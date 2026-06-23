#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#define TMA_IBGDA_THREADS_PER_BLOCK 512
#define TMA_IBGDA_WARP_SIZE 32
#define TMA_IBGDA_WARPS_PER_BLOCK \
  (TMA_IBGDA_THREADS_PER_BLOCK / TMA_IBGDA_WARP_SIZE)

#define _NVSHMEM_MIN_SM_ARCH 700
#define _NVSHMEM_DEVICELIB_SUPPORTED 1
#if defined(__CUDA_ARCH__)
#if (__CUDA_ARCH__ < _NVSHMEM_MIN_SM_ARCH) || (__CUDA_ARCH__ == 1100)
#undef _NVSHMEM_DEVICELIB_SUPPORTED
#endif
#endif

#ifndef _NVSHMEM_DEVICELIB_SUPPORTED
#define NVSHMEM_HOSTLIB_ONLY
#endif

#include <nvshmem.h>
#include <nvshmemx.h>

#include "olmo_bf16_tma_ibgda_ep/metadata.cuh"

namespace {

struct TmaIbgdaState {
  std::mutex mutex;
  bool initialized = false;
  int rank = -1;
  int world_size = -1;
  int device_idx = -1;
  std::vector<void*> allocations;
};

TmaIbgdaState& tma_ibgda_state() {
  static TmaIbgdaState state;
  return state;
}

void tma_ibgda_maybe_initialize_env_vars() {
  const char* nccl_socket_if_name = std::getenv("NCCL_SOCKET_IFNAME");
  if (std::getenv("NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME") == nullptr &&
      nccl_socket_if_name != nullptr) {
    setenv("NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME", nccl_socket_if_name, 0);
  }
  const char* nccl_hca_list = std::getenv("NCCL_IB_HCA");
  if (std::getenv("NVSHMEM_HCA_LIST") == nullptr && nccl_hca_list != nullptr) {
    setenv("NVSHMEM_ENABLE_NIC_PE_MAPPING", "1", 0);
    setenv("NVSHMEM_HCA_LIST", nccl_hca_list, 0);
  }
  const char* nccl_ib_gid_index = std::getenv("NCCL_IB_GID_INDEX");
  if (std::getenv("NVSHMEM_IB_GID_INDEX") == nullptr &&
      nccl_ib_gid_index != nullptr) {
    setenv("NVSHMEM_IB_GID_INDEX", nccl_ib_gid_index, 0);
  }
}

std::string cu_result_string(CUresult result) {
  const char* name = nullptr;
  const char* desc = nullptr;
  (void)cuGetErrorName(result, &name);
  (void)cuGetErrorString(result, &desc);
  std::string msg;
  if (name != nullptr) {
    msg += name;
  }
  if (desc != nullptr) {
    if (!msg.empty()) {
      msg += ": ";
    }
    msg += desc;
  }
  return msg;
}

void maybe_init_nvshmem_cumodule(const void* kernel_symbol) {
  static std::once_flag once;
  std::call_once(once, [kernel_symbol]() {
    cudaFunction_t cuda_func{};
    auto rt_status = cudaGetFuncBySymbol(&cuda_func, kernel_symbol);
    TORCH_CHECK(
        rt_status == cudaSuccess,
        "cudaGetFuncBySymbol failed while initializing TMA/IBGDA NVSHMEM module: ",
        cudaGetErrorString(rt_status));

    CUmodule cu_module{};
    auto cu_status = cuFuncGetModule(
        &cu_module, reinterpret_cast<CUfunction>(cuda_func));
    TORCH_CHECK(
        cu_status == CUDA_SUCCESS,
        "cuFuncGetModule failed while initializing TMA/IBGDA NVSHMEM module (",
        static_cast<int>(cu_status),
        "): ",
        cu_result_string(cu_status));

    int nv_status = nvshmemx_cumodule_init(cu_module);
    TORCH_CHECK(
        nv_status == 0,
        "nvshmemx_cumodule_init failed for TMA/IBGDA module with status ",
        nv_status);
  });
}

int resolve_route_blocks(int64_t num_routes, int64_t requested_blocks) {
  if (requested_blocks > 0) {
    return static_cast<int>(requested_blocks);
  }
  int blocks = static_cast<int>((num_routes + TMA_IBGDA_WARPS_PER_BLOCK - 1) /
      TMA_IBGDA_WARPS_PER_BLOCK);
  if (blocks < 1) {
    blocks = 1;
  }
  if (blocks > 1024) {
    blocks = 1024;
  }
  return blocks;
}

__device__ __forceinline__ __nv_bfloat16 bf16_mul_prob(
    __nv_bfloat16 value,
    float prob) {
  return __float2bfloat16(__bfloat162float(value) * prob);
}

struct alignas(8) TmaMBarrier {
  uint64_t storage;
};

using TmaArrivalPhase = uint32_t;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
__device__ __forceinline__ uint32_t shared_addr_u32(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void tma_mbarrier_init(
    TmaMBarrier* ptr,
    int arrive_count) {
  asm volatile(
      "mbarrier.init.shared::cta.b64 [%1], %0;" ::
      "r"(arrive_count),
      "r"(shared_addr_u32(ptr)));
  asm volatile("fence.mbarrier_init.release.cluster;" ::);
}

__device__ __forceinline__ void tma_mbarrier_arrive_and_set_tx(
    TmaMBarrier* ptr,
    int num_bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0;" ::
      "r"(num_bytes),
      "r"(shared_addr_u32(ptr)));
}

__device__ __forceinline__ void tma_mbarrier_wait_and_flip_phase(
    TmaMBarrier* ptr,
    TmaArrivalPhase& phase) {
  asm volatile(
      "{\n\t"
      ".reg .pred P1;\n\t"
      "LAB_WAIT:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n\t"
      "@P1 bra DONE;\n\t"
      "bra LAB_WAIT;\n\t"
      "DONE:\n\t"
      "}" ::
      "r"(shared_addr_u32(ptr)),
      "r"(phase),
      "r"(0x989680));
  phase ^= 1;
}

__device__ __forceinline__ void tma_load_1d(
    void* dst,
    const void* src,
    TmaMBarrier* barrier,
    int num_bytes) {
  constexpr int64_t kEvictFirst = 0x12f0000000000000ll;
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
      "[%0], [%1], %2, [%3], %4;\n" ::
      "r"(shared_addr_u32(dst)),
      "l"(src),
      "r"(num_bytes),
      "r"(shared_addr_u32(barrier)),
      "l"(kEvictFirst)
      : "memory");
}

__device__ __forceinline__ bool tma_load_chunk_if_aligned(
    __nv_bfloat16* shared_row,
    const __nv_bfloat16* src,
    int64_t chunk_elems,
    TmaMBarrier* barrier,
    TmaArrivalPhase& phase,
    int lane_id) {
  int64_t num_bytes64 = chunk_elems * static_cast<int64_t>(sizeof(__nv_bfloat16));
  if (num_bytes64 < 32 || (num_bytes64 % 32) != 0) {
    return false;
  }
  int num_bytes = static_cast<int>(num_bytes64);
  if (lane_id == 0) {
    tma_load_1d(shared_row, src, barrier, num_bytes);
    tma_mbarrier_arrive_and_set_tx(barrier, num_bytes);
    tma_mbarrier_wait_and_flip_phase(barrier, phase);
  }
  __syncwarp();
  return true;
}
#endif

__global__ void preprocess_routes_kernel(
    const int64_t* route_ranks,
    const int64_t* route_rows,
    const float* route_probs,
    olmo::tma_ibgda_ep::RouteRecord* route_records,
    int64_t* routes_per_rank,
    int64_t* route_ordinals,
    int32_t* errors,
    int64_t num_routes,
    int64_t top_k,
    int32_t ep_world_size,
    int32_t rank_capacity) {
  int64_t route_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (; route_id < num_routes; route_id += stride) {
    int64_t peer64 = route_ranks[route_id];
    int64_t row64 = route_rows[route_id];
    bool rank_dropped = peer64 < 0;
    bool row_dropped = row64 < 0;
    bool mismatched_drop = rank_dropped != row_dropped;
    bool valid = !(rank_dropped || row_dropped || mismatched_drop);

    auto& record = route_records[route_id];
    record.source_row = static_cast<int32_t>(route_id / top_k);
    record.topk_slot = static_cast<int32_t>(route_id - (route_id / top_k) * top_k);
    record.peer_rank = static_cast<int32_t>(peer64);
    record.peer_row = static_cast<int32_t>(row64);
    record.prob = route_probs == nullptr ? 1.0f : route_probs[route_id];
    record.flags = 0;
    record.reserved0 = 0;
    record.reserved1 = 0;

    if (mismatched_drop) {
      atomicOr(errors + 0, 1);
      continue;
    }
    if (!valid) {
      continue;
    }
    if (peer64 >= ep_world_size) {
      atomicOr(errors + 1, 1);
      continue;
    }
    if (row64 >= rank_capacity) {
      atomicOr(errors + 2, 1);
      continue;
    }
    record.flags = olmo::tma_ibgda_ep::ROUTE_FLAG_VALID;
    unsigned long long ordinal = atomicAdd(
        reinterpret_cast<unsigned long long*>(routes_per_rank + peer64),
        static_cast<unsigned long long>(1));
    route_ordinals[route_id] = static_cast<int64_t>(ordinal);
  }
}

__global__ void finalize_route_preprocess_kernel(
    const int64_t* routes_per_rank,
    int64_t* rank_offsets,
    bool* overflow_by_rank,
    int32_t ep_world_size,
    int32_t static_route_budget) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  int64_t running = 0;
  rank_offsets[0] = 0;
  for (int32_t rank = 0; rank < ep_world_size; ++rank) {
    int64_t count = routes_per_rank[rank];
    running += count;
    rank_offsets[rank + 1] = running;
    overflow_by_rank[rank] = static_route_budget > 0 && count > static_route_budget;
  }
}

__global__ void route_records_with_probs_kernel(
    const olmo::tma_ibgda_ep::RouteRecord* route_records,
    const float* probs,
    olmo::tma_ibgda_ep::RouteRecord* out_records,
    int64_t num_routes) {
  int64_t route_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (; route_id < num_routes; route_id += stride) {
    auto record = route_records[route_id];
    record.prob = probs[route_id];
    out_records[route_id] = record;
  }
}

__global__ void dispatch_bf16_peer_kernel(
    const __nv_bfloat16* input,
    int64_t input_stride,
    int64_t num_input_rows,
    int64_t top_k,
    int64_t hidden,
    const int64_t* dst_ranks,
    const int64_t* dst_rows,
    const float* probs,
    const int64_t* peer_out_ptrs,
    int64_t out_stride,
    int64_t out_capacity_rows,
    int64_t world_size) {
  int warp_id = threadIdx.x / TMA_IBGDA_WARP_SIZE;
  int lane_id = threadIdx.x % TMA_IBGDA_WARP_SIZE;
  int64_t num_routes = num_input_rows * top_k;
  int64_t route_id = static_cast<int64_t>(blockIdx.x) * TMA_IBGDA_WARPS_PER_BLOCK + warp_id;
  int64_t route_stride = static_cast<int64_t>(gridDim.x) * TMA_IBGDA_WARPS_PER_BLOCK;

  for (; route_id < num_routes; route_id += route_stride) {
    int64_t peer = dst_ranks[route_id];
    int64_t dst_row = dst_rows[route_id];
    if (peer < 0 || dst_row < 0) {
      continue;
    }
    CUDA_KERNEL_ASSERT(peer < world_size);
    CUDA_KERNEL_ASSERT(dst_row < out_capacity_rows);
    auto peer_addr = static_cast<uint64_t>(peer_out_ptrs[peer]);
    CUDA_KERNEL_ASSERT(peer_addr != 0);
    auto* dst = reinterpret_cast<__nv_bfloat16*>(peer_addr) + dst_row * out_stride;
    const __nv_bfloat16* src = input + (route_id / top_k) * input_stride;
    float prob = probs == nullptr ? 1.0f : probs[route_id];
    for (int64_t col = lane_id; col < hidden; col += TMA_IBGDA_WARP_SIZE) {
      __nv_bfloat16 value = src[col];
      if (probs != nullptr) {
        value = bf16_mul_prob(value, prob);
      }
      dst[col] = value;
    }
  }
}

__global__ void combine_bf16_peer_kernel(
    const __nv_bfloat16* local_expert_out,
    __nv_bfloat16* out,
    int64_t out_stride,
    int64_t num_out_rows,
    int64_t top_k,
    int64_t hidden,
    const int64_t* src_ranks,
    const int64_t* src_rows,
    const float* probs,
    const int64_t* peer_expert_out_ptrs,
    int64_t expert_stride,
    int64_t expert_capacity_rows,
    int64_t world_size) {
  (void)local_expert_out;
  int64_t row = blockIdx.x;
  int64_t col = static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
  if (row >= num_out_rows || col >= hidden) {
    return;
  }

  float acc = 0.0f;
  int64_t route_base = row * top_k;
  for (int64_t k = 0; k < top_k; ++k) {
    int64_t route_id = route_base + k;
    int64_t peer = src_ranks[route_id];
    int64_t src_row = src_rows[route_id];
    if (peer < 0 || src_row < 0) {
      continue;
    }
    CUDA_KERNEL_ASSERT(peer < world_size);
    CUDA_KERNEL_ASSERT(src_row < expert_capacity_rows);
    auto peer_addr = static_cast<uint64_t>(peer_expert_out_ptrs[peer]);
    CUDA_KERNEL_ASSERT(peer_addr != 0);
    const auto* src =
        reinterpret_cast<const __nv_bfloat16*>(peer_addr) + src_row * expert_stride;
    float value = __bfloat162float(src[col]);
    if (probs != nullptr) {
      value *= probs[route_id];
    }
    acc += value;
  }
  out[row * out_stride + col] = __float2bfloat16(acc);
}

__global__ void dispatch_bf16_ibgda_kernel(
    const __nv_bfloat16* input,
    __nv_bfloat16* out,
    int64_t input_stride,
    int64_t out_stride,
    int64_t num_input_rows,
    int64_t top_k,
    int64_t hidden,
    const int64_t* dst_ranks,
    const int64_t* dst_rows,
    const float* probs,
    int64_t out_capacity_rows) {
#ifndef _NVSHMEM_DEVICELIB_SUPPORTED
  CUDA_KERNEL_ASSERT(false);
#else
  constexpr int ELEMS_PER_THREAD = 4;
  constexpr int CHUNK_ELEMS = TMA_IBGDA_WARP_SIZE * ELEMS_PER_THREAD;
  __shared__ __nv_bfloat16 shared_rows[TMA_IBGDA_WARPS_PER_BLOCK][CHUNK_ELEMS];

  int npes = nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD);
  int warp_id = threadIdx.x / TMA_IBGDA_WARP_SIZE;
  int lane_id = threadIdx.x % TMA_IBGDA_WARP_SIZE;
  int64_t num_routes = num_input_rows * top_k;
  int64_t route_id =
      static_cast<int64_t>(blockIdx.x) * TMA_IBGDA_WARPS_PER_BLOCK + warp_id;
  int64_t route_stride =
      static_cast<int64_t>(gridDim.x) * TMA_IBGDA_WARPS_PER_BLOCK;

  for (; route_id < num_routes; route_id += route_stride) {
    int64_t peer = dst_ranks[route_id];
    int64_t dst_row = dst_rows[route_id];
    if (peer < 0 || dst_row < 0) {
      continue;
    }
    CUDA_KERNEL_ASSERT(peer < npes);
    CUDA_KERNEL_ASSERT(dst_row < out_capacity_rows);
    int64_t src_row = route_id / top_k;
    const __nv_bfloat16* src = input + src_row * input_stride;
    float prob = probs == nullptr ? 1.0f : probs[route_id];

    for (int64_t col_base = 0; col_base < hidden; col_base += CHUNK_ELEMS) {
      int64_t remaining = hidden - col_base;
      int64_t chunk_elems = remaining < CHUNK_ELEMS ? remaining : CHUNK_ELEMS;
      auto* shared_row = shared_rows[warp_id];
#pragma unroll
      for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        int elem = i * TMA_IBGDA_WARP_SIZE + lane_id;
        if (elem < chunk_elems) {
          __nv_bfloat16 value = src[col_base + elem];
          if (probs != nullptr) {
            value = bf16_mul_prob(value, prob);
          }
          shared_row[elem] = value;
        }
      }
      __syncwarp();
      nvshmemx_putmem_warp(
          out + dst_row * out_stride + col_base,
          shared_row,
          static_cast<size_t>(chunk_elems) * sizeof(__nv_bfloat16),
          static_cast<int>(peer));
      __syncwarp();
    }
  }
  nvshmem_quiet();
#endif
}

__global__ void dispatch_bf16_ibgda_records_kernel(
    const __nv_bfloat16* input,
    __nv_bfloat16* out,
    int64_t input_stride,
    int64_t out_stride,
    int64_t num_routes,
    int64_t hidden,
    const olmo::tma_ibgda_ep::RouteRecord* route_records,
    int64_t out_capacity_rows) {
#ifndef _NVSHMEM_DEVICELIB_SUPPORTED
  CUDA_KERNEL_ASSERT(false);
#else
  constexpr int ELEMS_PER_THREAD = 4;
  constexpr int CHUNK_ELEMS = TMA_IBGDA_WARP_SIZE * ELEMS_PER_THREAD;
  __shared__ __nv_bfloat16 shared_rows[TMA_IBGDA_WARPS_PER_BLOCK][CHUNK_ELEMS];

  int npes = nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD);
  int warp_id = threadIdx.x / TMA_IBGDA_WARP_SIZE;
  int lane_id = threadIdx.x % TMA_IBGDA_WARP_SIZE;
  int64_t route_id =
      static_cast<int64_t>(blockIdx.x) * TMA_IBGDA_WARPS_PER_BLOCK + warp_id;
  int64_t route_stride =
      static_cast<int64_t>(gridDim.x) * TMA_IBGDA_WARPS_PER_BLOCK;

  for (; route_id < num_routes; route_id += route_stride) {
    const auto& record = route_records[route_id];
    if ((record.flags & olmo::tma_ibgda_ep::ROUTE_FLAG_VALID) == 0) {
      continue;
    }
    int peer = record.peer_rank;
    int dst_row = record.peer_row;
    CUDA_KERNEL_ASSERT(peer >= 0 && peer < npes);
    CUDA_KERNEL_ASSERT(dst_row >= 0 && dst_row < out_capacity_rows);
    const __nv_bfloat16* src = input + static_cast<int64_t>(record.source_row) * input_stride;
    float prob = record.prob;

    for (int64_t col_base = 0; col_base < hidden; col_base += CHUNK_ELEMS) {
      int64_t remaining = hidden - col_base;
      int64_t chunk_elems = remaining < CHUNK_ELEMS ? remaining : CHUNK_ELEMS;
      auto* shared_row = shared_rows[warp_id];
#pragma unroll
      for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        int elem = i * TMA_IBGDA_WARP_SIZE + lane_id;
        if (elem < chunk_elems) {
          __nv_bfloat16 value = src[col_base + elem];
          if (prob != 1.0f) {
            value = bf16_mul_prob(value, prob);
          }
          shared_row[elem] = value;
        }
      }
      __syncwarp();
      nvshmemx_putmem_warp(
          out + static_cast<int64_t>(dst_row) * out_stride + col_base,
          shared_row,
          static_cast<size_t>(chunk_elems) * sizeof(__nv_bfloat16),
          peer);
      __syncwarp();
    }
  }
  nvshmem_quiet();
#endif
}

__global__ void dispatch_bf16_ibgda_records_tma_kernel(
    const __nv_bfloat16* input,
    __nv_bfloat16* out,
    int64_t input_stride,
    int64_t out_stride,
    int64_t num_routes,
    int64_t hidden,
    const olmo::tma_ibgda_ep::RouteRecord* route_records,
    int64_t out_capacity_rows) {
#ifndef _NVSHMEM_DEVICELIB_SUPPORTED
  CUDA_KERNEL_ASSERT(false);
#else
  constexpr int ELEMS_PER_THREAD = 4;
  constexpr int CHUNK_ELEMS = TMA_IBGDA_WARP_SIZE * ELEMS_PER_THREAD;
  __shared__ __align__(32) __nv_bfloat16 shared_rows[TMA_IBGDA_WARPS_PER_BLOCK][CHUNK_ELEMS];
  __shared__ __align__(8) TmaMBarrier barriers[TMA_IBGDA_WARPS_PER_BLOCK];

  int npes = nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD);
  int warp_id = threadIdx.x / TMA_IBGDA_WARP_SIZE;
  int lane_id = threadIdx.x % TMA_IBGDA_WARP_SIZE;
  TmaArrivalPhase phase = 0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if (lane_id == 0) {
    tma_mbarrier_init(barriers + warp_id, 1);
  }
  __syncwarp();
#endif

  int64_t route_id =
      static_cast<int64_t>(blockIdx.x) * TMA_IBGDA_WARPS_PER_BLOCK + warp_id;
  int64_t route_stride =
      static_cast<int64_t>(gridDim.x) * TMA_IBGDA_WARPS_PER_BLOCK;

  for (; route_id < num_routes; route_id += route_stride) {
    const auto& record = route_records[route_id];
    if ((record.flags & olmo::tma_ibgda_ep::ROUTE_FLAG_VALID) == 0) {
      continue;
    }
    int peer = record.peer_rank;
    int dst_row = record.peer_row;
    CUDA_KERNEL_ASSERT(peer >= 0 && peer < npes);
    CUDA_KERNEL_ASSERT(dst_row >= 0 && dst_row < out_capacity_rows);
    const __nv_bfloat16* src =
        input + static_cast<int64_t>(record.source_row) * input_stride;
    float prob = record.prob;

    for (int64_t col_base = 0; col_base < hidden; col_base += CHUNK_ELEMS) {
      int64_t remaining = hidden - col_base;
      int64_t chunk_elems = remaining < CHUNK_ELEMS ? remaining : CHUNK_ELEMS;
      auto* shared_row = shared_rows[warp_id];
      const auto* chunk_src = src + col_base;
      bool used_tma = false;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
      used_tma = tma_load_chunk_if_aligned(
          shared_row,
          chunk_src,
          chunk_elems,
          barriers + warp_id,
          phase,
          lane_id);
#endif
      if (!used_tma) {
#pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
          int elem = i * TMA_IBGDA_WARP_SIZE + lane_id;
          if (elem < chunk_elems) {
            shared_row[elem] = chunk_src[elem];
          }
        }
        __syncwarp();
      }
      if (prob != 1.0f) {
#pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
          int elem = i * TMA_IBGDA_WARP_SIZE + lane_id;
          if (elem < chunk_elems) {
            shared_row[elem] = bf16_mul_prob(shared_row[elem], prob);
          }
        }
        __syncwarp();
      }
      nvshmemx_putmem_warp(
          out + static_cast<int64_t>(dst_row) * out_stride + col_base,
          shared_row,
          static_cast<size_t>(chunk_elems) * sizeof(__nv_bfloat16),
          peer);
      __syncwarp();
    }
  }
  nvshmem_quiet();
#endif
}

__global__ void combine_bf16_ibgda_fused_kernel(
    const __nv_bfloat16* expert_out,
    __nv_bfloat16* out,
    int64_t out_stride,
    int64_t expert_stride,
    int64_t num_out_rows,
    int64_t top_k,
    int64_t hidden,
    const int64_t* src_ranks,
    const int64_t* src_rows,
    const float* probs,
    int64_t expert_capacity_rows,
    int64_t num_hidden_chunks) {
#ifndef _NVSHMEM_DEVICELIB_SUPPORTED
  CUDA_KERNEL_ASSERT(false);
#else
  constexpr int ELEMS_PER_THREAD = 4;
  constexpr int CHUNK_ELEMS = TMA_IBGDA_WARP_SIZE * ELEMS_PER_THREAD;
  __shared__ __nv_bfloat16 shared_rows[TMA_IBGDA_WARPS_PER_BLOCK][CHUNK_ELEMS];

  int npes = nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD);
  int warp_id = threadIdx.x / TMA_IBGDA_WARP_SIZE;
  int lane_id = threadIdx.x % TMA_IBGDA_WARP_SIZE;
  int64_t num_tiles = num_out_rows * num_hidden_chunks;
  int64_t tile_id =
      static_cast<int64_t>(blockIdx.x) * TMA_IBGDA_WARPS_PER_BLOCK + warp_id;
  int64_t tile_stride =
      static_cast<int64_t>(gridDim.x) * TMA_IBGDA_WARPS_PER_BLOCK;

  for (; tile_id < num_tiles; tile_id += tile_stride) {
    int64_t row = tile_id / num_hidden_chunks;
    int64_t chunk = tile_id - row * num_hidden_chunks;
    int64_t col_base = chunk * CHUNK_ELEMS;
    int64_t chunk_elems = hidden - col_base;
    if (chunk_elems > CHUNK_ELEMS) {
      chunk_elems = CHUNK_ELEMS;
    }

    float acc[ELEMS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
      acc[i] = 0.0f;
    }
    auto* shared_row = shared_rows[warp_id];

    int64_t route_base = row * top_k;
    for (int64_t k = 0; k < top_k; ++k) {
      int64_t route_id = route_base + k;
      int64_t peer = src_ranks[route_id];
      int64_t src_row = src_rows[route_id];
      if (peer < 0 || src_row < 0) {
        continue;
      }
      CUDA_KERNEL_ASSERT(peer < npes);
      CUDA_KERNEL_ASSERT(src_row < expert_capacity_rows);
      const __nv_bfloat16* remote = expert_out + src_row * expert_stride + col_base;
      nvshmemx_getmem_warp(
          shared_row,
          remote,
          static_cast<size_t>(chunk_elems) * sizeof(__nv_bfloat16),
          static_cast<int>(peer));
      __syncwarp();
      float prob = probs == nullptr ? 1.0f : probs[route_id];
#pragma unroll
      for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        int elem = i * TMA_IBGDA_WARP_SIZE + lane_id;
        if (elem < chunk_elems) {
          acc[i] += __bfloat162float(shared_row[elem]) * prob;
        }
      }
      __syncwarp();
    }

#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
      int elem = i * TMA_IBGDA_WARP_SIZE + lane_id;
      if (elem < chunk_elems) {
        out[row * out_stride + col_base + elem] = __float2bfloat16(acc[i]);
      }
    }
  }
  nvshmem_quiet();
#endif
}

__global__ void combine_bf16_ibgda_records_fused_kernel(
    const __nv_bfloat16* expert_out,
    __nv_bfloat16* out,
    int64_t out_stride,
    int64_t expert_stride,
    int64_t num_out_rows,
    int64_t top_k,
    int64_t hidden,
    const olmo::tma_ibgda_ep::RouteRecord* route_records,
    int64_t expert_capacity_rows,
    int64_t num_hidden_chunks) {
#ifndef _NVSHMEM_DEVICELIB_SUPPORTED
  CUDA_KERNEL_ASSERT(false);
#else
  constexpr int ELEMS_PER_THREAD = 4;
  constexpr int CHUNK_ELEMS = TMA_IBGDA_WARP_SIZE * ELEMS_PER_THREAD;
  __shared__ __nv_bfloat16 shared_rows[TMA_IBGDA_WARPS_PER_BLOCK][CHUNK_ELEMS];

  int npes = nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD);
  int warp_id = threadIdx.x / TMA_IBGDA_WARP_SIZE;
  int lane_id = threadIdx.x % TMA_IBGDA_WARP_SIZE;
  int64_t num_tiles = num_out_rows * num_hidden_chunks;
  int64_t tile_id =
      static_cast<int64_t>(blockIdx.x) * TMA_IBGDA_WARPS_PER_BLOCK + warp_id;
  int64_t tile_stride =
      static_cast<int64_t>(gridDim.x) * TMA_IBGDA_WARPS_PER_BLOCK;
  for (; tile_id < num_tiles; tile_id += tile_stride) {
    int64_t row = tile_id / num_hidden_chunks;
    int64_t chunk = tile_id - row * num_hidden_chunks;
    int64_t col_base = chunk * CHUNK_ELEMS;
    int64_t chunk_elems = hidden - col_base;
    if (chunk_elems > CHUNK_ELEMS) {
      chunk_elems = CHUNK_ELEMS;
    }

    float acc[ELEMS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
      acc[i] = 0.0f;
    }
    auto* shared_row = shared_rows[warp_id];

    int64_t route_base = row * top_k;
    for (int64_t k = 0; k < top_k; ++k) {
      int64_t route_id = route_base + k;
      const auto& record = route_records[route_id];
      if ((record.flags & olmo::tma_ibgda_ep::ROUTE_FLAG_VALID) == 0) {
        continue;
      }
      int peer = record.peer_rank;
      int src_row = record.peer_row;
      CUDA_KERNEL_ASSERT(peer >= 0 && peer < npes);
      CUDA_KERNEL_ASSERT(src_row >= 0 && src_row < expert_capacity_rows);
      const __nv_bfloat16* remote =
          expert_out + static_cast<int64_t>(src_row) * expert_stride + col_base;
      nvshmemx_getmem_warp(
          shared_row,
          remote,
          static_cast<size_t>(chunk_elems) * sizeof(__nv_bfloat16),
          peer);
      __syncwarp();
      float prob = record.prob;
#pragma unroll
      for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        int elem = i * TMA_IBGDA_WARP_SIZE + lane_id;
        if (elem < chunk_elems) {
          acc[i] += __bfloat162float(shared_row[elem]) * prob;
        }
      }
      __syncwarp();
    }

#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
      int elem = i * TMA_IBGDA_WARP_SIZE + lane_id;
      if (elem < chunk_elems) {
        out[row * out_stride + col_base + elem] = __float2bfloat16(acc[i]);
      }
    }
  }
  nvshmem_quiet();
#endif
}

__global__ void route_dot_bf16_peer_kernel(
    const __nv_bfloat16* local_expert_out,
    const __nv_bfloat16* grad_out,
    int64_t grad_stride,
    int64_t num_grad_rows,
    int64_t top_k,
    int64_t hidden,
    const int64_t* src_ranks,
    const int64_t* src_rows,
    const int64_t* peer_expert_out_ptrs,
    int64_t expert_stride,
    int64_t expert_capacity_rows,
    int64_t world_size,
    float* out) {
  (void)local_expert_out;
  __shared__ float partials[256];
  int64_t route_id = blockIdx.x;
  int tid = threadIdx.x;
  int64_t num_routes = num_grad_rows * top_k;
  if (route_id >= num_routes) {
    return;
  }

  int64_t peer = src_ranks[route_id];
  int64_t src_row = src_rows[route_id];
  float acc = 0.0f;
  if (peer >= 0 && src_row >= 0) {
    CUDA_KERNEL_ASSERT(peer < world_size);
    CUDA_KERNEL_ASSERT(src_row < expert_capacity_rows);
    auto peer_addr = static_cast<uint64_t>(peer_expert_out_ptrs[peer]);
    CUDA_KERNEL_ASSERT(peer_addr != 0);
    const auto* expert =
        reinterpret_cast<const __nv_bfloat16*>(peer_addr) + src_row * expert_stride;
    const auto* grad = grad_out + (route_id / top_k) * grad_stride;
    for (int64_t col = tid; col < hidden; col += blockDim.x) {
      acc += __bfloat162float(expert[col]) * __bfloat162float(grad[col]);
    }
  }

  partials[tid] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partials[tid] += partials[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out[route_id] = partials[0];
  }
}

__global__ void route_dot_bf16_ibgda_kernel(
    const __nv_bfloat16* expert_out,
    const __nv_bfloat16* grad_out,
    int64_t grad_stride,
    int64_t num_grad_rows,
    int64_t top_k,
    int64_t hidden,
    const int64_t* src_ranks,
    const int64_t* src_rows,
    int64_t expert_stride,
    int64_t expert_capacity_rows,
    float* out) {
#ifndef _NVSHMEM_DEVICELIB_SUPPORTED
  CUDA_KERNEL_ASSERT(false);
#else
  constexpr int CHUNK_ELEMS = TMA_IBGDA_WARP_SIZE;
  __shared__ __nv_bfloat16 shared_row[CHUNK_ELEMS];
  int lane_id = threadIdx.x;
  int64_t route_id = blockIdx.x;
  int64_t num_routes = num_grad_rows * top_k;
  if (route_id >= num_routes) {
    return;
  }

  int64_t peer = src_ranks[route_id];
  int64_t src_row = src_rows[route_id];
  float acc = 0.0f;
  if (peer >= 0 && src_row >= 0) {
    int npes = nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD);
    CUDA_KERNEL_ASSERT(peer < npes);
    CUDA_KERNEL_ASSERT(src_row < expert_capacity_rows);
    const __nv_bfloat16* remote = expert_out + src_row * expert_stride;
    const __nv_bfloat16* grad = grad_out + (route_id / top_k) * grad_stride;
    for (int64_t col_base = 0; col_base < hidden; col_base += CHUNK_ELEMS) {
      int64_t chunk_elems = hidden - col_base;
      if (chunk_elems > CHUNK_ELEMS) {
        chunk_elems = CHUNK_ELEMS;
      }
      nvshmemx_getmem_warp(
          shared_row,
          remote + col_base,
          static_cast<size_t>(chunk_elems) * sizeof(__nv_bfloat16),
          static_cast<int>(peer));
      __syncwarp();
      if (lane_id < chunk_elems) {
        int64_t col = col_base + lane_id;
        acc += __bfloat162float(shared_row[lane_id]) * __bfloat162float(grad[col]);
      }
      __syncwarp();
    }
  }
  for (int offset = TMA_IBGDA_WARP_SIZE / 2; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xffffffff, acc, offset);
  }
  if (lane_id == 0) {
    out[route_id] = acc;
  }
  nvshmem_quiet();
#endif
}

__global__ void route_dot_bf16_ibgda_records_kernel(
    const __nv_bfloat16* expert_out,
    const __nv_bfloat16* grad_out,
    int64_t grad_stride,
    int64_t top_k,
    int64_t hidden,
    const olmo::tma_ibgda_ep::RouteRecord* route_records,
    int64_t num_routes,
    int64_t expert_stride,
    int64_t expert_capacity_rows,
    float* out) {
#ifndef _NVSHMEM_DEVICELIB_SUPPORTED
  CUDA_KERNEL_ASSERT(false);
#else
  constexpr int CHUNK_ELEMS = TMA_IBGDA_WARP_SIZE;
  __shared__ __nv_bfloat16 shared_row[CHUNK_ELEMS];
  int lane_id = threadIdx.x;
  int64_t route_id = blockIdx.x;
  if (route_id >= num_routes) {
    return;
  }

  const auto& record = route_records[route_id];
  int64_t out_idx =
      static_cast<int64_t>(record.source_row) * top_k + record.topk_slot;
  float acc = 0.0f;
  if ((record.flags & olmo::tma_ibgda_ep::ROUTE_FLAG_VALID) != 0) {
    int npes = nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD);
    CUDA_KERNEL_ASSERT(record.peer_rank >= 0 && record.peer_rank < npes);
    CUDA_KERNEL_ASSERT(record.peer_row >= 0 && record.peer_row < expert_capacity_rows);
    const __nv_bfloat16* remote =
        expert_out + static_cast<int64_t>(record.peer_row) * expert_stride;
    const __nv_bfloat16* grad =
        grad_out + static_cast<int64_t>(record.source_row) * grad_stride;
    for (int64_t col_base = 0; col_base < hidden; col_base += CHUNK_ELEMS) {
      int64_t chunk_elems = hidden - col_base;
      if (chunk_elems > CHUNK_ELEMS) {
        chunk_elems = CHUNK_ELEMS;
      }
      nvshmemx_getmem_warp(
          shared_row,
          remote + col_base,
          static_cast<size_t>(chunk_elems) * sizeof(__nv_bfloat16),
          record.peer_rank);
      __syncwarp();
      if (lane_id < chunk_elems) {
        int64_t col = col_base + lane_id;
        acc += __bfloat162float(shared_row[lane_id]) * __bfloat162float(grad[col]);
      }
      __syncwarp();
    }
  }
  for (int offset = TMA_IBGDA_WARP_SIZE / 2; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xffffffff, acc, offset);
  }
  if (lane_id == 0) {
    out[out_idx] = acc;
  }
  nvshmem_quiet();
#endif
}

}  // namespace

std::vector<uint8_t> tma_ibgda_get_unique_id() {
  nvshmemx_uniqueid_t unique_id;
  int status = nvshmemx_get_uniqueid(&unique_id);
  TORCH_CHECK(status == 0, "nvshmemx_get_uniqueid failed with status ", status);
  const auto* begin = reinterpret_cast<const uint8_t*>(&unique_id);
  return std::vector<uint8_t>(begin, begin + sizeof(nvshmemx_uniqueid_t));
}

void tma_ibgda_init(
    const std::vector<std::vector<uint8_t>>& unique_ids,
    int64_t rank,
    int64_t world_size,
    int64_t device_idx) {
  auto& state = tma_ibgda_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.initialized) {
    TORCH_CHECK(
        state.rank == rank && state.world_size == world_size &&
            state.device_idx == device_idx,
        "OLMo TMA/IBGDA NVSHMEM state is already initialized with rank=",
        state.rank,
        ", world_size=",
        state.world_size,
        ", device_idx=",
        state.device_idx,
        " but got rank=",
        rank,
        ", world_size=",
        world_size,
        ", device_idx=",
        device_idx);
    return;
  }
  TORCH_CHECK(world_size > 0, "world_size must be positive");
  TORCH_CHECK(rank >= 0 && rank < world_size, "rank must be in [0, world_size)");
  TORCH_CHECK(
      static_cast<int64_t>(unique_ids.size()) == world_size,
      "unique_ids length must equal world_size");

  std::vector<nvshmemx_uniqueid_t> ids(static_cast<size_t>(world_size));
  for (int64_t i = 0; i < world_size; ++i) {
    TORCH_CHECK(
        unique_ids[i].size() == sizeof(nvshmemx_uniqueid_t),
        "NVSHMEM unique ID has unexpected size: ",
        unique_ids[i].size(),
        " expected ",
        sizeof(nvshmemx_uniqueid_t));
    std::memcpy(
        &ids[static_cast<size_t>(i)],
        unique_ids[i].data(),
        sizeof(nvshmemx_uniqueid_t));
  }

  c10::cuda::CUDAGuard guard(static_cast<int>(device_idx));
  tma_ibgda_maybe_initialize_env_vars();
  C10_CUDA_CHECK(cudaFree(nullptr));

  nvshmemx_init_attr_t attr;
  int set_status = nvshmemx_set_attr_uniqueid_args(
      static_cast<int>(rank), static_cast<int>(world_size), ids.data(), &attr);
  TORCH_CHECK(
      set_status == 0,
      "nvshmemx_set_attr_uniqueid_args failed with status ",
      set_status);
  int init_status = nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
  TORCH_CHECK(
      init_status == 0,
      "nvshmemx_init_attr failed with status ",
      init_status);

  state.initialized = true;
  state.rank = static_cast<int>(rank);
  state.world_size = static_cast<int>(world_size);
  state.device_idx = static_cast<int>(device_idx);
}

torch::Tensor tma_ibgda_empty(
    const std::vector<int64_t>& sizes,
    c10::ScalarType dtype,
    c10::Device device) {
  auto& state = tma_ibgda_state();
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    TORCH_CHECK(state.initialized, "OLMo TMA/IBGDA NVSHMEM state is not initialized");
  }
  TORCH_CHECK(device.is_cuda(), "OLMo TMA/IBGDA tensors must be CUDA tensors");
  c10::cuda::CUDAGuard guard(device);

  size_t numel = 1;
  for (auto dim : sizes) {
    TORCH_CHECK(dim >= 0, "negative tensor dimension: ", dim);
    numel *= static_cast<size_t>(dim);
  }
  size_t alloc_size = numel * c10::elementSize(dtype);
  void* ptr = nvshmem_malloc(alloc_size);
  TORCH_CHECK(ptr != nullptr || alloc_size == 0, "nvshmem_malloc failed");

  std::vector<int64_t> strides(sizes.size());
  int64_t stride = 1;
  for (int64_t i = static_cast<int64_t>(sizes.size()) - 1; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = stride;
    stride *= sizes[static_cast<size_t>(i)];
  }
  auto options = at::TensorOptions().dtype(dtype).device(device);
  auto tensor = at::from_blob(ptr, sizes, strides, [](void*) {}, options);
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    state.allocations.push_back(ptr);
  }
  return tensor;
}

void tma_ibgda_barrier_all_on_stream(c10::Device device) {
  auto& state = tma_ibgda_state();
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    TORCH_CHECK(state.initialized, "OLMo TMA/IBGDA NVSHMEM state is not initialized");
  }
  TORCH_CHECK(device.is_cuda(), "OLMo TMA/IBGDA stream barriers require a CUDA device");
  c10::cuda::CUDAGuard guard(device);
  auto stream = at::cuda::getCurrentCUDAStream();
  nvshmemx_quiet_on_stream(stream.stream());
  nvshmemx_barrier_all_on_stream(stream.stream());
  C10_CUDA_CHECK(cudaGetLastError());
}

void tma_ibgda_signal_all_and_wait(
    const torch::Tensor& signals,
    int64_t generation,
    int64_t world_size) {
  TORCH_CHECK(signals.is_cuda(), "TMA/IBGDA signals must be CUDA");
  TORCH_CHECK(signals.scalar_type() == torch::kInt64, "TMA/IBGDA signals must be int64");
  TORCH_CHECK(signals.is_contiguous(), "TMA/IBGDA signals must be contiguous");
  TORCH_CHECK(world_size > 0, "world_size must be positive");
  TORCH_CHECK(signals.numel() >= world_size, "signals must have at least world_size elements");
  TORCH_CHECK(generation > 0, "generation must be positive");
  auto& state = tma_ibgda_state();
  int rank = -1;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    TORCH_CHECK(state.initialized, "OLMo TMA/IBGDA NVSHMEM state is not initialized");
    TORCH_CHECK(
        state.world_size == world_size,
        "signal world_size mismatch: NVSHMEM world has ",
        state.world_size,
        " PEs, got ",
        world_size);
    rank = state.rank;
  }
  c10::cuda::CUDAGuard guard(signals.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* signal_ptr = reinterpret_cast<uint64_t*>(signals.data_ptr<int64_t>());
  uint64_t gen = static_cast<uint64_t>(generation);
  nvshmemx_quiet_on_stream(stream.stream());
  for (int pe = 0; pe < static_cast<int>(world_size); ++pe) {
    nvshmemx_signal_op_on_stream(
        signal_ptr + rank,
        gen,
        NVSHMEM_SIGNAL_SET,
        pe,
        stream.stream());
  }
  for (int peer = 0; peer < static_cast<int>(world_size); ++peer) {
    nvshmemx_signal_wait_until_on_stream(
        signal_ptr + peer,
        NVSHMEM_CMP_GE,
        gen,
        stream.stream());
  }
  C10_CUDA_CHECK(cudaGetLastError());
}

void tma_ibgda_preprocess_routes_launcher(
    const torch::Tensor& dst_ranks,
    const torch::Tensor& dst_rows,
    const std::optional<torch::Tensor>& probs,
    const torch::Tensor& route_records,
    const torch::Tensor& routes_per_rank,
    const torch::Tensor& rank_offsets,
    const torch::Tensor& overflow_by_rank,
    const torch::Tensor& route_ordinals,
    const torch::Tensor& errors,
    int64_t ep_world_size,
    int64_t rank_capacity,
    int64_t static_route_budget,
    int64_t nblocks) {
  int64_t num_routes = dst_ranks.numel();
  auto stream = at::cuda::getCurrentCUDAStream();
  routes_per_rank.zero_();
  rank_offsets.zero_();
  overflow_by_rank.zero_();
  route_ordinals.fill_(-1);
  errors.zero_();
  if (num_routes == 0) {
    finalize_route_preprocess_kernel<<<1, 1, 0, stream>>>(
        routes_per_rank.data_ptr<int64_t>(),
        rank_offsets.data_ptr<int64_t>(),
        overflow_by_rank.data_ptr<bool>(),
        static_cast<int32_t>(ep_world_size),
        static_cast<int32_t>(static_route_budget));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  int blocks = resolve_route_blocks(num_routes, nblocks);
  preprocess_routes_kernel<<<blocks, TMA_IBGDA_THREADS_PER_BLOCK, 0, stream>>>(
      dst_ranks.data_ptr<int64_t>(),
      dst_rows.data_ptr<int64_t>(),
      probs.has_value() ? probs->data_ptr<float>() : nullptr,
      reinterpret_cast<olmo::tma_ibgda_ep::RouteRecord*>(
          route_records.data_ptr<int32_t>()),
      routes_per_rank.data_ptr<int64_t>(),
      route_ordinals.data_ptr<int64_t>(),
      errors.data_ptr<int32_t>(),
      num_routes,
      dst_ranks.size(1),
      static_cast<int32_t>(ep_world_size),
      static_cast<int32_t>(rank_capacity));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  finalize_route_preprocess_kernel<<<1, 1, 0, stream>>>(
      routes_per_rank.data_ptr<int64_t>(),
      rank_offsets.data_ptr<int64_t>(),
      overflow_by_rank.data_ptr<bool>(),
      static_cast<int32_t>(ep_world_size),
      static_cast<int32_t>(static_route_budget));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tma_ibgda_route_records_with_probs_launcher(
    const torch::Tensor& route_records,
    const torch::Tensor& probs,
    const torch::Tensor& out_records) {
  int64_t num_routes = route_records.size(0);
  if (num_routes == 0) {
    return;
  }
  constexpr int threads = 256;
  int blocks = static_cast<int>((num_routes + threads - 1) / threads);
  if (blocks > 1024) {
    blocks = 1024;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  route_records_with_probs_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const olmo::tma_ibgda_ep::RouteRecord*>(
          route_records.data_ptr<int32_t>()),
      probs.data_ptr<float>(),
      reinterpret_cast<olmo::tma_ibgda_ep::RouteRecord*>(
          out_records.data_ptr<int32_t>()),
      num_routes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tma_ibgda_dispatch_bf16_peer_launcher(
    const torch::Tensor& input,
    const torch::Tensor& out,
    const torch::Tensor& dst_ranks,
    const torch::Tensor& dst_rows,
    const torch::Tensor& peer_out_ptrs,
    const std::optional<torch::Tensor>& probs,
    int64_t nblocks) {
  int64_t num_input_rows = input.size(0);
  int64_t top_k = dst_ranks.size(1);
  int64_t hidden = input.size(1);
  int64_t num_routes = num_input_rows * top_k;
  int blocks = resolve_route_blocks(num_routes, nblocks);
  auto stream = at::cuda::getCurrentCUDAStream();
  dispatch_bf16_peer_kernel<<<blocks, TMA_IBGDA_THREADS_PER_BLOCK, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
      input.stride(0),
      num_input_rows,
      top_k,
      hidden,
      dst_ranks.data_ptr<int64_t>(),
      dst_rows.data_ptr<int64_t>(),
      probs.has_value() ? probs->data_ptr<float>() : nullptr,
      peer_out_ptrs.data_ptr<int64_t>(),
      out.stride(0),
      out.size(0),
      peer_out_ptrs.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tma_ibgda_combine_bf16_peer_launcher(
    const torch::Tensor& expert_out,
    const torch::Tensor& out,
    const torch::Tensor& src_ranks,
    const torch::Tensor& src_rows,
    const torch::Tensor& peer_expert_out_ptrs,
    const std::optional<torch::Tensor>& probs) {
  constexpr int threads = 256;
  int64_t num_out_rows = out.size(0);
  int64_t hidden = out.size(1);
  dim3 grid(
      static_cast<unsigned int>(num_out_rows),
      static_cast<unsigned int>((hidden + threads - 1) / threads));
  auto stream = at::cuda::getCurrentCUDAStream();
  combine_bf16_peer_kernel<<<grid, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(expert_out.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      out.stride(0),
      num_out_rows,
      src_ranks.size(1),
      hidden,
      src_ranks.data_ptr<int64_t>(),
      src_rows.data_ptr<int64_t>(),
      probs.has_value() ? probs->data_ptr<float>() : nullptr,
      peer_expert_out_ptrs.data_ptr<int64_t>(),
      expert_out.stride(0),
      expert_out.size(0),
      peer_expert_out_ptrs.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tma_ibgda_dispatch_bf16_ibgda_launcher(
    const torch::Tensor& input,
    const torch::Tensor& out,
    const torch::Tensor& dst_ranks,
    const torch::Tensor& dst_rows,
    const std::optional<torch::Tensor>& probs,
    int64_t nblocks) {
  int64_t num_input_rows = input.size(0);
  int64_t top_k = dst_ranks.size(1);
  int64_t num_routes = num_input_rows * top_k;
  int blocks = resolve_route_blocks(num_routes, nblocks);
  auto stream = at::cuda::getCurrentCUDAStream();
  maybe_init_nvshmem_cumodule(reinterpret_cast<const void*>(dispatch_bf16_ibgda_kernel));
  dispatch_bf16_ibgda_kernel<<<blocks, TMA_IBGDA_THREADS_PER_BLOCK, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      input.stride(0),
      out.stride(0),
      num_input_rows,
      top_k,
      input.size(1),
      dst_ranks.data_ptr<int64_t>(),
      dst_rows.data_ptr<int64_t>(),
      probs.has_value() ? probs->data_ptr<float>() : nullptr,
      out.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tma_ibgda_dispatch_bf16_ibgda_records_launcher(
    const torch::Tensor& input,
    const torch::Tensor& out,
    const torch::Tensor& route_records,
    int64_t nblocks) {
  int64_t num_routes = route_records.size(0);
  int blocks = resolve_route_blocks(num_routes, nblocks);
  auto stream = at::cuda::getCurrentCUDAStream();
  maybe_init_nvshmem_cumodule(
      reinterpret_cast<const void*>(dispatch_bf16_ibgda_records_kernel));
  dispatch_bf16_ibgda_records_kernel<<<blocks, TMA_IBGDA_THREADS_PER_BLOCK, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      input.stride(0),
      out.stride(0),
      num_routes,
      input.size(1),
      reinterpret_cast<const olmo::tma_ibgda_ep::RouteRecord*>(
          route_records.data_ptr<int32_t>()),
      out.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tma_ibgda_dispatch_bf16_ibgda_records_tma_launcher(
    const torch::Tensor& input,
    const torch::Tensor& out,
    const torch::Tensor& route_records,
    int64_t nblocks) {
  int64_t num_routes = route_records.size(0);
  int blocks = resolve_route_blocks(num_routes, nblocks);
  auto stream = at::cuda::getCurrentCUDAStream();
  maybe_init_nvshmem_cumodule(
      reinterpret_cast<const void*>(dispatch_bf16_ibgda_records_tma_kernel));
  dispatch_bf16_ibgda_records_tma_kernel<<<blocks, TMA_IBGDA_THREADS_PER_BLOCK, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      input.stride(0),
      out.stride(0),
      num_routes,
      input.size(1),
      reinterpret_cast<const olmo::tma_ibgda_ep::RouteRecord*>(
          route_records.data_ptr<int32_t>()),
      out.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tma_ibgda_combine_bf16_ibgda_launcher(
    const torch::Tensor& expert_out,
    const torch::Tensor& out,
    const torch::Tensor& src_ranks,
    const torch::Tensor& src_rows,
    const std::optional<torch::Tensor>& probs,
    int64_t nblocks) {
  int64_t num_out_rows = out.size(0);
  int64_t top_k = src_ranks.size(1);
  int64_t hidden = out.size(1);
  if (num_out_rows == 0) {
    return;
  }
  constexpr int chunk_elems = TMA_IBGDA_WARP_SIZE * 4;
  int64_t num_hidden_chunks = (hidden + chunk_elems - 1) / chunk_elems;
  int64_t num_tiles = num_out_rows * num_hidden_chunks;
  int blocks = resolve_route_blocks(num_tiles, nblocks);
  auto stream = at::cuda::getCurrentCUDAStream();
  maybe_init_nvshmem_cumodule(reinterpret_cast<const void*>(combine_bf16_ibgda_fused_kernel));
  combine_bf16_ibgda_fused_kernel<<<blocks, TMA_IBGDA_THREADS_PER_BLOCK, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(expert_out.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      out.stride(0),
      expert_out.stride(0),
      num_out_rows,
      top_k,
      hidden,
      src_ranks.data_ptr<int64_t>(),
      src_rows.data_ptr<int64_t>(),
      probs.has_value() ? probs->data_ptr<float>() : nullptr,
      expert_out.size(0),
      num_hidden_chunks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tma_ibgda_combine_bf16_ibgda_records_launcher(
    const torch::Tensor& expert_out,
    const torch::Tensor& out,
    const torch::Tensor& route_records,
    int64_t top_k,
    int64_t nblocks) {
  int64_t num_out_rows = out.size(0);
  int64_t hidden = out.size(1);
  if (num_out_rows == 0) {
    return;
  }
  constexpr int chunk_elems = TMA_IBGDA_WARP_SIZE * 4;
  int64_t num_hidden_chunks = (hidden + chunk_elems - 1) / chunk_elems;
  int64_t num_tiles = num_out_rows * num_hidden_chunks;
  int blocks = resolve_route_blocks(num_tiles, nblocks);
  auto stream = at::cuda::getCurrentCUDAStream();
  maybe_init_nvshmem_cumodule(
      reinterpret_cast<const void*>(combine_bf16_ibgda_records_fused_kernel));
  combine_bf16_ibgda_records_fused_kernel<<<blocks, TMA_IBGDA_THREADS_PER_BLOCK, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(expert_out.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      out.stride(0),
      expert_out.stride(0),
      num_out_rows,
      top_k,
      hidden,
      reinterpret_cast<const olmo::tma_ibgda_ep::RouteRecord*>(
          route_records.data_ptr<int32_t>()),
      expert_out.size(0),
      num_hidden_chunks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tma_ibgda_route_dot_bf16_peer_launcher(
    const torch::Tensor& expert_out,
    const torch::Tensor& grad_out,
    const torch::Tensor& src_ranks,
    const torch::Tensor& src_rows,
    const torch::Tensor& peer_expert_out_ptrs,
    const torch::Tensor& out) {
  constexpr int threads = 256;
  int64_t num_grad_rows = grad_out.size(0);
  int64_t top_k = src_ranks.size(1);
  int64_t num_routes = num_grad_rows * top_k;
  auto stream = at::cuda::getCurrentCUDAStream();
  route_dot_bf16_peer_kernel<<<static_cast<unsigned int>(num_routes), threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(expert_out.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(grad_out.data_ptr()),
      grad_out.stride(0),
      num_grad_rows,
      top_k,
      grad_out.size(1),
      src_ranks.data_ptr<int64_t>(),
      src_rows.data_ptr<int64_t>(),
      peer_expert_out_ptrs.data_ptr<int64_t>(),
      expert_out.stride(0),
      expert_out.size(0),
      peer_expert_out_ptrs.numel(),
      out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tma_ibgda_route_dot_bf16_ibgda_launcher(
    const torch::Tensor& expert_out,
    const torch::Tensor& grad_out,
    const torch::Tensor& src_ranks,
    const torch::Tensor& src_rows,
    const torch::Tensor& out) {
  int64_t num_grad_rows = grad_out.size(0);
  int64_t top_k = src_ranks.size(1);
  int64_t num_routes = num_grad_rows * top_k;
  if (num_routes == 0) {
    return;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  maybe_init_nvshmem_cumodule(reinterpret_cast<const void*>(route_dot_bf16_ibgda_kernel));
  route_dot_bf16_ibgda_kernel<<<static_cast<unsigned int>(num_routes), TMA_IBGDA_WARP_SIZE, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(expert_out.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(grad_out.data_ptr()),
      grad_out.stride(0),
      num_grad_rows,
      top_k,
      grad_out.size(1),
      src_ranks.data_ptr<int64_t>(),
      src_rows.data_ptr<int64_t>(),
      expert_out.stride(0),
      expert_out.size(0),
      out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tma_ibgda_route_dot_bf16_ibgda_records_launcher(
    const torch::Tensor& expert_out,
    const torch::Tensor& grad_out,
    const torch::Tensor& route_records,
    int64_t top_k,
    const torch::Tensor& out) {
  int64_t num_routes = route_records.size(0);
  if (num_routes == 0) {
    return;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  maybe_init_nvshmem_cumodule(
      reinterpret_cast<const void*>(route_dot_bf16_ibgda_records_kernel));
  route_dot_bf16_ibgda_records_kernel<<<
      static_cast<unsigned int>(num_routes),
      TMA_IBGDA_WARP_SIZE,
      0,
      stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(expert_out.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(grad_out.data_ptr()),
      grad_out.stride(0),
      top_k,
      grad_out.size(1),
      reinterpret_cast<const olmo::tma_ibgda_ep::RouteRecord*>(
          route_records.data_ptr<int32_t>()),
      num_routes,
      expert_out.stride(0),
      expert_out.size(0),
      out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#include <ATen/ceil_div.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nvshmem_team_manager.hpp>

#include <algorithm>
#include <cub/cub.cuh>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// NVSHMEM minimum SM arch
#define _NVSHMEM_MIN_SM_ARCH 700

// If CUDA_ARCH is less than sm_70, or on sm_110, skip NVSHMEM device APIs
#define _NVSHMEM_DEVICELIB_SUPPORTED 1
#if defined(__CUDA_ARCH__)
#if (__CUDA_ARCH__ < _NVSHMEM_MIN_SM_ARCH) || (__CUDA_ARCH__ == 1100)
#undef _NVSHMEM_DEVICELIB_SUPPORTED
#endif
#endif

// Some NVSHMEM device APIs do not compile on older SM archs
#ifndef _NVSHMEM_DEVICELIB_SUPPORTED
#define NVSHMEM_HOSTLIB_ONLY
#endif

#include <nvshmem.h>
#include <nvshmemx.h>

#define THREADS_PER_BLOCK 512
#define WARP_SIZE 32
#define ROWWISE_THREADS_PER_BLOCK 512
#define ROWWISE_WARPS_PER_BLOCK (ROWWISE_THREADS_PER_BLOCK / WARP_SIZE)
#define ROWWISE_COMBINE_FUSED_THREADS_PER_BLOCK 256
#define ROWWISE_COMBINE_FUSED_WARPS_PER_BLOCK \
  (ROWWISE_COMBINE_FUSED_THREADS_PER_BLOCK / WARP_SIZE)
#define ROWWISE_COMBINE_FUSED_VECS_PER_THREAD 16
#define ROWWISE_WEIGHTED_PUT_ELEMS_PER_THREAD 32

namespace {

struct OlmoSymmGroupInfo {
  int world_size = 0;
  int* rank_to_pe_dev = nullptr;
  std::vector<int> rank_to_pe_host;
};

struct OlmoSymmState {
  bool initialized = false;
  int rank = -1;
  int world_size = -1;
  int device_idx = -1;
  std::mutex mutex;
  std::vector<void*> allocations;
  std::unordered_map<std::string, OlmoSymmGroupInfo> groups;
};

OlmoSymmState& olmo_symm_state() {
  static OlmoSymmState state;
  return state;
}

void olmo_maybe_initialize_env_vars() {
  const char* nccl_socket_if_name = std::getenv("NCCL_SOCKET_IFNAME");
  const char* nccl_hca_list = std::getenv("NCCL_IB_HCA");
  const char* nccl_ib_gid_index = std::getenv("NCCL_IB_GID_INDEX");

  if (std::getenv("NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME") == nullptr &&
      nccl_socket_if_name != nullptr) {
    setenv("NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME", nccl_socket_if_name, 0);
  }
  if (std::getenv("NVSHMEM_HCA_LIST") == nullptr && nccl_hca_list != nullptr) {
    setenv("NVSHMEM_ENABLE_NIC_PE_MAPPING", "1", 0);
    setenv("NVSHMEM_HCA_LIST", nccl_hca_list, 0);
  }
  if (std::getenv("NVSHMEM_IB_GID_INDEX") == nullptr &&
      nccl_ib_gid_index != nullptr) {
    setenv("NVSHMEM_IB_GID_INDEX", nccl_ib_gid_index, 0);
  }
}

OlmoSymmGroupInfo* olmo_symm_find_group(const std::string& group_name) {
  auto& state = olmo_symm_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  auto it = state.groups.find(group_name);
  if (it == state.groups.end()) {
    return nullptr;
  }
  return &it->second;
}

int olmo_symm_group_local_rank(const OlmoSymmGroupInfo& group, int global_rank) {
  for (size_t rank_idx = 0; rank_idx < group.rank_to_pe_host.size(); ++rank_idx) {
    if (group.rank_to_pe_host[rank_idx] == global_rank) {
      return static_cast<int>(rank_idx);
    }
  }
  TORCH_CHECK(
      false,
      "Current NVSHMEM PE ",
      global_rank,
      " is not a member of the requested OLMo symmetric-memory group");
  return -1;
}

__device__ __forceinline__ int olmo_route_npes(
    nvshmem_team_t team,
    const int* rank_to_pe,
    int group_size) {
  return rank_to_pe == nullptr ? nvshmem_team_n_pes(team) : group_size;
}

__device__ __forceinline__ int olmo_route_peer_global(
    nvshmem_team_t team,
    const int* rank_to_pe,
    int peer) {
  return rank_to_pe == nullptr
      ? nvshmem_team_translate_pe(team, peer, NVSHMEM_TEAM_WORLD)
      : rank_to_pe[peer];
}

__device__ __forceinline__ int64_t olmo_atomic_add_i64(
    int64_t* ptr,
    int64_t value) {
  return static_cast<int64_t>(atomicAdd(
      reinterpret_cast<unsigned long long*>(ptr),
      static_cast<unsigned long long>(value)));
}

__device__ __forceinline__ uint64_t olmo_load_acquire_sys_u64(
    const uint64_t* ptr) {
  uint64_t ret;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
  return ret;
}

__device__ int64_t prefixSum(int64_t* odata, int64_t* idata, int n) {
  using BlockScanT =
      cub::BlockScan<int64_t, THREADS_PER_BLOCK, cub::BLOCK_SCAN_WARP_SCANS>;
  __shared__ typename BlockScanT::TempStorage temp_storage;

  CUDA_KERNEL_ASSERT(n <= THREADS_PER_BLOCK);

  int tid = threadIdx.x;
  int64_t thread_data = (tid < n) ? idata[tid] : 0;

  int64_t block_aggregate;
  BlockScanT(temp_storage).ExclusiveSum(thread_data, thread_data, block_aggregate);

  odata[tid] = thread_data;
  return block_aggregate;
}

template <int NUM_WARPS>
__device__ int64_t prefixSum_warp(int64_t* odata, int64_t* idata, int n) {
  CUDA_KERNEL_ASSERT(n <= WARP_SIZE);

  using WarpScan = cub::WarpScan<int64_t>;
  __shared__ typename WarpScan::TempStorage temp_storage[NUM_WARPS];

  int warp_id = threadIdx.x / WARP_SIZE;
  if (warp_id >= NUM_WARPS) {
    return 0;
  }

  int tid = threadIdx.x % WARP_SIZE;
  int64_t thread_data = (tid < n) ? idata[tid] : 0;

  int64_t warp_aggregate;
  WarpScan(temp_storage[warp_id]).ExclusiveSum(
      thread_data, thread_data, warp_aggregate);

  // Fix OOB write when n < WARP_SIZE.
  if (tid < n) {
    odata[tid] = thread_data;
  }
  return warp_aggregate;
}

__global__ void waitSignalPeersKernel(
    uint64_t* signals,
    int64_t signal_row,
    int64_t group_size,
    uint64_t generation) {
  int64_t peer = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (peer >= group_size) {
    return;
  }
  uint64_t* slot =
      signals + static_cast<size_t>(signal_row) * static_cast<size_t>(group_size) +
      static_cast<size_t>(peer);
  while (olmo_load_acquire_sys_u64(slot) < generation) {
    __nanosleep(100);
  }
}

template <bool HAS_IN_OFFSETS>
__global__ void exchangeSplitAndOffset_2d(
    int64_t* in_splits_offsets,
    int64_t* out_splits_offsets,
    nvshmem_team_t team,
    int ne,
    size_t input_dim0,
    bool rank_is_row_in) {
#ifndef _NVSHMEM_DEVICELIB_SUPPORTED
  CUDA_KERNEL_ASSERT_MSG(false, "SM arch unsupported for NVSHMEM");
#else
  CUDA_KERNEL_ASSERT(team != NVSHMEM_TEAM_INVALID);
  int mype = nvshmem_team_my_pe(team);
  int npes = nvshmem_team_n_pes(team);
  int nsplits = npes * ne;
  auto input_splits = in_splits_offsets;
  auto output_splits = out_splits_offsets;
  auto source_offsets = out_splits_offsets + nsplits;
  int tid = threadIdx.x;

  int64_t* input_offsets = nullptr;
  if (HAS_IN_OFFSETS) {
    input_offsets = in_splits_offsets + nsplits;
  } else {
    __shared__ int64_t peer_offsets[THREADS_PER_BLOCK];
    auto sum_of_splits = prefixSum(peer_offsets, input_splits, nsplits);
    __syncthreads();
    CUDA_KERNEL_ASSERT(sum_of_splits <= input_dim0 && "sum of splits is larger than input dim\n");
    input_offsets = peer_offsets;
  }

  if (tid < nsplits) {
    int peer;
    int dst_offset;
    if (rank_is_row_in) {
      peer = tid / ne;
      int e = tid % ne;
      dst_offset = e * npes + mype;
    } else {
      peer = tid % npes;
      int e = tid / npes;
      dst_offset = mype * ne + e;
    }

    auto split_val = input_splits[tid];
    CUDA_KERNEL_ASSERT(split_val >= 0 && "split value is negative\n");
    auto peer_global = nvshmem_team_translate_pe(team, peer, NVSHMEM_TEAM_WORLD);
    nvshmem_int64_p(source_offsets + dst_offset, input_offsets[tid], peer_global);
    nvshmem_int64_p(output_splits + dst_offset, split_val, peer_global);
  }
  nvshmemx_barrier_block(team);
#endif
}

#define A2AV_TILE_SIZE WARP_SIZE

__global__ void allToAllV_2d(
    void* send_data,
    void* recv_data,
    int64_t* in_splits,
    int64_t* out_splits_offsets,
    size_t stride,
    int minor_size,
    int major_size,
    int64_t major_align,
    bool rank_is_row_out,
    nvshmem_team_t team) {
#ifndef _NVSHMEM_DEVICELIB_SUPPORTED
  CUDA_KERNEL_ASSERT_MSG(false, "SM arch unsupported for NVSHMEM");
#else
  int nsplits = minor_size * major_size;
  auto output_splits = out_splits_offsets;
  auto source_offsets = out_splits_offsets + nsplits;
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  constexpr int NUM_TILES = THREADS_PER_BLOCK / A2AV_TILE_SIZE;
  int tile_id = tid / A2AV_TILE_SIZE;
  int lane_id = tid % A2AV_TILE_SIZE;

  __shared__ int64_t tile_prefix_sums[NUM_TILES][A2AV_TILE_SIZE];

  int remaining = nsplits - tile_id * minor_size;
  int nsplits_per_tile = remaining > 0 ? min(minor_size, remaining) : 0;

  CUDA_KERNEL_ASSERT(minor_size <= A2AV_TILE_SIZE && "minor_size is too large\n");
  CUDA_KERNEL_ASSERT(major_size <= NUM_TILES && "major_size is too large\n");

  __shared__ int64_t len_per_tile[NUM_TILES];
  // Fix uninitialized len_per_tile entries for inactive tiles.
  if (lane_id == 0) {
    len_per_tile[tile_id] = 0;
  }

  if (nsplits_per_tile > 0) {
    int64_t my_tile_len = prefixSum_warp<NUM_TILES>(
        tile_prefix_sums[tile_id],
        output_splits + tile_id * minor_size,
        nsplits_per_tile);
    if (lane_id == A2AV_TILE_SIZE - 1) {
      if (major_align != 0) {
        auto aligned_len =
            (my_tile_len + major_align - 1) / major_align * major_align;
        len_per_tile[tile_id] = max(aligned_len, major_align);
      } else {
        len_per_tile[tile_id] = my_tile_len;
      }
    }
  }
  __syncthreads();

  __shared__ int64_t start_offset_per_tile[NUM_TILES];
  static_assert(NUM_TILES <= WARP_SIZE);
  prefixSum_warp<1>(start_offset_per_tile, len_per_tile, NUM_TILES);
  __syncthreads();

  if (lane_id < nsplits_per_tile) {
    tile_prefix_sums[tile_id][lane_id] += start_offset_per_tile[tile_id];
  }
  __syncthreads();

  // Parallelize each split over multiple blocks to improve bandwidth when
  // nsplits is small (e.g., ne=1) and payload is large.
  int blocks_per_split = max(gridDim.x / nsplits, 1);
  int split_groups = max(gridDim.x / blocks_per_split, 1);
  int block_in_split = bid % blocks_per_split;

  for (int eid = bid / blocks_per_split; eid < nsplits; eid += split_groups) {
    int row = eid / minor_size;
    int col = eid % minor_size;
    size_t peer_size = static_cast<size_t>(output_splits[eid]) * stride;
    if (peer_size == 0) {
      continue;
    }
    size_t chunk_start =
        (peer_size * static_cast<size_t>(block_in_split)) /
        static_cast<size_t>(blocks_per_split);
    size_t chunk_end =
        (peer_size * static_cast<size_t>(block_in_split + 1)) /
        static_cast<size_t>(blocks_per_split);
    size_t chunk_size = chunk_end - chunk_start;
    if (chunk_size == 0) {
      continue;
    }

    size_t source_offset = static_cast<size_t>(source_offsets[eid]) * stride + chunk_start;
    auto e_offset = tile_prefix_sums[row][col];
    size_t write_offset = static_cast<size_t>(e_offset) * stride + chunk_start;
    auto peer_global = nvshmem_team_translate_pe(
        team, rank_is_row_out ? row : col, NVSHMEM_TEAM_WORLD);
    nvshmemx_getmem_nbi_block(
        (char*)recv_data + write_offset,
        (char*)send_data + source_offset,
        chunk_size,
        peer_global);
  }

  if (bid == 0 && tid < nsplits) {
    source_offsets[tid] = tile_prefix_sums[tid / minor_size][tid % minor_size];
  }

  nvshmem_quiet();
#endif
}

__global__ void dispatchRowsPut(
    const void* input_data,
    void* out_data,
    const int64_t* dst_ranks,
    const int64_t* dst_rows,
    size_t row_bytes,
    int64_t num_input_rows,
    int64_t top_k,
    int64_t out_capacity_rows,
    nvshmem_team_t team,
    const int* rank_to_pe,
    int group_size) {
#ifndef _NVSHMEM_DEVICELIB_SUPPORTED
  CUDA_KERNEL_ASSERT_MSG(false, "SM arch unsupported for NVSHMEM");
#else
  CUDA_KERNEL_ASSERT(team != NVSHMEM_TEAM_INVALID);
  int npes = olmo_route_npes(team, rank_to_pe, group_size);
  int64_t num_routes = num_input_rows * top_k;
  int warp_id = threadIdx.x / WARP_SIZE;

  int64_t route_id = static_cast<int64_t>(blockIdx.x) * ROWWISE_WARPS_PER_BLOCK + warp_id;
  int64_t route_stride =
      static_cast<int64_t>(gridDim.x) * ROWWISE_WARPS_PER_BLOCK;
  for (; route_id < num_routes; route_id += route_stride) {
    int64_t peer = dst_ranks[route_id];
    int64_t dst_row = dst_rows[route_id];
    if (peer < 0 || dst_row < 0) {
      continue;
    }

    CUDA_KERNEL_ASSERT(peer < npes);
    CUDA_KERNEL_ASSERT(dst_row < out_capacity_rows);

    int64_t src_row = route_id / top_k;
    auto peer_global = olmo_route_peer_global(team, rank_to_pe, static_cast<int>(peer));
    nvshmemx_putmem_warp(
        (char*)out_data + static_cast<size_t>(dst_row) * row_bytes,
        (const char*)input_data + static_cast<size_t>(src_row) * row_bytes,
        row_bytes,
        peer_global);
  }
#endif
}

template <typename route_expert_t>
__global__ void countCompactRowwiseRoutes(
    const int64_t* dst_ranks,
    const int64_t* dst_rows,
    const route_expert_t* route_experts,
    int64_t* wave_counts,
    int64_t num_routes,
    int64_t top_k,
    int64_t num_local_experts,
    int64_t num_waves,
    int64_t experts_per_wave) {
  int64_t route_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (; route_id < num_routes; route_id += stride) {
    int64_t peer = dst_ranks[route_id];
    int64_t dst_row = dst_rows[route_id];
    int64_t expert = static_cast<int64_t>(route_experts[route_id]);
    if (peer < 0 || dst_row < 0 || expert < 0) {
      continue;
    }
    int64_t local_expert = expert % num_local_experts;
    int64_t wave = local_expert / experts_per_wave;
    wave = wave >= num_waves ? num_waves - 1 : wave;
    olmo_atomic_add_i64(wave_counts + wave, 1);
  }
}

__global__ void prefixCompactRowwiseRouteCounts(
    const int64_t* wave_counts,
    int64_t* wave_fill_counts,
    int64_t* wave_offsets,
    int64_t num_waves) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  int64_t acc = 0;
  wave_offsets[0] = 0;
  for (int64_t wave = 0; wave < num_waves; ++wave) {
    wave_fill_counts[wave] = 0;
    acc += wave_counts[wave];
    wave_offsets[wave + 1] = acc;
  }
}

template <typename route_expert_t>
__global__ void fillCompactRowwiseRoutes(
    const int64_t* dst_ranks,
    const int64_t* dst_rows,
    const route_expert_t* route_experts,
    int64_t* route_records,
    int64_t* wave_fill_counts,
    const int64_t* wave_offsets,
    int64_t num_routes,
    int64_t top_k,
    int64_t num_local_experts,
    int64_t num_waves,
    int64_t experts_per_wave) {
  int64_t route_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (; route_id < num_routes; route_id += stride) {
    int64_t peer = dst_ranks[route_id];
    int64_t dst_row = dst_rows[route_id];
    int64_t expert = static_cast<int64_t>(route_experts[route_id]);
    if (peer < 0 || dst_row < 0 || expert < 0) {
      continue;
    }
    int64_t local_expert = expert % num_local_experts;
    int64_t wave = local_expert / experts_per_wave;
    wave = wave >= num_waves ? num_waves - 1 : wave;
    int64_t compact_idx =
        wave_offsets[wave] + olmo_atomic_add_i64(wave_fill_counts + wave, 1);
    int64_t* record = route_records + compact_idx * 4;
    record[0] = route_id / top_k;
    record[1] = dst_row;
    record[2] = peer;
    record[3] = route_id;
  }
}

__global__ void dispatchRowsPutCompact(
    const void* input_data,
    void* out_data,
    const int64_t* route_records,
    const int64_t* wave_offsets,
    int64_t wave_idx,
    size_t row_bytes,
    int64_t num_input_rows,
    int64_t out_capacity_rows,
    nvshmem_team_t team,
    const int* rank_to_pe,
    int group_size) {
#ifndef _NVSHMEM_DEVICELIB_SUPPORTED
  CUDA_KERNEL_ASSERT_MSG(false, "SM arch unsupported for NVSHMEM");
#else
  CUDA_KERNEL_ASSERT(team != NVSHMEM_TEAM_INVALID);
  int npes = olmo_route_npes(team, rank_to_pe, group_size);
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;

  int64_t route_start = wave_offsets[wave_idx];
  int64_t route_end = wave_offsets[wave_idx + 1];
  int64_t local_idx =
      static_cast<int64_t>(blockIdx.x) * ROWWISE_WARPS_PER_BLOCK + warp_id;
  int64_t stride = static_cast<int64_t>(gridDim.x) * ROWWISE_WARPS_PER_BLOCK;
  for (; route_start + local_idx < route_end; local_idx += stride) {
    const int64_t* record = route_records + (route_start + local_idx) * 4;
    int64_t src_row = record[0];
    int64_t dst_row = record[1];
    int64_t peer = record[2];
    CUDA_KERNEL_ASSERT(src_row >= 0 && src_row < num_input_rows);
    CUDA_KERNEL_ASSERT(dst_row >= 0 && dst_row < out_capacity_rows);
    CUDA_KERNEL_ASSERT(peer >= 0 && peer < npes);

    int peer_global = 0;
    if (lane_id == 0) {
      peer_global = olmo_route_peer_global(team, rank_to_pe, static_cast<int>(peer));
    }
    peer_global = __shfl_sync(0xffffffff, peer_global, 0);
    nvshmemx_putmem_warp(
        (char*)out_data + static_cast<size_t>(dst_row) * row_bytes,
        (const char*)input_data + static_cast<size_t>(src_row) * row_bytes,
        row_bytes,
        peer_global);
  }
#endif
}

template <typename scalar_t>
__global__ void dispatchRowsPutCompactWeighted(
    const scalar_t* input_data,
    scalar_t* out_data,
    const int64_t* route_records,
    const int64_t* wave_offsets,
    int64_t wave_idx,
    const float* probs,
    int64_t top_k,
    int64_t dim,
    int64_t input_row_stride,
    int64_t out_row_stride,
    int64_t num_input_rows,
    int64_t out_capacity_rows,
    nvshmem_team_t team,
    const int* rank_to_pe,
    int group_size) {
#ifndef _NVSHMEM_DEVICELIB_SUPPORTED
  CUDA_KERNEL_ASSERT_MSG(false, "SM arch unsupported for NVSHMEM");
#else
  CUDA_KERNEL_ASSERT(team != NVSHMEM_TEAM_INVALID);
  constexpr int ELEMS_PER_THREAD = ROWWISE_WEIGHTED_PUT_ELEMS_PER_THREAD;
  constexpr int CHUNK_ELEMS = WARP_SIZE * ELEMS_PER_THREAD;
  __shared__ scalar_t shared_rows[ROWWISE_WARPS_PER_BLOCK][CHUNK_ELEMS];

  int npes = olmo_route_npes(team, rank_to_pe, group_size);
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;

  int64_t route_start = wave_offsets[wave_idx];
  int64_t route_end = wave_offsets[wave_idx + 1];
  int64_t local_idx =
      static_cast<int64_t>(blockIdx.x) * ROWWISE_WARPS_PER_BLOCK + warp_id;
  int64_t stride = static_cast<int64_t>(gridDim.x) * ROWWISE_WARPS_PER_BLOCK;
  for (; route_start + local_idx < route_end; local_idx += stride) {
    const int64_t* record = route_records + (route_start + local_idx) * 4;
    int64_t src_row = record[0];
    int64_t dst_row = record[1];
    int64_t peer = record[2];
    int64_t route_id = record[3];
    CUDA_KERNEL_ASSERT(src_row >= 0 && src_row < num_input_rows);
    CUDA_KERNEL_ASSERT(dst_row >= 0 && dst_row < out_capacity_rows);
    CUDA_KERNEL_ASSERT(peer >= 0 && peer < npes);

    float p = probs[route_id];
    int peer_global = 0;
    if (lane_id == 0) {
      peer_global = olmo_route_peer_global(team, rank_to_pe, static_cast<int>(peer));
    }
    peer_global = __shfl_sync(0xffffffff, peer_global, 0);

    const scalar_t* src_ptr = input_data + src_row * input_row_stride;
    scalar_t* shared_row = shared_rows[warp_id];
    for (int64_t col_base = 0; col_base < dim; col_base += CHUNK_ELEMS) {
      int64_t remaining = dim - col_base;
      int64_t chunk_elems = remaining < CHUNK_ELEMS ? remaining : CHUNK_ELEMS;
#pragma unroll
      for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        int elem = i * WARP_SIZE + lane_id;
        if (elem < chunk_elems) {
          float v = static_cast<float>(src_ptr[col_base + elem]);
          shared_row[elem] = static_cast<scalar_t>(v * p);
        }
      }
      __syncwarp();
      nvshmemx_putmem_warp(
          (char*)out_data +
              static_cast<size_t>(dst_row * out_row_stride + col_base) *
                  sizeof(scalar_t),
          shared_row,
          static_cast<size_t>(chunk_elems) * sizeof(scalar_t),
          peer_global);
      __syncwarp();
    }
  }
#endif
}

__global__ void inverseRouteMetaPutCompact(
    int64_t* inverse_route_meta,
    const int64_t* route_records,
    const int64_t* wave_offsets,
    int64_t num_waves,
    int64_t src_rank,
    int64_t inverse_capacity_rows,
    nvshmem_team_t team,
    const int* rank_to_pe,
    int group_size) {
#ifndef _NVSHMEM_DEVICELIB_SUPPORTED
  CUDA_KERNEL_ASSERT_MSG(false, "SM arch unsupported for NVSHMEM");
#else
  CUDA_KERNEL_ASSERT(team != NVSHMEM_TEAM_INVALID);
  __shared__ int64_t shared_meta[ROWWISE_WARPS_PER_BLOCK][2];
  int npes = olmo_route_npes(team, rank_to_pe, group_size);
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;

  int64_t route_end = wave_offsets[num_waves];
  int64_t compact_idx =
      static_cast<int64_t>(blockIdx.x) * ROWWISE_WARPS_PER_BLOCK + warp_id;
  int64_t stride = static_cast<int64_t>(gridDim.x) * ROWWISE_WARPS_PER_BLOCK;
  for (; compact_idx < route_end; compact_idx += stride) {
    const int64_t* record = route_records + compact_idx * 4;
    int64_t dst_row = record[1];
    int64_t peer = record[2];
    int64_t route_id = record[3];
    CUDA_KERNEL_ASSERT(dst_row >= 0 && dst_row < inverse_capacity_rows);
    CUDA_KERNEL_ASSERT(peer >= 0 && peer < npes);
    if (lane_id == 0) {
      shared_meta[warp_id][0] = src_rank;
    } else if (lane_id == 1) {
      shared_meta[warp_id][1] = route_id;
    }
    __syncwarp();

    auto peer_global = olmo_route_peer_global(team, rank_to_pe, static_cast<int>(peer));
    nvshmemx_putmem_warp(
        inverse_route_meta + dst_row * 2,
        shared_meta[warp_id],
        2 * sizeof(int64_t),
        peer_global);
  }
#endif
}

__global__ void combineRowsPutRange(
    const void* expert_out_data,
    void* gathered_data,
    const int64_t* inverse_route_meta,
    const int64_t* row_start_ptr,
    const int64_t* num_rows_ptr,
    size_t row_bytes,
    int64_t expert_capacity_rows,
    int64_t gathered_capacity_rows,
    nvshmem_team_t team,
    const int* rank_to_pe,
    int group_size) {
#ifndef _NVSHMEM_DEVICELIB_SUPPORTED
  CUDA_KERNEL_ASSERT_MSG(false, "SM arch unsupported for NVSHMEM");
#else
  CUDA_KERNEL_ASSERT(team != NVSHMEM_TEAM_INVALID);
  int npes = olmo_route_npes(team, rank_to_pe, group_size);
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;

  int64_t row_start = *row_start_ptr;
  int64_t num_rows = *num_rows_ptr;
  int64_t row_end = row_start + num_rows;
  CUDA_KERNEL_ASSERT(row_start >= 0);
  CUDA_KERNEL_ASSERT(num_rows >= 0);
  CUDA_KERNEL_ASSERT(row_end <= expert_capacity_rows);

  int64_t local_row =
      static_cast<int64_t>(blockIdx.x) * ROWWISE_WARPS_PER_BLOCK + warp_id;
  int64_t row_stride =
      static_cast<int64_t>(gridDim.x) * ROWWISE_WARPS_PER_BLOCK;
  for (; local_row < num_rows; local_row += row_stride) {
    int64_t row = row_start + local_row;
    int64_t peer = inverse_route_meta[row * 2];
    int64_t gathered_row = inverse_route_meta[row * 2 + 1];
    if (peer < 0 || gathered_row < 0) {
      continue;
    }

    CUDA_KERNEL_ASSERT(peer < npes);
    CUDA_KERNEL_ASSERT(gathered_row < gathered_capacity_rows);

    int peer_global = 0;
    if (lane_id == 0) {
      peer_global = olmo_route_peer_global(team, rank_to_pe, static_cast<int>(peer));
    }
    peer_global = __shfl_sync(0xffffffff, peer_global, 0);
    nvshmemx_putmem_warp(
        (char*)gathered_data + static_cast<size_t>(gathered_row) * row_bytes,
        (const char*)expert_out_data + static_cast<size_t>(row) * row_bytes,
        row_bytes,
        peer_global);
  }
#endif
}

template <typename scalar_t>
__global__ void dispatchRowsPutWeighted(
    const scalar_t* input_data,
    scalar_t* out_data,
    const int64_t* dst_ranks,
    const int64_t* dst_rows,
    const float* probs,
    int64_t num_input_rows,
    int64_t top_k,
    int64_t dim,
    int64_t input_row_stride,
    int64_t out_row_stride,
    int64_t out_capacity_rows,
    nvshmem_team_t team,
    const int* rank_to_pe,
    int group_size) {
#ifndef _NVSHMEM_DEVICELIB_SUPPORTED
  CUDA_KERNEL_ASSERT_MSG(false, "SM arch unsupported for NVSHMEM");
#else
  CUDA_KERNEL_ASSERT(team != NVSHMEM_TEAM_INVALID);
  constexpr int ELEMS_PER_THREAD = ROWWISE_WEIGHTED_PUT_ELEMS_PER_THREAD;
  constexpr int CHUNK_ELEMS = WARP_SIZE * ELEMS_PER_THREAD;
  __shared__ scalar_t shared_rows[ROWWISE_WARPS_PER_BLOCK][CHUNK_ELEMS];

  int npes = olmo_route_npes(team, rank_to_pe, group_size);
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;

  int64_t num_routes = num_input_rows * top_k;
  int64_t route_id =
      static_cast<int64_t>(blockIdx.x) * ROWWISE_WARPS_PER_BLOCK + warp_id;
  int64_t route_stride =
      static_cast<int64_t>(gridDim.x) * ROWWISE_WARPS_PER_BLOCK;

  for (; route_id < num_routes; route_id += route_stride) {
    int64_t peer = dst_ranks[route_id];
    int64_t dst_row = dst_rows[route_id];
    if (peer < 0 || dst_row < 0) {
      continue;
    }

    CUDA_KERNEL_ASSERT(peer < npes);
    CUDA_KERNEL_ASSERT(dst_row < out_capacity_rows);

    int64_t src_row = route_id / top_k;
    float p = probs[route_id];
    int peer_global = 0;
    if (lane_id == 0) {
      peer_global = olmo_route_peer_global(team, rank_to_pe, static_cast<int>(peer));
    }
    peer_global = __shfl_sync(0xffffffff, peer_global, 0);

    const scalar_t* src_ptr = input_data + src_row * input_row_stride;
    scalar_t* shared_row = shared_rows[warp_id];
    for (int64_t col_base = 0; col_base < dim; col_base += CHUNK_ELEMS) {
      int64_t remaining = dim - col_base;
      int64_t chunk_elems = remaining < CHUNK_ELEMS ? remaining : CHUNK_ELEMS;
#pragma unroll
      for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        int elem = i * WARP_SIZE + lane_id;
        if (elem < chunk_elems) {
          float v = static_cast<float>(src_ptr[col_base + elem]);
          shared_row[elem] = static_cast<scalar_t>(v * p);
        }
      }
      __syncwarp();
      nvshmemx_putmem_warp(
          (char*)out_data +
              static_cast<size_t>(dst_row * out_row_stride + col_base) *
                  sizeof(scalar_t),
          shared_row,
          static_cast<size_t>(chunk_elems) * sizeof(scalar_t),
          peer_global);
      __syncwarp();
    }
  }
#endif
}

template <bool ZERO_INVALID_ROWS>
__global__ void gatherRowsGet(
    const void* expert_out_data,
    void* gathered_data,
    const int64_t* src_ranks,
    const int64_t* src_rows,
    size_t row_bytes,
    int64_t num_out_rows,
    int64_t top_k,
    int64_t expert_capacity_rows,
    nvshmem_team_t team,
    const int* rank_to_pe,
    int group_size) {
#ifndef _NVSHMEM_DEVICELIB_SUPPORTED
  CUDA_KERNEL_ASSERT_MSG(false, "SM arch unsupported for NVSHMEM");
#else
  CUDA_KERNEL_ASSERT(team != NVSHMEM_TEAM_INVALID);
  int npes = olmo_route_npes(team, rank_to_pe, group_size);
  int64_t num_routes = num_out_rows * top_k;
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;

  int64_t route_id = static_cast<int64_t>(blockIdx.x) * ROWWISE_WARPS_PER_BLOCK + warp_id;
  int64_t route_stride =
      static_cast<int64_t>(gridDim.x) * ROWWISE_WARPS_PER_BLOCK;
  for (; route_id < num_routes; route_id += route_stride) {
    int64_t peer = src_ranks[route_id];
    int64_t src_row = src_rows[route_id];
    char* dst_ptr = (char*)gathered_data + static_cast<size_t>(route_id) * row_bytes;
    if (peer < 0 || src_row < 0) {
      if constexpr (ZERO_INVALID_ROWS) {
        for (size_t i = static_cast<size_t>(lane_id); i < row_bytes; i += WARP_SIZE) {
          dst_ptr[i] = 0;
        }
      }
      continue;
    }

    CUDA_KERNEL_ASSERT(peer < npes);
    CUDA_KERNEL_ASSERT(src_row < expert_capacity_rows);
    auto peer_global = olmo_route_peer_global(team, rank_to_pe, static_cast<int>(peer));
    nvshmemx_getmem_warp(
        dst_ptr,
        (const char*)expert_out_data + static_cast<size_t>(src_row) * row_bytes,
        row_bytes,
        peer_global);
  }
#endif
}

template <typename scalar_t, bool HAS_PROBS>
__global__ void combineRowsReduceKernel(
    const scalar_t* gathered,
    scalar_t* out,
    const float* probs,
    int64_t num_out_rows,
    int64_t top_k,
    int64_t dim) {
  int64_t row = blockIdx.x;
  int64_t col = static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
  if (row >= num_out_rows || col >= dim) {
    return;
  }

  float acc = 0.0f;
  int64_t base = row * top_k * dim + col;
  for (int64_t k = 0; k < top_k; ++k) {
    float v = static_cast<float>(gathered[base + k * dim]);
    if constexpr (HAS_PROBS) {
      v *= probs[row * top_k + k];
    }
    acc += v;
  }

  out[row * dim + col] = static_cast<scalar_t>(acc);
}

template <typename scalar_t, typename prob_t, bool HAS_ROUTE_RANKS>
__global__ void combineGatheredRowsWeightedReduceKernel(
    const scalar_t* gathered,
    const prob_t* probs,
    const int64_t* route_ranks,
    scalar_t* out,
    int64_t num_out_rows,
    int64_t top_k,
    int64_t dim) {
  int64_t row = blockIdx.x;
  int64_t col = static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
  if (row >= num_out_rows) {
    return;
  }

  if (col >= dim) {
    return;
  }

  float acc = 0.0f;
  int64_t base = row * top_k * dim + col;
  int64_t prob_base = row * top_k;
  for (int64_t k = 0; k < top_k; ++k) {
    if constexpr (HAS_ROUTE_RANKS) {
      if (route_ranks[prob_base + k] < 0) {
        continue;
      }
    }
    float p = static_cast<float>(probs[prob_base + k]);
    if (p == 0.0f) {
      continue;
    }
    acc += static_cast<float>(gathered[base + k * dim]) * p;
  }

  out[row * dim + col] = static_cast<scalar_t>(acc);
}

template <typename scalar_t, bool HAS_ROUTE_RANKS>
__global__ void combineGatheredRowsReduceKernel(
    const scalar_t* gathered,
    const int64_t* route_ranks,
    scalar_t* out,
    int64_t num_out_rows,
    int64_t top_k,
    int64_t dim) {
  int64_t row = blockIdx.x;
  int64_t col = static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
  if (row >= num_out_rows || col >= dim) {
    return;
  }

  float acc = 0.0f;
  int64_t base = row * top_k * dim + col;
  int64_t route_base = row * top_k;
  for (int64_t k = 0; k < top_k; ++k) {
    if constexpr (HAS_ROUTE_RANKS) {
      if (route_ranks[route_base + k] < 0) {
        continue;
      }
    }
    acc += static_cast<float>(gathered[base + k * dim]);
  }

  out[row * dim + col] = static_cast<scalar_t>(acc);
}

template <typename scalar_t, bool HAS_PROBS>
__global__ void combineRowsGetKernel(
    const scalar_t* expert_out,
    scalar_t* out,
    scalar_t* gathered_out,
    const int64_t* src_ranks,
    const int64_t* src_rows,
    const float* probs,
    int64_t num_out_rows,
    int64_t top_k,
    int64_t dim,
    int64_t expert_row_stride,
    int64_t out_row_stride,
    int64_t expert_capacity_rows,
    nvshmem_team_t team,
    const int* rank_to_pe,
    int group_size) {
#ifndef _NVSHMEM_DEVICELIB_SUPPORTED
  CUDA_KERNEL_ASSERT_MSG(false, "SM arch unsupported for NVSHMEM");
#else
  CUDA_KERNEL_ASSERT(team != NVSHMEM_TEAM_INVALID);
  constexpr int WARP_TILE_ELEMS =
      WARP_SIZE * ROWWISE_COMBINE_FUSED_VECS_PER_THREAD;
  constexpr int BLOCK_TILE_ELEMS =
      ROWWISE_COMBINE_FUSED_THREADS_PER_BLOCK *
      ROWWISE_COMBINE_FUSED_VECS_PER_THREAD;

  int npes = olmo_route_npes(team, rank_to_pe, group_size);
  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;

  int64_t block_col_base = static_cast<int64_t>(blockIdx.y) * BLOCK_TILE_ELEMS;
  int64_t warp_col_base = block_col_base + static_cast<int64_t>(warp_id) * WARP_TILE_ELEMS;
  int64_t warp_chunk_elems = 0;
  if (warp_col_base < dim) {
    warp_chunk_elems = dim - warp_col_base;
    if (warp_chunk_elems > WARP_TILE_ELEMS) {
      warp_chunk_elems = WARP_TILE_ELEMS;
    }
  }

  if (warp_chunk_elems == 0) {
    return;
  }

  __shared__
      scalar_t shared_rows[ROWWISE_COMBINE_FUSED_WARPS_PER_BLOCK][WARP_TILE_ELEMS];
  scalar_t* warp_shared_row = shared_rows[warp_id];

  for (int64_t row = blockIdx.x; row < num_out_rows; row += gridDim.x) {
    float acc[ROWWISE_COMBINE_FUSED_VECS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < ROWWISE_COMBINE_FUSED_VECS_PER_THREAD; ++i) {
      acc[i] = 0.0f;
    }

    int64_t route_base = row * top_k;
    for (int64_t k = 0; k < top_k; ++k) {
      int64_t route = route_base + k;
      int64_t peer = src_ranks[route];
      int64_t src_row = src_rows[route];
      if (peer < 0 || src_row < 0) {
        if (gathered_out != nullptr) {
          scalar_t* gathered_route_ptr =
              gathered_out + (row * top_k + k) * dim + warp_col_base;
#pragma unroll
          for (int i = 0; i < ROWWISE_COMBINE_FUSED_VECS_PER_THREAD; ++i) {
            int elem = i * WARP_SIZE + lane_id;
            if (elem < warp_chunk_elems) {
              gathered_route_ptr[elem] = static_cast<scalar_t>(0.0f);
            }
          }
        }
        continue;
      }

      CUDA_KERNEL_ASSERT(peer < npes);
      CUDA_KERNEL_ASSERT(src_row < expert_capacity_rows);

      int peer_global = 0;
      if (lane_id == 0) {
        peer_global = olmo_route_peer_global(team, rank_to_pe, static_cast<int>(peer));
      }
      peer_global = __shfl_sync(0xffffffff, peer_global, 0);

      nvshmemx_getmem_warp(
          warp_shared_row,
          expert_out + src_row * expert_row_stride + warp_col_base,
          static_cast<size_t>(warp_chunk_elems) * sizeof(scalar_t),
          peer_global);
      __syncwarp();

      float p = 1.0f;
      if constexpr (HAS_PROBS) {
        p = probs[route];
      }

#pragma unroll
      for (int i = 0; i < ROWWISE_COMBINE_FUSED_VECS_PER_THREAD; ++i) {
        int elem = i * WARP_SIZE + lane_id;
        if (elem < warp_chunk_elems) {
          float v = static_cast<float>(warp_shared_row[elem]);
          if (gathered_out != nullptr) {
            scalar_t* gathered_route_ptr =
                gathered_out + (row * top_k + k) * dim + warp_col_base;
            gathered_route_ptr[elem] = warp_shared_row[elem];
          }
          acc[i] += v * p;
        }
      }
    }

    scalar_t* out_ptr = out + row * out_row_stride + warp_col_base;
#pragma unroll
    for (int i = 0; i < ROWWISE_COMBINE_FUSED_VECS_PER_THREAD; ++i) {
      int elem = i * WARP_SIZE + lane_id;
      if (elem < warp_chunk_elems) {
        out_ptr[elem] = static_cast<scalar_t>(acc[i]);
      }
    }
  }
#endif
}

int resolve_num_blocks(int world_size, int ne, int64_t requested_nblocks) {
  if (requested_nblocks > 0) {
    TORCH_CHECK(
        requested_nblocks <= std::numeric_limits<int>::max(),
        "nblocks is too large");
    return static_cast<int>(requested_nblocks);
  }
  return 0;
}

int resolve_num_blocks_auto(
    size_t input_size_bytes,
    int world_size,
    int ne,
    bool intra_node) {
  constexpr size_t chunk_size = 16 * THREADS_PER_BLOCK * 8;
  int nsplits = world_size * ne;
  TORCH_CHECK(nsplits > 0, "nsplits must be > 0");

  int num_blocks = at::ceil_div(input_size_bytes, chunk_size);
  num_blocks = std::max(num_blocks, nsplits);
  num_blocks = at::round_up(num_blocks, nsplits);
  int max_blocks = intra_node ? 256 : 64;
  return std::min(num_blocks, max_blocks);
}

int resolve_num_blocks_rowwise(
    int64_t num_routes,
    int64_t requested_nblocks,
    bool intra_node) {
  if (requested_nblocks > 0) {
    TORCH_CHECK(
        requested_nblocks <= std::numeric_limits<int>::max(),
        "nblocks is too large");
    return static_cast<int>(requested_nblocks);
  }
  if (num_routes <= 0) {
    return 1;
  }

  auto* props = at::cuda::getCurrentDeviceProperties();
  int sm_count = std::max(props->multiProcessorCount, 1);
  int target_blocks = sm_count * 4;
  int max_blocks = intra_node ? 2048 : 512;
  int64_t capped = std::min<int64_t>(num_routes, max_blocks);
  return std::max<int>(1, static_cast<int>(std::min<int64_t>(target_blocks, capped)));
}

int64_t resolve_num_row_blocks_fused(int64_t num_out_rows, int num_blocks) {
  static int factor = []() {
    constexpr int kDefault = 16;
    constexpr int kMin = 1;
    constexpr int kMax = 64;
    const char* env = std::getenv("OLMO_ROWWISE_COMBINE_FUSED_ROW_BLOCK_FACTOR");
    if (env == nullptr || env[0] == '\0') {
      return kDefault;
    }
    char* end = nullptr;
    long parsed = std::strtol(env, &end, 10);
    if (end == env || *end != '\0') {
      return kDefault;
    }
    parsed = std::max<long>(parsed, kMin);
    parsed = std::min<long>(parsed, kMax);
    return static_cast<int>(parsed);
  }();

  int64_t row_blocks = std::min<int64_t>(
      num_out_rows, static_cast<int64_t>(num_blocks) * factor);
  return std::max<int64_t>(row_blocks, 1);
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
        "cudaGetFuncBySymbol failed while initializing NVSHMEM module: ",
        cudaGetErrorString(rt_status));

    CUmodule cu_module{};
    auto cu_status = cuFuncGetModule(
        &cu_module, reinterpret_cast<CUfunction>(cuda_func));
    TORCH_CHECK(
        cu_status == CUDA_SUCCESS,
        "cuFuncGetModule failed while initializing NVSHMEM module (",
        static_cast<int>(cu_status),
        "): ",
        cu_result_string(cu_status));

    int nv_status = nvshmemx_cumodule_init(cu_module);
    TORCH_CHECK(
        nv_status == 0,
        "nvshmemx_cumodule_init failed with status ",
        nv_status);
  });
}

} // namespace

std::vector<uint8_t> olmo_symm_get_unique_id() {
  nvshmemx_uniqueid_t unique_id;
  int status = nvshmemx_get_uniqueid(&unique_id);
  TORCH_CHECK(status == 0, "nvshmemx_get_uniqueid failed with status ", status);
  const auto* begin = reinterpret_cast<const uint8_t*>(&unique_id);
  return std::vector<uint8_t>(begin, begin + sizeof(nvshmemx_uniqueid_t));
}

void olmo_symm_init(
    const std::vector<std::vector<uint8_t>>& unique_ids,
    int64_t rank,
    int64_t world_size,
    int64_t device_idx) {
  auto& state = olmo_symm_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.initialized) {
    TORCH_CHECK(
        state.rank == rank && state.world_size == world_size &&
            state.device_idx == device_idx,
        "OLMo symmetric memory is already initialized with rank=",
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
  olmo_maybe_initialize_env_vars();
  AT_CUDA_CHECK(cudaFree(nullptr));

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

at::Tensor olmo_symm_empty(
    const std::vector<int64_t>& sizes,
    c10::ScalarType dtype,
    c10::Device device) {
  auto& state = olmo_symm_state();
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    TORCH_CHECK(state.initialized, "OLMo symmetric memory is not initialized");
  }
  TORCH_CHECK(device.is_cuda(), "OLMo symmetric memory tensors must be CUDA tensors");
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

at::Tensor olmo_symm_peer_base_ptrs(
    at::Tensor& tensor,
    const std::string& group_name) {
  auto& state = olmo_symm_state();
  std::vector<int> rank_to_pe;
  int device_idx = -1;
  int my_pe = -1;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    TORCH_CHECK(state.initialized, "OLMo symmetric memory is not initialized");
    auto it = state.groups.find(group_name);
    TORCH_CHECK(
        it != state.groups.end(),
        "OLMo symmetric-memory group ",
        group_name,
        " is not registered");
    rank_to_pe = it->second.rank_to_pe_host;
    device_idx = state.device_idx;
    my_pe = state.rank;
  }

  TORCH_CHECK(tensor.is_cuda(), "symmetric tensor must be CUDA");
  TORCH_CHECK(tensor.numel() > 0, "symmetric tensor must be non-empty");
  TORCH_CHECK(
      tensor.get_device() == device_idx,
      "symmetric tensor must be on the NVSHMEM bootstrap device ",
      device_idx,
      ", got ",
      tensor.get_device());
  c10::cuda::CUDAGuard guard(tensor.device());

  void* local_ptr = tensor.mutable_data_ptr();
  std::vector<int64_t> host_ptrs(rank_to_pe.size(), 0);
  for (size_t rank_idx = 0; rank_idx < rank_to_pe.size(); ++rank_idx) {
    const int peer = rank_to_pe[rank_idx];
    void* peer_ptr = peer == my_pe ? local_ptr : nvshmem_ptr(local_ptr, peer);
    TORCH_CHECK(
        peer_ptr != nullptr,
        "NVSHMEM symmetric allocation for group ",
        group_name,
        " is not directly addressable for group rank ",
        rank_idx,
        " (PE ",
        peer,
        "). The BF16 wave peer-window path currently supports only "
        "directly peer-visible intra-node workspaces.");
    host_ptrs[rank_idx] = static_cast<int64_t>(reinterpret_cast<uintptr_t>(peer_ptr));
  }

  auto out = at::empty(
      {static_cast<int64_t>(host_ptrs.size())},
      at::TensorOptions().device(tensor.device()).dtype(at::kLong));
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_CUDA_CHECK(cudaMemcpyAsync(
      out.mutable_data_ptr<int64_t>(),
      host_ptrs.data(),
      host_ptrs.size() * sizeof(int64_t),
      cudaMemcpyHostToDevice,
      stream.stream()));
  return out;
}

void olmo_symm_register_group(
    const std::string& group_name,
    const std::vector<int64_t>& rank_to_pe) {
  auto& state = olmo_symm_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  TORCH_CHECK(state.initialized, "OLMo symmetric memory is not initialized");
  TORCH_CHECK(!rank_to_pe.empty(), "rank_to_pe must not be empty");

  auto existing = state.groups.find(group_name);
  if (existing != state.groups.end()) {
    TORCH_CHECK(
        existing->second.world_size == static_cast<int>(rank_to_pe.size()),
        "OLMo symmetric-memory group ",
        group_name,
        " was already registered with a different size");
    return;
  }

  std::vector<int> host_rank_to_pe(rank_to_pe.size());
  for (size_t i = 0; i < rank_to_pe.size(); ++i) {
    TORCH_CHECK(
        rank_to_pe[i] >= 0 && rank_to_pe[i] < state.world_size,
        "rank_to_pe entry is outside the NVSHMEM bootstrap world: ",
        rank_to_pe[i]);
    host_rank_to_pe[i] = static_cast<int>(rank_to_pe[i]);
  }

  c10::cuda::CUDAGuard guard(state.device_idx);
  int* rank_to_pe_dev = nullptr;
  AT_CUDA_CHECK(cudaMalloc(&rank_to_pe_dev, sizeof(int) * host_rank_to_pe.size()));
  AT_CUDA_CHECK(cudaMemcpy(
      rank_to_pe_dev,
      host_rank_to_pe.data(),
      sizeof(int) * host_rank_to_pe.size(),
      cudaMemcpyHostToDevice));

  OlmoSymmGroupInfo info;
  info.world_size = static_cast<int>(host_rank_to_pe.size());
  info.rank_to_pe_dev = rank_to_pe_dev;
  info.rank_to_pe_host = std::move(host_rank_to_pe);
  state.groups.emplace(group_name, info);
}

bool olmo_symm_has_group(const std::string& group_name) {
  return olmo_symm_find_group(group_name) != nullptr;
}

void olmo_symm_world_barrier() {
  auto& state = olmo_symm_state();
  int device_idx = -1;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    TORCH_CHECK(state.initialized, "OLMo symmetric memory is not initialized");
    device_idx = state.device_idx;
  }

  c10::cuda::CUDAGuard guard(device_idx);
  auto stream = at::cuda::getCurrentCUDAStream();
  int barrier_status = nvshmemx_barrier_on_stream(NVSHMEM_TEAM_WORLD, stream.stream());
  TORCH_CHECK(
      barrier_status == 0,
      "nvshmemx_barrier_on_stream (world) failed with status ",
      barrier_status);
}

void rowwise_signal_peers_on_stream(
    at::Tensor& signals,
    int64_t signal_row,
    int64_t generation,
    const std::string& group_name,
    bool quiet_before_signal) {
  auto& state = olmo_symm_state();
  std::vector<int> rank_to_pe;
  int local_rank = -1;
  int device_idx = -1;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    TORCH_CHECK(state.initialized, "OLMo symmetric memory is not initialized");
    auto it = state.groups.find(group_name);
    TORCH_CHECK(
        it != state.groups.end(),
        "OLMo symmetric-memory group ",
        group_name,
        " is not registered");
    rank_to_pe = it->second.rank_to_pe_host;
    local_rank = olmo_symm_group_local_rank(it->second, state.rank);
    device_idx = state.device_idx;
  }

  TORCH_CHECK(signals.is_cuda(), "signals must be a CUDA tensor");
  TORCH_CHECK(signals.scalar_type() == at::kLong, "signals must be int64");
  TORCH_CHECK(signals.is_contiguous(), "signals must be contiguous");
  TORCH_CHECK(signals.dim() == 2, "signals must be rank-2 [rows, group_size]");
  TORCH_CHECK(
      signals.size(1) == static_cast<int64_t>(rank_to_pe.size()),
      "signals second dim must match group size: got ",
      signals.size(1),
      ", expected ",
      rank_to_pe.size());
  TORCH_CHECK(
      signal_row >= 0 && signal_row < signals.size(0),
      "signal_row out of range: ",
      signal_row,
      " for signals rows ",
      signals.size(0));
  TORCH_CHECK(generation > 0, "generation must be positive");
  TORCH_CHECK(
      signals.get_device() == device_idx,
      "signals must be on the NVSHMEM bootstrap device ",
      device_idx,
      ", got ",
      signals.get_device());

  c10::cuda::CUDAGuard guard(signals.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* signal_ptr = reinterpret_cast<uint64_t*>(signals.mutable_data_ptr<int64_t>());
  uint64_t* local_slot =
      signal_ptr + static_cast<size_t>(signal_row) * rank_to_pe.size() + local_rank;
  uint64_t gen = static_cast<uint64_t>(generation);

  if (quiet_before_signal) {
    nvshmemx_quiet_on_stream(stream.stream());
  }
  for (int peer_global : rank_to_pe) {
    nvshmemx_signal_op_on_stream(
        local_slot,
        gen,
        NVSHMEM_SIGNAL_SET,
        peer_global,
        stream.stream());
  }
  C10_CUDA_CHECK(cudaGetLastError());
}

void rowwise_wait_signal_peers_on_stream(
    at::Tensor& signals,
    int64_t signal_row,
    int64_t generation,
    const std::string& group_name) {
  auto& state = olmo_symm_state();
  int group_size = -1;
  int device_idx = -1;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    TORCH_CHECK(state.initialized, "OLMo symmetric memory is not initialized");
    auto it = state.groups.find(group_name);
    TORCH_CHECK(
        it != state.groups.end(),
        "OLMo symmetric-memory group ",
        group_name,
        " is not registered");
    group_size = it->second.world_size;
    device_idx = state.device_idx;
  }

  TORCH_CHECK(signals.is_cuda(), "signals must be a CUDA tensor");
  TORCH_CHECK(signals.scalar_type() == at::kLong, "signals must be int64");
  TORCH_CHECK(signals.is_contiguous(), "signals must be contiguous");
  TORCH_CHECK(signals.dim() == 2, "signals must be rank-2 [rows, group_size]");
  TORCH_CHECK(
      signals.size(1) == group_size,
      "signals second dim must match group size: got ",
      signals.size(1),
      ", expected ",
      group_size);
  TORCH_CHECK(
      signal_row >= 0 && signal_row < signals.size(0),
      "signal_row out of range: ",
      signal_row,
      " for signals rows ",
      signals.size(0));
  TORCH_CHECK(generation > 0, "generation must be positive");
  TORCH_CHECK(
      signals.get_device() == device_idx,
      "signals must be on the NVSHMEM bootstrap device ",
      device_idx,
      ", got ",
      signals.get_device());

  c10::cuda::CUDAGuard guard(signals.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* signal_ptr = reinterpret_cast<uint64_t*>(signals.mutable_data_ptr<int64_t>());
  uint64_t gen = static_cast<uint64_t>(generation);
  constexpr int threads = 256;
  int blocks = static_cast<int>(at::ceil_div(static_cast<int64_t>(group_size), static_cast<int64_t>(threads)));
  waitSignalPeersKernel<<<blocks, threads, 0, stream>>>(
      signal_ptr,
      signal_row,
      static_cast<int64_t>(group_size),
      gen);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  C10_CUDA_CHECK(cudaGetLastError());
}

void all_to_all_vdev_2d_nblocks(
    at::Tensor& input,
    at::Tensor& out,
    at::Tensor& in_splits,
    at::Tensor& out_splits_offsets,
    const std::string& group_name,
    int64_t major_align,
    int64_t nblocks) {
  auto input_hdl = c10d::symmetric_memory::rendezvous(input, group_name);
  auto out_hdl = c10d::symmetric_memory::rendezvous(out, group_name);
  auto in_splits_hdl = c10d::symmetric_memory::rendezvous(in_splits, group_name);
  auto out_splits_offsets_hdl =
      c10d::symmetric_memory::rendezvous(out_splits_offsets, group_name);
  (void)out_hdl;
  (void)in_splits_hdl;
  (void)out_splits_offsets_hdl;

  int world_size = input_hdl->get_world_size();
  TORCH_CHECK(
      major_align > 0, "major_align must be positive, got ", major_align);
  TORCH_CHECK(
      nblocks >= 0, "nblocks must be non-negative (0 means auto), got ", nblocks);

  void* input_ptr = input.data_ptr();
  void* output_ptr = out.mutable_data_ptr();
  int64_t* in_splits_ptr = reinterpret_cast<int64_t*>(in_splits.data_ptr());
  int64_t* out_splits_offsets_ptr =
      reinterpret_cast<int64_t*>(out_splits_offsets.mutable_data_ptr());

  TORCH_CHECK(
      in_splits.is_contiguous() && out_splits_offsets.is_contiguous() &&
          input.is_contiguous() && out.is_contiguous(),
      "input, out, in_splits and out_splits_offsets must be contiguous");
  auto in_split_shape = in_splits.sizes();
  auto out_split_shape = out_splits_offsets.sizes();
  TORCH_CHECK(
      out_split_shape.size() == 2 && out_split_shape[0] == 2 &&
          out_split_shape[1] == in_split_shape[0] &&
          in_split_shape[0] % world_size == 0,
      "out_splits_offsets must be 2D with 2 rows, each row must be a multiple of world_size");

  TORCH_CHECK(
      input.dtype() == out.dtype() && input.stride(0) == out.stride(0),
      "input and out must have the same dtype and same stride at dim 0");
  TORCH_CHECK(
      in_splits.scalar_type() == at::kLong &&
          out_splits_offsets.scalar_type() == at::kLong,
      "splits and offsets must be int64");

  int ne = in_split_shape[0] / world_size;
  constexpr int NUM_TILES = THREADS_PER_BLOCK / A2AV_TILE_SIZE;
  TORCH_CHECK(
      world_size <= A2AV_TILE_SIZE,
      "world_size must be smaller than A2AV_TILE_SIZE",
      A2AV_TILE_SIZE);
  TORCH_CHECK(
      ne <= NUM_TILES,
      "Number of experts must be smaller than NUM_TILES",
      NUM_TILES);

  auto device = input.device();
  TORCH_CHECK(
      device.type() == at::DeviceType::CUDA && out.device() == device &&
          in_splits.device() == device && out_splits_offsets.device() == device,
      "all tensor arguments must be on the same CUDA device");
  c10::cuda::CUDAGuard guard(device);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto& team_manager = c10d::nvshmem_extension::TeamManager::get(device);
  auto team = team_manager.get_team(group_name, input_hdl->get_rank_to_global_rank());
  maybe_init_nvshmem_cumodule(reinterpret_cast<const void*>(allToAllV_2d));

  // Ensure all peers have completed prior stream work (input writes / split setup)
  // before 2D all-to-all starts issuing remote gets.
  int pre_barrier_status = nvshmemx_barrier_on_stream(team, stream.stream());
  TORCH_CHECK(
      pre_barrier_status == 0,
      "nvshmemx_barrier_on_stream (pre) failed with status ",
      pre_barrier_status);

  auto input_dim0 = input.size(0);
  bool rank_is_row_in = true;
  void* args0[] = {
      &in_splits_ptr,
      &out_splits_offsets_ptr,
      &team,
      &ne,
      &input_dim0,
      &rank_is_row_in};
  nvshmemx_collective_launch(
      (const void*)exchangeSplitAndOffset_2d<false>,
      dim3(1),
      dim3(THREADS_PER_BLOCK),
      args0,
      0,
      stream);

  int num_blocks = resolve_num_blocks(world_size, ne, nblocks);
  if (num_blocks == 0) {
    auto input_size_bytes = static_cast<size_t>(input.numel()) * input.element_size();
    num_blocks = resolve_num_blocks_auto(
        input_size_bytes,
        world_size,
        ne,
        input_hdl->world_within_direct_access());
  }
  TORCH_CHECK(num_blocks > 0, "resolved nblocks must be > 0");

  size_t stride_bytes = input.stride(0) * input.element_size();
  bool rank_is_row_out = !rank_is_row_in;

  void* args1[] = {
      &input_ptr,
      &output_ptr,
      &in_splits_ptr,
      &out_splits_offsets_ptr,
      &stride_bytes,
      &world_size,
      &ne,
      &major_align,
      &rank_is_row_out,
      &team};
  nvshmemx_collective_launch(
      (const void*)allToAllV_2d,
      dim3(num_blocks),
      dim3(THREADS_PER_BLOCK),
      args1,
      0,
      stream);

  // Ensure all peers have completed the collective before buffers can be reused.
  int post_barrier_status = nvshmemx_barrier_on_stream(team, stream.stream());
  TORCH_CHECK(
      post_barrier_status == 0,
      "nvshmemx_barrier_on_stream (post) failed with status ",
      post_barrier_status);
}

void all_to_all_vdev_2d_offset_nblocks(
    at::Tensor& input,
    at::Tensor& out,
    at::Tensor& in_splits_offsets,
    at::Tensor& out_splits_offsets,
    const std::string& group_name,
    int64_t nblocks) {
  auto input_hdl = c10d::symmetric_memory::rendezvous(input, group_name);
  auto out_hdl = c10d::symmetric_memory::rendezvous(out, group_name);
  auto out_splits_offsets_hdl =
      c10d::symmetric_memory::rendezvous(out_splits_offsets, group_name);
  auto in_splits_offsets_hdl =
      c10d::symmetric_memory::rendezvous(in_splits_offsets, group_name);
  (void)out_hdl;
  (void)out_splits_offsets_hdl;
  (void)in_splits_offsets_hdl;

  int world_size = input_hdl->get_world_size();
  TORCH_CHECK(
      nblocks >= 0, "nblocks must be non-negative (0 means auto), got ", nblocks);

  int64_t major_align_val = 0;

  void* input_ptr = input.data_ptr();
  void* output_ptr = out.mutable_data_ptr();
  int64_t* out_splits_offsets_ptr =
      reinterpret_cast<int64_t*>(out_splits_offsets.mutable_data_ptr());
  int64_t* in_splits_offsets_ptr =
      reinterpret_cast<int64_t*>(in_splits_offsets.data_ptr());

  TORCH_CHECK(
      out_splits_offsets.is_contiguous() && in_splits_offsets.is_contiguous() &&
          input.is_contiguous() && out.is_contiguous(),
      "input, out, in_splits_offsets and out_splits_offsets must be contiguous");
  auto out_split_shape = out_splits_offsets.sizes();
  auto in_split_shape = in_splits_offsets.sizes();
  TORCH_CHECK(
      in_split_shape.size() == 2 && in_split_shape[0] == 2 &&
          in_split_shape[1] % world_size == 0,
      "in_splits_offsets must be 2D with 2 rows, each row must be a multiple of world_size");

  TORCH_CHECK(
      input.dtype() == out.dtype() && input.stride(0) == out.stride(0),
      "input and out must have the same dtype and same stride at dim 0");
  TORCH_CHECK(
      out_splits_offsets.scalar_type() == at::kLong &&
          in_splits_offsets.scalar_type() == at::kLong,
      "splits and offsets must be int64");

  int ne = in_split_shape[1] / world_size;
  constexpr int NUM_TILES = THREADS_PER_BLOCK / A2AV_TILE_SIZE;
  TORCH_CHECK(
      world_size <= NUM_TILES,
      "world_size must be smaller than NUM_TILES",
      NUM_TILES);
  TORCH_CHECK(
      ne <= A2AV_TILE_SIZE,
      "Number of experts must be smaller than A2AV_TILE_SIZE",
      A2AV_TILE_SIZE);

  auto device = input.device();
  TORCH_CHECK(
      device.type() == at::DeviceType::CUDA && out.device() == device &&
          in_splits_offsets.device() == device &&
          out_splits_offsets.device() == device,
      "all tensor arguments must be on the same CUDA device");
  c10::cuda::CUDAGuard guard(device);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto& team_manager = c10d::nvshmem_extension::TeamManager::get(device);
  auto team = team_manager.get_team(group_name, input_hdl->get_rank_to_global_rank());
  maybe_init_nvshmem_cumodule(reinterpret_cast<const void*>(allToAllV_2d));

  // Ensure all peers have completed prior stream work (input writes / split setup)
  // before 2D all-to-all starts issuing remote gets.
  int pre_barrier_status = nvshmemx_barrier_on_stream(team, stream.stream());
  TORCH_CHECK(
      pre_barrier_status == 0,
      "nvshmemx_barrier_on_stream (pre) failed with status ",
      pre_barrier_status);

  auto input_dim0 = input.size(0);
  bool rank_is_row_in = false;
  void* args0[] = {
      &in_splits_offsets_ptr,
      &out_splits_offsets_ptr,
      &team,
      &ne,
      &input_dim0,
      &rank_is_row_in};
  nvshmemx_collective_launch(
      (const void*)exchangeSplitAndOffset_2d<true>,
      dim3(1),
      dim3(THREADS_PER_BLOCK),
      args0,
      0,
      stream);

  int num_blocks = resolve_num_blocks(world_size, ne, nblocks);
  if (num_blocks == 0) {
    auto input_size_bytes = static_cast<size_t>(input.numel()) * input.element_size();
    num_blocks = resolve_num_blocks_auto(
        input_size_bytes,
        world_size,
        ne,
        input_hdl->world_within_direct_access());
  }
  TORCH_CHECK(num_blocks > 0, "resolved nblocks must be > 0");

  size_t stride_bytes = input.stride(0) * input.element_size();
  bool rank_is_row_out = !rank_is_row_in;

  void* args1[] = {
      &input_ptr,
      &output_ptr,
      &in_splits_offsets_ptr,
      &out_splits_offsets_ptr,
      &stride_bytes,
      &ne,
      &world_size,
      &major_align_val,
      &rank_is_row_out,
      &team};
  nvshmemx_collective_launch(
      (const void*)allToAllV_2d,
      dim3(num_blocks),
      dim3(THREADS_PER_BLOCK),
      args1,
      0,
      stream);

  // Ensure all peers have completed the collective before buffers can be reused.
  int post_barrier_status = nvshmemx_barrier_on_stream(team, stream.stream());
  TORCH_CHECK(
      post_barrier_status == 0,
      "nvshmemx_barrier_on_stream (post) failed with status ",
      post_barrier_status);
}

void rowwise_dispatch_put(
    at::Tensor& input,
    at::Tensor& out,
    at::Tensor& dst_ranks,
    at::Tensor& dst_rows,
    const std::optional<at::Tensor>& probs,
    const std::string& group_name,
    int64_t nblocks,
    bool pre_barrier,
    bool post_barrier) {
  auto* olmo_group = olmo_symm_find_group(group_name);
  TORCH_CHECK(
      olmo_group != nullptr,
      "OLMo rowwise dispatch requires registered OLMo symmetric-memory group ",
      group_name);

  TORCH_CHECK(
      nblocks >= 0, "nblocks must be non-negative (0 means auto), got ", nblocks);
  TORCH_CHECK(input.dim() == 2, "input must be rank-2 [N, D]");
  TORCH_CHECK(out.dim() == 2, "out must be rank-2 [C, D]");
  TORCH_CHECK(
      dst_ranks.dim() == 2 && dst_rows.dim() == 2,
      "dst_ranks and dst_rows must be rank-2 [N, K]");
  TORCH_CHECK(
      dst_ranks.sizes() == dst_rows.sizes(),
      "dst_ranks and dst_rows must have identical shapes");
  TORCH_CHECK(
      dst_ranks.size(0) == input.size(0),
      "dst_ranks/dst_rows first dim (N) must match input rows");
  TORCH_CHECK(
      input.size(1) == out.size(1),
      "input and out must have the same hidden dim (D)");

  TORCH_CHECK(
      input.is_contiguous() && out.is_contiguous() && dst_ranks.is_contiguous() &&
          dst_rows.is_contiguous(),
      "input, out, dst_ranks and dst_rows must be contiguous");
  TORCH_CHECK(
      input.dtype() == out.dtype(),
      "input and out must have the same dtype");
  TORCH_CHECK(
      dst_ranks.scalar_type() == at::kLong && dst_rows.scalar_type() == at::kLong,
      "dst_ranks and dst_rows must be int64");

  auto device = input.device();
  TORCH_CHECK(
      device.type() == at::DeviceType::CUDA && out.device() == device &&
          dst_ranks.device() == device && dst_rows.device() == device,
      "all tensor arguments must be on the same CUDA device");
  c10::cuda::CUDAGuard guard(device);

  auto stream = at::cuda::getCurrentCUDAStream();
  nvshmem_team_t team = NVSHMEM_TEAM_WORLD;
  const int* rank_to_pe_dev = nullptr;
  int group_size = 0;
  bool world_within_direct_access = true;
  rank_to_pe_dev = olmo_group->rank_to_pe_dev;
  group_size = olmo_group->world_size;
  const float* probs_ptr = nullptr;
  if (probs.has_value()) {
    TORCH_CHECK(probs->defined(), "probs optional tensor must be defined");
    TORCH_CHECK(
        probs->device() == device,
        "probs must be on the same CUDA device as other arguments");
    TORCH_CHECK(probs->is_contiguous(), "probs must be contiguous");
    TORCH_CHECK(
        probs->sizes() == dst_ranks.sizes(),
        "probs must have shape [N, K] matching dst_ranks/dst_rows");
    TORCH_CHECK(probs->scalar_type() == at::kFloat, "probs must be float32");
    probs_ptr = probs->data_ptr<float>();
    maybe_init_nvshmem_cumodule(
        reinterpret_cast<const void*>(dispatchRowsPutWeighted<float>));
  } else {
    maybe_init_nvshmem_cumodule(reinterpret_cast<const void*>(dispatchRowsPut));
  }

  const void* input_ptr = input.data_ptr();
  void* out_ptr = out.mutable_data_ptr();
  const int64_t* dst_ranks_ptr = reinterpret_cast<const int64_t*>(dst_ranks.data_ptr());
  const int64_t* dst_rows_ptr = reinterpret_cast<const int64_t*>(dst_rows.data_ptr());

  int64_t num_input_rows = input.size(0);
  int64_t top_k = dst_ranks.size(1);
  int64_t dim = input.size(1);
  int64_t input_row_stride = input.stride(0);
  int64_t out_row_stride = out.stride(0);
  int64_t out_capacity_rows = out.size(0);
  size_t row_bytes = static_cast<size_t>(input.stride(0)) * input.element_size();
  int num_blocks = resolve_num_blocks_rowwise(
      num_input_rows * top_k, nblocks, world_within_direct_access);
  TORCH_CHECK(num_blocks > 0, "resolved nblocks must be > 0");

  if (pre_barrier) {
    int pre_barrier_status = nvshmemx_barrier_on_stream(team, stream.stream());
    TORCH_CHECK(
        pre_barrier_status == 0,
        "nvshmemx_barrier_on_stream (pre) failed with status ",
        pre_barrier_status);
  }

  if (probs_ptr == nullptr) {
    void* args[] = {
        &input_ptr,
        &out_ptr,
        &dst_ranks_ptr,
        &dst_rows_ptr,
        &row_bytes,
        &num_input_rows,
        &top_k,
        &out_capacity_rows,
        &team,
        &rank_to_pe_dev,
        &group_size};
    nvshmemx_collective_launch(
        (const void*)dispatchRowsPut,
        dim3(num_blocks),
        dim3(ROWWISE_THREADS_PER_BLOCK),
        args,
        0,
        stream);
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        input.scalar_type(),
        "dispatchRowsPutWeighted",
        [&] {
          const scalar_t* input_typed = input.data_ptr<scalar_t>();
          scalar_t* out_typed = out.mutable_data_ptr<scalar_t>();
          void* args[] = {
              &input_typed,
              &out_typed,
              &dst_ranks_ptr,
              &dst_rows_ptr,
              &probs_ptr,
              &num_input_rows,
              &top_k,
              &dim,
              &input_row_stride,
              &out_row_stride,
              &out_capacity_rows,
              &team,
              &rank_to_pe_dev,
              &group_size};
          nvshmemx_collective_launch(
              (const void*)dispatchRowsPutWeighted<scalar_t>,
              dim3(num_blocks),
              dim3(ROWWISE_THREADS_PER_BLOCK),
              args,
              0,
              stream);
        });
  }
  if (post_barrier) {
    int post_barrier_status = nvshmemx_barrier_on_stream(team, stream.stream());
    TORCH_CHECK(
        post_barrier_status == 0,
        "nvshmemx_barrier_on_stream (post) failed with status ",
        post_barrier_status);
  }
}

void rowwise_build_compact_route_records(
    at::Tensor& dst_ranks,
    at::Tensor& dst_rows,
    at::Tensor& route_experts,
    at::Tensor& route_records,
    at::Tensor& wave_counts,
    at::Tensor& wave_fill_counts,
    at::Tensor& wave_offsets,
    int64_t num_local_experts,
    int64_t num_waves,
    int64_t nblocks) {
  TORCH_CHECK(
      nblocks >= 0, "nblocks must be non-negative (0 means auto), got ", nblocks);
  TORCH_CHECK(num_local_experts > 0, "num_local_experts must be > 0");
  TORCH_CHECK(num_waves > 0, "num_waves must be > 0");
  TORCH_CHECK(
      dst_ranks.dim() == 2 && dst_rows.dim() == 2 && route_experts.dim() == 2,
      "dst_ranks, dst_rows, and route_experts must be rank-2 [N, K]");
  TORCH_CHECK(
      dst_ranks.sizes() == dst_rows.sizes() &&
          dst_ranks.sizes() == route_experts.sizes(),
      "dst_ranks, dst_rows, and route_experts must have identical shapes");
  TORCH_CHECK(
      route_records.dim() == 2 && route_records.size(1) == 4,
      "route_records must be rank-2 [N*K, 4]");
  TORCH_CHECK(
      route_records.size(0) == dst_ranks.numel(),
      "route_records first dim must equal dst_ranks.numel()");
  TORCH_CHECK(
      wave_counts.dim() == 1 && wave_counts.size(0) == num_waves,
      "wave_counts must be rank-1 [num_waves]");
  TORCH_CHECK(
      wave_fill_counts.dim() == 1 && wave_fill_counts.size(0) == num_waves,
      "wave_fill_counts must be rank-1 [num_waves]");
  TORCH_CHECK(
      wave_offsets.dim() == 1 && wave_offsets.size(0) == num_waves + 1,
      "wave_offsets must be rank-1 [num_waves + 1]");

  TORCH_CHECK(
      dst_ranks.is_contiguous() && dst_rows.is_contiguous() &&
          route_experts.is_contiguous() && route_records.is_contiguous() &&
          wave_counts.is_contiguous() && wave_fill_counts.is_contiguous() &&
          wave_offsets.is_contiguous(),
      "all route tensors must be contiguous");
  TORCH_CHECK(
      dst_ranks.scalar_type() == at::kLong && dst_rows.scalar_type() == at::kLong,
      "dst_ranks and dst_rows must be int64");
  TORCH_CHECK(
      route_experts.scalar_type() == at::kInt ||
          route_experts.scalar_type() == at::kLong,
      "route_experts must be int32 or int64");
  TORCH_CHECK(
      route_records.scalar_type() == at::kLong &&
          wave_counts.scalar_type() == at::kLong &&
          wave_fill_counts.scalar_type() == at::kLong &&
          wave_offsets.scalar_type() == at::kLong,
      "route_records, wave_counts, wave_fill_counts, and wave_offsets must be int64");

  auto device = dst_ranks.device();
  TORCH_CHECK(
      device.type() == at::DeviceType::CUDA && dst_rows.device() == device &&
          route_experts.device() == device && route_records.device() == device &&
          wave_counts.device() == device && wave_fill_counts.device() == device &&
          wave_offsets.device() == device,
      "all tensor arguments must be on the same CUDA device");
  c10::cuda::CUDAGuard guard(device);

  auto stream = at::cuda::getCurrentCUDAStream();
  int64_t num_routes = dst_ranks.numel();
  int64_t top_k = dst_ranks.size(1);
  int64_t experts_per_wave = at::ceil_div(num_local_experts, num_waves);
  int num_blocks = resolve_num_blocks_rowwise(num_routes, nblocks, true);
  TORCH_CHECK(num_blocks > 0, "resolved nblocks must be > 0");

  C10_CUDA_CHECK(cudaMemsetAsync(
      wave_counts.mutable_data_ptr<int64_t>(),
      0,
      static_cast<size_t>(num_waves) * sizeof(int64_t),
      stream.stream()));

  const int64_t* dst_ranks_ptr =
      reinterpret_cast<const int64_t*>(dst_ranks.data_ptr());
  const int64_t* dst_rows_ptr =
      reinterpret_cast<const int64_t*>(dst_rows.data_ptr());
  int64_t* route_records_ptr =
      reinterpret_cast<int64_t*>(route_records.mutable_data_ptr());
  int64_t* wave_counts_ptr =
      reinterpret_cast<int64_t*>(wave_counts.mutable_data_ptr());
  int64_t* wave_fill_counts_ptr =
      reinterpret_cast<int64_t*>(wave_fill_counts.mutable_data_ptr());
  int64_t* wave_offsets_ptr =
      reinterpret_cast<int64_t*>(wave_offsets.mutable_data_ptr());

  if (route_experts.scalar_type() == at::kInt) {
    const int32_t* route_experts_ptr = route_experts.data_ptr<int32_t>();
    countCompactRowwiseRoutes<int32_t><<<
        dim3(num_blocks),
        dim3(ROWWISE_THREADS_PER_BLOCK),
        0,
        stream>>>(
        dst_ranks_ptr,
        dst_rows_ptr,
        route_experts_ptr,
        wave_counts_ptr,
        num_routes,
        top_k,
        num_local_experts,
        num_waves,
        experts_per_wave);
  } else {
    const int64_t* route_experts_ptr =
        reinterpret_cast<const int64_t*>(route_experts.data_ptr());
    countCompactRowwiseRoutes<int64_t><<<
        dim3(num_blocks),
        dim3(ROWWISE_THREADS_PER_BLOCK),
        0,
        stream>>>(
        dst_ranks_ptr,
        dst_rows_ptr,
        route_experts_ptr,
        wave_counts_ptr,
        num_routes,
        top_k,
        num_local_experts,
        num_waves,
        experts_per_wave);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  prefixCompactRowwiseRouteCounts<<<dim3(1), dim3(1), 0, stream>>>(
      wave_counts_ptr,
      wave_fill_counts_ptr,
      wave_offsets_ptr,
      num_waves);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (route_experts.scalar_type() == at::kInt) {
    const int32_t* route_experts_ptr = route_experts.data_ptr<int32_t>();
    fillCompactRowwiseRoutes<int32_t><<<
        dim3(num_blocks),
        dim3(ROWWISE_THREADS_PER_BLOCK),
        0,
        stream>>>(
        dst_ranks_ptr,
        dst_rows_ptr,
        route_experts_ptr,
        route_records_ptr,
        wave_fill_counts_ptr,
        wave_offsets_ptr,
        num_routes,
        top_k,
        num_local_experts,
        num_waves,
        experts_per_wave);
  } else {
    const int64_t* route_experts_ptr =
        reinterpret_cast<const int64_t*>(route_experts.data_ptr());
    fillCompactRowwiseRoutes<int64_t><<<
        dim3(num_blocks),
        dim3(ROWWISE_THREADS_PER_BLOCK),
        0,
        stream>>>(
        dst_ranks_ptr,
        dst_rows_ptr,
        route_experts_ptr,
        route_records_ptr,
        wave_fill_counts_ptr,
        wave_offsets_ptr,
        num_routes,
        top_k,
        num_local_experts,
        num_waves,
        experts_per_wave);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rowwise_dispatch_put_compact(
    at::Tensor& input,
    at::Tensor& out,
    at::Tensor& route_records,
    at::Tensor& wave_offsets,
    int64_t wave_idx,
    const std::string& group_name,
    int64_t nblocks,
    bool pre_barrier,
    bool post_barrier) {
  auto* olmo_group = olmo_symm_find_group(group_name);
  TORCH_CHECK(
      olmo_group != nullptr,
      "OLMo compact rowwise dispatch requires registered OLMo symmetric-memory group ",
      group_name);
  TORCH_CHECK(
      nblocks >= 0, "nblocks must be non-negative (0 means auto), got ", nblocks);
  TORCH_CHECK(input.dim() == 2, "input must be rank-2 [N, D]");
  TORCH_CHECK(out.dim() == 2, "out must be rank-2 [C, D]");
  TORCH_CHECK(
      route_records.dim() == 2 && route_records.size(1) == 4,
      "route_records must be rank-2 [R, 4]");
  TORCH_CHECK(wave_offsets.dim() == 1, "wave_offsets must be rank-1");
  TORCH_CHECK(wave_idx >= 0, "wave_idx must be non-negative");
  TORCH_CHECK(
      wave_idx + 1 < wave_offsets.size(0),
      "wave_idx is outside wave_offsets range");
  TORCH_CHECK(
      input.size(1) == out.size(1),
      "input and out must have the same hidden dim (D)");
  TORCH_CHECK(
      input.is_contiguous() && out.is_contiguous() &&
          route_records.is_contiguous() && wave_offsets.is_contiguous(),
      "input, out, route_records, and wave_offsets must be contiguous");
  TORCH_CHECK(input.dtype() == out.dtype(), "input and out dtype mismatch");
  TORCH_CHECK(
      route_records.scalar_type() == at::kLong &&
          wave_offsets.scalar_type() == at::kLong,
      "route_records and wave_offsets must be int64");

  auto device = input.device();
  TORCH_CHECK(
      device.type() == at::DeviceType::CUDA && out.device() == device &&
          route_records.device() == device && wave_offsets.device() == device,
      "all tensor arguments must be on the same CUDA device");
  c10::cuda::CUDAGuard guard(device);

  auto stream = at::cuda::getCurrentCUDAStream();
  nvshmem_team_t team = NVSHMEM_TEAM_WORLD;
  const int* rank_to_pe_dev = olmo_group->rank_to_pe_dev;
  int group_size = olmo_group->world_size;
  maybe_init_nvshmem_cumodule(reinterpret_cast<const void*>(dispatchRowsPutCompact));

  const void* input_ptr = input.data_ptr();
  void* out_ptr = out.mutable_data_ptr();
  const int64_t* route_records_ptr =
      reinterpret_cast<const int64_t*>(route_records.data_ptr());
  const int64_t* wave_offsets_ptr =
      reinterpret_cast<const int64_t*>(wave_offsets.data_ptr());
  int64_t num_input_rows = input.size(0);
  int64_t out_capacity_rows = out.size(0);
  size_t row_bytes = static_cast<size_t>(input.stride(0)) * input.element_size();
  int num_blocks = resolve_num_blocks_rowwise(
      route_records.size(0), nblocks, true);
  TORCH_CHECK(num_blocks > 0, "resolved nblocks must be > 0");

  if (pre_barrier) {
    int pre_barrier_status = nvshmemx_barrier_on_stream(team, stream.stream());
    TORCH_CHECK(
        pre_barrier_status == 0,
        "nvshmemx_barrier_on_stream (pre) failed with status ",
        pre_barrier_status);
  }

  void* args[] = {
      &input_ptr,
      &out_ptr,
      &route_records_ptr,
      &wave_offsets_ptr,
      &wave_idx,
      &row_bytes,
      &num_input_rows,
      &out_capacity_rows,
      &team,
      &rank_to_pe_dev,
      &group_size};
  nvshmemx_collective_launch(
      (const void*)dispatchRowsPutCompact,
      dim3(num_blocks),
      dim3(ROWWISE_THREADS_PER_BLOCK),
      args,
      0,
      stream);

  if (post_barrier) {
    int post_barrier_status = nvshmemx_barrier_on_stream(team, stream.stream());
    TORCH_CHECK(
        post_barrier_status == 0,
        "nvshmemx_barrier_on_stream (post) failed with status ",
        post_barrier_status);
  }
}

void rowwise_dispatch_put_compact_weighted(
    at::Tensor& input,
    at::Tensor& out,
    at::Tensor& route_records,
    at::Tensor& wave_offsets,
    int64_t wave_idx,
    at::Tensor& probs,
    const std::string& group_name,
    int64_t nblocks,
    bool pre_barrier,
    bool post_barrier) {
  auto* olmo_group = olmo_symm_find_group(group_name);
  TORCH_CHECK(
      olmo_group != nullptr,
      "OLMo compact weighted rowwise dispatch requires registered OLMo symmetric-memory group ",
      group_name);
  TORCH_CHECK(
      nblocks >= 0, "nblocks must be non-negative (0 means auto), got ", nblocks);
  TORCH_CHECK(input.dim() == 2, "input must be rank-2 [N, D]");
  TORCH_CHECK(out.dim() == 2, "out must be rank-2 [C, D]");
  TORCH_CHECK(
      route_records.dim() == 2 && route_records.size(1) == 4,
      "route_records must be rank-2 [R, 4]");
  TORCH_CHECK(wave_offsets.dim() == 1, "wave_offsets must be rank-1");
  TORCH_CHECK(wave_idx >= 0, "wave_idx must be non-negative");
  TORCH_CHECK(
      wave_idx + 1 < wave_offsets.size(0),
      "wave_idx is outside wave_offsets range");
  TORCH_CHECK(probs.dim() == 2, "probs must be rank-2 [N, K]");
  TORCH_CHECK(
      probs.size(0) == input.size(0),
      "probs first dim must match input rows");
  TORCH_CHECK(
      input.size(1) == out.size(1),
      "input and out must have the same hidden dim (D)");
  TORCH_CHECK(
      input.is_contiguous() && out.is_contiguous() &&
          route_records.is_contiguous() && wave_offsets.is_contiguous() &&
          probs.is_contiguous(),
      "input, out, route_records, wave_offsets, and probs must be contiguous");
  TORCH_CHECK(input.dtype() == out.dtype(), "input and out dtype mismatch");
  TORCH_CHECK(probs.scalar_type() == at::kFloat, "probs must be float32");
  TORCH_CHECK(
      route_records.scalar_type() == at::kLong &&
          wave_offsets.scalar_type() == at::kLong,
      "route_records and wave_offsets must be int64");

  auto device = input.device();
  TORCH_CHECK(
      device.type() == at::DeviceType::CUDA && out.device() == device &&
          route_records.device() == device && wave_offsets.device() == device &&
          probs.device() == device,
      "all tensor arguments must be on the same CUDA device");
  c10::cuda::CUDAGuard guard(device);

  auto stream = at::cuda::getCurrentCUDAStream();
  nvshmem_team_t team = NVSHMEM_TEAM_WORLD;
  const int* rank_to_pe_dev = olmo_group->rank_to_pe_dev;
  int group_size = olmo_group->world_size;
  maybe_init_nvshmem_cumodule(
      reinterpret_cast<const void*>(dispatchRowsPutCompactWeighted<float>));

  const int64_t* route_records_ptr =
      reinterpret_cast<const int64_t*>(route_records.data_ptr());
  const int64_t* wave_offsets_ptr =
      reinterpret_cast<const int64_t*>(wave_offsets.data_ptr());
  const float* probs_ptr = probs.data_ptr<float>();
  int64_t num_input_rows = input.size(0);
  int64_t top_k = probs.size(1);
  int64_t dim = input.size(1);
  int64_t input_row_stride = input.stride(0);
  int64_t out_row_stride = out.stride(0);
  int64_t out_capacity_rows = out.size(0);
  int num_blocks = resolve_num_blocks_rowwise(
      route_records.size(0), nblocks, true);
  TORCH_CHECK(num_blocks > 0, "resolved nblocks must be > 0");

  if (pre_barrier) {
    int pre_barrier_status = nvshmemx_barrier_on_stream(team, stream.stream());
    TORCH_CHECK(
        pre_barrier_status == 0,
        "nvshmemx_barrier_on_stream (pre) failed with status ",
        pre_barrier_status);
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      input.scalar_type(),
      "dispatchRowsPutCompactWeighted",
      [&] {
        const scalar_t* input_typed = input.data_ptr<scalar_t>();
        scalar_t* out_typed = out.mutable_data_ptr<scalar_t>();
        void* args[] = {
            &input_typed,
            &out_typed,
            &route_records_ptr,
            &wave_offsets_ptr,
            &wave_idx,
            &probs_ptr,
            &top_k,
            &dim,
            &input_row_stride,
            &out_row_stride,
            &num_input_rows,
            &out_capacity_rows,
            &team,
            &rank_to_pe_dev,
            &group_size};
        nvshmemx_collective_launch(
            (const void*)dispatchRowsPutCompactWeighted<scalar_t>,
            dim3(num_blocks),
            dim3(ROWWISE_THREADS_PER_BLOCK),
            args,
            0,
            stream);
      });

  if (post_barrier) {
    int post_barrier_status = nvshmemx_barrier_on_stream(team, stream.stream());
    TORCH_CHECK(
        post_barrier_status == 0,
        "nvshmemx_barrier_on_stream (post) failed with status ",
        post_barrier_status);
  }
}

void rowwise_inverse_route_meta_put_compact(
    at::Tensor& inverse_route_meta,
    at::Tensor& route_records,
    at::Tensor& wave_offsets,
    int64_t src_rank,
    const std::string& group_name,
    int64_t nblocks,
    bool pre_barrier,
    bool post_barrier) {
  auto* olmo_group = olmo_symm_find_group(group_name);
  TORCH_CHECK(
      olmo_group != nullptr,
      "OLMo compact inverse route metadata PUT requires registered OLMo symmetric-memory group ",
      group_name);
  TORCH_CHECK(
      nblocks >= 0, "nblocks must be non-negative (0 means auto), got ", nblocks);
  TORCH_CHECK(
      inverse_route_meta.dim() == 2 && inverse_route_meta.size(1) == 2,
      "inverse_route_meta must be rank-2 [C, 2]");
  TORCH_CHECK(
      route_records.dim() == 2 && route_records.size(1) == 4,
      "route_records must be rank-2 [R, 4]");
  TORCH_CHECK(wave_offsets.dim() == 1, "wave_offsets must be rank-1");
  TORCH_CHECK(wave_offsets.size(0) >= 2, "wave_offsets must contain at least one wave");
  TORCH_CHECK(src_rank >= 0, "src_rank must be non-negative");
  TORCH_CHECK(
      inverse_route_meta.is_contiguous() && route_records.is_contiguous() &&
          wave_offsets.is_contiguous(),
      "inverse_route_meta, route_records, and wave_offsets must be contiguous");
  TORCH_CHECK(
      inverse_route_meta.scalar_type() == at::kLong &&
          route_records.scalar_type() == at::kLong &&
          wave_offsets.scalar_type() == at::kLong,
      "inverse_route_meta, route_records, and wave_offsets must be int64");

  auto device = inverse_route_meta.device();
  TORCH_CHECK(
      device.type() == at::DeviceType::CUDA &&
          route_records.device() == device && wave_offsets.device() == device,
      "all tensor arguments must be on the same CUDA device");
  c10::cuda::CUDAGuard guard(device);

  auto stream = at::cuda::getCurrentCUDAStream();
  nvshmem_team_t team = NVSHMEM_TEAM_WORLD;
  const int* rank_to_pe_dev = olmo_group->rank_to_pe_dev;
  int group_size = olmo_group->world_size;
  maybe_init_nvshmem_cumodule(
      reinterpret_cast<const void*>(inverseRouteMetaPutCompact));

  int64_t* inverse_route_meta_ptr =
      reinterpret_cast<int64_t*>(inverse_route_meta.mutable_data_ptr());
  const int64_t* route_records_ptr =
      reinterpret_cast<const int64_t*>(route_records.data_ptr());
  const int64_t* wave_offsets_ptr =
      reinterpret_cast<const int64_t*>(wave_offsets.data_ptr());
  int64_t num_waves = wave_offsets.size(0) - 1;
  int64_t inverse_capacity_rows = inverse_route_meta.size(0);
  int num_blocks = resolve_num_blocks_rowwise(route_records.size(0), nblocks, true);
  TORCH_CHECK(num_blocks > 0, "resolved nblocks must be > 0");

  if (pre_barrier) {
    int pre_barrier_status = nvshmemx_barrier_on_stream(team, stream.stream());
    TORCH_CHECK(
        pre_barrier_status == 0,
        "nvshmemx_barrier_on_stream (pre) failed with status ",
        pre_barrier_status);
  }

  void* args[] = {
      &inverse_route_meta_ptr,
      &route_records_ptr,
      &wave_offsets_ptr,
      &num_waves,
      &src_rank,
      &inverse_capacity_rows,
      &team,
      &rank_to_pe_dev,
      &group_size};
  nvshmemx_collective_launch(
      (const void*)inverseRouteMetaPutCompact,
      dim3(num_blocks),
      dim3(ROWWISE_THREADS_PER_BLOCK),
      args,
      0,
      stream);

  if (post_barrier) {
    int post_barrier_status = nvshmemx_barrier_on_stream(team, stream.stream());
    TORCH_CHECK(
        post_barrier_status == 0,
        "nvshmemx_barrier_on_stream (post) failed with status ",
        post_barrier_status);
  }
}

void rowwise_combine_get(
    at::Tensor& expert_out,
    at::Tensor& out,
    at::Tensor& src_ranks,
    at::Tensor& src_rows,
    const std::optional<at::Tensor>& probs,
    const std::string& group_name,
    int64_t nblocks,
    const std::optional<at::Tensor>& gathered_out,
    bool pre_barrier,
    bool post_barrier) {
  auto* olmo_group = olmo_symm_find_group(group_name);
  TORCH_CHECK(
      olmo_group != nullptr,
      "OLMo rowwise combine requires registered OLMo symmetric-memory group ",
      group_name);

  TORCH_CHECK(
      nblocks >= 0, "nblocks must be non-negative (0 means auto), got ", nblocks);
  TORCH_CHECK(expert_out.dim() == 2, "expert_out must be rank-2 [C, D]");
  TORCH_CHECK(out.dim() == 2, "out must be rank-2 [N, D]");
  TORCH_CHECK(
      src_ranks.dim() == 2 && src_rows.dim() == 2,
      "src_ranks and src_rows must be rank-2 [N, K]");
  TORCH_CHECK(
      src_ranks.sizes() == src_rows.sizes(),
      "src_ranks and src_rows must have identical shapes");
  TORCH_CHECK(
      src_ranks.size(0) == out.size(0),
      "src_ranks/src_rows first dim (N) must match out rows");
  TORCH_CHECK(
      expert_out.size(1) == out.size(1),
      "expert_out and out must have the same hidden dim (D)");

  TORCH_CHECK(
      expert_out.is_contiguous() && out.is_contiguous() && src_ranks.is_contiguous() &&
          src_rows.is_contiguous(),
      "expert_out, out, src_ranks and src_rows must be contiguous");
  TORCH_CHECK(
      expert_out.dtype() == out.dtype(),
      "expert_out and out must have the same dtype");
  TORCH_CHECK(
      src_ranks.scalar_type() == at::kLong && src_rows.scalar_type() == at::kLong,
      "src_ranks and src_rows must be int64");

  auto device = expert_out.device();
  TORCH_CHECK(
      device.type() == at::DeviceType::CUDA && out.device() == device &&
          src_ranks.device() == device && src_rows.device() == device,
      "all tensor arguments must be on the same CUDA device");

  const float* probs_ptr = nullptr;
  if (probs.has_value()) {
    TORCH_CHECK(probs->defined(), "probs optional tensor must be defined");
    TORCH_CHECK(
        probs->device() == device,
        "probs must be on the same CUDA device as other arguments");
    TORCH_CHECK(
        probs->is_contiguous(),
        "probs must be contiguous");
    TORCH_CHECK(
        probs->sizes() == src_ranks.sizes(),
        "probs must have shape [N, K] matching src_ranks/src_rows");
    TORCH_CHECK(
        probs->scalar_type() == at::kFloat,
        "probs must be float32");
    probs_ptr = probs->data_ptr<float>();
  }

  if (gathered_out.has_value()) {
    TORCH_CHECK(
        gathered_out->defined(),
        "gathered_out optional tensor must be defined");
    TORCH_CHECK(
        gathered_out->device() == device,
        "gathered_out must be on the same CUDA device as other arguments");
    TORCH_CHECK(
        gathered_out->is_contiguous(),
        "gathered_out must be contiguous");
    TORCH_CHECK(
        gathered_out->scalar_type() == out.scalar_type(),
        "gathered_out must have the same dtype as out");
  }

  c10::cuda::CUDAGuard guard(device);
  auto stream = at::cuda::getCurrentCUDAStream();
  nvshmem_team_t team = NVSHMEM_TEAM_WORLD;
  const int* rank_to_pe_dev = nullptr;
  int group_size = 0;
  bool world_within_direct_access = true;
  rank_to_pe_dev = olmo_group->rank_to_pe_dev;
  group_size = olmo_group->world_size;
  maybe_init_nvshmem_cumodule(reinterpret_cast<const void*>(gatherRowsGet<true>));

  int64_t num_out_rows = out.size(0);
  int64_t top_k = src_ranks.size(1);
  int64_t dim = out.size(1);
  int64_t expert_capacity_rows = expert_out.size(0);
  size_t row_bytes =
      static_cast<size_t>(expert_out.stride(0)) * expert_out.element_size();
  int num_blocks = resolve_num_blocks_rowwise(
      num_out_rows * top_k, nblocks, world_within_direct_access);
  TORCH_CHECK(num_blocks > 0, "resolved nblocks must be > 0");

  if (pre_barrier) {
    int pre_barrier_status = nvshmemx_barrier_on_stream(team, stream.stream());
    TORCH_CHECK(
        pre_barrier_status == 0,
        "nvshmemx_barrier_on_stream (pre) failed with status ",
        pre_barrier_status);
  }

  at::Tensor gathered;
  if (gathered_out.has_value()) {
    TORCH_CHECK(
        gathered_out->dim() == 3,
        "gathered_out must be rank-3 [N, K, D]");
    TORCH_CHECK(
        gathered_out->size(0) == num_out_rows &&
            gathered_out->size(1) == top_k &&
            gathered_out->size(2) == dim,
        "gathered_out shape mismatch: expected [",
        num_out_rows,
        ", ",
        top_k,
        ", ",
        dim,
        "]");
    gathered = *gathered_out;
  } else {
    // Local temporary gather buffer [N, K, D] before reduction to [N, D].
    gathered = at::empty({num_out_rows, top_k, dim}, out.options());
  }

  const void* expert_out_ptr = expert_out.data_ptr();
  void* gathered_ptr = gathered.mutable_data_ptr();
  const int64_t* src_ranks_ptr = reinterpret_cast<const int64_t*>(src_ranks.data_ptr());
  const int64_t* src_rows_ptr = reinterpret_cast<const int64_t*>(src_rows.data_ptr());

  void* args[] = {
      &expert_out_ptr,
      &gathered_ptr,
      &src_ranks_ptr,
      &src_rows_ptr,
      &row_bytes,
      &num_out_rows,
      &top_k,
      &expert_capacity_rows,
      &team,
      &rank_to_pe_dev,
      &group_size};
  nvshmemx_collective_launch(
      (const void*)gatherRowsGet<true>,
      dim3(num_blocks),
      dim3(ROWWISE_THREADS_PER_BLOCK),
      args,
      0,
      stream);

  constexpr int THREADS = 256;
  dim3 block(THREADS);
  dim3 grid(
      static_cast<unsigned int>(num_out_rows),
      static_cast<unsigned int>(at::ceil_div(dim, static_cast<int64_t>(THREADS))));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      out.scalar_type(),
      "combineRowsReduceKernel",
      [&] {
        const scalar_t* gathered_typed = gathered.data_ptr<scalar_t>();
        scalar_t* out_typed = out.mutable_data_ptr<scalar_t>();
        if (probs_ptr == nullptr) {
          combineRowsReduceKernel<scalar_t, false>
              <<<grid, block, 0, stream>>>(
                  gathered_typed,
                  out_typed,
                  nullptr,
                  num_out_rows,
                  top_k,
                  dim);
        } else {
          combineRowsReduceKernel<scalar_t, true>
              <<<grid, block, 0, stream>>>(
                  gathered_typed,
                  out_typed,
                  probs_ptr,
                  num_out_rows,
                  top_k,
                  dim);
        }
      });

  if (post_barrier) {
    // TBO reuses symmetric slots across adjacent blocks, so every rank has to
    // wait until all peers have finished their GETs before a fast rank can
    // overwrite its local expert_out/combine_in slot.
    int post_barrier_status = nvshmemx_barrier_on_stream(team, stream.stream());
    TORCH_CHECK(
        post_barrier_status == 0,
        "nvshmemx_barrier_on_stream (post) failed with status ",
        post_barrier_status);
  }
}

void rowwise_combine_put(
    at::Tensor& expert_out,
    at::Tensor& gathered_out,
    at::Tensor& inverse_route_meta,
    at::Tensor& row_start,
    at::Tensor& num_rows,
    const std::string& group_name,
    int64_t nblocks,
    bool pre_barrier,
    bool post_barrier) {
  auto* olmo_group = olmo_symm_find_group(group_name);
  TORCH_CHECK(
      olmo_group != nullptr,
      "OLMo rowwise combine PUT requires registered OLMo symmetric-memory group ",
      group_name);

  TORCH_CHECK(
      nblocks >= 0, "nblocks must be non-negative (0 means auto), got ", nblocks);
  TORCH_CHECK(expert_out.dim() == 2, "expert_out must be rank-2 [C, D]");
  TORCH_CHECK(gathered_out.dim() == 2, "gathered_out must be rank-2 [R, D]");
  TORCH_CHECK(
      inverse_route_meta.dim() == 2 && inverse_route_meta.size(1) == 2,
      "inverse_route_meta must be rank-2 [C, 2]");
  TORCH_CHECK(
      inverse_route_meta.size(0) == expert_out.size(0),
      "inverse_route_meta first dim must match expert_out rows");
  TORCH_CHECK(
      expert_out.size(1) == gathered_out.size(1),
      "expert_out and gathered_out must have the same hidden dim (D)");
  TORCH_CHECK(row_start.numel() == 1, "row_start must be a scalar tensor");
  TORCH_CHECK(num_rows.numel() == 1, "num_rows must be a scalar tensor");

  TORCH_CHECK(
      expert_out.is_contiguous() && gathered_out.is_contiguous() &&
          inverse_route_meta.is_contiguous(),
      "expert_out, gathered_out, and inverse_route_meta must be contiguous");
  TORCH_CHECK(
      expert_out.dtype() == gathered_out.dtype(),
      "expert_out and gathered_out must have the same dtype");
  TORCH_CHECK(
      inverse_route_meta.scalar_type() == at::kLong,
      "inverse_route_meta must be int64");
  TORCH_CHECK(
      row_start.scalar_type() == at::kLong && num_rows.scalar_type() == at::kLong,
      "row_start and num_rows must be int64");

  auto device = expert_out.device();
  TORCH_CHECK(
      device.type() == at::DeviceType::CUDA && gathered_out.device() == device &&
          inverse_route_meta.device() == device && row_start.device() == device &&
          num_rows.device() == device,
      "all tensor arguments must be on the same CUDA device");
  c10::cuda::CUDAGuard guard(device);

  auto stream = at::cuda::getCurrentCUDAStream();
  nvshmem_team_t team = NVSHMEM_TEAM_WORLD;
  const int* rank_to_pe_dev = nullptr;
  int group_size = 0;
  bool world_within_direct_access = true;
  rank_to_pe_dev = olmo_group->rank_to_pe_dev;
  group_size = olmo_group->world_size;
  maybe_init_nvshmem_cumodule(reinterpret_cast<const void*>(combineRowsPutRange));

  int64_t expert_capacity_rows = expert_out.size(0);
  int64_t gathered_capacity_rows = gathered_out.size(0);
  size_t row_bytes =
      static_cast<size_t>(expert_out.stride(0)) * expert_out.element_size();
  int num_blocks = resolve_num_blocks_rowwise(
      expert_capacity_rows, nblocks, world_within_direct_access);
  TORCH_CHECK(num_blocks > 0, "resolved nblocks must be > 0");

  if (pre_barrier) {
    int pre_barrier_status = nvshmemx_barrier_on_stream(team, stream.stream());
    TORCH_CHECK(
        pre_barrier_status == 0,
        "nvshmemx_barrier_on_stream (pre) failed with status ",
        pre_barrier_status);
  }

  const void* expert_out_ptr = expert_out.data_ptr();
  void* gathered_out_ptr = gathered_out.mutable_data_ptr();
  const int64_t* inverse_route_meta_ptr =
      reinterpret_cast<const int64_t*>(inverse_route_meta.data_ptr());
  const int64_t* row_start_ptr =
      reinterpret_cast<const int64_t*>(row_start.data_ptr());
  const int64_t* num_rows_ptr =
      reinterpret_cast<const int64_t*>(num_rows.data_ptr());

  void* args[] = {
      &expert_out_ptr,
      &gathered_out_ptr,
      &inverse_route_meta_ptr,
      &row_start_ptr,
      &num_rows_ptr,
      &row_bytes,
      &expert_capacity_rows,
      &gathered_capacity_rows,
      &team,
      &rank_to_pe_dev,
      &group_size};
  nvshmemx_collective_launch(
      (const void*)combineRowsPutRange,
      dim3(num_blocks),
      dim3(ROWWISE_THREADS_PER_BLOCK),
      args,
      0,
      stream);

  if (post_barrier) {
    int post_barrier_status = nvshmemx_barrier_on_stream(team, stream.stream());
    TORCH_CHECK(
        post_barrier_status == 0,
        "nvshmemx_barrier_on_stream (post) failed with status ",
        post_barrier_status);
  }
}

void rowwise_reduce_gathered_routes(
    at::Tensor& gathered,
    at::Tensor& probs,
    at::Tensor& out,
    const std::optional<at::Tensor>& route_ranks) {
  TORCH_CHECK(gathered.dim() == 3, "gathered must be rank-3 [N, K, D]");
  TORCH_CHECK(probs.dim() == 2, "probs must be rank-2 [N, K]");
  TORCH_CHECK(out.dim() == 2, "out must be rank-2 [N, D]");
  TORCH_CHECK(
      probs.size(0) == gathered.size(0) && probs.size(1) == gathered.size(1),
      "probs shape must match gathered [N, K]");
  TORCH_CHECK(
      out.size(0) == gathered.size(0) && out.size(1) == gathered.size(2),
      "out shape must match gathered [N, D]");
  TORCH_CHECK(
      gathered.is_contiguous() && probs.is_contiguous() && out.is_contiguous(),
      "gathered, probs, and out must be contiguous");
  TORCH_CHECK(
      gathered.scalar_type() == out.scalar_type(),
      "gathered and out must have the same dtype");
  TORCH_CHECK(
      probs.scalar_type() == at::kFloat || probs.scalar_type() == at::kHalf ||
          probs.scalar_type() == at::kBFloat16,
      "probs must be float32, float16, or bfloat16");

  auto device = gathered.device();
  TORCH_CHECK(
      device.type() == at::DeviceType::CUDA && probs.device() == device &&
          out.device() == device,
      "all tensor arguments must be on the same CUDA device");
  const int64_t* route_ranks_ptr = nullptr;
  if (route_ranks.has_value()) {
    TORCH_CHECK(route_ranks->defined(), "route_ranks optional tensor must be defined");
    TORCH_CHECK(route_ranks->dim() == 2, "route_ranks must be rank-2 [N, K]");
    TORCH_CHECK(
        route_ranks->size(0) == gathered.size(0) &&
            route_ranks->size(1) == gathered.size(1),
        "route_ranks shape must match gathered [N, K]");
    TORCH_CHECK(
        route_ranks->device() == device,
        "route_ranks must be on the same CUDA device as other arguments");
    TORCH_CHECK(route_ranks->is_contiguous(), "route_ranks must be contiguous");
    TORCH_CHECK(route_ranks->scalar_type() == at::kLong, "route_ranks must be int64");
    route_ranks_ptr = reinterpret_cast<const int64_t*>(route_ranks->data_ptr());
  }
  c10::cuda::CUDAGuard guard(device);

  int64_t num_out_rows = gathered.size(0);
  int64_t top_k = gathered.size(1);
  int64_t dim = gathered.size(2);
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int THREADS = 256;
  dim3 block(THREADS);
  dim3 grid(
      static_cast<unsigned int>(num_out_rows),
      static_cast<unsigned int>(at::ceil_div(dim, static_cast<int64_t>(THREADS))));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      gathered.scalar_type(),
      "combineGatheredRowsWeightedReduceKernel",
      [&] {
        const scalar_t* gathered_typed = gathered.data_ptr<scalar_t>();
        scalar_t* out_typed = out.mutable_data_ptr<scalar_t>();
        if (probs.scalar_type() == at::kFloat) {
          const float* probs_typed = probs.data_ptr<float>();
          if (route_ranks_ptr == nullptr) {
            combineGatheredRowsWeightedReduceKernel<scalar_t, float, false>
                <<<grid, block, 0, stream>>>(
                    gathered_typed,
                    probs_typed,
                    nullptr,
                    out_typed,
                    num_out_rows,
                    top_k,
                    dim);
          } else {
            combineGatheredRowsWeightedReduceKernel<scalar_t, float, true>
                <<<grid, block, 0, stream>>>(
                    gathered_typed,
                    probs_typed,
                    route_ranks_ptr,
                    out_typed,
                    num_out_rows,
                    top_k,
                    dim);
          }
        } else if (probs.scalar_type() == at::kHalf) {
          const at::Half* probs_typed = probs.data_ptr<at::Half>();
          if (route_ranks_ptr == nullptr) {
            combineGatheredRowsWeightedReduceKernel<scalar_t, at::Half, false>
                <<<grid, block, 0, stream>>>(
                    gathered_typed,
                    probs_typed,
                    nullptr,
                    out_typed,
                    num_out_rows,
                    top_k,
                    dim);
          } else {
            combineGatheredRowsWeightedReduceKernel<scalar_t, at::Half, true>
                <<<grid, block, 0, stream>>>(
                    gathered_typed,
                    probs_typed,
                    route_ranks_ptr,
                    out_typed,
                    num_out_rows,
                    top_k,
                    dim);
          }
        } else {
          const at::BFloat16* probs_typed = probs.data_ptr<at::BFloat16>();
          if (route_ranks_ptr == nullptr) {
            combineGatheredRowsWeightedReduceKernel<scalar_t, at::BFloat16, false>
                <<<grid, block, 0, stream>>>(
                    gathered_typed,
                    probs_typed,
                    nullptr,
                    out_typed,
                    num_out_rows,
                    top_k,
                    dim);
          } else {
            combineGatheredRowsWeightedReduceKernel<scalar_t, at::BFloat16, true>
                <<<grid, block, 0, stream>>>(
                    gathered_typed,
                    probs_typed,
                    route_ranks_ptr,
                    out_typed,
                    num_out_rows,
                    top_k,
                    dim);
          }
        }
      });
}

void rowwise_reduce_gathered_routes_unweighted(
    at::Tensor& gathered,
    at::Tensor& out,
    const std::optional<at::Tensor>& route_ranks) {
  TORCH_CHECK(gathered.dim() == 3, "gathered must be rank-3 [N, K, D]");
  TORCH_CHECK(out.dim() == 2, "out must be rank-2 [N, D]");
  TORCH_CHECK(
      out.size(0) == gathered.size(0) && out.size(1) == gathered.size(2),
      "out shape must match gathered [N, D]");
  TORCH_CHECK(
      gathered.is_contiguous() && out.is_contiguous(),
      "gathered and out must be contiguous");
  TORCH_CHECK(
      gathered.scalar_type() == out.scalar_type(),
      "gathered and out must have the same dtype");

  auto device = gathered.device();
  TORCH_CHECK(
      device.type() == at::DeviceType::CUDA && out.device() == device,
      "all tensor arguments must be on the same CUDA device");

  const int64_t* route_ranks_ptr = nullptr;
  if (route_ranks.has_value()) {
    TORCH_CHECK(route_ranks->defined(), "route_ranks optional tensor must be defined");
    TORCH_CHECK(route_ranks->dim() == 2, "route_ranks must be rank-2 [N, K]");
    TORCH_CHECK(
        route_ranks->size(0) == gathered.size(0) &&
            route_ranks->size(1) == gathered.size(1),
        "route_ranks shape must match gathered [N, K]");
    TORCH_CHECK(
        route_ranks->device() == device,
        "route_ranks must be on the same CUDA device as other arguments");
    TORCH_CHECK(route_ranks->is_contiguous(), "route_ranks must be contiguous");
    TORCH_CHECK(route_ranks->scalar_type() == at::kLong, "route_ranks must be int64");
    route_ranks_ptr = reinterpret_cast<const int64_t*>(route_ranks->data_ptr());
  }
  c10::cuda::CUDAGuard guard(device);

  int64_t num_out_rows = gathered.size(0);
  int64_t top_k = gathered.size(1);
  int64_t dim = gathered.size(2);
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int THREADS = 256;
  dim3 block(THREADS);
  dim3 grid(
      static_cast<unsigned int>(num_out_rows),
      static_cast<unsigned int>(at::ceil_div(dim, static_cast<int64_t>(THREADS))));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      gathered.scalar_type(),
      "combineGatheredRowsReduceKernel",
      [&] {
        const scalar_t* gathered_typed = gathered.data_ptr<scalar_t>();
        scalar_t* out_typed = out.mutable_data_ptr<scalar_t>();
        if (route_ranks_ptr == nullptr) {
          combineGatheredRowsReduceKernel<scalar_t, false>
              <<<grid, block, 0, stream>>>(
                  gathered_typed,
                  nullptr,
                  out_typed,
                  num_out_rows,
                  top_k,
                  dim);
        } else {
          combineGatheredRowsReduceKernel<scalar_t, true>
              <<<grid, block, 0, stream>>>(
                  gathered_typed,
                  route_ranks_ptr,
                  out_typed,
                  num_out_rows,
                  top_k,
                  dim);
        }
      });
}

void rowwise_combine_get_fused(
    at::Tensor& expert_out,
    at::Tensor& out,
    at::Tensor& src_ranks,
    at::Tensor& src_rows,
    const std::optional<at::Tensor>& probs,
    const std::string& group_name,
    int64_t nblocks,
    const std::optional<at::Tensor>& gathered_out,
    bool pre_barrier,
    bool post_barrier) {
  auto* olmo_group = olmo_symm_find_group(group_name);
  TORCH_CHECK(
      olmo_group != nullptr,
      "OLMo rowwise fused combine requires registered OLMo symmetric-memory group ",
      group_name);

  TORCH_CHECK(
      nblocks >= 0, "nblocks must be non-negative (0 means auto), got ", nblocks);
  TORCH_CHECK(expert_out.dim() == 2, "expert_out must be rank-2 [C, D]");
  TORCH_CHECK(out.dim() == 2, "out must be rank-2 [N, D]");
  TORCH_CHECK(
      src_ranks.dim() == 2 && src_rows.dim() == 2,
      "src_ranks and src_rows must be rank-2 [N, K]");
  TORCH_CHECK(
      src_ranks.sizes() == src_rows.sizes(),
      "src_ranks and src_rows must have identical shapes");
  TORCH_CHECK(
      src_ranks.size(0) == out.size(0),
      "src_ranks/src_rows first dim (N) must match out rows");
  TORCH_CHECK(
      expert_out.size(1) == out.size(1),
      "expert_out and out must have the same hidden dim (D)");

  TORCH_CHECK(
      expert_out.is_contiguous() && out.is_contiguous() && src_ranks.is_contiguous() &&
          src_rows.is_contiguous(),
      "expert_out, out, src_ranks and src_rows must be contiguous");
  TORCH_CHECK(
      expert_out.dtype() == out.dtype(),
      "expert_out and out must have the same dtype");
  TORCH_CHECK(
      src_ranks.scalar_type() == at::kLong && src_rows.scalar_type() == at::kLong,
      "src_ranks and src_rows must be int64");

  auto device = expert_out.device();
  TORCH_CHECK(
      device.type() == at::DeviceType::CUDA && out.device() == device &&
          src_ranks.device() == device && src_rows.device() == device,
      "all tensor arguments must be on the same CUDA device");

  const float* probs_ptr = nullptr;
  if (probs.has_value()) {
    TORCH_CHECK(probs->defined(), "probs optional tensor must be defined");
    TORCH_CHECK(
        probs->device() == device,
        "probs must be on the same CUDA device as other arguments");
    TORCH_CHECK(
        probs->is_contiguous(),
        "probs must be contiguous");
    TORCH_CHECK(
        probs->sizes() == src_ranks.sizes(),
        "probs must have shape [N, K] matching src_ranks/src_rows");
    TORCH_CHECK(
        probs->scalar_type() == at::kFloat,
        "probs must be float32");
    probs_ptr = probs->data_ptr<float>();
  }

  if (gathered_out.has_value()) {
    TORCH_CHECK(
        gathered_out->defined(),
        "gathered_out optional tensor must be defined");
    TORCH_CHECK(
        gathered_out->device() == device,
        "gathered_out must be on the same CUDA device as other arguments");
    TORCH_CHECK(
        gathered_out->is_contiguous(),
        "gathered_out must be contiguous");
    TORCH_CHECK(
        gathered_out->scalar_type() == out.scalar_type(),
        "gathered_out must have the same dtype as out");
  }

  c10::cuda::CUDAGuard guard(device);
  auto stream = at::cuda::getCurrentCUDAStream();
  nvshmem_team_t team = NVSHMEM_TEAM_WORLD;
  const int* rank_to_pe_dev = nullptr;
  int group_size = 0;
  bool world_within_direct_access = true;
  rank_to_pe_dev = olmo_group->rank_to_pe_dev;
  group_size = olmo_group->world_size;
  maybe_init_nvshmem_cumodule(
      reinterpret_cast<const void*>(combineRowsGetKernel<float, false>));

  int64_t num_out_rows = out.size(0);
  int64_t top_k = src_ranks.size(1);
  int64_t dim = out.size(1);
  int64_t expert_capacity_rows = expert_out.size(0);
  int64_t expert_row_stride = expert_out.stride(0);
  int64_t out_row_stride = out.stride(0);
  if (gathered_out.has_value()) {
    TORCH_CHECK(
        gathered_out->dim() == 3,
        "gathered_out must be rank-3 [N, K, D]");
    TORCH_CHECK(
        gathered_out->size(0) == num_out_rows &&
            gathered_out->size(1) == top_k &&
            gathered_out->size(2) == dim,
        "gathered_out shape mismatch: expected [",
        num_out_rows,
        ", ",
        top_k,
        ", ",
        dim,
        "]");
  }
  int num_blocks = resolve_num_blocks_rowwise(
      num_out_rows * top_k, nblocks, world_within_direct_access);
  TORCH_CHECK(num_blocks > 0, "resolved nblocks must be > 0");

  if (pre_barrier) {
    int pre_barrier_status = nvshmemx_barrier_on_stream(team, stream.stream());
    TORCH_CHECK(
        pre_barrier_status == 0,
        "nvshmemx_barrier_on_stream (pre) failed with status ",
        pre_barrier_status);
  }

  dim3 block(ROWWISE_COMBINE_FUSED_THREADS_PER_BLOCK);
  constexpr int64_t cols_per_block =
      static_cast<int64_t>(ROWWISE_COMBINE_FUSED_THREADS_PER_BLOCK) *
      ROWWISE_COMBINE_FUSED_VECS_PER_THREAD;
  // Fused path is not collective-launched, so we can oversubscribe row blocks
  // to hide remote-get latency; factor is tunable via env var.
  int64_t row_blocks = resolve_num_row_blocks_fused(num_out_rows, num_blocks);
  dim3 grid(
      static_cast<unsigned int>(row_blocks),
      static_cast<unsigned int>(at::ceil_div(dim, cols_per_block)));

  const int64_t* src_ranks_ptr = reinterpret_cast<const int64_t*>(src_ranks.data_ptr());
  const int64_t* src_rows_ptr = reinterpret_cast<const int64_t*>(src_rows.data_ptr());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      out.scalar_type(),
      "combineRowsGetKernel",
      [&] {
        const scalar_t* expert_out_typed = expert_out.data_ptr<scalar_t>();
        scalar_t* out_typed = out.mutable_data_ptr<scalar_t>();
        scalar_t* gathered_out_typed = gathered_out.has_value()
            ? gathered_out->mutable_data_ptr<scalar_t>()
            : nullptr;
        if (probs_ptr == nullptr) {
          combineRowsGetKernel<scalar_t, false>
              <<<grid, block, 0, stream>>>(
                  expert_out_typed,
                  out_typed,
                  gathered_out_typed,
                  src_ranks_ptr,
                  src_rows_ptr,
                  nullptr,
                  num_out_rows,
                  top_k,
                  dim,
                  expert_row_stride,
                  out_row_stride,
                  expert_capacity_rows,
                  team,
                  rank_to_pe_dev,
                  group_size);
        } else {
          combineRowsGetKernel<scalar_t, true>
              <<<grid, block, 0, stream>>>(
                  expert_out_typed,
                  out_typed,
                  gathered_out_typed,
                  src_ranks_ptr,
                  src_rows_ptr,
                  probs_ptr,
                  num_out_rows,
                  top_k,
                  dim,
                  expert_row_stride,
                  out_row_stride,
                  expert_capacity_rows,
                  team,
                  rank_to_pe_dev,
                  group_size);
        }
      });
  if (post_barrier) {
    int post_barrier_status = nvshmemx_barrier_on_stream(team, stream.stream());
    TORCH_CHECK(
        post_barrier_status == 0,
        "nvshmemx_barrier_on_stream (post) failed with status ",
        post_barrier_status);
  }
}

void rowwise_gather_get(
    at::Tensor& expert_out,
    at::Tensor& out,
    at::Tensor& src_ranks,
    at::Tensor& src_rows,
    const std::string& group_name,
    int64_t nblocks,
    bool pre_barrier,
    bool post_barrier) {
  auto* olmo_group = olmo_symm_find_group(group_name);
  TORCH_CHECK(
      olmo_group != nullptr,
      "OLMo rowwise gather requires registered OLMo symmetric-memory group ",
      group_name);

  TORCH_CHECK(
      nblocks >= 0, "nblocks must be non-negative (0 means auto), got ", nblocks);
  TORCH_CHECK(expert_out.dim() == 2, "expert_out must be rank-2 [C, D]");
  TORCH_CHECK(out.dim() == 2, "out must be rank-2 [R, D]");
  TORCH_CHECK(
      src_ranks.dim() == 2 && src_rows.dim() == 2,
      "src_ranks and src_rows must be rank-2 [R, 1]");
  TORCH_CHECK(
      src_ranks.sizes() == src_rows.sizes(),
      "src_ranks and src_rows must have identical shapes");
  TORCH_CHECK(
      src_ranks.size(1) == 1,
      "rowwise_gather_get expects src_ranks/src_rows shape [R, 1]");
  TORCH_CHECK(
      src_ranks.size(0) == out.size(0),
      "src_ranks/src_rows first dim (R) must match out rows");
  TORCH_CHECK(
      expert_out.size(1) == out.size(1),
      "expert_out and out must have the same hidden dim (D)");
  TORCH_CHECK(
      expert_out.is_contiguous() && out.is_contiguous() && src_ranks.is_contiguous() &&
          src_rows.is_contiguous(),
      "expert_out, out, src_ranks and src_rows must be contiguous");
  TORCH_CHECK(
      expert_out.dtype() == out.dtype(),
      "expert_out and out must have the same dtype");
  TORCH_CHECK(
      src_ranks.scalar_type() == at::kLong && src_rows.scalar_type() == at::kLong,
      "src_ranks and src_rows must be int64");

  auto device = expert_out.device();
  TORCH_CHECK(
      device.type() == at::DeviceType::CUDA && out.device() == device &&
          src_ranks.device() == device && src_rows.device() == device,
      "all tensor arguments must be on the same CUDA device");
  c10::cuda::CUDAGuard guard(device);

  auto stream = at::cuda::getCurrentCUDAStream();
  nvshmem_team_t team = NVSHMEM_TEAM_WORLD;
  const int* rank_to_pe_dev = nullptr;
  int group_size = 0;
  bool world_within_direct_access = true;
  rank_to_pe_dev = olmo_group->rank_to_pe_dev;
  group_size = olmo_group->world_size;
  // rowwise_gather_get is used by combine-2d-offset, where dropped routes are
  // masked downstream by packed_keep_mask. Skipping per-route zero-fill here
  // avoids substantial extra work on ranks with more dropped routes.
  maybe_init_nvshmem_cumodule(reinterpret_cast<const void*>(gatherRowsGet<false>));

  int64_t num_out_rows = out.size(0);
  int64_t top_k = 1;
  int64_t expert_capacity_rows = expert_out.size(0);
  size_t row_bytes =
      static_cast<size_t>(expert_out.stride(0)) * expert_out.element_size();
  int num_blocks = resolve_num_blocks_rowwise(
      num_out_rows, nblocks, world_within_direct_access);
  TORCH_CHECK(num_blocks > 0, "resolved nblocks must be > 0");

  if (pre_barrier) {
    int pre_barrier_status = nvshmemx_barrier_on_stream(team, stream.stream());
    TORCH_CHECK(
        pre_barrier_status == 0,
        "nvshmemx_barrier_on_stream (pre) failed with status ",
        pre_barrier_status);
  }

  const void* expert_out_ptr = expert_out.data_ptr();
  void* out_ptr = out.mutable_data_ptr();
  const int64_t* src_ranks_ptr =
      reinterpret_cast<const int64_t*>(src_ranks.data_ptr());
  const int64_t* src_rows_ptr =
      reinterpret_cast<const int64_t*>(src_rows.data_ptr());

  void* args[] = {
      &expert_out_ptr,
      &out_ptr,
      &src_ranks_ptr,
      &src_rows_ptr,
      &row_bytes,
      &num_out_rows,
      &top_k,
      &expert_capacity_rows,
      &team,
      &rank_to_pe_dev,
      &group_size};
  nvshmemx_collective_launch(
      (const void*)gatherRowsGet<false>,
      dim3(num_blocks),
      dim3(ROWWISE_THREADS_PER_BLOCK),
      args,
      0,
      stream);
  if (post_barrier) {
    int post_barrier_status = nvshmemx_barrier_on_stream(team, stream.stream());
    TORCH_CHECK(
        post_barrier_status == 0,
        "nvshmemx_barrier_on_stream (post) failed with status ",
        post_barrier_status);
  }
}

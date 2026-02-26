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
#include <limits>
#include <mutex>
#include <optional>

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

namespace {

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
    nvshmem_team_t team) {
#ifndef _NVSHMEM_DEVICELIB_SUPPORTED
  CUDA_KERNEL_ASSERT_MSG(false, "SM arch unsupported for NVSHMEM");
#else
  CUDA_KERNEL_ASSERT(team != NVSHMEM_TEAM_INVALID);
  int npes = nvshmem_team_n_pes(team);
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
    auto peer_global = nvshmem_team_translate_pe(
        team, static_cast<int>(peer), NVSHMEM_TEAM_WORLD);
    nvshmemx_putmem_warp(
        (char*)out_data + static_cast<size_t>(dst_row) * row_bytes,
        (const char*)input_data + static_cast<size_t>(src_row) * row_bytes,
        row_bytes,
        peer_global);
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
    nvshmem_team_t team) {
#ifndef _NVSHMEM_DEVICELIB_SUPPORTED
  CUDA_KERNEL_ASSERT_MSG(false, "SM arch unsupported for NVSHMEM");
#else
  CUDA_KERNEL_ASSERT(team != NVSHMEM_TEAM_INVALID);
  int npes = nvshmem_team_n_pes(team);
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
    auto peer_global = nvshmem_team_translate_pe(
        team, static_cast<int>(peer), NVSHMEM_TEAM_WORLD);
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
    const std::string& group_name,
    int64_t nblocks) {
  auto out_hdl = c10d::symmetric_memory::rendezvous(out, group_name);

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
  auto& team_manager = c10d::nvshmem_extension::TeamManager::get(device);
  auto team = team_manager.get_team(group_name, out_hdl->get_rank_to_global_rank());
  maybe_init_nvshmem_cumodule(reinterpret_cast<const void*>(dispatchRowsPut));

  const void* input_ptr = input.data_ptr();
  void* out_ptr = out.mutable_data_ptr();
  const int64_t* dst_ranks_ptr = reinterpret_cast<const int64_t*>(dst_ranks.data_ptr());
  const int64_t* dst_rows_ptr = reinterpret_cast<const int64_t*>(dst_rows.data_ptr());

  int64_t num_input_rows = input.size(0);
  int64_t top_k = dst_ranks.size(1);
  int64_t out_capacity_rows = out.size(0);
  size_t row_bytes = static_cast<size_t>(input.stride(0)) * input.element_size();
  int num_blocks = resolve_num_blocks_rowwise(
      num_input_rows * top_k, nblocks, out_hdl->world_within_direct_access());
  TORCH_CHECK(num_blocks > 0, "resolved nblocks must be > 0");

  // Ensure all peers have completed prior stream work (e.g., local zero_/copy_)
  // before remote puts in this launch start.
  int pre_barrier_status = nvshmemx_barrier_on_stream(team, stream.stream());
  TORCH_CHECK(
      pre_barrier_status == 0,
      "nvshmemx_barrier_on_stream (pre) failed with status ",
      pre_barrier_status);

  void* args[] = {
      &input_ptr,
      &out_ptr,
      &dst_ranks_ptr,
      &dst_rows_ptr,
      &row_bytes,
      &num_input_rows,
      &top_k,
      &out_capacity_rows,
      &team};
  nvshmemx_collective_launch(
      (const void*)dispatchRowsPut,
      dim3(num_blocks),
      dim3(ROWWISE_THREADS_PER_BLOCK),
      args,
      0,
      stream);
  int barrier_status = nvshmemx_barrier_on_stream(team, stream.stream());
  TORCH_CHECK(
      barrier_status == 0,
      "nvshmemx_barrier_on_stream failed with status ",
      barrier_status);
}

void rowwise_combine_get(
    at::Tensor& expert_out,
    at::Tensor& out,
    at::Tensor& src_ranks,
    at::Tensor& src_rows,
    const std::optional<at::Tensor>& probs,
    const std::string& group_name,
    int64_t nblocks) {
  auto expert_out_hdl = c10d::symmetric_memory::rendezvous(expert_out, group_name);

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

  c10::cuda::CUDAGuard guard(device);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto& team_manager = c10d::nvshmem_extension::TeamManager::get(device);
  auto team =
      team_manager.get_team(group_name, expert_out_hdl->get_rank_to_global_rank());
  maybe_init_nvshmem_cumodule(reinterpret_cast<const void*>(gatherRowsGet<true>));

  int64_t num_out_rows = out.size(0);
  int64_t top_k = src_ranks.size(1);
  int64_t dim = out.size(1);
  int64_t expert_capacity_rows = expert_out.size(0);
  size_t row_bytes =
      static_cast<size_t>(expert_out.stride(0)) * expert_out.element_size();
  int num_blocks = resolve_num_blocks_rowwise(
      num_out_rows * top_k, nblocks, expert_out_hdl->world_within_direct_access());
  TORCH_CHECK(num_blocks > 0, "resolved nblocks must be > 0");

  // Local temporary gather buffer [N, K, D] before reduction to [N, D].
  auto gathered = at::empty({num_out_rows, top_k, dim}, out.options());

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
      &team};
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
}

void rowwise_gather_get(
    at::Tensor& expert_out,
    at::Tensor& out,
    at::Tensor& src_ranks,
    at::Tensor& src_rows,
    const std::string& group_name,
    int64_t nblocks) {
  auto expert_out_hdl = c10d::symmetric_memory::rendezvous(expert_out, group_name);

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
  auto& team_manager = c10d::nvshmem_extension::TeamManager::get(device);
  auto team =
      team_manager.get_team(group_name, expert_out_hdl->get_rank_to_global_rank());
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
      num_out_rows, nblocks, expert_out_hdl->world_within_direct_access());
  TORCH_CHECK(num_blocks > 0, "resolved nblocks must be > 0");

  // Ensure all peers have completed prior stream work before remote gets start.
  int pre_barrier_status = nvshmemx_barrier_on_stream(team, stream.stream());
  TORCH_CHECK(
      pre_barrier_status == 0,
      "nvshmemx_barrier_on_stream (pre) failed with status ",
      pre_barrier_status);

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
      &team};
  nvshmemx_collective_launch(
      (const void*)gatherRowsGet<false>,
      dim3(num_blocks),
      dim3(ROWWISE_THREADS_PER_BLOCK),
      args,
      0,
      stream);
}

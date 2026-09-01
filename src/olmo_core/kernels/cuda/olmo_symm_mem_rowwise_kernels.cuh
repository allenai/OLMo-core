#include <cuda_fp8.h>
#include <type_traits>

template <typename scalar_t>
__device__ __forceinline__ float rowwise_weighted_value_like_torch(
    scalar_t value,
    float prob) {
  scalar_t prob_t = static_cast<scalar_t>(prob);
  scalar_t weighted_t =
      static_cast<scalar_t>(static_cast<float>(value) * static_cast<float>(prob_t));
  return static_cast<float>(weighted_t);
}

template <>
__device__ __forceinline__ float rowwise_weighted_value_like_torch<float>(
    float value,
    float prob) {
  return value * prob;
}

template <typename scalar_t>
__device__ __forceinline__ uint8_t rowwise_mxfp8_scale_byte(float max_abs) {
  constexpr float FP32_TINY = 1.1754943508222875e-38f;
  max_abs = max_abs == max_abs ? max_abs : 0.0f;
  max_abs = fmaxf(max_abs, FP32_TINY);

  int largest_p2 = 0;
  if constexpr (std::is_same<scalar_t, at::BFloat16>::value) {
    // Match the Triton quantizer's BF16 exponent shortcut.
    __nv_bfloat16 max_abs_bf16 = __float2bfloat16(max_abs);
    __nv_bfloat16_raw max_abs_raw = static_cast<__nv_bfloat16_raw>(max_abs_bf16);
    uint16_t max_abs_bits = max_abs_raw.x;
    largest_p2 = static_cast<int>((max_abs_bits >> 7) & 0xFF) - 127;
  } else {
    largest_p2 = ilogbf(max_abs);
  }

  int scale_unbiased = largest_p2 - 8;
  scale_unbiased = max(scale_unbiased, -127);
  scale_unbiased = min(scale_unbiased, 127);
  return static_cast<uint8_t>(scale_unbiased + 127);
}

__device__ __forceinline__ float rowwise_mxfp8_inv_scale(uint8_t scale_byte) {
  return ldexpf(1.0f, 127 - static_cast<int>(scale_byte));
}

__device__ __forceinline__ uint8_t rowwise_mxfp8_q_byte(float value) {
  constexpr float F8E4M3_MAX = 448.0f;
  value = value == value ? value : 0.0f;
  value = fmaxf(value, -F8E4M3_MAX);
  value = fminf(value, F8E4M3_MAX);
  return __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
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

__global__ void dispatchRowsPutPair(
    const void* input_q_data,
    const void* input_scales_data,
    void* out_q_data,
    void* out_scales_data,
    const int64_t* dst_ranks,
    const int64_t* dst_rows,
    size_t q_row_bytes,
    size_t scales_row_bytes,
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
    auto peer_global = olmo_route_peer_global(team, rank_to_pe, static_cast<int>(peer));
    nvshmemx_putmem_warp(
        (char*)out_q_data + static_cast<size_t>(dst_row) * q_row_bytes,
        (const char*)input_q_data + static_cast<size_t>(src_row) * q_row_bytes,
        q_row_bytes,
        peer_global);
    __syncwarp();
    nvshmemx_putmem_warp(
        (char*)out_scales_data + static_cast<size_t>(dst_row) * scales_row_bytes,
        (const char*)input_scales_data + static_cast<size_t>(src_row) * scales_row_bytes,
        scales_row_bytes,
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

__global__ void inverseRouteMetaPutCompactScalar(
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
      auto peer_global = olmo_route_peer_global(team, rank_to_pe, static_cast<int>(peer));
      nvshmem_int64_p(inverse_route_meta + dst_row * 2, src_rank, peer_global);
      nvshmem_int64_p(inverse_route_meta + dst_row * 2 + 1, route_id, peer_global);
    }
  }
#endif
}

__global__ void buildInverseRouteMetaFromGlobalRecords(
    int64_t* inverse_route_meta,
    const int64_t* global_route_records,
    const int64_t* global_wave_offsets,
    int64_t world_size,
    int64_t records_per_rank,
    int64_t wave_offsets_per_rank,
    int64_t num_waves,
    int64_t local_rank,
    int64_t inverse_capacity_rows) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  int64_t total_records = world_size * records_per_rank;
  for (; idx < total_records; idx += stride) {
    int64_t src_rank = idx / records_per_rank;
    int64_t local_idx = idx - src_rank * records_per_rank;
    int64_t valid_records =
        global_wave_offsets[src_rank * wave_offsets_per_rank + num_waves];
    if (local_idx >= valid_records) {
      continue;
    }

    const int64_t* record = global_route_records + idx * 4;
    int64_t dst_row = record[1];
    int64_t peer = record[2];
    int64_t route_id = record[3];
    if (peer != local_rank) {
      continue;
    }

    CUDA_KERNEL_ASSERT(dst_row >= 0 && dst_row < inverse_capacity_rows);
    int64_t* meta = inverse_route_meta + dst_row * 2;
    meta[0] = src_rank;
    meta[1] = route_id;
  }
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

// Experimental weighted MXFP8 dispatch kernel. It exists because materializing
// [N, top_k, D] weighted grads is an obvious-looking overhead in MoE combine
// backward. Benchmarks on B300 showed this direct NVSHMEM path is bit-exact
// against the materialized path, but not faster at the normal rowwise nblocks
// setting; keep it as a reference/tuning target rather than production code.
template <typename scalar_t>
__global__ void dispatchRowsPutScaledWeighted(
    const scalar_t* input_data,
    uint8_t* out_q_data,
    uint8_t* out_scales_data,
    const int64_t* dst_ranks,
    const int64_t* dst_rows,
    const float* probs,
    int64_t num_input_rows,
    int64_t top_k,
    int64_t dim,
    int64_t scale_dim,
    int64_t input_row_stride,
    int64_t out_q_row_stride,
    int64_t out_scales_row_stride,
    int64_t out_capacity_rows,
    nvshmem_team_t team,
    const int* rank_to_pe,
    int group_size) {
#ifndef _NVSHMEM_DEVICELIB_SUPPORTED
  CUDA_KERNEL_ASSERT_MSG(false, "SM arch unsupported for NVSHMEM");
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
  // The per-warp shared staging buffers exceed the 48 KB static shared-memory
  // limit on pre-SM100 GPUs. This is an SM100+ reference/tuning kernel and is
  // fail-closed at runtime below SM100, so emit a trap stub for older archs.
  CUDA_KERNEL_ASSERT_MSG(false, "dispatchRowsPutScaledWeighted requires SM100+");
#else
  CUDA_KERNEL_ASSERT(team != NVSHMEM_TEAM_INVALID);
  constexpr int BLOCK_SIZE = ROWWISE_MXFP8_BLOCK_SIZE;
  constexpr int CHUNK_BLOCKS = ROWWISE_MXFP8_Q_CHUNK_BLOCKS;
  constexpr int CHUNK_Q_BYTES = BLOCK_SIZE * CHUNK_BLOCKS;

  __shared__ uint8_t shared_q[ROWWISE_WARPS_PER_BLOCK][CHUNK_Q_BYTES];
  __shared__ uint8_t shared_scales[ROWWISE_WARPS_PER_BLOCK]
                                  [ROWWISE_MXFP8_MAX_SCALE_GROUPS];

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
    CUDA_KERNEL_ASSERT(scale_dim <= ROWWISE_MXFP8_MAX_SCALE_GROUPS);

    int64_t src_row = route_id / top_k;
    float p = probs[route_id];
    int peer_global = 0;
    if (lane_id == 0) {
      peer_global = olmo_route_peer_global(team, rank_to_pe, static_cast<int>(peer));
    }
    peer_global = __shfl_sync(0xffffffff, peer_global, 0);

    const scalar_t* src_ptr = input_data + src_row * input_row_stride;
    uint8_t* q_shared = shared_q[warp_id];
    uint8_t* scale_shared = shared_scales[warp_id];

    for (int64_t scale_base = 0; scale_base < scale_dim; scale_base += CHUNK_BLOCKS) {
      int64_t remaining_scale_groups = scale_dim - scale_base;
      int chunk_groups = static_cast<int>(
          remaining_scale_groups < CHUNK_BLOCKS ? remaining_scale_groups : CHUNK_BLOCKS);

      for (int local_group = 0; local_group < chunk_groups; ++local_group) {
        int64_t scale_group = scale_base + local_group;
        int64_t col = scale_group * BLOCK_SIZE + lane_id;
        float weighted = rowwise_weighted_value_like_torch(src_ptr[col], p);
        float max_abs = fabsf(weighted);
        max_abs = max_abs == max_abs ? max_abs : 0.0f;

#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
          max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffff, max_abs, offset));
        }
        max_abs = __shfl_sync(0xffffffff, max_abs, 0);

        uint8_t scale_byte = 0;
        if (lane_id == 0) {
          scale_byte = rowwise_mxfp8_scale_byte<scalar_t>(max_abs);
          scale_shared[scale_group] = scale_byte;
        }
        scale_byte = static_cast<uint8_t>(__shfl_sync(
            0xffffffff, static_cast<unsigned int>(scale_byte), 0));
        float q_hp = weighted * rowwise_mxfp8_inv_scale(scale_byte);
        q_shared[local_group * BLOCK_SIZE + lane_id] = rowwise_mxfp8_q_byte(q_hp);
      }

      __syncwarp();
      nvshmemx_putmem_warp(
          out_q_data +
              static_cast<size_t>(dst_row * out_q_row_stride + scale_base * BLOCK_SIZE),
          q_shared,
          static_cast<size_t>(chunk_groups * BLOCK_SIZE),
          peer_global);
      __syncwarp();
    }

    nvshmemx_putmem_warp(
        out_scales_data + static_cast<size_t>(dst_row * out_scales_row_stride),
        scale_shared,
        static_cast<size_t>(scale_dim),
        peer_global);
    __syncwarp();
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

template <bool ZERO_INVALID_ROWS>
__global__ void gatherRowsGetPair(
    const void* expert_q_data,
    const void* expert_scales_data,
    void* gathered_q_data,
    void* gathered_scales_data,
    const int64_t* src_ranks,
    const int64_t* src_rows,
    size_t q_row_bytes,
    size_t scales_row_bytes,
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

  int64_t route_id =
      static_cast<int64_t>(blockIdx.x) * ROWWISE_WARPS_PER_BLOCK + warp_id;
  int64_t route_stride =
      static_cast<int64_t>(gridDim.x) * ROWWISE_WARPS_PER_BLOCK;
  for (; route_id < num_routes; route_id += route_stride) {
    int64_t peer = src_ranks[route_id];
    int64_t src_row = src_rows[route_id];
    char* q_dst_ptr = (char*)gathered_q_data + static_cast<size_t>(route_id) * q_row_bytes;
    char* scales_dst_ptr =
        (char*)gathered_scales_data + static_cast<size_t>(route_id) * scales_row_bytes;
    if (peer < 0 || src_row < 0) {
      if constexpr (ZERO_INVALID_ROWS) {
        for (size_t i = static_cast<size_t>(lane_id); i < q_row_bytes; i += WARP_SIZE) {
          q_dst_ptr[i] = 0;
        }
        for (size_t i = static_cast<size_t>(lane_id); i < scales_row_bytes; i += WARP_SIZE) {
          scales_dst_ptr[i] = 0;
        }
      }
      continue;
    }

    CUDA_KERNEL_ASSERT(peer < npes);
    CUDA_KERNEL_ASSERT(src_row < expert_capacity_rows);
    auto peer_global = olmo_route_peer_global(team, rank_to_pe, static_cast<int>(peer));
    nvshmemx_getmem_warp(
        q_dst_ptr,
        (const char*)expert_q_data + static_cast<size_t>(src_row) * q_row_bytes,
        q_row_bytes,
        peer_global);
    __syncwarp();
    nvshmemx_getmem_warp(
        scales_dst_ptr,
        (const char*)expert_scales_data + static_cast<size_t>(src_row) * scales_row_bytes,
        scales_row_bytes,
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

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

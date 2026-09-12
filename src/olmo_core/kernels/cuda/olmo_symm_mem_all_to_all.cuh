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
  checked_collective_launch(
      "exchangeSplitAndOffset_2d<false>",
      (const void*)exchangeSplitAndOffset_2d<false>,
      1,
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
  checked_collective_launch(
      "allToAllV_2d",
      (const void*)allToAllV_2d,
      num_blocks,
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
  checked_collective_launch(
      "exchangeSplitAndOffset_2d<true>",
      (const void*)exchangeSplitAndOffset_2d<true>,
      1,
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
  checked_collective_launch(
      "allToAllV_2d",
      (const void*)allToAllV_2d,
      num_blocks,
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

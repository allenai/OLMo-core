void preflight_rowwise_collective_launches(int64_t nblocks) {
  TORCH_CHECK(
      nblocks > 0,
      "strict NVSHMEM rowwise collective preflight requires explicit "
      "rowwise_nblocks > 0 (0 means auto and cannot be validated before "
      "runtime)");
  TORCH_CHECK(
      nblocks <= std::numeric_limits<int>::max(),
      "rowwise_nblocks is too large: ",
      nblocks);
  int requested_blocks = static_cast<int>(nblocks);

  auto preflight = [requested_blocks](
                       const char* kernel_name,
                       const void* kernel,
                       dim3 block_dims,
                       void** args,
                       size_t shared_mem) {
    maybe_init_nvshmem_cumodule(kernel);
    int max_grid =
        query_collective_launch_max_grid(kernel_name, kernel, block_dims, args, shared_mem);
    TORCH_CHECK(
        requested_blocks <= max_grid,
        "NVSHMEM rowwise collective launch preflight failed for ",
        kernel_name,
        ": requested rowwise_nblocks=",
        requested_blocks,
        " exceeds max_grid=",
        max_grid,
        " for block=(",
        block_dims.x,
        ",",
        block_dims.y,
        ",",
        block_dims.z,
        "). Lower ep.rowwise_nblocks / ROWWISE_A2A_NBLOCKS "
        "(for the OLMoE3 testrun script, set "
        "OLMOE3_TESTRUN_ROWWISE_NBLOCKS).");
  };

  const void* const_data_ptr = nullptr;
  void* data_ptr = nullptr;
  const int64_t* const_i64_ptr = nullptr;
  int64_t* i64_ptr = nullptr;
  const int* rank_to_pe_dev = nullptr;
  const float* probs_ptr = nullptr;
  const at::Half* half_input_ptr = nullptr;
  at::Half* half_out_ptr = nullptr;
  const at::BFloat16* bf16_input_ptr = nullptr;
  at::BFloat16* bf16_out_ptr = nullptr;
  nvshmem_team_t team = NVSHMEM_TEAM_WORLD;
  size_t row_bytes = 0;
  int64_t num_input_rows = 0;
  int64_t num_out_rows = 0;
  int64_t top_k = 0;
  int64_t dim = 0;
  int64_t input_row_stride = 0;
  int64_t out_row_stride = 0;
  int64_t out_capacity_rows = 0;
  int64_t wave_idx = 0;
  int64_t num_waves = 0;
  int64_t src_rank = 0;
  int64_t inverse_capacity_rows = 0;
  int64_t expert_capacity_rows = 0;
  int64_t gathered_capacity_rows = 0;
  int group_size = 0;

  {
    void* args[] = {
        &const_data_ptr,
        &data_ptr,
        &const_i64_ptr,
        &const_i64_ptr,
        &row_bytes,
        &num_input_rows,
        &top_k,
        &out_capacity_rows,
        &team,
        &rank_to_pe_dev,
        &group_size};
    preflight(
        "dispatchRowsPut",
        (const void*)dispatchRowsPut,
        dim3(ROWWISE_THREADS_PER_BLOCK),
        args,
        0);
  }

  {
    void* args[] = {
        &half_input_ptr,
        &half_out_ptr,
        &const_i64_ptr,
        &const_i64_ptr,
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
    preflight(
        "dispatchRowsPutWeighted<Half>",
        (const void*)dispatchRowsPutWeighted<at::Half>,
        dim3(ROWWISE_THREADS_PER_BLOCK),
        args,
        0);
  }

  {
    void* args[] = {
        &bf16_input_ptr,
        &bf16_out_ptr,
        &const_i64_ptr,
        &const_i64_ptr,
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
    preflight(
        "dispatchRowsPutWeighted<BFloat16>",
        (const void*)dispatchRowsPutWeighted<at::BFloat16>,
        dim3(ROWWISE_THREADS_PER_BLOCK),
        args,
        0);
  }

  {
    void* args[] = {
        &const_data_ptr,
        &data_ptr,
        &const_i64_ptr,
        &const_i64_ptr,
        &wave_idx,
        &row_bytes,
        &num_input_rows,
        &out_capacity_rows,
        &team,
        &rank_to_pe_dev,
        &group_size};
    preflight(
        "dispatchRowsPutCompact",
        (const void*)dispatchRowsPutCompact,
        dim3(ROWWISE_THREADS_PER_BLOCK),
        args,
        0);
  }

  {
    void* args[] = {
        &half_input_ptr,
        &half_out_ptr,
        &const_i64_ptr,
        &const_i64_ptr,
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
    preflight(
        "dispatchRowsPutCompactWeighted<Half>",
        (const void*)dispatchRowsPutCompactWeighted<at::Half>,
        dim3(ROWWISE_THREADS_PER_BLOCK),
        args,
        0);
  }

  {
    void* args[] = {
        &bf16_input_ptr,
        &bf16_out_ptr,
        &const_i64_ptr,
        &const_i64_ptr,
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
    preflight(
        "dispatchRowsPutCompactWeighted<BFloat16>",
        (const void*)dispatchRowsPutCompactWeighted<at::BFloat16>,
        dim3(ROWWISE_THREADS_PER_BLOCK),
        args,
        0);
  }

  {
    void* args[] = {
        &i64_ptr,
        &const_i64_ptr,
        &const_i64_ptr,
        &num_waves,
        &src_rank,
        &inverse_capacity_rows,
        &team,
        &rank_to_pe_dev,
        &group_size};
    preflight(
        "inverseRouteMetaPutCompact",
        (const void*)inverseRouteMetaPutCompact,
        dim3(ROWWISE_THREADS_PER_BLOCK),
        args,
        0);
  }

  {
    void* args[] = {
        &i64_ptr,
        &const_i64_ptr,
        &const_i64_ptr,
        &num_waves,
        &src_rank,
        &inverse_capacity_rows,
        &team,
        &rank_to_pe_dev,
        &group_size};
    preflight(
        "inverseRouteMetaPutCompactScalar",
        (const void*)inverseRouteMetaPutCompactScalar,
        dim3(ROWWISE_THREADS_PER_BLOCK),
        args,
        0);
  }

  {
    void* args[] = {
        &const_data_ptr,
        &data_ptr,
        &const_i64_ptr,
        &const_i64_ptr,
        &row_bytes,
        &num_out_rows,
        &top_k,
        &expert_capacity_rows,
        &team,
        &rank_to_pe_dev,
        &group_size};
    preflight(
        "gatherRowsGet<true>",
        (const void*)gatherRowsGet<true>,
        dim3(ROWWISE_THREADS_PER_BLOCK),
        args,
        0);
  }

  {
    const int64_t* row_start_ptr = nullptr;
    const int64_t* num_rows_ptr = nullptr;
    void* args[] = {
        &const_data_ptr,
        &data_ptr,
        &const_i64_ptr,
        &row_start_ptr,
        &num_rows_ptr,
        &row_bytes,
        &expert_capacity_rows,
        &gathered_capacity_rows,
        &team,
        &rank_to_pe_dev,
        &group_size};
    preflight(
        "combineRowsPutRange",
        (const void*)combineRowsPutRange,
        dim3(ROWWISE_THREADS_PER_BLOCK),
        args,
        0);
  }

  {
    void* args[] = {
        &const_data_ptr,
        &data_ptr,
        &const_i64_ptr,
        &const_i64_ptr,
        &row_bytes,
        &num_out_rows,
        &top_k,
        &expert_capacity_rows,
        &team,
        &rank_to_pe_dev,
        &group_size};
    preflight(
        "gatherRowsGet<false>",
        (const void*)gatherRowsGet<false>,
        dim3(ROWWISE_THREADS_PER_BLOCK),
        args,
        0);
  }
}

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

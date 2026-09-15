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

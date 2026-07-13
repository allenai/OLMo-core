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
        reinterpret_cast<const void*>(dispatchRowsPutWeighted<at::BFloat16>));
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
    checked_collective_launch(
        "dispatchRowsPut",
        (const void*)dispatchRowsPut,
        num_blocks,
        dim3(ROWWISE_THREADS_PER_BLOCK),
        args,
        0,
        stream);
  } else {
    AT_DISPATCH_SWITCH(
        input.scalar_type(),
        "dispatchRowsPutWeighted",
        AT_DISPATCH_CASE(at::kHalf, [&] {
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
          checked_collective_launch(
              "dispatchRowsPutWeighted",
              (const void*)dispatchRowsPutWeighted<scalar_t>,
              num_blocks,
              dim3(ROWWISE_THREADS_PER_BLOCK),
              args,
              0,
              stream);
        })
        AT_DISPATCH_CASE(at::kBFloat16, [&] {
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
          checked_collective_launch(
              "dispatchRowsPutWeighted",
              (const void*)dispatchRowsPutWeighted<scalar_t>,
              num_blocks,
              dim3(ROWWISE_THREADS_PER_BLOCK),
              args,
              0,
              stream);
        }));
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
  checked_collective_launch(
      "dispatchRowsPutCompact",
      (const void*)dispatchRowsPutCompact,
      num_blocks,
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
      reinterpret_cast<const void*>(dispatchRowsPutCompactWeighted<at::BFloat16>));

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

  AT_DISPATCH_SWITCH(
      input.scalar_type(),
      "dispatchRowsPutCompactWeighted",
      AT_DISPATCH_CASE(at::kHalf, [&] {
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
        checked_collective_launch(
            "dispatchRowsPutCompactWeighted",
            (const void*)dispatchRowsPutCompactWeighted<scalar_t>,
            num_blocks,
            dim3(ROWWISE_THREADS_PER_BLOCK),
            args,
            0,
            stream);
      })
      AT_DISPATCH_CASE(at::kBFloat16, [&] {
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
        checked_collective_launch(
            "dispatchRowsPutCompactWeighted",
            (const void*)dispatchRowsPutCompactWeighted<scalar_t>,
            num_blocks,
            dim3(ROWWISE_THREADS_PER_BLOCK),
            args,
            0,
            stream);
      }));

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
    bool post_barrier,
    bool scalar_put) {
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
  const void* kernel = scalar_put
      ? reinterpret_cast<const void*>(inverseRouteMetaPutCompactScalar)
      : reinterpret_cast<const void*>(inverseRouteMetaPutCompact);
  maybe_init_nvshmem_cumodule(kernel);

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
  checked_collective_launch(
      scalar_put ? "inverseRouteMetaPutCompactScalar" : "inverseRouteMetaPutCompact",
      kernel,
      num_blocks,
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

void rowwise_build_inverse_route_meta_from_global_records(
    at::Tensor& inverse_route_meta,
    at::Tensor& global_route_records,
    at::Tensor& global_wave_offsets,
    int64_t local_rank,
    int64_t nblocks) {
  TORCH_CHECK(
      nblocks >= 0, "nblocks must be non-negative (0 means auto), got ", nblocks);
  TORCH_CHECK(
      inverse_route_meta.dim() == 2 && inverse_route_meta.size(1) == 2,
      "inverse_route_meta must be rank-2 [C, 2]");
  TORCH_CHECK(
      global_route_records.dim() == 3 && global_route_records.size(2) == 4,
      "global_route_records must be rank-3 [world_size, R, 4]");
  TORCH_CHECK(
      global_wave_offsets.dim() == 2,
      "global_wave_offsets must be rank-2 [world_size, num_waves + 1]");
  TORCH_CHECK(
      global_route_records.size(0) == global_wave_offsets.size(0),
      "global_route_records and global_wave_offsets world sizes must match");
  TORCH_CHECK(
      global_wave_offsets.size(1) >= 2,
      "global_wave_offsets must contain at least one wave");
  TORCH_CHECK(local_rank >= 0, "local_rank must be non-negative");
  TORCH_CHECK(
      local_rank < global_route_records.size(0),
      "local_rank must be less than world_size");
  TORCH_CHECK(
      inverse_route_meta.is_contiguous() && global_route_records.is_contiguous() &&
          global_wave_offsets.is_contiguous(),
      "inverse_route_meta, global_route_records, and global_wave_offsets must be contiguous");
  TORCH_CHECK(
      inverse_route_meta.scalar_type() == at::kLong &&
          global_route_records.scalar_type() == at::kLong &&
          global_wave_offsets.scalar_type() == at::kLong,
      "inverse_route_meta, global_route_records, and global_wave_offsets must be int64");

  auto device = inverse_route_meta.device();
  TORCH_CHECK(
      device.type() == at::DeviceType::CUDA &&
          global_route_records.device() == device &&
          global_wave_offsets.device() == device,
      "all tensor arguments must be on the same CUDA device");
  c10::cuda::CUDAGuard guard(device);

  int64_t world_size = global_route_records.size(0);
  int64_t records_per_rank = global_route_records.size(1);
  int64_t wave_offsets_per_rank = global_wave_offsets.size(1);
  int64_t num_waves = wave_offsets_per_rank - 1;
  int64_t inverse_capacity_rows = inverse_route_meta.size(0);
  int num_blocks = resolve_num_blocks_rowwise(
      world_size * records_per_rank, nblocks, true);
  TORCH_CHECK(num_blocks > 0, "resolved nblocks must be > 0");

  auto stream = at::cuda::getCurrentCUDAStream();
  int64_t* inverse_route_meta_ptr =
      reinterpret_cast<int64_t*>(inverse_route_meta.mutable_data_ptr());
  const int64_t* global_route_records_ptr =
      reinterpret_cast<const int64_t*>(global_route_records.data_ptr());
  const int64_t* global_wave_offsets_ptr =
      reinterpret_cast<const int64_t*>(global_wave_offsets.data_ptr());
  buildInverseRouteMetaFromGlobalRecords<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
      inverse_route_meta_ptr,
      global_route_records_ptr,
      global_wave_offsets_ptr,
      world_size,
      records_per_rank,
      wave_offsets_per_rank,
      num_waves,
      local_rank,
      inverse_capacity_rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
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
  checked_collective_launch(
      "gatherRowsGet<true>",
      (const void*)gatherRowsGet<true>,
      num_blocks,
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
  checked_collective_launch(
      "combineRowsPutRange",
      (const void*)combineRowsPutRange,
      num_blocks,
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
  checked_collective_launch(
      "gatherRowsGet<false>",
      (const void*)gatherRowsGet<false>,
      num_blocks,
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

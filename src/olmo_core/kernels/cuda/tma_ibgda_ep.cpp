#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <pybind11/stl.h>

#include <optional>
#include <limits>
#include <tuple>
#include <vector>

#include "olmo_bf16_tma_ibgda_ep/metadata.cuh"
#include "olmo_bf16_tma_ibgda_ep/workspace.cuh"

namespace py = pybind11;

size_t align_up(size_t value, size_t alignment) {
  TORCH_CHECK(alignment > 0, "alignment must be positive");
  return ((value + alignment - 1) / alignment) * alignment;
}

py::dict peer_window_layout_to_dict(
    const olmo::tma_ibgda_ep::PeerWindowLayout& layout) {
  py::dict out;
  out["route_records_offset"] = py::int_(layout.route_records_offset);
  out["routes_per_rank_offset"] = py::int_(layout.routes_per_rank_offset);
  out["rank_offsets_offset"] = py::int_(layout.rank_offsets_offset);
  out["overflow_by_rank_offset"] = py::int_(layout.overflow_by_rank_offset);
  out["payload_window_offset"] = py::int_(layout.payload_window_offset);
  out["send_doorbells_offset"] = py::int_(layout.send_doorbells_offset);
  out["recv_completions_offset"] = py::int_(layout.recv_completions_offset);
  out["rank_stride_bytes"] = py::int_(layout.rank_stride_bytes);
  out["route_records_bytes"] = py::int_(layout.route_records_bytes);
  out["routes_per_rank_bytes"] = py::int_(layout.routes_per_rank_bytes);
  out["rank_offsets_bytes"] = py::int_(layout.rank_offsets_bytes);
  out["overflow_by_rank_bytes"] = py::int_(layout.overflow_by_rank_bytes);
  out["payload_window_bytes_per_rank"] =
      py::int_(layout.payload_window_bytes_per_rank);
  out["send_doorbells_bytes"] = py::int_(layout.send_doorbells_bytes);
  out["recv_completions_bytes"] = py::int_(layout.recv_completions_bytes);
  out["ep_world_size"] = py::int_(layout.ep_world_size);
  out["rank_capacity"] = py::int_(layout.rank_capacity);
  out["hidden_size"] = py::int_(layout.hidden_size);
  out["dtype_bytes"] = py::int_(layout.dtype_bytes);
  out["total_peer_window_bytes"] =
      py::int_(layout.rank_stride_bytes * layout.ep_world_size);
  return out;
}

py::dict plan_peer_window_layout(
    int64_t num_routes,
    int64_t ep_world_size,
    int64_t rank_capacity,
    int64_t hidden_size,
    int64_t dtype_bytes) {
  TORCH_CHECK(num_routes >= 0, "num_routes must be non-negative");
  TORCH_CHECK(ep_world_size > 0, "ep_world_size must be positive");
  TORCH_CHECK(rank_capacity >= 0, "rank_capacity must be non-negative");
  TORCH_CHECK(hidden_size > 0, "hidden_size must be positive");
  TORCH_CHECK(dtype_bytes > 0, "dtype_bytes must be positive");
  TORCH_CHECK(
      ep_world_size <= std::numeric_limits<int32_t>::max(),
      "ep_world_size too large for PeerWindowLayout");
  TORCH_CHECK(
      rank_capacity <= std::numeric_limits<int32_t>::max(),
      "rank_capacity too large for PeerWindowLayout");
  TORCH_CHECK(
      hidden_size <= std::numeric_limits<int32_t>::max(),
      "hidden_size too large for PeerWindowLayout");
  TORCH_CHECK(
      dtype_bytes <= std::numeric_limits<int32_t>::max(),
      "dtype_bytes too large for PeerWindowLayout");

  constexpr size_t alignment = olmo::tma_ibgda_ep::kWorkspaceAlignment;
  olmo::tma_ibgda_ep::PeerWindowLayout layout{};
  layout.route_records_bytes =
      static_cast<size_t>(num_routes) * sizeof(olmo::tma_ibgda_ep::RouteRecord);
  layout.routes_per_rank_bytes =
      static_cast<size_t>(ep_world_size) * sizeof(int64_t);
  layout.rank_offsets_bytes =
      static_cast<size_t>(ep_world_size + 1) * sizeof(int64_t);
  layout.overflow_by_rank_bytes = static_cast<size_t>(ep_world_size);
  layout.payload_window_bytes_per_rank =
      static_cast<size_t>(rank_capacity) * static_cast<size_t>(hidden_size) *
      static_cast<size_t>(dtype_bytes);
  layout.send_doorbells_bytes =
      static_cast<size_t>(ep_world_size) * olmo::tma_ibgda_ep::kDoorbellBytes;
  layout.recv_completions_bytes =
      static_cast<size_t>(ep_world_size) * olmo::tma_ibgda_ep::kCompletionBytes;

  size_t cursor = 0;
  layout.route_records_offset = cursor;
  cursor += layout.route_records_bytes;
  cursor = align_up(cursor, alignment);

  layout.routes_per_rank_offset = cursor;
  cursor += layout.routes_per_rank_bytes;
  cursor = align_up(cursor, alignment);

  layout.rank_offsets_offset = cursor;
  cursor += layout.rank_offsets_bytes;
  cursor = align_up(cursor, alignment);

  layout.overflow_by_rank_offset = cursor;
  cursor += layout.overflow_by_rank_bytes;
  cursor = align_up(cursor, alignment);

  layout.payload_window_offset = cursor;
  cursor += layout.payload_window_bytes_per_rank;
  cursor = align_up(cursor, alignment);

  layout.send_doorbells_offset = cursor;
  cursor += layout.send_doorbells_bytes;
  cursor = align_up(cursor, alignment);

  layout.recv_completions_offset = cursor;
  cursor += layout.recv_completions_bytes;
  layout.rank_stride_bytes = align_up(cursor, alignment);

  layout.ep_world_size = static_cast<int32_t>(ep_world_size);
  layout.rank_capacity = static_cast<int32_t>(rank_capacity);
  layout.hidden_size = static_cast<int32_t>(hidden_size);
  layout.dtype_bytes = static_cast<int32_t>(dtype_bytes);

  return peer_window_layout_to_dict(layout);
}

void tma_ibgda_dispatch_bf16_peer_launcher(
    const torch::Tensor& input,
    const torch::Tensor& out,
    const torch::Tensor& dst_ranks,
    const torch::Tensor& dst_rows,
    const torch::Tensor& peer_out_ptrs,
    const std::optional<torch::Tensor>& probs,
    int64_t nblocks);

void tma_ibgda_combine_bf16_peer_launcher(
    const torch::Tensor& expert_out,
    const torch::Tensor& out,
    const torch::Tensor& src_ranks,
    const torch::Tensor& src_rows,
    const torch::Tensor& peer_expert_out_ptrs,
    const std::optional<torch::Tensor>& probs);

void tma_ibgda_dispatch_bf16_ibgda_launcher(
    const torch::Tensor& input,
    const torch::Tensor& out,
    const torch::Tensor& dst_ranks,
    const torch::Tensor& dst_rows,
    const std::optional<torch::Tensor>& probs,
    int64_t nblocks);

void tma_ibgda_dispatch_bf16_ibgda_records_launcher(
    const torch::Tensor& input,
    const torch::Tensor& out,
    const torch::Tensor& route_records,
    int64_t nblocks);

void tma_ibgda_dispatch_bf16_ibgda_records_tma_launcher(
    const torch::Tensor& input,
    const torch::Tensor& out,
    const torch::Tensor& route_records,
    int64_t nblocks);

void tma_ibgda_combine_bf16_ibgda_launcher(
    const torch::Tensor& expert_out,
    const torch::Tensor& out,
    const torch::Tensor& src_ranks,
    const torch::Tensor& src_rows,
    const std::optional<torch::Tensor>& probs,
    int64_t nblocks);

void tma_ibgda_combine_bf16_ibgda_records_launcher(
    const torch::Tensor& expert_out,
    const torch::Tensor& out,
    const torch::Tensor& route_records,
    int64_t top_k,
    int64_t nblocks);

void tma_ibgda_route_dot_bf16_peer_launcher(
    const torch::Tensor& expert_out,
    const torch::Tensor& grad_out,
    const torch::Tensor& src_ranks,
    const torch::Tensor& src_rows,
    const torch::Tensor& peer_expert_out_ptrs,
    const torch::Tensor& out);

void tma_ibgda_route_dot_bf16_ibgda_launcher(
    const torch::Tensor& expert_out,
    const torch::Tensor& grad_out,
    const torch::Tensor& src_ranks,
    const torch::Tensor& src_rows,
    const torch::Tensor& out);

void tma_ibgda_route_dot_bf16_ibgda_records_launcher(
    const torch::Tensor& expert_out,
    const torch::Tensor& grad_out,
    const torch::Tensor& route_records,
    int64_t top_k,
    const torch::Tensor& out);

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
    int64_t nblocks);

void tma_ibgda_route_records_with_probs_launcher(
    const torch::Tensor& route_records,
    const torch::Tensor& probs,
    const torch::Tensor& out_records);

std::vector<uint8_t> tma_ibgda_get_unique_id();

void tma_ibgda_init(
    const std::vector<std::vector<uint8_t>>& unique_ids,
    int64_t rank,
    int64_t world_size,
    int64_t device_idx);

torch::Tensor tma_ibgda_empty(
    const std::vector<int64_t>& sizes,
    c10::ScalarType dtype,
    c10::Device device);

void tma_ibgda_barrier_all_on_stream(c10::Device device);

void tma_ibgda_signal_all_and_wait(
    const torch::Tensor& signals,
    int64_t generation,
    int64_t world_size);

namespace {

std::optional<torch::Tensor> normalize_optional_tensor(
    const c10::optional<torch::Tensor>& tensor) {
  if (tensor.has_value() && tensor->defined()) {
    return std::optional<torch::Tensor>(tensor.value());
  }
  return std::nullopt;
}

void check_route_maps(
    const torch::Tensor& ranks,
    const torch::Tensor& rows,
    int64_t num_rows) {
  TORCH_CHECK(ranks.is_cuda(), "route ranks must be CUDA");
  TORCH_CHECK(rows.is_cuda(), "route rows must be CUDA");
  TORCH_CHECK(ranks.scalar_type() == torch::kInt64, "route ranks must be int64");
  TORCH_CHECK(rows.scalar_type() == torch::kInt64, "route rows must be int64");
  TORCH_CHECK(ranks.dim() == 2, "route ranks must be rank-2 [N, K]");
  TORCH_CHECK(rows.dim() == 2, "route rows must be rank-2 [N, K]");
  TORCH_CHECK(ranks.sizes() == rows.sizes(), "route ranks/rows shape mismatch");
  TORCH_CHECK(ranks.size(0) == num_rows, "route maps first dim must match rows");
  TORCH_CHECK(ranks.is_contiguous(), "route ranks must be contiguous");
  TORCH_CHECK(rows.is_contiguous(), "route rows must be contiguous");
}

void check_probs(
    const std::optional<torch::Tensor>& probs,
    const torch::Tensor& route_ranks) {
  if (!probs.has_value()) {
    return;
  }
  TORCH_CHECK(probs->is_cuda(), "probs must be CUDA");
  TORCH_CHECK(probs->scalar_type() == torch::kFloat32, "probs must be float32");
  TORCH_CHECK(probs->sizes() == route_ranks.sizes(), "probs shape mismatch");
  TORCH_CHECK(probs->is_contiguous(), "probs must be contiguous");
}

void check_bf16_matrix(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16, name, " must be bf16");
  TORCH_CHECK(tensor.dim() == 2, name, " must be rank-2 [rows, hidden]");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_peer_ptrs(const torch::Tensor& peer_ptrs, const char* name) {
  TORCH_CHECK(peer_ptrs.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(peer_ptrs.scalar_type() == torch::kInt64, name, " must be int64");
  TORCH_CHECK(peer_ptrs.dim() == 1, name, " must be rank-1 [world_size]");
  TORCH_CHECK(peer_ptrs.is_contiguous(), name, " must be contiguous");
}

void check_route_records(const torch::Tensor& route_records, const char* name) {
  TORCH_CHECK(route_records.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(route_records.scalar_type() == torch::kInt32, name, " must be int32");
  TORCH_CHECK(route_records.dim() == 2, name, " must be rank-2 [num_routes, 8]");
  TORCH_CHECK(route_records.size(1) == 8, name, " second dim must be 8 int32 words");
  TORCH_CHECK(route_records.is_contiguous(), name, " must be contiguous");
}

void check_preprocess_outputs(
    const torch::Tensor& dst_ranks,
    const torch::Tensor& route_records,
    const torch::Tensor& routes_per_rank,
    const torch::Tensor& rank_offsets,
    const torch::Tensor& overflow_by_rank,
    const torch::Tensor& route_ordinals,
    const torch::Tensor& errors,
    int64_t ep_world_size) {
  check_route_records(route_records, "route_records");
  TORCH_CHECK(route_records.size(0) == dst_ranks.numel(), "route_records first dim must match route map numel");
  TORCH_CHECK(route_records.device() == dst_ranks.device(), "route_records device mismatch");
  TORCH_CHECK(routes_per_rank.is_cuda(), "routes_per_rank must be CUDA");
  TORCH_CHECK(routes_per_rank.scalar_type() == torch::kInt64, "routes_per_rank must be int64");
  TORCH_CHECK(
      routes_per_rank.dim() == 1 && routes_per_rank.size(0) == ep_world_size,
      "routes_per_rank shape mismatch");
  TORCH_CHECK(routes_per_rank.device() == dst_ranks.device(), "routes_per_rank device mismatch");
  TORCH_CHECK(routes_per_rank.is_contiguous(), "routes_per_rank must be contiguous");
  TORCH_CHECK(rank_offsets.is_cuda(), "rank_offsets must be CUDA");
  TORCH_CHECK(rank_offsets.scalar_type() == torch::kInt64, "rank_offsets must be int64");
  TORCH_CHECK(
      rank_offsets.dim() == 1 && rank_offsets.size(0) == ep_world_size + 1,
      "rank_offsets shape mismatch");
  TORCH_CHECK(rank_offsets.device() == dst_ranks.device(), "rank_offsets device mismatch");
  TORCH_CHECK(rank_offsets.is_contiguous(), "rank_offsets must be contiguous");
  TORCH_CHECK(overflow_by_rank.is_cuda(), "overflow_by_rank must be CUDA");
  TORCH_CHECK(overflow_by_rank.scalar_type() == torch::kBool, "overflow_by_rank must be bool");
  TORCH_CHECK(
      overflow_by_rank.dim() == 1 && overflow_by_rank.size(0) == ep_world_size,
      "overflow_by_rank shape mismatch");
  TORCH_CHECK(overflow_by_rank.device() == dst_ranks.device(), "overflow_by_rank device mismatch");
  TORCH_CHECK(overflow_by_rank.is_contiguous(), "overflow_by_rank must be contiguous");
  TORCH_CHECK(route_ordinals.is_cuda(), "route_ordinals must be CUDA");
  TORCH_CHECK(route_ordinals.scalar_type() == torch::kInt64, "route_ordinals must be int64");
  TORCH_CHECK(route_ordinals.sizes() == dst_ranks.sizes(), "route_ordinals shape mismatch");
  TORCH_CHECK(route_ordinals.device() == dst_ranks.device(), "route_ordinals device mismatch");
  TORCH_CHECK(route_ordinals.is_contiguous(), "route_ordinals must be contiguous");
  TORCH_CHECK(errors.is_cuda(), "errors must be CUDA");
  TORCH_CHECK(errors.scalar_type() == torch::kInt32, "errors must be int32");
  TORCH_CHECK(errors.dim() == 1 && errors.size(0) == 3, "errors shape mismatch");
  TORCH_CHECK(errors.device() == dst_ranks.device(), "errors device mismatch");
  TORCH_CHECK(errors.is_contiguous(), "errors must be contiguous");
}

}  // namespace

void enable_peer_access_for_all_visible_devices() {
  int current_device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&current_device));
  int device_count = 0;
  C10_CUDA_CHECK(cudaGetDeviceCount(&device_count));
  for (int peer = 0; peer < device_count; ++peer) {
    if (peer == current_device) {
      continue;
    }
    int can_access = 0;
    C10_CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, current_device, peer));
    if (!can_access) {
      continue;
    }
    cudaError_t status = cudaDeviceEnablePeerAccess(peer, 0);
    if (status == cudaErrorPeerAccessAlreadyEnabled) {
      cudaGetLastError();
      continue;
    }
    C10_CUDA_CHECK(status);
  }
}

void preprocess_routes_into(
    torch::Tensor dst_ranks,
    torch::Tensor dst_rows,
    torch::Tensor route_records,
    torch::Tensor routes_per_rank,
    torch::Tensor rank_offsets,
    torch::Tensor overflow_by_rank,
    torch::Tensor route_ordinals,
    torch::Tensor errors,
    int64_t ep_world_size,
    int64_t rank_capacity,
    int64_t static_route_budget,
    int64_t nblocks,
    const c10::optional<torch::Tensor>& probs) {
  check_route_maps(dst_ranks, dst_rows, dst_ranks.size(0));
  auto probs_norm = normalize_optional_tensor(probs);
  check_probs(probs_norm, dst_ranks);
  TORCH_CHECK(ep_world_size > 0, "ep_world_size must be positive");
  TORCH_CHECK(rank_capacity > 0, "rank_capacity must be positive");
  TORCH_CHECK(
      static_route_budget == -1 || static_route_budget > 0,
      "static_route_budget must be -1 or positive");
  TORCH_CHECK(
      ep_world_size <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
      "ep_world_size exceeds int32 metadata range");
  TORCH_CHECK(
      rank_capacity <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
      "rank_capacity exceeds int32 metadata range");
  TORCH_CHECK(
      static_route_budget <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
      "static_route_budget exceeds int32 metadata range");
  check_preprocess_outputs(
      dst_ranks,
      route_records,
      routes_per_rank,
      rank_offsets,
      overflow_by_rank,
      route_ordinals,
      errors,
      ep_world_size);

  c10::cuda::CUDAGuard guard(dst_ranks.device());
  tma_ibgda_preprocess_routes_launcher(
      dst_ranks,
      dst_rows,
      probs_norm,
      route_records,
      routes_per_rank,
      rank_offsets,
      overflow_by_rank,
      route_ordinals,
      errors,
      ep_world_size,
      rank_capacity,
      static_route_budget,
      nblocks);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
preprocess_routes(
    torch::Tensor dst_ranks,
    torch::Tensor dst_rows,
    int64_t ep_world_size,
    int64_t rank_capacity,
    int64_t static_route_budget,
    int64_t nblocks,
    const c10::optional<torch::Tensor>& probs) {
  check_route_maps(dst_ranks, dst_rows, dst_ranks.size(0));
  TORCH_CHECK(ep_world_size > 0, "ep_world_size must be positive");
  TORCH_CHECK(rank_capacity > 0, "rank_capacity must be positive");

  c10::cuda::CUDAGuard guard(dst_ranks.device());
  auto int32_options = dst_ranks.options().dtype(torch::kInt32);
  auto int64_options = dst_ranks.options().dtype(torch::kInt64);
  auto bool_options = dst_ranks.options().dtype(torch::kBool);
  auto route_records = torch::empty({dst_ranks.numel(), 8}, int32_options);
  auto routes_per_rank = torch::empty({ep_world_size}, int64_options);
  auto rank_offsets = torch::empty({ep_world_size + 1}, int64_options);
  auto overflow_by_rank = torch::empty({ep_world_size}, bool_options);
  auto route_ordinals = torch::empty(dst_ranks.sizes(), int64_options);
  auto errors = torch::empty({3}, int32_options);
  preprocess_routes_into(
      dst_ranks,
      dst_rows,
      route_records,
      routes_per_rank,
      rank_offsets,
      overflow_by_rank,
      route_ordinals,
      errors,
      ep_world_size,
      rank_capacity,
      static_route_budget,
      nblocks,
      probs);
  return std::make_tuple(
      route_records,
      routes_per_rank,
      rank_offsets,
      overflow_by_rank,
      route_ordinals,
      errors);
}

torch::Tensor route_records_with_probs(
    torch::Tensor route_records,
    torch::Tensor probs) {
  check_route_records(route_records, "route_records");
  TORCH_CHECK(probs.is_cuda(), "probs must be CUDA");
  TORCH_CHECK(probs.scalar_type() == torch::kFloat32, "probs must be float32");
  TORCH_CHECK(probs.dim() == 2, "probs must be rank-2 [num_rows, top_k]");
  TORCH_CHECK(probs.numel() == route_records.size(0), "probs numel must match route_records rows");
  TORCH_CHECK(probs.device() == route_records.device(), "probs and route_records must be on the same device");
  TORCH_CHECK(probs.is_contiguous(), "probs must be contiguous");
  c10::cuda::CUDAGuard guard(route_records.device());
  auto out_records = torch::empty_like(route_records);
  tma_ibgda_route_records_with_probs_launcher(route_records, probs, out_records);
  return out_records;
}

void dispatch_bf16_peer(
    torch::Tensor input,
    torch::Tensor out,
    torch::Tensor dst_ranks,
    torch::Tensor dst_rows,
    torch::Tensor peer_out_ptrs,
    const c10::optional<torch::Tensor>& probs,
    int64_t nblocks) {
  check_bf16_matrix(input, "input");
  check_bf16_matrix(out, "out");
  TORCH_CHECK(input.size(1) == out.size(1), "input/out hidden dim mismatch");
  check_route_maps(dst_ranks, dst_rows, input.size(0));
  check_peer_ptrs(peer_out_ptrs, "peer_out_ptrs");
  auto probs_norm = normalize_optional_tensor(probs);
  check_probs(probs_norm, dst_ranks);
  c10::cuda::CUDAGuard guard(input.device());
  tma_ibgda_dispatch_bf16_peer_launcher(
      input,
      out,
      dst_ranks,
      dst_rows,
      peer_out_ptrs,
      probs_norm,
      nblocks);
}

void combine_bf16_peer(
    torch::Tensor expert_out,
    torch::Tensor out,
    torch::Tensor src_ranks,
    torch::Tensor src_rows,
    torch::Tensor peer_expert_out_ptrs,
    const c10::optional<torch::Tensor>& probs) {
  check_bf16_matrix(expert_out, "expert_out");
  check_bf16_matrix(out, "out");
  TORCH_CHECK(expert_out.size(1) == out.size(1), "expert_out/out hidden dim mismatch");
  check_route_maps(src_ranks, src_rows, out.size(0));
  check_peer_ptrs(peer_expert_out_ptrs, "peer_expert_out_ptrs");
  auto probs_norm = normalize_optional_tensor(probs);
  check_probs(probs_norm, src_ranks);
  c10::cuda::CUDAGuard guard(out.device());
  tma_ibgda_combine_bf16_peer_launcher(
      expert_out,
      out,
      src_ranks,
      src_rows,
      peer_expert_out_ptrs,
      probs_norm);
}

void dispatch_bf16_ibgda(
    torch::Tensor input,
    torch::Tensor out,
    torch::Tensor dst_ranks,
    torch::Tensor dst_rows,
    const c10::optional<torch::Tensor>& probs,
    int64_t nblocks) {
  check_bf16_matrix(input, "input");
  check_bf16_matrix(out, "out");
  TORCH_CHECK(input.size(1) == out.size(1), "input/out hidden dim mismatch");
  check_route_maps(dst_ranks, dst_rows, input.size(0));
  auto probs_norm = normalize_optional_tensor(probs);
  check_probs(probs_norm, dst_ranks);
  c10::cuda::CUDAGuard guard(input.device());
  tma_ibgda_dispatch_bf16_ibgda_launcher(
      input,
      out,
      dst_ranks,
      dst_rows,
      probs_norm,
      nblocks);
}

void dispatch_bf16_ibgda_records(
    torch::Tensor input,
    torch::Tensor out,
    torch::Tensor route_records,
    int64_t nblocks) {
  check_bf16_matrix(input, "input");
  check_bf16_matrix(out, "out");
  TORCH_CHECK(input.size(1) == out.size(1), "input/out hidden dim mismatch");
  check_route_records(route_records, "route_records");
  c10::cuda::CUDAGuard guard(input.device());
  tma_ibgda_dispatch_bf16_ibgda_records_launcher(
      input,
      out,
      route_records,
      nblocks);
}

void dispatch_bf16_ibgda_records_tma(
    torch::Tensor input,
    torch::Tensor out,
    torch::Tensor route_records,
    int64_t nblocks) {
  check_bf16_matrix(input, "input");
  check_bf16_matrix(out, "out");
  TORCH_CHECK(input.size(1) == out.size(1), "input/out hidden dim mismatch");
  check_route_records(route_records, "route_records");
  c10::cuda::CUDAGuard guard(input.device());
  tma_ibgda_dispatch_bf16_ibgda_records_tma_launcher(
      input,
      out,
      route_records,
      nblocks);
}

void route_dot_bf16_peer(
    torch::Tensor expert_out,
    torch::Tensor grad_out,
    torch::Tensor src_ranks,
    torch::Tensor src_rows,
    torch::Tensor peer_expert_out_ptrs,
    torch::Tensor out) {
  check_bf16_matrix(expert_out, "expert_out");
  check_bf16_matrix(grad_out, "grad_out");
  TORCH_CHECK(expert_out.size(1) == grad_out.size(1), "expert_out/grad_out hidden dim mismatch");
  check_route_maps(src_ranks, src_rows, grad_out.size(0));
  check_peer_ptrs(peer_expert_out_ptrs, "peer_expert_out_ptrs");
  TORCH_CHECK(out.is_cuda(), "out must be CUDA");
  TORCH_CHECK(out.scalar_type() == torch::kFloat32, "out must be float32");
  TORCH_CHECK(out.sizes() == src_ranks.sizes(), "out shape must match route maps");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  c10::cuda::CUDAGuard guard(grad_out.device());
  tma_ibgda_route_dot_bf16_peer_launcher(
      expert_out,
      grad_out,
      src_ranks,
      src_rows,
      peer_expert_out_ptrs,
      out);
}

void combine_bf16_ibgda(
    torch::Tensor expert_out,
    torch::Tensor out,
    torch::Tensor src_ranks,
    torch::Tensor src_rows,
    const c10::optional<torch::Tensor>& probs,
    int64_t nblocks) {
  check_bf16_matrix(expert_out, "expert_out");
  check_bf16_matrix(out, "out");
  TORCH_CHECK(expert_out.size(1) == out.size(1), "expert_out/out hidden dim mismatch");
  check_route_maps(src_ranks, src_rows, out.size(0));
  auto probs_norm = normalize_optional_tensor(probs);
  check_probs(probs_norm, src_ranks);
  c10::cuda::CUDAGuard guard(out.device());
  tma_ibgda_combine_bf16_ibgda_launcher(
      expert_out,
      out,
      src_ranks,
      src_rows,
      probs_norm,
      nblocks);
}

void combine_bf16_ibgda_records(
    torch::Tensor expert_out,
    torch::Tensor out,
    torch::Tensor route_records,
    int64_t top_k,
    int64_t nblocks) {
  check_bf16_matrix(expert_out, "expert_out");
  check_bf16_matrix(out, "out");
  TORCH_CHECK(expert_out.size(1) == out.size(1), "expert_out/out hidden dim mismatch");
  TORCH_CHECK(top_k > 0, "top_k must be positive");
  check_route_records(route_records, "route_records");
  TORCH_CHECK(
      route_records.size(0) == out.size(0) * top_k,
      "route_records first dim must equal out rows * top_k");
  c10::cuda::CUDAGuard guard(out.device());
  tma_ibgda_combine_bf16_ibgda_records_launcher(
      expert_out,
      out,
      route_records,
      top_k,
      nblocks);
}

void route_dot_bf16_ibgda(
    torch::Tensor expert_out,
    torch::Tensor grad_out,
    torch::Tensor src_ranks,
    torch::Tensor src_rows,
    torch::Tensor out) {
  check_bf16_matrix(expert_out, "expert_out");
  check_bf16_matrix(grad_out, "grad_out");
  TORCH_CHECK(expert_out.size(1) == grad_out.size(1), "expert_out/grad_out hidden dim mismatch");
  check_route_maps(src_ranks, src_rows, grad_out.size(0));
  TORCH_CHECK(out.is_cuda(), "out must be CUDA");
  TORCH_CHECK(out.scalar_type() == torch::kFloat32, "out must be float32");
  TORCH_CHECK(out.sizes() == src_ranks.sizes(), "out shape must match route maps");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  c10::cuda::CUDAGuard guard(grad_out.device());
  tma_ibgda_route_dot_bf16_ibgda_launcher(
      expert_out,
      grad_out,
      src_ranks,
      src_rows,
      out);
}

void route_dot_bf16_ibgda_records(
    torch::Tensor expert_out,
    torch::Tensor grad_out,
    torch::Tensor route_records,
    int64_t top_k,
    torch::Tensor out) {
  check_bf16_matrix(expert_out, "expert_out");
  check_bf16_matrix(grad_out, "grad_out");
  TORCH_CHECK(top_k > 0, "top_k must be positive");
  TORCH_CHECK(expert_out.size(1) == grad_out.size(1), "expert_out/grad_out hidden dim mismatch");
  check_route_records(route_records, "route_records");
  TORCH_CHECK(
      route_records.size(0) == grad_out.size(0) * top_k,
      "route_records first dim must equal grad_out rows * top_k");
  TORCH_CHECK(out.is_cuda(), "out must be CUDA");
  TORCH_CHECK(out.scalar_type() == torch::kFloat32, "out must be float32");
  TORCH_CHECK(out.dim() == 2, "out must be rank-2 [num_rows, top_k]");
  TORCH_CHECK(out.size(0) == grad_out.size(0), "out first dim must match grad_out rows");
  TORCH_CHECK(out.size(1) == top_k, "out second dim must match top_k");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  c10::cuda::CUDAGuard guard(grad_out.device());
  tma_ibgda_route_dot_bf16_ibgda_records_launcher(
      expert_out,
      grad_out,
      route_records,
      top_k,
      out);
}

py::dict extension_contract() {
  py::dict contract;
  contract["extension_module"] = "_tma_ibgda_ep_ext_gpu";
  contract["route_record_bytes"] =
      py::int_(sizeof(olmo::tma_ibgda_ep::RouteRecord));
  contract["route_record_words"] = py::int_(
      sizeof(olmo::tma_ibgda_ep::RouteRecord) / sizeof(int32_t));
  contract["route_flag_valid"] =
      py::int_(static_cast<int>(olmo::tma_ibgda_ep::ROUTE_FLAG_VALID));
  contract["workspace_alignment"] =
      py::int_(olmo::tma_ibgda_ep::kWorkspaceAlignment);
  contract["doorbell_bytes"] = py::int_(olmo::tma_ibgda_ep::kDoorbellBytes);
  contract["completion_bytes"] =
      py::int_(olmo::tma_ibgda_ep::kCompletionBytes);
  contract["peer_window_layout_bytes"] =
      py::int_(sizeof(olmo::tma_ibgda_ep::PeerWindowLayout));
  contract["kernel_launch_config_bytes"] =
      py::int_(sizeof(olmo::tma_ibgda_ep::KernelLaunchConfig));
  contract["bf16_only"] = py::bool_(true);
  contract["has_gpu_route_preprocess"] = py::bool_(true);
  contract["has_ibgda_dispatch"] = py::bool_(true);
  contract["has_tma_load_dispatch"] = py::bool_(true);
  contract["has_ibgda_combine"] = py::bool_(true);
  contract["has_route_dot_backward"] = py::bool_(true);
  contract["has_peer_window_layout_planner"] = py::bool_(true);
  return contract;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "extension_contract",
      &extension_contract,
      "Return the OLMo TMA/IBGDA extension ABI/capability contract");
  m.def(
      "plan_peer_window_layout",
      &plan_peer_window_layout,
      "Plan the OLMo TMA/IBGDA peer-window layout using the CUDA-side ABI");
  m.def(
      "get_unique_id",
      &tma_ibgda_get_unique_id,
      "Create an NVSHMEM unique ID for OLMo TMA/IBGDA EP");
  m.def(
      "init",
      &tma_ibgda_init,
      "Initialize OLMo TMA/IBGDA EP NVSHMEM state",
      py::arg("unique_ids"),
      py::arg("rank"),
      py::arg("world_size"),
      py::arg("device_idx"));
  m.def(
      "empty",
      &tma_ibgda_empty,
      "Allocate an OLMo TMA/IBGDA EP NVSHMEM symmetric tensor",
      py::arg("sizes"),
      py::arg("dtype"),
      py::arg("device"));
  m.def(
      "barrier_all_on_stream",
      &tma_ibgda_barrier_all_on_stream,
      "Enqueue an OLMo TMA/IBGDA NVSHMEM TEAM_WORLD barrier on the current CUDA stream",
      py::arg("device"));
  m.def(
      "signal_all_and_wait",
      &tma_ibgda_signal_all_and_wait,
      "Signal every OLMo TMA/IBGDA peer and wait for all peer signals on the current CUDA stream",
      py::arg("signals"),
      py::arg("generation"),
      py::arg("world_size"));
  m.def(
      "enable_peer_access_for_all_visible_devices",
      &enable_peer_access_for_all_visible_devices,
      "Enable CUDA peer access from the current device to visible peer devices");
  m.def(
      "preprocess_routes",
      &preprocess_routes,
      "Build OLMo TMA/IBGDA route records and per-rank metadata on GPU",
      py::arg("dst_ranks"),
      py::arg("dst_rows"),
      py::arg("ep_world_size"),
      py::arg("rank_capacity"),
      py::arg("static_route_budget") = -1,
      py::arg("nblocks") = 0,
      py::arg("probs") = py::none());
  m.def(
      "preprocess_routes_into",
      &preprocess_routes_into,
      "Build OLMo TMA/IBGDA route metadata into caller-provided tensors",
      py::arg("dst_ranks"),
      py::arg("dst_rows"),
      py::arg("route_records"),
      py::arg("routes_per_rank"),
      py::arg("rank_offsets"),
      py::arg("overflow_by_rank"),
      py::arg("route_ordinals"),
      py::arg("errors"),
      py::arg("ep_world_size"),
      py::arg("rank_capacity"),
      py::arg("static_route_budget") = -1,
      py::arg("nblocks") = 0,
      py::arg("probs") = py::none());
  m.def(
      "route_records_with_probs",
      &route_records_with_probs,
      "Copy packed OLMo TMA/IBGDA route records and patch the float32 probability field",
      py::arg("route_records"),
      py::arg("probs"));
  m.def(
      "dispatch_bf16_peer",
      &dispatch_bf16_peer,
      "Rowwise BF16 dispatch through peer-visible pointer table",
      py::arg("input"),
      py::arg("out"),
      py::arg("dst_ranks"),
      py::arg("dst_rows"),
      py::arg("peer_out_ptrs"),
      py::arg("probs") = py::none(),
      py::arg("nblocks") = 0);
  m.def(
      "combine_bf16_peer",
      &combine_bf16_peer,
      "Rowwise BF16 combine through peer-visible pointer table",
      py::arg("expert_out"),
      py::arg("out"),
      py::arg("src_ranks"),
      py::arg("src_rows"),
      py::arg("peer_expert_out_ptrs"),
      py::arg("probs") = py::none());
  m.def(
      "dispatch_bf16_ibgda",
      &dispatch_bf16_ibgda,
      "Rowwise BF16 dispatch using NVSHMEM/IBGDA PUTs",
      py::arg("input"),
      py::arg("out"),
      py::arg("dst_ranks"),
      py::arg("dst_rows"),
      py::arg("probs") = py::none(),
      py::arg("nblocks") = 0);
  m.def(
      "dispatch_bf16_ibgda_records",
      &dispatch_bf16_ibgda_records,
      "Rowwise BF16 dispatch using packed TMA/IBGDA route records",
      py::arg("input"),
      py::arg("out"),
      py::arg("route_records"),
      py::arg("nblocks") = 0);
  m.def(
      "dispatch_bf16_ibgda_records_tma",
      &dispatch_bf16_ibgda_records_tma,
      "Rowwise BF16 dispatch using packed route records, TMA global-to-shared loads, and NVSHMEM/IBGDA PUTs",
      py::arg("input"),
      py::arg("out"),
      py::arg("route_records"),
      py::arg("nblocks") = 0);
  m.def(
      "combine_bf16_ibgda",
      &combine_bf16_ibgda,
      "Rowwise BF16 combine using NVSHMEM/IBGDA GETs",
      py::arg("expert_out"),
      py::arg("out"),
      py::arg("src_ranks"),
      py::arg("src_rows"),
      py::arg("probs") = py::none(),
      py::arg("nblocks") = 0);
  m.def(
      "combine_bf16_ibgda_records",
      &combine_bf16_ibgda_records,
      "Rowwise BF16 combine using packed TMA/IBGDA route records",
      py::arg("expert_out"),
      py::arg("out"),
      py::arg("route_records"),
      py::arg("top_k"),
      py::arg("nblocks") = 0);
  m.def(
      "route_dot_bf16_peer",
      &route_dot_bf16_peer,
      "Per-route BF16 peer read dot product for weighted combine backward",
      py::arg("expert_out"),
      py::arg("grad_out"),
      py::arg("src_ranks"),
      py::arg("src_rows"),
      py::arg("peer_expert_out_ptrs"),
      py::arg("out"));
  m.def(
      "route_dot_bf16_ibgda",
      &route_dot_bf16_ibgda,
      "Per-route BF16 NVSHMEM/IBGDA GET dot product for weighted combine backward",
      py::arg("expert_out"),
      py::arg("grad_out"),
      py::arg("src_ranks"),
      py::arg("src_rows"),
      py::arg("out"));
  m.def(
      "route_dot_bf16_ibgda_records",
      &route_dot_bf16_ibgda_records,
      "Per-route BF16 NVSHMEM/IBGDA GET dot product using packed route records",
      py::arg("expert_out"),
      py::arg("grad_out"),
      py::arg("route_records"),
      py::arg("top_k"),
      py::arg("out"));
}

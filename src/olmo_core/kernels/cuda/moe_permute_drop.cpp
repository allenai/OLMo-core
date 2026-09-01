#include <torch/extension.h>

namespace py = pybind11;

std::tuple<torch::Tensor, torch::Tensor> moe_permute_drop_fwd_cuda_launcher(
    const torch::Tensor& input,
    const torch::Tensor& routing_map,
    const torch::Tensor& requested_offsets,
    const torch::Tensor& keep_offsets,
    const torch::Tensor& keep_splits,
    int64_t num_out_tokens,
    const torch::Tensor& sorted_indices_workspace,
    const torch::Tensor& row_id_workspace,
    const torch::Tensor& sorted_row_id_workspace,
    const torch::Tensor& temp_storage_workspace,
    const c10::optional<torch::Tensor>& out);

int64_t moe_permute_drop_temp_storage_bytes_cuda(int64_t num_items);

std::tuple<torch::Tensor, torch::Tensor> moe_permute_drop_fwd_cuda(
    const torch::Tensor& input,
    const torch::Tensor& routing_map,
    const torch::Tensor& requested_offsets,
    const torch::Tensor& keep_offsets,
    const torch::Tensor& keep_splits,
    int64_t num_out_tokens,
    const torch::Tensor& sorted_indices_workspace,
    const torch::Tensor& row_id_workspace,
    const torch::Tensor& sorted_row_id_workspace,
    const torch::Tensor& temp_storage_workspace,
    const c10::optional<torch::Tensor>& out) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(routing_map.is_cuda(), "routing_map must be CUDA");
  TORCH_CHECK(requested_offsets.is_cuda(), "requested_offsets must be CUDA");
  TORCH_CHECK(keep_offsets.is_cuda(), "keep_offsets must be CUDA");
  TORCH_CHECK(keep_splits.is_cuda(), "keep_splits must be CUDA");
  TORCH_CHECK(sorted_indices_workspace.is_cuda(), "sorted_indices_workspace must be CUDA");
  TORCH_CHECK(row_id_workspace.is_cuda(), "row_id_workspace must be CUDA");
  TORCH_CHECK(sorted_row_id_workspace.is_cuda(), "sorted_row_id_workspace must be CUDA");
  TORCH_CHECK(temp_storage_workspace.is_cuda(), "temp_storage_workspace must be CUDA");

  TORCH_CHECK(input.dim() == 2, "input must be rank-2");
  TORCH_CHECK(routing_map.dim() == 2, "routing_map must be rank-2 [num_tokens, topK]");
  TORCH_CHECK(routing_map.size(0) == input.size(0), "routing_map/input rows must match");
  TORCH_CHECK(routing_map.scalar_type() == torch::kInt32, "routing_map must be int32");

  TORCH_CHECK(requested_offsets.dim() == 1, "requested_offsets must be rank-1");
  TORCH_CHECK(keep_offsets.dim() == 1, "keep_offsets must be rank-1");
  TORCH_CHECK(keep_splits.dim() == 1, "keep_splits must be rank-1");
  TORCH_CHECK(
      requested_offsets.scalar_type() == torch::kInt64,
      "requested_offsets must be int64");
  TORCH_CHECK(keep_offsets.scalar_type() == torch::kInt64, "keep_offsets must be int64");
  TORCH_CHECK(keep_splits.scalar_type() == torch::kInt64, "keep_splits must be int64");
  TORCH_CHECK(
      requested_offsets.size(0) == keep_offsets.size(0) &&
          requested_offsets.size(0) == keep_splits.size(0),
      "requested_offsets/keep_offsets/keep_splits sizes must match");

  TORCH_CHECK(sorted_indices_workspace.dim() == 1, "sorted_indices_workspace must be rank-1");
  TORCH_CHECK(row_id_workspace.dim() == 1, "row_id_workspace must be rank-1");
  TORCH_CHECK(sorted_row_id_workspace.dim() == 1, "sorted_row_id_workspace must be rank-1");
  TORCH_CHECK(temp_storage_workspace.dim() == 1, "temp_storage_workspace must be rank-1");
  TORCH_CHECK(
      sorted_indices_workspace.scalar_type() == torch::kInt32,
      "sorted_indices_workspace must be int32");
  TORCH_CHECK(row_id_workspace.scalar_type() == torch::kInt32, "row_id_workspace must be int32");
  TORCH_CHECK(
      sorted_row_id_workspace.scalar_type() == torch::kInt32,
      "sorted_row_id_workspace must be int32");
  TORCH_CHECK(
      temp_storage_workspace.scalar_type() == torch::kUInt8,
      "temp_storage_workspace must be uint8");

  return moe_permute_drop_fwd_cuda_launcher(
      input,
      routing_map,
      requested_offsets,
      keep_offsets,
      keep_splits,
      num_out_tokens,
      sorted_indices_workspace,
      row_id_workspace,
      sorted_row_id_workspace,
      temp_storage_workspace,
      out);
}

int64_t moe_permute_drop_temp_storage_bytes(int64_t num_items) {
  TORCH_CHECK(num_items >= 0, "num_items must be non-negative");
  return moe_permute_drop_temp_storage_bytes_cuda(num_items);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "moe_permute_drop_fwd_cuda",
      &moe_permute_drop_fwd_cuda,
      "MoE one-shot permute+drop forward (CUDA)",
      py::arg("input"),
      py::arg("routing_map"),
      py::arg("requested_offsets"),
      py::arg("keep_offsets"),
      py::arg("keep_splits"),
      py::arg("num_out_tokens"),
      py::arg("sorted_indices_workspace"),
      py::arg("row_id_workspace"),
      py::arg("sorted_row_id_workspace"),
      py::arg("temp_storage_workspace"),
      py::arg("out") = py::none());

  m.def(
      "moe_permute_drop_temp_storage_bytes",
      &moe_permute_drop_temp_storage_bytes,
      "Temporary storage bytes required by MoE one-shot permute+drop radix sort");
}


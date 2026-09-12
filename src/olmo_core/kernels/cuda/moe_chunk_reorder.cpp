#include <torch/extension.h>

namespace py = pybind11;

std::tuple<torch::Tensor, torch::Tensor> chunk_permute_fwd_cuda_launcher(
    const torch::Tensor& input,
    const torch::Tensor& routing_map,
    int64_t num_out_tokens,
    const c10::optional<torch::Tensor>& out);

torch::Tensor chunk_permute_by_row_id_map_cuda_launcher(
    const torch::Tensor& input,
    const torch::Tensor& row_id_map,
    int64_t num_out_tokens,
    const c10::optional<torch::Tensor>& out);

torch::Tensor chunk_unpermute_fwd_cuda_launcher(
    const torch::Tensor& input,
    const torch::Tensor& row_id_map,
    int64_t num_tokens,
    const c10::optional<torch::Tensor>& out);

std::tuple<torch::Tensor, torch::Tensor> chunk_permute_fwd_cuda(
    const torch::Tensor& input,
    const torch::Tensor& routing_map,
    int64_t num_out_tokens,
    const c10::optional<torch::Tensor>& out) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(routing_map.is_cuda(), "routing_map must be CUDA");
  TORCH_CHECK(input.dim() == 2, "input must be rank-2");
  TORCH_CHECK(routing_map.dim() == 2, "routing_map must be rank-2 [num_tokens, topK]");
  TORCH_CHECK(routing_map.size(0) == input.size(0), "routing_map/input rows must match");
  TORCH_CHECK(routing_map.size(1) == 1, "only topK=1 is supported");
  TORCH_CHECK(routing_map.scalar_type() == torch::kInt32, "routing_map must be int32");

  return chunk_permute_fwd_cuda_launcher(input, routing_map, num_out_tokens, out);
}

torch::Tensor chunk_permute_by_row_id_map_cuda(
    const torch::Tensor& input,
    const torch::Tensor& row_id_map,
    int64_t num_out_tokens,
    const c10::optional<torch::Tensor>& out) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(row_id_map.is_cuda(), "row_id_map must be CUDA");
  TORCH_CHECK(input.dim() == 2, "input must be rank-2");
  TORCH_CHECK(row_id_map.dim() == 1, "row_id_map must be rank-1");
  TORCH_CHECK(row_id_map.size(0) == input.size(0), "row_id_map/input rows must match");
  TORCH_CHECK(row_id_map.scalar_type() == torch::kInt32, "row_id_map must be int32");

  return chunk_permute_by_row_id_map_cuda_launcher(input, row_id_map, num_out_tokens, out);
}

torch::Tensor chunk_unpermute_fwd_cuda(
    const torch::Tensor& input,
    const torch::Tensor& row_id_map,
    int64_t num_tokens,
    const c10::optional<torch::Tensor>& out) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(row_id_map.is_cuda(), "row_id_map must be CUDA");
  TORCH_CHECK(input.dim() == 2, "input must be rank-2");
  TORCH_CHECK(row_id_map.dim() == 1, "row_id_map must be rank-1");
  TORCH_CHECK(row_id_map.scalar_type() == torch::kInt32, "row_id_map must be int32");

  return chunk_unpermute_fwd_cuda_launcher(input, row_id_map, num_tokens, out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "chunk_permute_fwd_cuda",
      &chunk_permute_fwd_cuda,
      "Chunk permute forward (CUDA)",
      py::arg("input"),
      py::arg("routing_map"),
      py::arg("num_out_tokens"),
      py::arg("out") = py::none());

  m.def(
      "chunk_permute_by_row_id_map_cuda",
      &chunk_permute_by_row_id_map_cuda,
      "Chunk permute by row_id_map (CUDA)",
      py::arg("input"),
      py::arg("row_id_map"),
      py::arg("num_out_tokens"),
      py::arg("out") = py::none());

  m.def(
      "chunk_unpermute_fwd_cuda",
      &chunk_unpermute_fwd_cuda,
      "Chunk unpermute forward (CUDA)",
      py::arg("input"),
      py::arg("row_id_map"),
      py::arg("num_tokens"),
      py::arg("out") = py::none());
}

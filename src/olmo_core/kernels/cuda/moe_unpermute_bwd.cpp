#include <torch/extension.h>

namespace py = pybind11;

std::tuple<torch::Tensor, torch::Tensor> moe_unpermute_bwd_cuda_launcher(
    const torch::Tensor& grad_output,
    const torch::Tensor& input_fwd,
    const torch::Tensor& row_id_map,
    const torch::Tensor& probs,
    const c10::optional<torch::Tensor>& keep_mask,
    const c10::optional<torch::Tensor>& out);

std::tuple<torch::Tensor, torch::Tensor> moe_unpermute_bwd_cuda(
    const torch::Tensor& grad_output,
    const torch::Tensor& input_fwd,
    const torch::Tensor& row_id_map,
    const torch::Tensor& probs,
    const c10::optional<torch::Tensor>& keep_mask,
    const c10::optional<torch::Tensor>& out) {
  TORCH_CHECK(grad_output.is_cuda(), "grad_output must be CUDA");
  TORCH_CHECK(input_fwd.is_cuda(), "input_fwd must be CUDA");
  TORCH_CHECK(row_id_map.is_cuda(), "row_id_map must be CUDA");
  TORCH_CHECK(probs.is_cuda(), "probs must be CUDA");

  TORCH_CHECK(grad_output.dim() == 2, "grad_output must be rank-2 [num_tokens, hidden]");
  TORCH_CHECK(input_fwd.dim() == 2, "input_fwd must be rank-2 [input_rows, hidden]");
  TORCH_CHECK(row_id_map.dim() == 1, "row_id_map must be rank-1 [topK * num_tokens]");
  TORCH_CHECK(probs.dim() == 2, "probs must be rank-2 [num_tokens, topK]");

  TORCH_CHECK(grad_output.scalar_type() == input_fwd.scalar_type(), "dtype mismatch between grad_output and input_fwd");
  TORCH_CHECK(row_id_map.scalar_type() == torch::kInt32, "row_id_map must be int32");
  TORCH_CHECK(probs.scalar_type() == torch::kFloat32, "probs must be float32");

  TORCH_CHECK(grad_output.size(1) == input_fwd.size(1), "hidden-size mismatch between grad_output and input_fwd");
  TORCH_CHECK(grad_output.size(0) == probs.size(0), "num_tokens mismatch between grad_output and probs");

  const int64_t num_tokens = probs.size(0);
  const int64_t topk = probs.size(1);
  TORCH_CHECK(row_id_map.numel() == num_tokens * topk, "row_id_map size must equal num_tokens * topK");
  if (keep_mask.has_value()) {
    const auto& keep_mask_tensor = keep_mask.value();
    TORCH_CHECK(keep_mask_tensor.is_cuda(), "keep_mask must be CUDA");
    TORCH_CHECK(keep_mask_tensor.dim() == 1, "keep_mask must be rank-1 [input_rows]");
    TORCH_CHECK(keep_mask_tensor.scalar_type() == torch::kBool, "keep_mask must be bool");
    TORCH_CHECK(
        keep_mask_tensor.size(0) == input_fwd.size(0),
        "keep_mask length must equal input_fwd rows");
  }

  return moe_unpermute_bwd_cuda_launcher(
      grad_output, input_fwd, row_id_map, probs, keep_mask, out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "moe_unpermute_bwd_cuda",
      &moe_unpermute_bwd_cuda,
      "MoE unpermute backward (CUDA)",
      py::arg("grad_output"),
      py::arg("input_fwd"),
      py::arg("row_id_map"),
      py::arg("probs"),
      py::arg("keep_mask") = py::none(),
      py::arg("out") = py::none());
}

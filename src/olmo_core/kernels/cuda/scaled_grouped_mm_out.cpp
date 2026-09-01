#include <torch/extension.h>

#include <ATen/core/Tensor.h>

#include <optional>

namespace py = pybind11;

// Exported by libtorch_cuda when PyTorch is built with USE_FBGEMM_GENAI.
// We intentionally call this backend directly so we can write into `out`
// without creating a temporary tensor and copying.
namespace fbgemm_gpu {
at::Tensor mx8mx8bf16_grouped_mm(
    at::Tensor mat_a,
    at::Tensor mat_b,
    at::Tensor scale_a,
    at::Tensor scale_b,
    at::Tensor offs,
    std::optional<at::Tensor> out);
}  // namespace fbgemm_gpu

namespace {

std::optional<at::Tensor> normalize_optional_tensor(
    const c10::optional<at::Tensor>& tensor) {
  if (tensor.has_value() && tensor->defined()) {
    return std::optional<at::Tensor>(tensor.value());
  }
  return std::nullopt;
}

bool is_transposed_last2(const at::Tensor& t) {
  if (t.dim() < 2) {
    return false;
  }
  return t.stride(-2) == 1 && t.stride(-1) == t.size(-2);
}

}  // namespace

at::Tensor scaled_grouped_mm_v2_out_cuda(
    const at::Tensor& mat_a_q,
    const at::Tensor& mat_b_q,
    const at::Tensor& scale_a,
    const at::Tensor& scale_b,
    at::Tensor out,
    const c10::optional<at::Tensor>& offs_opt,
    bool use_fast_accum) {
  TORCH_CHECK(mat_a_q.is_cuda(), "mat_a_q must be CUDA");
  TORCH_CHECK(mat_b_q.is_cuda(), "mat_b_q must be CUDA");
  TORCH_CHECK(scale_a.is_cuda(), "scale_a must be CUDA");
  TORCH_CHECK(scale_b.is_cuda(), "scale_b must be CUDA");
  TORCH_CHECK(out.is_cuda(), "out must be CUDA");
  TORCH_CHECK(mat_a_q.scalar_type() == at::kFloat8_e4m3fn, "mat_a_q must be float8_e4m3fn");
  TORCH_CHECK(mat_b_q.scalar_type() == at::kFloat8_e4m3fn, "mat_b_q must be float8_e4m3fn");
  TORCH_CHECK(scale_a.scalar_type() == at::kFloat8_e8m0fnu, "scale_a must be float8_e8m0fnu");
  TORCH_CHECK(scale_b.scalar_type() == at::kFloat8_e8m0fnu, "scale_b must be float8_e8m0fnu");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16, "out must be bfloat16");
  TORCH_CHECK(mat_a_q.dim() == 2, "mat_a_q must be rank-2 [M, K]");
  TORCH_CHECK(mat_b_q.dim() == 3, "mat_b_q must be rank-3 [G, K, N]");
  TORCH_CHECK(is_transposed_last2(mat_b_q), "Expected mat_b_q to be transposed in trailing dims");
  TORCH_CHECK(
      mat_a_q.size(1) == mat_b_q.size(1),
      "mat_a_q K dimension must match mat_b_q K dimension, got ",
      mat_a_q.size(1),
      " vs ",
      mat_b_q.size(1));
  TORCH_CHECK(
      out.dim() == 2 && out.size(0) == mat_a_q.size(0) && out.size(1) == mat_b_q.size(2),
      "out shape must be [M, N] = [",
      mat_a_q.size(0),
      ", ",
      mat_b_q.size(2),
      "], got ",
      out.sizes());
  (void)use_fast_accum;

  auto offs = normalize_optional_tensor(offs_opt);
  TORCH_CHECK(offs.has_value(), "offs is required for MXFP8 grouped mm");
  TORCH_CHECK(offs->is_cuda(), "offs must be CUDA");
  TORCH_CHECK(offs->scalar_type() == at::kInt, "offs must be int32");
  TORCH_CHECK(offs->dim() == 1, "offs must be rank-1");

  auto result = fbgemm_gpu::mx8mx8bf16_grouped_mm(
      mat_a_q,
      mat_b_q,
      scale_a,
      scale_b,
      offs.value(),
      std::optional<at::Tensor>(out));
  TORCH_CHECK(result.defined(), "mx8mx8bf16_grouped_mm returned undefined tensor");
  TORCH_CHECK(
      result.data_ptr() == out.data_ptr(),
      "mx8mx8bf16_grouped_mm did not write into provided out buffer");
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "scaled_grouped_mm_v2_out_cuda",
      &scaled_grouped_mm_v2_out_cuda,
      "Scaled grouped_mm_v2 with explicit output tensor",
      py::arg("mat_a_q"),
      py::arg("mat_b_q"),
      py::arg("scale_a"),
      py::arg("scale_b"),
      py::arg("out"),
      py::arg("offs") = py::none(),
      py::arg("use_fast_accum") = true);
}

#include <torch/extension.h>

#include <ATen/core/Tensor.h>
#include <ATen/native/GroupedMMUtils.h>
#include <ATen/native/cuda/GroupMM.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace py = pybind11;

namespace {

std::optional<at::Tensor> normalize_optional_tensor(
    const c10::optional<at::Tensor>& tensor) {
  if (tensor.has_value() && tensor->defined()) {
    return std::optional<at::Tensor>(tensor.value());
  }
  return std::nullopt;
}

std::vector<int64_t> grouped_mm_expected_output_size(
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    const std::optional<at::Tensor>& offs) {
  const bool a_is_2d = mat_a.dim() == 2;
  const bool b_is_2d = mat_b.dim() == 2;

  if (a_is_2d) {
    if (b_is_2d) {
      TORCH_CHECK(offs.has_value(), "offs is required for 2D/2D grouped_mm");
      return {offs->size(0), mat_a.size(0), mat_b.size(1)};
    } else {
      TORCH_CHECK(offs.has_value(), "offs is required for 2D/3D grouped_mm");
      TORCH_CHECK(
          offs->size(0) == mat_b.size(0),
          "matrix batch sizes have to match");
      return {mat_a.size(0), mat_b.size(-1)};
    }
  } else {
    if (b_is_2d) {
      TORCH_CHECK(offs.has_value(), "offs is required for 3D/2D grouped_mm");
      TORCH_CHECK(
          offs->size(0) == mat_a.size(0),
          "matrix batch sizes have to match");
      return {mat_a.size(1), mat_b.size(1)};
    }

    TORCH_CHECK(
        mat_a.size(0) == mat_b.size(0), "batched dimension has to match");
    return {mat_a.size(0), mat_a.size(1), mat_b.size(-1)};
  }
}

void check_out_tensor_shape(
    const at::Tensor& out,
    const std::vector<int64_t>& expected_sizes) {
  TORCH_CHECK(
      out.dim() == static_cast<int64_t>(expected_sizes.size()),
      "out rank mismatch: expected ",
      expected_sizes.size(),
      "D but got ",
      out.dim(),
      "D");
  for (size_t i = 0; i < expected_sizes.size(); ++i) {
    TORCH_CHECK(
        out.size(static_cast<int64_t>(i)) == expected_sizes[i],
        "out shape mismatch at dim ",
        i,
        ": expected ",
        expected_sizes[i],
        " but got ",
        out.size(static_cast<int64_t>(i)));
  }
}

bool has_fast_path_output_layout(
    const at::Tensor& out,
    const std::vector<int64_t>& expected_sizes,
    const bool a_is_2d,
    const bool b_is_2d) {
  if (reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 != 0) {
    return false;
  }
  const int alignment = 16 / out.element_size();
  const auto last_dim = static_cast<int64_t>(expected_sizes.size() - 1);
  const int64_t size_padded =
      (expected_sizes[last_dim] + alignment - 1) / alignment * alignment;

  if (a_is_2d != b_is_2d) {
    return out.stride(0) == size_padded && out.stride(1) == 1;
  }

  return out.stride(0) == expected_sizes[1] * size_padded &&
      out.stride(1) == size_padded && out.stride(2) == 1;
}

bool should_use_fast_path(
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    const at::Tensor& out,
    const std::vector<int64_t>& expected_sizes,
    const bool a_is_2d,
    const bool b_is_2d) {
  if (mat_a.scalar_type() != at::kBFloat16 ||
      mat_b.scalar_type() != at::kBFloat16 ||
      out.scalar_type() != at::kBFloat16) {
    return false;
  }

#ifdef USE_ROCM
  return false;
#else
  return has_fast_path_output_layout(out, expected_sizes, a_is_2d, b_is_2d);
#endif
}

} // namespace

at::Tensor grouped_mm_out_cuda(
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    at::Tensor out,
    const c10::optional<at::Tensor>& offs_opt,
    const c10::optional<at::Tensor>& bias_opt) {
  TORCH_CHECK(mat_a.is_cuda(), "mat_a must be CUDA");
  TORCH_CHECK(mat_b.is_cuda(), "mat_b must be CUDA");
  TORCH_CHECK(out.is_cuda(), "out must be CUDA");
  TORCH_CHECK(
      mat_a.device() == mat_b.device(),
      "mat_a/mat_b device mismatch: ",
      mat_a.device(),
      " vs ",
      mat_b.device());
  TORCH_CHECK(
      out.device() == mat_a.device(),
      "out/mat_a device mismatch: ",
      out.device(),
      " vs ",
      mat_a.device());

  const auto offs = normalize_optional_tensor(offs_opt);
  const auto bias = normalize_optional_tensor(bias_opt);

  const std::optional<c10::ScalarType> out_dtype = out.scalar_type();
  at::native::_grouped_mm_validate_inputs(mat_a, mat_b, offs, bias, out_dtype);
  const auto resolved_out_dtype =
      at::native::_resolve_grouped_mm_out_dtype(mat_a, mat_b, out_dtype);
  TORCH_CHECK(
      out.scalar_type() == resolved_out_dtype,
      "out dtype must match grouped_mm resolved dtype. got out=",
      out.scalar_type(),
      ", expected=",
      resolved_out_dtype);

  const auto expected_sizes = grouped_mm_expected_output_size(mat_a, mat_b, offs);
  check_out_tensor_shape(out, expected_sizes);

  const bool a_is_2d = mat_a.dim() == 2;
  const bool b_is_2d = mat_b.dim() == 2;
  const bool use_fast_path = should_use_fast_path(
      mat_a, mat_b, out, expected_sizes, a_is_2d, b_is_2d);

  if (use_fast_path) {
    try {
      at::cuda::detail::bf16bf16_grouped_mm(mat_a, mat_b, offs, bias, out);
      return out;
    } catch (const c10::Error&) {
      // Fall back if the fast kernel is unavailable for the current build/device.
    }
  }

  at::native::_grouped_mm_fallback(mat_a, mat_b, offs, bias, out_dtype, out);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "grouped_mm_out_cuda",
      &grouped_mm_out_cuda,
      "GroupedMM with explicit output buffer (CUDA)",
      py::arg("mat_a"),
      py::arg("mat_b"),
      py::arg("out"),
      py::arg("offs") = py::none(),
      py::arg("bias") = py::none());
}

/*
 * OLMo BF16 MegaMoE tensor-map helpers.
 *
 * Host-side CUDA-driver descriptor creation for TMA. This replaces CUTE's
 * TmaDescriptor dependency with native CUtensorMap descriptors owned by OLMo.
 */
#pragma once

#include <ATen/core/TensorBody.h>
#include <torch/extension.h>

#include <cuda.h>

#include <array>
#include <cstdint>

namespace olmo::bf16_mega_moe {

struct Bf16TensorMap2D {
  CUtensorMap map;
};

inline const char* cuda_driver_error_string(CUresult status) {
  const char* message = nullptr;
  cuGetErrorString(status, &message);
  return message == nullptr ? "unknown CUDA driver error" : message;
}

inline CUtensorMapSwizzle tensor_map_swizzle_for_bytes(int64_t swizzle_bytes) {
  switch (swizzle_bytes) {
    case 0:
      return CU_TENSOR_MAP_SWIZZLE_NONE;
    case 32:
      return CU_TENSOR_MAP_SWIZZLE_32B;
    case 64:
      return CU_TENSOR_MAP_SWIZZLE_64B;
    case 128:
      return CU_TENSOR_MAP_SWIZZLE_128B;
    default:
      TORCH_CHECK(
          false,
          "unsupported tensor-map swizzle size ",
          swizzle_bytes,
          "; expected 0, 32, 64, or 128");
  }
}

inline Bf16TensorMap2D make_bf16_tma_2d_desc(
    const at::Tensor& tensor,
    int64_t box_cols,
    int64_t box_rows,
    int64_t swizzle_bytes = 128) {
  TORCH_CHECK(tensor.is_cuda(), "TMA tensor must be CUDA");
  TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, "TMA tensor must be BF16");
  TORCH_CHECK(tensor.dim() == 2, "TMA tensor must be rank-2 [rows, cols]");
  TORCH_CHECK(tensor.stride(1) == 1, "TMA tensor must have contiguous columns");
  TORCH_CHECK(box_cols > 0 && box_rows > 0, "TMA box dimensions must be positive");
  TORCH_CHECK(box_cols <= tensor.size(1), "TMA box cols exceed tensor cols");
  TORCH_CHECK(box_rows <= tensor.size(0), "TMA box rows exceed tensor rows");
  TORCH_CHECK(box_cols % 8 == 0, "BF16 TMA box cols must be a multiple of 8");
  if (swizzle_bytes != 0) {
    TORCH_CHECK(
        box_cols * static_cast<int64_t>(tensor.element_size()) == swizzle_bytes,
        "TMA swizzled box inner dimension must equal the swizzle atom size; got box_cols=",
        box_cols,
        ", elem_size=",
        tensor.element_size(),
        ", swizzle_bytes=",
        swizzle_bytes);
  }

  std::array<cuuint64_t, 2> global_dim{
      static_cast<cuuint64_t>(tensor.size(1)),
      static_cast<cuuint64_t>(tensor.size(0)),
  };
  std::array<cuuint64_t, 1> global_strides{
      static_cast<cuuint64_t>(tensor.stride(0) * tensor.element_size()),
  };
  std::array<cuuint32_t, 2> box_dim{
      static_cast<cuuint32_t>(box_cols),
      static_cast<cuuint32_t>(box_rows),
  };
  std::array<cuuint32_t, 2> element_strides{1, 1};

  Bf16TensorMap2D desc;
  const CUresult status = cuTensorMapEncodeTiled(
      &desc.map,
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      /*tensorRank=*/2,
      tensor.data_ptr(),
      global_dim.data(),
      global_strides.data(),
      box_dim.data(),
      element_strides.data(),
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      tensor_map_swizzle_for_bytes(swizzle_bytes),
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(
      status == CUDA_SUCCESS,
      "cuTensorMapEncodeTiled failed: ",
      cuda_driver_error_string(status));
  return desc;
}

}  // namespace olmo::bf16_mega_moe

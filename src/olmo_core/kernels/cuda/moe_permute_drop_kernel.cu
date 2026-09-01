#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cub/cub.cuh>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t, int kElementsPerAccess>
__global__ void moe_permute_drop_vec_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int* __restrict__ sorted_indices,
    const int* __restrict__ sorted_row_id,
    const int64_t* __restrict__ requested_offsets,
    const int64_t* __restrict__ keep_offsets,
    const int64_t* __restrict__ keep_splits,
    int* __restrict__ row_id_map,
    int num_tokens,
    int topk,
    int num_cols,
    int num_experts,
    int num_out_tokens,
    int expanded_rows) {
  const int sorted_pos = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  if (sorted_pos >= expanded_rows) {
    return;
  }

  const int expert = sorted_indices[sorted_pos];
  int dest_row = -1;
  int source_token = 0;
  int source_topk = 0;

  if (expert >= 0 && expert < num_experts) {
    const int64_t exp_offset = requested_offsets[expert];
    const int64_t keep_offset = keep_offsets[expert];
    const int64_t keep_count = keep_splits[expert];
    const int64_t rank_in_expert = static_cast<int64_t>(sorted_pos) - exp_offset;

    if (rank_in_expert >= 0 && rank_in_expert < keep_count) {
      const int64_t dst64 = keep_offset + rank_in_expert;
      if (dst64 >= 0 && dst64 < static_cast<int64_t>(num_out_tokens)) {
        dest_row = static_cast<int>(dst64);
      }
    }
  }

  const int source_row = sorted_row_id[sorted_pos];
  source_token = source_row / topk;
  source_topk = source_row - source_token * topk;

  if (tid == 0) {
    const int map_idx = source_topk * num_tokens + source_token;
    row_id_map[map_idx] = dest_row;
  }

  if (dest_row < 0) {
    return;
  }

  const scalar_t* source_ptr = input + static_cast<int64_t>(source_token) * num_cols;
  scalar_t* dest_ptr = output + static_cast<int64_t>(dest_row) * num_cols;

  for (int col = tid * kElementsPerAccess; col < num_cols;
       col += static_cast<int>(blockDim.x) * kElementsPerAccess) {
    float4 frag = reinterpret_cast<const float4*>(source_ptr + col)[0];
    reinterpret_cast<float4*>(dest_ptr + col)[0] = frag;
  }
}

template <typename scalar_t>
__global__ void moe_permute_drop_scalar_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int* __restrict__ sorted_indices,
    const int* __restrict__ sorted_row_id,
    const int64_t* __restrict__ requested_offsets,
    const int64_t* __restrict__ keep_offsets,
    const int64_t* __restrict__ keep_splits,
    int* __restrict__ row_id_map,
    int num_tokens,
    int topk,
    int num_cols,
    int num_experts,
    int num_out_tokens,
    int expanded_rows) {
  const int sorted_pos = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  if (sorted_pos >= expanded_rows) {
    return;
  }

  const int expert = sorted_indices[sorted_pos];
  int dest_row = -1;
  int source_token = 0;
  int source_topk = 0;

  if (expert >= 0 && expert < num_experts) {
    const int64_t exp_offset = requested_offsets[expert];
    const int64_t keep_offset = keep_offsets[expert];
    const int64_t keep_count = keep_splits[expert];
    const int64_t rank_in_expert = static_cast<int64_t>(sorted_pos) - exp_offset;

    if (rank_in_expert >= 0 && rank_in_expert < keep_count) {
      const int64_t dst64 = keep_offset + rank_in_expert;
      if (dst64 >= 0 && dst64 < static_cast<int64_t>(num_out_tokens)) {
        dest_row = static_cast<int>(dst64);
      }
    }
  }

  const int source_row = sorted_row_id[sorted_pos];
  source_token = source_row / topk;
  source_topk = source_row - source_token * topk;

  if (tid == 0) {
    const int map_idx = source_topk * num_tokens + source_token;
    row_id_map[map_idx] = dest_row;
  }

  if (dest_row < 0) {
    return;
  }

  const scalar_t* source_ptr = input + static_cast<int64_t>(source_token) * num_cols;
  scalar_t* dest_ptr = output + static_cast<int64_t>(dest_row) * num_cols;
  for (int col = tid; col < num_cols; col += static_cast<int>(blockDim.x)) {
    dest_ptr[col] = source_ptr[col];
  }
}

torch::Tensor get_or_create_output(
    const torch::Tensor& input,
    int64_t out_rows,
    const c10::optional<torch::Tensor>& out) {
  if (out.has_value()) {
    const auto& out_tensor = out.value();
    TORCH_CHECK(out_tensor.is_cuda(), "moe_permute_drop_fwd_cuda: out must be CUDA");
    TORCH_CHECK(out_tensor.dim() == 2, "moe_permute_drop_fwd_cuda: out must be rank-2");
    TORCH_CHECK(
        out_tensor.size(0) == out_rows && out_tensor.size(1) == input.size(1),
        "moe_permute_drop_fwd_cuda: out shape mismatch, expected [",
        out_rows,
        ", ",
        input.size(1),
        "] got ",
        out_tensor.sizes());
    TORCH_CHECK(
        out_tensor.scalar_type() == input.scalar_type(),
        "moe_permute_drop_fwd_cuda: out dtype mismatch");
    TORCH_CHECK(
        out_tensor.device() == input.device(),
        "moe_permute_drop_fwd_cuda: out device mismatch");
    TORCH_CHECK(out_tensor.is_contiguous(), "moe_permute_drop_fwd_cuda: out must be contiguous");
    return out_tensor;
  }

  return torch::empty(
      {out_rows, input.size(1)},
      torch::TensorOptions()
          .dtype(input.scalar_type())
          .device(input.device())
          .requires_grad(false));
}

template <typename scalar_t>
void launch_moe_permute_drop_scatter(
    const torch::Tensor& input,
    const torch::Tensor& output,
    const torch::Tensor& sorted_indices,
    const torch::Tensor& sorted_row_id,
    const torch::Tensor& requested_offsets,
    const torch::Tensor& keep_offsets,
    const torch::Tensor& keep_splits,
    const torch::Tensor& row_id_map,
    int num_tokens,
    int topk,
    int num_experts,
    int num_out_tokens,
    int expanded_rows,
    cudaStream_t stream) {
  const int num_cols = static_cast<int>(input.size(1));
  if (expanded_rows == 0 || num_cols == 0) {
    return;
  }

  constexpr int kElementsPerAccess = 16 / sizeof(scalar_t);
  if (num_cols % kElementsPerAccess == 0) {
    const int threads = std::max(1, std::min(num_cols / kElementsPerAccess, 256));
    const int blocks = expanded_rows;
    moe_permute_drop_vec_kernel<scalar_t, kElementsPerAccess><<<blocks, threads, 0, stream>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        sorted_indices.data_ptr<int>(),
        sorted_row_id.data_ptr<int>(),
        requested_offsets.data_ptr<int64_t>(),
        keep_offsets.data_ptr<int64_t>(),
        keep_splits.data_ptr<int64_t>(),
        row_id_map.data_ptr<int>(),
        num_tokens,
        topk,
        num_cols,
        num_experts,
        num_out_tokens,
        expanded_rows);
  } else {
    const int threads = std::max(1, std::min(num_cols, 256));
    const int blocks = expanded_rows;
    moe_permute_drop_scalar_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        sorted_indices.data_ptr<int>(),
        sorted_row_id.data_ptr<int>(),
        requested_offsets.data_ptr<int64_t>(),
        keep_offsets.data_ptr<int64_t>(),
        keep_splits.data_ptr<int64_t>(),
        row_id_map.data_ptr<int>(),
        num_tokens,
        topk,
        num_cols,
        num_experts,
        num_out_tokens,
        expanded_rows);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int64_t moe_permute_drop_temp_storage_bytes_cuda(int64_t num_items) {
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(
      nullptr,
      temp_storage_bytes,
      static_cast<int*>(nullptr),
      static_cast<int*>(nullptr),
      static_cast<int*>(nullptr),
      static_cast<int*>(nullptr),
      static_cast<int>(num_items));
  TORCH_CHECK(
      temp_storage_bytes <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
      "moe_permute_drop_temp_storage_bytes_cuda overflow");
  return static_cast<int64_t>(temp_storage_bytes);
}

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
    const c10::optional<torch::Tensor>& out) {
  c10::cuda::CUDAGuard device_guard(input.device());
  auto cuda_stream = at::cuda::getCurrentCUDAStream(input.device().index());
  cudaStream_t stream = cuda_stream.stream();

  TORCH_CHECK(input.is_contiguous(), "moe_permute_drop_fwd_cuda: input must be contiguous");
  TORCH_CHECK(
      routing_map.is_contiguous(), "moe_permute_drop_fwd_cuda: routing_map must be contiguous");
  TORCH_CHECK(
      requested_offsets.is_contiguous(),
      "moe_permute_drop_fwd_cuda: requested_offsets must be contiguous");
  TORCH_CHECK(
      keep_offsets.is_contiguous(), "moe_permute_drop_fwd_cuda: keep_offsets must be contiguous");
  TORCH_CHECK(
      keep_splits.is_contiguous(), "moe_permute_drop_fwd_cuda: keep_splits must be contiguous");

  const int64_t num_tokens = input.size(0);
  const int64_t topk = routing_map.size(1);
  const int64_t expanded_rows = num_tokens * topk;
  const int64_t num_experts = keep_splits.numel();

  TORCH_CHECK(topk > 0, "moe_permute_drop_fwd_cuda: topK must be > 0");
  TORCH_CHECK(
      requested_offsets.numel() == num_experts && keep_offsets.numel() == num_experts,
      "moe_permute_drop_fwd_cuda: offsets/splits size mismatch");
  TORCH_CHECK(
      sorted_indices_workspace.numel() >= expanded_rows,
      "moe_permute_drop_fwd_cuda: sorted_indices_workspace too small");
  TORCH_CHECK(
      row_id_workspace.numel() >= expanded_rows,
      "moe_permute_drop_fwd_cuda: row_id_workspace too small");
  TORCH_CHECK(
      sorted_row_id_workspace.numel() >= expanded_rows,
      "moe_permute_drop_fwd_cuda: sorted_row_id_workspace too small");

  if (num_out_tokens <= 0) {
    num_out_tokens = expanded_rows;
  }
  TORCH_CHECK(num_out_tokens >= 0, "moe_permute_drop_fwd_cuda: num_out_tokens must be >= 0");
  TORCH_CHECK(
      num_out_tokens <= expanded_rows,
      "moe_permute_drop_fwd_cuda: num_out_tokens must be <= expanded rows");

  auto output = get_or_create_output(input, num_out_tokens, out);
  auto row_id_map = torch::empty(
      {expanded_rows},
      torch::TensorOptions()
          .dtype(torch::kInt32)
          .device(input.device())
          .requires_grad(false));

  if (expanded_rows == 0 || input.size(1) == 0 || num_out_tokens == 0) {
    return std::make_tuple(output, row_id_map);
  }

  auto routing_flat = routing_map.reshape({expanded_rows}).contiguous();
  auto sorted_indices = sorted_indices_workspace.narrow(0, 0, expanded_rows);
  auto row_id = row_id_workspace.narrow(0, 0, expanded_rows);
  auto sorted_row_id = sorted_row_id_workspace.narrow(0, 0, expanded_rows);

  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(
      nullptr,
      temp_storage_bytes,
      routing_flat.data_ptr<int>(),
      sorted_indices.data_ptr<int>(),
      row_id.data_ptr<int>(),
      sorted_row_id.data_ptr<int>(),
      static_cast<int>(expanded_rows),
      0,
      static_cast<int>(sizeof(int) * 8),
      stream);
  TORCH_CHECK(
      temp_storage_workspace.numel() >= static_cast<int64_t>(temp_storage_bytes),
      "moe_permute_drop_fwd_cuda: temp_storage_workspace too small");

  cub::DeviceRadixSort::SortPairs(
      temp_storage_workspace.data_ptr<uint8_t>(),
      temp_storage_bytes,
      routing_flat.data_ptr<int>(),
      sorted_indices.data_ptr<int>(),
      row_id.data_ptr<int>(),
      sorted_row_id.data_ptr<int>(),
      static_cast<int>(expanded_rows),
      0,
      static_cast<int>(sizeof(int) * 8),
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      input.scalar_type(),
      "moe_permute_drop_fwd_cuda",
      [&] {
        launch_moe_permute_drop_scatter<scalar_t>(
            input,
            output,
            sorted_indices,
            sorted_row_id,
            requested_offsets,
            keep_offsets,
            keep_splits,
            row_id_map,
            static_cast<int>(num_tokens),
            static_cast<int>(topk),
            static_cast<int>(num_experts),
            static_cast<int>(num_out_tokens),
            static_cast<int>(expanded_rows),
            stream);
      });

  return std::make_tuple(output, row_id_map);
}

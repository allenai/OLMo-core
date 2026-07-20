#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cub/cub.cuh>

#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void moe_permute_row_map_kernel(
    const int* __restrict__ sorted_row_id,
    int* __restrict__ row_id_map,
    int num_rows,
    int topk,
    int num_out_tokens) {
  const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= num_rows * topk) {
    return;
  }

  const int source_row = sorted_row_id[idx];
  const int source_token_id = source_row / topk;
  const int source_topk_id = source_row % topk;

  if (idx >= num_out_tokens) {
    row_id_map[source_topk_id * num_rows + source_token_id] = -1;
  } else {
    row_id_map[source_topk_id * num_rows + source_token_id] = idx;
  }
}

template <typename scalar_t, int kElementsPerAccess>
__global__ void permute_by_row_id_map_vec_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int* __restrict__ row_id_map,
    int num_rows,
    int num_cols,
    int num_out_tokens) {
  const int source_token = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  if (source_token >= num_rows) {
    return;
  }

  const int dest_row = row_id_map[source_token];
  if (dest_row < 0 || dest_row >= num_out_tokens) {
    return;
  }

  const scalar_t* source_row_ptr = input + static_cast<int64_t>(source_token) * num_cols;
  scalar_t* dest_row_ptr = output + static_cast<int64_t>(dest_row) * num_cols;

  for (int col = tid * kElementsPerAccess; col < num_cols;
       col += static_cast<int>(blockDim.x) * kElementsPerAccess) {
    float4 frag = reinterpret_cast<const float4*>(source_row_ptr + col)[0];
    reinterpret_cast<float4*>(dest_row_ptr + col)[0] = frag;
  }
}

template <typename scalar_t>
__global__ void permute_by_row_id_map_scalar_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int* __restrict__ row_id_map,
    int num_rows,
    int num_cols,
    int num_out_tokens) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(num_rows) * num_cols;
  if (idx >= total) {
    return;
  }

  const int source_token = static_cast<int>(idx / num_cols);
  const int col = static_cast<int>(idx % num_cols);
  const int dest_row = row_id_map[source_token];
  if (dest_row < 0 || dest_row >= num_out_tokens) {
    return;
  }

  output[static_cast<int64_t>(dest_row) * num_cols + col] = input[idx];
}

template <typename scalar_t, int kElementsPerAccess>
__global__ void unpermute_vec_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int* __restrict__ row_id_map,
    int input_rows,
    int num_tokens,
    int num_cols) {
  const int token = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  if (token >= num_tokens) {
    return;
  }

  const int source_row = row_id_map[token];
  scalar_t* dest_row_ptr = output + static_cast<int64_t>(token) * num_cols;

  if (source_row < 0 || source_row >= input_rows) {
    float4 zero_frag;
    scalar_t* zero_ptr = reinterpret_cast<scalar_t*>(&zero_frag);
#pragma unroll
    for (int i = 0; i < kElementsPerAccess; ++i) {
      zero_ptr[i] = scalar_t(0);
    }
    for (int col = tid * kElementsPerAccess; col < num_cols;
         col += static_cast<int>(blockDim.x) * kElementsPerAccess) {
      reinterpret_cast<float4*>(dest_row_ptr + col)[0] = zero_frag;
    }
    return;
  }

  const scalar_t* source_row_ptr = input + static_cast<int64_t>(source_row) * num_cols;
  for (int col = tid * kElementsPerAccess; col < num_cols;
       col += static_cast<int>(blockDim.x) * kElementsPerAccess) {
    float4 frag = reinterpret_cast<const float4*>(source_row_ptr + col)[0];
    reinterpret_cast<float4*>(dest_row_ptr + col)[0] = frag;
  }
}

template <typename scalar_t>
__global__ void unpermute_scalar_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int* __restrict__ row_id_map,
    int input_rows,
    int num_tokens,
    int num_cols) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(num_tokens) * num_cols;
  if (idx >= total) {
    return;
  }

  const int token = static_cast<int>(idx / num_cols);
  const int col = static_cast<int>(idx % num_cols);
  const int source_row = row_id_map[token];
  if (source_row < 0 || source_row >= input_rows) {
    output[idx] = scalar_t(0);
    return;
  }

  output[idx] = input[static_cast<int64_t>(source_row) * num_cols + col];
}

template <typename scalar_t>
void launch_permute_by_row_id_map(
    const torch::Tensor& input,
    const torch::Tensor& row_id_map,
    const torch::Tensor& output,
    cudaStream_t stream) {
  constexpr int kElementsPerAccess = 16 / sizeof(scalar_t);
  const int num_rows = static_cast<int>(input.size(0));
  const int num_cols = static_cast<int>(input.size(1));
  const int num_out_tokens = static_cast<int>(output.size(0));

  if (num_rows == 0 || num_cols == 0 || num_out_tokens == 0) {
    return;
  }

  if (num_cols % kElementsPerAccess == 0) {
    const int threads = std::max(1, std::min(num_cols / kElementsPerAccess, 1024));
    const int blocks = num_rows;
    permute_by_row_id_map_vec_kernel<scalar_t, kElementsPerAccess><<<blocks, threads, 0, stream>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        row_id_map.data_ptr<int>(),
        num_rows,
        num_cols,
        num_out_tokens);
  } else {
    constexpr int threads = 256;
    const int64_t total = static_cast<int64_t>(num_rows) * num_cols;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    permute_by_row_id_map_scalar_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        row_id_map.data_ptr<int>(),
        num_rows,
        num_cols,
        num_out_tokens);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void launch_unpermute(
    const torch::Tensor& input,
    const torch::Tensor& row_id_map,
    const torch::Tensor& output,
    cudaStream_t stream) {
  constexpr int kElementsPerAccess = 16 / sizeof(scalar_t);
  const int input_rows = static_cast<int>(input.size(0));
  const int num_tokens = static_cast<int>(output.size(0));
  const int num_cols = static_cast<int>(input.size(1));

  if (num_tokens == 0 || num_cols == 0) {
    return;
  }

  if (num_cols % kElementsPerAccess == 0) {
    const int threads = std::max(1, std::min(num_cols / kElementsPerAccess, 1024));
    const int blocks = num_tokens;
    unpermute_vec_kernel<scalar_t, kElementsPerAccess><<<blocks, threads, 0, stream>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        row_id_map.data_ptr<int>(),
        input_rows,
        num_tokens,
        num_cols);
  } else {
    constexpr int threads = 256;
    const int64_t total = static_cast<int64_t>(num_tokens) * num_cols;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    unpermute_scalar_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        row_id_map.data_ptr<int>(),
        input_rows,
        num_tokens,
        num_cols);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor get_or_create_output(
    const torch::Tensor& input,
    int64_t out_rows,
    const c10::optional<torch::Tensor>& out,
    const char* op_name) {
  if (out.has_value()) {
    const auto& out_tensor = out.value();
    TORCH_CHECK(out_tensor.is_cuda(), op_name, ": out must be CUDA");
    TORCH_CHECK(out_tensor.dim() == 2, op_name, ": out must be rank-2");
    TORCH_CHECK(
        out_tensor.size(0) == out_rows && out_tensor.size(1) == input.size(1),
        op_name,
        ": out shape mismatch, expected [",
        out_rows,
        ", ",
        input.size(1),
        "] got ",
        out_tensor.sizes());
    TORCH_CHECK(
        out_tensor.scalar_type() == input.scalar_type(),
        op_name,
        ": out dtype mismatch");
    TORCH_CHECK(out_tensor.device() == input.device(), op_name, ": out device mismatch");
    TORCH_CHECK(out_tensor.is_contiguous(), op_name, ": out must be contiguous");
    return out_tensor;
  }

  return torch::empty(
      {out_rows, input.size(1)},
      torch::TensorOptions()
          .dtype(input.scalar_type())
          .device(input.device())
          .requires_grad(false));
}

std::tuple<torch::Tensor, torch::Tensor> chunk_permute_fwd_cuda_launcher(
    const torch::Tensor& input,
    const torch::Tensor& routing_map,
    int64_t num_out_tokens,
    const c10::optional<torch::Tensor>& out) {
  c10::cuda::CUDAGuard device_guard(input.device());
  auto cuda_stream = at::cuda::getCurrentCUDAStream(input.device().index());
  cudaStream_t stream = cuda_stream.stream();

  TORCH_CHECK(input.is_contiguous(), "chunk_permute_fwd_cuda: input must be contiguous");
  TORCH_CHECK(
      routing_map.is_contiguous(),
      "chunk_permute_fwd_cuda: routing_map must be contiguous");

  const int64_t num_tokens = input.size(0);
  if (num_out_tokens <= 0) {
    num_out_tokens = num_tokens;
  }
  TORCH_CHECK(
      num_out_tokens <= num_tokens,
      "chunk_permute_fwd_cuda: num_out_tokens must be <= num_tokens, got ",
      num_out_tokens,
      " > ",
      num_tokens);

  auto output = get_or_create_output(input, num_out_tokens, out, "chunk_permute_fwd_cuda");
  auto row_id_map = torch::empty(
      {num_tokens},
      torch::TensorOptions()
          .dtype(torch::kInt32)
          .device(input.device())
          .requires_grad(false));

  if (num_tokens == 0 || input.size(1) == 0 || num_out_tokens == 0) {
    return std::make_tuple(output, row_id_map);
  }

  auto routing_flat = routing_map.reshape({num_tokens}).contiguous();
  auto sorted_indices = torch::empty_like(routing_flat);
  auto row_id = torch::arange(
      num_tokens,
      torch::TensorOptions()
          .dtype(torch::kInt32)
          .device(input.device())
          .requires_grad(false));
  auto sorted_row_id = torch::empty_like(row_id);

  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(
      nullptr,
      temp_storage_bytes,
      routing_flat.data_ptr<int>(),
      sorted_indices.data_ptr<int>(),
      row_id.data_ptr<int>(),
      sorted_row_id.data_ptr<int>(),
      static_cast<int>(num_tokens),
      0,
      static_cast<int>(sizeof(int) * 8),
      stream);

  auto temp_storage = torch::empty(
      {static_cast<int64_t>(temp_storage_bytes)},
      torch::TensorOptions()
          .dtype(torch::kUInt8)
          .device(input.device())
          .requires_grad(false));

  cub::DeviceRadixSort::SortPairs(
      temp_storage.data_ptr<uint8_t>(),
      temp_storage_bytes,
      routing_flat.data_ptr<int>(),
      sorted_indices.data_ptr<int>(),
      row_id.data_ptr<int>(),
      sorted_row_id.data_ptr<int>(),
      static_cast<int>(num_tokens),
      0,
      static_cast<int>(sizeof(int) * 8),
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  constexpr int kMapThreads = 64;
  const int map_blocks = static_cast<int>((num_tokens + kMapThreads - 1) / kMapThreads);
  moe_permute_row_map_kernel<<<map_blocks, kMapThreads, 0, stream>>>(
      sorted_row_id.data_ptr<int>(),
      row_id_map.data_ptr<int>(),
      static_cast<int>(num_tokens),
      1,
      static_cast<int>(num_out_tokens));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      input.scalar_type(),
      "chunk_permute_fwd_cuda",
      [&] {
        launch_permute_by_row_id_map<scalar_t>(input, row_id_map, output, stream);
      });

  return std::make_tuple(output, row_id_map);
}

torch::Tensor chunk_permute_by_row_id_map_cuda_launcher(
    const torch::Tensor& input,
    const torch::Tensor& row_id_map,
    int64_t num_out_tokens,
    const c10::optional<torch::Tensor>& out) {
  c10::cuda::CUDAGuard device_guard(input.device());
  auto cuda_stream = at::cuda::getCurrentCUDAStream(input.device().index());
  cudaStream_t stream = cuda_stream.stream();

  TORCH_CHECK(
      input.is_contiguous(),
      "chunk_permute_by_row_id_map_cuda: input must be contiguous");
  TORCH_CHECK(
      row_id_map.is_contiguous(),
      "chunk_permute_by_row_id_map_cuda: row_id_map must be contiguous");
  TORCH_CHECK(
      row_id_map.size(0) == input.size(0),
      "chunk_permute_by_row_id_map_cuda: row_id_map/input rows must match");

  if (num_out_tokens <= 0) {
    num_out_tokens = input.size(0);
  }

  auto output = get_or_create_output(
      input,
      num_out_tokens,
      out,
      "chunk_permute_by_row_id_map_cuda");

  if (input.size(0) == 0 || input.size(1) == 0 || num_out_tokens == 0) {
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      input.scalar_type(),
      "chunk_permute_by_row_id_map_cuda",
      [&] {
        launch_permute_by_row_id_map<scalar_t>(input, row_id_map, output, stream);
      });

  return output;
}

torch::Tensor chunk_unpermute_fwd_cuda_launcher(
    const torch::Tensor& input,
    const torch::Tensor& row_id_map,
    int64_t num_tokens,
    const c10::optional<torch::Tensor>& out) {
  c10::cuda::CUDAGuard device_guard(input.device());
  auto cuda_stream = at::cuda::getCurrentCUDAStream(input.device().index());
  cudaStream_t stream = cuda_stream.stream();

  TORCH_CHECK(input.is_contiguous(), "chunk_unpermute_fwd_cuda: input must be contiguous");
  TORCH_CHECK(
      row_id_map.is_contiguous(),
      "chunk_unpermute_fwd_cuda: row_id_map must be contiguous");

  if (num_tokens <= 0) {
    num_tokens = row_id_map.size(0);
  }
  TORCH_CHECK(
      row_id_map.size(0) == num_tokens,
      "chunk_unpermute_fwd_cuda: row_id_map length must equal num_tokens");

  auto output = get_or_create_output(input, num_tokens, out, "chunk_unpermute_fwd_cuda");

  if (num_tokens == 0 || input.size(1) == 0) {
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      input.scalar_type(),
      "chunk_unpermute_fwd_cuda",
      [&] { launch_unpermute<scalar_t>(input, row_id_map, output, stream); });

  return output;
}

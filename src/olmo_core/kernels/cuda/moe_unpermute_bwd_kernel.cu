#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cub/cub.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <type_traits>

namespace {

torch::Tensor get_or_create_output(
    const torch::Tensor& input_fwd,
    const c10::optional<torch::Tensor>& out) {
  if (out.has_value()) {
    auto out_tensor = out.value();
    TORCH_CHECK(out_tensor.is_cuda(), "moe_unpermute_bwd_cuda: out must be CUDA");
    TORCH_CHECK(out_tensor.dim() == 2, "moe_unpermute_bwd_cuda: out must be rank-2");
    TORCH_CHECK(
        out_tensor.size(0) == input_fwd.size(0) && out_tensor.size(1) == input_fwd.size(1),
        "moe_unpermute_bwd_cuda: out shape mismatch, expected [",
        input_fwd.size(0),
        ", ",
        input_fwd.size(1),
        "] got ",
        out_tensor.sizes());
    TORCH_CHECK(
        out_tensor.scalar_type() == input_fwd.scalar_type(),
        "moe_unpermute_bwd_cuda: out dtype mismatch");
    TORCH_CHECK(
        out_tensor.device() == input_fwd.device(),
        "moe_unpermute_bwd_cuda: out device mismatch");
    TORCH_CHECK(out_tensor.is_contiguous(), "moe_unpermute_bwd_cuda: out must be contiguous");

    return out_tensor;
  }

  return torch::empty(
      {input_fwd.size(0), input_fwd.size(1)},
      torch::TensorOptions()
          .dtype(input_fwd.scalar_type())
          .device(input_fwd.device())
          .requires_grad(false));
}

template <typename scalar_t, int kThreads>
__global__ void zero_rows_by_keep_mask_scalar_kernel(
    scalar_t* __restrict__ output,
    const bool* __restrict__ keep_mask,
    int num_rows,
    int num_cols) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(num_rows) * num_cols;
  if (idx >= total) {
    return;
  }
  const int row = static_cast<int>(idx / num_cols);
  if (!keep_mask[row]) {
    output[idx] = scalar_t(0);
  }
}

template <typename scalar_t, int kElementsPerAccess>
__global__ void zero_rows_by_keep_mask_vec_kernel(
    scalar_t* __restrict__ output,
    const bool* __restrict__ keep_mask,
    int num_rows,
    int num_cols) {
  const int row = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  if (row >= num_rows || keep_mask[row]) {
    return;
  }

  float4 zero_frag;
  scalar_t* zero_ptr = reinterpret_cast<scalar_t*>(&zero_frag);
#pragma unroll
  for (int i = 0; i < kElementsPerAccess; ++i) {
    zero_ptr[i] = scalar_t(0);
  }

  scalar_t* row_ptr = output + static_cast<int64_t>(row) * num_cols;
  for (int col = tid * kElementsPerAccess; col < num_cols;
       col += static_cast<int>(blockDim.x) * kElementsPerAccess) {
    *reinterpret_cast<float4*>(row_ptr + col) = zero_frag;
  }
}

template <typename scalar_t>
void launch_zero_rows_by_keep_mask(
    const torch::Tensor& output,
    const torch::Tensor& keep_mask,
    cudaStream_t stream) {
  const int num_rows = static_cast<int>(output.size(0));
  const int num_cols = static_cast<int>(output.size(1));
  if (num_rows == 0 || num_cols == 0) {
    return;
  }

  static constexpr int kElementsPerAccess = 16 / sizeof(scalar_t);
  if (num_cols % kElementsPerAccess == 0) {
    const int threads = std::max(1, std::min(num_cols / kElementsPerAccess, 1024));
    const int blocks = num_rows;
    zero_rows_by_keep_mask_vec_kernel<scalar_t, kElementsPerAccess><<<blocks, threads, 0, stream>>>(
        output.data_ptr<scalar_t>(),
        keep_mask.data_ptr<bool>(),
        num_rows,
        num_cols);
  } else {
    constexpr int kThreads = 256;
    const int64_t total = static_cast<int64_t>(num_rows) * num_cols;
    const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
    zero_rows_by_keep_mask_scalar_kernel<scalar_t, kThreads><<<blocks, kThreads, 0, stream>>>(
        output.data_ptr<scalar_t>(),
        keep_mask.data_ptr<bool>(),
        num_rows,
        num_cols);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, typename compute_t, int topk_tile, bool has_prob>
__global__ void moe_unpermute_bwd_vec_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ input_fwd,
    scalar_t* __restrict__ act_grad,
    const int* __restrict__ row_id_map,
    const float* __restrict__ probs,
    float* __restrict__ prob_grad,
    int num_tokens,
    int topk,
    int num_cols,
    int input_rows) {
  extern __shared__ int8_t shared_mem[];
  compute_t* shared_probs = reinterpret_cast<compute_t*>(shared_mem);

  const int token = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  if (token >= num_tokens) {
    return;
  }

  if (has_prob) {
    for (int i = tid; i < topk; i += blockDim.x) {
      shared_probs[i] = static_cast<compute_t>(probs[token * topk + i]);
    }
    __syncthreads();
  }

  float accum[topk_tile] = {0.0f};
  float4 frag_load_store;
  scalar_t* frag_load_store_ptr = reinterpret_cast<scalar_t*>(&frag_load_store);
  static constexpr int kElementsPerAccess = 16 / sizeof(scalar_t);

  const scalar_t* grad_row_ptr = grad_output + static_cast<int64_t>(token) * num_cols;
  for (int col = tid * kElementsPerAccess; col < num_cols;
       col += static_cast<int>(blockDim.x) * kElementsPerAccess) {
    compute_t grad_frag[kElementsPerAccess];
    frag_load_store = __ldlu(reinterpret_cast<const float4*>(grad_row_ptr + col));
    for (int e = 0; e < kElementsPerAccess; ++e) {
      grad_frag[e] = static_cast<compute_t>(frag_load_store_ptr[e]);
    }

    int index = token;
    for (int k = 0; k < topk_tile; ++k) {
      if (k == topk) {
        break;
      }
      const int dest_row = row_id_map[index];
      index += num_tokens;

      if (dest_row < 0 || dest_row >= input_rows) {
        continue;
      }

      if (has_prob) {
        const compute_t p = shared_probs[k];
        for (int e = 0; e < kElementsPerAccess; ++e) {
          frag_load_store_ptr[e] = static_cast<scalar_t>(grad_frag[e] * p);
        }
      } else {
        for (int e = 0; e < kElementsPerAccess; ++e) {
          frag_load_store_ptr[e] = static_cast<scalar_t>(grad_frag[e]);
        }
      }

      scalar_t* out_row_ptr = act_grad + static_cast<int64_t>(dest_row) * num_cols;
      *reinterpret_cast<float4*>(out_row_ptr + col) = frag_load_store;

      if (has_prob) {
        const scalar_t* input_fwd_ptr = input_fwd + static_cast<int64_t>(dest_row) * num_cols;
        frag_load_store = __ldlu(reinterpret_cast<const float4*>(input_fwd_ptr + col));
        for (int e = 0; e < kElementsPerAccess; ++e) {
          const compute_t input_val = static_cast<compute_t>(frag_load_store_ptr[e]);
          accum[k] += static_cast<float>(grad_frag[e] * input_val);
        }
      }
    }
  }

  if (has_prob) {
#pragma unroll
    for (int k = 0; k < topk_tile; ++k) {
      if (k == topk) {
        break;
      }
      for (int mask = 16; mask > 0; mask /= 2) {
        accum[k] += __shfl_xor_sync(0xffffffff, accum[k], mask, 32);
      }
    }

    if (tid == 0) {
      for (int k = 0; k < topk_tile; ++k) {
        if (k == topk) {
          break;
        }
        prob_grad[token * topk + k] = accum[k];
      }
    }
  }
}

template <typename scalar_t, int kThreads>
__global__ void moe_unpermute_bwd_scalar_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ input_fwd,
    scalar_t* __restrict__ act_grad,
    const int* __restrict__ row_id_map,
    const float* __restrict__ probs,
    float* __restrict__ prob_grad,
    int num_tokens,
    int topk,
    int num_cols,
    int input_rows) {
  const int token = static_cast<int>(blockIdx.x);
  const int k = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);
  if (token >= num_tokens || k >= topk) {
    return;
  }

  const int dest_row = row_id_map[k * num_tokens + token];
  float local_prob_grad = 0.0f;
  if (dest_row >= 0 && dest_row < input_rows) {
    const float p = probs[token * topk + k];
    const scalar_t* grad_row_ptr = grad_output + static_cast<int64_t>(token) * num_cols;
    const scalar_t* input_row_ptr = input_fwd + static_cast<int64_t>(dest_row) * num_cols;
    scalar_t* out_row_ptr = act_grad + static_cast<int64_t>(dest_row) * num_cols;
    for (int col = tid; col < num_cols; col += kThreads) {
      const float grad_val = static_cast<float>(grad_row_ptr[col]);
      out_row_ptr[col] = static_cast<scalar_t>(grad_val * p);
      local_prob_grad += grad_val * static_cast<float>(input_row_ptr[col]);
    }
  }

  using BlockReduce = cub::BlockReduce<float, kThreads>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  const float sum = BlockReduce(temp_storage).Sum(local_prob_grad);
  if (tid == 0) {
    prob_grad[token * topk + k] = (dest_row >= 0 && dest_row < input_rows) ? sum : 0.0f;
  }
}

template <typename scalar_t>
void launch_moe_unpermute_bwd(
    const torch::Tensor& grad_output,
    const torch::Tensor& input_fwd,
    const torch::Tensor& row_id_map,
    const torch::Tensor& probs,
    const torch::Tensor& act_grad,
    const torch::Tensor& prob_grad,
    cudaStream_t stream) {
  const int num_tokens = static_cast<int>(grad_output.size(0));
  const int topk = static_cast<int>(probs.size(1));
  const int num_cols = static_cast<int>(grad_output.size(1));
  const int input_rows = static_cast<int>(input_fwd.size(0));

  if (num_tokens == 0 || topk == 0 || num_cols == 0) {
    return;
  }

  using compute_t = scalar_t;
  static constexpr int kElementsPerAccess = 16 / sizeof(scalar_t);

  if (num_cols % kElementsPerAccess == 0) {
    const int threads = 32;
    const int blocks = num_tokens;
    const size_t smem_bytes = static_cast<size_t>(topk) * sizeof(compute_t);
    if (topk <= 8) {
      moe_unpermute_bwd_vec_kernel<scalar_t, compute_t, 8, true><<<blocks, threads, smem_bytes, stream>>>(
          grad_output.data_ptr<scalar_t>(),
          input_fwd.data_ptr<scalar_t>(),
          act_grad.data_ptr<scalar_t>(),
          row_id_map.data_ptr<int>(),
          probs.data_ptr<float>(),
          prob_grad.data_ptr<float>(),
          num_tokens,
          topk,
          num_cols,
          input_rows);
    } else if (topk <= 16) {
      moe_unpermute_bwd_vec_kernel<scalar_t, compute_t, 16, true><<<blocks, threads, smem_bytes, stream>>>(
          grad_output.data_ptr<scalar_t>(),
          input_fwd.data_ptr<scalar_t>(),
          act_grad.data_ptr<scalar_t>(),
          row_id_map.data_ptr<int>(),
          probs.data_ptr<float>(),
          prob_grad.data_ptr<float>(),
          num_tokens,
          topk,
          num_cols,
          input_rows);
    } else if (topk <= 32) {
      moe_unpermute_bwd_vec_kernel<scalar_t, compute_t, 32, true><<<blocks, threads, smem_bytes, stream>>>(
          grad_output.data_ptr<scalar_t>(),
          input_fwd.data_ptr<scalar_t>(),
          act_grad.data_ptr<scalar_t>(),
          row_id_map.data_ptr<int>(),
          probs.data_ptr<float>(),
          prob_grad.data_ptr<float>(),
          num_tokens,
          topk,
          num_cols,
          input_rows);
    } else if (topk <= 64) {
      moe_unpermute_bwd_vec_kernel<scalar_t, compute_t, 64, true><<<blocks, threads, smem_bytes, stream>>>(
          grad_output.data_ptr<scalar_t>(),
          input_fwd.data_ptr<scalar_t>(),
          act_grad.data_ptr<scalar_t>(),
          row_id_map.data_ptr<int>(),
          probs.data_ptr<float>(),
          prob_grad.data_ptr<float>(),
          num_tokens,
          topk,
          num_cols,
          input_rows);
    } else {
      moe_unpermute_bwd_vec_kernel<scalar_t, compute_t, 128, true><<<blocks, threads, smem_bytes, stream>>>(
          grad_output.data_ptr<scalar_t>(),
          input_fwd.data_ptr<scalar_t>(),
          act_grad.data_ptr<scalar_t>(),
          row_id_map.data_ptr<int>(),
          probs.data_ptr<float>(),
          prob_grad.data_ptr<float>(),
          num_tokens,
          topk,
          num_cols,
          input_rows);
    }
  } else {
    constexpr int kThreads = 256;
    const dim3 grid(num_tokens, topk);
    moe_unpermute_bwd_scalar_kernel<scalar_t, kThreads><<<grid, kThreads, 0, stream>>>(
        grad_output.data_ptr<scalar_t>(),
        input_fwd.data_ptr<scalar_t>(),
        act_grad.data_ptr<scalar_t>(),
        row_id_map.data_ptr<int>(),
        probs.data_ptr<float>(),
        prob_grad.data_ptr<float>(),
        num_tokens,
        topk,
        num_cols,
        input_rows);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> moe_unpermute_bwd_cuda_launcher(
    const torch::Tensor& grad_output,
    const torch::Tensor& input_fwd,
    const torch::Tensor& row_id_map,
    const torch::Tensor& probs,
    const c10::optional<torch::Tensor>& keep_mask,
    const c10::optional<torch::Tensor>& out) {
  c10::cuda::CUDAGuard device_guard(grad_output.device());
  auto cuda_stream = at::cuda::getCurrentCUDAStream(grad_output.device().index());
  cudaStream_t stream = cuda_stream.stream();

  TORCH_CHECK(grad_output.is_contiguous(), "moe_unpermute_bwd_cuda: grad_output must be contiguous");
  TORCH_CHECK(input_fwd.is_contiguous(), "moe_unpermute_bwd_cuda: input_fwd must be contiguous");
  TORCH_CHECK(row_id_map.is_contiguous(), "moe_unpermute_bwd_cuda: row_id_map must be contiguous");
  TORCH_CHECK(probs.is_contiguous(), "moe_unpermute_bwd_cuda: probs must be contiguous");

  const int64_t num_tokens = grad_output.size(0);
  const int64_t topk = probs.size(1);

  auto act_grad = get_or_create_output(input_fwd, out);
  TORCH_CHECK(topk > 0, "moe_unpermute_bwd_cuda: topk must be > 0");
  TORCH_CHECK(topk <= 128, "moe_unpermute_bwd_cuda: topk must be <= 128");
  if (keep_mask.has_value()) {
    const auto& keep_mask_tensor = keep_mask.value();
    TORCH_CHECK(
        keep_mask_tensor.is_contiguous(),
        "moe_unpermute_bwd_cuda: keep_mask must be contiguous");
  } else {
    act_grad.zero_();
  }

  auto prob_grad = torch::empty(
      {num_tokens, topk},
      torch::TensorOptions().dtype(torch::kFloat32).device(grad_output.device()).requires_grad(false));

  if (num_tokens == 0 || topk == 0 || grad_output.size(1) == 0) {
    return std::make_tuple(act_grad, prob_grad);
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      grad_output.scalar_type(),
      "moe_unpermute_bwd_cuda",
      [&] {
        if (keep_mask.has_value()) {
          launch_zero_rows_by_keep_mask<scalar_t>(act_grad, keep_mask.value(), stream);
        }
        launch_moe_unpermute_bwd<scalar_t>(
            grad_output,
            input_fwd,
            row_id_map,
            probs,
            act_grad,
            prob_grad,
            stream);
      });

  return std::make_tuple(act_grad, prob_grad);
}

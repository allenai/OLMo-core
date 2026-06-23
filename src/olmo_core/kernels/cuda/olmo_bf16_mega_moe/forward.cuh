/*
 * OLMo BF16 MegaMoE forward API contract.
 *
 * This header is the forward-facing host surface for the OLMo-owned MegaMoE
 * runtime. It checks the BF16 OLMo RoutedExperts tensor contract and produces
 * the same launch metadata consumed by the persistent kernel implementation.
 */
#pragma once

#include "barrier.cuh"
#include "dispatch.cuh"
#include "mma.cuh"
#include "ptx.cuh"
#include "runtime.cuh"
#include "scheduler.cuh"
#include "shared_storage.cuh"
#include "tensor_map.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cstdint>

namespace olmo::bf16_mega_moe {

struct ForwardProblem {
  int64_t num_tokens = 0;
  int64_t top_k = 0;
  int64_t hidden = 0;
  int64_t intermediate = 0;
  int64_t num_local_experts = 0;
  int64_t num_ranks = 1;
  int64_t num_total_experts = 0;
  int64_t num_max_tokens_per_rank = 0;
  LaunchConfig launch;
  Bf16TensorMap2D source_input_map;
  Bf16TensorMap2D out_map;
  Bf16TensorMap2D up_gate_weight_map;
  Bf16TensorMap2D down_weight_map;
};

inline void check_same_cuda_device(const at::Tensor& base, const at::Tensor& other, const char* name) {
  TORCH_CHECK(other.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(
      other.get_device() == base.get_device(),
      name,
      " must be on the same CUDA device as source_input");
}

inline void check_bf16_tensor(const at::Tensor& base, const at::Tensor& tensor, const char* name) {
  check_same_cuda_device(base, tensor, name);
  TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, name, " must be BF16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

inline void check_integer_route_tensor(
    const at::Tensor& base,
    const at::Tensor& tensor,
    const char* name,
    int64_t num_tokens,
    int64_t top_k) {
  check_same_cuda_device(base, tensor, name);
  TORCH_CHECK(
      tensor.scalar_type() == at::kLong || tensor.scalar_type() == at::kInt,
      name,
      " must be int64 or int32");
  TORCH_CHECK(tensor.dim() == 2, name, " must be rank-2 [tokens, top_k]");
  TORCH_CHECK(tensor.size(0) == num_tokens, name, " token dimension mismatch");
  TORCH_CHECK(tensor.size(1) == top_k, name, " top_k dimension mismatch");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

inline ForwardProblem make_forward_problem(
    const at::Tensor& source_input,
    const at::Tensor& gathered_out,
    const at::Tensor& out,
    const at::Tensor& route_dst_ranks,
    const at::Tensor& route_dst_rows,
    const at::Tensor& route_expert_indices,
    const at::Tensor& probs,
    const at::Tensor& up_gate_weight,
    const at::Tensor& down_weight,
    const at::Tensor& expert_offsets,
    int64_t num_ranks,
    int64_t num_total_experts,
    int64_t num_max_tokens_per_rank,
    int64_t num_sms) {
  TORCH_CHECK(source_input.is_cuda(), "source_input must be a CUDA tensor");
  check_bf16_tensor(source_input, source_input, "source_input");
  TORCH_CHECK(source_input.dim() == 2, "source_input must be rank-2 [tokens, hidden]");
  const int64_t num_tokens = source_input.size(0);
  const int64_t hidden = source_input.size(1);

  check_bf16_tensor(source_input, out, "out");
  TORCH_CHECK(out.dim() == 2, "out must be rank-2 [tokens, hidden]");
  TORCH_CHECK(out.size(0) == num_tokens, "out token dimension mismatch");
  TORCH_CHECK(out.size(1) == hidden, "out hidden dimension mismatch");

  check_bf16_tensor(source_input, gathered_out, "gathered_out");
  TORCH_CHECK(gathered_out.dim() == 3, "gathered_out must be rank-3 [tokens, top_k, hidden]");
  TORCH_CHECK(gathered_out.size(0) == num_tokens, "gathered_out token dimension mismatch");
  TORCH_CHECK(gathered_out.size(2) == hidden, "gathered_out hidden dimension mismatch");
  const int64_t top_k = gathered_out.size(1);

  check_integer_route_tensor(source_input, route_dst_ranks, "route_dst_ranks", num_tokens, top_k);
  check_integer_route_tensor(source_input, route_dst_rows, "route_dst_rows", num_tokens, top_k);
  check_integer_route_tensor(
      source_input,
      route_expert_indices,
      "route_expert_indices",
      num_tokens,
      top_k);

  check_same_cuda_device(source_input, probs, "probs");
  TORCH_CHECK(probs.scalar_type() == at::kFloat, "probs must be FP32");
  TORCH_CHECK(probs.dim() == 2, "probs must be rank-2 [tokens, top_k]");
  TORCH_CHECK(probs.size(0) == num_tokens, "probs token dimension mismatch");
  TORCH_CHECK(probs.size(1) == top_k, "probs top_k dimension mismatch");
  TORCH_CHECK(probs.is_contiguous(), "probs must be contiguous");

  check_bf16_tensor(source_input, up_gate_weight, "up_gate_weight");
  check_bf16_tensor(source_input, down_weight, "down_weight");
  TORCH_CHECK(up_gate_weight.dim() == 3, "up_gate_weight must be rank-3 [experts, 2*intermediate, hidden]");
  TORCH_CHECK(down_weight.dim() == 3, "down_weight must be rank-3 [experts, intermediate, hidden]");
  const int64_t num_local_experts = up_gate_weight.size(0);
  const int64_t intermediate = down_weight.size(1);
  TORCH_CHECK(num_local_experts > 0, "num_local_experts must be > 0");
  TORCH_CHECK(down_weight.size(0) == num_local_experts, "expert count mismatch between weights");
  TORCH_CHECK(up_gate_weight.size(1) == 2 * intermediate, "up_gate_weight must have 2*intermediate rows");
  TORCH_CHECK(up_gate_weight.size(2) == hidden, "up_gate_weight hidden dimension mismatch");
  TORCH_CHECK(down_weight.size(2) == hidden, "down_weight hidden dimension mismatch");

  check_same_cuda_device(source_input, expert_offsets, "expert_offsets");
  TORCH_CHECK(
      expert_offsets.scalar_type() == at::kInt || expert_offsets.scalar_type() == at::kLong,
      "expert_offsets must be int32 or int64");
  TORCH_CHECK(expert_offsets.dim() == 1, "expert_offsets must be rank-1");
  TORCH_CHECK(
      expert_offsets.numel() == num_local_experts + 1 ||
          expert_offsets.numel() == num_local_experts,
      "expert_offsets must have local_experts or local_experts + 1 entries");

  ForwardProblem problem;
  problem.num_tokens = num_tokens;
  problem.top_k = top_k;
  problem.hidden = hidden;
  problem.intermediate = intermediate;
  problem.num_local_experts = num_local_experts;
  problem.num_ranks = num_ranks;
  problem.num_total_experts = num_total_experts;
  problem.num_max_tokens_per_rank = num_max_tokens_per_rank;
  problem.launch = make_launch_config(
      num_ranks,
      num_total_experts,
      num_local_experts,
      num_max_tokens_per_rank,
      num_tokens,
      top_k,
      hidden,
      intermediate,
      num_sms);
  problem.source_input_map = make_bf16_tma_2d_desc(
      source_input,
      /*box_cols=*/problem.launch.block_k,
      /*box_rows=*/problem.launch.load_block_m,
      /*swizzle_bytes=*/128);
  problem.out_map = make_bf16_tma_2d_desc(
      out,
      /*box_cols=*/problem.launch.block_k,
      /*box_rows=*/problem.launch.store_block_m,
      /*swizzle_bytes=*/128);
  problem.up_gate_weight_map = make_bf16_tma_2d_desc(
      up_gate_weight.view({num_local_experts * 2 * intermediate, hidden}),
      /*box_cols=*/problem.launch.block_k,
      /*box_rows=*/problem.launch.block_n,
      /*swizzle_bytes=*/128);
  problem.down_weight_map = make_bf16_tma_2d_desc(
      down_weight.view({num_local_experts * intermediate, hidden}),
      /*box_cols=*/problem.launch.block_k,
      /*box_rows=*/problem.launch.block_n,
      /*swizzle_bytes=*/128);
  return problem;
}

inline int64_t current_device_sm_count() {
  return at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
}

inline void fail_unimplemented_forward_body() {
  TORCH_CHECK(
      false,
      "rowwise_bf16_mega_moe_forward_persistent is the OLMo-owned production "
      "BF16 MegaMoE entry point. CUDA BF16 single-rank local forward is "
      "implemented, but this execution mode is unsupported; peer-window "
      "transport, persistent multi-rank scheduling, and backward are not "
      "implemented yet.");
}

}  // namespace olmo::bf16_mega_moe

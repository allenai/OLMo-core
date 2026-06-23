/*
 * OLMo BF16 MegaMoE buffer layout.
 *
 * DeepGEMM MegaMoE lays out a workspace followed by registered inputs,
 * dispatch metadata, per-expert pooled activations, and combine buffers. This
 * file keeps that runtime shape for BF16 while removing FP8/FP4 scale-factor
 * regions that do not exist in the first OLMo BF16 target.
 */
#pragma once

#include "layout.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace olmo::bf16_mega_moe::buffers {

struct Bf16ForwardBufferLayout {
  layout::Buffer input_token_buffer;
  layout::Buffer input_topk_idx_buffer;
  layout::Buffer input_topk_weight_buffer;
  layout::Buffer l1_token_buffer;
  layout::Buffer l1_topk_weight_buffer;
  layout::Buffer l2_token_buffer;
  layout::Buffer combine_token_buffer;

  OLMO_BF16_MEGA_HOST_DEVICE void* end_ptr() const {
    return combine_token_buffer.end_ptr();
  }

  OLMO_BF16_MEGA_HOST_DEVICE uint64_t num_bytes_from_base(void* base) const {
    return static_cast<uint64_t>(
        reinterpret_cast<uint8_t*>(end_ptr()) - reinterpret_cast<uint8_t*>(base));
  }
};

OLMO_BF16_MEGA_HOST_DEVICE inline Bf16ForwardBufferLayout make_bf16_forward_buffer_layout(
    void* base,
    uint32_t num_max_tokens_per_rank,
    uint32_t top_k,
    uint32_t hidden,
    uint32_t intermediate,
    uint32_t num_max_pool_tokens) {
  const layout::Data bf16_hidden_token(
      hidden * static_cast<uint32_t>(sizeof(__nv_bfloat16)));
  const layout::Data bf16_intermediate_token(
      intermediate * static_cast<uint32_t>(sizeof(__nv_bfloat16)));
  const layout::Data topk_idx_token(top_k * static_cast<uint32_t>(sizeof(int64_t)), false);
  const layout::Data topk_weight_token(top_k * static_cast<uint32_t>(sizeof(float)), false);
  const layout::Data pooled_topk_weight(static_cast<uint32_t>(sizeof(float)), false);

  Bf16ForwardBufferLayout buffers{
      /*input_token_buffer=*/layout::Buffer(bf16_hidden_token, 1, num_max_tokens_per_rank, base),
      /*input_topk_idx_buffer=*/layout::Buffer(topk_idx_token, 1, num_max_tokens_per_rank),
      /*input_topk_weight_buffer=*/layout::Buffer(topk_weight_token, 1, num_max_tokens_per_rank),
      /*l1_token_buffer=*/layout::Buffer(bf16_hidden_token, 1, num_max_pool_tokens),
      /*l1_topk_weight_buffer=*/layout::Buffer(pooled_topk_weight, 1, num_max_pool_tokens),
      /*l2_token_buffer=*/layout::Buffer(bf16_intermediate_token, 1, num_max_pool_tokens),
      /*combine_token_buffer=*/layout::Buffer(bf16_hidden_token, top_k, num_max_tokens_per_rank),
  };

  buffers.input_topk_idx_buffer.base = buffers.input_token_buffer.end_ptr();
  buffers.input_topk_weight_buffer.base = buffers.input_topk_idx_buffer.end_ptr();
  buffers.l1_token_buffer.base = buffers.input_topk_weight_buffer.end_ptr();
  buffers.l1_topk_weight_buffer.base = buffers.l1_token_buffer.end_ptr();
  buffers.l2_token_buffer.base = buffers.l1_topk_weight_buffer.end_ptr();
  buffers.combine_token_buffer.base = buffers.l2_token_buffer.end_ptr();
  return buffers;
}

OLMO_BF16_MEGA_HOST_DEVICE inline uint64_t bf16_forward_buffer_bytes(
    uint32_t num_max_tokens_per_rank,
    uint32_t top_k,
    uint32_t hidden,
    uint32_t intermediate,
    uint32_t num_max_pool_tokens) {
  const uint64_t hidden_token_bytes =
      static_cast<uint64_t>(hidden) * sizeof(__nv_bfloat16);
  const uint64_t intermediate_token_bytes =
      static_cast<uint64_t>(intermediate) * sizeof(__nv_bfloat16);
  uint64_t bytes = 0;
  bytes += static_cast<uint64_t>(num_max_tokens_per_rank) * hidden_token_bytes;
  bytes += static_cast<uint64_t>(num_max_tokens_per_rank) * top_k * sizeof(int64_t);
  bytes += static_cast<uint64_t>(num_max_tokens_per_rank) * top_k * sizeof(float);
  bytes += static_cast<uint64_t>(num_max_pool_tokens) * hidden_token_bytes;
  bytes += static_cast<uint64_t>(num_max_pool_tokens) * sizeof(float);
  bytes += static_cast<uint64_t>(num_max_pool_tokens) * intermediate_token_bytes;
  bytes += static_cast<uint64_t>(top_k) * num_max_tokens_per_rank * hidden_token_bytes;
  return bytes;
}

}  // namespace olmo::bf16_mega_moe::buffers

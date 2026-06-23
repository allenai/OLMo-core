/*
 * OLMo BF16 MegaMoE runtime contract.
 *
 * This file deliberately adapts the structure of DeepGEMM's MegaMoE runtime
 * and scheduler into OLMo-owned code. It keeps the same high-level concepts:
 * pool sizing, expert waves, persistent launch thread roles, and BF16 SM100
 * launch metadata. It does not include DeepGEMM runtime headers or kernels.
 */
#pragma once

#include "buffers.cuh"
#include "layout.cuh"
#include "megakernel_plan.cuh"

#include <torch/extension.h>

#include <algorithm>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

namespace olmo::bf16_mega_moe {

constexpr int64_t kBlockN = 128;
constexpr int64_t kBf16BlockK = 64;
constexpr int64_t kDispatchThreads = 128;
constexpr int64_t kNonEpilogueThreads = 128;
constexpr int64_t kSm100SmemCapacity = 232448;
constexpr int64_t kSmemAlignment = 1024;
constexpr int64_t kPullThresholdBytes = 4096;

struct LaunchConfig {
  int64_t cluster_size = 2;
  int64_t block_m = 0;
  int64_t block_n = kBlockN;
  int64_t block_k = kBf16BlockK;
  int64_t load_block_m = 0;
  int64_t load_block_n = kBlockN;
  int64_t store_block_m = 0;
  int64_t num_max_pool_tokens = 0;
  int64_t num_max_pool_blocks = 0;
  int64_t workspace_bytes = 0;
  int64_t num_experts_per_wave = 0;
  int64_t num_expert_waves = 0;
  int64_t num_stages = 0;
  int64_t smem_size = 0;
  int64_t runtime_buffer_bytes = 0;
  int64_t total_symmetric_bytes = 0;
  int64_t num_dispatch_threads = kDispatchThreads;
  int64_t num_non_epilogue_threads = kNonEpilogueThreads;
  int64_t num_epilogue_threads = 0;
  int64_t num_bytes_per_pull = 0;
  int64_t grid_sms = 0;
  int64_t num_ranks = 1;
  int64_t num_experts_per_rank = 0;
  megakernel::ForwardPlan forward_plan;
};

inline int64_t ceil_div(int64_t x, int64_t y) {
  TORCH_CHECK(y > 0, "ceil_div denominator must be > 0");
  return (x + y - 1) / y;
}

inline int64_t align(int64_t x, int64_t alignment) {
  TORCH_CHECK(alignment > 0, "alignment must be > 0");
  return ceil_div(x, alignment) * alignment;
}

inline std::tuple<int64_t, int64_t, int64_t> block_m_store_m_epilogue_threads(
    int64_t num_ranks,
    int64_t num_tokens,
    int64_t top_k,
    int64_t num_total_experts) {
  const double expected =
      static_cast<double>(num_tokens) * static_cast<double>(num_ranks) *
      static_cast<double>(top_k) / static_cast<double>(num_total_experts);
  if (expected <= 8.5) {
    return {16, 8, 256};
  }
  if (expected <= 16.5) {
    return {32, 16, 256};
  }
  if (expected <= 32.5) {
    return {64, 32, 128};
  }
  if (expected <= 64.5) {
    return {96, 16, 256};
  }
  if (expected <= 96.5) {
    return {128, 32, 256};
  }
  return {128, 32, 256};
}

inline int64_t num_max_pool_tokens(
    int64_t num_ranks,
    int64_t num_max_tokens_per_rank,
    int64_t top_k,
    int64_t num_experts_per_rank) {
  const int64_t max_recv_tokens = num_ranks * num_max_tokens_per_rank;
  const int64_t max_experts_per_token = std::min(top_k, num_experts_per_rank);
  return align(
      max_recv_tokens * max_experts_per_token +
          num_experts_per_rank *
              (static_cast<int64_t>(layout::kMaxCandidateBlockM) - 1),
      static_cast<int64_t>(layout::kLcmCandidateBlockM));
}

inline int64_t workspace_bytes(
    int64_t num_ranks,
    int64_t num_total_experts,
    int64_t num_experts_per_rank,
    int64_t num_max_tokens_per_rank,
    int64_t top_k,
    int64_t pool_tokens) {
  const int64_t max_recv_tokens_per_expert = num_ranks * num_max_tokens_per_rank;
  const int64_t max_pool_blocks =
      pool_tokens / static_cast<int64_t>(layout::kMinCandidateBlockM);
  int64_t bytes = 0;
  bytes += 32;  // grid/NVLink barrier signal region
  bytes += num_total_experts * static_cast<int64_t>(sizeof(uint64_t)) * 2;
  bytes += num_experts_per_rank * static_cast<int64_t>(sizeof(uint64_t));
  bytes += align(max_pool_blocks, 2) * static_cast<int64_t>(sizeof(uint32_t));
  bytes += max_pool_blocks * static_cast<int64_t>(sizeof(uint64_t));
  bytes += num_experts_per_rank * num_ranks * max_recv_tokens_per_expert *
      static_cast<int64_t>(sizeof(int32_t));
  bytes += pool_tokens * static_cast<int64_t>(sizeof(layout::TokenSrcMetadata));
  (void)top_k;
  return align(bytes, 16);
}

inline int64_t num_experts_per_wave(
    int64_t num_tokens,
    int64_t top_k,
    int64_t intermediate,
    int64_t num_experts_per_rank,
    int64_t block_m,
    int64_t block_n,
    int64_t num_sms) {
  TORCH_CHECK(num_sms > 0, "num_sms must be > 0");
  const double expected =
      static_cast<double>(num_tokens) * static_cast<double>(top_k) /
      static_cast<double>(num_experts_per_rank);
  if (expected < 1.0) {
    return num_experts_per_rank;
  }

  constexpr int64_t imbalance_factor = 2;
  const int64_t expected_ceil = static_cast<int64_t>(std::ceil(expected));
  const int64_t m_blocks = ceil_div(expected_ceil, block_m);
  const int64_t n_blocks = ceil_div(2 * intermediate, block_n);
  const int64_t l1_blocks_per_expert = m_blocks * n_blocks;
  int64_t min_wave = l1_blocks_per_expert > 0
      ? ceil_div(imbalance_factor * num_sms, l1_blocks_per_expert)
      : 1;
  min_wave = std::max<int64_t>(min_wave, 1);
  if (min_wave >= num_experts_per_rank) {
    return num_experts_per_rank;
  }
  if (l1_blocks_per_expert >= num_sms) {
    return min_wave;
  }

  const int64_t max_wave = std::min(num_experts_per_rank, min_wave * 2);
  int64_t best_wave = min_wave;
  double best_tail_ratio = -1.0;
  for (int64_t wave = min_wave; wave <= max_wave; ++wave) {
    const int64_t rem = num_experts_per_rank % wave;
    const double ratio = rem == 0 ? 1.0 : static_cast<double>(rem) / static_cast<double>(wave);
    if (ratio > best_tail_ratio) {
      best_tail_ratio = ratio;
      best_wave = wave;
    }
  }
  return best_wave;
}

inline std::pair<int64_t, int64_t> pipeline_config(
    int64_t num_experts_per_rank,
    int64_t block_m,
    int64_t block_n,
    int64_t block_k,
    int64_t num_bytes_per_pull,
    int64_t store_block_m,
    int64_t num_dispatch_warps,
    int64_t num_epilogue_warps) {
  constexpr int64_t num_epilogue_stages = 2;
  constexpr int64_t num_tma_store_stages = 2;
  const int64_t load_block_m = block_m / 2;
  const int64_t load_block_n = block_n;
  const int64_t epilogue_warpgroups = std::max<int64_t>(num_epilogue_warps / 4, 1);

  const int64_t expert_count_size =
      align(num_experts_per_rank * static_cast<int64_t>(sizeof(uint32_t)), kSmemAlignment);
  const int64_t send_buffers_size =
      align(num_dispatch_warps * num_bytes_per_pull, kSmemAlignment);
  const int64_t dispatch_size = expert_count_size + send_buffers_size;

  const int64_t cd_l1 = epilogue_warpgroups * store_block_m * (block_n / 2) *
      static_cast<int64_t>(sizeof(__nv_bfloat16)) * num_tma_store_stages;
  const int64_t cd_l2 = epilogue_warpgroups * store_block_m * block_n *
      static_cast<int64_t>(sizeof(__nv_bfloat16));
  const int64_t cd_size = align(std::max(cd_l1, cd_l2), kSmemAlignment);

  const int64_t barriers =
      (num_dispatch_warps + num_epilogue_stages * 2 + num_epilogue_warps * 2) *
      static_cast<int64_t>(sizeof(uint64_t));
  const int64_t tmem_ptr = static_cast<int64_t>(sizeof(uint32_t));

  const int64_t a_per_stage = load_block_m * block_k * static_cast<int64_t>(sizeof(__nv_bfloat16));
  const int64_t b_per_stage = load_block_n * block_k * static_cast<int64_t>(sizeof(__nv_bfloat16));
  const int64_t stage_barriers = 2 * static_cast<int64_t>(sizeof(uint64_t));
  const int64_t per_stage = a_per_stage + b_per_stage + stage_barriers;
  const int64_t fixed = dispatch_size + cd_size + barriers + tmem_ptr;
  const int64_t stages = std::max<int64_t>((kSm100SmemCapacity - fixed) / per_stage, 2);
  return {stages, fixed + stages * per_stage};
}

inline LaunchConfig make_launch_config(
    int64_t num_ranks,
    int64_t num_total_experts,
    int64_t num_experts_per_rank,
    int64_t num_max_tokens_per_rank,
    int64_t num_tokens,
    int64_t top_k,
    int64_t hidden,
    int64_t intermediate,
    int64_t num_sms) {
  TORCH_CHECK(num_ranks > 0, "num_ranks must be > 0");
  TORCH_CHECK(num_total_experts > 0, "num_total_experts must be > 0");
  TORCH_CHECK(num_experts_per_rank > 0, "num_experts_per_rank must be > 0");
  TORCH_CHECK(num_total_experts == num_ranks * num_experts_per_rank,
              "num_total_experts must equal num_ranks * num_experts_per_rank");
  TORCH_CHECK(num_tokens > 0, "num_tokens must be > 0");
  TORCH_CHECK(num_max_tokens_per_rank >= num_tokens,
              "num_max_tokens_per_rank must be >= num_tokens");
  TORCH_CHECK(top_k > 0, "top_k must be > 0");
  TORCH_CHECK(hidden > 0 && intermediate > 0, "hidden/intermediate must be > 0");
  TORCH_CHECK(hidden % kBf16BlockK == 0, "hidden must be divisible by BF16 block_k=64");
  TORCH_CHECK(intermediate % kBlockN == 0, "intermediate must be divisible by block_n=128");

  auto [block_m, store_block_m, epilogue_threads] =
      block_m_store_m_epilogue_threads(num_ranks, num_tokens, top_k, num_total_experts);
  int64_t pull_bytes = hidden * static_cast<int64_t>(sizeof(__nv_bfloat16));
  while (pull_bytes > kPullThresholdBytes) {
    TORCH_CHECK(pull_bytes % 2 == 0, "pull byte count must stay divisible by 2");
    pull_bytes /= 2;
  }

  const int64_t experts_per_wave = num_experts_per_wave(
      num_tokens,
      top_k,
      intermediate,
      num_experts_per_rank,
      block_m,
      kBlockN,
      num_sms);
  const int64_t pool_tokens = num_max_pool_tokens(
      num_ranks,
      num_max_tokens_per_rank,
      top_k,
      num_experts_per_rank);
  const auto [num_stages, smem_size] = pipeline_config(
      num_experts_per_rank,
      block_m,
      kBlockN,
      kBf16BlockK,
      pull_bytes,
      store_block_m,
      kDispatchThreads / 32,
      epilogue_threads / 32);

  LaunchConfig cfg;
  cfg.block_m = block_m;
  cfg.block_k = kBf16BlockK;
  cfg.load_block_m = block_m / 2;
  cfg.store_block_m = store_block_m;
  cfg.num_max_pool_tokens = pool_tokens;
  cfg.num_max_pool_blocks =
      pool_tokens / static_cast<int64_t>(layout::kMinCandidateBlockM);
  cfg.workspace_bytes = workspace_bytes(
      num_ranks,
      num_total_experts,
      num_experts_per_rank,
      num_max_tokens_per_rank,
      top_k,
      pool_tokens);
  cfg.runtime_buffer_bytes = static_cast<int64_t>(buffers::bf16_forward_buffer_bytes(
      static_cast<uint32_t>(num_max_tokens_per_rank),
      static_cast<uint32_t>(top_k),
      static_cast<uint32_t>(hidden),
      static_cast<uint32_t>(intermediate),
      static_cast<uint32_t>(pool_tokens)));
  cfg.total_symmetric_bytes = cfg.workspace_bytes + cfg.runtime_buffer_bytes;
  cfg.num_experts_per_wave = experts_per_wave;
  cfg.num_expert_waves = ceil_div(num_experts_per_rank, experts_per_wave);
  cfg.num_stages = num_stages;
  cfg.smem_size = smem_size;
  cfg.num_epilogue_threads = epilogue_threads;
  cfg.num_bytes_per_pull = pull_bytes;
  cfg.grid_sms = num_sms;
  cfg.num_ranks = num_ranks;
  cfg.num_experts_per_rank = num_experts_per_rank;
  cfg.forward_plan = megakernel::make_forward_plan(
      num_ranks,
      num_total_experts,
      num_experts_per_rank,
      num_tokens,
      top_k,
      hidden,
      intermediate,
      block_m,
      kBlockN,
      num_sms);
  return cfg;
}

inline std::vector<int64_t> to_vector(const LaunchConfig& cfg) {
  // Preserve the first ten fields from the original fail-closed wrapper, then
  // append the richer MegaMoE runtime layout/config fields.
  return {
      cfg.block_m,
      cfg.block_n,
      cfg.block_k,
      cfg.store_block_m,
      cfg.num_experts_per_wave,
      cfg.num_expert_waves,
      cfg.num_dispatch_threads,
      cfg.num_non_epilogue_threads,
      cfg.num_epilogue_threads,
      cfg.num_bytes_per_pull,
      cfg.cluster_size,
      cfg.load_block_m,
      cfg.load_block_n,
      cfg.num_max_pool_tokens,
      cfg.num_max_pool_blocks,
      cfg.workspace_bytes,
      cfg.runtime_buffer_bytes,
      cfg.total_symmetric_bytes,
      cfg.num_stages,
      cfg.smem_size,
      cfg.grid_sms,
      cfg.num_ranks,
      cfg.num_experts_per_rank,
      cfg.forward_plan.f1_dispatch_sms,
      cfg.forward_plan.f1_finalize_sms,
      cfg.forward_plan.f1_gemm_sms,
      cfg.forward_plan.f1_expected_tokens_per_expert,
      cfg.forward_plan.f1_gemm_m_tiles_per_expert,
      cfg.forward_plan.f1_gemm_n_tiles,
      cfg.forward_plan.f1_dispatch_route_tasks,
      cfg.forward_plan.f1_finalize_expert_tasks,
      cfg.forward_plan.f1_gemm_tasks,
      cfg.forward_plan.f1_total_tasks,
      cfg.forward_plan.f2_combine_sms,
      cfg.forward_plan.f2_reduce_sms,
      cfg.forward_plan.f2_gemm_sms,
      cfg.forward_plan.f2_expected_tokens_per_expert,
      cfg.forward_plan.f2_gemm_m_tiles_per_expert,
      cfg.forward_plan.f2_gemm_n_tiles,
      cfg.forward_plan.f2_combine_scatter_tasks,
      cfg.forward_plan.f2_combine_reduce_tasks,
      cfg.forward_plan.f2_gemm_tasks,
      cfg.forward_plan.f2_total_tasks,
  };
}

}  // namespace olmo::bf16_mega_moe

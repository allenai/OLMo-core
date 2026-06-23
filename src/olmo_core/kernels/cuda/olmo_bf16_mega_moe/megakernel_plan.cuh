/*
 * OLMo BF16 MegaMoE megakernel task plan.
 *
 * The training target is split into two forward megakernels:
 *
 *   F1: route dispatch plus W1/up-gate GroupGEMM
 *   F2: W2/down GroupGEMM plus weighted combine
 *
 * This header owns the launch-time task counts and SM role split for that
 * shape. The CUDA bodies will consume this metadata directly; there is no
 * Python-level wave scheduler in the hot path.
 */
#pragma once

#include "layout.cuh"

#include <algorithm>
#include <cstdint>

namespace olmo::bf16_mega_moe::megakernel {

constexpr int64_t kRoutesPerDispatchTask = 256;
constexpr int64_t kRoutesPerCombineTask = 256;
constexpr int64_t kTokensPerReduceTask = 256;

enum class ForwardKernelKind : int64_t {
  DispatchGemm = 1,
  GemmCombine = 2,
};

enum class ForwardTaskKind : int64_t {
  None = 0,
  DispatchRoute = 1,
  DispatchFinalizeExpert = 2,
  W1GemmTile = 3,
  W2GemmTile = 4,
  CombineScatter = 5,
  CombineReduce = 6,
};

struct ForwardPlan {
  int64_t f1_dispatch_sms = 0;
  int64_t f1_finalize_sms = 0;
  int64_t f1_gemm_sms = 0;
  int64_t f1_expected_tokens_per_expert = 0;
  int64_t f1_gemm_m_tiles_per_expert = 0;
  int64_t f1_gemm_n_tiles = 0;
  int64_t f1_dispatch_route_tasks = 0;
  int64_t f1_finalize_expert_tasks = 0;
  int64_t f1_gemm_tasks = 0;
  int64_t f1_total_tasks = 0;

  int64_t f2_combine_sms = 0;
  int64_t f2_reduce_sms = 0;
  int64_t f2_gemm_sms = 0;
  int64_t f2_expected_tokens_per_expert = 0;
  int64_t f2_gemm_m_tiles_per_expert = 0;
  int64_t f2_gemm_n_tiles = 0;
  int64_t f2_combine_scatter_tasks = 0;
  int64_t f2_combine_reduce_tasks = 0;
  int64_t f2_gemm_tasks = 0;
  int64_t f2_total_tasks = 0;
};

struct ForwardTask {
  ForwardTaskKind kind = ForwardTaskKind::None;
  int64_t ordinal = -1;
  int64_t local_expert = -1;
  int64_t m_tile = -1;
  int64_t n_tile = -1;
  int64_t route_task = -1;
};

inline int64_t ceil_div_i64(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

inline int64_t clamp_i64(int64_t value, int64_t low, int64_t high) {
  return std::min(std::max(value, low), high);
}

inline int64_t primary_role_sms(int64_t num_sms) {
  if (num_sms <= 1) {
    return 1;
  }
  const int64_t max_role = std::max<int64_t>(1, std::min<int64_t>(40, num_sms - 1));
  const int64_t min_role = std::min<int64_t>(8, max_role);
  return clamp_i64(ceil_div_i64(num_sms, 4), min_role, max_role);
}

inline int64_t secondary_role_sms(int64_t num_sms) {
  if (num_sms <= 4) {
    return 1;
  }
  return std::min<int64_t>(8, std::max<int64_t>(1, ceil_div_i64(num_sms, 16)));
}

inline int64_t remaining_gemm_sms(int64_t num_sms, int64_t first_role_sms, int64_t second_role_sms) {
  return std::max<int64_t>(1, num_sms - first_role_sms - second_role_sms);
}

inline ForwardPlan make_forward_plan(
    int64_t num_ranks,
    int64_t num_total_experts,
    int64_t num_experts_per_rank,
    int64_t num_tokens,
    int64_t top_k,
    int64_t hidden,
    int64_t intermediate,
    int64_t block_m,
    int64_t block_n,
    int64_t num_sms) {
  const int64_t local_route_tasks =
      ceil_div_i64(num_tokens * top_k, kRoutesPerDispatchTask);
  const int64_t total_routes = num_tokens * num_ranks * top_k;
  const int64_t expected_tokens_per_expert = ceil_div_i64(total_routes, num_total_experts);
  const int64_t m_tiles_per_expert = ceil_div_i64(expected_tokens_per_expert, block_m);
  const int64_t f1_n_tiles = ceil_div_i64(2 * intermediate, block_n);
  const int64_t f2_n_tiles = ceil_div_i64(hidden, block_n);
  const int64_t f1_gemm_tasks =
      num_experts_per_rank * m_tiles_per_expert * f1_n_tiles;
  const int64_t f2_gemm_tasks =
      num_experts_per_rank * m_tiles_per_expert * f2_n_tiles;

  ForwardPlan plan;
  plan.f1_dispatch_sms = primary_role_sms(num_sms);
  plan.f1_finalize_sms = secondary_role_sms(num_sms);
  plan.f1_gemm_sms = remaining_gemm_sms(num_sms, plan.f1_dispatch_sms, plan.f1_finalize_sms);
  plan.f1_expected_tokens_per_expert = expected_tokens_per_expert;
  plan.f1_gemm_m_tiles_per_expert = m_tiles_per_expert;
  plan.f1_gemm_n_tiles = f1_n_tiles;
  plan.f1_dispatch_route_tasks = local_route_tasks;
  plan.f1_finalize_expert_tasks = num_experts_per_rank;
  plan.f1_gemm_tasks = f1_gemm_tasks;
  plan.f1_total_tasks =
      plan.f1_dispatch_route_tasks + plan.f1_finalize_expert_tasks + plan.f1_gemm_tasks;

  plan.f2_combine_sms = primary_role_sms(num_sms);
  plan.f2_reduce_sms = top_k > 1 ? secondary_role_sms(num_sms) : 0;
  plan.f2_gemm_sms = remaining_gemm_sms(num_sms, plan.f2_combine_sms, plan.f2_reduce_sms);
  plan.f2_expected_tokens_per_expert = expected_tokens_per_expert;
  plan.f2_gemm_m_tiles_per_expert = m_tiles_per_expert;
  plan.f2_gemm_n_tiles = f2_n_tiles;
  plan.f2_combine_scatter_tasks =
      ceil_div_i64(num_tokens * top_k, kRoutesPerCombineTask);
  plan.f2_combine_reduce_tasks =
      top_k > 1 ? ceil_div_i64(num_tokens, kTokensPerReduceTask) : 0;
  plan.f2_gemm_tasks = f2_gemm_tasks;
  plan.f2_total_tasks =
      plan.f2_gemm_tasks + plan.f2_combine_scatter_tasks + plan.f2_combine_reduce_tasks;
  return plan;
}

OLMO_BF16_MEGA_HOST_DEVICE inline ForwardTask none_task() {
  return {};
}

OLMO_BF16_MEGA_HOST_DEVICE inline ForwardTask dispatch_route_task(int64_t task_idx) {
  ForwardTask task;
  task.kind = ForwardTaskKind::DispatchRoute;
  task.ordinal = task_idx;
  task.route_task = task_idx;
  return task;
}

OLMO_BF16_MEGA_HOST_DEVICE inline ForwardTask finalize_expert_task(int64_t task_idx) {
  ForwardTask task;
  task.kind = ForwardTaskKind::DispatchFinalizeExpert;
  task.ordinal = task_idx;
  task.local_expert = task_idx;
  return task;
}

OLMO_BF16_MEGA_HOST_DEVICE inline ForwardTask gemm_task(
    ForwardTaskKind kind,
    int64_t task_idx,
    int64_t m_tiles_per_expert,
    int64_t n_tiles) {
  ForwardTask task;
  task.kind = kind;
  task.ordinal = task_idx;
  task.n_tile = task_idx % n_tiles;
  int64_t remaining = task_idx / n_tiles;
  task.m_tile = remaining % m_tiles_per_expert;
  task.local_expert = remaining / m_tiles_per_expert;
  return task;
}

OLMO_BF16_MEGA_HOST_DEVICE inline ForwardTask combine_scatter_task(int64_t task_idx) {
  ForwardTask task;
  task.kind = ForwardTaskKind::CombineScatter;
  task.ordinal = task_idx;
  task.route_task = task_idx;
  return task;
}

OLMO_BF16_MEGA_HOST_DEVICE inline ForwardTask combine_reduce_task(int64_t task_idx) {
  ForwardTask task;
  task.kind = ForwardTaskKind::CombineReduce;
  task.ordinal = task_idx;
  task.route_task = task_idx;
  return task;
}

OLMO_BF16_MEGA_HOST_DEVICE inline ForwardTask decode_f1_task(
    const ForwardPlan& plan,
    int64_t task_idx) {
  if (task_idx < 0 || task_idx >= plan.f1_total_tasks) {
    return none_task();
  }

  if (task_idx < plan.f1_dispatch_route_tasks) {
    return dispatch_route_task(task_idx);
  }
  task_idx -= plan.f1_dispatch_route_tasks;

  if (task_idx < plan.f1_finalize_expert_tasks) {
    return finalize_expert_task(task_idx);
  }
  task_idx -= plan.f1_finalize_expert_tasks;

  return gemm_task(
      ForwardTaskKind::W1GemmTile,
      task_idx,
      plan.f1_gemm_m_tiles_per_expert,
      plan.f1_gemm_n_tiles);
}

OLMO_BF16_MEGA_HOST_DEVICE inline ForwardTask decode_f2_task(
    const ForwardPlan& plan,
    int64_t task_idx) {
  if (task_idx < 0 || task_idx >= plan.f2_total_tasks) {
    return none_task();
  }

  if (task_idx < plan.f2_gemm_tasks) {
    return gemm_task(
        ForwardTaskKind::W2GemmTile,
        task_idx,
        plan.f2_gemm_m_tiles_per_expert,
        plan.f2_gemm_n_tiles);
  }
  task_idx -= plan.f2_gemm_tasks;

  if (task_idx < plan.f2_combine_scatter_tasks) {
    return combine_scatter_task(task_idx);
  }
  task_idx -= plan.f2_combine_scatter_tasks;

  if (task_idx < plan.f2_combine_reduce_tasks) {
    return combine_reduce_task(task_idx);
  }
  return none_task();
}

}  // namespace olmo::bf16_mega_moe::megakernel

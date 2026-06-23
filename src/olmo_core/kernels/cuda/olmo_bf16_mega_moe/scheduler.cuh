/*
 * OLMo BF16 MegaMoE wave scheduler.
 *
 * This mirrors DeepGEMM MegaMoE's persistent-kernel scheduling model: local
 * experts are processed in waves, each wave runs all Linear1 CTA tiles before
 * switching to Linear2 CTA tiles, and every CTA advances by grid-stride block
 * indices. The code is OLMo-owned and uses only CUDA intrinsics plus the OLMo
 * layout contract.
 */
#pragma once

#include "layout.cuh"

#include <cstdint>

namespace olmo::bf16_mega_moe::sched {

template <
    uint32_t BLOCK_M,
    uint32_t BLOCK_N,
    uint32_t BLOCK_K,
    uint32_t L1_SHAPE_N,
    uint32_t L1_SHAPE_K,
    uint32_t L2_SHAPE_N,
    uint32_t L2_SHAPE_K,
    uint32_t kNumExpertsPerRank,
    uint32_t kNumExpertsPerWave,
    uint32_t kNumSMs,
    uint32_t kNumRanks,
    uint32_t kNumExpertsPerLane = layout::math::ceil_div(kNumExpertsPerRank, 32u),
    uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N,
    uint32_t kNumL2BlockNs = L2_SHAPE_N / BLOCK_N,
    uint32_t kNumL1BlockKs = L1_SHAPE_K / BLOCK_K,
    uint32_t kNumL2BlockKs = L2_SHAPE_K / BLOCK_K>
struct MegaMoEScheduler {
  static_assert(L1_SHAPE_N % BLOCK_N == 0, "invalid L1 N tile shape");
  static_assert(L2_SHAPE_N % BLOCK_N == 0, "invalid L2 N tile shape");
  static_assert(L1_SHAPE_K % BLOCK_K == 0, "invalid L1 K tile shape");
  static_assert(L2_SHAPE_K % BLOCK_K == 0, "invalid L2 K tile shape");
  static_assert(kNumExpertsPerWave > 0, "invalid zero experts per wave");
  static_assert(kNumExpertsPerWave <= kNumExpertsPerRank, "invalid experts per wave");
  static_assert(kNumSMs % 2 == 0, "2-CTA cluster scheduling needs even SM count");
  static_assert(kNumL1BlockNs % 2 == 0, "L1 N blocks must be even for 2-CTA clusters");
  static_assert(kNumL2BlockNs % 2 == 0, "L2 N blocks must be even for 2-CTA clusters");

  const layout::Workspace workspace;

  layout::BlockPhase next_phase = layout::BlockPhase::Linear1;
  uint32_t current_local_expert_idx = 0;
  uint32_t current_num_tokens = 0;
  uint32_t current_pool_block_offset = 0;
  uint32_t block_idx = 0;
  uint32_t m_block_idx = 0;
  uint32_t n_block_idx = 0;
  uint32_t stored_num_tokens_per_expert[kNumExpertsPerLane] = {};

  OLMO_BF16_MEGA_DEVICE explicit MegaMoEScheduler(layout::Workspace workspace_)
      : workspace(workspace_) {
    block_idx = blockIdx.x;
  }

  OLMO_BF16_MEGA_DEVICE uint32_t lane_idx() const {
    return static_cast<uint32_t>(threadIdx.x) & 31u;
  }

  OLMO_BF16_MEGA_DEVICE uint32_t wave_expert_end_idx() const {
    const uint32_t aligned =
        layout::math::align(current_local_expert_idx + 1u, kNumExpertsPerWave);
    return layout::math::min(aligned, kNumExpertsPerRank);
  }

  OLMO_BF16_MEGA_DEVICE uint32_t num_tokens(uint32_t expert_idx) const {
    uint32_t value = 0;
#pragma unroll
    for (uint32_t i = 0; i < kNumExpertsPerLane; ++i) {
      if (expert_idx == i * 32u + lane_idx()) {
        value = stored_num_tokens_per_expert[i];
      }
    }
    return __shfl_sync(0xffffffffu, value, expert_idx % 32u);
  }

  OLMO_BF16_MEGA_DEVICE uint32_t pool_block_offset(uint32_t expert_idx) {
    uint32_t blocks = 0;
#pragma unroll
    for (uint32_t i = 0; i < kNumExpertsPerLane; ++i) {
      if (i * 32u + lane_idx() < expert_idx) {
        blocks += layout::math::ceil_div(stored_num_tokens_per_expert[i], BLOCK_M);
      }
    }
    return __reduce_add_sync(0xffffffffu, blocks);
  }

  OLMO_BF16_MEGA_DEVICE uint32_t current_num_m_blocks() const {
    return layout::math::ceil_div(current_num_tokens, BLOCK_M);
  }

  OLMO_BF16_MEGA_DEVICE uint32_t current_pool_block_offset_value() const {
    return current_pool_block_offset;
  }

  OLMO_BF16_MEGA_DEVICE uint32_t valid_m(bool align_to_umma = false) const {
    const uint32_t remaining = current_num_tokens - m_block_idx * BLOCK_M;
    const uint32_t m = layout::math::min(remaining, BLOCK_M);
    return align_to_umma ? layout::math::align(m, 16u) : m;
  }

  OLMO_BF16_MEGA_DEVICE void advance_expert_idx() {
    current_pool_block_offset += current_num_m_blocks();
    current_local_expert_idx += 1;
    current_num_tokens = num_tokens(current_local_expert_idx);
  }

  OLMO_BF16_MEGA_DEVICE void set_expert_idx(uint32_t expert_idx) {
    current_local_expert_idx = expert_idx;
    current_num_tokens = num_tokens(expert_idx);
    current_pool_block_offset = pool_block_offset(expert_idx);
  }

  OLMO_BF16_MEGA_DEVICE bool fetch_next_l1_block() {
    const uint32_t wave_end = wave_expert_end_idx();
    while (current_local_expert_idx < wave_end) {
      const uint32_t m_blocks = current_num_m_blocks();
      m_block_idx = block_idx / kNumL1BlockNs;
      if (m_block_idx < m_blocks) {
        return true;
      }
      block_idx -= m_blocks * kNumL1BlockNs;
      advance_expert_idx();
    }
    return false;
  }

  OLMO_BF16_MEGA_DEVICE bool fetch_next_l2_block() {
    const uint32_t wave_end = wave_expert_end_idx();
    while (current_local_expert_idx < wave_end) {
      const uint32_t m_blocks = current_num_m_blocks();
      if (block_idx < m_blocks * kNumL2BlockNs) {
        m_block_idx = block_idx / kNumL2BlockNs;
        return true;
      }
      block_idx -= m_blocks * kNumL2BlockNs;
      advance_expert_idx();
    }
    return false;
  }

  OLMO_BF16_MEGA_DEVICE layout::BlockPhase get_next_block(
      uint32_t* local_expert_idx,
      uint32_t* k_blocks,
      uint32_t* m_idx,
      uint32_t* n_idx) {
    while (true) {
      if (current_local_expert_idx >= kNumExpertsPerRank) {
        break;
      }

      if (next_phase == layout::BlockPhase::Linear1) {
        if (fetch_next_l1_block()) {
          n_block_idx = block_idx - m_block_idx * kNumL1BlockNs;
          block_idx += kNumSMs;
          *local_expert_idx = current_local_expert_idx;
          *k_blocks = kNumL1BlockKs;
          *m_idx = m_block_idx;
          *n_idx = n_block_idx;
          return layout::BlockPhase::Linear1;
        }
        next_phase = layout::BlockPhase::Linear2;
        set_expert_idx(((current_local_expert_idx - 1u) / kNumExpertsPerWave) * kNumExpertsPerWave);
      } else {
        if (fetch_next_l2_block()) {
          n_block_idx = block_idx - m_block_idx * kNumL2BlockNs;
          block_idx += kNumSMs;
          *local_expert_idx = current_local_expert_idx;
          *k_blocks = kNumL2BlockKs;
          *m_idx = m_block_idx;
          *n_idx = n_block_idx;
          return layout::BlockPhase::Linear2;
        }
        next_phase = layout::BlockPhase::Linear1;
      }
    }

    *local_expert_idx = 0;
    *k_blocks = 0;
    *m_idx = 0;
    *n_idx = 0;
    return layout::BlockPhase::None;
  }

  OLMO_BF16_MEGA_DEVICE void fetch_expert_recv_count() {
#pragma unroll
    for (uint32_t i = 0; i < kNumExpertsPerLane; ++i) {
      const uint32_t expert_idx = i * 32u + lane_idx();
      uint64_t value = 0;
      if (expert_idx < kNumExpertsPerRank) {
        do {
          value = *reinterpret_cast<volatile uint64_t*>(
              workspace.expert_recv_count_sum_ptr(expert_idx));
        } while (static_cast<uint32_t>(value >> 32) != kNumSMs * kNumRanks);
      }
      stored_num_tokens_per_expert[i] = static_cast<uint32_t>(value);
    }
    __syncwarp();
  }

  template <typename Func>
  OLMO_BF16_MEGA_DEVICE void for_each_block(Func&& func) {
    fetch_expert_recv_count();
    set_expert_idx(0);

    while (true) {
      uint32_t local_expert_idx = 0;
      uint32_t k_blocks = 0;
      uint32_t m_idx = 0;
      uint32_t n_idx = 0;
      const layout::BlockPhase phase =
          get_next_block(&local_expert_idx, &k_blocks, &m_idx, &n_idx);
      if (phase == layout::BlockPhase::None) {
        break;
      }
      func(phase, local_expert_idx, k_blocks, m_idx, n_idx);
    }
  }
};

}  // namespace olmo::bf16_mega_moe::sched

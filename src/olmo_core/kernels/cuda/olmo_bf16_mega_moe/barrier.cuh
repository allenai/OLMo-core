/*
 * OLMo BF16 MegaMoE device barriers.
 *
 * Persistent MegaMoE needs a cheap same-grid barrier and a cross-rank signal
 * barrier around dispatch/pull/combine phases. This file keeps those mechanics
 * OLMo-owned and parameterized by an address map, so the transport can evolve
 * from local testing to a lower-level RDMA/IBGDA mapping without changing the
 * scheduler or dispatch metadata.
 */
#pragma once

#include "layout.cuh"

#include <cstdint>
#include <cstdio>

namespace olmo::bf16_mega_moe::barrier {

OLMO_BF16_MEGA_DEVICE inline uint32_t atomic_add_release_u32(uint32_t* ptr, uint32_t value) {
  __threadfence();
  return atomicAdd(ptr, value);
}

OLMO_BF16_MEGA_DEVICE inline void atomic_add_release_system_i32(int* ptr, int value) {
  __threadfence_system();
  atomicAdd(ptr, value);
}

OLMO_BF16_MEGA_DEVICE inline uint32_t load_acquire_u32(const uint32_t* ptr) {
  const uint32_t value = *reinterpret_cast<const volatile uint32_t*>(ptr);
  __threadfence();
  return value;
}

OLMO_BF16_MEGA_DEVICE inline int load_acquire_system_i32(const int* ptr) {
  const int value = *reinterpret_cast<const volatile int*>(ptr);
  __threadfence_system();
  return value;
}

template <uint32_t kNumSMs, uint32_t kGridSyncIndex = 0, typename SyncScope>
OLMO_BF16_MEGA_DEVICE void grid_sync(
    const layout::Workspace& workspace,
    uint32_t sm_idx,
    uint32_t thread_idx,
    SyncScope sync_scope) {
  static_assert(kNumSMs > 0, "grid sync needs a nonzero SM count");
  static constexpr uint32_t kFinishSumTag = 0x80000000u;

  sync_scope();
  if (thread_idx == 0) {
    uint32_t* count_ptr = workspace.grid_sync_count_ptr(kGridSyncIndex);
    const uint32_t increment = sm_idx == 0 ? (kFinishSumTag - (kNumSMs - 1)) : 1u;
    const uint32_t old_value = atomic_add_release_u32(count_ptr, increment);
    uint32_t new_value = 0;
    do {
      new_value = load_acquire_u32(count_ptr);
    } while (((new_value ^ old_value) & kFinishSumTag) == 0);
  }
  sync_scope();
}

template <
    uint32_t kNumRanks,
    uint32_t kNumSMs,
    uint32_t kNumThreads,
    uint32_t kGridSyncIndex,
    uint32_t kTag,
    typename AddressMap,
    typename SyncScope>
OLMO_BF16_MEGA_DEVICE void cross_rank_barrier(
    const layout::Workspace& workspace,
    AddressMap address_map,
    uint32_t sm_idx,
    uint32_t thread_idx,
    SyncScope sync_scope,
    bool sync_prologue = true,
    bool sync_epilogue = true) {
  static_assert(kNumRanks > 0, "cross-rank barrier needs at least one rank");
  static_assert(kNumThreads >= kNumRanks, "barrier needs at least one thread per rank signal");

  if (sync_prologue) {
    grid_sync<kNumSMs, kGridSyncIndex>(workspace, sm_idx, thread_idx, sync_scope);
  }

  // Every thread may have written peer-visible workspace data before this
  // phase barrier. Fence from every thread, then reconverge the CTA before the
  // rank-level signal is published by SM0.
  __threadfence_system();
  sync_scope();

  if (sm_idx == 0) {
    uint32_t* counter_ptr = workspace.nvl_barrier_counter_ptr();
    const uint32_t status = *reinterpret_cast<volatile uint32_t*>(counter_ptr) & 3u;
    const uint32_t signal_phase = status & 1u;
    const uint32_t signal_sign = status >> 1u;
    int* signal_ptr = workspace.nvl_barrier_signal_ptr(signal_phase);

    if (thread_idx < kNumRanks) {
      int* remote_signal = address_map.map(signal_ptr, thread_idx);
      atomic_add_release_system_i32(remote_signal, signal_sign ? -1 : 1);
    }
    sync_scope();

    if (thread_idx == 0) {
      atomicAdd(counter_ptr, 1u);
      const int target = signal_sign ? 0 : static_cast<int>(kNumRanks);
      const uint64_t start_clock = clock64();
      constexpr uint64_t kTimeoutCycles = 300ull * 2000000000ull;
      while (load_acquire_system_i32(signal_ptr) != target) {
        if (clock64() - start_clock >= kTimeoutCycles) {
          printf(
              "OLMo BF16 MegaMoE cross-rank barrier timeout: rank=%u counter=%u signal=%d target=%d phase=%u sign=%u tag=%u\n",
              address_map.rank_idx,
              *counter_ptr,
              load_acquire_system_i32(signal_ptr),
              target,
              signal_phase,
              signal_sign,
              kTag);
          asm volatile("trap;");
        }
      }
    }
  }

  if (sync_epilogue) {
    grid_sync<kNumSMs, kGridSyncIndex>(workspace, sm_idx, thread_idx, sync_scope);
  }
}

}  // namespace olmo::bf16_mega_moe::barrier

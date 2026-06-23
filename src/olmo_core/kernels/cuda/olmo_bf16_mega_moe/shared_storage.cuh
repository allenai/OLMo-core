/*
 * OLMo BF16 MegaMoE shared-memory contract.
 *
 * The persistent kernel divides warps into dispatch, TMA load, MMA issue, and
 * epilogue roles. This storage mirrors that role split in one statically
 * described shared-memory object, keeping the BF16 path close to the final
 * fused-kernel shape instead of staging work through Python.
 */
#pragma once

#include "layout.cuh"
#include "ptx.cuh"

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>

namespace olmo::bf16_mega_moe::shared {

template <
    uint32_t kNumExperts,
    uint32_t kBlockM,
    uint32_t kBlockN,
    uint32_t kBlockK,
    uint32_t kLoadBlockM,
    uint32_t kStoreBlockM,
    uint32_t kNumStages,
    uint32_t kNumBytesPerPull,
    uint32_t kNumDispatchWarps,
    uint32_t kNumEpilogueWarps>
struct SharedStorage {
  static_assert(kNumExperts > 0, "expert count must be positive");
  static_assert(kBlockM % 16 == 0, "block_m must be UMMA aligned");
  static_assert(kBlockN == 128, "BF16 MegaMoE stores 128-wide output tiles");
  static_assert(kBlockK == 64, "BF16 MegaMoE starts with 64-wide K tiles");
  static_assert(kLoadBlockM * 2 == kBlockM, "2-CTA multicast expects load_block_m=block_m/2");
  static_assert(kStoreBlockM > 0 && kBlockM % kStoreBlockM == 0,
                "store_block_m must divide block_m");
  static_assert(kNumStages >= 2, "persistent MMA pipeline needs at least two stages");
  static_assert(kNumDispatchWarps > 0, "dispatch warps must be positive");
  static_assert(kNumEpilogueWarps % 4 == 0, "epilogue warps must form full warpgroups");

  static constexpr uint32_t kSharedMemoryAlignment = 1024;
  static constexpr uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4;
  static constexpr uint32_t kNumEpilogueStages = 2;
  static constexpr uint32_t kNumTmaStoreStages = 2;
  static constexpr uint32_t kL1OutBlockN = kBlockN / 2;

  alignas(kSharedMemoryAlignment) uint32_t expert_token_count[kNumExperts];
  alignas(kSharedMemoryAlignment) uint8_t dispatch_send_buffer[kNumDispatchWarps][kNumBytesPerPull];

  union {
    alignas(kSharedMemoryAlignment)
        __nv_bfloat16 l1[kNumEpilogueWarpgroups][kNumTmaStoreStages][kStoreBlockM * kL1OutBlockN];
    alignas(kSharedMemoryAlignment)
        __nv_bfloat16 l2[kNumEpilogueWarpgroups][kStoreBlockM * kBlockN];
  } smem_d;

  alignas(kSharedMemoryAlignment)
      __nv_bfloat16 smem_a[kNumStages][kLoadBlockM * kBlockK];
  alignas(kSharedMemoryAlignment)
      __nv_bfloat16 smem_b[kNumStages][kBlockN * kBlockK];

  ptx::MBarrier dispatch_barriers[kNumDispatchWarps];
  ptx::MBarrier full_barriers[kNumStages];
  ptx::MBarrier empty_barriers[kNumStages];
  ptx::MBarrier tmem_full_barriers[kNumEpilogueStages];
  ptx::MBarrier tmem_empty_barriers[kNumEpilogueStages];
  ptx::MBarrier combine_barriers[kNumEpilogueWarps * 2];
  uint32_t tmem_ptr_in_smem;

  static constexpr uint32_t reusable_smem_bytes() {
    return static_cast<uint32_t>(offsetof(SharedStorage, dispatch_barriers));
  }
};

}  // namespace olmo::bf16_mega_moe::shared

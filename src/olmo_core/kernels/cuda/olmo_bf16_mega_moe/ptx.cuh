/*
 * OLMo BF16 MegaMoE PTX helpers.
 *
 * OLMo-owned wrappers for the small set of SM100 primitives needed by the
 * persistent BF16 MegaMoE mainloop. These intentionally avoid DeepGEMM,
 * CUTLASS, and CUTE dependencies; they expose raw CUDA/PTX building blocks
 * that the fused forward kernel can compose.
 */
#pragma once

#include "layout.cuh"

#include <cuda.h>

#include <cstdint>

namespace olmo::bf16_mega_moe::ptx {

#if defined(__CUDA_ARCH__) && \
    (defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM101_ALL) || \
     defined(__CUDA_ARCH_FEAT_SM103_ALL))
#define OLMO_BF16_MEGA_HAS_TCGEN05_TMEM 1
#else
#define OLMO_BF16_MEGA_HAS_TCGEN05_TMEM 0
#endif

struct alignas(8) MBarrier {
  uint64_t state;
};

OLMO_BF16_MEGA_DEVICE inline uint32_t smem_addr(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

OLMO_BF16_MEGA_DEVICE inline void mbarrier_init(MBarrier* barrier, uint32_t expected_count) {
  asm volatile(
      "mbarrier.init.shared.b64 [%0], %1;"
      :
      : "r"(smem_addr(barrier)), "r"(expected_count)
      : "memory");
}

OLMO_BF16_MEGA_DEVICE inline void mbarrier_arrive(MBarrier* barrier) {
  asm volatile(
      "mbarrier.arrive.shared::cta.b64 _, [%0];"
      :
      : "r"(smem_addr(barrier))
      : "memory");
}

OLMO_BF16_MEGA_DEVICE inline void mbarrier_arrive_expect_tx(
    MBarrier* barrier,
    uint32_t num_bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0;"
      :
      : "r"(num_bytes), "r"(smem_addr(barrier))
      : "memory");
}

OLMO_BF16_MEGA_DEVICE inline void mbarrier_wait_parity(
    MBarrier* barrier,
    uint32_t phase) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "L_wait_%=:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2;\n\t"
      "@p bra L_done_%=;\n\t"
      "bra L_wait_%=;\n\t"
      "L_done_%=:\n\t"
      "}\n"
      :
      : "r"(smem_addr(barrier)), "r"(phase), "r"(0x989680)
      : "memory");
}

OLMO_BF16_MEGA_DEVICE inline void tma_load_1d(
    void* dst_smem,
    const void* src_global,
    MBarrier* barrier,
    uint32_t num_bytes) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];"
      :
      : "r"(smem_addr(dst_smem)),
        "l"(src_global),
        "r"(num_bytes),
        "r"(smem_addr(barrier))
      : "memory");
}

template <int kCtaGroup = 1>
OLMO_BF16_MEGA_DEVICE inline void tma_load_2d(
    void* dst_smem,
    const CUtensorMap* tensor_map,
    int32_t coord0,
    int32_t coord1,
    MBarrier* barrier) {
  static_assert(kCtaGroup == 1 || kCtaGroup == 2, "TMA CTA group must be 1 or 2");
  if constexpr (kCtaGroup == 1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
        ".cta_group::1 [%0], [%1, {%2, %3}], [%4];"
        :
        : "r"(smem_addr(dst_smem)),
          "l"(tensor_map),
          "r"(coord0),
          "r"(coord1),
          "r"(smem_addr(barrier))
        : "memory");
  } else {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
        ".cta_group::2 [%0], [%1, {%2, %3}], [%4];"
        :
        : "r"(smem_addr(dst_smem)),
          "l"(tensor_map),
          "r"(coord0),
          "r"(coord1),
          "r"(smem_addr(barrier))
        : "memory");
  }
}

OLMO_BF16_MEGA_DEVICE inline void tma_store_1d(
    void* dst_global,
    const void* src_smem,
    uint32_t num_bytes) {
  asm volatile(
      "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
      :
      : "l"(dst_global), "r"(smem_addr(src_smem)), "r"(num_bytes)
      : "memory");
}

template <int kNumRemainingWaits = 0>
OLMO_BF16_MEGA_DEVICE inline void tma_store_wait() {
  asm volatile("cp.async.bulk.wait_group %0;" ::"n"(kNumRemainingWaits) : "memory");
}

OLMO_BF16_MEGA_DEVICE inline void tcgen05_before_thread_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

OLMO_BF16_MEGA_DEVICE inline void tcgen05_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

OLMO_BF16_MEGA_DEVICE inline void tcgen05_wait_tmem_load() {
#if OLMO_BF16_MEGA_HAS_TCGEN05_TMEM
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#endif
}

OLMO_BF16_MEGA_DEVICE inline uint32_t tcgen05_tmem_feature_supported() {
  return OLMO_BF16_MEGA_HAS_TCGEN05_TMEM;
}

template <int kCtaGroup = 1>
OLMO_BF16_MEGA_DEVICE inline void tcgen05_tmem_alloc(
    uint32_t* dst_tmem_ptr_smem,
    uint32_t num_cols) {
  static_assert(kCtaGroup == 1 || kCtaGroup == 2, "TMEM CTA group must be 1 or 2");
#if OLMO_BF16_MEGA_HAS_TCGEN05_TMEM
  if constexpr (kCtaGroup == 1) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :
        : "r"(smem_addr(dst_tmem_ptr_smem)), "r"(num_cols)
        : "memory");
  } else {
    asm volatile(
        "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
        :
        : "r"(smem_addr(dst_tmem_ptr_smem)), "r"(num_cols)
        : "memory");
  }
#else
  (void)dst_tmem_ptr_smem;
  (void)num_cols;
#endif
}

template <int kCtaGroup = 1>
OLMO_BF16_MEGA_DEVICE inline void tcgen05_tmem_dealloc(
    uint32_t tmem_ptr,
    uint32_t num_cols) {
  static_assert(kCtaGroup == 1 || kCtaGroup == 2, "TMEM CTA group must be 1 or 2");
#if OLMO_BF16_MEGA_HAS_TCGEN05_TMEM
  if constexpr (kCtaGroup == 1) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        :
        : "r"(tmem_ptr), "r"(num_cols)
        : "memory");
  } else {
    asm volatile(
        "tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
        :
        : "r"(tmem_ptr), "r"(num_cols)
        : "memory");
  }
#else
  (void)tmem_ptr;
  (void)num_cols;
#endif
}

template <int kCtaGroup = 1>
OLMO_BF16_MEGA_DEVICE inline void tcgen05_tmem_relinquish_alloc_permit() {
  static_assert(kCtaGroup == 1 || kCtaGroup == 2, "TMEM CTA group must be 1 or 2");
#if OLMO_BF16_MEGA_HAS_TCGEN05_TMEM
  if constexpr (kCtaGroup == 1) {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  } else {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
  }
#endif
}

template <int kCtaGroup = 1>
OLMO_BF16_MEGA_DEVICE inline void tcgen05_commit_mbarrier_arrive_one(MBarrier* barrier) {
  static_assert(kCtaGroup == 1 || kCtaGroup == 2, "TMEM CTA group must be 1 or 2");
#if OLMO_BF16_MEGA_HAS_TCGEN05_TMEM
  if constexpr (kCtaGroup == 1) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
        :
        : "r"(smem_addr(barrier))
        : "memory");
  } else {
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];"
        :
        : "r"(smem_addr(barrier))
        : "memory");
  }
#else
  (void)barrier;
#endif
}

OLMO_BF16_MEGA_DEVICE inline void tcgen05_tmem_load_16x256b(
    uint32_t tmem_addr,
    uint32_t (&out)[4]) {
#if OLMO_BF16_MEGA_HAS_TCGEN05_TMEM
  asm volatile(
      "tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0, %1, %2, %3}, [%4];"
      : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
      : "r"(tmem_addr)
      : "memory");
#else
  (void)tmem_addr;
  out[0] = 0;
  out[1] = 0;
  out[2] = 0;
  out[3] = 0;
#endif
}

OLMO_BF16_MEGA_DEVICE inline void tcgen05_tmem_load_32x32b_x8(
    uint32_t tmem_addr,
    uint32_t (&out)[8]) {
#if OLMO_BF16_MEGA_HAS_TCGEN05_TMEM
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(out[0]),
        "=r"(out[1]),
        "=r"(out[2]),
        "=r"(out[3]),
        "=r"(out[4]),
        "=r"(out[5]),
        "=r"(out[6]),
        "=r"(out[7])
      : "r"(tmem_addr)
      : "memory");
#else
  (void)tmem_addr;
  for (uint32_t idx = 0; idx < 8; ++idx) {
    out[idx] = 0;
  }
#endif
}

template <int kCtaGroup = 1>
OLMO_BF16_MEGA_DEVICE inline void tcgen05_mma_f16(
    uint32_t tmem_c,
    uint64_t desc_a,
    uint64_t desc_b,
    uint32_t scale_c,
    uint64_t instr_desc) {
  static_assert(kCtaGroup == 1 || kCtaGroup == 2, "MMA CTA group must be 1 or 2");
  if constexpr (kCtaGroup == 1) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
        "}\n"
        :
        : "r"(tmem_c),
          "l"(desc_a),
          "l"(desc_b),
          "r"(static_cast<uint32_t>(instr_desc >> 32)),
          "r"(scale_c)
        : "memory");
  } else {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t"
        "}\n"
        :
        : "r"(tmem_c),
          "l"(desc_a),
          "l"(desc_b),
          "r"(static_cast<uint32_t>(instr_desc >> 32)),
          "r"(scale_c)
        : "memory");
  }
}

}  // namespace olmo::bf16_mega_moe::ptx

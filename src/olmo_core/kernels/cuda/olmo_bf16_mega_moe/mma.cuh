/*
 * OLMo BF16 MegaMoE SM100 MMA descriptors.
 *
 * These helpers own the small slice of SM100 UMMA descriptor packing needed by
 * the BF16 persistent forward path. They mirror the reference MegaMoE runtime
 * shape without depending on CUTE or DeepGEMM headers.
 */
#pragma once

#include "layout.cuh"
#include "ptx.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace olmo::bf16_mega_moe::mma {

enum class Major : uint8_t {
  K = 0,
  MN = 1,
};

enum class LayoutType : uint8_t {
  SwizzleNone = 0,
  Swizzle128Base32 = 1,
  Swizzle128 = 2,
  Swizzle64 = 4,
  Swizzle32 = 6,
};

union SmemDescriptor {
  uint64_t desc = 0;
  struct {
    uint16_t start_address : 14, : 2;
    uint16_t leading_byte_offset : 14, : 2;
    uint16_t stride_byte_offset : 14, version : 2;
    uint8_t : 1, base_offset : 3, lbo_mode : 1, : 3;
    uint8_t : 5, layout_type : 3;
  };
  struct {
    uint32_t lo;
    uint32_t hi;
  };
};

union InstrDescriptor {
  uint32_t desc = 0;
  struct {
    uint16_t sparse_id2 : 2,
             sparse_flag : 1,
             saturate : 1,
             c_format : 2,
             : 1,
             a_format : 3,
             b_format : 3,
             a_negate : 1,
             b_negate : 1,
             a_major : 1;
    uint16_t b_major : 1,
             n_dim : 6,
             : 1,
             m_dim : 5,
             : 1,
             max_shift : 2;
  };
};

OLMO_BF16_MEGA_DEVICE inline SmemDescriptor make_smem_desc(
    LayoutType layout_type,
    void* smem_ptr,
    uint32_t stride_byte_offset,
    uint32_t leading_byte_offset) {
  SmemDescriptor desc;
  desc.desc = 0;
  desc.version = 1;
  desc.lbo_mode = 0;
  desc.layout_type = static_cast<uint8_t>(layout_type);
  desc.start_address = static_cast<uint16_t>(ptx::smem_addr(smem_ptr) >> 4);
  desc.base_offset = 0;
  desc.stride_byte_offset = static_cast<uint16_t>(stride_byte_offset >> 4);
  desc.leading_byte_offset = static_cast<uint16_t>(leading_byte_offset >> 4);
  return desc;
}

OLMO_BF16_MEGA_DEVICE inline void replace_smem_desc_addr(
    SmemDescriptor& desc,
    const void* smem_ptr) {
  desc.start_address = static_cast<uint16_t>(ptx::smem_addr(smem_ptr) >> 4);
}

OLMO_BF16_MEGA_HOST_DEVICE constexpr LayoutType swizzle_layout_type(
    uint32_t swizzle_bytes,
    bool use_base32 = false) {
  return use_base32 ? LayoutType::Swizzle128Base32 :
      swizzle_bytes == 128 ? LayoutType::Swizzle128 :
      swizzle_bytes == 64 ? LayoutType::Swizzle64 :
      swizzle_bytes == 32 ? LayoutType::Swizzle32 :
      LayoutType::SwizzleNone;
}

OLMO_BF16_MEGA_HOST_DEVICE constexpr uint32_t atom_base(LayoutType layout_type) {
  return layout_type == LayoutType::Swizzle128Base32 ? 32u : 16u;
}

template <uint32_t kBlockMN, uint32_t kBlockK, uint32_t kSwizzleBytes>
OLMO_BF16_MEGA_DEVICE SmemDescriptor make_bf16_k_major_desc(
    __nv_bfloat16* base_smem_ptr,
    uint32_t mn_idx,
    uint32_t k_idx) {
  static_assert(kSwizzleBytes == kBlockK * sizeof(__nv_bfloat16),
                "BF16 K-major descriptor expects one swizzle atom along K");
  const LayoutType layout_type = swizzle_layout_type(kSwizzleBytes);
  const uint32_t num_non_contiguous = 128u / atom_base(layout_type);
  const uint32_t stride_byte_offset =
      num_non_contiguous * kBlockK * static_cast<uint32_t>(sizeof(__nv_bfloat16));
  constexpr uint32_t leading_byte_offset = 0;
  return make_smem_desc(
      layout_type,
      base_smem_ptr + mn_idx * kBlockK + k_idx,
      stride_byte_offset,
      leading_byte_offset);
}

template <uint32_t kSwizzleBytes>
OLMO_BF16_MEGA_HOST_DEVICE constexpr uint32_t bf16_k_major_stride_k() {
  (void)kSwizzleBytes;
  return 1u;
}

template <uint32_t kSwizzleBytes>
OLMO_BF16_MEGA_HOST_DEVICE constexpr uint32_t advance_bf16_k_major_desc_lo(
    uint32_t base,
    uint32_t byte_offset,
    uint32_t k_idx) {
  const uint32_t element_offset =
      byte_offset / static_cast<uint32_t>(sizeof(__nv_bfloat16)) +
      k_idx * bf16_k_major_stride_k<kSwizzleBytes>();
  return base + ((element_offset * static_cast<uint32_t>(sizeof(__nv_bfloat16))) >> 4u);
}

template <uint32_t kBlockMN, uint32_t kBlockK, uint32_t kSwizzleBytes>
OLMO_BF16_MEGA_DEVICE SmemDescriptor make_bf16_mn_major_desc(
    __nv_bfloat16* base_smem_ptr,
    uint32_t mn_idx,
    uint32_t k_idx) {
  static_assert(kSwizzleBytes > 0, "BF16 MN-major descriptor expects swizzled shared memory");
  static_assert(kBlockMN % (kSwizzleBytes / sizeof(__nv_bfloat16)) == 0,
                "BF16 MN-major block must contain whole swizzle atoms");
  const LayoutType layout_type = swizzle_layout_type(kSwizzleBytes);
  const uint32_t num_non_contiguous = 128u / atom_base(layout_type);
  constexpr uint32_t kBlockMnAtom =
      kSwizzleBytes / static_cast<uint32_t>(sizeof(__nv_bfloat16));
  const uint32_t stride_byte_offset =
      num_non_contiguous * kBlockMnAtom * static_cast<uint32_t>(sizeof(__nv_bfloat16));
  const uint32_t leading_byte_offset =
      kBlockK * kBlockMnAtom * static_cast<uint32_t>(sizeof(__nv_bfloat16));
  return make_smem_desc(
      layout_type,
      base_smem_ptr + mn_idx * kBlockK + k_idx * kBlockMnAtom,
      stride_byte_offset,
      leading_byte_offset);
}

template <uint32_t kSwizzleBytes>
OLMO_BF16_MEGA_HOST_DEVICE constexpr uint32_t bf16_mn_major_stride_k() {
  return kSwizzleBytes / static_cast<uint32_t>(sizeof(__nv_bfloat16));
}

template <uint32_t kSwizzleBytes>
OLMO_BF16_MEGA_HOST_DEVICE constexpr uint32_t advance_bf16_mn_major_desc_lo(
    uint32_t base,
    uint32_t byte_offset,
    uint32_t k_idx) {
  const uint32_t element_offset =
      byte_offset / static_cast<uint32_t>(sizeof(__nv_bfloat16)) +
      k_idx * bf16_mn_major_stride_k<kSwizzleBytes>();
  return base + ((element_offset * static_cast<uint32_t>(sizeof(__nv_bfloat16))) >> 4u);
}

template <uint32_t kUmmaM, uint32_t kUmmaN, Major kAMajor = Major::K, Major kBMajor = Major::K>
OLMO_BF16_MEGA_HOST_DEVICE constexpr InstrDescriptor make_bf16_f32_instr_desc() {
  static_assert(kUmmaM == 64 || kUmmaM == 128 || kUmmaM == 256,
                "SM100 BF16 UMMA M must be 64, 128, or 256");
  static_assert(kUmmaN >= 8 && kUmmaN <= 256 && kUmmaN % 8 == 0,
                "SM100 BF16 UMMA N must be in [8, 256] and divisible by 8");
  InstrDescriptor desc;
  desc.desc = 0;
  desc.a_format = 1;  // BF16
  desc.b_format = 1;  // BF16
  desc.c_format = 1;  // F32
  desc.m_dim = kUmmaM >> 4;
  desc.n_dim = kUmmaN >> 3;
  desc.a_major = static_cast<uint8_t>(kAMajor);
  desc.b_major = static_cast<uint8_t>(kBMajor);
  desc.a_negate = 0;
  desc.b_negate = 0;
  desc.saturate = 0;
  desc.sparse_flag = 0;
  desc.sparse_id2 = 0;
  desc.max_shift = 0;
  return desc;
}

OLMO_BF16_MEGA_HOST_DEVICE constexpr uint64_t make_runtime_instr_desc(
    InstrDescriptor desc) {
  return static_cast<uint64_t>(desc.desc) << 32;
}

OLMO_BF16_MEGA_HOST_DEVICE constexpr InstrDescriptor update_instr_desc_umma_n(
    InstrDescriptor desc,
    uint32_t umma_n) {
  desc.n_dim = static_cast<uint16_t>(umma_n >> 3);
  return desc;
}

}  // namespace olmo::bf16_mega_moe::mma

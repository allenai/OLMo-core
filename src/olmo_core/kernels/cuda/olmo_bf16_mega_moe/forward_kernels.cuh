/*
 * OLMo BF16 MegaMoE forward kernel launch surfaces.
 *
 * These kernels establish the CUDA-owned F1/F2 task contract before the BF16
 * tensor-core tile bodies are filled in. They intentionally decode the same
 * task stream the production megakernels will consume.
 */
#pragma once

#include "barrier.cuh"
#include "dispatch.cuh"
#include "mma.cuh"
#include "megakernel_plan.cuh"
#include "ptx.cuh"
#include "scheduler.cuh"

#include <cuda_bf16.h>
#include <mma.h>

#include <cstdio>
#include <cstdint>

namespace olmo::bf16_mega_moe::kernels {

namespace wmma = nvcuda::wmma;

constexpr int64_t kForwardPlanDebugRows = 4;
constexpr int64_t kForwardPlanDebugCols = 8;
constexpr int64_t kForwardTaskKindCount = 7;
constexpr int64_t kWmmaM = 16;
constexpr int64_t kWmmaN = 16;
constexpr int64_t kWmmaK = 16;
constexpr uint32_t kStandardNumMaxTokensPerRank = 16384;
constexpr uint32_t kStandardHidden = 4096;
constexpr uint32_t kStandardIntermediate = 4096;
constexpr uint32_t kStandardNumTotalExperts = 32;
constexpr uint32_t kStandardTopK = 4;
constexpr uint32_t kStandardNumRanks = 4;
constexpr uint32_t kStandardNumLocalExperts = kStandardNumTotalExperts / kStandardNumRanks;
constexpr uint32_t kStandardExpertsPerWave = 4;
constexpr uint32_t kStandardSchedulerSms = 8;
constexpr uint32_t kStandardDispatchWarps = 4;
constexpr uint32_t kStandardBlockM = 128;
constexpr uint32_t kStandardBlockN = 128;
constexpr uint32_t kStandardBlockK = 64;
constexpr int64_t kSm100TmaUmmaContractDebugValues = 16;
constexpr uint32_t kTmaContractRows = 16;
constexpr uint32_t kTmaContractCols = 64;
constexpr int64_t kTmaContractDebugValues = 8;
constexpr uint32_t kSm100TileContractM = 128;
constexpr uint32_t kSm100TileContractN = 128;
constexpr uint32_t kSm100TileContractK = 64;
constexpr uint32_t kStandardUmmaBlockThreads = kSm100TileContractM;
constexpr int64_t kSm100TmaUmmaTileContractDebugHeaderValues = 16;
constexpr int64_t kSm100TmaUmmaTileContractDebugRegisterValues = 32 * 8;
constexpr int64_t kSm100TmaUmmaTileContractDebugValues =
    kSm100TmaUmmaTileContractDebugHeaderValues +
    kSm100TmaUmmaTileContractDebugRegisterValues;

OLMO_BF16_MEGA_HOST_DEVICE inline uint64_t standard_ep_metadata_bytes() {
  const layout::Workspace workspace(
      nullptr,
      kStandardNumRanks,
      kStandardNumTotalExperts,
      kStandardNumMaxTokensPerRank,
      kStandardTopK);
  return workspace.num_bytes();
}

OLMO_BF16_MEGA_HOST_DEVICE inline uint64_t standard_ep_source_input_bytes(
    int64_t hidden) {
  return layout::math::align<uint64_t>(
      static_cast<uint64_t>(kStandardNumMaxTokensPerRank) *
          static_cast<uint64_t>(hidden) * sizeof(__nv_bfloat16),
      16);
}

OLMO_BF16_MEGA_HOST_DEVICE inline uint64_t standard_ep_output_bytes(
    int64_t hidden) {
  return layout::math::align<uint64_t>(
      static_cast<uint64_t>(kStandardNumMaxTokensPerRank) *
          static_cast<uint64_t>(kStandardTopK) *
          static_cast<uint64_t>(hidden) * sizeof(__nv_bfloat16),
      16);
}

OLMO_BF16_MEGA_HOST_DEVICE inline uint64_t standard_ep_workspace_stride_bytes(
    int64_t hidden) {
  return layout::math::align<uint64_t>(
      standard_ep_metadata_bytes() +
          standard_ep_source_input_bytes(hidden) +
          standard_ep_output_bytes(hidden),
      16);
}

OLMO_BF16_MEGA_HOST_DEVICE inline uint64_t standard_ep_local_packed_capacity(
    int64_t num_tokens) {
  return layout::num_max_pool_tokens<uint64_t>(
      static_cast<uint64_t>(kStandardNumRanks),
      static_cast<uint64_t>(num_tokens),
      static_cast<uint64_t>(kStandardTopK),
      static_cast<uint64_t>(kStandardNumLocalExperts));
}

OLMO_BF16_MEGA_HOST_DEVICE inline __nv_bfloat16* standard_ep_source_input_window(
    const uint64_t* rank_workspace_bases,
    uint32_t rank_idx,
    int64_t hidden) {
  (void)hidden;
  return reinterpret_cast<__nv_bfloat16*>(
      rank_workspace_bases[rank_idx] + standard_ep_metadata_bytes());
}

OLMO_BF16_MEGA_HOST_DEVICE inline __nv_bfloat16* standard_ep_output_window(
    const uint64_t* rank_workspace_bases,
    uint32_t rank_idx,
    int64_t hidden) {
  return reinterpret_cast<__nv_bfloat16*>(
      rank_workspace_bases[rank_idx] +
      standard_ep_metadata_bytes() +
      standard_ep_source_input_bytes(hidden));
}

struct StandardEpPackedSourceRoute {
  int64_t token_topk_idx = -1;
  uint32_t source_rank_idx = 0;
};

OLMO_BF16_MEGA_HOST_DEVICE inline int64_t standard_ep_encode_source_route(
    uint32_t source_rank_idx,
    int64_t token_topk_idx,
    int64_t num_route_slots) {
  return static_cast<int64_t>(source_rank_idx) * num_route_slots + token_topk_idx;
}

OLMO_BF16_MEGA_HOST_DEVICE inline StandardEpPackedSourceRoute standard_ep_decode_source_route(
    int64_t encoded,
    int64_t num_route_slots) {
  if (encoded < 0 || num_route_slots <= 0) {
    return {};
  }
  return {
      encoded % num_route_slots,
      static_cast<uint32_t>(encoded / num_route_slots),
  };
}

OLMO_BF16_MEGA_HOST_DEVICE inline StandardEpPackedSourceRoute standard_ep_load_source_route(
    const uint64_t* rank_workspace_bases,
    uint32_t dst_rank_idx,
    uint32_t dst_local_expert_idx,
    int64_t dst_expert_slot) {
  const void* rank_base = reinterpret_cast<const void*>(rank_workspace_bases[dst_rank_idx]);
  const layout::Workspace rank_workspace(
      const_cast<void*>(rank_base),
      kStandardNumRanks,
      kStandardNumTotalExperts,
      kStandardNumMaxTokensPerRank,
      kStandardTopK);
  int64_t source_slot_base = 0;
  for (uint32_t src_rank_idx = 0; src_rank_idx < kStandardNumRanks; ++src_rank_idx) {
    const int64_t source_count = static_cast<int64_t>(
        *rank_workspace.expert_recv_count_ptr(src_rank_idx, dst_local_expert_idx) &
        0xffffffffu);
    if (dst_expert_slot < source_slot_base + source_count) {
      const uint32_t source_slot =
          static_cast<uint32_t>(dst_expert_slot - source_slot_base);
      return {
          static_cast<int64_t>(*rank_workspace.src_token_topk_idx_ptr(
              dst_local_expert_idx,
              src_rank_idx,
              source_slot)),
          src_rank_idx,
      };
    }
    source_slot_base += source_count;
  }
  return {};
}

struct GroupedGemmTile {
  int64_t local_expert = -1;
  int64_t global_m_tile = -1;
  int64_t local_m_tile = -1;
  int64_t n_tile = -1;
};

OLMO_BF16_MEGA_DEVICE inline GroupedGemmTile decode_grouped_gemm_tile(
    int64_t task_idx,
    const int64_t* tile_offsets,
    int64_t num_local_experts,
    int64_t n_tiles);

__global__ void sm100_tma_umma_contract_kernel(int64_t* debug) {
  __shared__ alignas(1024) __nv_bfloat16 smem_a[kStandardBlockM * kStandardBlockK];
  __shared__ alignas(1024) __nv_bfloat16 smem_b[kStandardBlockN * kStandardBlockK];
  __shared__ alignas(8) ptx::MBarrier commit_barrier;
  __shared__ uint32_t tmem_ptr_smem;

  if (blockIdx.x != 0) {
    return;
  }

  if (threadIdx.x == 0) {
    debug[0] = 0;
    debug[1] = 0;
    debug[2] = kStandardBlockM;
    debug[3] = kStandardBlockN;
    debug[4] = kStandardBlockK;
    debug[5] = kStandardHidden;
    debug[6] = kStandardIntermediate;
    debug[7] = kStandardNumTotalExperts;
    debug[8] = kStandardTopK;
    debug[9] = kStandardNumRanks;
    debug[10] = 0;
    debug[11] = 0;
    debug[12] = 0;
    debug[13] = 0;
    debug[14] = 0;
    debug[15] = 0;
  }
  __syncthreads();

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (threadIdx.x == 0) {
    debug[0] = static_cast<int64_t>(__CUDA_ARCH__);
    debug[1] = static_cast<int64_t>(ptx::tcgen05_tmem_feature_supported());
    tmem_ptr_smem = 0;
    ptx::mbarrier_init(&commit_barrier, 1);
  }
  __syncthreads();

  constexpr uint32_t kContractUmmaM = 128;
  constexpr uint32_t kContractUmmaN = kStandardBlockN;
  constexpr uint32_t kContractSwizzleBytes =
      kStandardBlockK * static_cast<uint32_t>(sizeof(__nv_bfloat16));
  const mma::SmemDescriptor desc_a =
      mma::make_bf16_k_major_desc<kContractUmmaM, kStandardBlockK, kContractSwizzleBytes>(
          smem_a,
          /*mn_idx=*/0,
          /*k_idx=*/0);
  const mma::SmemDescriptor desc_b =
      mma::make_bf16_k_major_desc<kContractUmmaN, kStandardBlockK, kContractSwizzleBytes>(
          smem_b,
          /*mn_idx=*/0,
          /*k_idx=*/0);
  const mma::InstrDescriptor instr_desc =
      mma::make_bf16_f32_instr_desc<kContractUmmaM, kContractUmmaN>();

  if (threadIdx.x == 0) {
    debug[10] = desc_a.desc != 0 ? 1 : 0;
    debug[11] = desc_b.desc != 0 ? 1 : 0;
    debug[12] = instr_desc.desc != 0 ? 1 : 0;
    debug[13] = static_cast<int64_t>(mma::make_runtime_instr_desc(instr_desc) >> 32);
  }

#if OLMO_BF16_MEGA_HAS_TCGEN05_TMEM
  if (threadIdx.x < 32) {
    ptx::tcgen05_tmem_alloc<1>(&tmem_ptr_smem, /*num_cols=*/kStandardBlockN);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    debug[14] = static_cast<int64_t>(tmem_ptr_smem);
  }
  if (threadIdx.x < 32) {
    ptx::tcgen05_tmem_dealloc<1>(tmem_ptr_smem, /*num_cols=*/kStandardBlockN);
    ptx::tcgen05_tmem_relinquish_alloc_permit<1>();
  }
  if (threadIdx.x == 0) {
    debug[15] = 1;
  }
#endif
#endif
}

__global__ void sm100_bf16_tma_load_contract_kernel(
    const CUtensorMap* tensor_map,
    int64_t* debug) {
  __shared__ alignas(1024) __nv_bfloat16 tile[kTmaContractRows * kTmaContractCols];
  __shared__ alignas(8) ptx::MBarrier barrier;

  if (blockIdx.x != 0) {
    return;
  }

  if (threadIdx.x == 0) {
    for (uint32_t idx = 0; idx < kTmaContractDebugValues; ++idx) {
      debug[idx] = 0;
    }
  }
  __syncthreads();

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if (threadIdx.x == 0) {
    debug[0] = static_cast<int64_t>(__CUDA_ARCH__);
    ptx::mbarrier_init(&barrier, 1);
  }
  for (uint32_t idx = threadIdx.x; idx < kTmaContractRows * kTmaContractCols; idx += blockDim.x) {
    tile[idx] = __float2bfloat16(0.0f);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    ptx::mbarrier_arrive_expect_tx(
        &barrier,
        kTmaContractRows * kTmaContractCols * static_cast<uint32_t>(sizeof(__nv_bfloat16)));
    ptx::tma_load_2d<1>(
        tile,
        tensor_map,
        /*coord0=*/0,
        /*coord1=*/0,
        &barrier);
  }
  ptx::mbarrier_wait_parity(&barrier, 0);
  __syncthreads();

  if (threadIdx.x == 0) {
    debug[1] = 1;
    debug[2] = static_cast<int64_t>(__bfloat162float(tile[0]));
    debug[3] = static_cast<int64_t>(__bfloat162float(tile[1]));
    debug[4] = static_cast<int64_t>(__bfloat162float(tile[kTmaContractCols - 1]));
    debug[5] = static_cast<int64_t>(__bfloat162float(tile[kTmaContractCols]));
    debug[6] = kTmaContractRows;
    debug[7] = kTmaContractCols;
  }
#else
  (void)tensor_map;
#endif
}

__global__ void sm100_bf16_tma_umma_tile_contract_kernel(
    const CUtensorMap* a_tensor_map,
    const CUtensorMap* b_tensor_map,
    int64_t* debug,
    float* out,
    bool b_mn_major) {
  __shared__ alignas(1024) __nv_bfloat16 smem_a[
      kSm100TileContractM * kSm100TileContractK];
  __shared__ alignas(1024) __nv_bfloat16 smem_b[
      kSm100TileContractN * kSm100TileContractK];
  __shared__ alignas(8) ptx::MBarrier tma_barrier;
  __shared__ alignas(8) ptx::MBarrier umma_full_barrier;
  __shared__ uint32_t tmem_ptr_smem;

  if (blockIdx.x != 0) {
    return;
  }

  for (uint32_t idx = threadIdx.x;
       idx < kSm100TmaUmmaTileContractDebugValues;
       idx += blockDim.x) {
    debug[idx] = 0;
  }
  __syncthreads();

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  constexpr uint32_t kTileBytesA =
      kSm100TileContractM * kSm100TileContractK *
      static_cast<uint32_t>(sizeof(__nv_bfloat16));
  constexpr uint32_t kTileBytesB =
      kSm100TileContractN * kSm100TileContractK *
      static_cast<uint32_t>(sizeof(__nv_bfloat16));
  constexpr uint32_t kSwizzleBytes =
      kSm100TileContractK * static_cast<uint32_t>(sizeof(__nv_bfloat16));

  if (threadIdx.x == 0) {
    debug[0] = static_cast<int64_t>(__CUDA_ARCH__);
    debug[1] = static_cast<int64_t>(ptx::tcgen05_tmem_feature_supported());
    debug[2] = kSm100TileContractM;
    debug[3] = kSm100TileContractN;
    debug[4] = kSm100TileContractK;
    debug[5] = kSwizzleBytes;
    debug[6] = 0;
    debug[7] = 0;
    debug[14] = b_mn_major ? 1 : 0;
    tmem_ptr_smem = 0;
    ptx::mbarrier_init(&tma_barrier, 1);
    ptx::mbarrier_init(&umma_full_barrier, 1);
  }

  for (uint32_t idx = threadIdx.x;
       idx < kSm100TileContractM * kSm100TileContractK;
       idx += blockDim.x) {
    smem_a[idx] = __float2bfloat16(0.0f);
  }
  for (uint32_t idx = threadIdx.x;
       idx < kSm100TileContractN * kSm100TileContractK;
       idx += blockDim.x) {
    smem_b[idx] = __float2bfloat16(0.0f);
  }
  __syncthreads();

#if OLMO_BF16_MEGA_HAS_TCGEN05_TMEM
  if (threadIdx.x < 32) {
    ptx::tcgen05_tmem_alloc<1>(&tmem_ptr_smem, /*num_cols=*/kSm100TileContractN);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    debug[6] = static_cast<int64_t>(tmem_ptr_smem);
    ptx::mbarrier_arrive_expect_tx(&tma_barrier, kTileBytesA + kTileBytesB);
    ptx::tma_load_2d<1>(
        smem_a,
        a_tensor_map,
        /*coord0=*/0,
        /*coord1=*/0,
        &tma_barrier);
    if (b_mn_major) {
      constexpr uint32_t kBlockMnAtom =
          kSwizzleBytes / static_cast<uint32_t>(sizeof(__nv_bfloat16));
      ptx::tma_load_2d<1>(
          smem_b,
          b_tensor_map,
          /*coord0=*/0,
          /*coord1=*/0,
          &tma_barrier);
      ptx::tma_load_2d<1>(
          smem_b + kBlockMnAtom * kSm100TileContractK,
          b_tensor_map,
          /*coord0=*/static_cast<int32_t>(kBlockMnAtom),
          /*coord1=*/0,
          &tma_barrier);
    } else {
      ptx::tma_load_2d<1>(
          smem_b,
          b_tensor_map,
          /*coord0=*/0,
          /*coord1=*/0,
          &tma_barrier);
    }
  }
  ptx::mbarrier_wait_parity(&tma_barrier, 0);
  __syncthreads();

  const mma::SmemDescriptor desc_a =
      mma::make_bf16_k_major_desc<
          kSm100TileContractM,
          kSm100TileContractK,
          kSwizzleBytes>(
          smem_a,
          /*mn_idx=*/0,
          /*k_idx=*/0);
  mma::SmemDescriptor desc_b;
  mma::InstrDescriptor instr_desc;
  if (b_mn_major) {
    desc_b =
        mma::make_bf16_mn_major_desc<
            kSm100TileContractN,
            kSm100TileContractK,
            kSwizzleBytes>(
            smem_b,
            /*mn_idx=*/0,
            /*k_idx=*/0);
    instr_desc = mma::make_bf16_f32_instr_desc<
        kSm100TileContractM,
        kSm100TileContractN,
        mma::Major::K,
        mma::Major::MN>();
  } else {
    desc_b =
        mma::make_bf16_k_major_desc<
            kSm100TileContractN,
            kSm100TileContractK,
            kSwizzleBytes>(
            smem_b,
            /*mn_idx=*/0,
            /*k_idx=*/0);
    instr_desc = mma::make_bf16_f32_instr_desc<kSm100TileContractM, kSm100TileContractN>();
  }
  const uint64_t runtime_instr_desc = mma::make_runtime_instr_desc(instr_desc);

  if (threadIdx.x == 0) {
    debug[7] = 1;
    debug[8] = desc_a.desc != 0 ? 1 : 0;
    debug[9] = desc_b.desc != 0 ? 1 : 0;
    debug[10] = instr_desc.desc != 0 ? 1 : 0;
    debug[11] = static_cast<int64_t>(runtime_instr_desc >> 32);
  }

  if (threadIdx.x >= 32 && threadIdx.x < 64) {
    ptx::tcgen05_after_thread_sync();
    if ((threadIdx.x & 31) == 0) {
      constexpr uint32_t kUmmaK = 16;
      #pragma unroll
      for (uint32_t k_idx = 0; k_idx < kSm100TileContractK; k_idx += kUmmaK) {
        mma::SmemDescriptor k_desc_a = desc_a;
        mma::SmemDescriptor k_desc_b = desc_b;
        k_desc_a.lo = mma::advance_bf16_k_major_desc_lo<kSwizzleBytes>(
            desc_a.lo,
            /*byte_offset=*/0,
            k_idx);
        k_desc_b.lo = mma::advance_bf16_k_major_desc_lo<kSwizzleBytes>(
            desc_b.lo,
            /*byte_offset=*/0,
            k_idx);
        if (b_mn_major) {
          k_desc_b.lo = mma::advance_bf16_mn_major_desc_lo<kSwizzleBytes>(
              desc_b.lo,
              /*byte_offset=*/0,
              k_idx);
        }
        ptx::tcgen05_mma_f16<1>(
            tmem_ptr_smem,
            k_desc_a.desc,
            k_desc_b.desc,
            /*scale_c=*/k_idx == 0 ? 0 : 1,
            runtime_instr_desc);
      }
    }
    __syncwarp();
    if ((threadIdx.x & 31) == 0) {
      ptx::tcgen05_commit_mbarrier_arrive_one<1>(&umma_full_barrier);
    }
  }

  ptx::mbarrier_wait_parity(&umma_full_barrier, 0);
  __syncthreads();

  if (threadIdx.x < 32) {
    ptx::tcgen05_after_thread_sync();
    uint32_t regs[8];
    ptx::tcgen05_tmem_load_32x32b_x8(tmem_ptr_smem, regs);
    ptx::tcgen05_wait_tmem_load();
    const uint32_t lane = threadIdx.x & 31;
    #pragma unroll
    for (uint32_t idx = 0; idx < 8; ++idx) {
      debug[kSm100TmaUmmaTileContractDebugHeaderValues + lane * 8 + idx] =
          static_cast<int64_t>(static_cast<uint64_t>(regs[idx]));
    }
    if (lane == 0) {
      debug[12] = 1;
    }
  }
  __syncthreads();

  if (out != nullptr && threadIdx.x < kSm100TileContractM) {
    const uint32_t row = threadIdx.x;
    #pragma unroll
    for (uint32_t col = 0; col < kSm100TileContractN; col += 8) {
      uint32_t regs[8];
      ptx::tcgen05_tmem_load_32x32b_x8(tmem_ptr_smem + col, regs);
      ptx::tcgen05_wait_tmem_load();
      #pragma unroll
      for (uint32_t idx = 0; idx < 8; ++idx) {
        out[row * kSm100TileContractN + col + idx] = __uint_as_float(regs[idx]);
      }
    }
  }
  __syncthreads();

  if (threadIdx.x < 32) {
    ptx::tcgen05_tmem_dealloc<1>(tmem_ptr_smem, /*num_cols=*/kSm100TileContractN);
    ptx::tcgen05_tmem_relinquish_alloc_permit<1>();
  }
  if (threadIdx.x == 0) {
    debug[13] = 1;
  }
#else
  (void)a_tensor_map;
  (void)b_tensor_map;
#endif
#else
  (void)a_tensor_map;
  (void)b_tensor_map;
#endif
}

template <bool kBMnMajor, bool kManageTmem = true>
OLMO_BF16_MEGA_DEVICE inline void compute_bf16_umma_linear_tile_to_bf16(
    const CUtensorMap* a_tensor_map,
    const CUtensorMap* b_tensor_map,
    const int64_t* expert_counts,
    const int64_t* token_offsets,
    int64_t expert_idx,
    int64_t local_m_tile,
    int64_t n_tile,
    int64_t in_features,
    int64_t out_features,
    int64_t b_row_base,
    __nv_bfloat16* out,
    __nv_bfloat16* smem_a,
    __nv_bfloat16* smem_b,
    ptx::MBarrier* tma_barrier,
    ptx::MBarrier* umma_full_barrier,
    uint32_t* tmem_ptr_smem) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && OLMO_BF16_MEGA_HAS_TCGEN05_TMEM
  constexpr uint32_t kTileM = kSm100TileContractM;
  constexpr uint32_t kTileN = kSm100TileContractN;
  constexpr uint32_t kTileK = kSm100TileContractK;
  constexpr uint32_t kUmmaK = 16;
  constexpr uint32_t kSwizzleBytes =
      kTileK * static_cast<uint32_t>(sizeof(__nv_bfloat16));
  constexpr uint32_t kBMnAtom =
      kSwizzleBytes / static_cast<uint32_t>(sizeof(__nv_bfloat16));
  constexpr uint32_t kTileBytesA =
      kTileM * kTileK * static_cast<uint32_t>(sizeof(__nv_bfloat16));
  constexpr uint32_t kTileBytesB =
      kTileN * kTileK * static_cast<uint32_t>(sizeof(__nv_bfloat16));

  const int64_t expert_count = expert_counts[expert_idx];
  const int64_t m_start = token_offsets[expert_idx] + local_m_tile * kTileM;
  const int64_t n_start = n_tile * kTileN;
  if (n_start >= out_features || local_m_tile * kTileM >= expert_count) {
    return;
  }

  if (threadIdx.x == 0) {
    ptx::mbarrier_init(tma_barrier, 1);
    ptx::mbarrier_init(umma_full_barrier, 1);
    if constexpr (kManageTmem) {
      *tmem_ptr_smem = 0;
    }
  }
  __syncthreads();

  if constexpr (kManageTmem) {
    if (threadIdx.x < 32) {
      ptx::tcgen05_after_thread_sync();
      ptx::tcgen05_tmem_alloc<1>(tmem_ptr_smem, /*num_cols=*/kTileN);
    }
    __syncthreads();
  }

  const mma::SmemDescriptor desc_a_base =
      mma::make_bf16_k_major_desc<kTileM, kTileK, kSwizzleBytes>(
          smem_a,
          /*mn_idx=*/0,
          /*k_idx=*/0);
  mma::SmemDescriptor desc_b_base;
  mma::InstrDescriptor instr_desc;
  if constexpr (kBMnMajor) {
    desc_b_base = mma::make_bf16_mn_major_desc<kTileN, kTileK, kSwizzleBytes>(
        smem_b,
        /*mn_idx=*/0,
        /*k_idx=*/0);
    instr_desc = mma::make_bf16_f32_instr_desc<kTileM, kTileN, mma::Major::K, mma::Major::MN>();
  } else {
    desc_b_base = mma::make_bf16_k_major_desc<kTileN, kTileK, kSwizzleBytes>(
        smem_b,
        /*mn_idx=*/0,
        /*k_idx=*/0);
    instr_desc = mma::make_bf16_f32_instr_desc<kTileM, kTileN>();
  }
  const uint64_t runtime_instr_desc = mma::make_runtime_instr_desc(instr_desc);

  uint32_t tma_phase = 0;
  for (int64_t k_start = 0; k_start < in_features; k_start += kTileK) {
    const bool need_pre_tma_zero =
        (k_start + static_cast<int64_t>(kTileK) > in_features) ||
        (n_start + static_cast<int64_t>(kTileN) > out_features);
    if (need_pre_tma_zero) {
      for (uint32_t idx = threadIdx.x; idx < kTileM * kTileK; idx += blockDim.x) {
        smem_a[idx] = __float2bfloat16(0.0f);
      }
      for (uint32_t idx = threadIdx.x; idx < kTileN * kTileK; idx += blockDim.x) {
        smem_b[idx] = __float2bfloat16(0.0f);
      }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      ptx::mbarrier_arrive_expect_tx(tma_barrier, kTileBytesA + kTileBytesB);
      ptx::tma_load_2d<1>(
          smem_a,
          a_tensor_map,
          static_cast<int32_t>(k_start),
          static_cast<int32_t>(m_start),
          tma_barrier);
      if constexpr (kBMnMajor) {
        ptx::tma_load_2d<1>(
            smem_b,
            b_tensor_map,
            static_cast<int32_t>(n_start),
            static_cast<int32_t>(b_row_base + k_start),
            tma_barrier);
        ptx::tma_load_2d<1>(
            smem_b + kBMnAtom * kTileK,
            b_tensor_map,
            static_cast<int32_t>(n_start + kBMnAtom),
            static_cast<int32_t>(b_row_base + k_start),
            tma_barrier);
      } else {
        ptx::tma_load_2d<1>(
            smem_b,
            b_tensor_map,
            static_cast<int32_t>(k_start),
            static_cast<int32_t>(b_row_base + n_start),
            tma_barrier);
      }
    }
    ptx::mbarrier_wait_parity(tma_barrier, tma_phase);
    tma_phase ^= 1u;
    __syncthreads();

    for (uint32_t idx = threadIdx.x; idx < kTileM * kTileK; idx += blockDim.x) {
      const uint32_t row = idx / kTileK;
      const int64_t expert_local_row = local_m_tile * kTileM + row;
      if (expert_local_row >= expert_count) {
        smem_a[idx] = __float2bfloat16(0.0f);
      }
    }
    if constexpr (!kBMnMajor) {
      for (uint32_t idx = threadIdx.x; idx < kTileN * kTileK; idx += blockDim.x) {
        const uint32_t col = idx / kTileK;
        if (n_start + col >= out_features) {
          smem_b[idx] = __float2bfloat16(0.0f);
        }
      }
    }
    __syncthreads();

    if (threadIdx.x >= 32 && threadIdx.x < 64) {
      ptx::tcgen05_after_thread_sync();
      if ((threadIdx.x & 31) == 0) {
#pragma unroll
        for (uint32_t k_idx = 0; k_idx < kTileK; k_idx += kUmmaK) {
          mma::SmemDescriptor desc_a = desc_a_base;
          mma::SmemDescriptor desc_b = desc_b_base;
          desc_a.lo = mma::advance_bf16_k_major_desc_lo<kSwizzleBytes>(
              desc_a_base.lo,
              /*byte_offset=*/0,
              k_idx);
          if constexpr (kBMnMajor) {
            desc_b.lo = mma::advance_bf16_mn_major_desc_lo<kSwizzleBytes>(
                desc_b_base.lo,
                /*byte_offset=*/0,
                k_idx);
          } else {
            desc_b.lo = mma::advance_bf16_k_major_desc_lo<kSwizzleBytes>(
                desc_b_base.lo,
                /*byte_offset=*/0,
                k_idx);
          }
          ptx::tcgen05_mma_f16<1>(
              *tmem_ptr_smem,
              desc_a.desc,
              desc_b.desc,
              /*scale_c=*/(k_start == 0 && k_idx == 0) ? 0 : 1,
              runtime_instr_desc);
        }
      }
    }
    __syncthreads();
  }

  if (threadIdx.x >= 32 && threadIdx.x < 64 && (threadIdx.x & 31) == 0) {
    ptx::tcgen05_commit_mbarrier_arrive_one<1>(umma_full_barrier);
  }
  ptx::mbarrier_wait_parity(umma_full_barrier, 0);
  __syncthreads();

  if (threadIdx.x < kTileM) {
    const uint32_t row = threadIdx.x;
    const int64_t expert_local_row = local_m_tile * kTileM + row;
    const bool valid_row = expert_local_row < expert_count;
#pragma unroll
    for (uint32_t col = 0; col < kTileN; col += 8) {
      uint32_t regs[8];
      ptx::tcgen05_tmem_load_32x32b_x8(*tmem_ptr_smem + col, regs);
      ptx::tcgen05_wait_tmem_load();
#pragma unroll
      for (uint32_t idx = 0; idx < 8; ++idx) {
        if (valid_row && n_start + col + idx < out_features) {
          out[(m_start + row) * out_features + n_start + col + idx] =
              __float2bfloat16(__uint_as_float(regs[idx]));
        }
      }
    }
  }
  __syncthreads();

  if constexpr (kManageTmem) {
    if (threadIdx.x < 32) {
      ptx::tcgen05_tmem_dealloc<1>(*tmem_ptr_smem, /*num_cols=*/kTileN);
      ptx::tcgen05_tmem_relinquish_alloc_permit<1>();
    }
    __syncthreads();
  }
#else
  (void)a_tensor_map;
  (void)b_tensor_map;
  (void)expert_counts;
  (void)token_offsets;
  (void)expert_idx;
  (void)local_m_tile;
  (void)n_tile;
  (void)in_features;
  (void)out_features;
  (void)b_row_base;
  (void)out;
  (void)smem_a;
  (void)smem_b;
  (void)tma_barrier;
  (void)umma_full_barrier;
  (void)tmem_ptr_smem;
#endif
}

__global__ void grouped_w1_umma_linear_kernel(
    const CUtensorMap* packed_input_map,
    const CUtensorMap* up_gate_weight_map,
    const int64_t* expert_counts,
    const int64_t* token_offsets,
    const int64_t* tile_offsets,
    int64_t num_local_experts,
    int64_t total_tasks,
    int64_t hidden,
    int64_t intermediate,
    int64_t gate_offset,
    __nv_bfloat16* out) {
  __shared__ alignas(1024) __nv_bfloat16 smem_a[
      kSm100TileContractM * kSm100TileContractK];
  __shared__ alignas(1024) __nv_bfloat16 smem_b[
      kSm100TileContractN * kSm100TileContractK];
  __shared__ alignas(8) ptx::MBarrier tma_barrier;
  __shared__ alignas(8) ptx::MBarrier umma_full_barrier;
  __shared__ uint32_t tmem_ptr_smem;

  const int64_t n_tiles = (intermediate + kSm100TileContractN - 1) /
      kSm100TileContractN;
  for (int64_t task_idx = blockIdx.x; task_idx < total_tasks; task_idx += gridDim.x) {
    const GroupedGemmTile tile =
        decode_grouped_gemm_tile(task_idx, tile_offsets, num_local_experts, n_tiles);
    if (tile.local_expert < 0) {
      continue;
    }
    compute_bf16_umma_linear_tile_to_bf16</*kBMnMajor=*/false>(
        packed_input_map,
        up_gate_weight_map,
        expert_counts,
        token_offsets,
        tile.local_expert,
        tile.local_m_tile,
        tile.n_tile,
        hidden,
        intermediate,
        tile.local_expert * (2 * intermediate) + gate_offset,
        out,
        smem_a,
        smem_b,
        &tma_barrier,
        &umma_full_barrier,
        &tmem_ptr_smem);
  }
}

__global__ void grouped_w2_umma_linear_kernel(
    const CUtensorMap* h_map,
    const CUtensorMap* down_weight_map,
    const int64_t* expert_counts,
    const int64_t* token_offsets,
    const int64_t* tile_offsets,
    int64_t num_local_experts,
    int64_t total_tasks,
    int64_t intermediate,
    int64_t hidden,
    __nv_bfloat16* packed_expert_out) {
  __shared__ alignas(1024) __nv_bfloat16 smem_a[
      kSm100TileContractM * kSm100TileContractK];
  __shared__ alignas(1024) __nv_bfloat16 smem_b[
      kSm100TileContractN * kSm100TileContractK];
  __shared__ alignas(8) ptx::MBarrier tma_barrier;
  __shared__ alignas(8) ptx::MBarrier umma_full_barrier;
  __shared__ uint32_t tmem_ptr_smem;

  const int64_t n_tiles = (hidden + kSm100TileContractN - 1) /
      kSm100TileContractN;
  for (int64_t task_idx = blockIdx.x; task_idx < total_tasks; task_idx += gridDim.x) {
    const GroupedGemmTile tile =
        decode_grouped_gemm_tile(task_idx, tile_offsets, num_local_experts, n_tiles);
    if (tile.local_expert < 0) {
      continue;
    }
    compute_bf16_umma_linear_tile_to_bf16</*kBMnMajor=*/true>(
        h_map,
        down_weight_map,
        expert_counts,
        token_offsets,
        tile.local_expert,
        tile.local_m_tile,
        tile.n_tile,
        intermediate,
        hidden,
        tile.local_expert * intermediate,
        packed_expert_out,
        smem_a,
        smem_b,
        &tma_barrier,
        &umma_full_barrier,
        &tmem_ptr_smem);
  }
}

OLMO_BF16_MEGA_DEVICE inline void atomic_add_i64(int64_t* ptr, int64_t value) {
  atomicAdd(
      reinterpret_cast<unsigned long long*>(ptr),
      static_cast<unsigned long long>(value));
}

OLMO_BF16_MEGA_DEVICE inline int64_t atomic_fetch_add_i64(int64_t* ptr, int64_t value) {
  return static_cast<int64_t>(atomicAdd(
      reinterpret_cast<unsigned long long*>(ptr),
      static_cast<unsigned long long>(value)));
}

OLMO_BF16_MEGA_DEVICE inline void write_task_debug_row(
    int64_t* row,
    megakernel::ForwardTask task,
    int64_t total_tasks,
    int64_t gemm_tasks) {
  row[0] = static_cast<int64_t>(task.kind);
  row[1] = task.ordinal;
  row[2] = task.local_expert;
  row[3] = task.m_tile;
  row[4] = task.n_tile;
  row[5] = task.route_task;
  row[6] = total_tasks;
  row[7] = gemm_tasks;
}

OLMO_BF16_MEGA_DEVICE inline void persistent_grid_barrier(
    uint32_t* barrier_count,
    uint32_t* barrier_sense);

OLMO_BF16_MEGA_DEVICE inline GroupedGemmTile decode_grouped_gemm_tile(
    int64_t task_idx,
    const int64_t* tile_offsets,
    int64_t num_local_experts,
    int64_t n_tiles) {
  GroupedGemmTile tile;
  tile.global_m_tile = task_idx / n_tiles;
  tile.n_tile = task_idx - tile.global_m_tile * n_tiles;
  for (int64_t expert_idx = 0; expert_idx < num_local_experts; ++expert_idx) {
    const int64_t start = tile_offsets[expert_idx];
    const int64_t end = tile_offsets[expert_idx + 1];
    if (tile.global_m_tile >= start && tile.global_m_tile < end) {
      tile.local_expert = expert_idx;
      tile.local_m_tile = tile.global_m_tile - start;
      return tile;
    }
  }
  return tile;
}

__global__ void f1_forward_contract_kernel(
    megakernel::ForwardPlan plan,
    int64_t* debug) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t task_idx = blockIdx.x * blockDim.x + threadIdx.x;
       task_idx < plan.f1_total_tasks;
       task_idx += stride) {
    const megakernel::ForwardTask task = megakernel::decode_f1_task(plan, task_idx);
    const int64_t kind = static_cast<int64_t>(task.kind);
    if (kind >= 0 && kind < kForwardTaskKindCount) {
      atomic_add_i64(debug + kind, 1);
    }
  }

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    write_task_debug_row(
        debug + 2 * kForwardPlanDebugCols,
        megakernel::decode_f1_task(plan, plan.f1_total_tasks - 1),
        plan.f1_total_tasks,
        plan.f1_gemm_tasks);
  }
}

template <typename IndexT>
__global__ void f1_count_route_experts_kernel(
    const IndexT* route_expert_indices,
    int64_t num_routes,
    int64_t num_local_experts,
    int64_t* expert_counts) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t route_idx = blockIdx.x * blockDim.x + threadIdx.x;
       route_idx < num_routes;
       route_idx += stride) {
    const int64_t expert_idx = static_cast<int64_t>(route_expert_indices[route_idx]);
    if (expert_idx >= 0 && expert_idx < num_local_experts) {
      atomic_add_i64(expert_counts + expert_idx, 1);
    }
  }
}

__global__ void f1_prefix_route_counts_kernel(
    const int64_t* expert_counts,
    int64_t num_local_experts,
    int64_t* expert_offsets,
    int64_t* expert_cursors) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  int64_t offset = 0;
  for (int64_t expert_idx = 0; expert_idx < num_local_experts; ++expert_idx) {
    expert_offsets[expert_idx] = offset;
    expert_cursors[expert_idx] = offset;
    offset += expert_counts[expert_idx];
  }
  expert_offsets[num_local_experts] = offset;
}

__global__ void f1_grouped_gemm_metadata_kernel(
    const int64_t* expert_counts,
    int64_t num_local_experts,
    int64_t block_m,
    int64_t* token_offsets,
    int64_t* token_cursors,
    int64_t* tile_counts,
    int64_t* tile_offsets,
    int64_t* num_total_m_tiles) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  int64_t token_offset = 0;
  int64_t tile_offset = 0;
  for (int64_t expert_idx = 0; expert_idx < num_local_experts; ++expert_idx) {
    const int64_t count = expert_counts[expert_idx];
    const int64_t tiles = (count + block_m - 1) / block_m;
    token_offsets[expert_idx] = token_offset;
    token_cursors[expert_idx] = token_offset;
    tile_counts[expert_idx] = tiles;
    tile_offsets[expert_idx] = tile_offset;
    token_offset += count;
    tile_offset += tiles;
  }
  token_offsets[num_local_experts] = token_offset;
  tile_offsets[num_local_experts] = tile_offset;
  num_total_m_tiles[0] = tile_offset;
}

template <typename IndexT>
__global__ void f1_pack_route_indices_kernel(
    const IndexT* route_expert_indices,
    int64_t num_routes,
    int64_t num_local_experts,
    int64_t* expert_cursors,
    int64_t* packed_token_topk_indices) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t route_idx = blockIdx.x * blockDim.x + threadIdx.x;
       route_idx < num_routes;
       route_idx += stride) {
    const int64_t expert_idx = static_cast<int64_t>(route_expert_indices[route_idx]);
    if (expert_idx >= 0 && expert_idx < num_local_experts) {
      const int64_t packed_idx = atomic_fetch_add_i64(expert_cursors + expert_idx, 1);
      packed_token_topk_indices[packed_idx] = route_idx;
    }
  }
}

__global__ void f1_pack_input_rows_kernel(
    const __nv_bfloat16* source_input,
    const float* probs,
    const int64_t* packed_token_topk_indices,
    int64_t num_route_slots,
    int64_t top_k,
    int64_t hidden,
    __nv_bfloat16* packed_input,
    float* packed_probs) {
  const int64_t num_values = num_route_slots * hidden;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t value_idx = blockIdx.x * blockDim.x + threadIdx.x;
       value_idx < num_values;
       value_idx += stride) {
    const int64_t route_slot = value_idx / hidden;
    const int64_t hidden_idx = value_idx - route_slot * hidden;
    const int64_t token_topk_idx = packed_token_topk_indices[route_slot];
    if (token_topk_idx < 0) {
      continue;
    }
    const int64_t token_idx = token_topk_idx / top_k;
    packed_input[value_idx] = source_input[token_idx * hidden + hidden_idx];
    if (hidden_idx == 0) {
      packed_probs[route_slot] = probs[token_topk_idx];
    }
  }
}

__global__ void f2_forward_contract_kernel(
    megakernel::ForwardPlan plan,
    int64_t* debug) {
  int64_t* f2_counts = debug + kForwardPlanDebugCols;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t task_idx = blockIdx.x * blockDim.x + threadIdx.x;
       task_idx < plan.f2_total_tasks;
       task_idx += stride) {
    const megakernel::ForwardTask task = megakernel::decode_f2_task(plan, task_idx);
    const int64_t kind = static_cast<int64_t>(task.kind);
    if (kind >= 0 && kind < kForwardTaskKindCount) {
      atomic_add_i64(f2_counts + kind, 1);
    }
  }

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    write_task_debug_row(
        debug + 3 * kForwardPlanDebugCols,
        megakernel::decode_f2_task(plan, plan.f2_gemm_tasks - 1),
        plan.f2_total_tasks,
        plan.f2_gemm_tasks);
  }
}

__global__ void grouped_gemm_tile_contract_kernel(
    const int64_t* tile_offsets,
    int64_t num_local_experts,
    int64_t n_tiles,
    int64_t max_tasks,
    int64_t* debug) {
  const int64_t total_m_tiles = tile_offsets[num_local_experts];
  const int64_t total_tasks = total_m_tiles * n_tiles;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t task_idx = blockIdx.x * blockDim.x + threadIdx.x;
       task_idx < max_tasks;
       task_idx += stride) {
    if (task_idx >= total_tasks) {
      continue;
    }
    const GroupedGemmTile tile =
        decode_grouped_gemm_tile(task_idx, tile_offsets, num_local_experts, n_tiles);
    if (tile.local_expert >= 0) {
      atomic_add_i64(debug + tile.local_expert * 4, 1);
    }
  }

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int64_t* sample = debug + num_local_experts * 4;
    const GroupedGemmTile last =
        decode_grouped_gemm_tile(total_tasks - 1, tile_offsets, num_local_experts, n_tiles);
    sample[0] = last.local_expert;
    sample[1] = last.global_m_tile;
    sample[2] = last.local_m_tile;
    sample[3] = last.n_tile;

    int64_t* summary = debug + (num_local_experts + 1) * 4;
    summary[0] = total_tasks;
    summary[1] = total_m_tiles;
    summary[2] = n_tiles;
    summary[3] = max_tasks;
  }
}

__global__ void f2_scatter_packed_rows_kernel(
    const __nv_bfloat16* packed_expert_out,
    const int64_t* packed_token_topk_indices,
    int64_t num_route_slots,
    int64_t hidden,
    __nv_bfloat16* gathered_out) {
  const int64_t num_values = num_route_slots * hidden;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t value_idx = blockIdx.x * blockDim.x + threadIdx.x;
       value_idx < num_values;
       value_idx += stride) {
    const int64_t route_slot = value_idx / hidden;
    const int64_t hidden_idx = value_idx - route_slot * hidden;
    const int64_t token_topk_idx = packed_token_topk_indices[route_slot];
    if (token_topk_idx >= 0) {
      gathered_out[token_topk_idx * hidden + hidden_idx] = packed_expert_out[value_idx];
    }
  }
}

__global__ void f2_reduce_topk_rows_kernel(
    const __nv_bfloat16* gathered_out,
    const float* probs,
    int64_t num_tokens,
    int64_t top_k,
    int64_t hidden,
    __nv_bfloat16* out) {
  const int64_t num_values = num_tokens * hidden;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t value_idx = blockIdx.x * blockDim.x + threadIdx.x;
       value_idx < num_values;
       value_idx += stride) {
    const int64_t token_idx = value_idx / hidden;
    const int64_t hidden_idx = value_idx - token_idx * hidden;
    float acc = 0.0f;
    for (int64_t topk_idx = 0; topk_idx < top_k; ++topk_idx) {
      const int64_t token_topk_idx = token_idx * top_k + topk_idx;
      const float value =
          __bfloat162float(gathered_out[token_topk_idx * hidden + hidden_idx]);
      acc += value * probs[token_topk_idx];
    }
    out[value_idx] = __float2bfloat16(acc);
  }
}

__global__ void copy_bf16_rows_kernel(
    const __nv_bfloat16* src,
    int64_t rows,
    int64_t cols,
    __nv_bfloat16* dst) {
  const int64_t num_values = rows * cols;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t value_idx = blockIdx.x * blockDim.x + threadIdx.x;
       value_idx < num_values;
       value_idx += stride) {
    dst[value_idx] = src[value_idx];
  }
}

__global__ void grouped_w1_wmma_kernel(
    const __nv_bfloat16* packed_input,
    const __nv_bfloat16* up_gate_weight,
    const int64_t* expert_counts,
    const int64_t* token_offsets,
    const int64_t* tile_offsets,
    int64_t num_local_experts,
    int64_t num_route_slots,
    int64_t hidden,
    int64_t out_features,
    int64_t n_tiles,
    int64_t max_tasks,
    __nv_bfloat16* w1_out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  __shared__ __nv_bfloat16 a_tile[kWmmaM * kWmmaK];
  __shared__ __nv_bfloat16 b_tile[kWmmaK * kWmmaN];
  __shared__ float c_tile[kWmmaM * kWmmaN];

  for (int64_t task_idx = blockIdx.x; task_idx < max_tasks; task_idx += gridDim.x) {
    const GroupedGemmTile tile =
        decode_grouped_gemm_tile(task_idx, tile_offsets, num_local_experts, n_tiles);
    if (tile.local_expert < 0) {
      continue;
    }

    const int64_t expert_count = expert_counts[tile.local_expert];
    const int64_t m_start = token_offsets[tile.local_expert] + tile.local_m_tile * kWmmaM;
    const int64_t n_start = tile.n_tile * kWmmaN;
    if (m_start >= num_route_slots || n_start >= out_features) {
      continue;
    }

    wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    for (int64_t k_start = 0; k_start < hidden; k_start += kWmmaK) {
      for (int64_t idx = threadIdx.x; idx < kWmmaM * kWmmaK; idx += blockDim.x) {
        const int64_t m = idx / kWmmaK;
        const int64_t k = idx - m * kWmmaK;
        const int64_t expert_local_row = tile.local_m_tile * kWmmaM + m;
        const bool valid = expert_local_row < expert_count && k_start + k < hidden;
        a_tile[idx] = valid ? packed_input[(m_start + m) * hidden + k_start + k]
                            : __float2bfloat16(0.0f);
      }
      for (int64_t idx = threadIdx.x; idx < kWmmaK * kWmmaN; idx += blockDim.x) {
        const int64_t k = idx % kWmmaK;
        const int64_t n = idx / kWmmaK;
        const bool valid = k_start + k < hidden && n_start + n < out_features;
        b_tile[k + n * kWmmaK] = valid
            ? up_gate_weight[
                  (tile.local_expert * out_features + n_start + n) * hidden + k_start + k]
            : __float2bfloat16(0.0f);
      }
      __syncthreads();

      wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, __nv_bfloat16, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, __nv_bfloat16, wmma::col_major> b_frag;
      wmma::load_matrix_sync(a_frag, a_tile, kWmmaK);
      wmma::load_matrix_sync(b_frag, b_tile, kWmmaK);
      wmma::mma_sync(acc, a_frag, b_frag, acc);
      __syncthreads();
    }

    wmma::store_matrix_sync(c_tile, acc, kWmmaN, wmma::mem_row_major);
    __syncthreads();
    for (int64_t idx = threadIdx.x; idx < kWmmaM * kWmmaN; idx += blockDim.x) {
      const int64_t m = idx / kWmmaN;
      const int64_t n = idx - m * kWmmaN;
      const int64_t expert_local_row = tile.local_m_tile * kWmmaM + m;
      if (expert_local_row < expert_count && n_start + n < out_features) {
        w1_out[(m_start + m) * out_features + n_start + n] =
            __float2bfloat16(c_tile[idx]);
      }
    }
    __syncthreads();
  }
#else
  (void)packed_input;
  (void)up_gate_weight;
  (void)expert_counts;
  (void)token_offsets;
  (void)tile_offsets;
  (void)num_local_experts;
  (void)num_route_slots;
  (void)hidden;
  (void)out_features;
  (void)n_tiles;
  (void)max_tasks;
  (void)w1_out;
#endif
}

__global__ void swiglu_forward_kernel(
    const __nv_bfloat16* up_gate,
    int64_t num_route_slots,
    int64_t intermediate,
    __nv_bfloat16* h) {
  const int64_t out_features = 2 * intermediate;
  const int64_t num_values = num_route_slots * intermediate;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t value_idx = blockIdx.x * blockDim.x + threadIdx.x;
       value_idx < num_values;
       value_idx += stride) {
    const int64_t route_slot = value_idx / intermediate;
    const int64_t col = value_idx - route_slot * intermediate;
    const float up =
        __bfloat162float(up_gate[route_slot * out_features + col]);
    const float gate =
        __bfloat162float(up_gate[route_slot * out_features + intermediate + col]);
    const float silu_gate = gate / (1.0f + expf(-gate));
    h[value_idx] = __float2bfloat16(up * silu_gate);
  }
}

__global__ void swiglu_forward_split_kernel(
    const __nv_bfloat16* up,
    const __nv_bfloat16* gate,
    int64_t num_route_slots,
    int64_t intermediate,
    __nv_bfloat16* h) {
  const int64_t num_values = num_route_slots * intermediate;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t value_idx = blockIdx.x * blockDim.x + threadIdx.x;
       value_idx < num_values;
       value_idx += stride) {
    const float up_value = __bfloat162float(up[value_idx]);
    const float gate_value = __bfloat162float(gate[value_idx]);
    const float silu_gate = gate_value / (1.0f + expf(-gate_value));
    h[value_idx] = __float2bfloat16(up_value * silu_gate);
  }
}

__global__ void grouped_w2_wmma_kernel(
    const __nv_bfloat16* h,
    const __nv_bfloat16* down_weight,
    const int64_t* expert_counts,
    const int64_t* token_offsets,
    const int64_t* tile_offsets,
    int64_t num_local_experts,
    int64_t num_route_slots,
    int64_t intermediate,
    int64_t hidden,
    int64_t n_tiles,
    int64_t max_tasks,
    __nv_bfloat16* packed_expert_out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  __shared__ __nv_bfloat16 a_tile[kWmmaM * kWmmaK];
  __shared__ __nv_bfloat16 b_tile[kWmmaK * kWmmaN];
  __shared__ float c_tile[kWmmaM * kWmmaN];

  for (int64_t task_idx = blockIdx.x; task_idx < max_tasks; task_idx += gridDim.x) {
    const GroupedGemmTile tile =
        decode_grouped_gemm_tile(task_idx, tile_offsets, num_local_experts, n_tiles);
    if (tile.local_expert < 0) {
      continue;
    }

    const int64_t expert_count = expert_counts[tile.local_expert];
    const int64_t m_start = token_offsets[tile.local_expert] + tile.local_m_tile * kWmmaM;
    const int64_t n_start = tile.n_tile * kWmmaN;
    if (m_start >= num_route_slots || n_start >= hidden) {
      continue;
    }

    wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    for (int64_t k_start = 0; k_start < intermediate; k_start += kWmmaK) {
      for (int64_t idx = threadIdx.x; idx < kWmmaM * kWmmaK; idx += blockDim.x) {
        const int64_t m = idx / kWmmaK;
        const int64_t k = idx - m * kWmmaK;
        const int64_t expert_local_row = tile.local_m_tile * kWmmaM + m;
        const bool valid = expert_local_row < expert_count && k_start + k < intermediate;
        a_tile[idx] = valid ? h[(m_start + m) * intermediate + k_start + k]
                            : __float2bfloat16(0.0f);
      }
      for (int64_t idx = threadIdx.x; idx < kWmmaK * kWmmaN; idx += blockDim.x) {
        const int64_t k = idx / kWmmaN;
        const int64_t n = idx - k * kWmmaN;
        const bool valid = k_start + k < intermediate && n_start + n < hidden;
        b_tile[k * kWmmaN + n] = valid
            ? down_weight[
                  (tile.local_expert * intermediate + k_start + k) * hidden + n_start + n]
            : __float2bfloat16(0.0f);
      }
      __syncthreads();

      wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, __nv_bfloat16, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, __nv_bfloat16, wmma::row_major> b_frag;
      wmma::load_matrix_sync(a_frag, a_tile, kWmmaK);
      wmma::load_matrix_sync(b_frag, b_tile, kWmmaN);
      wmma::mma_sync(acc, a_frag, b_frag, acc);
      __syncthreads();
    }

    wmma::store_matrix_sync(c_tile, acc, kWmmaN, wmma::mem_row_major);
    __syncthreads();
    for (int64_t idx = threadIdx.x; idx < kWmmaM * kWmmaN; idx += blockDim.x) {
      const int64_t m = idx / kWmmaN;
      const int64_t n = idx - m * kWmmaN;
      const int64_t expert_local_row = tile.local_m_tile * kWmmaM + m;
      if (expert_local_row < expert_count && n_start + n < hidden) {
        packed_expert_out[(m_start + m) * hidden + n_start + n] =
            __float2bfloat16(c_tile[idx]);
      }
    }
    __syncthreads();
  }
#else
  (void)h;
  (void)down_weight;
  (void)expert_counts;
  (void)token_offsets;
  (void)tile_offsets;
  (void)num_local_experts;
  (void)num_route_slots;
  (void)intermediate;
  (void)hidden;
  (void)n_tiles;
  (void)max_tasks;
  (void)packed_expert_out;
#endif
}

__global__ void standard_scheduler_seed_counts_kernel(
    void* workspace_base,
    const int64_t* expert_counts) {
  const layout::Workspace workspace(
      workspace_base,
      kStandardNumRanks,
      kStandardNumTotalExperts,
      kStandardNumMaxTokensPerRank,
      kStandardTopK);
  const uint32_t expert_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert_idx >= kStandardNumLocalExperts) {
    return;
  }
  const uint64_t ready_count =
      static_cast<uint64_t>(kStandardSchedulerSms * kStandardNumRanks);
  const uint64_t token_count =
      static_cast<uint64_t>(expert_counts[expert_idx]);
  *workspace.expert_recv_count_sum_ptr(expert_idx) =
      (ready_count << 32) | token_count;
}

__global__ void standard_scheduler_debug_kernel(
    void* workspace_base,
    int64_t* debug) {
  using Scheduler = sched::MegaMoEScheduler<
      kStandardBlockM,
      kStandardBlockN,
      kStandardBlockK,
      kStandardIntermediate * 2,
      kStandardHidden,
      kStandardHidden,
      kStandardIntermediate,
      kStandardNumLocalExperts,
      kStandardExpertsPerWave,
      kStandardSchedulerSms,
      kStandardNumRanks>;

  const layout::Workspace workspace(
      workspace_base,
      kStandardNumRanks,
      kStandardNumTotalExperts,
      kStandardNumMaxTokensPerRank,
      kStandardTopK);
  Scheduler scheduler(workspace);
  scheduler.for_each_block(
      [&](layout::BlockPhase phase,
          uint32_t local_expert_idx,
          uint32_t k_blocks,
          uint32_t m_idx,
          uint32_t n_idx) {
        if (threadIdx.x != 0 || local_expert_idx >= kStandardNumLocalExperts) {
          return;
        }
        int64_t* row = debug + static_cast<int64_t>(local_expert_idx) * 4;
        if (phase == layout::BlockPhase::Linear1) {
          atomic_add_i64(row + 0, 1);
        } else if (phase == layout::BlockPhase::Linear2) {
          atomic_add_i64(row + 1, 1);
        }
        (void)k_blocks;
        (void)m_idx;
        (void)n_idx;
      });

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int64_t* summary = debug + static_cast<int64_t>(kStandardNumLocalExperts) * 4;
    summary[0] = kStandardSchedulerSms;
    summary[1] = kStandardExpertsPerWave;
    summary[2] = kStandardIntermediate * 2 / kStandardBlockN;
    summary[3] = kStandardHidden / kStandardBlockN;
  }
}

__global__ void standard_ep_dispatch_metadata_debug_kernel(
    const int64_t* route_expert_indices,
    uint32_t num_tokens,
    void* workspace_base,
    uint64_t workspace_stride_bytes,
    uint32_t* barrier_state) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
  using Builder = dispatch::MetadataBuilder<
      kStandardNumTotalExperts,
      kStandardTopK,
      kStandardSchedulerSms,
      kStandardDispatchWarps,
      kStandardNumLocalExperts>;

  __shared__ uint32_t shared_expert_token_count[kStandardNumTotalExperts];
  const layout::Workspace workspace(
      workspace_base,
      kStandardNumRanks,
      kStandardNumTotalExperts,
      kStandardNumMaxTokensPerRank,
      kStandardTopK);
  Builder builder(workspace, shared_expert_token_count);
  const uint32_t thread_idx = threadIdx.x;
  const uint32_t warp_idx = threadIdx.x / 32u;
  const uint32_t lane_idx = threadIdx.x & 31u;
  const uint32_t num_threads = blockDim.x;
  const uint32_t sm_idx = blockIdx.x;
  dispatch::DebugMultiRankAddressMap address_map{
      workspace_base,
      workspace_stride_bytes,
      /*rank_idx=*/0u,
  };

  builder.clear_shared_counts(thread_idx, num_threads);
  __syncthreads();
  if (warp_idx < kStandardDispatchWarps) {
    builder.count_routes(route_expert_indices, num_tokens, sm_idx, warp_idx, lane_idx);
  }
  __syncthreads();
  builder.publish_sm_offsets(thread_idx, num_threads);

  persistent_grid_barrier(barrier_state, barrier_state + 1);

  if (warp_idx < kStandardDispatchWarps) {
    builder.write_source_indices(
        route_expert_indices,
        num_tokens,
        sm_idx,
        warp_idx,
        lane_idx,
        address_map);
  }

  persistent_grid_barrier(barrier_state, barrier_state + 1);

  if (blockIdx.x == 0) {
    builder.publish_recv_counts(thread_idx, num_threads, address_map);
  }
#else
  (void)route_expert_indices;
  (void)num_tokens;
  (void)workspace_base;
  (void)workspace_stride_bytes;
  (void)barrier_state;
#endif
}

__global__ void standard_ep_fill_workspace_base_ptrs_kernel(
    void* workspace_base,
    uint64_t workspace_stride_bytes,
    uint64_t* rank_workspace_bases) {
  const uint32_t rank_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (rank_idx < kStandardNumRanks) {
    rank_workspace_bases[rank_idx] =
        reinterpret_cast<uintptr_t>(workspace_base) +
        static_cast<uint64_t>(rank_idx) * workspace_stride_bytes;
  }
}

__global__ void standard_ep_dispatch_metadata_peer_map_debug_kernel(
    const int64_t* route_expert_indices,
    uint32_t num_tokens,
    const uint64_t* rank_workspace_bases,
    uint32_t* barrier_state) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
  using Builder = dispatch::MetadataBuilder<
      kStandardNumTotalExperts,
      kStandardTopK,
      kStandardSchedulerSms,
      kStandardDispatchWarps,
      kStandardNumLocalExperts>;

  __shared__ uint32_t shared_expert_token_count[kStandardNumTotalExperts];
  void* local_workspace_base = reinterpret_cast<void*>(rank_workspace_bases[0]);
  const layout::Workspace workspace(
      local_workspace_base,
      kStandardNumRanks,
      kStandardNumTotalExperts,
      kStandardNumMaxTokensPerRank,
      kStandardTopK);
  Builder builder(workspace, shared_expert_token_count);
  const uint32_t thread_idx = threadIdx.x;
  const uint32_t warp_idx = threadIdx.x / 32u;
  const uint32_t lane_idx = threadIdx.x & 31u;
  const uint32_t num_threads = blockDim.x;
  const uint32_t sm_idx = blockIdx.x;
  dispatch::PeerWorkspaceAddressMap address_map{
      local_workspace_base,
      rank_workspace_bases,
      /*rank_idx=*/0u,
  };

  builder.clear_shared_counts(thread_idx, num_threads);
  __syncthreads();
  if (warp_idx < kStandardDispatchWarps) {
    builder.count_routes(route_expert_indices, num_tokens, sm_idx, warp_idx, lane_idx);
  }
  __syncthreads();
  builder.publish_sm_offsets(thread_idx, num_threads);

  persistent_grid_barrier(barrier_state, barrier_state + 1);

  if (warp_idx < kStandardDispatchWarps) {
    builder.write_source_indices(
        route_expert_indices,
        num_tokens,
        sm_idx,
        warp_idx,
        lane_idx,
        address_map);
  }

  persistent_grid_barrier(barrier_state, barrier_state + 1);

  if (blockIdx.x == 0) {
    builder.publish_recv_counts(thread_idx, num_threads, address_map);
  }
#else
  (void)route_expert_indices;
  (void)num_tokens;
  (void)rank_workspace_bases;
  (void)barrier_state;
#endif
}

__global__ void standard_ep_dispatch_metadata_extract_kernel(
    const void* workspace_base,
    uint64_t workspace_stride_bytes,
    int64_t num_route_slots,
    int64_t* recv_counts,
    int64_t* recv_ready_counts,
    int64_t* src_token_topk_indices) {
  const int64_t total_pairs =
      static_cast<int64_t>(kStandardNumRanks) * kStandardNumLocalExperts;
  const int64_t total_src_values = total_pairs * num_route_slots;
  const int64_t total_values = total_pairs + total_src_values;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
       linear_idx < total_values;
       linear_idx += stride) {
    if (linear_idx < total_pairs) {
      const uint32_t rank_idx = static_cast<uint32_t>(linear_idx / kStandardNumLocalExperts);
      const uint32_t local_expert_idx =
          static_cast<uint32_t>(linear_idx - rank_idx * kStandardNumLocalExperts);
      const void* rank_base = reinterpret_cast<const void*>(
          reinterpret_cast<uintptr_t>(workspace_base) +
          static_cast<uint64_t>(rank_idx) * workspace_stride_bytes);
      const layout::Workspace workspace(
          const_cast<void*>(rank_base),
          kStandardNumRanks,
          kStandardNumTotalExperts,
          kStandardNumMaxTokensPerRank,
          kStandardTopK);
      const uint64_t value = *workspace.expert_recv_count_sum_ptr(local_expert_idx);
      recv_counts[linear_idx] = static_cast<int64_t>(value & 0xffffffffu);
      recv_ready_counts[linear_idx] = static_cast<int64_t>(value >> 32);
      continue;
    }

    const int64_t src_linear = linear_idx - total_pairs;
    const int64_t slot_idx = src_linear % num_route_slots;
    const int64_t pair_idx = src_linear / num_route_slots;
    const uint32_t rank_idx = static_cast<uint32_t>(pair_idx / kStandardNumLocalExperts);
    const uint32_t local_expert_idx =
        static_cast<uint32_t>(pair_idx - rank_idx * kStandardNumLocalExperts);
    const void* rank_base = reinterpret_cast<const void*>(
        reinterpret_cast<uintptr_t>(workspace_base) +
        static_cast<uint64_t>(rank_idx) * workspace_stride_bytes);
    const layout::Workspace workspace(
        const_cast<void*>(rank_base),
        kStandardNumRanks,
        kStandardNumTotalExperts,
        kStandardNumMaxTokensPerRank,
        kStandardTopK);
    const int64_t count =
        static_cast<int64_t>(*workspace.expert_recv_count_sum_ptr(local_expert_idx) & 0xffffffffu);
    const int64_t out_idx = pair_idx * num_route_slots + slot_idx;
    src_token_topk_indices[out_idx] = slot_idx < count
        ? static_cast<int64_t>(*workspace.src_token_topk_idx_ptr(
              local_expert_idx,
              /*rank_idx=*/0u,
              static_cast<uint32_t>(slot_idx)))
        : -1;
  }
}

__global__ void standard_ep_dispatch_global_counts_kernel(
    const void* workspace_base,
    uint64_t workspace_stride_bytes,
    int64_t* global_expert_counts) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t global_expert_idx = blockIdx.x * blockDim.x + threadIdx.x;
       global_expert_idx < kStandardNumTotalExperts;
       global_expert_idx += stride) {
    const uint32_t rank_idx =
        static_cast<uint32_t>(global_expert_idx / kStandardNumLocalExperts);
    const uint32_t local_expert_idx =
        static_cast<uint32_t>(global_expert_idx - rank_idx * kStandardNumLocalExperts);
    const void* rank_base = reinterpret_cast<const void*>(
        reinterpret_cast<uintptr_t>(workspace_base) +
        static_cast<uint64_t>(rank_idx) * workspace_stride_bytes);
    const layout::Workspace workspace(
        const_cast<void*>(rank_base),
        kStandardNumRanks,
        kStandardNumTotalExperts,
        kStandardNumMaxTokensPerRank,
        kStandardTopK);
    global_expert_counts[global_expert_idx] =
        static_cast<int64_t>(*workspace.expert_recv_count_sum_ptr(local_expert_idx) & 0xffffffffu);
  }
}

__global__ void standard_ep_dispatch_global_offsets_kernel(
    const int64_t* global_expert_counts,
    int64_t* global_expert_offsets) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  int64_t offset = 0;
  for (int64_t expert_idx = 0; expert_idx < kStandardNumTotalExperts; ++expert_idx) {
    global_expert_offsets[expert_idx] = offset;
    offset += global_expert_counts[expert_idx];
  }
  global_expert_offsets[kStandardNumTotalExperts] = offset;
}

__global__ void standard_ep_dispatch_pack_inputs_from_workspace_kernel(
    const void* workspace_base,
    uint64_t workspace_stride_bytes,
    const int64_t* global_expert_counts,
    const int64_t* global_expert_offsets,
    int64_t* packed_token_topk_indices) {
  const int64_t total_routes = global_expert_offsets[kStandardNumTotalExperts];
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
       linear_idx < total_routes;
       linear_idx += stride) {
    int64_t expert_idx = 0;
    while (expert_idx + 1 <= kStandardNumTotalExperts &&
           linear_idx >= global_expert_offsets[expert_idx + 1]) {
      ++expert_idx;
    }
    const int64_t slot = linear_idx - global_expert_offsets[expert_idx];
    const uint32_t rank_idx =
        static_cast<uint32_t>(expert_idx / kStandardNumLocalExperts);
    const uint32_t local_expert_idx =
        static_cast<uint32_t>(expert_idx - rank_idx * kStandardNumLocalExperts);
    const void* rank_base = reinterpret_cast<const void*>(
        reinterpret_cast<uintptr_t>(workspace_base) +
        static_cast<uint64_t>(rank_idx) * workspace_stride_bytes);
    const layout::Workspace workspace(
        const_cast<void*>(rank_base),
        kStandardNumRanks,
        kStandardNumTotalExperts,
        kStandardNumMaxTokensPerRank,
        kStandardTopK);
    packed_token_topk_indices[linear_idx] = slot < global_expert_counts[expert_idx]
        ? static_cast<int64_t>(*workspace.src_token_topk_idx_ptr(
              local_expert_idx,
              /*rank_idx=*/0u,
              static_cast<uint32_t>(slot)))
        : -1;
  }
}

__global__ void standard_ep_dispatch_copy_packed_inputs_kernel(
    const __nv_bfloat16* source_input,
    const int64_t* packed_token_topk_indices,
    int64_t total_routes,
    int64_t hidden,
    __nv_bfloat16* packed_input) {
  const int64_t total_values = total_routes * hidden;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t value_idx = blockIdx.x * blockDim.x + threadIdx.x;
       value_idx < total_values;
       value_idx += stride) {
    const int64_t packed_idx = value_idx / hidden;
    const int64_t hidden_idx = value_idx - packed_idx * hidden;
    const int64_t token_topk_idx = packed_token_topk_indices[packed_idx];
    if (token_topk_idx >= 0) {
      const int64_t token_idx = token_topk_idx / kStandardTopK;
      packed_input[value_idx] = source_input[token_idx * hidden + hidden_idx];
    }
  }
}

struct WaveTile {
  int64_t local_expert = -1;
  int64_t local_m_tile = -1;
  int64_t n_tile = -1;
};

OLMO_BF16_MEGA_DEVICE inline int64_t expert_m_tiles(
    const int64_t* expert_counts,
    int64_t expert_idx,
    int64_t block_m) {
  return (expert_counts[expert_idx] + block_m - 1) / block_m;
}

OLMO_BF16_MEGA_DEVICE inline int64_t wave_task_count(
    const int64_t* expert_counts,
    int64_t wave_start_expert,
    int64_t wave_end_expert,
    int64_t block_m,
    int64_t n_tiles) {
  int64_t tasks = 0;
  for (int64_t expert_idx = wave_start_expert;
       expert_idx < wave_end_expert;
       ++expert_idx) {
    tasks += expert_m_tiles(expert_counts, expert_idx, block_m) * n_tiles;
  }
  return tasks;
}

OLMO_BF16_MEGA_DEVICE inline WaveTile decode_wave_tile(
    int64_t task_idx,
    const int64_t* expert_counts,
    int64_t wave_start_expert,
    int64_t wave_end_expert,
    int64_t block_m,
    int64_t n_tiles) {
  WaveTile tile;
  for (int64_t expert_idx = wave_start_expert;
       expert_idx < wave_end_expert;
       ++expert_idx) {
    const int64_t m_tiles = expert_m_tiles(expert_counts, expert_idx, block_m);
    const int64_t expert_tasks = m_tiles * n_tiles;
    if (task_idx < expert_tasks) {
      tile.local_expert = expert_idx;
      tile.local_m_tile = task_idx / n_tiles;
      tile.n_tile = task_idx - tile.local_m_tile * n_tiles;
      return tile;
    }
    task_idx -= expert_tasks;
  }
  return tile;
}

OLMO_BF16_MEGA_DEVICE inline void persistent_grid_barrier(
    uint32_t* barrier_count,
    uint32_t* barrier_sense) {
  __syncthreads();
  if (threadIdx.x == 0) {
    const uint32_t target =
        *reinterpret_cast<volatile uint32_t*>(barrier_sense) + 1u;
    __threadfence();
    const uint32_t old = atomicAdd(barrier_count, 1u);
    if (old == gridDim.x - 1u) {
      *barrier_count = 0u;
      __threadfence();
      *reinterpret_cast<volatile uint32_t*>(barrier_sense) = target;
    } else {
      while (*reinterpret_cast<volatile uint32_t*>(barrier_sense) < target) {
      }
    }
  }
  __syncthreads();
}

template <uint32_t kTag, typename AddressMap>
OLMO_BF16_MEGA_DEVICE inline void standard_ep_phase_barrier(
    bool enable_cross_rank_barrier,
    const layout::Workspace& workspace,
    AddressMap address_map,
    uint32_t sm_idx,
    uint32_t thread_idx,
    uint32_t* barrier_state) {
  if (enable_cross_rank_barrier) {
    barrier::cross_rank_barrier<
        kStandardNumRanks,
        kStandardSchedulerSms,
        32,
        /*kGridSyncIndex=*/0,
        kTag>(
        workspace,
        address_map,
        sm_idx,
        thread_idx,
        []() { __syncthreads(); });
  } else {
    persistent_grid_barrier(barrier_state, barrier_state + 1);
  }
}

OLMO_BF16_MEGA_DEVICE inline void compute_w1_swiglu_wmma_tile(
    const __nv_bfloat16* packed_input,
    const __nv_bfloat16* up_gate_weight,
    const int64_t* expert_counts,
    const int64_t* token_offsets,
    int64_t expert_idx,
    int64_t local_m_tile,
    int64_t n_tile,
    int64_t num_route_slots,
    int64_t hidden,
    int64_t intermediate,
    __nv_bfloat16* h,
    __nv_bfloat16* a_tile,
    __nv_bfloat16* b_tile,
    float* c_up_tile,
    float* c_gate_tile) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  const int64_t expert_count = expert_counts[expert_idx];
  const int64_t m_start = token_offsets[expert_idx] + local_m_tile * kWmmaM;
  const int64_t n_start = n_tile * kWmmaN;
  if (m_start >= num_route_slots || n_start >= intermediate) {
    return;
  }

  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> up_acc;
  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> gate_acc;
  wmma::fill_fragment(up_acc, 0.0f);
  wmma::fill_fragment(gate_acc, 0.0f);

  for (int64_t k_start = 0; k_start < hidden; k_start += kWmmaK) {
    for (int64_t idx = threadIdx.x; idx < kWmmaM * kWmmaK; idx += blockDim.x) {
      const int64_t m = idx / kWmmaK;
      const int64_t k = idx - m * kWmmaK;
      const int64_t expert_local_row = local_m_tile * kWmmaM + m;
      const bool valid = expert_local_row < expert_count && k_start + k < hidden;
      a_tile[idx] = valid ? packed_input[(m_start + m) * hidden + k_start + k]
                          : __float2bfloat16(0.0f);
    }
    for (int64_t idx = threadIdx.x; idx < kWmmaK * kWmmaN; idx += blockDim.x) {
      const int64_t k = idx % kWmmaK;
      const int64_t n = idx / kWmmaK;
      const bool valid = k_start + k < hidden && n_start + n < intermediate;
      b_tile[k + n * kWmmaK] = valid
          ? up_gate_weight[
                (expert_idx * (2 * intermediate) + n_start + n) * hidden + k_start + k]
          : __float2bfloat16(0.0f);
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::load_matrix_sync(a_frag, a_tile, kWmmaK);
    wmma::load_matrix_sync(b_frag, b_tile, kWmmaK);
    wmma::mma_sync(up_acc, a_frag, b_frag, up_acc);
    __syncthreads();

    for (int64_t idx = threadIdx.x; idx < kWmmaK * kWmmaN; idx += blockDim.x) {
      const int64_t k = idx % kWmmaK;
      const int64_t n = idx / kWmmaK;
      const bool valid = k_start + k < hidden && n_start + n < intermediate;
      b_tile[k + n * kWmmaK] = valid
          ? up_gate_weight[
                (expert_idx * (2 * intermediate) + intermediate + n_start + n) *
                    hidden +
                k_start + k]
          : __float2bfloat16(0.0f);
    }
    __syncthreads();
    wmma::load_matrix_sync(b_frag, b_tile, kWmmaK);
    wmma::mma_sync(gate_acc, a_frag, b_frag, gate_acc);
    __syncthreads();
  }

  wmma::store_matrix_sync(c_up_tile, up_acc, kWmmaN, wmma::mem_row_major);
  wmma::store_matrix_sync(c_gate_tile, gate_acc, kWmmaN, wmma::mem_row_major);
  __syncthreads();
  for (int64_t idx = threadIdx.x; idx < kWmmaM * kWmmaN; idx += blockDim.x) {
    const int64_t m = idx / kWmmaN;
    const int64_t n = idx - m * kWmmaN;
    const int64_t expert_local_row = local_m_tile * kWmmaM + m;
    if (expert_local_row < expert_count && n_start + n < intermediate) {
      const float up = __bfloat162float(__float2bfloat16(c_up_tile[idx]));
      const float gate = __bfloat162float(__float2bfloat16(c_gate_tile[idx]));
      const float silu_gate = gate / (1.0f + expf(-gate));
      h[(m_start + m) * intermediate + n_start + n] =
          __float2bfloat16(up * silu_gate);
    }
  }
  __syncthreads();
#else
  (void)packed_input;
  (void)up_gate_weight;
  (void)expert_counts;
  (void)token_offsets;
  (void)expert_idx;
  (void)local_m_tile;
  (void)n_tile;
  (void)num_route_slots;
  (void)hidden;
  (void)intermediate;
  (void)h;
  (void)a_tile;
  (void)b_tile;
  (void)c_up_tile;
  (void)c_gate_tile;
#endif
}

OLMO_BF16_MEGA_DEVICE inline void compute_w2_wmma_tile(
    const __nv_bfloat16* h,
    const __nv_bfloat16* down_weight,
    const int64_t* expert_counts,
    const int64_t* token_offsets,
    int64_t expert_idx,
    int64_t local_m_tile,
    int64_t n_tile,
    int64_t num_route_slots,
    int64_t intermediate,
    int64_t hidden,
    __nv_bfloat16* packed_expert_out,
    __nv_bfloat16* a_tile,
    __nv_bfloat16* b_tile,
    float* c_tile) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  const int64_t expert_count = expert_counts[expert_idx];
  const int64_t m_start = token_offsets[expert_idx] + local_m_tile * kWmmaM;
  const int64_t n_start = n_tile * kWmmaN;
  if (m_start >= num_route_slots || n_start >= hidden) {
    return;
  }

  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> acc;
  wmma::fill_fragment(acc, 0.0f);

  for (int64_t k_start = 0; k_start < intermediate; k_start += kWmmaK) {
    for (int64_t idx = threadIdx.x; idx < kWmmaM * kWmmaK; idx += blockDim.x) {
      const int64_t m = idx / kWmmaK;
      const int64_t k = idx - m * kWmmaK;
      const int64_t expert_local_row = local_m_tile * kWmmaM + m;
      const bool valid = expert_local_row < expert_count && k_start + k < intermediate;
      a_tile[idx] = valid ? h[(m_start + m) * intermediate + k_start + k]
                          : __float2bfloat16(0.0f);
    }
    for (int64_t idx = threadIdx.x; idx < kWmmaK * kWmmaN; idx += blockDim.x) {
      const int64_t k = idx / kWmmaN;
      const int64_t n = idx - k * kWmmaN;
      const bool valid = k_start + k < intermediate && n_start + n < hidden;
      b_tile[k * kWmmaN + n] = valid
          ? down_weight[(expert_idx * intermediate + k_start + k) * hidden + n_start + n]
          : __float2bfloat16(0.0f);
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, __nv_bfloat16, wmma::row_major> b_frag;
    wmma::load_matrix_sync(a_frag, a_tile, kWmmaK);
    wmma::load_matrix_sync(b_frag, b_tile, kWmmaN);
    wmma::mma_sync(acc, a_frag, b_frag, acc);
    __syncthreads();
  }

  wmma::store_matrix_sync(c_tile, acc, kWmmaN, wmma::mem_row_major);
  __syncthreads();
  for (int64_t idx = threadIdx.x; idx < kWmmaM * kWmmaN; idx += blockDim.x) {
    const int64_t m = idx / kWmmaN;
    const int64_t n = idx - m * kWmmaN;
    const int64_t expert_local_row = local_m_tile * kWmmaM + m;
    if (expert_local_row < expert_count && n_start + n < hidden) {
      packed_expert_out[(m_start + m) * hidden + n_start + n] =
          __float2bfloat16(c_tile[idx]);
    }
  }
  __syncthreads();
#else
  (void)h;
  (void)down_weight;
  (void)expert_counts;
  (void)token_offsets;
  (void)expert_idx;
  (void)local_m_tile;
  (void)n_tile;
  (void)num_route_slots;
  (void)intermediate;
  (void)hidden;
  (void)packed_expert_out;
  (void)a_tile;
  (void)b_tile;
  (void)c_tile;
#endif
}

OLMO_BF16_MEGA_DEVICE inline void compute_swiglu_split_tile(
    const __nv_bfloat16* up,
    const __nv_bfloat16* gate,
    const int64_t* expert_counts,
    const int64_t* token_offsets,
    int64_t expert_idx,
    int64_t local_m_tile,
    int64_t n_tile,
    int64_t intermediate,
    __nv_bfloat16* h) {
  const int64_t expert_count = expert_counts[expert_idx];
  const int64_t m_start = token_offsets[expert_idx] +
      local_m_tile * static_cast<int64_t>(kSm100TileContractM);
  const int64_t n_start = n_tile * static_cast<int64_t>(kSm100TileContractN);
  for (int64_t idx = threadIdx.x;
       idx < static_cast<int64_t>(kSm100TileContractM) *
               static_cast<int64_t>(kSm100TileContractN);
       idx += blockDim.x) {
    const int64_t m = idx / static_cast<int64_t>(kSm100TileContractN);
    const int64_t n = idx - m * static_cast<int64_t>(kSm100TileContractN);
    const int64_t expert_local_row =
        local_m_tile * static_cast<int64_t>(kSm100TileContractM) + m;
    if (expert_local_row < expert_count && n_start + n < intermediate) {
      const int64_t value_idx = (m_start + m) * intermediate + n_start + n;
      const float up_value = __bfloat162float(up[value_idx]);
      const float gate_value = __bfloat162float(gate[value_idx]);
      const float silu_gate = gate_value / (1.0f + expf(-gate_value));
      h[value_idx] = __float2bfloat16(up_value * silu_gate);
    }
  }
  __syncthreads();
}

struct alignas(1024) StandardEpFullForwardSharedStorage {
  alignas(1024) __nv_bfloat16 umma_smem_a[
      kSm100TileContractM * kSm100TileContractK];
  alignas(1024) __nv_bfloat16 umma_smem_b[
      kSm100TileContractN * kSm100TileContractK];
  alignas(8) ptx::MBarrier umma_tma_barrier;
  alignas(8) ptx::MBarrier umma_full_barrier;
  uint32_t umma_tmem_ptr_smem;
  alignas(16) uint32_t shared_expert_token_count[kStandardNumTotalExperts];
  alignas(16) __nv_bfloat16 a_tile[kWmmaM * kWmmaK];
  alignas(16) __nv_bfloat16 b_tile[kWmmaK * kWmmaN];
  alignas(16) float c_tile[kWmmaM * kWmmaN];
  alignas(16) float c_gate_tile[kWmmaM * kWmmaN];
};

__global__ void local_persistent_forward_debug_kernel(
    const __nv_bfloat16* packed_input,
    const __nv_bfloat16* up_gate_weight,
    const __nv_bfloat16* down_weight,
    const int64_t* expert_counts,
    const int64_t* token_offsets,
    int64_t num_local_experts,
    int64_t num_experts_per_wave,
    int64_t num_route_slots,
    int64_t hidden,
    int64_t intermediate,
    uint32_t* barrier_state,
    __nv_bfloat16* h,
    __nv_bfloat16* packed_expert_out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  __shared__ __nv_bfloat16 a_tile[kWmmaM * kWmmaK];
  __shared__ __nv_bfloat16 b_tile[kWmmaK * kWmmaN];
  __shared__ float c_tile[kWmmaM * kWmmaN];
  __shared__ float c_gate_tile[kWmmaM * kWmmaN];

  const int64_t l1_n_tiles = (intermediate + kWmmaN - 1) / kWmmaN;
  const int64_t l2_n_tiles = (hidden + kWmmaN - 1) / kWmmaN;

  for (int64_t wave_start = 0;
       wave_start < num_local_experts;
       wave_start += num_experts_per_wave) {
    const int64_t wave_end = wave_start + num_experts_per_wave < num_local_experts
        ? wave_start + num_experts_per_wave
        : num_local_experts;
    const int64_t l1_tasks = wave_task_count(
        expert_counts,
        wave_start,
        wave_end,
        kWmmaM,
        l1_n_tiles);
    for (int64_t task_idx = blockIdx.x; task_idx < l1_tasks; task_idx += gridDim.x) {
      const WaveTile tile =
          decode_wave_tile(task_idx, expert_counts, wave_start, wave_end, kWmmaM, l1_n_tiles);
      if (tile.local_expert >= 0) {
        compute_w1_swiglu_wmma_tile(
            packed_input,
            up_gate_weight,
            expert_counts,
            token_offsets,
            tile.local_expert,
            tile.local_m_tile,
            tile.n_tile,
            num_route_slots,
            hidden,
            intermediate,
            h,
            a_tile,
            b_tile,
            c_tile,
            c_gate_tile);
      }
    }

    persistent_grid_barrier(barrier_state, barrier_state + 1);

    const int64_t l2_tasks = wave_task_count(
        expert_counts,
        wave_start,
        wave_end,
        kWmmaM,
        l2_n_tiles);
    for (int64_t task_idx = blockIdx.x; task_idx < l2_tasks; task_idx += gridDim.x) {
      const WaveTile tile =
          decode_wave_tile(task_idx, expert_counts, wave_start, wave_end, kWmmaM, l2_n_tiles);
      if (tile.local_expert >= 0) {
        compute_w2_wmma_tile(
            h,
            down_weight,
            expert_counts,
            token_offsets,
            tile.local_expert,
            tile.local_m_tile,
            tile.n_tile,
            num_route_slots,
            intermediate,
            hidden,
            packed_expert_out,
            a_tile,
            b_tile,
            c_tile);
      }
    }

    persistent_grid_barrier(barrier_state, barrier_state + 1);
  }
#else
  (void)packed_input;
  (void)up_gate_weight;
  (void)down_weight;
  (void)expert_counts;
  (void)token_offsets;
  (void)num_local_experts;
  (void)num_experts_per_wave;
  (void)num_route_slots;
  (void)hidden;
  (void)intermediate;
  (void)barrier_state;
  (void)h;
  (void)packed_expert_out;
#endif
}

__global__ void local_full_forward_megakernel_debug_kernel(
    const __nv_bfloat16* source_input,
    const int64_t* route_expert_indices,
    const float* probs,
    const __nv_bfloat16* up_gate_weight,
    const __nv_bfloat16* down_weight,
    int64_t num_tokens,
    int64_t top_k,
    int64_t num_local_experts,
    int64_t num_experts_per_wave,
    int64_t hidden,
    int64_t intermediate,
    int64_t num_route_slots,
    int64_t* expert_counts,
    int64_t* token_offsets,
    int64_t* expert_cursors,
    int64_t* packed_token_topk_indices,
    int64_t* route_to_slot,
    uint32_t* barrier_state,
    __nv_bfloat16* packed_input,
    __nv_bfloat16* h,
    __nv_bfloat16* packed_expert_out,
    __nv_bfloat16* gathered_out,
    __nv_bfloat16* out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  __shared__ __nv_bfloat16 a_tile[kWmmaM * kWmmaK];
  __shared__ __nv_bfloat16 b_tile[kWmmaK * kWmmaN];
  __shared__ float c_tile[kWmmaM * kWmmaN];
  __shared__ float c_gate_tile[kWmmaM * kWmmaN];

  const int64_t thread_stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  const int64_t thread_start = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  for (int64_t idx = thread_start; idx < num_local_experts; idx += thread_stride) {
    expert_counts[idx] = 0;
    expert_cursors[idx] = 0;
  }
  for (int64_t idx = thread_start; idx <= num_local_experts; idx += thread_stride) {
    token_offsets[idx] = 0;
  }
  for (int64_t idx = thread_start; idx < num_route_slots; idx += thread_stride) {
    packed_token_topk_indices[idx] = -1;
  }
  for (int64_t idx = thread_start; idx < num_route_slots; idx += thread_stride) {
    route_to_slot[idx] = -1;
  }
  for (int64_t idx = thread_start; idx < num_route_slots * hidden; idx += thread_stride) {
    packed_input[idx] = __float2bfloat16(0.0f);
    packed_expert_out[idx] = __float2bfloat16(0.0f);
  }
  for (int64_t idx = thread_start; idx < num_route_slots * hidden; idx += thread_stride) {
    gathered_out[idx] = __float2bfloat16(0.0f);
  }
  for (int64_t idx = thread_start; idx < num_route_slots * intermediate; idx += thread_stride) {
    h[idx] = __float2bfloat16(0.0f);
  }
  for (int64_t idx = thread_start; idx < num_tokens * hidden; idx += thread_stride) {
    out[idx] = __float2bfloat16(0.0f);
  }

  persistent_grid_barrier(barrier_state, barrier_state + 1);

  for (int64_t route_idx = thread_start; route_idx < num_route_slots; route_idx += thread_stride) {
    const int64_t expert_idx = route_expert_indices[route_idx];
    if (expert_idx >= 0 && expert_idx < num_local_experts) {
      atomic_add_i64(expert_counts + expert_idx, 1);
    }
  }

  persistent_grid_barrier(barrier_state, barrier_state + 1);

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int64_t offset = 0;
    for (int64_t expert_idx = 0; expert_idx < num_local_experts; ++expert_idx) {
      token_offsets[expert_idx] = offset;
      expert_cursors[expert_idx] = offset;
      offset += expert_counts[expert_idx];
    }
    token_offsets[num_local_experts] = offset;
  }

  persistent_grid_barrier(barrier_state, barrier_state + 1);

  for (int64_t route_idx = thread_start; route_idx < num_route_slots; route_idx += thread_stride) {
    const int64_t expert_idx = route_expert_indices[route_idx];
    if (expert_idx >= 0 && expert_idx < num_local_experts) {
      const int64_t packed_idx = atomic_fetch_add_i64(expert_cursors + expert_idx, 1);
      packed_token_topk_indices[packed_idx] = route_idx;
      route_to_slot[route_idx] = packed_idx;
    }
  }

  persistent_grid_barrier(barrier_state, barrier_state + 1);

  for (int64_t value_idx = thread_start;
       value_idx < num_route_slots * hidden;
       value_idx += thread_stride) {
    const int64_t route_idx = value_idx / hidden;
    const int64_t hidden_idx = value_idx - route_idx * hidden;
    const int64_t packed_idx = route_to_slot[route_idx];
    if (packed_idx >= 0) {
      const int64_t token_idx = route_idx / top_k;
      packed_input[packed_idx * hidden + hidden_idx] =
          source_input[token_idx * hidden + hidden_idx];
    }
  }

  persistent_grid_barrier(barrier_state, barrier_state + 1);

  const int64_t l1_n_tiles = (intermediate + kWmmaN - 1) / kWmmaN;
  const int64_t l2_n_tiles = (hidden + kWmmaN - 1) / kWmmaN;
  for (int64_t wave_start = 0;
       wave_start < num_local_experts;
       wave_start += num_experts_per_wave) {
    const int64_t wave_end = wave_start + num_experts_per_wave < num_local_experts
        ? wave_start + num_experts_per_wave
        : num_local_experts;
    const int64_t l1_tasks = wave_task_count(
        expert_counts,
        wave_start,
        wave_end,
        kWmmaM,
        l1_n_tiles);
    for (int64_t task_idx = blockIdx.x; task_idx < l1_tasks; task_idx += gridDim.x) {
      const WaveTile tile =
          decode_wave_tile(task_idx, expert_counts, wave_start, wave_end, kWmmaM, l1_n_tiles);
      if (tile.local_expert >= 0) {
        compute_w1_swiglu_wmma_tile(
            packed_input,
            up_gate_weight,
            expert_counts,
            token_offsets,
            tile.local_expert,
            tile.local_m_tile,
            tile.n_tile,
            num_route_slots,
            hidden,
            intermediate,
            h,
            a_tile,
            b_tile,
            c_tile,
            c_gate_tile);
      }
    }

    persistent_grid_barrier(barrier_state, barrier_state + 1);

    const int64_t l2_tasks = wave_task_count(
        expert_counts,
        wave_start,
        wave_end,
        kWmmaM,
        l2_n_tiles);
    for (int64_t task_idx = blockIdx.x; task_idx < l2_tasks; task_idx += gridDim.x) {
      const WaveTile tile =
          decode_wave_tile(task_idx, expert_counts, wave_start, wave_end, kWmmaM, l2_n_tiles);
      if (tile.local_expert >= 0) {
        compute_w2_wmma_tile(
            h,
            down_weight,
            expert_counts,
            token_offsets,
            tile.local_expert,
            tile.local_m_tile,
            tile.n_tile,
            num_route_slots,
            intermediate,
            hidden,
            packed_expert_out,
            a_tile,
            b_tile,
            c_tile);
      }
    }

    persistent_grid_barrier(barrier_state, barrier_state + 1);
  }

  for (int64_t value_idx = thread_start;
       value_idx < num_route_slots * hidden;
       value_idx += thread_stride) {
    const int64_t route_slot = value_idx / hidden;
    const int64_t hidden_idx = value_idx - route_slot * hidden;
    const int64_t token_topk_idx = packed_token_topk_indices[route_slot];
    if (token_topk_idx >= 0) {
      gathered_out[token_topk_idx * hidden + hidden_idx] =
          packed_expert_out[route_slot * hidden + hidden_idx];
    }
  }

  persistent_grid_barrier(barrier_state, barrier_state + 1);

  for (int64_t value_idx = thread_start;
       value_idx < num_tokens * hidden;
       value_idx += thread_stride) {
    const int64_t token_idx = value_idx / hidden;
    const int64_t hidden_idx = value_idx - token_idx * hidden;
    float acc = 0.0f;
    for (int64_t topk_idx = 0; topk_idx < top_k; ++topk_idx) {
      const int64_t token_topk_idx = token_idx * top_k + topk_idx;
      const float value =
          __bfloat162float(gathered_out[token_topk_idx * hidden + hidden_idx]);
      acc += value * probs[token_topk_idx];
    }
    out[value_idx] = __float2bfloat16(acc);
  }
#else
  (void)source_input;
  (void)route_expert_indices;
  (void)probs;
  (void)up_gate_weight;
  (void)down_weight;
  (void)num_tokens;
  (void)top_k;
  (void)num_local_experts;
  (void)num_experts_per_wave;
  (void)hidden;
  (void)intermediate;
  (void)num_route_slots;
  (void)expert_counts;
  (void)token_offsets;
  (void)expert_cursors;
  (void)packed_token_topk_indices;
  (void)route_to_slot;
  (void)barrier_state;
  (void)packed_input;
  (void)h;
  (void)packed_expert_out;
  (void)gathered_out;
  (void)out;
#endif
}

__global__ void standard_ep_full_forward_megakernel_debug_kernel(
    const __nv_bfloat16* source_input,
    const int64_t* route_expert_indices,
    const float* probs,
    const __nv_bfloat16* up_gate_weight,
    const __nv_bfloat16* down_weight,
    int64_t num_tokens,
    int64_t hidden,
    int64_t intermediate,
    int64_t num_route_slots,
    int64_t packed_capacity,
    const uint64_t* rank_workspace_bases,
    int64_t caller_rank_idx,
    bool enable_cross_rank_barriers,
    bool rank_local_expert_owner,
    int64_t* global_expert_counts,
    int64_t* global_expert_offsets,
    int64_t* expert_cursors,
    int64_t* packed_token_topk_indices,
    int64_t* route_to_slot,
    __nv_bfloat16* packed_input,
    __nv_bfloat16* h,
    __nv_bfloat16* packed_expert_out,
    __nv_bfloat16* gathered_out,
    __nv_bfloat16* out,
    int64_t* recv_counts,
    int64_t* recv_ready_counts,
    int64_t* src_token_topk_indices,
    uint32_t* barrier_state,
    bool use_umma_compute,
    int64_t umma_debug_phase_limit,
    const CUtensorMap* packed_input_map,
    const CUtensorMap* h_map,
    const CUtensorMap* up_gate_weight_map,
    const CUtensorMap* down_weight_map,
    __nv_bfloat16* w1_up,
    __nv_bfloat16* w1_gate) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  constexpr uint32_t kMegakernelDispatchWarps = 1;
  using Builder = dispatch::MetadataBuilder<
      kStandardNumTotalExperts,
      kStandardTopK,
      kStandardSchedulerSms,
      kMegakernelDispatchWarps,
      kStandardNumLocalExperts>;

  __shared__ alignas(1024) StandardEpFullForwardSharedStorage shared;
  __nv_bfloat16* umma_smem_a = shared.umma_smem_a;
  __nv_bfloat16* umma_smem_b = shared.umma_smem_b;
  ptx::MBarrier& umma_tma_barrier = shared.umma_tma_barrier;
  ptx::MBarrier& umma_full_barrier = shared.umma_full_barrier;
  uint32_t& umma_tmem_ptr_smem = shared.umma_tmem_ptr_smem;
  uint32_t* shared_expert_token_count = shared.shared_expert_token_count;
  __nv_bfloat16* a_tile = shared.a_tile;
  __nv_bfloat16* b_tile = shared.b_tile;
  float* c_tile = shared.c_tile;
  float* c_gate_tile = shared.c_gate_tile;

  const int64_t thread_stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  const int64_t thread_start = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t metadata_pairs =
      static_cast<int64_t>(kStandardNumRanks) * kStandardNumLocalExperts;
  const int64_t metadata_src_values = metadata_pairs * num_route_slots;
  const uint32_t source_rank_idx = static_cast<uint32_t>(caller_rank_idx);
  if (source_rank_idx >= kStandardNumRanks) {
    asm volatile("trap;");
    return;
  }
  if (packed_capacity < num_route_slots) {
    asm volatile("trap;");
    return;
  }
  const int64_t num_active_experts = rank_local_expert_owner
      ? static_cast<int64_t>(kStandardNumLocalExperts)
      : static_cast<int64_t>(kStandardNumTotalExperts);
  const uint32_t owner_rank_idx = source_rank_idx;
  const uint32_t thread_idx = threadIdx.x;
  const uint32_t lane_idx = threadIdx.x & 31u;
  const uint32_t num_threads = blockDim.x;
  const uint32_t sm_idx = blockIdx.x;
  void* local_workspace_base = reinterpret_cast<void*>(rank_workspace_bases[source_rank_idx]);
  const layout::Workspace workspace(
      local_workspace_base,
      kStandardNumRanks,
      kStandardNumTotalExperts,
      kStandardNumMaxTokensPerRank,
      kStandardTopK);
  dispatch::PeerWorkspaceAddressMap address_map{
      local_workspace_base,
      rank_workspace_bases,
      source_rank_idx,
  };

  for (int64_t idx = thread_start; idx < kStandardNumTotalExperts; idx += thread_stride) {
    global_expert_counts[idx] = 0;
    expert_cursors[idx] = 0;
  }
  for (int64_t idx = thread_start; idx <= kStandardNumTotalExperts; idx += thread_stride) {
    global_expert_offsets[idx] = 0;
  }
  for (int64_t idx = thread_start; idx < packed_capacity; idx += thread_stride) {
    packed_token_topk_indices[idx] = -1;
  }
  for (int64_t idx = thread_start; idx < num_route_slots; idx += thread_stride) {
    route_to_slot[idx] = -1;
  }
  if (recv_counts != nullptr && recv_ready_counts != nullptr) {
    for (int64_t idx = thread_start; idx < metadata_pairs; idx += thread_stride) {
      recv_counts[idx] = 0;
      recv_ready_counts[idx] = 0;
    }
  }
  if (src_token_topk_indices != nullptr) {
    for (int64_t idx = thread_start; idx < metadata_src_values; idx += thread_stride) {
      src_token_topk_indices[idx] = -1;
    }
  }
  const bool clear_full_runtime_buffers = !use_umma_compute;
  if (clear_full_runtime_buffers) {
    for (int64_t idx = thread_start; idx < packed_capacity * hidden; idx += thread_stride) {
      packed_input[idx] = __float2bfloat16(0.0f);
      packed_expert_out[idx] = __float2bfloat16(0.0f);
    }
    for (int64_t idx = thread_start; idx < num_route_slots * hidden; idx += thread_stride) {
      gathered_out[idx] = __float2bfloat16(0.0f);
    }
    for (int64_t idx = thread_start; idx < packed_capacity * intermediate; idx += thread_stride) {
      h[idx] = __float2bfloat16(0.0f);
    }
    for (int64_t idx = thread_start; idx < num_tokens * hidden; idx += thread_stride) {
      out[idx] = __float2bfloat16(0.0f);
    }
  }

  __nv_bfloat16* source_input_window =
      standard_ep_source_input_window(rank_workspace_bases, source_rank_idx, hidden);
  __nv_bfloat16* local_output_window =
      standard_ep_output_window(rank_workspace_bases, source_rank_idx, hidden);
  for (int64_t idx = thread_start; idx < num_tokens * hidden; idx += thread_stride) {
    source_input_window[idx] = source_input[idx];
  }
  if (clear_full_runtime_buffers) {
    for (int64_t idx = thread_start; idx < num_route_slots * hidden; idx += thread_stride) {
      local_output_window[idx] = __float2bfloat16(0.0f);
    }
  }

  standard_ep_phase_barrier</*kTag=*/1>(
      enable_cross_rank_barriers,
      workspace,
      address_map,
      sm_idx,
      thread_idx,
      barrier_state);

  Builder builder(workspace, shared_expert_token_count);

  builder.clear_shared_counts(thread_idx, num_threads);
  __syncthreads();
  if (thread_idx < 32) {
    builder.count_routes(
        route_expert_indices,
        static_cast<uint32_t>(num_tokens),
        sm_idx,
        0,
        lane_idx);
  }
  __syncthreads();
  builder.publish_sm_offsets(thread_idx, num_threads);

  persistent_grid_barrier(barrier_state, barrier_state + 1);

  if (thread_idx < 32) {
    builder.write_source_indices(
        route_expert_indices,
        static_cast<uint32_t>(num_tokens),
        sm_idx,
        0,
        lane_idx,
        address_map);
  }

  persistent_grid_barrier(barrier_state, barrier_state + 1);

  if (blockIdx.x == 0) {
    builder.publish_recv_counts(thread_idx, num_threads, address_map);
  }

  standard_ep_phase_barrier</*kTag=*/2>(
      enable_cross_rank_barriers,
      workspace,
      address_map,
      sm_idx,
      thread_idx,
      barrier_state);

  for (int64_t logical_expert_idx = thread_start;
       logical_expert_idx < num_active_experts;
       logical_expert_idx += thread_stride) {
    const uint32_t rank_idx = rank_local_expert_owner
        ? owner_rank_idx
        : static_cast<uint32_t>(logical_expert_idx / kStandardNumLocalExperts);
    const uint32_t local_expert_idx = rank_local_expert_owner
        ? static_cast<uint32_t>(logical_expert_idx)
        : static_cast<uint32_t>(
              logical_expert_idx - rank_idx * kStandardNumLocalExperts);
    const void* rank_base = reinterpret_cast<const void*>(rank_workspace_bases[rank_idx]);
    const layout::Workspace rank_workspace(
        const_cast<void*>(rank_base),
        kStandardNumRanks,
        kStandardNumTotalExperts,
        kStandardNumMaxTokensPerRank,
        kStandardTopK);
    const uint64_t value = *rank_workspace.expert_recv_count_sum_ptr(local_expert_idx);
    global_expert_counts[logical_expert_idx] = static_cast<int64_t>(value & 0xffffffffu);
  }

  if (recv_counts != nullptr && recv_ready_counts != nullptr) {
    for (int64_t pair_idx = thread_start; pair_idx < metadata_pairs; pair_idx += thread_stride) {
      const uint32_t rank_idx = static_cast<uint32_t>(pair_idx / kStandardNumLocalExperts);
      const uint32_t local_expert_idx =
          static_cast<uint32_t>(pair_idx - rank_idx * kStandardNumLocalExperts);
      const void* rank_base = reinterpret_cast<const void*>(rank_workspace_bases[rank_idx]);
      const layout::Workspace rank_workspace(
          const_cast<void*>(rank_base),
          kStandardNumRanks,
          kStandardNumTotalExperts,
          kStandardNumMaxTokensPerRank,
          kStandardTopK);
      const uint64_t value = *rank_workspace.expert_recv_count_sum_ptr(local_expert_idx);
      recv_counts[pair_idx] = static_cast<int64_t>(value & 0xffffffffu);
      recv_ready_counts[pair_idx] = static_cast<int64_t>(value >> 32);
    }
  }

  if (src_token_topk_indices != nullptr) {
    for (int64_t src_linear = thread_start;
         src_linear < metadata_src_values;
         src_linear += thread_stride) {
      const int64_t slot_idx = src_linear % num_route_slots;
      const int64_t pair_idx = src_linear / num_route_slots;
      const uint32_t rank_idx = static_cast<uint32_t>(pair_idx / kStandardNumLocalExperts);
      const uint32_t local_expert_idx =
          static_cast<uint32_t>(pair_idx - rank_idx * kStandardNumLocalExperts);
      const void* rank_base = reinterpret_cast<const void*>(rank_workspace_bases[rank_idx]);
      const layout::Workspace rank_workspace(
          const_cast<void*>(rank_base),
          kStandardNumRanks,
          kStandardNumTotalExperts,
          kStandardNumMaxTokensPerRank,
          kStandardTopK);
      const int64_t count =
          static_cast<int64_t>(*rank_workspace.expert_recv_count_sum_ptr(local_expert_idx) & 0xffffffffu);
      src_token_topk_indices[src_linear] = slot_idx < count
          ? static_cast<int64_t>(*rank_workspace.src_token_topk_idx_ptr(
                local_expert_idx,
                source_rank_idx,
                static_cast<uint32_t>(slot_idx)))
          : -1;
    }
  }

  standard_ep_phase_barrier</*kTag=*/3>(
      enable_cross_rank_barriers,
      workspace,
      address_map,
      sm_idx,
      thread_idx,
      barrier_state);

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int64_t offset = 0;
    for (int64_t expert_idx = 0; expert_idx < num_active_experts; ++expert_idx) {
      global_expert_offsets[expert_idx] = offset;
      expert_cursors[expert_idx] = offset;
      offset += global_expert_counts[expert_idx];
    }
    for (int64_t expert_idx = num_active_experts;
         expert_idx <= kStandardNumTotalExperts;
         ++expert_idx) {
      global_expert_offsets[expert_idx] = offset;
      if (expert_idx < kStandardNumTotalExperts) {
        expert_cursors[expert_idx] = offset;
      }
    }
  }

  persistent_grid_barrier(barrier_state, barrier_state + 1);

  const int64_t total_routes = global_expert_offsets[num_active_experts];
  if (total_routes > packed_capacity) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      printf(
          "OLMo BF16 MegaMoE packed capacity exceeded: routes=%lld capacity=%lld rank=%lld\n",
          static_cast<long long>(total_routes),
          static_cast<long long>(packed_capacity),
          static_cast<long long>(caller_rank_idx));
    }
    asm volatile("trap;");
    return;
  }
  for (int64_t linear_idx = thread_start; linear_idx < total_routes; linear_idx += thread_stride) {
    int64_t expert_idx = 0;
    while (expert_idx + 1 <= num_active_experts &&
           linear_idx >= global_expert_offsets[expert_idx + 1]) {
      ++expert_idx;
    }
    const int64_t slot = linear_idx - global_expert_offsets[expert_idx];
    const uint32_t rank_idx = rank_local_expert_owner
        ? owner_rank_idx
        : static_cast<uint32_t>(expert_idx / kStandardNumLocalExperts);
    const uint32_t local_expert_idx = rank_local_expert_owner
        ? static_cast<uint32_t>(expert_idx)
        : static_cast<uint32_t>(expert_idx - rank_idx * kStandardNumLocalExperts);
    const StandardEpPackedSourceRoute source_route = slot < global_expert_counts[expert_idx]
        ? standard_ep_load_source_route(
              rank_workspace_bases,
              rank_idx,
              local_expert_idx,
              slot)
        : StandardEpPackedSourceRoute{};
    const int64_t token_topk_idx = source_route.token_topk_idx;
    const int64_t encoded_source_route =
        token_topk_idx >= 0 && token_topk_idx < num_route_slots
        ? standard_ep_encode_source_route(
              source_route.source_rank_idx,
              token_topk_idx,
              num_route_slots)
        : -1;
    packed_token_topk_indices[linear_idx] = encoded_source_route;
    if (source_route.source_rank_idx == source_rank_idx &&
        token_topk_idx >= 0 &&
        token_topk_idx < num_route_slots) {
      route_to_slot[token_topk_idx] = linear_idx;
    }
  }

  persistent_grid_barrier(barrier_state, barrier_state + 1);

  for (int64_t value_idx = thread_start;
       value_idx < total_routes * hidden;
       value_idx += thread_stride) {
    const int64_t packed_idx = value_idx / hidden;
    const int64_t hidden_idx = value_idx - packed_idx * hidden;
    const StandardEpPackedSourceRoute source_route =
        standard_ep_decode_source_route(packed_token_topk_indices[packed_idx], num_route_slots);
    if (source_route.token_topk_idx >= 0 &&
        source_route.token_topk_idx < num_route_slots &&
        source_route.source_rank_idx < kStandardNumRanks) {
      const int64_t token_idx = source_route.token_topk_idx / kStandardTopK;
      const __nv_bfloat16* source_rank_input =
          standard_ep_source_input_window(rank_workspace_bases, source_route.source_rank_idx, hidden);
      packed_input[value_idx] = source_rank_input[token_idx * hidden + hidden_idx];
    }
  }

  persistent_grid_barrier(barrier_state, barrier_state + 1);

  const int64_t l1_n_tiles = use_umma_compute
      ? (intermediate + static_cast<int64_t>(kSm100TileContractN) - 1) /
          static_cast<int64_t>(kSm100TileContractN)
      : (intermediate + kWmmaN - 1) / kWmmaN;
  const int64_t l2_n_tiles = use_umma_compute
      ? (hidden + static_cast<int64_t>(kSm100TileContractN) - 1) /
          static_cast<int64_t>(kSm100TileContractN)
      : (hidden + kWmmaN - 1) / kWmmaN;
  const int64_t block_m = use_umma_compute
      ? static_cast<int64_t>(kSm100TileContractM)
      : static_cast<int64_t>(kWmmaM);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && OLMO_BF16_MEGA_HAS_TCGEN05_TMEM
  const bool owns_umma_tmem = use_umma_compute && umma_debug_phase_limit >= 1;
  if (owns_umma_tmem) {
    if (threadIdx.x == 0) {
      umma_tmem_ptr_smem = 0;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
      ptx::tcgen05_after_thread_sync();
      ptx::tcgen05_tmem_alloc<1>(
          &umma_tmem_ptr_smem,
          /*num_cols=*/kSm100TileContractN);
    }
    __syncthreads();
  }
#endif
  for (int64_t wave_start = 0;
       wave_start < num_active_experts;
       wave_start += kStandardExpertsPerWave) {
    const int64_t wave_end = wave_start + kStandardExpertsPerWave < num_active_experts
        ? wave_start + kStandardExpertsPerWave
        : num_active_experts;
    const int64_t l1_tasks = wave_task_count(
        global_expert_counts,
        wave_start,
        wave_end,
        block_m,
        l1_n_tiles);
    for (int64_t task_idx = blockIdx.x; task_idx < l1_tasks; task_idx += gridDim.x) {
      const WaveTile tile = decode_wave_tile(
          task_idx,
          global_expert_counts,
          wave_start,
          wave_end,
          block_m,
          l1_n_tiles);
      if (tile.local_expert >= 0) {
        if (use_umma_compute) {
          if (umma_debug_phase_limit >= 1) {
            compute_bf16_umma_linear_tile_to_bf16<
                /*kBMnMajor=*/false,
                /*kManageTmem=*/false>(
                packed_input_map,
                up_gate_weight_map,
                global_expert_counts,
                global_expert_offsets,
                tile.local_expert,
                tile.local_m_tile,
                tile.n_tile,
                hidden,
                intermediate,
                tile.local_expert * (2 * intermediate),
                w1_up,
                umma_smem_a,
                umma_smem_b,
                &umma_tma_barrier,
                &umma_full_barrier,
                &umma_tmem_ptr_smem);
          }
          if (umma_debug_phase_limit >= 2) {
            compute_bf16_umma_linear_tile_to_bf16<
                /*kBMnMajor=*/false,
                /*kManageTmem=*/false>(
                packed_input_map,
                up_gate_weight_map,
                global_expert_counts,
                global_expert_offsets,
                tile.local_expert,
                tile.local_m_tile,
                tile.n_tile,
                hidden,
                intermediate,
                tile.local_expert * (2 * intermediate) + intermediate,
                w1_gate,
                umma_smem_a,
                umma_smem_b,
                &umma_tma_barrier,
                &umma_full_barrier,
                &umma_tmem_ptr_smem);
            compute_swiglu_split_tile(
                w1_up,
                w1_gate,
                global_expert_counts,
                global_expert_offsets,
                tile.local_expert,
                tile.local_m_tile,
                tile.n_tile,
                intermediate,
                h);
          }
        } else {
          compute_w1_swiglu_wmma_tile(
              packed_input,
              up_gate_weight,
              global_expert_counts,
              global_expert_offsets,
              tile.local_expert,
              tile.local_m_tile,
              tile.n_tile,
              packed_capacity,
              hidden,
              intermediate,
              h,
              a_tile,
              b_tile,
              c_tile,
              c_gate_tile);
        }
      }
    }

    persistent_grid_barrier(barrier_state, barrier_state + 1);

    const int64_t l2_tasks = wave_task_count(
        global_expert_counts,
        wave_start,
        wave_end,
        block_m,
        l2_n_tiles);
    for (int64_t task_idx = blockIdx.x; task_idx < l2_tasks; task_idx += gridDim.x) {
      const WaveTile tile = decode_wave_tile(
          task_idx,
          global_expert_counts,
          wave_start,
          wave_end,
          block_m,
          l2_n_tiles);
      if (tile.local_expert >= 0) {
        if (use_umma_compute && umma_debug_phase_limit >= 3) {
          compute_bf16_umma_linear_tile_to_bf16<
              /*kBMnMajor=*/true,
              /*kManageTmem=*/false>(
              h_map,
              down_weight_map,
              global_expert_counts,
              global_expert_offsets,
              tile.local_expert,
              tile.local_m_tile,
              tile.n_tile,
              intermediate,
              hidden,
              tile.local_expert * intermediate,
              packed_expert_out,
              umma_smem_a,
              umma_smem_b,
              &umma_tma_barrier,
              &umma_full_barrier,
              &umma_tmem_ptr_smem);
        } else {
          compute_w2_wmma_tile(
              h,
              down_weight,
              global_expert_counts,
              global_expert_offsets,
              tile.local_expert,
              tile.local_m_tile,
              tile.n_tile,
              packed_capacity,
              intermediate,
              hidden,
              packed_expert_out,
              a_tile,
              b_tile,
              c_tile);
        }
      }
    }

    persistent_grid_barrier(barrier_state, barrier_state + 1);
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && OLMO_BF16_MEGA_HAS_TCGEN05_TMEM
  if (owns_umma_tmem) {
    __syncthreads();
    if (threadIdx.x < 32) {
      ptx::tcgen05_tmem_dealloc<1>(
          umma_tmem_ptr_smem,
          /*num_cols=*/kSm100TileContractN);
      ptx::tcgen05_tmem_relinquish_alloc_permit<1>();
    }
    __syncthreads();
  }
#endif

  for (int64_t value_idx = thread_start;
       value_idx < total_routes * hidden;
       value_idx += thread_stride) {
    const int64_t route_slot = value_idx / hidden;
    const int64_t hidden_idx = value_idx - route_slot * hidden;
    const StandardEpPackedSourceRoute source_route =
        standard_ep_decode_source_route(packed_token_topk_indices[route_slot], num_route_slots);
    if (source_route.source_rank_idx == source_rank_idx &&
        source_route.token_topk_idx >= 0 &&
        source_route.token_topk_idx < num_route_slots) {
      local_output_window[source_route.token_topk_idx * hidden + hidden_idx] =
          packed_expert_out[route_slot * hidden + hidden_idx];
    } else if (source_route.source_rank_idx < kStandardNumRanks &&
               source_route.token_topk_idx >= 0 &&
               source_route.token_topk_idx < num_route_slots) {
      __nv_bfloat16* remote_output_window =
          standard_ep_output_window(rank_workspace_bases, source_route.source_rank_idx, hidden);
      remote_output_window[source_route.token_topk_idx * hidden + hidden_idx] =
          packed_expert_out[route_slot * hidden + hidden_idx];
    }
  }

  standard_ep_phase_barrier</*kTag=*/4>(
      enable_cross_rank_barriers,
      workspace,
      address_map,
      sm_idx,
      thread_idx,
      barrier_state);

  for (int64_t value_idx = thread_start;
       value_idx < num_route_slots * hidden;
       value_idx += thread_stride) {
    gathered_out[value_idx] = local_output_window[value_idx];
  }

  persistent_grid_barrier(barrier_state, barrier_state + 1);

  for (int64_t value_idx = thread_start; value_idx < num_tokens * hidden; value_idx += thread_stride) {
    const int64_t token_idx = value_idx / hidden;
    const int64_t hidden_idx = value_idx - token_idx * hidden;
    float acc = 0.0f;
    for (int64_t topk_idx = 0; topk_idx < kStandardTopK; ++topk_idx) {
      const int64_t token_topk_idx = token_idx * kStandardTopK + topk_idx;
      const float value =
          __bfloat162float(gathered_out[token_topk_idx * hidden + hidden_idx]);
      acc += value * probs[token_topk_idx];
    }
    out[value_idx] = __float2bfloat16(acc);
  }
#else
  (void)source_input;
  (void)route_expert_indices;
  (void)probs;
  (void)up_gate_weight;
  (void)down_weight;
  (void)num_tokens;
  (void)hidden;
  (void)intermediate;
  (void)num_route_slots;
  (void)packed_capacity;
  (void)rank_workspace_bases;
  (void)caller_rank_idx;
  (void)enable_cross_rank_barriers;
  (void)rank_local_expert_owner;
  (void)global_expert_counts;
  (void)global_expert_offsets;
  (void)expert_cursors;
  (void)packed_token_topk_indices;
  (void)route_to_slot;
  (void)packed_input;
  (void)h;
  (void)packed_expert_out;
  (void)gathered_out;
  (void)out;
  (void)recv_counts;
  (void)recv_ready_counts;
  (void)src_token_topk_indices;
  (void)barrier_state;
  (void)use_umma_compute;
  (void)umma_debug_phase_limit;
  (void)packed_input_map;
  (void)h_map;
  (void)up_gate_weight_map;
  (void)down_weight_map;
  (void)w1_up;
  (void)w1_gate;
#endif
}

}  // namespace olmo::bf16_mega_moe::kernels

/*
 * OLMo BF16 MegaMoE dispatch metadata.
 *
 * This is the OLMo-owned counterpart to the MegaMoE dispatch phase: dispatch
 * warps read the router's top-k expert ids, count tokens per expert, publish
 * per-SM offsets, and write token/top-k source records into the symmetric
 * workspace consumed by the persistent wave scheduler.
 */
#pragma once

#include "layout.cuh"

#include <cstdint>

namespace olmo::bf16_mega_moe::dispatch {

template <typename IndexT>
OLMO_BF16_MEGA_DEVICE int32_t route_index_to_i32(IndexT value) {
  return static_cast<int32_t>(value);
}

template <
    uint32_t kNumExperts,
    uint32_t kNumTopK,
    uint32_t kNumSMs,
    uint32_t kNumDispatchWarps>
struct TopKRouteReaderForGrid {
  static_assert(kNumTopK > 0, "top_k must be positive");
  static_assert(kNumTopK <= 32, "one dispatch warp supports top_k <= 32");
  static_assert(32 % kNumTopK == 0, "top_k must divide a warp for lane-token mapping");
  static_assert(kNumDispatchWarps > 0, "dispatch warp count must be positive");
  static_assert(kNumSMs > 0, "SM count must be positive");

  static constexpr uint32_t kNumTokensPerWarp = 32 / kNumTopK;
  static constexpr uint32_t kNumActiveLanes = kNumTokensPerWarp * kNumTopK;

  template <typename IndexT, typename Process>
  OLMO_BF16_MEGA_DEVICE static void for_each(
      const IndexT* route_expert_indices,
      uint32_t num_tokens,
      uint32_t sm_idx,
      uint32_t warp_idx,
      uint32_t lane_idx,
      Process process) {
    const uint32_t token_start =
        (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
    const uint32_t token_stride = kNumSMs * kNumDispatchWarps * kNumTokensPerWarp;
    for (uint32_t token_base = token_start; token_base < num_tokens;
         token_base += token_stride) {
      if (lane_idx < kNumActiveLanes) {
        const uint32_t token_idx = token_base + lane_idx / kNumTopK;
        const uint32_t topk_idx = lane_idx % kNumTopK;
        if (token_idx < num_tokens) {
          const uint32_t token_topk_idx = token_idx * kNumTopK + topk_idx;
          const int32_t expert_idx =
              route_index_to_i32(__ldg(route_expert_indices + token_topk_idx));
          if (expert_idx >= 0 && static_cast<uint32_t>(expert_idx) < kNumExperts) {
            process(token_idx, topk_idx, token_topk_idx, static_cast<uint32_t>(expert_idx));
          }
        }
      }
      __syncwarp();
    }
  }
};

struct LocalAddressMap {
  uint32_t rank_idx = 0;

  template <typename T>
  OLMO_BF16_MEGA_DEVICE T* map(T* ptr, uint32_t dst_rank_idx) const {
    (void)dst_rank_idx;
    return ptr;
  }
};

struct DebugMultiRankAddressMap {
  void* rank0_workspace_base = nullptr;
  uint64_t workspace_stride_bytes = 0;
  uint32_t rank_idx = 0;

  template <typename T>
  OLMO_BF16_MEGA_DEVICE T* map(T* ptr, uint32_t dst_rank_idx) const {
    const uint64_t offset =
        static_cast<uint64_t>(
            reinterpret_cast<uintptr_t>(ptr) -
            reinterpret_cast<uintptr_t>(rank0_workspace_base));
    return reinterpret_cast<T*>(
        reinterpret_cast<uintptr_t>(rank0_workspace_base) +
        static_cast<uint64_t>(dst_rank_idx) * workspace_stride_bytes +
        offset);
  }
};

struct PeerWorkspaceAddressMap {
  void* local_workspace_base = nullptr;
  const uint64_t* rank_workspace_bases = nullptr;
  uint32_t rank_idx = 0;

  template <typename T>
  OLMO_BF16_MEGA_DEVICE T* map(T* ptr, uint32_t dst_rank_idx) const {
    const uint64_t offset =
        static_cast<uint64_t>(
            reinterpret_cast<uintptr_t>(ptr) -
            reinterpret_cast<uintptr_t>(local_workspace_base));
    return reinterpret_cast<T*>(rank_workspace_bases[dst_rank_idx] + offset);
  }
};

OLMO_BF16_MEGA_DEVICE uint64_t atomic_add_u64(uint64_t* ptr, uint64_t value) {
  return static_cast<uint64_t>(
      atomicAdd(reinterpret_cast<unsigned long long*>(ptr),
                static_cast<unsigned long long>(value)));
}

template <
    uint32_t kNumExperts,
    uint32_t kNumTopK,
    uint32_t kNumSMs,
    uint32_t kNumDispatchWarps,
    uint32_t kNumExpertsPerRank = kNumExperts>
struct MetadataBuilder {
  using Reader = TopKRouteReaderForGrid<kNumExperts, kNumTopK, kNumSMs, kNumDispatchWarps>;

  layout::Workspace workspace;
  uint32_t* shared_expert_token_count;

  OLMO_BF16_MEGA_DEVICE MetadataBuilder(
      layout::Workspace workspace_,
      uint32_t* shared_expert_token_count_)
      : workspace(workspace_),
        shared_expert_token_count(shared_expert_token_count_) {}

  OLMO_BF16_MEGA_DEVICE void clear_shared_counts(uint32_t thread_idx, uint32_t num_threads) {
    for (uint32_t expert_idx = thread_idx; expert_idx < kNumExperts; expert_idx += num_threads) {
      shared_expert_token_count[expert_idx] = 0;
    }
  }

  template <typename IndexT>
  OLMO_BF16_MEGA_DEVICE void count_routes(
      const IndexT* route_expert_indices,
      uint32_t num_tokens,
      uint32_t sm_idx,
      uint32_t warp_idx,
      uint32_t lane_idx) {
    Reader::for_each(
        route_expert_indices,
        num_tokens,
        sm_idx,
        warp_idx,
        lane_idx,
        [=](uint32_t, uint32_t, uint32_t, uint32_t expert_idx) {
          atomicAdd(shared_expert_token_count + expert_idx, 1u);
        });
  }

  OLMO_BF16_MEGA_DEVICE void publish_sm_offsets(uint32_t thread_idx, uint32_t num_threads) {
    for (uint32_t expert_idx = thread_idx; expert_idx < kNumExperts; expert_idx += num_threads) {
      const uint64_t send_value =
          (uint64_t{1} << 32) | static_cast<uint64_t>(shared_expert_token_count[expert_idx]);
      shared_expert_token_count[expert_idx] =
          static_cast<uint32_t>(atomic_add_u64(workspace.expert_send_count_ptr(expert_idx), send_value));
    }
  }

  template <typename IndexT, typename AddressMap>
  OLMO_BF16_MEGA_DEVICE void write_source_indices(
      const IndexT* route_expert_indices,
      uint32_t num_tokens,
      uint32_t sm_idx,
      uint32_t warp_idx,
      uint32_t lane_idx,
      AddressMap address_map) {
    Reader::for_each(
        route_expert_indices,
        num_tokens,
        sm_idx,
        warp_idx,
        lane_idx,
        [=](uint32_t, uint32_t, uint32_t token_topk_idx, uint32_t expert_idx) {
          const uint32_t dst_rank_idx = expert_idx / kNumExpertsPerRank;
          const uint32_t dst_local_expert_idx = expert_idx % kNumExpertsPerRank;
          const uint32_t dst_slot_idx =
              atomicAdd(shared_expert_token_count + expert_idx, 1u);
          uint32_t* dst_ptr = workspace.src_token_topk_idx_ptr(
              dst_local_expert_idx,
              address_map.rank_idx,
              dst_slot_idx);
          *address_map.map(dst_ptr, dst_rank_idx) = token_topk_idx;
        });
  }

  template <typename AddressMap>
  OLMO_BF16_MEGA_DEVICE void publish_recv_counts(
      uint32_t thread_idx,
      uint32_t num_threads,
      AddressMap address_map) {
    for (uint32_t expert_idx = thread_idx; expert_idx < kNumExperts; expert_idx += num_threads) {
      const uint32_t dst_rank_idx = expert_idx / kNumExpertsPerRank;
      const uint32_t dst_local_expert_idx = expert_idx % kNumExpertsPerRank;
      const uint64_t expert_status = *workspace.expert_send_count_ptr(expert_idx);
      *address_map.map(
          workspace.expert_recv_count_ptr(address_map.rank_idx, dst_local_expert_idx),
          dst_rank_idx) = expert_status & 0xffffffffu;
      atomic_add_u64(
          address_map.map(workspace.expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx),
          expert_status);
    }
  }
};

}  // namespace olmo::bf16_mega_moe::dispatch

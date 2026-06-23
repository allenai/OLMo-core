/*
 * OLMo BF16 MegaMoE layout helpers.
 *
 * This is an OLMo-owned adaptation of the DeepGEMM MegaMoE runtime layout:
 * a symmetric workspace followed by input, intermediate, and combine buffers.
 * It intentionally avoids depending on DeepGEMM headers or kernels.
 */
#pragma once

#include <cstdint>

namespace olmo::bf16_mega_moe {

#if defined(__CUDACC__)
#define OLMO_BF16_MEGA_HOST_DEVICE __host__ __device__
#define OLMO_BF16_MEGA_DEVICE __device__
#else
#define OLMO_BF16_MEGA_HOST_DEVICE
#define OLMO_BF16_MEGA_DEVICE
#endif

namespace layout {

constexpr uint32_t kNumCandidateBlockMs = 7;
constexpr uint32_t kCandidateBlockM[kNumCandidateBlockMs] = {8, 16, 32, 64, 96, 128, 192};
constexpr uint32_t kMaxCandidateBlockM = 192;
constexpr uint32_t kMinCandidateBlockM = 8;
constexpr uint32_t kLcmCandidateBlockM = 384;
constexpr uint64_t kNumBarrierSignalBytes = 32;
constexpr uint32_t kNumMaxGridSyncCounters = 4;

enum class BlockPhase : uint32_t {
  None = 0,
  Linear1 = 1,
  Linear2 = 2,
};

namespace math {

template <typename T>
OLMO_BF16_MEGA_HOST_DEVICE constexpr T ceil_div(T x, T y) {
  return (x + y - 1) / y;
}

template <typename T>
OLMO_BF16_MEGA_HOST_DEVICE constexpr T min(T x, T y) {
  return x < y ? x : y;
}

template <typename T>
OLMO_BF16_MEGA_HOST_DEVICE constexpr T align(T x, T alignment) {
  return ceil_div(x, alignment) * alignment;
}

template <typename T = void>
OLMO_BF16_MEGA_HOST_DEVICE T* advance_ptr(void* ptr, uint64_t bytes) {
  return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(ptr) + bytes);
}

template <typename T = void>
OLMO_BF16_MEGA_HOST_DEVICE const T* advance_ptr(const void* ptr, uint64_t bytes) {
  return reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(ptr) + bytes);
}

}  // namespace math

template <typename T>
OLMO_BF16_MEGA_HOST_DEVICE constexpr T num_max_pool_tokens(
    T num_ranks,
    T num_max_tokens_per_rank,
    T top_k,
    T num_experts_per_rank) {
  const T max_recv_tokens = num_ranks * num_max_tokens_per_rank;
  const T max_experts_per_token = math::min(top_k, num_experts_per_rank);
  return math::align(
      max_recv_tokens * max_experts_per_token +
          num_experts_per_rank * (static_cast<T>(kMaxCandidateBlockM) - 1),
      static_cast<T>(kLcmCandidateBlockM));
}

struct TokenSrcMetadata {
  uint32_t rank_idx;
  uint32_t token_idx;
  uint32_t topk_idx;
};

struct Workspace {
  void* base;
  uint32_t num_ranks;
  uint32_t num_experts;
  uint32_t num_experts_per_rank;
  uint32_t num_max_tokens_per_rank;
  uint32_t num_max_recv_tokens_per_expert;
  uint32_t num_max_pool_tokens;
  uint32_t num_max_pool_blocks;

  OLMO_BF16_MEGA_HOST_DEVICE Workspace(
      void* base_,
      uint32_t num_ranks_,
      uint32_t num_experts_,
      uint32_t num_max_tokens_per_rank_,
      uint32_t top_k)
      : base(base_),
        num_ranks(num_ranks_),
        num_experts(num_experts_),
        num_experts_per_rank(num_experts_ / num_ranks_),
        num_max_tokens_per_rank(num_max_tokens_per_rank_),
        num_max_recv_tokens_per_expert(num_ranks_ * num_max_tokens_per_rank_),
        num_max_pool_tokens(layout::num_max_pool_tokens(
            num_ranks_,
            num_max_tokens_per_rank_,
            top_k,
            num_experts_ / num_ranks_)),
        num_max_pool_blocks(num_max_pool_tokens / kMinCandidateBlockM) {}

  OLMO_BF16_MEGA_HOST_DEVICE uint64_t num_bytes() const {
    uint64_t bytes = 0;
    bytes += kNumBarrierSignalBytes;
    bytes += static_cast<uint64_t>(num_experts) * sizeof(uint64_t) * 2;
    bytes += static_cast<uint64_t>(num_experts_per_rank) * sizeof(uint64_t);
    bytes += math::align(num_max_pool_blocks, 2u) * sizeof(uint32_t);
    bytes += static_cast<uint64_t>(num_max_pool_blocks) * sizeof(uint64_t);
    bytes += static_cast<uint64_t>(num_experts_per_rank) * num_ranks *
        num_max_recv_tokens_per_expert * sizeof(uint32_t);
    bytes += static_cast<uint64_t>(num_max_pool_tokens) * sizeof(TokenSrcMetadata);
    return math::align<uint64_t>(bytes, 16);
  }

  OLMO_BF16_MEGA_HOST_DEVICE void* end_ptr() const {
    return math::advance_ptr(base, num_bytes());
  }

  OLMO_BF16_MEGA_HOST_DEVICE uint32_t* grid_sync_count_ptr(uint32_t idx = 0) const {
    return static_cast<uint32_t*>(base) + idx;
  }

  OLMO_BF16_MEGA_HOST_DEVICE uint32_t* nvl_barrier_counter_ptr() const {
    return static_cast<uint32_t*>(base) + kNumMaxGridSyncCounters;
  }

  OLMO_BF16_MEGA_HOST_DEVICE int* nvl_barrier_signal_ptr(uint32_t phase) const {
    return math::advance_ptr<int>(
        base,
        (kNumMaxGridSyncCounters + 1) * sizeof(uint32_t) + phase * sizeof(int));
  }

  OLMO_BF16_MEGA_HOST_DEVICE uint64_t* expert_send_count_ptr(uint32_t expert_idx = 0) const {
    return math::advance_ptr<uint64_t>(base, kNumBarrierSignalBytes) + expert_idx;
  }

  OLMO_BF16_MEGA_HOST_DEVICE uint64_t* expert_recv_count_ptr(
      uint32_t rank_idx = 0,
      uint32_t expert_idx = 0) const {
    return expert_send_count_ptr(num_experts) +
        rank_idx * num_experts_per_rank + expert_idx;
  }

  OLMO_BF16_MEGA_HOST_DEVICE uint64_t* expert_recv_count_sum_ptr(uint32_t expert_idx = 0) const {
    return expert_send_count_ptr(num_experts * 2) + expert_idx;
  }

  OLMO_BF16_MEGA_HOST_DEVICE uint32_t* l1_arrival_count_ptr(uint32_t pool_block_idx = 0) const {
    return reinterpret_cast<uint32_t*>(expert_recv_count_sum_ptr(num_experts_per_rank)) +
        pool_block_idx;
  }

  OLMO_BF16_MEGA_HOST_DEVICE uint64_t* l2_arrival_mask_ptr(uint32_t pool_block_idx = 0) const {
    return reinterpret_cast<uint64_t*>(
               l1_arrival_count_ptr(math::align(num_max_pool_blocks, 2u))) +
        pool_block_idx;
  }

  OLMO_BF16_MEGA_HOST_DEVICE uint32_t* src_token_topk_idx_ptr(
      uint32_t expert_idx = 0,
      uint32_t rank_idx = 0,
      uint32_t token_idx = 0) const {
    return reinterpret_cast<uint32_t*>(l2_arrival_mask_ptr(num_max_pool_blocks)) +
        expert_idx * (num_ranks * num_max_recv_tokens_per_expert) +
        rank_idx * num_max_recv_tokens_per_expert + token_idx;
  }

  OLMO_BF16_MEGA_HOST_DEVICE TokenSrcMetadata* token_src_metadata_ptr(
      uint32_t pool_token_idx = 0) const {
    return reinterpret_cast<TokenSrcMetadata*>(src_token_topk_idx_ptr(num_experts_per_rank)) +
        pool_token_idx;
  }
};

struct Data {
  uint32_t num_bytes;
  bool require_tma_alignment;
  void* base;

  OLMO_BF16_MEGA_HOST_DEVICE constexpr Data(
      uint32_t num_bytes_,
      bool require_tma_alignment_ = true,
      void* base_ = nullptr)
      : num_bytes(num_bytes_),
        require_tma_alignment(require_tma_alignment_),
        base(base_) {}

  template <typename T = void>
  OLMO_BF16_MEGA_HOST_DEVICE T* base_ptr() const {
    return static_cast<T*>(base);
  }

  OLMO_BF16_MEGA_HOST_DEVICE void set_base_ptr(void* ptr) {
    base = ptr;
  }
};

struct Buffer {
  Data data_layout;
  uint32_t num_ranks;
  uint32_t num_max_tokens_per_rank;
  void* base;

  OLMO_BF16_MEGA_HOST_DEVICE Buffer(
      Data data_layout_,
      uint32_t num_ranks_,
      uint32_t num_max_tokens_per_rank_,
      void* base_ = nullptr)
      : data_layout(data_layout_),
        num_ranks(num_ranks_),
        num_max_tokens_per_rank(num_max_tokens_per_rank_),
        base(base_) {}

  OLMO_BF16_MEGA_HOST_DEVICE uint64_t num_bytes_per_rank() const {
    return static_cast<uint64_t>(num_max_tokens_per_rank) * data_layout.num_bytes;
  }

  OLMO_BF16_MEGA_HOST_DEVICE uint64_t num_bytes() const {
    return num_bytes_per_rank() * num_ranks;
  }

  template <typename T = void>
  OLMO_BF16_MEGA_HOST_DEVICE T* base_ptr() const {
    return static_cast<T*>(base);
  }

  OLMO_BF16_MEGA_HOST_DEVICE void* end_ptr() const {
    return math::advance_ptr(base, num_bytes());
  }

  OLMO_BF16_MEGA_HOST_DEVICE Buffer rank_buffer(uint32_t rank_idx) const {
    return {data_layout, 1, num_max_tokens_per_rank, math::advance_ptr(base, num_bytes_per_rank() * rank_idx)};
  }

  OLMO_BF16_MEGA_HOST_DEVICE Data data_buffer(uint32_t token_idx, bool global = false) const {
    (void)global;
    return Data(
        data_layout.num_bytes,
        data_layout.require_tma_alignment,
        math::advance_ptr(base, static_cast<uint64_t>(data_layout.num_bytes) * token_idx));
  }
};

}  // namespace layout

}  // namespace olmo::bf16_mega_moe

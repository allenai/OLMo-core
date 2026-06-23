#pragma once

#include <stddef.h>
#include <stdint.h>

#include "metadata.cuh"

namespace olmo::tma_ibgda_ep {

constexpr size_t kWorkspaceAlignment = 128;
constexpr size_t kDoorbellBytes = 8;
constexpr size_t kCompletionBytes = 8;

struct Doorbell {
  uint64_t value;
};

struct CompletionCounter {
  uint64_t value;
};

static_assert(sizeof(Doorbell) == kDoorbellBytes, "Doorbell size changed");
static_assert(sizeof(CompletionCounter) == kCompletionBytes, "CompletionCounter size changed");

struct WorkspaceLayout {
  RouteRecord* route_records;
  int64_t* routes_per_rank;
  int64_t* rank_offsets;
  uint8_t* overflow_by_rank;
  void* payload_window;
  size_t route_records_bytes;
  size_t routes_per_rank_bytes;
  size_t rank_offsets_bytes;
  size_t overflow_by_rank_bytes;
  size_t payload_window_bytes;
  size_t total_bytes;
};

struct PeerWindowLayout {
  size_t route_records_offset;
  size_t routes_per_rank_offset;
  size_t rank_offsets_offset;
  size_t overflow_by_rank_offset;
  size_t payload_window_offset;
  size_t send_doorbells_offset;
  size_t recv_completions_offset;
  size_t rank_stride_bytes;
  size_t route_records_bytes;
  size_t routes_per_rank_bytes;
  size_t rank_offsets_bytes;
  size_t overflow_by_rank_bytes;
  size_t payload_window_bytes_per_rank;
  size_t send_doorbells_bytes;
  size_t recv_completions_bytes;
  int32_t ep_world_size;
  int32_t rank_capacity;
  int32_t hidden_size;
  int32_t dtype_bytes;
};

struct PeerWindowView {
  void* local_base;
  const uint64_t* peer_base_addrs;
  PeerWindowLayout layout;

  template <typename T = void>
  __host__ __device__ T* local_ptr(size_t offset) const {
    return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(local_base) + offset);
  }

  template <typename T = void>
  __host__ __device__ T* peer_ptr(int32_t peer_rank, size_t offset) const {
    return reinterpret_cast<T*>(peer_base_addrs[peer_rank] + offset);
  }

  __host__ __device__ RouteRecord* local_route_records() const {
    return local_ptr<RouteRecord>(layout.route_records_offset);
  }

  __host__ __device__ int64_t* local_routes_per_rank() const {
    return local_ptr<int64_t>(layout.routes_per_rank_offset);
  }

  __host__ __device__ int64_t* local_rank_offsets() const {
    return local_ptr<int64_t>(layout.rank_offsets_offset);
  }

  __host__ __device__ uint8_t* local_overflow_by_rank() const {
    return local_ptr<uint8_t>(layout.overflow_by_rank_offset);
  }

  __host__ __device__ void* local_payload_window() const {
    return local_ptr(layout.payload_window_offset);
  }

  __host__ __device__ Doorbell* peer_send_doorbells(int32_t peer_rank) const {
    return peer_ptr<Doorbell>(peer_rank, layout.send_doorbells_offset);
  }

  __host__ __device__ CompletionCounter* peer_recv_completions(int32_t peer_rank) const {
    return peer_ptr<CompletionCounter>(peer_rank, layout.recv_completions_offset);
  }

  __host__ __device__ void* peer_payload_window(int32_t peer_rank) const {
    return peer_ptr(peer_rank, layout.payload_window_offset);
  }
};

struct KernelLaunchConfig {
  int32_t num_sms_dispatch;
  int32_t num_sms_combine;
  int32_t num_sms_preprocess;
  int32_t ep_world_size;
  int32_t rank_capacity;
  int32_t hidden_size;
};

}  // namespace olmo::tma_ibgda_ep

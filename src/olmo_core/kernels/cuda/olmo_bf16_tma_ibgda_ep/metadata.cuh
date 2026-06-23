#pragma once

#include <stdint.h>

namespace olmo::tma_ibgda_ep {

enum RouteFlags : int32_t {
  ROUTE_FLAG_VALID = 1 << 0,
};

struct RouteRecord {
  int32_t source_row;
  int32_t topk_slot;
  int32_t peer_rank;
  int32_t peer_row;
  float prob;
  int32_t flags;
  int32_t reserved0;
  int32_t reserved1;
};

static_assert(sizeof(RouteRecord) == 32, "RouteRecord must stay 32 bytes");
static_assert(alignof(RouteRecord) == 4, "RouteRecord alignment changed");

struct RouteMetadataView {
  const int64_t* route_ranks;
  const int64_t* route_rows;
  const float* route_probs;
  RouteRecord* route_records;
  int64_t* routes_per_rank;
  int64_t* rank_offsets;
  uint8_t* overflow_by_rank;
  int64_t num_tokens;
  int64_t top_k;
  int32_t ep_world_size;
  int32_t rank_capacity;
  int32_t static_route_budget;
};

}  // namespace olmo::tma_ibgda_ep

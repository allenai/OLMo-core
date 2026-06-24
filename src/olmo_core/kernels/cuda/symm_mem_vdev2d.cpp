#include <torch/extension.h>
#include <optional>
#include <pybind11/stl.h>

namespace py = pybind11;

std::vector<uint8_t> olmo_symm_get_unique_id();

void olmo_symm_init(
    const std::vector<std::vector<uint8_t>>& unique_ids,
    int64_t rank,
    int64_t world_size,
    int64_t device_idx);

torch::Tensor olmo_symm_empty(
    const std::vector<int64_t>& sizes,
    c10::ScalarType dtype,
    c10::Device device);

torch::Tensor olmo_symm_peer_base_ptrs(
    torch::Tensor& tensor,
    const std::string& group_name);

void olmo_symm_register_group(
    const std::string& group_name,
    const std::vector<int64_t>& rank_to_pe);

bool olmo_symm_has_group(const std::string& group_name);

void olmo_symm_world_barrier();

void rowwise_signal_peers_on_stream(
    torch::Tensor& signals,
    int64_t signal_row,
    int64_t generation,
    const std::string& group_name,
    bool quiet_before_signal);

void rowwise_wait_signal_peers_on_stream(
    torch::Tensor& signals,
    int64_t signal_row,
    int64_t generation,
    const std::string& group_name);

void all_to_all_vdev_2d_nblocks(
    torch::Tensor& input,
    torch::Tensor& out,
    torch::Tensor& in_splits,
    torch::Tensor& out_splits_offsets,
    const std::string& group_name,
    int64_t major_align,
    int64_t nblocks);

void all_to_all_vdev_2d_offset_nblocks(
    torch::Tensor& input,
    torch::Tensor& out,
    torch::Tensor& in_splits_offsets,
    torch::Tensor& out_splits_offsets,
    const std::string& group_name,
    int64_t nblocks);

void rowwise_dispatch_put(
    torch::Tensor& input,
    torch::Tensor& out,
    torch::Tensor& dst_ranks,
    torch::Tensor& dst_rows,
    const std::optional<torch::Tensor>& probs,
    const std::string& group_name,
    int64_t nblocks,
    bool pre_barrier,
    bool post_barrier);

void rowwise_build_compact_route_records(
    torch::Tensor& dst_ranks,
    torch::Tensor& dst_rows,
    torch::Tensor& route_experts,
    torch::Tensor& route_records,
    torch::Tensor& wave_counts,
    torch::Tensor& wave_fill_counts,
    torch::Tensor& wave_offsets,
    int64_t num_local_experts,
    int64_t num_waves,
    int64_t nblocks);

void rowwise_dispatch_put_compact(
    torch::Tensor& input,
    torch::Tensor& out,
    torch::Tensor& route_records,
    torch::Tensor& wave_offsets,
    int64_t wave_idx,
    const std::string& group_name,
    int64_t nblocks,
    bool pre_barrier,
    bool post_barrier);

void rowwise_dispatch_put_compact_weighted(
    torch::Tensor& input,
    torch::Tensor& out,
    torch::Tensor& route_records,
    torch::Tensor& wave_offsets,
    int64_t wave_idx,
    torch::Tensor& probs,
    const std::string& group_name,
    int64_t nblocks,
    bool pre_barrier,
    bool post_barrier);

void rowwise_inverse_route_meta_put_compact(
    torch::Tensor& inverse_route_meta,
    torch::Tensor& route_records,
    torch::Tensor& wave_offsets,
    int64_t src_rank,
    const std::string& group_name,
    int64_t nblocks,
    bool pre_barrier,
    bool post_barrier);

void rowwise_combine_get(
    torch::Tensor& expert_out,
    torch::Tensor& out,
    torch::Tensor& src_ranks,
    torch::Tensor& src_rows,
    const std::optional<torch::Tensor>& probs,
    const std::string& group_name,
    int64_t nblocks,
    const std::optional<torch::Tensor>& gathered_out,
    bool pre_barrier,
    bool post_barrier);

void rowwise_combine_put(
    torch::Tensor& expert_out,
    torch::Tensor& gathered_out,
    torch::Tensor& inverse_route_meta,
    torch::Tensor& row_start,
    torch::Tensor& num_rows,
    const std::string& group_name,
    int64_t nblocks,
    bool pre_barrier,
    bool post_barrier);

void rowwise_reduce_gathered_routes(
    torch::Tensor& gathered,
    torch::Tensor& probs,
    torch::Tensor& out,
    const std::optional<torch::Tensor>& route_ranks);

void rowwise_reduce_gathered_routes_unweighted(
    torch::Tensor& gathered,
    torch::Tensor& out,
    const std::optional<torch::Tensor>& route_ranks);

void rowwise_combine_get_fused(
    torch::Tensor& expert_out,
    torch::Tensor& out,
    torch::Tensor& src_ranks,
    torch::Tensor& src_rows,
    const std::optional<torch::Tensor>& probs,
    const std::string& group_name,
    int64_t nblocks,
    const std::optional<torch::Tensor>& gathered_out,
    bool pre_barrier,
    bool post_barrier);

void rowwise_gather_get(
    torch::Tensor& expert_out,
    torch::Tensor& out,
    torch::Tensor& src_ranks,
    torch::Tensor& src_rows,
    const std::string& group_name,
    int64_t nblocks,
    bool pre_barrier,
    bool post_barrier);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "olmo_symm_get_unique_id",
      &olmo_symm_get_unique_id,
      "Create an NVSHMEM unique ID for OLMo-owned symmetric memory");
  m.def(
      "olmo_symm_init",
      &olmo_symm_init,
      "Initialize OLMo-owned NVSHMEM symmetric memory",
      py::arg("unique_ids"),
      py::arg("rank"),
      py::arg("world_size"),
      py::arg("device_idx"));
  m.def(
      "olmo_symm_empty",
      &olmo_symm_empty,
      "Allocate an OLMo-owned NVSHMEM symmetric tensor",
      py::arg("sizes"),
      py::arg("dtype"),
      py::arg("device"));
  m.def(
      "olmo_symm_peer_base_ptrs",
      &olmo_symm_peer_base_ptrs,
      "Return direct peer-visible base pointers for an OLMo symmetric tensor",
      py::arg("tensor"),
      py::arg("group_name"));
  m.def(
      "olmo_symm_register_group",
      &olmo_symm_register_group,
      "Register an OLMo symmetric-memory group mapping",
      py::arg("group_name"),
      py::arg("rank_to_pe"));
  m.def(
      "olmo_symm_has_group",
      &olmo_symm_has_group,
      "Return whether an OLMo symmetric-memory group mapping exists",
      py::arg("group_name"));
  m.def(
      "olmo_symm_world_barrier",
      &olmo_symm_world_barrier,
      "Enqueue an NVSHMEM_TEAM_WORLD barrier on the current CUDA stream for the OLMo NVSHMEM bootstrap world");
  m.def(
      "rowwise_signal_peers_on_stream",
      &rowwise_signal_peers_on_stream,
      "Quiet prior rowwise NVSHMEM work on the current stream and signal all peers for one signal row",
      py::arg("signals"),
      py::arg("signal_row"),
      py::arg("generation"),
      py::arg("group_name"),
      py::arg("quiet_before_signal") = true);
  m.def(
      "rowwise_wait_signal_peers_on_stream",
      &rowwise_wait_signal_peers_on_stream,
      "Wait on the current stream until all peers have signaled one row",
      py::arg("signals"),
      py::arg("signal_row"),
      py::arg("generation"),
      py::arg("group_name"));

  m.def(
      "all_to_all_vdev_2d_nblocks",
      &all_to_all_vdev_2d_nblocks,
      "NVSHMEM 2D all_to_all_vdev with explicit nblocks",
      py::arg("input"),
      py::arg("out"),
      py::arg("in_splits"),
      py::arg("out_splits_offsets"),
      py::arg("group_name"),
      py::arg("major_align") = 1,
      py::arg("nblocks") = 0);

  m.def(
      "all_to_all_vdev_2d_offset_nblocks",
      &all_to_all_vdev_2d_offset_nblocks,
      "NVSHMEM 2D all_to_all_vdev_offset with explicit nblocks",
      py::arg("input"),
      py::arg("out"),
      py::arg("in_splits_offsets"),
      py::arg("out_splits_offsets"),
      py::arg("group_name"),
      py::arg("nblocks") = 0);

  m.def(
      "rowwise_dispatch_put",
      &rowwise_dispatch_put,
      "NVSHMEM row-wise dispatch: put input rows directly to remote output rows",
      py::arg("input"),
      py::arg("out"),
      py::arg("dst_ranks"),
      py::arg("dst_rows"),
      py::arg("probs") = std::nullopt,
      py::arg("group_name"),
      py::arg("nblocks") = 0,
      py::arg("pre_barrier") = false,
      py::arg("post_barrier") = true);

  m.def(
      "rowwise_build_compact_route_records",
      &rowwise_build_compact_route_records,
      "Build compact row-wise route records and per-wave offsets on GPU",
      py::arg("dst_ranks"),
      py::arg("dst_rows"),
      py::arg("route_experts"),
      py::arg("route_records"),
      py::arg("wave_counts"),
      py::arg("wave_fill_counts"),
      py::arg("wave_offsets"),
      py::arg("num_local_experts"),
      py::arg("num_waves"),
      py::arg("nblocks") = 0);

  m.def(
      "rowwise_dispatch_put_compact",
      &rowwise_dispatch_put_compact,
      "NVSHMEM compact row-wise dispatch: put only active compact route records",
      py::arg("input"),
      py::arg("out"),
      py::arg("route_records"),
      py::arg("wave_offsets"),
      py::arg("wave_idx"),
      py::arg("group_name"),
      py::arg("nblocks") = 0,
      py::arg("pre_barrier") = false,
      py::arg("post_barrier") = true);

  m.def(
      "rowwise_dispatch_put_compact_weighted",
      &rowwise_dispatch_put_compact_weighted,
      "NVSHMEM compact weighted row-wise dispatch: put active compact route records scaled by route probabilities",
      py::arg("input"),
      py::arg("out"),
      py::arg("route_records"),
      py::arg("wave_offsets"),
      py::arg("wave_idx"),
      py::arg("probs"),
      py::arg("group_name"),
      py::arg("nblocks") = 0,
      py::arg("pre_barrier") = false,
      py::arg("post_barrier") = true);

  m.def(
      "rowwise_inverse_route_meta_put_compact",
      &rowwise_inverse_route_meta_put_compact,
      "NVSHMEM compact row-wise metadata PUT for inverse route maps",
      py::arg("inverse_route_meta"),
      py::arg("route_records"),
      py::arg("wave_offsets"),
      py::arg("src_rank"),
      py::arg("group_name"),
      py::arg("nblocks") = 0,
      py::arg("pre_barrier") = false,
      py::arg("post_barrier") = true);

  m.def(
      "rowwise_combine_get",
      &rowwise_combine_get,
      "NVSHMEM row-wise combine: get remote rows and merge to token outputs",
      py::arg("expert_out"),
      py::arg("out"),
      py::arg("src_ranks"),
      py::arg("src_rows"),
      py::arg("probs") = std::nullopt,
      py::arg("group_name"),
      py::arg("nblocks") = 0,
      py::arg("gathered_out") = std::nullopt,
      py::arg("pre_barrier") = true,
      py::arg("post_barrier") = false);

  m.def(
      "rowwise_combine_put",
      &rowwise_combine_put,
      "NVSHMEM row-wise combine PUT: put local expert rows to remote gathered route slots",
      py::arg("expert_out"),
      py::arg("gathered_out"),
      py::arg("inverse_route_meta"),
      py::arg("row_start"),
      py::arg("num_rows"),
      py::arg("group_name"),
      py::arg("nblocks") = 0,
      py::arg("pre_barrier") = false,
      py::arg("post_barrier") = true);

  m.def(
      "rowwise_reduce_gathered_routes",
      &rowwise_reduce_gathered_routes,
      "Reduce gathered row-wise route slots with route probabilities",
      py::arg("gathered"),
      py::arg("probs"),
      py::arg("out"),
      py::arg("route_ranks") = std::nullopt);

  m.def(
      "rowwise_reduce_gathered_routes_unweighted",
      &rowwise_reduce_gathered_routes_unweighted,
      "Reduce gathered row-wise route slots without route probabilities",
      py::arg("gathered"),
      py::arg("out"),
      py::arg("route_ranks") = std::nullopt);

  m.def(
      "rowwise_combine_get_fused",
      &rowwise_combine_get_fused,
      "NVSHMEM row-wise combine fused: get remote rows and reduce in one kernel",
      py::arg("expert_out"),
      py::arg("out"),
      py::arg("src_ranks"),
      py::arg("src_rows"),
      py::arg("probs") = std::nullopt,
      py::arg("group_name"),
      py::arg("nblocks") = 0,
      py::arg("gathered_out") = std::nullopt,
      py::arg("pre_barrier") = true,
      py::arg("post_barrier") = false);

  m.def(
      "rowwise_gather_get",
      &rowwise_gather_get,
      "NVSHMEM row-wise gather: one-to-one get remote rows to local rows",
      py::arg("expert_out"),
      py::arg("out"),
      py::arg("src_ranks"),
      py::arg("src_rows"),
      py::arg("group_name"),
      py::arg("nblocks") = 0,
      py::arg("pre_barrier") = true,
      py::arg("post_barrier") = false);
}

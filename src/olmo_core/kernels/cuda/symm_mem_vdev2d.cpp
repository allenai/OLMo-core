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

void olmo_symm_register_group(
    const std::string& group_name,
    const std::vector<int64_t>& rank_to_pe);

bool olmo_symm_has_group(const std::string& group_name);

void olmo_symm_world_barrier();

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

void rowwise_combine_get_fused(
    torch::Tensor& expert_out,
    torch::Tensor& out,
    torch::Tensor& src_ranks,
    torch::Tensor& src_rows,
    const std::optional<torch::Tensor>& probs,
    const std::string& group_name,
    int64_t nblocks,
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

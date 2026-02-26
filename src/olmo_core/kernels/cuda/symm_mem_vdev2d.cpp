#include <torch/extension.h>
#include <optional>

namespace py = pybind11;

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
    const std::string& group_name,
    int64_t nblocks);

void rowwise_combine_get(
    torch::Tensor& expert_out,
    torch::Tensor& out,
    torch::Tensor& src_ranks,
    torch::Tensor& src_rows,
    const std::optional<torch::Tensor>& probs,
    const std::string& group_name,
    int64_t nblocks);

void rowwise_gather_get(
    torch::Tensor& expert_out,
    torch::Tensor& out,
    torch::Tensor& src_ranks,
    torch::Tensor& src_rows,
    const std::string& group_name,
    int64_t nblocks);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
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
      py::arg("group_name"),
      py::arg("nblocks") = 0);

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
      py::arg("nblocks") = 0);

  m.def(
      "rowwise_gather_get",
      &rowwise_gather_get,
      "NVSHMEM row-wise gather: one-to-one get remote rows to local rows",
      py::arg("expert_out"),
      py::arg("out"),
      py::arg("src_ranks"),
      py::arg("src_rows"),
      py::arg("group_name"),
      py::arg("nblocks") = 0);
}

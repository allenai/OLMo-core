#include <torch/extension.h>

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
}

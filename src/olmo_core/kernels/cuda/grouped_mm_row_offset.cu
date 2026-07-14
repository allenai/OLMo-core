#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/GroupedMMUtils.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/macros/Macros.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace py = pybind11;

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wset-but-not-used")
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-but-set-parameter")
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-but-set-variable")

#if !defined(USE_ROCM) && !defined(_WIN32) && defined(CUDA_VERSION)
#define OLMO_BUILD_GG_ROW_OFFSET_KERNEL
#endif

#if defined(OLMO_BUILD_GG_ROW_OFFSET_KERNEL)

#include <cute/tensor.hpp>
#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/trace.h>
#include <cutlass/version.h>

#include <ATen/native/cuda/GroupMMCommon.cuh>

#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>

#include <cute/atom/mma_atom.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>

#include <ATen/native/cuda/cutlass_common.cuh>

namespace {

using Strides = at::cuda::detail::Strides;

int ceildiv(int a, int b) {
  return (a + b - 1) / b;
}

int round_up_to_nearest_multiple(int a, int b) {
  return ceildiv(a, b) * b;
}

template <typename ArchTag, bool PONGOr2SM, typename TB_M, typename TB_N, typename TB_K>
struct Schedule {
  using CooperativeSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
  using PongSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
  using CooperativeEpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
  using PongEpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using MMA1SMKernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100;
  using MMA1SMEpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
  using MMA2SMKernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100;
  using MMA2SMEpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;

  using KernelSchedule = cute::conditional_t<std::is_same_v<ArchTag, cutlass::arch::Sm100>,
    cute::conditional_t<PONGOr2SM, MMA2SMKernelSchedule, MMA1SMKernelSchedule>,
    cute::conditional_t<PONGOr2SM, PongSchedule, CooperativeSchedule>>;
  using EpilogueSchedule = cute::conditional_t<std::is_same_v<ArchTag, cutlass::arch::Sm100>,
    cute::conditional_t<PONGOr2SM, MMA2SMEpilogueSchedule, MMA1SMEpilogueSchedule>,
    cute::conditional_t<PONGOr2SM, PongEpilogueSchedule, CooperativeEpilogueSchedule>>;
};

template <
    typename DtypeA,
    typename DtypeB,
    typename DtypeOutput,
    typename RowStartT,
    typename ProblemShape,
    typename StrideA,
    typename StrideB,
    typename StrideOutput,
    bool a_row_major,
    bool b_row_major>
__global__ void prepare_grouped_gemm_data_row_offset_2d3d(
    DtypeA* A,
    DtypeB* B,
    DtypeOutput* output,
    DtypeA** A_ptrs,
    DtypeB** B_ptrs,
    DtypeOutput** output_ptrs,
    ProblemShape* problem_sizes,
    StrideA* stride_A,
    StrideB* stride_B,
    StrideOutput* stride_output,
    const int32_t* offs,
    const RowStartT* row_start,
    int32_t N,
    int32_t K,
    Strides tensor_StrideA,
    Strides tensor_StrideB,
    Strides tensor_StrideOutput,
    int64_t tensor_rows) {
  const int32_t tid = threadIdx.x;
  const int32_t start = tid == 0 ? 0 : offs[tid - 1];
  const int32_t end = offs[tid];
  const int32_t M = end - start;
  CUDA_KERNEL_ASSERT(M >= 0 && "expected group row count to be non-negative\n");

  const int64_t base = static_cast<int64_t>(*row_start);
  const int64_t row = base + static_cast<int64_t>(start);
  const int64_t row_end = base + static_cast<int64_t>(end);
  CUDA_KERNEL_ASSERT(base >= 0 && "expected row_start to be non-negative\n");
  CUDA_KERNEL_ASSERT(row_end <= tensor_rows && "expected wave rows to fit mat_a/out\n");

  if (tid < blockDim.x - 1) {
    int align_a = 128 / cutlass::sizeof_bits<DtypeA>::value;
    CUDA_KERNEL_ASSERT(
        M % align_a == 0 &&
        "expected input tensor dynamic dimension byte size to be non-negative multiple of 16\n");
    int align_output = 128 / cutlass::sizeof_bits<DtypeOutput>::value;
    CUDA_KERNEL_ASSERT(
        M % align_output == 0 &&
        "expected output tensor dynamic dimension byte size to be non-negative multiple of 16\n");
  }

  const int64_t lda = a_row_major ? tensor_StrideA[0] : tensor_StrideA[1];
  const int64_t ldb = b_row_major ? tensor_StrideB[1] : tensor_StrideB[2];
  const int64_t ldoutput = tensor_StrideOutput[0];

  A_ptrs[tid] = A + row * tensor_StrideA[0];
  B_ptrs[tid] = B + static_cast<int64_t>(tid) * tensor_StrideB[0];
  output_ptrs[tid] = output + row * ldoutput;
  problem_sizes[tid] = ProblemShape(M, N, K);

  stride_A[tid] = cutlass::make_cute_packed_stride(StrideA{}, {lda, lda, 1});
  stride_B[tid] = cutlass::make_cute_packed_stride(StrideB{}, {ldb, ldb, 1});
  stride_output[tid] =
      cutlass::make_cute_packed_stride(StrideOutput{}, {M, ldoutput, 1});
}

template <
    typename ArchTag,
    bool a_row_major,
    bool b_row_major,
    bool PONGOr2SM,
    typename TB_M,
    typename TB_N,
    typename TB_K,
    typename RowStartT>
void bf16bf16_grouped_gemm_row_offset_impl(
    at::Tensor mat_a,
    at::Tensor mat_b,
    at::Tensor offs,
    at::Tensor row_start,
    at::Tensor& out) {
  using DtypeA = cutlass::bfloat16_t;
  using DtypeB = cutlass::bfloat16_t;
  using DtypeOutput = cutlass::bfloat16_t;
  using DtypeAccum = float;
  using LayoutA = cute::conditional_t<
      a_row_major,
      cutlass::layout::RowMajor,
      cutlass::layout::ColumnMajor>;
  constexpr int AlignmentA = 16 / sizeof(DtypeA);
  using LayoutB = cute::conditional_t<
      b_row_major,
      cutlass::layout::RowMajor,
      cutlass::layout::ColumnMajor>;
  constexpr int AlignmentB = 16 / sizeof(DtypeB);
  using LayoutOutput = cutlass::layout::RowMajor;
  constexpr int AlignmentOutput = 16 / sizeof(DtypeOutput);
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using TileShape = cute::Shape<TB_M, TB_N, TB_K>;
  using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;
  using KernelSchedule =
      typename Schedule<ArchTag, PONGOr2SM, TB_M, TB_N, TB_K>::KernelSchedule;
  using EpilogueSchedule =
      typename Schedule<ArchTag, PONGOr2SM, TB_M, TB_N, TB_K>::EpilogueSchedule;
  using ProblemShape = cutlass::gemm::GroupProblemShape<
      cute::Shape<int32_t, int32_t, int32_t>>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          TileShape,
          ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          DtypeAccum,
          DtypeAccum,
          void,
          LayoutOutput*,
          AlignmentOutput,
          DtypeOutput,
          LayoutOutput*,
          AlignmentOutput,
          EpilogueSchedule,
          cutlass::epilogue::fusion::
              LinearCombination<DtypeOutput, DtypeAccum>>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          DtypeA,
          LayoutA*,
          AlignmentA,
          DtypeB,
          LayoutB*,
          AlignmentB,
          DtypeAccum,
          TileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using GemmKernelBase = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape,
      CollectiveMainloop,
      CollectiveEpilogue>;

  using GemmKernel = std::conditional_t<
      std::is_same_v<ArchTag, cutlass::arch::Sm100>,
      at::cuda::detail::enable_3x_kernel_for_sm10<GemmKernelBase>,
      at::cuda::detail::enable_3x_kernel_for_sm9x<GemmKernelBase>>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideOutput = typename Gemm::GemmKernel::InternalStrideD;

  const int32_t group_count = static_cast<int32_t>(mat_b.size(0));
  TORCH_CHECK(group_count > 0, "group_count must be > 0");
  TORCH_CHECK(group_count < 1024, "Can't process more than 1024 groups");

  const int32_t approx_m = static_cast<int32_t>(mat_a.size(0) / group_count);
  const int32_t n = static_cast<int32_t>(mat_b.size(-1));
  const int32_t k = static_cast<int32_t>(mat_a.size(-1));

  const int64_t problem_shape_size =
      group_count * static_cast<int64_t>(sizeof(ProblemShape::UnderlyingProblemShape));
  const int64_t stride_size = 3 * group_count * static_cast<int64_t>(sizeof(StrideA));
  const int group_alignment = 16 / sizeof(void*);
  const int aligned_group_count =
      round_up_to_nearest_multiple(group_count, group_alignment);
  const int64_t input_args_size = aligned_group_count * 3 * sizeof(void*) +
      problem_shape_size + stride_size;

  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  auto input_buf = allocator.allocate(input_args_size);
  void* buf_ptr = input_buf.get();
  DtypeA** inputA_ptrs = reinterpret_cast<DtypeA**>(buf_ptr);
  DtypeB** inputB_ptrs =
      reinterpret_cast<DtypeB**>(inputA_ptrs + aligned_group_count);
  DtypeOutput** output_ptrs =
      reinterpret_cast<DtypeOutput**>(inputB_ptrs + aligned_group_count);
  static_assert(
      sizeof(StrideA) == 8, "expected StrideA to be 8 bytes for alignment");
  StrideA* stride_A =
      reinterpret_cast<StrideA*>(output_ptrs + aligned_group_count);
  StrideB* stride_B = reinterpret_cast<StrideB*>(stride_A + group_count);
  StrideOutput* stride_output =
      reinterpret_cast<StrideOutput*>(stride_B + group_count);
  ProblemShape::UnderlyingProblemShape* problem_sizes =
      reinterpret_cast<ProblemShape::UnderlyingProblemShape*>(
          stride_output + group_count);

  auto make_strides = [](at::IntArrayRef strides) -> Strides {
    Strides out_strides;
    std::copy(strides.begin(), strides.end(), out_strides.begin());
    return out_strides;
  };

  const Strides tensor_stride_a = make_strides(mat_a.strides());
  const Strides tensor_stride_b = make_strides(mat_b.strides());
  const Strides tensor_stride_output = make_strides(out.strides());

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  prepare_grouped_gemm_data_row_offset_2d3d<
      DtypeA,
      DtypeB,
      DtypeOutput,
      RowStartT,
      typename ProblemShape::UnderlyingProblemShape,
      StrideA,
      StrideB,
      StrideOutput,
      a_row_major,
      b_row_major><<<1, group_count, 0, stream>>>(
      reinterpret_cast<DtypeA*>(mat_a.data_ptr()),
      reinterpret_cast<DtypeB*>(mat_b.data_ptr()),
      reinterpret_cast<DtypeOutput*>(out.data_ptr()),
      inputA_ptrs,
      inputB_ptrs,
      output_ptrs,
      problem_sizes,
      stride_A,
      stride_B,
      stride_output,
      offs.const_data_ptr<int32_t>(),
      row_start.const_data_ptr<RowStartT>(),
      n,
      k,
      tensor_stride_a,
      tensor_stride_b,
      tensor_stride_output,
      mat_a.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {group_count, problem_sizes, nullptr},
      {(const DtypeA**)inputA_ptrs,
       stride_A,
       (const DtypeB**)inputB_ptrs,
       stride_B},
      {{},
       nullptr,
       stride_output,
       output_ptrs,
       stride_output}};
  arguments.epilogue.thread.alpha = 1.0;
  arguments.epilogue.thread.dAlpha = {cute::_0{}, cute::_0{}, 0};

  int sm_count =
      at::cuda::getDeviceProperties(out.device().index())->multiProcessorCount;
  if (at::globalContext()._SMCarveout_EXPERIMENTAL().has_value()) {
    sm_count -= at::globalContext()._SMCarveout_EXPERIMENTAL().value();
  }
  arguments.hw_info.sm_count = sm_count;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace = allocator.allocate(workspace_size);
  Gemm gemm;
  TORCH_CHECK(
      gemm.can_implement(arguments) == cutlass::Status::kSuccess,
      "cutlass cannot implement");
  TORCH_CHECK(
      gemm.initialize(arguments, workspace.get()) == cutlass::Status::kSuccess,
      "cutlass cannot initialize");
  auto status = gemm(at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "cutlass cannot run, error ",
      int(status));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool a_row_major, bool b_row_major, typename RowStartT>
void dispatch_bf16_grouped_row_offset_on_tile_size(
    at::Tensor mat_a,
    at::Tensor mat_b,
    at::Tensor offs,
    at::Tensor row_start,
    at::Tensor& out) {
  int32_t m = static_cast<int32_t>(mat_a.size(-2));
  int32_t n = static_cast<int32_t>(mat_b.size(-1));
  int32_t group_count = static_cast<int32_t>(mat_b.size(0));
  m = m / group_count;
  bool small = (m <= 128 || n <= 128);
  cudaDeviceProp* properties = at::cuda::getCurrentDeviceProperties();
  const bool sm10x = properties != nullptr && properties->major == 10;
  const bool sm11x = properties != nullptr && properties->major == 11;

  if (sm10x || sm11x) {
    if (small) {
      bf16bf16_grouped_gemm_row_offset_impl<
          cutlass::arch::Sm100,
          a_row_major,
          b_row_major,
          false,
          cute::_128,
          cute::_256,
          cute::_64,
          RowStartT>(mat_a, mat_b, offs, row_start, out);
    } else {
      bf16bf16_grouped_gemm_row_offset_impl<
          cutlass::arch::Sm100,
          a_row_major,
          b_row_major,
          true,
          cute::_256,
          cute::_256,
          cute::_64,
          RowStartT>(mat_a, mat_b, offs, row_start, out);
    }
  } else {
    if (small) {
      bf16bf16_grouped_gemm_row_offset_impl<
          cutlass::arch::Sm90,
          a_row_major,
          b_row_major,
          true,
          cute::_64,
          cute::_128,
          cute::_128,
          RowStartT>(mat_a, mat_b, offs, row_start, out);
    } else {
      bf16bf16_grouped_gemm_row_offset_impl<
          cutlass::arch::Sm90,
          a_row_major,
          b_row_major,
          false,
          cute::_128,
          cute::_256,
          cute::_64,
          RowStartT>(mat_a, mat_b, offs, row_start, out);
    }
  }
}

template <typename RowStartT>
void dispatch_bf16_grouped_row_offset_on_layout(
    at::Tensor mat_a,
    at::Tensor mat_b,
    at::Tensor offs,
    at::Tensor row_start,
    at::Tensor& out) {
  bool a_row_major = mat_a.stride(-1) == 1;
  bool b_row_major = mat_b.stride(-1) == 1;
  if (a_row_major && b_row_major) {
    dispatch_bf16_grouped_row_offset_on_tile_size<true, true, RowStartT>(
        mat_a, mat_b, offs, row_start, out);
  } else if (a_row_major && !b_row_major) {
    dispatch_bf16_grouped_row_offset_on_tile_size<true, false, RowStartT>(
        mat_a, mat_b, offs, row_start, out);
  } else if (!a_row_major && b_row_major) {
    dispatch_bf16_grouped_row_offset_on_tile_size<false, true, RowStartT>(
        mat_a, mat_b, offs, row_start, out);
  } else {
    dispatch_bf16_grouped_row_offset_on_tile_size<false, false, RowStartT>(
        mat_a, mat_b, offs, row_start, out);
  }
}

void bf16bf16_grouped_mm_row_offset(
    at::Tensor mat_a,
    at::Tensor mat_b,
    at::Tensor offs,
    at::Tensor row_start,
    at::Tensor& out) {
  if (row_start.scalar_type() == at::kInt) {
    dispatch_bf16_grouped_row_offset_on_layout<int32_t>(
        mat_a, mat_b, offs, row_start, out);
  } else if (row_start.scalar_type() == at::kLong) {
    dispatch_bf16_grouped_row_offset_on_layout<int64_t>(
        mat_a, mat_b, offs, row_start, out);
  } else {
    TORCH_CHECK(false, "row_start must be int32 or int64");
  }
}

} // namespace

#endif

void check_row_offset_grouped_mm_inputs(
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    const at::Tensor& out,
    const at::Tensor& offs,
    const at::Tensor& row_start) {
  TORCH_CHECK(mat_a.is_cuda(), "mat_a must be CUDA");
  TORCH_CHECK(mat_b.is_cuda(), "mat_b must be CUDA");
  TORCH_CHECK(out.is_cuda(), "out must be CUDA");
  TORCH_CHECK(offs.is_cuda(), "offs must be CUDA");
  TORCH_CHECK(row_start.is_cuda(), "row_start must be CUDA");
  TORCH_CHECK(mat_a.device() == mat_b.device(), "mat_a/mat_b device mismatch");
  TORCH_CHECK(out.device() == mat_a.device(), "out/mat_a device mismatch");
  TORCH_CHECK(offs.device() == mat_a.device(), "offs/mat_a device mismatch");
  TORCH_CHECK(row_start.device() == mat_a.device(), "row_start/mat_a device mismatch");
  TORCH_CHECK(mat_a.dim() == 2, "mat_a must be rank-2 [M, K]");
  TORCH_CHECK(mat_b.dim() == 3, "mat_b must be rank-3 [G, K, N]");
  TORCH_CHECK(out.dim() == 2, "out must be rank-2 [M, N]");
  TORCH_CHECK(offs.dim() == 1, "offs must be rank-1");
  TORCH_CHECK(row_start.numel() == 1, "row_start must be a scalar tensor");
  TORCH_CHECK(offs.scalar_type() == at::kInt, "offs must be int32");
  TORCH_CHECK(
      row_start.scalar_type() == at::kInt || row_start.scalar_type() == at::kLong,
      "row_start must be int32 or int64");
  TORCH_CHECK(mat_a.scalar_type() == at::kBFloat16, "mat_a must be bf16");
  TORCH_CHECK(mat_b.scalar_type() == at::kBFloat16, "mat_b must be bf16");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16, "out must be bf16");
  TORCH_CHECK(mat_a.size(1) == mat_b.size(1), "mat_a K must match mat_b K");
  TORCH_CHECK(out.size(0) == mat_a.size(0), "out rows must match mat_a rows");
  TORCH_CHECK(out.size(1) == mat_b.size(2), "out N must match mat_b N");
  TORCH_CHECK(offs.size(0) == mat_b.size(0), "offs length must match mat_b groups");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(mat_a.data_ptr()) % 16 == 0, "mat_a data_ptr must be 16-byte aligned");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(mat_b.data_ptr()) % 16 == 0, "mat_b data_ptr must be 16-byte aligned");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0, "out data_ptr must be 16-byte aligned");
  at::native::check_valid_strides_and_return_transposed(mat_a);
  at::native::check_valid_strides_and_return_transposed(mat_b);
  at::native::check_valid_strides_and_return_transposed(out);
}

at::Tensor grouped_mm_out_row_offset_cuda(
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    at::Tensor out,
    const at::Tensor& offs,
    const at::Tensor& row_start) {
  check_row_offset_grouped_mm_inputs(mat_a, mat_b, out, offs, row_start);
#if defined(OLMO_BUILD_GG_ROW_OFFSET_KERNEL)
  bf16bf16_grouped_mm_row_offset(mat_a, mat_b, offs, row_start, out);
  return out;
#else
  TORCH_CHECK(false, "grouped_mm row-offset CUDA kernel is not supported on this system");
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "grouped_mm_out_row_offset_cuda",
      &grouped_mm_out_row_offset_cuda,
      "BF16 grouped MM with explicit output and CUDA row offset (2D x 3D)",
      py::arg("mat_a"),
      py::arg("mat_b"),
      py::arg("out"),
      py::arg("offs"),
      py::arg("row_start"));
}

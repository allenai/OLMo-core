#include <ATen/ceil_div.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "olmo_bf16_mega_moe/forward.cuh"
#include "olmo_bf16_mega_moe/forward_kernels.cuh"
#include "olmo_bf16_mega_moe/tensor_map.cuh"

namespace {

std::string wave_cu_result_string(CUresult result) {
  const char* name = nullptr;
  const char* desc = nullptr;
  (void)cuGetErrorName(result, &name);
  (void)cuGetErrorString(result, &desc);
  std::string msg;
  if (name != nullptr) {
    msg += name;
  }
  if (desc != nullptr) {
    if (!msg.empty()) {
      msg += ": ";
    }
    msg += desc;
  }
  return msg;
}

void maybe_init_wave_nvshmem_cumodule(const void* kernel_symbol) {
  static std::once_flag once;
  std::call_once(once, [kernel_symbol]() {
    cudaFunction_t cuda_func{};
    const cudaError_t rt_status = cudaGetFuncBySymbol(&cuda_func, kernel_symbol);
    TORCH_CHECK(
        rt_status == cudaSuccess,
        "cudaGetFuncBySymbol failed while initializing BF16 wave NVSHMEM module: ",
        cudaGetErrorString(rt_status));

    CUmodule cu_module{};
    const CUresult cu_status = cuFuncGetModule(
        &cu_module,
        reinterpret_cast<CUfunction>(cuda_func));
    TORCH_CHECK(
        cu_status == CUDA_SUCCESS,
        "cuFuncGetModule failed while initializing BF16 wave NVSHMEM module (",
        static_cast<int>(cu_status),
        "): ",
        wave_cu_result_string(cu_status));

    const int nv_status = nvshmemx_cumodule_init(cu_module);
    TORCH_CHECK(
        nv_status == 0,
        "nvshmemx_cumodule_init failed for BF16 wave module with status ",
        nv_status);
  });
}

olmo::bf16_mega_moe::LaunchConfig make_wave_config(
    int64_t num_rows,
    int64_t top_k,
    int64_t hidden,
    int64_t intermediate,
    int64_t num_local_experts,
    int64_t num_sms) {
  return olmo::bf16_mega_moe::make_launch_config(
      /*num_ranks=*/1,
      /*num_total_experts=*/num_local_experts,
      /*num_experts_per_rank=*/num_local_experts,
      /*num_max_tokens_per_rank=*/num_rows,
      /*num_tokens=*/num_rows,
      top_k,
      hidden,
      intermediate,
      num_sms);
}

void check_route_expert_indices(const at::Tensor& route_expert_indices) {
  TORCH_CHECK(route_expert_indices.is_cuda(), "route_expert_indices must be CUDA");
  TORCH_CHECK(
      route_expert_indices.dim() == 2,
      "route_expert_indices must be rank-2 [tokens, top_k]");
  TORCH_CHECK(route_expert_indices.is_contiguous(), "route_expert_indices must be contiguous");
  TORCH_CHECK(
      route_expert_indices.scalar_type() == at::kLong ||
          route_expert_indices.scalar_type() == at::kInt,
      "route_expert_indices must be int64 or int32");
}

int64_t route_blocks(int64_t num_routes) {
  constexpr int64_t threads = 256;
  return std::max<int64_t>(
      1,
      std::min<int64_t>(at::ceil_div(num_routes, threads), 1024));
}

template <typename FnLong, typename FnInt>
void dispatch_by_route_dtype(
    const at::Tensor& route_expert_indices,
    FnLong&& fn_long,
    FnInt&& fn_int) {
  if (route_expert_indices.scalar_type() == at::kLong) {
    fn_long(route_expert_indices.data_ptr<int64_t>());
  } else {
    fn_int(route_expert_indices.data_ptr<int32_t>());
  }
}

__global__ void peer_route_count_kernel(
    const int64_t* dst_ranks,
    const int64_t* dst_rows,
    int64_t num_routes,
    int64_t ep_world_size,
    int64_t rank_capacity,
    int64_t* routes_per_rank) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t route_idx = blockIdx.x * blockDim.x + threadIdx.x;
       route_idx < num_routes;
       route_idx += stride) {
    const int64_t rank = dst_ranks[route_idx];
    const int64_t row = dst_rows[route_idx];
    if (rank >= 0 && row >= 0 && rank < ep_world_size && row < rank_capacity) {
      atomicAdd(
          reinterpret_cast<unsigned long long*>(routes_per_rank + rank),
          static_cast<unsigned long long>(1));
    }
  }
}

__global__ void peer_route_prefix_kernel(
    const int64_t* routes_per_rank,
    int64_t ep_world_size,
    int64_t static_route_budget,
    int64_t* rank_offsets,
    int64_t* route_cursors,
    uint8_t* overflow_by_rank) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  int64_t offset = 0;
  for (int64_t rank = 0; rank < ep_world_size; ++rank) {
    rank_offsets[rank] = offset;
    route_cursors[rank] = offset;
    const int64_t count = routes_per_rank[rank];
    if (overflow_by_rank != nullptr) {
      overflow_by_rank[rank] =
          static_cast<uint8_t>(static_route_budget > 0 && count > static_route_budget);
    }
    offset += count;
  }
  rank_offsets[ep_world_size] = offset;
}

__global__ void peer_route_records_kernel(
    const int64_t* dst_ranks,
    const int64_t* dst_rows,
    const float* probs,
    int64_t num_tokens,
    int64_t top_k,
    int64_t ep_world_size,
    int64_t rank_capacity,
    int64_t* route_cursors,
    int32_t* route_records_i32,
    float* route_record_probs) {
  const int64_t num_routes = num_tokens * top_k;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t route_idx = blockIdx.x * blockDim.x + threadIdx.x;
       route_idx < num_routes;
       route_idx += stride) {
    const int64_t rank = dst_ranks[route_idx];
    const int64_t row = dst_rows[route_idx];
    if (rank < 0 || row < 0 || rank >= ep_world_size || row >= rank_capacity) {
      continue;
    }
    const int64_t record_idx = atomicAdd(
        reinterpret_cast<unsigned long long*>(route_cursors + rank),
        static_cast<unsigned long long>(1));
    const int64_t source_row = route_idx / top_k;
    const int64_t topk_slot = route_idx - source_row * top_k;
    int32_t* record = route_records_i32 + record_idx * 6;
    record[0] = static_cast<int32_t>(source_row);
    record[1] = static_cast<int32_t>(topk_slot);
    record[2] = static_cast<int32_t>(rank);
    record[3] = static_cast<int32_t>(row);
    record[4] = 1;
    record[5] = static_cast<int32_t>(record_idx);
    route_record_probs[record_idx] = probs == nullptr ? 1.0f : probs[route_idx];
  }
}

__global__ void peer_window_dispatch_bf16_kernel(
    const __nv_bfloat16* source_input,
    const int64_t* dst_ranks,
    const int64_t* dst_rows,
    int64_t num_tokens,
    int64_t top_k,
    int64_t ep_world_size,
    int64_t rank_capacity,
    int64_t hidden,
    __nv_bfloat16* peer_payload) {
  const int64_t total_values = num_tokens * top_k * hidden;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t value_idx = blockIdx.x * blockDim.x + threadIdx.x;
       value_idx < total_values;
       value_idx += stride) {
    const int64_t hidden_idx = value_idx % hidden;
    const int64_t route_idx = value_idx / hidden;
    const int64_t rank = dst_ranks[route_idx];
    const int64_t row = dst_rows[route_idx];
    if (rank < 0 || row < 0 || rank >= ep_world_size || row >= rank_capacity) {
      continue;
    }
    const int64_t token_idx = route_idx / top_k;
    peer_payload[(rank * rank_capacity + row) * hidden + hidden_idx] =
        source_input[token_idx * hidden + hidden_idx];
  }
}

__global__ void peer_window_gather_bf16_kernel(
    const __nv_bfloat16* peer_payload,
    const int64_t* src_ranks,
    const int64_t* src_rows,
    int64_t num_tokens,
    int64_t top_k,
    int64_t ep_world_size,
    int64_t rank_capacity,
    int64_t hidden,
    __nv_bfloat16* gathered_out) {
  const int64_t total_values = num_tokens * top_k * hidden;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t value_idx = blockIdx.x * blockDim.x + threadIdx.x;
       value_idx < total_values;
       value_idx += stride) {
    const int64_t hidden_idx = value_idx % hidden;
    const int64_t route_idx = value_idx / hidden;
    const int64_t rank = src_ranks[route_idx];
    const int64_t row = src_rows[route_idx];
    if (rank < 0 || row < 0 || rank >= ep_world_size || row >= rank_capacity) {
      continue;
    }
    gathered_out[value_idx] =
        peer_payload[(rank * rank_capacity + row) * hidden + hidden_idx];
  }
}

at::Tensor count_routes(
    at::Tensor& route_expert_indices,
    int64_t num_local_experts) {
  check_route_expert_indices(route_expert_indices);
  TORCH_CHECK(num_local_experts > 0, "num_local_experts must be > 0");

  auto counts = at::zeros(
      {num_local_experts},
      at::TensorOptions().device(route_expert_indices.device()).dtype(at::kLong));
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int64_t threads = 256;
  const int64_t num_routes = route_expert_indices.numel();
  const int64_t blocks = route_blocks(num_routes);

  dispatch_by_route_dtype(
      route_expert_indices,
      [&](const int64_t* route_ptr) {
        olmo::bf16_mega_moe::kernels::f1_count_route_experts_kernel<int64_t>
            <<<static_cast<unsigned int>(blocks),
               static_cast<unsigned int>(threads),
               0,
               stream.stream()>>>(
                route_ptr,
                num_routes,
                num_local_experts,
                counts.data_ptr<int64_t>());
      },
      [&](const int32_t* route_ptr) {
        olmo::bf16_mega_moe::kernels::f1_count_route_experts_kernel<int32_t>
            <<<static_cast<unsigned int>(blocks),
               static_cast<unsigned int>(threads),
               0,
               stream.stream()>>>(
                route_ptr,
                num_routes,
                num_local_experts,
                counts.data_ptr<int64_t>());
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return counts;
}

std::vector<at::Tensor> build_route_pack(
    at::Tensor& route_expert_indices,
    int64_t num_local_experts) {
  at::Tensor expert_counts = count_routes(route_expert_indices, num_local_experts);
  const int64_t num_routes = route_expert_indices.numel();
  auto options =
      at::TensorOptions().device(route_expert_indices.device()).dtype(at::kLong);
  auto expert_offsets = at::zeros({num_local_experts + 1}, options);
  auto expert_cursors = at::zeros({num_local_experts}, options);
  auto packed_token_topk_indices = at::full({num_routes}, -1, options);

  auto stream = at::cuda::getCurrentCUDAStream();
  olmo::bf16_mega_moe::kernels::f1_prefix_route_counts_kernel<<<
      1,
      1,
      0,
      stream.stream()>>>(
      expert_counts.data_ptr<int64_t>(),
      num_local_experts,
      expert_offsets.data_ptr<int64_t>(),
      expert_cursors.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  constexpr int64_t threads = 256;
  const int64_t blocks = route_blocks(num_routes);
  dispatch_by_route_dtype(
      route_expert_indices,
      [&](const int64_t* route_ptr) {
        olmo::bf16_mega_moe::kernels::f1_pack_route_indices_kernel<int64_t>
            <<<static_cast<unsigned int>(blocks),
               static_cast<unsigned int>(threads),
               0,
               stream.stream()>>>(
                route_ptr,
                num_routes,
                num_local_experts,
                expert_cursors.data_ptr<int64_t>(),
                packed_token_topk_indices.data_ptr<int64_t>());
      },
      [&](const int32_t* route_ptr) {
        olmo::bf16_mega_moe::kernels::f1_pack_route_indices_kernel<int32_t>
            <<<static_cast<unsigned int>(blocks),
               static_cast<unsigned int>(threads),
               0,
               stream.stream()>>>(
                route_ptr,
                num_routes,
                num_local_experts,
                expert_cursors.data_ptr<int64_t>(),
                packed_token_topk_indices.data_ptr<int64_t>());
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {expert_offsets, packed_token_topk_indices};
}

std::vector<at::Tensor> build_grouped_gemm_metadata(
    at::Tensor& route_expert_indices,
    int64_t num_local_experts,
    int64_t block_m) {
  TORCH_CHECK(block_m > 0, "block_m must be > 0");
  at::Tensor expert_counts = count_routes(route_expert_indices, num_local_experts);
  auto options =
      at::TensorOptions().device(route_expert_indices.device()).dtype(at::kLong);
  auto token_offsets = at::zeros({num_local_experts + 1}, options);
  auto token_cursors = at::zeros({num_local_experts}, options);
  auto tile_counts = at::zeros({num_local_experts}, options);
  auto tile_offsets = at::zeros({num_local_experts + 1}, options);
  auto num_total_m_tiles = at::zeros({1}, options);

  auto stream = at::cuda::getCurrentCUDAStream();
  olmo::bf16_mega_moe::kernels::f1_grouped_gemm_metadata_kernel<<<
      1,
      1,
      0,
      stream.stream()>>>(
      expert_counts.data_ptr<int64_t>(),
      num_local_experts,
      block_m,
      token_offsets.data_ptr<int64_t>(),
      token_cursors.data_ptr<int64_t>(),
      tile_counts.data_ptr<int64_t>(),
      tile_offsets.data_ptr<int64_t>(),
      num_total_m_tiles.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {expert_counts, token_offsets, tile_counts, tile_offsets, num_total_m_tiles};
}

std::vector<at::Tensor> build_grouped_gemm_metadata_from_counts(
    at::Tensor& expert_counts,
    int64_t block_m) {
  TORCH_CHECK(expert_counts.is_cuda(), "expert_counts must be CUDA");
  TORCH_CHECK(expert_counts.scalar_type() == at::kLong, "expert_counts must be int64");
  TORCH_CHECK(expert_counts.dim() == 1, "expert_counts must be rank-1 [num_local_experts]");
  TORCH_CHECK(expert_counts.is_contiguous(), "expert_counts must be contiguous");
  TORCH_CHECK(block_m > 0, "block_m must be > 0");

  const int64_t num_local_experts = expert_counts.numel();
  TORCH_CHECK(num_local_experts > 0, "num_local_experts must be > 0");
  auto options = expert_counts.options();
  auto token_offsets = at::zeros({num_local_experts + 1}, options);
  auto token_cursors = at::zeros({num_local_experts}, options);
  auto tile_counts = at::zeros({num_local_experts}, options);
  auto tile_offsets = at::zeros({num_local_experts + 1}, options);
  auto num_total_m_tiles = at::zeros({1}, options);

  auto stream = at::cuda::getCurrentCUDAStream();
  olmo::bf16_mega_moe::kernels::f1_grouped_gemm_metadata_kernel<<<
      1,
      1,
      0,
      stream.stream()>>>(
      expert_counts.data_ptr<int64_t>(),
      num_local_experts,
      block_m,
      token_offsets.data_ptr<int64_t>(),
      token_cursors.data_ptr<int64_t>(),
      tile_counts.data_ptr<int64_t>(),
      tile_offsets.data_ptr<int64_t>(),
      num_total_m_tiles.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {expert_counts, token_offsets, tile_counts, tile_offsets, num_total_m_tiles};
}

}  // namespace

std::vector<int64_t> rowwise_bf16_mega_moe_forward_config(
    int64_t num_rows,
    int64_t top_k,
    int64_t hidden,
    int64_t intermediate,
    int64_t num_local_experts,
    int64_t num_sms) {
  return olmo::bf16_mega_moe::to_vector(make_wave_config(
      num_rows,
      top_k,
      hidden,
      intermediate,
      num_local_experts,
      num_sms));
}

at::Tensor rowwise_bf16_mega_moe_sm100_tma_umma_contract_debug() {
  auto debug = at::zeros(
      {olmo::bf16_mega_moe::kernels::kSm100TmaUmmaContractDebugValues},
      at::TensorOptions().device(at::kCUDA).dtype(at::kLong));
  auto stream = at::cuda::getCurrentCUDAStream();
  olmo::bf16_mega_moe::kernels::sm100_tma_umma_contract_kernel<<<
      1,
      128,
      0,
      stream.stream()>>>(debug.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return debug;
}

at::Tensor rowwise_bf16_mega_moe_sm100_tma_load_contract_debug(
    at::Tensor& source) {
  TORCH_CHECK(source.is_cuda(), "source must be CUDA");
  TORCH_CHECK(source.scalar_type() == at::kBFloat16, "source must be BF16");
  TORCH_CHECK(source.dim() == 2, "source must be rank-2 [rows, cols]");
  TORCH_CHECK(source.is_contiguous(), "source must be contiguous");
  TORCH_CHECK(
      source.size(0) >= olmo::bf16_mega_moe::kernels::kTmaContractRows,
      "source must have at least ",
      olmo::bf16_mega_moe::kernels::kTmaContractRows,
      " rows");
  TORCH_CHECK(
      source.size(1) >= olmo::bf16_mega_moe::kernels::kTmaContractCols,
      "source must have at least ",
      olmo::bf16_mega_moe::kernels::kTmaContractCols,
      " columns");

  const olmo::bf16_mega_moe::Bf16TensorMap2D tensor_map =
      olmo::bf16_mega_moe::make_bf16_tma_2d_desc(
          source,
          olmo::bf16_mega_moe::kernels::kTmaContractCols,
          olmo::bf16_mega_moe::kernels::kTmaContractRows,
          /*swizzle_bytes=*/0);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto tensor_map_storage = at::empty(
      {static_cast<int64_t>(sizeof(CUtensorMap))},
      at::TensorOptions().device(source.device()).dtype(at::kByte));
  C10_CUDA_CHECK(cudaMemcpyAsync(
      tensor_map_storage.data_ptr(),
      &tensor_map.map,
      sizeof(CUtensorMap),
      cudaMemcpyHostToDevice,
      stream.stream()));
  auto debug = at::zeros(
      {olmo::bf16_mega_moe::kernels::kTmaContractDebugValues},
      at::TensorOptions().device(source.device()).dtype(at::kLong));
  olmo::bf16_mega_moe::kernels::sm100_bf16_tma_load_contract_kernel<<<
      1,
      128,
      0,
      stream.stream()>>>(
      reinterpret_cast<const CUtensorMap*>(tensor_map_storage.data_ptr()),
      debug.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return debug;
}

at::Tensor rowwise_bf16_mega_moe_sm100_tma_umma_tile_contract_debug(
    at::Tensor& a,
    at::Tensor& b) {
  TORCH_CHECK(a.is_cuda(), "a must be CUDA");
  TORCH_CHECK(b.is_cuda(), "b must be CUDA");
  TORCH_CHECK(a.device() == b.device(), "a and b must be on the same CUDA device");
  TORCH_CHECK(a.scalar_type() == at::kBFloat16, "a must be BF16");
  TORCH_CHECK(b.scalar_type() == at::kBFloat16, "b must be BF16");
  TORCH_CHECK(a.dim() == 2, "a must be rank-2 [M, K]");
  TORCH_CHECK(b.dim() == 2, "b must be rank-2 [N, K]");
  TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
  TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
  TORCH_CHECK(
      a.size(0) >= olmo::bf16_mega_moe::kernels::kSm100TileContractM,
      "a must have at least ",
      olmo::bf16_mega_moe::kernels::kSm100TileContractM,
      " rows");
  TORCH_CHECK(
      a.size(1) >= olmo::bf16_mega_moe::kernels::kSm100TileContractK,
      "a must have at least ",
      olmo::bf16_mega_moe::kernels::kSm100TileContractK,
      " columns");
  TORCH_CHECK(
      b.size(0) >= olmo::bf16_mega_moe::kernels::kSm100TileContractN,
      "b must have at least ",
      olmo::bf16_mega_moe::kernels::kSm100TileContractN,
      " rows");
  TORCH_CHECK(
      b.size(1) >= olmo::bf16_mega_moe::kernels::kSm100TileContractK,
      "b must have at least ",
      olmo::bf16_mega_moe::kernels::kSm100TileContractK,
      " columns");

  constexpr int64_t swizzle_bytes =
      olmo::bf16_mega_moe::kernels::kSm100TileContractK *
      static_cast<int64_t>(sizeof(__nv_bfloat16));
  const olmo::bf16_mega_moe::Bf16TensorMap2D a_tensor_map =
      olmo::bf16_mega_moe::make_bf16_tma_2d_desc(
          a,
          olmo::bf16_mega_moe::kernels::kSm100TileContractK,
          olmo::bf16_mega_moe::kernels::kSm100TileContractM,
          swizzle_bytes);
  const olmo::bf16_mega_moe::Bf16TensorMap2D b_tensor_map =
      olmo::bf16_mega_moe::make_bf16_tma_2d_desc(
          b,
          olmo::bf16_mega_moe::kernels::kSm100TileContractK,
          olmo::bf16_mega_moe::kernels::kSm100TileContractN,
          swizzle_bytes);

  auto stream = at::cuda::getCurrentCUDAStream();
  auto a_tensor_map_storage = at::empty(
      {static_cast<int64_t>(sizeof(CUtensorMap))},
      at::TensorOptions().device(a.device()).dtype(at::kByte));
  auto b_tensor_map_storage = at::empty(
      {static_cast<int64_t>(sizeof(CUtensorMap))},
      at::TensorOptions().device(b.device()).dtype(at::kByte));
  C10_CUDA_CHECK(cudaMemcpyAsync(
      a_tensor_map_storage.data_ptr(),
      &a_tensor_map.map,
      sizeof(CUtensorMap),
      cudaMemcpyHostToDevice,
      stream.stream()));
  C10_CUDA_CHECK(cudaMemcpyAsync(
      b_tensor_map_storage.data_ptr(),
      &b_tensor_map.map,
      sizeof(CUtensorMap),
      cudaMemcpyHostToDevice,
      stream.stream()));

  auto debug = at::zeros(
      {olmo::bf16_mega_moe::kernels::kSm100TmaUmmaTileContractDebugValues},
      at::TensorOptions().device(a.device()).dtype(at::kLong));
  olmo::bf16_mega_moe::kernels::sm100_bf16_tma_umma_tile_contract_kernel<<<
      1,
      128,
      0,
      stream.stream()>>>(
      reinterpret_cast<const CUtensorMap*>(a_tensor_map_storage.data_ptr()),
      reinterpret_cast<const CUtensorMap*>(b_tensor_map_storage.data_ptr()),
      debug.data_ptr<int64_t>(),
      nullptr,
      /*b_mn_major=*/false);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return debug;
}

at::Tensor rowwise_bf16_mega_moe_sm100_tma_umma_tile_forward_debug(
    at::Tensor& a,
    at::Tensor& b) {
  TORCH_CHECK(a.is_cuda(), "a must be CUDA");
  TORCH_CHECK(b.is_cuda(), "b must be CUDA");
  TORCH_CHECK(a.device() == b.device(), "a and b must be on the same CUDA device");
  TORCH_CHECK(a.scalar_type() == at::kBFloat16, "a must be BF16");
  TORCH_CHECK(b.scalar_type() == at::kBFloat16, "b must be BF16");
  TORCH_CHECK(a.dim() == 2, "a must be rank-2 [M, K]");
  TORCH_CHECK(b.dim() == 2, "b must be rank-2 [N, K]");
  TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
  TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
  TORCH_CHECK(
      a.size(0) >= olmo::bf16_mega_moe::kernels::kSm100TileContractM,
      "a must have at least ",
      olmo::bf16_mega_moe::kernels::kSm100TileContractM,
      " rows");
  TORCH_CHECK(
      a.size(1) >= olmo::bf16_mega_moe::kernels::kSm100TileContractK,
      "a must have at least ",
      olmo::bf16_mega_moe::kernels::kSm100TileContractK,
      " columns");
  TORCH_CHECK(
      b.size(0) >= olmo::bf16_mega_moe::kernels::kSm100TileContractN,
      "b must have at least ",
      olmo::bf16_mega_moe::kernels::kSm100TileContractN,
      " rows");
  TORCH_CHECK(
      b.size(1) >= olmo::bf16_mega_moe::kernels::kSm100TileContractK,
      "b must have at least ",
      olmo::bf16_mega_moe::kernels::kSm100TileContractK,
      " columns");

  constexpr int64_t swizzle_bytes =
      olmo::bf16_mega_moe::kernels::kSm100TileContractK *
      static_cast<int64_t>(sizeof(__nv_bfloat16));
  const olmo::bf16_mega_moe::Bf16TensorMap2D a_tensor_map =
      olmo::bf16_mega_moe::make_bf16_tma_2d_desc(
          a,
          olmo::bf16_mega_moe::kernels::kSm100TileContractK,
          olmo::bf16_mega_moe::kernels::kSm100TileContractM,
          swizzle_bytes);
  const olmo::bf16_mega_moe::Bf16TensorMap2D b_tensor_map =
      olmo::bf16_mega_moe::make_bf16_tma_2d_desc(
          b,
          olmo::bf16_mega_moe::kernels::kSm100TileContractK,
          olmo::bf16_mega_moe::kernels::kSm100TileContractN,
          swizzle_bytes);

  auto stream = at::cuda::getCurrentCUDAStream();
  auto a_tensor_map_storage = at::empty(
      {static_cast<int64_t>(sizeof(CUtensorMap))},
      at::TensorOptions().device(a.device()).dtype(at::kByte));
  auto b_tensor_map_storage = at::empty(
      {static_cast<int64_t>(sizeof(CUtensorMap))},
      at::TensorOptions().device(b.device()).dtype(at::kByte));
  C10_CUDA_CHECK(cudaMemcpyAsync(
      a_tensor_map_storage.data_ptr(),
      &a_tensor_map.map,
      sizeof(CUtensorMap),
      cudaMemcpyHostToDevice,
      stream.stream()));
  C10_CUDA_CHECK(cudaMemcpyAsync(
      b_tensor_map_storage.data_ptr(),
      &b_tensor_map.map,
      sizeof(CUtensorMap),
      cudaMemcpyHostToDevice,
      stream.stream()));

  auto debug = at::zeros(
      {olmo::bf16_mega_moe::kernels::kSm100TmaUmmaTileContractDebugValues},
      at::TensorOptions().device(a.device()).dtype(at::kLong));
  auto out = at::empty(
      {olmo::bf16_mega_moe::kernels::kSm100TileContractM,
       olmo::bf16_mega_moe::kernels::kSm100TileContractN},
      at::TensorOptions().device(a.device()).dtype(at::kFloat));
  olmo::bf16_mega_moe::kernels::sm100_bf16_tma_umma_tile_contract_kernel<<<
      1,
      128,
      0,
      stream.stream()>>>(
      reinterpret_cast<const CUtensorMap*>(a_tensor_map_storage.data_ptr()),
      reinterpret_cast<const CUtensorMap*>(b_tensor_map_storage.data_ptr()),
      debug.data_ptr<int64_t>(),
      out.data_ptr<float>(),
      /*b_mn_major=*/false);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor rowwise_bf16_mega_moe_sm100_tma_umma_tile_forward_b_mn_debug(
    at::Tensor& a,
    at::Tensor& b) {
  TORCH_CHECK(a.is_cuda(), "a must be CUDA");
  TORCH_CHECK(b.is_cuda(), "b must be CUDA");
  TORCH_CHECK(a.device() == b.device(), "a and b must be on the same CUDA device");
  TORCH_CHECK(a.scalar_type() == at::kBFloat16, "a must be BF16");
  TORCH_CHECK(b.scalar_type() == at::kBFloat16, "b must be BF16");
  TORCH_CHECK(a.dim() == 2, "a must be rank-2 [M, K]");
  TORCH_CHECK(b.dim() == 2, "b must be rank-2 [K, N]");
  TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
  TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
  TORCH_CHECK(
      a.size(0) >= olmo::bf16_mega_moe::kernels::kSm100TileContractM,
      "a must have at least ",
      olmo::bf16_mega_moe::kernels::kSm100TileContractM,
      " rows");
  TORCH_CHECK(
      a.size(1) >= olmo::bf16_mega_moe::kernels::kSm100TileContractK,
      "a must have at least ",
      olmo::bf16_mega_moe::kernels::kSm100TileContractK,
      " columns");
  TORCH_CHECK(
      b.size(0) >= olmo::bf16_mega_moe::kernels::kSm100TileContractK,
      "b must have at least ",
      olmo::bf16_mega_moe::kernels::kSm100TileContractK,
      " rows");
  TORCH_CHECK(
      b.size(1) >= olmo::bf16_mega_moe::kernels::kSm100TileContractN,
      "b must have at least ",
      olmo::bf16_mega_moe::kernels::kSm100TileContractN,
      " columns");

  constexpr int64_t swizzle_bytes =
      olmo::bf16_mega_moe::kernels::kSm100TileContractK *
      static_cast<int64_t>(sizeof(__nv_bfloat16));
  constexpr int64_t b_atom_cols =
      swizzle_bytes / static_cast<int64_t>(sizeof(__nv_bfloat16));
  const olmo::bf16_mega_moe::Bf16TensorMap2D a_tensor_map =
      olmo::bf16_mega_moe::make_bf16_tma_2d_desc(
          a,
          olmo::bf16_mega_moe::kernels::kSm100TileContractK,
          olmo::bf16_mega_moe::kernels::kSm100TileContractM,
          swizzle_bytes);
  const olmo::bf16_mega_moe::Bf16TensorMap2D b_tensor_map =
      olmo::bf16_mega_moe::make_bf16_tma_2d_desc(
          b,
          b_atom_cols,
          olmo::bf16_mega_moe::kernels::kSm100TileContractK,
          swizzle_bytes);

  auto stream = at::cuda::getCurrentCUDAStream();
  auto a_tensor_map_storage = at::empty(
      {static_cast<int64_t>(sizeof(CUtensorMap))},
      at::TensorOptions().device(a.device()).dtype(at::kByte));
  auto b_tensor_map_storage = at::empty(
      {static_cast<int64_t>(sizeof(CUtensorMap))},
      at::TensorOptions().device(b.device()).dtype(at::kByte));
  C10_CUDA_CHECK(cudaMemcpyAsync(
      a_tensor_map_storage.data_ptr(),
      &a_tensor_map.map,
      sizeof(CUtensorMap),
      cudaMemcpyHostToDevice,
      stream.stream()));
  C10_CUDA_CHECK(cudaMemcpyAsync(
      b_tensor_map_storage.data_ptr(),
      &b_tensor_map.map,
      sizeof(CUtensorMap),
      cudaMemcpyHostToDevice,
      stream.stream()));

  auto debug = at::zeros(
      {olmo::bf16_mega_moe::kernels::kSm100TmaUmmaTileContractDebugValues},
      at::TensorOptions().device(a.device()).dtype(at::kLong));
  auto out = at::empty(
      {olmo::bf16_mega_moe::kernels::kSm100TileContractM,
       olmo::bf16_mega_moe::kernels::kSm100TileContractN},
      at::TensorOptions().device(a.device()).dtype(at::kFloat));
  olmo::bf16_mega_moe::kernels::sm100_bf16_tma_umma_tile_contract_kernel<<<
      1,
      128,
      0,
      stream.stream()>>>(
      reinterpret_cast<const CUtensorMap*>(a_tensor_map_storage.data_ptr()),
      reinterpret_cast<const CUtensorMap*>(b_tensor_map_storage.data_ptr()),
      debug.data_ptr<int64_t>(),
      out.data_ptr<float>(),
      /*b_mn_major=*/true);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor rowwise_bf16_mega_moe_forward_plan_debug(
    int64_t num_rows,
    int64_t top_k,
    int64_t hidden,
    int64_t intermediate,
    int64_t num_local_experts,
    int64_t num_sms) {
  const auto config = make_wave_config(
      num_rows,
      top_k,
      hidden,
      intermediate,
      num_local_experts,
      num_sms);
  auto debug = at::zeros(
      {olmo::bf16_mega_moe::kernels::kForwardPlanDebugRows,
       olmo::bf16_mega_moe::kernels::kForwardPlanDebugCols},
      at::TensorOptions().device(at::kCUDA).dtype(at::kLong));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t f1_blocks =
      std::max<int64_t>(1, std::min<int64_t>(config.forward_plan.f1_gemm_sms, 128));
  const int64_t f2_blocks =
      std::max<int64_t>(1, std::min<int64_t>(config.forward_plan.f2_gemm_sms, 128));
  constexpr int64_t threads = 256;
  olmo::bf16_mega_moe::kernels::f1_forward_contract_kernel<<<
      static_cast<unsigned int>(f1_blocks),
      static_cast<unsigned int>(threads),
      0,
      stream.stream()>>>(config.forward_plan, debug.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  olmo::bf16_mega_moe::kernels::f2_forward_contract_kernel<<<
      static_cast<unsigned int>(f2_blocks),
      static_cast<unsigned int>(threads),
      0,
      stream.stream()>>>(config.forward_plan, debug.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return debug;
}

at::Tensor rowwise_bf16_mega_moe_route_counts_debug(
    at::Tensor& route_expert_indices,
    int64_t num_local_experts) {
  return count_routes(route_expert_indices, num_local_experts);
}

std::vector<at::Tensor> rowwise_bf16_mega_moe_route_pack_debug(
    at::Tensor& route_expert_indices,
    int64_t num_local_experts) {
  return build_route_pack(route_expert_indices, num_local_experts);
}

std::vector<at::Tensor> rowwise_bf16_mega_moe_route_pack_inputs_debug(
    at::Tensor& source_input,
    at::Tensor& route_expert_indices,
    at::Tensor& probs,
    int64_t num_local_experts) {
  TORCH_CHECK(source_input.is_cuda(), "source_input must be CUDA");
  TORCH_CHECK(source_input.scalar_type() == at::kBFloat16, "source_input must be BF16");
  TORCH_CHECK(source_input.dim() == 2, "source_input must be rank-2 [tokens, hidden]");
  TORCH_CHECK(source_input.is_contiguous(), "source_input must be contiguous");
  TORCH_CHECK(probs.is_cuda(), "probs must be CUDA");
  TORCH_CHECK(probs.scalar_type() == at::kFloat, "probs must be FP32");
  TORCH_CHECK(probs.dim() == 2, "probs must be rank-2 [tokens, top_k]");
  TORCH_CHECK(probs.is_contiguous(), "probs must be contiguous");
  check_route_expert_indices(route_expert_indices);
  TORCH_CHECK(
      route_expert_indices.get_device() == source_input.get_device(),
      "route_expert_indices must be on the same device as source_input");
  TORCH_CHECK(
      probs.get_device() == source_input.get_device(),
      "probs must be on the same device as source_input");
  TORCH_CHECK(
      route_expert_indices.size(0) == source_input.size(0),
      "route_expert_indices token dimension mismatch");
  TORCH_CHECK(probs.size(0) == source_input.size(0), "probs token dimension mismatch");
  TORCH_CHECK(probs.size(1) == route_expert_indices.size(1), "probs top_k dimension mismatch");

  std::vector<at::Tensor> packed_meta =
      build_route_pack(route_expert_indices, num_local_experts);
  at::Tensor expert_offsets = packed_meta[0];
  at::Tensor packed_token_topk_indices = packed_meta[1];
  const int64_t num_route_slots = route_expert_indices.numel();
  const int64_t hidden = source_input.size(1);
  auto packed_input = at::zeros(
      {num_route_slots, hidden},
      at::TensorOptions().device(source_input.device()).dtype(at::kBFloat16));
  auto packed_probs = at::zeros(
      {num_route_slots},
      at::TensorOptions().device(source_input.device()).dtype(at::kFloat));

  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int64_t threads = 256;
  const int64_t num_values = num_route_slots * hidden;
  const int64_t blocks =
      std::max<int64_t>(1, std::min<int64_t>(at::ceil_div(num_values, threads), 4096));
  olmo::bf16_mega_moe::kernels::f1_pack_input_rows_kernel<<<
      static_cast<unsigned int>(blocks),
      static_cast<unsigned int>(threads),
      0,
      stream.stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(source_input.data_ptr<at::BFloat16>()),
      probs.data_ptr<float>(),
      packed_token_topk_indices.data_ptr<int64_t>(),
      num_route_slots,
      route_expert_indices.size(1),
      hidden,
      reinterpret_cast<__nv_bfloat16*>(packed_input.data_ptr<at::BFloat16>()),
      packed_probs.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {expert_offsets, packed_token_topk_indices, packed_input, packed_probs};
}

void check_peer_route_maps(
    const at::Tensor& dst_ranks,
    const at::Tensor& dst_rows,
    int64_t ep_world_size,
    int64_t rank_capacity) {
  TORCH_CHECK(dst_ranks.is_cuda(), "route ranks must be CUDA");
  TORCH_CHECK(dst_rows.is_cuda(), "route rows must be CUDA");
  TORCH_CHECK(dst_ranks.scalar_type() == at::kLong, "route ranks must be int64");
  TORCH_CHECK(dst_rows.scalar_type() == at::kLong, "route rows must be int64");
  TORCH_CHECK(dst_ranks.dim() == 2, "route ranks must be rank-2 [tokens, top_k]");
  TORCH_CHECK(dst_rows.dim() == 2, "route rows must be rank-2 [tokens, top_k]");
  TORCH_CHECK(dst_ranks.sizes() == dst_rows.sizes(), "route rank/row shape mismatch");
  TORCH_CHECK(dst_ranks.is_contiguous(), "route ranks must be contiguous");
  TORCH_CHECK(dst_rows.is_contiguous(), "route rows must be contiguous");
  TORCH_CHECK(dst_ranks.get_device() == dst_rows.get_device(),
              "route ranks and rows must be on the same device");
  TORCH_CHECK(ep_world_size > 0, "ep_world_size must be > 0");
  TORCH_CHECK(rank_capacity > 0, "rank_capacity must be > 0");
}

std::vector<at::Tensor> rowwise_bf16_mega_moe_peer_route_metadata_debug(
    at::Tensor& dst_ranks,
    at::Tensor& dst_rows,
    at::Tensor& probs,
    int64_t ep_world_size,
    int64_t rank_capacity,
    int64_t static_route_budget) {
  check_peer_route_maps(dst_ranks, dst_rows, ep_world_size, rank_capacity);
  TORCH_CHECK(probs.is_cuda(), "probs must be CUDA");
  TORCH_CHECK(probs.scalar_type() == at::kFloat, "probs must be FP32");
  TORCH_CHECK(probs.dim() == 2, "probs must be rank-2 [tokens, top_k]");
  TORCH_CHECK(probs.sizes() == dst_ranks.sizes(), "probs shape mismatch");
  TORCH_CHECK(probs.is_contiguous(), "probs must be contiguous");
  TORCH_CHECK(probs.get_device() == dst_ranks.get_device(),
              "probs must be on the same device as route maps");
  TORCH_CHECK(static_route_budget >= 0, "static_route_budget must be >= 0");

  const int64_t num_tokens = dst_ranks.size(0);
  const int64_t top_k = dst_ranks.size(1);
  const int64_t num_routes = num_tokens * top_k;
  auto long_options = at::TensorOptions().device(dst_ranks.device()).dtype(at::kLong);
  auto int_options = at::TensorOptions().device(dst_ranks.device()).dtype(at::kInt);
  auto byte_options = at::TensorOptions().device(dst_ranks.device()).dtype(at::kByte);
  auto float_options = at::TensorOptions().device(dst_ranks.device()).dtype(at::kFloat);

  auto routes_per_rank = at::zeros({ep_world_size}, long_options);
  auto rank_offsets = at::zeros({ep_world_size + 1}, long_options);
  auto route_cursors = at::zeros({ep_world_size}, long_options);
  auto overflow_by_rank = at::zeros({ep_world_size}, byte_options);
  auto route_records_i32 = at::full({num_routes, 6}, -1, int_options);
  auto route_record_probs = at::zeros({num_routes}, float_options);

  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int64_t threads = 256;
  const int64_t route_blocks_count =
      std::max<int64_t>(1, std::min<int64_t>(at::ceil_div(num_routes, threads), 4096));
  peer_route_count_kernel<<<
      static_cast<unsigned int>(route_blocks_count),
      static_cast<unsigned int>(threads),
      0,
      stream.stream()>>>(
      dst_ranks.data_ptr<int64_t>(),
      dst_rows.data_ptr<int64_t>(),
      num_routes,
      ep_world_size,
      rank_capacity,
      routes_per_rank.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  peer_route_prefix_kernel<<<1, 1, 0, stream.stream()>>>(
      routes_per_rank.data_ptr<int64_t>(),
      ep_world_size,
      static_route_budget,
      rank_offsets.data_ptr<int64_t>(),
      route_cursors.data_ptr<int64_t>(),
      overflow_by_rank.data_ptr<uint8_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  peer_route_records_kernel<<<
      static_cast<unsigned int>(route_blocks_count),
      static_cast<unsigned int>(threads),
      0,
      stream.stream()>>>(
      dst_ranks.data_ptr<int64_t>(),
      dst_rows.data_ptr<int64_t>(),
      probs.data_ptr<float>(),
      num_tokens,
      top_k,
      ep_world_size,
      rank_capacity,
      route_cursors.data_ptr<int64_t>(),
      route_records_i32.data_ptr<int32_t>(),
      route_record_probs.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {
      route_records_i32,
      route_record_probs,
      routes_per_rank,
      rank_offsets,
      overflow_by_rank,
  };
}

at::Tensor rowwise_bf16_mega_moe_peer_window_dispatch_debug(
    at::Tensor& source_input,
    at::Tensor& dst_ranks,
    at::Tensor& dst_rows,
    int64_t ep_world_size,
    int64_t rank_capacity) {
  TORCH_CHECK(source_input.is_cuda(), "source_input must be CUDA");
  TORCH_CHECK(source_input.scalar_type() == at::kBFloat16, "source_input must be BF16");
  TORCH_CHECK(source_input.dim() == 2, "source_input must be rank-2 [tokens, hidden]");
  TORCH_CHECK(source_input.is_contiguous(), "source_input must be contiguous");
  check_peer_route_maps(dst_ranks, dst_rows, ep_world_size, rank_capacity);
  TORCH_CHECK(dst_ranks.get_device() == source_input.get_device(),
              "route maps must be on the same device as source_input");
  TORCH_CHECK(dst_ranks.size(0) == source_input.size(0),
              "route map token dimension mismatch");

  const int64_t num_tokens = source_input.size(0);
  const int64_t top_k = dst_ranks.size(1);
  const int64_t hidden = source_input.size(1);
  auto out = at::zeros(
      {ep_world_size, rank_capacity, hidden},
      at::TensorOptions().device(source_input.device()).dtype(at::kBFloat16));
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int64_t threads = 256;
  const int64_t total_values = num_tokens * top_k * hidden;
  const int64_t blocks =
      std::max<int64_t>(1, std::min<int64_t>(at::ceil_div(total_values, threads), 4096));
  peer_window_dispatch_bf16_kernel<<<
      static_cast<unsigned int>(blocks),
      static_cast<unsigned int>(threads),
      0,
      stream.stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(source_input.data_ptr<at::BFloat16>()),
      dst_ranks.data_ptr<int64_t>(),
      dst_rows.data_ptr<int64_t>(),
      num_tokens,
      top_k,
      ep_world_size,
      rank_capacity,
      hidden,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<at::Tensor> rowwise_bf16_mega_moe_peer_window_combine_debug(
    at::Tensor& peer_payload,
    at::Tensor& src_ranks,
    at::Tensor& src_rows,
    at::Tensor& probs) {
  TORCH_CHECK(peer_payload.is_cuda(), "peer_payload must be CUDA");
  TORCH_CHECK(peer_payload.scalar_type() == at::kBFloat16, "peer_payload must be BF16");
  TORCH_CHECK(peer_payload.dim() == 3,
              "peer_payload must be rank-3 [ep_world_size, rank_capacity, hidden]");
  TORCH_CHECK(peer_payload.is_contiguous(), "peer_payload must be contiguous");
  const int64_t ep_world_size = peer_payload.size(0);
  const int64_t rank_capacity = peer_payload.size(1);
  const int64_t hidden = peer_payload.size(2);
  check_peer_route_maps(src_ranks, src_rows, ep_world_size, rank_capacity);
  TORCH_CHECK(probs.is_cuda(), "probs must be CUDA");
  TORCH_CHECK(probs.scalar_type() == at::kFloat, "probs must be FP32");
  TORCH_CHECK(probs.dim() == 2, "probs must be rank-2 [tokens, top_k]");
  TORCH_CHECK(probs.sizes() == src_ranks.sizes(), "probs shape mismatch");
  TORCH_CHECK(probs.is_contiguous(), "probs must be contiguous");
  TORCH_CHECK(src_ranks.get_device() == peer_payload.get_device(),
              "route maps must be on the same device as peer_payload");
  TORCH_CHECK(probs.get_device() == peer_payload.get_device(),
              "probs must be on the same device as peer_payload");

  const int64_t num_tokens = src_ranks.size(0);
  const int64_t top_k = src_ranks.size(1);
  auto bf16_options =
      at::TensorOptions().device(peer_payload.device()).dtype(at::kBFloat16);
  auto gathered_out = at::zeros({num_tokens, top_k, hidden}, bf16_options);
  auto out = at::zeros({num_tokens, hidden}, bf16_options);
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int64_t threads = 256;
  const int64_t gather_values = num_tokens * top_k * hidden;
  const int64_t gather_blocks =
      std::max<int64_t>(1, std::min<int64_t>(at::ceil_div(gather_values, threads), 4096));
  peer_window_gather_bf16_kernel<<<
      static_cast<unsigned int>(gather_blocks),
      static_cast<unsigned int>(threads),
      0,
      stream.stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(peer_payload.data_ptr<at::BFloat16>()),
      src_ranks.data_ptr<int64_t>(),
      src_rows.data_ptr<int64_t>(),
      num_tokens,
      top_k,
      ep_world_size,
      rank_capacity,
      hidden,
      reinterpret_cast<__nv_bfloat16*>(gathered_out.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t reduce_values = num_tokens * hidden;
  const int64_t reduce_blocks =
      std::max<int64_t>(1, std::min<int64_t>(at::ceil_div(reduce_values, threads), 4096));
  olmo::bf16_mega_moe::kernels::f2_reduce_topk_rows_kernel<<<
      static_cast<unsigned int>(reduce_blocks),
      static_cast<unsigned int>(threads),
      0,
      stream.stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(gathered_out.data_ptr<at::BFloat16>()),
      probs.data_ptr<float>(),
      num_tokens,
      top_k,
      hidden,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {gathered_out, out};
}

std::vector<at::Tensor> rowwise_bf16_mega_moe_grouped_gemm_metadata_debug(
    at::Tensor& route_expert_indices,
    int64_t num_local_experts,
    int64_t block_m) {
  return build_grouped_gemm_metadata(route_expert_indices, num_local_experts, block_m);
}

std::vector<at::Tensor> rowwise_bf16_mega_moe_grouped_gemm_tile_debug(
    at::Tensor& route_expert_indices,
    int64_t num_local_experts,
    int64_t block_m,
    int64_t n_tiles) {
  TORCH_CHECK(n_tiles > 0, "n_tiles must be > 0");
  std::vector<at::Tensor> metadata =
      build_grouped_gemm_metadata(route_expert_indices, num_local_experts, block_m);
  at::Tensor tile_offsets = metadata[3];
  auto debug = at::zeros(
      {num_local_experts + 2, 4},
      at::TensorOptions().device(route_expert_indices.device()).dtype(at::kLong));

  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int64_t threads = 256;
  const int64_t max_tasks = route_expert_indices.numel() * n_tiles;
  const int64_t blocks =
      std::max<int64_t>(1, std::min<int64_t>(at::ceil_div(max_tasks, threads), 4096));
  olmo::bf16_mega_moe::kernels::grouped_gemm_tile_contract_kernel<<<
      static_cast<unsigned int>(blocks),
      static_cast<unsigned int>(threads),
      0,
      stream.stream()>>>(
      tile_offsets.data_ptr<int64_t>(),
      num_local_experts,
      n_tiles,
      max_tasks,
      debug.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  metadata.push_back(debug);
  return metadata;
}

std::vector<at::Tensor> rowwise_bf16_mega_moe_combine_debug(
    at::Tensor& packed_expert_out,
    at::Tensor& packed_token_topk_indices,
    at::Tensor& probs) {
  TORCH_CHECK(packed_expert_out.is_cuda(), "packed_expert_out must be CUDA");
  TORCH_CHECK(packed_expert_out.scalar_type() == at::kBFloat16, "packed_expert_out must be BF16");
  TORCH_CHECK(packed_expert_out.dim() == 2, "packed_expert_out must be rank-2 [routes, hidden]");
  TORCH_CHECK(packed_expert_out.is_contiguous(), "packed_expert_out must be contiguous");
  TORCH_CHECK(packed_token_topk_indices.is_cuda(), "packed_token_topk_indices must be CUDA");
  TORCH_CHECK(packed_token_topk_indices.scalar_type() == at::kLong,
              "packed_token_topk_indices must be int64");
  TORCH_CHECK(packed_token_topk_indices.dim() == 1,
              "packed_token_topk_indices must be rank-1 [routes]");
  TORCH_CHECK(packed_token_topk_indices.is_contiguous(),
              "packed_token_topk_indices must be contiguous");
  TORCH_CHECK(probs.is_cuda(), "probs must be CUDA");
  TORCH_CHECK(probs.scalar_type() == at::kFloat, "probs must be FP32");
  TORCH_CHECK(probs.dim() == 2, "probs must be rank-2 [tokens, top_k]");
  TORCH_CHECK(probs.is_contiguous(), "probs must be contiguous");
  TORCH_CHECK(packed_token_topk_indices.get_device() == packed_expert_out.get_device(),
              "packed_token_topk_indices must be on the same device as packed_expert_out");
  TORCH_CHECK(probs.get_device() == packed_expert_out.get_device(),
              "probs must be on the same device as packed_expert_out");
  TORCH_CHECK(packed_token_topk_indices.numel() == packed_expert_out.size(0),
              "packed route count must match packed_expert_out rows");

  const int64_t num_tokens = probs.size(0);
  const int64_t top_k = probs.size(1);
  const int64_t hidden = packed_expert_out.size(1);
  const int64_t num_route_slots = packed_expert_out.size(0);
  auto gathered_out = at::zeros(
      {num_tokens, top_k, hidden},
      at::TensorOptions().device(packed_expert_out.device()).dtype(at::kBFloat16));
  auto out = at::zeros(
      {num_tokens, hidden},
      at::TensorOptions().device(packed_expert_out.device()).dtype(at::kBFloat16));

  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int64_t threads = 256;
  const int64_t scatter_values = num_route_slots * hidden;
  const int64_t scatter_blocks =
      std::max<int64_t>(1, std::min<int64_t>(at::ceil_div(scatter_values, threads), 4096));
  olmo::bf16_mega_moe::kernels::f2_scatter_packed_rows_kernel<<<
      static_cast<unsigned int>(scatter_blocks),
      static_cast<unsigned int>(threads),
      0,
      stream.stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(packed_expert_out.data_ptr<at::BFloat16>()),
      packed_token_topk_indices.data_ptr<int64_t>(),
      num_route_slots,
      hidden,
      reinterpret_cast<__nv_bfloat16*>(gathered_out.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t reduce_values = num_tokens * hidden;
  const int64_t reduce_blocks =
      std::max<int64_t>(1, std::min<int64_t>(at::ceil_div(reduce_values, threads), 4096));
  olmo::bf16_mega_moe::kernels::f2_reduce_topk_rows_kernel<<<
      static_cast<unsigned int>(reduce_blocks),
      static_cast<unsigned int>(threads),
      0,
      stream.stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(gathered_out.data_ptr<at::BFloat16>()),
      probs.data_ptr<float>(),
      num_tokens,
      top_k,
      hidden,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {gathered_out, out};
}

std::vector<at::Tensor> rowwise_bf16_mega_moe_w1_wmma_debug(
    at::Tensor& source_input,
    at::Tensor& route_expert_indices,
    at::Tensor& up_gate_weight) {
  TORCH_CHECK(source_input.is_cuda(), "source_input must be CUDA");
  TORCH_CHECK(source_input.scalar_type() == at::kBFloat16, "source_input must be BF16");
  TORCH_CHECK(source_input.dim() == 2, "source_input must be rank-2 [tokens, hidden]");
  TORCH_CHECK(source_input.is_contiguous(), "source_input must be contiguous");
  check_route_expert_indices(route_expert_indices);
  TORCH_CHECK(up_gate_weight.is_cuda(), "up_gate_weight must be CUDA");
  TORCH_CHECK(up_gate_weight.scalar_type() == at::kBFloat16, "up_gate_weight must be BF16");
  TORCH_CHECK(up_gate_weight.dim() == 3, "up_gate_weight must be rank-3 [experts, out, hidden]");
  TORCH_CHECK(up_gate_weight.is_contiguous(), "up_gate_weight must be contiguous");
  TORCH_CHECK(route_expert_indices.get_device() == source_input.get_device(),
              "route_expert_indices must be on the same device as source_input");
  TORCH_CHECK(up_gate_weight.get_device() == source_input.get_device(),
              "up_gate_weight must be on the same device as source_input");
  TORCH_CHECK(up_gate_weight.size(2) == source_input.size(1),
              "up_gate_weight hidden dimension mismatch");

  const int64_t num_local_experts = up_gate_weight.size(0);
  const int64_t hidden = source_input.size(1);
  const int64_t out_features = up_gate_weight.size(1);
  const int64_t num_route_slots = route_expert_indices.numel();
  auto probs = at::ones(
      route_expert_indices.sizes(),
      at::TensorOptions().device(source_input.device()).dtype(at::kFloat));
  std::vector<at::Tensor> packed_inputs =
      rowwise_bf16_mega_moe_route_pack_inputs_debug(
          source_input,
          route_expert_indices,
          probs,
          num_local_experts);
  at::Tensor packed_route = packed_inputs[1];
  at::Tensor packed_input = packed_inputs[2];
  std::vector<at::Tensor> metadata =
      build_grouped_gemm_metadata(
          route_expert_indices,
          num_local_experts,
          olmo::bf16_mega_moe::kernels::kWmmaM);
  at::Tensor expert_counts = metadata[0];
  at::Tensor token_offsets = metadata[1];
  at::Tensor tile_offsets = metadata[3];
  auto w1_out = at::zeros(
      {num_route_slots, out_features},
      at::TensorOptions().device(source_input.device()).dtype(at::kBFloat16));

  const int64_t n_tiles =
      at::ceil_div(out_features, olmo::bf16_mega_moe::kernels::kWmmaN);
  const int64_t max_tasks = num_route_slots * n_tiles;
  const int64_t blocks = std::max<int64_t>(1, std::min<int64_t>(max_tasks, 4096));
  auto stream = at::cuda::getCurrentCUDAStream();
  olmo::bf16_mega_moe::kernels::grouped_w1_wmma_kernel<<<
      static_cast<unsigned int>(blocks),
      32,
      0,
      stream.stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(packed_input.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(up_gate_weight.data_ptr<at::BFloat16>()),
      expert_counts.data_ptr<int64_t>(),
      token_offsets.data_ptr<int64_t>(),
      tile_offsets.data_ptr<int64_t>(),
      num_local_experts,
      num_route_slots,
      hidden,
      out_features,
      n_tiles,
      max_tasks,
      reinterpret_cast<__nv_bfloat16*>(w1_out.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  metadata.push_back(packed_route);
  metadata.push_back(packed_input);
  metadata.push_back(w1_out);
  return metadata;
}

std::vector<at::Tensor> rowwise_bf16_mega_moe_forward_debug(
    at::Tensor& source_input,
    at::Tensor& route_expert_indices,
    at::Tensor& probs,
    at::Tensor& up_gate_weight,
    at::Tensor& down_weight) {
  TORCH_CHECK(source_input.is_cuda(), "source_input must be CUDA");
  TORCH_CHECK(source_input.scalar_type() == at::kBFloat16, "source_input must be BF16");
  TORCH_CHECK(source_input.dim() == 2, "source_input must be rank-2 [tokens, hidden]");
  TORCH_CHECK(source_input.is_contiguous(), "source_input must be contiguous");
  check_route_expert_indices(route_expert_indices);
  TORCH_CHECK(probs.is_cuda(), "probs must be CUDA");
  TORCH_CHECK(probs.scalar_type() == at::kFloat, "probs must be FP32");
  TORCH_CHECK(probs.dim() == 2, "probs must be rank-2 [tokens, top_k]");
  TORCH_CHECK(probs.is_contiguous(), "probs must be contiguous");
  TORCH_CHECK(up_gate_weight.is_cuda(), "up_gate_weight must be CUDA");
  TORCH_CHECK(up_gate_weight.scalar_type() == at::kBFloat16, "up_gate_weight must be BF16");
  TORCH_CHECK(up_gate_weight.dim() == 3,
              "up_gate_weight must be rank-3 [experts, 2*intermediate, hidden]");
  TORCH_CHECK(up_gate_weight.is_contiguous(), "up_gate_weight must be contiguous");
  TORCH_CHECK(down_weight.is_cuda(), "down_weight must be CUDA");
  TORCH_CHECK(down_weight.scalar_type() == at::kBFloat16, "down_weight must be BF16");
  TORCH_CHECK(down_weight.dim() == 3,
              "down_weight must be rank-3 [experts, intermediate, hidden]");
  TORCH_CHECK(down_weight.is_contiguous(), "down_weight must be contiguous");
  TORCH_CHECK(route_expert_indices.get_device() == source_input.get_device(),
              "route_expert_indices must be on the same device as source_input");
  TORCH_CHECK(probs.get_device() == source_input.get_device(),
              "probs must be on the same device as source_input");
  TORCH_CHECK(up_gate_weight.get_device() == source_input.get_device(),
              "up_gate_weight must be on the same device as source_input");
  TORCH_CHECK(down_weight.get_device() == source_input.get_device(),
              "down_weight must be on the same device as source_input");
  TORCH_CHECK(route_expert_indices.size(0) == source_input.size(0),
              "route_expert_indices token dimension mismatch");
  TORCH_CHECK(probs.size(0) == source_input.size(0), "probs token dimension mismatch");
  TORCH_CHECK(probs.size(1) == route_expert_indices.size(1), "probs top_k dimension mismatch");

  const int64_t num_local_experts = up_gate_weight.size(0);
  const int64_t hidden = source_input.size(1);
  const int64_t intermediate = down_weight.size(1);
  const int64_t out_features = up_gate_weight.size(1);
  const int64_t num_route_slots = route_expert_indices.numel();
  TORCH_CHECK(num_local_experts > 0, "num_local_experts must be > 0");
  TORCH_CHECK(down_weight.size(0) == num_local_experts,
              "expert count mismatch between weights");
  TORCH_CHECK(up_gate_weight.size(1) == 2 * intermediate,
              "up_gate_weight must have 2*intermediate rows");
  TORCH_CHECK(up_gate_weight.size(2) == hidden,
              "up_gate_weight hidden dimension mismatch");
  TORCH_CHECK(down_weight.size(2) == hidden,
              "down_weight hidden dimension mismatch");

  std::vector<at::Tensor> packed_inputs =
      rowwise_bf16_mega_moe_route_pack_inputs_debug(
          source_input,
          route_expert_indices,
          probs,
          num_local_experts);
  at::Tensor packed_route = packed_inputs[1];
  at::Tensor packed_input = packed_inputs[2];
  std::vector<at::Tensor> metadata =
      build_grouped_gemm_metadata(
          route_expert_indices,
          num_local_experts,
          olmo::bf16_mega_moe::kernels::kWmmaM);
  at::Tensor expert_counts = metadata[0];
  at::Tensor token_offsets = metadata[1];
  at::Tensor tile_offsets = metadata[3];

  auto tensor_options =
      at::TensorOptions().device(source_input.device()).dtype(at::kBFloat16);
  auto w1_out = at::zeros({num_route_slots, out_features}, tensor_options);
  auto h = at::zeros({num_route_slots, intermediate}, tensor_options);
  auto packed_expert_out = at::zeros({num_route_slots, hidden}, tensor_options);

  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t w1_n_tiles =
      at::ceil_div(out_features, olmo::bf16_mega_moe::kernels::kWmmaN);
  const int64_t w1_max_tasks = num_route_slots * w1_n_tiles;
  const int64_t w1_blocks = std::max<int64_t>(1, std::min<int64_t>(w1_max_tasks, 4096));
  olmo::bf16_mega_moe::kernels::grouped_w1_wmma_kernel<<<
      static_cast<unsigned int>(w1_blocks),
      32,
      0,
      stream.stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(packed_input.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(up_gate_weight.data_ptr<at::BFloat16>()),
      expert_counts.data_ptr<int64_t>(),
      token_offsets.data_ptr<int64_t>(),
      tile_offsets.data_ptr<int64_t>(),
      num_local_experts,
      num_route_slots,
      hidden,
      out_features,
      w1_n_tiles,
      w1_max_tasks,
      reinterpret_cast<__nv_bfloat16*>(w1_out.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  constexpr int64_t threads = 256;
  const int64_t swiglu_values = num_route_slots * intermediate;
  const int64_t swiglu_blocks =
      std::max<int64_t>(1, std::min<int64_t>(at::ceil_div(swiglu_values, threads), 4096));
  olmo::bf16_mega_moe::kernels::swiglu_forward_kernel<<<
      static_cast<unsigned int>(swiglu_blocks),
      static_cast<unsigned int>(threads),
      0,
      stream.stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(w1_out.data_ptr<at::BFloat16>()),
      num_route_slots,
      intermediate,
      reinterpret_cast<__nv_bfloat16*>(h.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t w2_n_tiles =
      at::ceil_div(hidden, olmo::bf16_mega_moe::kernels::kWmmaN);
  const int64_t w2_max_tasks = num_route_slots * w2_n_tiles;
  const int64_t w2_blocks = std::max<int64_t>(1, std::min<int64_t>(w2_max_tasks, 4096));
  olmo::bf16_mega_moe::kernels::grouped_w2_wmma_kernel<<<
      static_cast<unsigned int>(w2_blocks),
      32,
      0,
      stream.stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(h.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(down_weight.data_ptr<at::BFloat16>()),
      expert_counts.data_ptr<int64_t>(),
      token_offsets.data_ptr<int64_t>(),
      tile_offsets.data_ptr<int64_t>(),
      num_local_experts,
      num_route_slots,
      intermediate,
      hidden,
      w2_n_tiles,
      w2_max_tasks,
      reinterpret_cast<__nv_bfloat16*>(packed_expert_out.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  std::vector<at::Tensor> combined =
      rowwise_bf16_mega_moe_combine_debug(packed_expert_out, packed_route, probs);
  metadata.push_back(packed_route);
  metadata.push_back(packed_input);
  metadata.push_back(w1_out);
  metadata.push_back(h);
  metadata.push_back(packed_expert_out);
  metadata.push_back(combined[0]);
  metadata.push_back(combined[1]);
  return metadata;
}

std::vector<at::Tensor> rowwise_bf16_mega_moe_local_persistent_forward_debug(
    at::Tensor& source_input,
    at::Tensor& route_expert_indices,
    at::Tensor& probs,
    at::Tensor& up_gate_weight,
    at::Tensor& down_weight) {
  TORCH_CHECK(source_input.is_cuda(), "source_input must be CUDA");
  TORCH_CHECK(source_input.scalar_type() == at::kBFloat16, "source_input must be BF16");
  TORCH_CHECK(source_input.dim() == 2, "source_input must be rank-2 [tokens, hidden]");
  TORCH_CHECK(source_input.is_contiguous(), "source_input must be contiguous");
  check_route_expert_indices(route_expert_indices);
  TORCH_CHECK(probs.is_cuda(), "probs must be CUDA");
  TORCH_CHECK(probs.scalar_type() == at::kFloat, "probs must be FP32");
  TORCH_CHECK(probs.dim() == 2, "probs must be rank-2 [tokens, top_k]");
  TORCH_CHECK(probs.is_contiguous(), "probs must be contiguous");
  TORCH_CHECK(up_gate_weight.is_cuda(), "up_gate_weight must be CUDA");
  TORCH_CHECK(up_gate_weight.scalar_type() == at::kBFloat16, "up_gate_weight must be BF16");
  TORCH_CHECK(up_gate_weight.dim() == 3,
              "up_gate_weight must be rank-3 [experts, 2*intermediate, hidden]");
  TORCH_CHECK(up_gate_weight.is_contiguous(), "up_gate_weight must be contiguous");
  TORCH_CHECK(down_weight.is_cuda(), "down_weight must be CUDA");
  TORCH_CHECK(down_weight.scalar_type() == at::kBFloat16, "down_weight must be BF16");
  TORCH_CHECK(down_weight.dim() == 3,
              "down_weight must be rank-3 [experts, intermediate, hidden]");
  TORCH_CHECK(down_weight.is_contiguous(), "down_weight must be contiguous");
  TORCH_CHECK(route_expert_indices.get_device() == source_input.get_device(),
              "route_expert_indices must be on the same device as source_input");
  TORCH_CHECK(probs.get_device() == source_input.get_device(),
              "probs must be on the same device as source_input");
  TORCH_CHECK(up_gate_weight.get_device() == source_input.get_device(),
              "up_gate_weight must be on the same device as source_input");
  TORCH_CHECK(down_weight.get_device() == source_input.get_device(),
              "down_weight must be on the same device as source_input");
  TORCH_CHECK(route_expert_indices.size(0) == source_input.size(0),
              "route_expert_indices token dimension mismatch");
  TORCH_CHECK(probs.size(0) == source_input.size(0), "probs token dimension mismatch");
  TORCH_CHECK(probs.size(1) == route_expert_indices.size(1), "probs top_k dimension mismatch");

  const int64_t num_local_experts = up_gate_weight.size(0);
  const int64_t hidden = source_input.size(1);
  const int64_t intermediate = down_weight.size(1);
  const int64_t out_features = up_gate_weight.size(1);
  const int64_t num_route_slots = route_expert_indices.numel();
  TORCH_CHECK(num_local_experts > 0, "num_local_experts must be > 0");
  TORCH_CHECK(down_weight.size(0) == num_local_experts,
              "expert count mismatch between weights");
  TORCH_CHECK(up_gate_weight.size(1) == 2 * intermediate,
              "up_gate_weight must have 2*intermediate rows");
  TORCH_CHECK(up_gate_weight.size(2) == hidden,
              "up_gate_weight hidden dimension mismatch");
  TORCH_CHECK(down_weight.size(2) == hidden,
              "down_weight hidden dimension mismatch");
  TORCH_CHECK(hidden % olmo::bf16_mega_moe::kernels::kWmmaK == 0,
              "local persistent debug path requires hidden divisible by 16");
  TORCH_CHECK(intermediate % olmo::bf16_mega_moe::kernels::kWmmaK == 0,
              "local persistent debug path requires intermediate divisible by 16");

  std::vector<at::Tensor> packed_inputs =
      rowwise_bf16_mega_moe_route_pack_inputs_debug(
          source_input,
          route_expert_indices,
          probs,
          num_local_experts);
  at::Tensor packed_route = packed_inputs[1];
  at::Tensor packed_input = packed_inputs[2];
  std::vector<at::Tensor> metadata =
      build_grouped_gemm_metadata(
          route_expert_indices,
          num_local_experts,
          olmo::bf16_mega_moe::kernels::kWmmaM);
  at::Tensor expert_counts = metadata[0];
  at::Tensor token_offsets = metadata[1];

  auto tensor_options =
      at::TensorOptions().device(source_input.device()).dtype(at::kBFloat16);
  auto h = at::zeros({num_route_slots, intermediate}, tensor_options);
  auto packed_expert_out = at::zeros({num_route_slots, hidden}, tensor_options);
  auto barrier_state = at::zeros(
      {2},
      at::TensorOptions().device(source_input.device()).dtype(at::kInt));

  const int64_t num_sms = olmo::bf16_mega_moe::current_device_sm_count();
  const int64_t blocks = std::max<int64_t>(1, std::min<int64_t>(num_sms, 128));
  const int64_t num_experts_per_wave =
      std::max<int64_t>(1, std::min<int64_t>(num_local_experts, 4));
  auto stream = at::cuda::getCurrentCUDAStream();
  olmo::bf16_mega_moe::kernels::local_persistent_forward_debug_kernel<<<
      static_cast<unsigned int>(blocks),
      32,
      0,
      stream.stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(packed_input.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(up_gate_weight.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(down_weight.data_ptr<at::BFloat16>()),
      expert_counts.data_ptr<int64_t>(),
      token_offsets.data_ptr<int64_t>(),
      num_local_experts,
      num_experts_per_wave,
      num_route_slots,
      hidden,
      intermediate,
      reinterpret_cast<uint32_t*>(barrier_state.data_ptr<int>()),
      reinterpret_cast<__nv_bfloat16*>(h.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(packed_expert_out.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  std::vector<at::Tensor> combined =
      rowwise_bf16_mega_moe_combine_debug(packed_expert_out, packed_route, probs);
  metadata.push_back(packed_route);
  metadata.push_back(packed_input);
  metadata.push_back(h);
  metadata.push_back(packed_expert_out);
  metadata.push_back(combined[0]);
  metadata.push_back(combined[1]);
  metadata.push_back(barrier_state);
  (void)out_features;
  return metadata;
}

std::vector<at::Tensor> rowwise_bf16_mega_moe_local_full_forward_megakernel_debug(
    at::Tensor& source_input,
    at::Tensor& route_expert_indices,
    at::Tensor& probs,
    at::Tensor& up_gate_weight,
    at::Tensor& down_weight) {
  TORCH_CHECK(source_input.is_cuda(), "source_input must be CUDA");
  TORCH_CHECK(source_input.scalar_type() == at::kBFloat16, "source_input must be BF16");
  TORCH_CHECK(source_input.dim() == 2, "source_input must be rank-2 [tokens, hidden]");
  TORCH_CHECK(source_input.is_contiguous(), "source_input must be contiguous");
  check_route_expert_indices(route_expert_indices);
  TORCH_CHECK(route_expert_indices.scalar_type() == at::kLong,
              "local full forward megakernel debug currently requires int64 route indices");
  TORCH_CHECK(probs.is_cuda(), "probs must be CUDA");
  TORCH_CHECK(probs.scalar_type() == at::kFloat, "probs must be FP32");
  TORCH_CHECK(probs.dim() == 2, "probs must be rank-2 [tokens, top_k]");
  TORCH_CHECK(probs.is_contiguous(), "probs must be contiguous");
  TORCH_CHECK(up_gate_weight.is_cuda(), "up_gate_weight must be CUDA");
  TORCH_CHECK(up_gate_weight.scalar_type() == at::kBFloat16, "up_gate_weight must be BF16");
  TORCH_CHECK(up_gate_weight.dim() == 3,
              "up_gate_weight must be rank-3 [experts, 2*intermediate, hidden]");
  TORCH_CHECK(up_gate_weight.is_contiguous(), "up_gate_weight must be contiguous");
  TORCH_CHECK(down_weight.is_cuda(), "down_weight must be CUDA");
  TORCH_CHECK(down_weight.scalar_type() == at::kBFloat16, "down_weight must be BF16");
  TORCH_CHECK(down_weight.dim() == 3,
              "down_weight must be rank-3 [experts, intermediate, hidden]");
  TORCH_CHECK(down_weight.is_contiguous(), "down_weight must be contiguous");
  TORCH_CHECK(route_expert_indices.get_device() == source_input.get_device(),
              "route_expert_indices must be on the same device as source_input");
  TORCH_CHECK(probs.get_device() == source_input.get_device(),
              "probs must be on the same device as source_input");
  TORCH_CHECK(up_gate_weight.get_device() == source_input.get_device(),
              "up_gate_weight must be on the same device as source_input");
  TORCH_CHECK(down_weight.get_device() == source_input.get_device(),
              "down_weight must be on the same device as source_input");
  TORCH_CHECK(route_expert_indices.size(0) == source_input.size(0),
              "route_expert_indices token dimension mismatch");
  TORCH_CHECK(probs.size(0) == source_input.size(0), "probs token dimension mismatch");
  TORCH_CHECK(probs.size(1) == route_expert_indices.size(1), "probs top_k dimension mismatch");

  const int64_t num_tokens = source_input.size(0);
  const int64_t top_k = route_expert_indices.size(1);
  const int64_t num_local_experts = up_gate_weight.size(0);
  const int64_t hidden = source_input.size(1);
  const int64_t intermediate = down_weight.size(1);
  const int64_t num_route_slots = route_expert_indices.numel();
  TORCH_CHECK(num_local_experts > 0, "num_local_experts must be > 0");
  TORCH_CHECK(down_weight.size(0) == num_local_experts,
              "expert count mismatch between weights");
  TORCH_CHECK(up_gate_weight.size(1) == 2 * intermediate,
              "up_gate_weight must have 2*intermediate rows");
  TORCH_CHECK(up_gate_weight.size(2) == hidden,
              "up_gate_weight hidden dimension mismatch");
  TORCH_CHECK(down_weight.size(2) == hidden,
              "down_weight hidden dimension mismatch");
  TORCH_CHECK(hidden % olmo::bf16_mega_moe::kernels::kWmmaK == 0,
              "local full forward megakernel debug requires hidden divisible by 16");
  TORCH_CHECK(intermediate % olmo::bf16_mega_moe::kernels::kWmmaK == 0,
              "local full forward megakernel debug requires intermediate divisible by 16");

  auto long_options =
      at::TensorOptions().device(source_input.device()).dtype(at::kLong);
  auto bf16_options =
      at::TensorOptions().device(source_input.device()).dtype(at::kBFloat16);
  auto expert_counts = at::empty({num_local_experts}, long_options);
  auto token_offsets = at::empty({num_local_experts + 1}, long_options);
  auto expert_cursors = at::empty({num_local_experts}, long_options);
  auto packed_route = at::empty({num_route_slots}, long_options);
  auto route_to_slot = at::empty({num_route_slots}, long_options);
  auto packed_input = at::empty({num_route_slots, hidden}, bf16_options);
  auto h = at::empty({num_route_slots, intermediate}, bf16_options);
  auto packed_expert_out = at::empty({num_route_slots, hidden}, bf16_options);
  auto gathered_out = at::empty({num_tokens, top_k, hidden}, bf16_options);
  auto out = at::empty({num_tokens, hidden}, bf16_options);
  auto barrier_state = at::zeros(
      {2},
      at::TensorOptions().device(source_input.device()).dtype(at::kInt));

  const int64_t num_sms = olmo::bf16_mega_moe::current_device_sm_count();
  const int64_t blocks = std::max<int64_t>(1, std::min<int64_t>(num_sms, 128));
  const int64_t num_experts_per_wave =
      std::max<int64_t>(1, std::min<int64_t>(num_local_experts, 4));
  auto stream = at::cuda::getCurrentCUDAStream();
  olmo::bf16_mega_moe::kernels::local_full_forward_megakernel_debug_kernel<<<
      static_cast<unsigned int>(blocks),
      32,
      0,
      stream.stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(source_input.data_ptr<at::BFloat16>()),
      route_expert_indices.data_ptr<int64_t>(),
      probs.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(up_gate_weight.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(down_weight.data_ptr<at::BFloat16>()),
      num_tokens,
      top_k,
      num_local_experts,
      num_experts_per_wave,
      hidden,
      intermediate,
      num_route_slots,
      expert_counts.data_ptr<int64_t>(),
      token_offsets.data_ptr<int64_t>(),
      expert_cursors.data_ptr<int64_t>(),
      packed_route.data_ptr<int64_t>(),
      route_to_slot.data_ptr<int64_t>(),
      reinterpret_cast<uint32_t*>(barrier_state.data_ptr<int>()),
      reinterpret_cast<__nv_bfloat16*>(packed_input.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(h.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(packed_expert_out.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(gathered_out.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {
      expert_counts,
      token_offsets,
      packed_route,
      route_to_slot,
      packed_input,
      h,
      packed_expert_out,
      gathered_out,
      out,
      barrier_state,
  };
}

std::vector<at::Tensor> rowwise_bf16_mega_moe_standard_scheduler_debug(
    at::Tensor& expert_counts) {
  TORCH_CHECK(expert_counts.is_cuda(), "expert_counts must be CUDA");
  TORCH_CHECK(expert_counts.scalar_type() == at::kLong, "expert_counts must be int64");
  TORCH_CHECK(expert_counts.dim() == 1, "expert_counts must be rank-1");
  TORCH_CHECK(expert_counts.is_contiguous(), "expert_counts must be contiguous");
  TORCH_CHECK(
      expert_counts.numel() == olmo::bf16_mega_moe::kernels::kStandardNumLocalExperts,
      "expert_counts must have 8 local expert entries for the standard EP=4 shape");

  const olmo::bf16_mega_moe::layout::Workspace workspace_contract(
      nullptr,
      olmo::bf16_mega_moe::kernels::kStandardNumRanks,
      olmo::bf16_mega_moe::kernels::kStandardNumTotalExperts,
      olmo::bf16_mega_moe::kernels::kStandardNumMaxTokensPerRank,
      olmo::bf16_mega_moe::kernels::kStandardTopK);
  auto workspace = at::zeros(
      {static_cast<int64_t>(workspace_contract.num_bytes())},
      at::TensorOptions().device(expert_counts.device()).dtype(at::kByte));
  auto debug = at::zeros(
      {static_cast<int64_t>(olmo::bf16_mega_moe::kernels::kStandardNumLocalExperts) + 1, 4},
      at::TensorOptions().device(expert_counts.device()).dtype(at::kLong));

  auto stream = at::cuda::getCurrentCUDAStream();
  olmo::bf16_mega_moe::kernels::standard_scheduler_seed_counts_kernel<<<
      1,
      olmo::bf16_mega_moe::kernels::kStandardNumLocalExperts,
      0,
      stream.stream()>>>(
      workspace.data_ptr(),
      expert_counts.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  olmo::bf16_mega_moe::kernels::standard_scheduler_debug_kernel<<<
      olmo::bf16_mega_moe::kernels::kStandardSchedulerSms,
      32,
      0,
      stream.stream()>>>(
      workspace.data_ptr(),
      debug.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {workspace, debug};
}

std::vector<at::Tensor> rowwise_bf16_mega_moe_standard_ep_dispatch_metadata_debug(
    at::Tensor& route_expert_indices) {
  check_route_expert_indices(route_expert_indices);
  TORCH_CHECK(route_expert_indices.scalar_type() == at::kLong,
              "standard EP dispatch metadata debug currently requires int64 route indices");
  TORCH_CHECK(
      route_expert_indices.size(1) == olmo::bf16_mega_moe::kernels::kStandardTopK,
      "standard EP dispatch metadata debug requires top_k=4");
  TORCH_CHECK(
      route_expert_indices.size(0) <=
          olmo::bf16_mega_moe::kernels::kStandardNumMaxTokensPerRank,
      "standard EP dispatch metadata debug token count exceeds 16384");

  const int64_t num_tokens = route_expert_indices.size(0);
  const int64_t num_route_slots = route_expert_indices.numel();
  const uint64_t workspace_stride_bytes =
      olmo::bf16_mega_moe::kernels::standard_ep_workspace_stride_bytes(
          olmo::bf16_mega_moe::kernels::kStandardHidden);
  auto workspace = at::zeros(
      {static_cast<int64_t>(
          workspace_stride_bytes * olmo::bf16_mega_moe::kernels::kStandardNumRanks)},
      at::TensorOptions().device(route_expert_indices.device()).dtype(at::kByte));
  auto recv_counts = at::zeros(
      {olmo::bf16_mega_moe::kernels::kStandardNumRanks,
       olmo::bf16_mega_moe::kernels::kStandardNumLocalExperts},
      at::TensorOptions().device(route_expert_indices.device()).dtype(at::kLong));
  auto recv_ready_counts = at::zeros(
      {olmo::bf16_mega_moe::kernels::kStandardNumRanks,
       olmo::bf16_mega_moe::kernels::kStandardNumLocalExperts},
      at::TensorOptions().device(route_expert_indices.device()).dtype(at::kLong));
  auto src_token_topk_indices = at::empty(
      {olmo::bf16_mega_moe::kernels::kStandardNumRanks,
       olmo::bf16_mega_moe::kernels::kStandardNumLocalExperts,
       num_route_slots},
      at::TensorOptions().device(route_expert_indices.device()).dtype(at::kLong));
  auto barrier_state = at::zeros(
      {2},
      at::TensorOptions().device(route_expert_indices.device()).dtype(at::kInt));

  auto stream = at::cuda::getCurrentCUDAStream();
  olmo::bf16_mega_moe::kernels::standard_ep_dispatch_metadata_debug_kernel<<<
      olmo::bf16_mega_moe::kernels::kStandardSchedulerSms,
      olmo::bf16_mega_moe::kernels::kStandardDispatchWarps * 32,
      0,
      stream.stream()>>>(
      route_expert_indices.data_ptr<int64_t>(),
      static_cast<uint32_t>(num_tokens),
      workspace.data_ptr(),
      workspace_stride_bytes,
      reinterpret_cast<uint32_t*>(barrier_state.data_ptr<int>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  constexpr int64_t threads = 256;
  const int64_t total_values =
      static_cast<int64_t>(olmo::bf16_mega_moe::kernels::kStandardNumRanks) *
      olmo::bf16_mega_moe::kernels::kStandardNumLocalExperts *
      (1 + num_route_slots);
  const int64_t blocks =
      std::max<int64_t>(1, std::min<int64_t>(at::ceil_div(total_values, threads), 4096));
  olmo::bf16_mega_moe::kernels::standard_ep_dispatch_metadata_extract_kernel<<<
      static_cast<unsigned int>(blocks),
      static_cast<unsigned int>(threads),
      0,
      stream.stream()>>>(
      workspace.data_ptr(),
      workspace_stride_bytes,
      num_route_slots,
      recv_counts.data_ptr<int64_t>(),
      recv_ready_counts.data_ptr<int64_t>(),
      src_token_topk_indices.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {workspace, recv_counts, recv_ready_counts, src_token_topk_indices, barrier_state};
}

std::vector<at::Tensor> rowwise_bf16_mega_moe_standard_ep_dispatch_metadata_peer_map_debug(
    at::Tensor& route_expert_indices) {
  check_route_expert_indices(route_expert_indices);
  TORCH_CHECK(route_expert_indices.scalar_type() == at::kLong,
              "standard EP dispatch peer-map metadata debug currently requires int64 route indices");
  TORCH_CHECK(
      route_expert_indices.size(1) == olmo::bf16_mega_moe::kernels::kStandardTopK,
      "standard EP dispatch peer-map metadata debug requires top_k=4");
  TORCH_CHECK(
      route_expert_indices.size(0) <=
          olmo::bf16_mega_moe::kernels::kStandardNumMaxTokensPerRank,
      "standard EP dispatch peer-map metadata debug token count exceeds 16384");

  const int64_t num_tokens = route_expert_indices.size(0);
  const int64_t num_route_slots = route_expert_indices.numel();
  const uint64_t workspace_stride_bytes =
      olmo::bf16_mega_moe::kernels::standard_ep_workspace_stride_bytes(
          olmo::bf16_mega_moe::kernels::kStandardHidden);
  auto byte_options =
      at::TensorOptions().device(route_expert_indices.device()).dtype(at::kByte);
  auto long_options =
      at::TensorOptions().device(route_expert_indices.device()).dtype(at::kLong);
  auto int_options =
      at::TensorOptions().device(route_expert_indices.device()).dtype(at::kInt);
  auto workspace = at::zeros(
      {static_cast<int64_t>(
          workspace_stride_bytes * olmo::bf16_mega_moe::kernels::kStandardNumRanks)},
      byte_options);
  auto rank_workspace_bases = at::zeros(
      {olmo::bf16_mega_moe::kernels::kStandardNumRanks},
      long_options);
  auto recv_counts = at::zeros(
      {olmo::bf16_mega_moe::kernels::kStandardNumRanks,
       olmo::bf16_mega_moe::kernels::kStandardNumLocalExperts},
      long_options);
  auto recv_ready_counts = at::zeros(
      {olmo::bf16_mega_moe::kernels::kStandardNumRanks,
       olmo::bf16_mega_moe::kernels::kStandardNumLocalExperts},
      long_options);
  auto src_token_topk_indices = at::empty(
      {olmo::bf16_mega_moe::kernels::kStandardNumRanks,
       olmo::bf16_mega_moe::kernels::kStandardNumLocalExperts,
       num_route_slots},
      long_options);
  auto barrier_state = at::zeros({2}, int_options);

  auto stream = at::cuda::getCurrentCUDAStream();
  olmo::bf16_mega_moe::kernels::standard_ep_fill_workspace_base_ptrs_kernel<<<
      1,
      olmo::bf16_mega_moe::kernels::kStandardNumRanks,
      0,
      stream.stream()>>>(
      workspace.data_ptr(),
      workspace_stride_bytes,
      reinterpret_cast<uint64_t*>(rank_workspace_bases.data_ptr<int64_t>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  olmo::bf16_mega_moe::kernels::standard_ep_dispatch_metadata_peer_map_debug_kernel<<<
      olmo::bf16_mega_moe::kernels::kStandardSchedulerSms,
      olmo::bf16_mega_moe::kernels::kStandardDispatchWarps * 32,
      0,
      stream.stream()>>>(
      route_expert_indices.data_ptr<int64_t>(),
      static_cast<uint32_t>(num_tokens),
      reinterpret_cast<const uint64_t*>(rank_workspace_bases.data_ptr<int64_t>()),
      reinterpret_cast<uint32_t*>(barrier_state.data_ptr<int>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  constexpr int64_t threads = 256;
  const int64_t total_values =
      static_cast<int64_t>(olmo::bf16_mega_moe::kernels::kStandardNumRanks) *
      olmo::bf16_mega_moe::kernels::kStandardNumLocalExperts *
      (1 + num_route_slots);
  const int64_t blocks =
      std::max<int64_t>(1, std::min<int64_t>(at::ceil_div(total_values, threads), 4096));
  olmo::bf16_mega_moe::kernels::standard_ep_dispatch_metadata_extract_kernel<<<
      static_cast<unsigned int>(blocks),
      static_cast<unsigned int>(threads),
      0,
      stream.stream()>>>(
      workspace.data_ptr(),
      workspace_stride_bytes,
      num_route_slots,
      recv_counts.data_ptr<int64_t>(),
      recv_ready_counts.data_ptr<int64_t>(),
      src_token_topk_indices.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {
      workspace,
      rank_workspace_bases,
      recv_counts,
      recv_ready_counts,
      src_token_topk_indices,
      barrier_state,
  };
}

std::vector<at::Tensor> rowwise_bf16_mega_moe_standard_ep_dispatch_pack_inputs_debug(
    at::Tensor& source_input,
    at::Tensor& route_expert_indices) {
  TORCH_CHECK(source_input.is_cuda(), "source_input must be CUDA");
  TORCH_CHECK(source_input.scalar_type() == at::kBFloat16, "source_input must be BF16");
  TORCH_CHECK(source_input.dim() == 2, "source_input must be rank-2 [tokens, hidden]");
  TORCH_CHECK(source_input.is_contiguous(), "source_input must be contiguous");
  check_route_expert_indices(route_expert_indices);
  TORCH_CHECK(route_expert_indices.scalar_type() == at::kLong,
              "standard EP dispatch pack debug currently requires int64 route indices");
  TORCH_CHECK(route_expert_indices.get_device() == source_input.get_device(),
              "route_expert_indices must be on the same device as source_input");
  TORCH_CHECK(route_expert_indices.size(0) == source_input.size(0),
              "route_expert_indices token dimension mismatch");
  TORCH_CHECK(
      route_expert_indices.size(1) == olmo::bf16_mega_moe::kernels::kStandardTopK,
      "standard EP dispatch pack debug requires top_k=4");

  std::vector<at::Tensor> metadata =
      rowwise_bf16_mega_moe_standard_ep_dispatch_metadata_debug(route_expert_indices);
  at::Tensor workspace = metadata[0];
  const int64_t num_route_slots = route_expert_indices.numel();
  const int64_t hidden = source_input.size(1);
  const uint64_t workspace_stride_bytes =
      olmo::bf16_mega_moe::kernels::standard_ep_workspace_stride_bytes(
          olmo::bf16_mega_moe::kernels::kStandardHidden);
  auto long_options =
      at::TensorOptions().device(source_input.device()).dtype(at::kLong);
  auto bf16_options =
      at::TensorOptions().device(source_input.device()).dtype(at::kBFloat16);
  auto global_counts = at::zeros(
      {olmo::bf16_mega_moe::kernels::kStandardNumTotalExperts},
      long_options);
  auto global_offsets = at::zeros(
      {olmo::bf16_mega_moe::kernels::kStandardNumTotalExperts + 1},
      long_options);
  auto packed_route = at::full({num_route_slots}, -1, long_options);
  auto packed_input = at::zeros({num_route_slots, hidden}, bf16_options);

  auto stream = at::cuda::getCurrentCUDAStream();
  olmo::bf16_mega_moe::kernels::standard_ep_dispatch_global_counts_kernel<<<
      1,
      128,
      0,
      stream.stream()>>>(
      workspace.data_ptr(),
      workspace_stride_bytes,
      global_counts.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  olmo::bf16_mega_moe::kernels::standard_ep_dispatch_global_offsets_kernel<<<
      1,
      1,
      0,
      stream.stream()>>>(
      global_counts.data_ptr<int64_t>(),
      global_offsets.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  constexpr int64_t threads = 256;
  const int64_t route_blocks =
      std::max<int64_t>(1, std::min<int64_t>(at::ceil_div(num_route_slots, threads), 4096));
  olmo::bf16_mega_moe::kernels::standard_ep_dispatch_pack_inputs_from_workspace_kernel<<<
      static_cast<unsigned int>(route_blocks),
      static_cast<unsigned int>(threads),
      0,
      stream.stream()>>>(
      workspace.data_ptr(),
      workspace_stride_bytes,
      global_counts.data_ptr<int64_t>(),
      global_offsets.data_ptr<int64_t>(),
      packed_route.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t value_count = num_route_slots * hidden;
  const int64_t value_blocks =
      std::max<int64_t>(1, std::min<int64_t>(at::ceil_div(value_count, threads), 4096));
  olmo::bf16_mega_moe::kernels::standard_ep_dispatch_copy_packed_inputs_kernel<<<
      static_cast<unsigned int>(value_blocks),
      static_cast<unsigned int>(threads),
      0,
      stream.stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(source_input.data_ptr<at::BFloat16>()),
      packed_route.data_ptr<int64_t>(),
      num_route_slots,
      hidden,
      reinterpret_cast<__nv_bfloat16*>(packed_input.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {
      workspace,
      global_counts,
      global_offsets,
      packed_route,
      packed_input,
      metadata[1],
      metadata[2],
      metadata[3],
      metadata[4],
  };
}

std::vector<at::Tensor> rowwise_bf16_mega_moe_standard_ep_forward_from_dispatch_debug(
    at::Tensor& source_input,
    at::Tensor& route_expert_indices,
    at::Tensor& probs,
    at::Tensor& up_gate_weight,
    at::Tensor& down_weight) {
  TORCH_CHECK(source_input.is_cuda(), "source_input must be CUDA");
  TORCH_CHECK(source_input.scalar_type() == at::kBFloat16, "source_input must be BF16");
  TORCH_CHECK(source_input.dim() == 2, "source_input must be rank-2 [tokens, hidden]");
  TORCH_CHECK(source_input.is_contiguous(), "source_input must be contiguous");
  check_route_expert_indices(route_expert_indices);
  TORCH_CHECK(route_expert_indices.scalar_type() == at::kLong,
              "standard EP forward from dispatch debug currently requires int64 route indices");
  TORCH_CHECK(probs.is_cuda(), "probs must be CUDA");
  TORCH_CHECK(probs.scalar_type() == at::kFloat, "probs must be FP32");
  TORCH_CHECK(probs.dim() == 2, "probs must be rank-2 [tokens, top_k]");
  TORCH_CHECK(probs.is_contiguous(), "probs must be contiguous");
  TORCH_CHECK(up_gate_weight.is_cuda(), "up_gate_weight must be CUDA");
  TORCH_CHECK(up_gate_weight.scalar_type() == at::kBFloat16, "up_gate_weight must be BF16");
  TORCH_CHECK(up_gate_weight.dim() == 3,
              "up_gate_weight must be rank-3 [32 experts, 2*intermediate, hidden]");
  TORCH_CHECK(up_gate_weight.is_contiguous(), "up_gate_weight must be contiguous");
  TORCH_CHECK(down_weight.is_cuda(), "down_weight must be CUDA");
  TORCH_CHECK(down_weight.scalar_type() == at::kBFloat16, "down_weight must be BF16");
  TORCH_CHECK(down_weight.dim() == 3,
              "down_weight must be rank-3 [32 experts, intermediate, hidden]");
  TORCH_CHECK(down_weight.is_contiguous(), "down_weight must be contiguous");
  TORCH_CHECK(route_expert_indices.get_device() == source_input.get_device(),
              "route_expert_indices must be on the same device as source_input");
  TORCH_CHECK(probs.get_device() == source_input.get_device(),
              "probs must be on the same device as source_input");
  TORCH_CHECK(up_gate_weight.get_device() == source_input.get_device(),
              "up_gate_weight must be on the same device as source_input");
  TORCH_CHECK(down_weight.get_device() == source_input.get_device(),
              "down_weight must be on the same device as source_input");
  TORCH_CHECK(route_expert_indices.size(0) == source_input.size(0),
              "route_expert_indices token dimension mismatch");
  TORCH_CHECK(probs.sizes() == route_expert_indices.sizes(),
              "probs must match route_expert_indices shape");
  TORCH_CHECK(
      route_expert_indices.size(1) == olmo::bf16_mega_moe::kernels::kStandardTopK,
      "standard EP forward from dispatch debug requires top_k=4");
  TORCH_CHECK(
      up_gate_weight.size(0) == olmo::bf16_mega_moe::kernels::kStandardNumTotalExperts,
      "standard EP forward from dispatch debug requires 32 global experts");
  TORCH_CHECK(
      down_weight.size(0) == olmo::bf16_mega_moe::kernels::kStandardNumTotalExperts,
      "standard EP forward from dispatch debug requires 32 global experts");

  const int64_t num_route_slots = route_expert_indices.numel();
  const int64_t hidden = source_input.size(1);
  const int64_t intermediate = down_weight.size(1);
  TORCH_CHECK(up_gate_weight.size(1) == 2 * intermediate,
              "up_gate_weight must have 2*intermediate rows");
  TORCH_CHECK(up_gate_weight.size(2) == hidden,
              "up_gate_weight hidden dimension mismatch");
  TORCH_CHECK(down_weight.size(2) == hidden,
              "down_weight hidden dimension mismatch");
  TORCH_CHECK(hidden % olmo::bf16_mega_moe::kernels::kWmmaK == 0,
              "standard EP forward from dispatch debug requires hidden divisible by 16");
  TORCH_CHECK(intermediate % olmo::bf16_mega_moe::kernels::kWmmaK == 0,
              "standard EP forward from dispatch debug requires intermediate divisible by 16");

  std::vector<at::Tensor> packed =
      rowwise_bf16_mega_moe_standard_ep_dispatch_pack_inputs_debug(
          source_input,
          route_expert_indices);
  at::Tensor workspace = packed[0];
  at::Tensor global_counts = packed[1];
  at::Tensor global_offsets = packed[2];
  at::Tensor packed_route = packed[3];
  at::Tensor packed_input = packed[4];

  auto bf16_options =
      at::TensorOptions().device(source_input.device()).dtype(at::kBFloat16);
  auto h = at::zeros({num_route_slots, intermediate}, bf16_options);
  auto packed_expert_out = at::zeros({num_route_slots, hidden}, bf16_options);
  auto barrier_state = at::zeros(
      {2},
      at::TensorOptions().device(source_input.device()).dtype(at::kInt));

  const int64_t num_sms = olmo::bf16_mega_moe::current_device_sm_count();
  const int64_t blocks = std::max<int64_t>(1, std::min<int64_t>(num_sms, 128));
  auto stream = at::cuda::getCurrentCUDAStream();
  olmo::bf16_mega_moe::kernels::local_persistent_forward_debug_kernel<<<
      static_cast<unsigned int>(blocks),
      32,
      0,
      stream.stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(packed_input.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(up_gate_weight.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(down_weight.data_ptr<at::BFloat16>()),
      global_counts.data_ptr<int64_t>(),
      global_offsets.data_ptr<int64_t>(),
      olmo::bf16_mega_moe::kernels::kStandardNumTotalExperts,
      olmo::bf16_mega_moe::kernels::kStandardExpertsPerWave,
      num_route_slots,
      hidden,
      intermediate,
      reinterpret_cast<uint32_t*>(barrier_state.data_ptr<int>()),
      reinterpret_cast<__nv_bfloat16*>(h.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(packed_expert_out.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  std::vector<at::Tensor> combined =
      rowwise_bf16_mega_moe_combine_debug(packed_expert_out, packed_route, probs);
  return {
      workspace,
      global_counts,
      global_offsets,
      packed_route,
      packed_input,
      h,
      packed_expert_out,
      combined[0],
      combined[1],
      barrier_state,
      packed[5],
      packed[6],
      packed[7],
      packed[8],
  };
}

std::vector<at::Tensor> rowwise_bf16_mega_moe_standard_ep_forward_from_dispatch_umma_debug(
    at::Tensor& source_input,
    at::Tensor& route_expert_indices,
    at::Tensor& probs,
    at::Tensor& up_gate_weight,
    at::Tensor& down_weight) {
  TORCH_CHECK(source_input.is_cuda(), "source_input must be CUDA");
  TORCH_CHECK(source_input.scalar_type() == at::kBFloat16, "source_input must be BF16");
  TORCH_CHECK(source_input.dim() == 2, "source_input must be rank-2 [tokens, hidden]");
  TORCH_CHECK(source_input.is_contiguous(), "source_input must be contiguous");
  check_route_expert_indices(route_expert_indices);
  TORCH_CHECK(route_expert_indices.scalar_type() == at::kLong,
              "standard EP UMMA forward from dispatch debug currently requires int64 route indices");
  TORCH_CHECK(probs.is_cuda(), "probs must be CUDA");
  TORCH_CHECK(probs.scalar_type() == at::kFloat, "probs must be FP32");
  TORCH_CHECK(probs.dim() == 2, "probs must be rank-2 [tokens, top_k]");
  TORCH_CHECK(probs.is_contiguous(), "probs must be contiguous");
  TORCH_CHECK(up_gate_weight.is_cuda(), "up_gate_weight must be CUDA");
  TORCH_CHECK(up_gate_weight.scalar_type() == at::kBFloat16, "up_gate_weight must be BF16");
  TORCH_CHECK(up_gate_weight.dim() == 3,
              "up_gate_weight must be rank-3 [32 experts, 2*intermediate, hidden]");
  TORCH_CHECK(up_gate_weight.is_contiguous(), "up_gate_weight must be contiguous");
  TORCH_CHECK(down_weight.is_cuda(), "down_weight must be CUDA");
  TORCH_CHECK(down_weight.scalar_type() == at::kBFloat16, "down_weight must be BF16");
  TORCH_CHECK(down_weight.dim() == 3,
              "down_weight must be rank-3 [32 experts, intermediate, hidden]");
  TORCH_CHECK(down_weight.is_contiguous(), "down_weight must be contiguous");
  TORCH_CHECK(route_expert_indices.get_device() == source_input.get_device(),
              "route_expert_indices must be on the same device as source_input");
  TORCH_CHECK(probs.get_device() == source_input.get_device(),
              "probs must be on the same device as source_input");
  TORCH_CHECK(up_gate_weight.get_device() == source_input.get_device(),
              "up_gate_weight must be on the same device as source_input");
  TORCH_CHECK(down_weight.get_device() == source_input.get_device(),
              "down_weight must be on the same device as source_input");
  TORCH_CHECK(route_expert_indices.size(0) == source_input.size(0),
              "route_expert_indices token dimension mismatch");
  TORCH_CHECK(probs.sizes() == route_expert_indices.sizes(),
              "probs must match route_expert_indices shape");
  TORCH_CHECK(
      route_expert_indices.size(1) == olmo::bf16_mega_moe::kernels::kStandardTopK,
      "standard EP UMMA forward from dispatch debug requires top_k=4");
  TORCH_CHECK(
      up_gate_weight.size(0) == olmo::bf16_mega_moe::kernels::kStandardNumTotalExperts,
      "standard EP UMMA forward from dispatch debug requires 32 global experts");
  TORCH_CHECK(
      down_weight.size(0) == olmo::bf16_mega_moe::kernels::kStandardNumTotalExperts,
      "standard EP UMMA forward from dispatch debug requires 32 global experts");

  const int64_t num_route_slots = route_expert_indices.numel();
  const int64_t hidden = source_input.size(1);
  const int64_t intermediate = down_weight.size(1);
  TORCH_CHECK(up_gate_weight.size(1) == 2 * intermediate,
              "up_gate_weight must have 2*intermediate rows");
  TORCH_CHECK(up_gate_weight.size(2) == hidden,
              "up_gate_weight hidden dimension mismatch");
  TORCH_CHECK(down_weight.size(2) == hidden,
              "down_weight hidden dimension mismatch");
  TORCH_CHECK(hidden >= olmo::bf16_mega_moe::kernels::kSm100TileContractN,
              "standard EP UMMA forward requires hidden >= 128");
  TORCH_CHECK(intermediate >= olmo::bf16_mega_moe::kernels::kSm100TileContractN,
              "standard EP UMMA forward requires intermediate >= 128");
  TORCH_CHECK(hidden % olmo::bf16_mega_moe::kernels::kSm100TileContractN == 0,
              "standard EP UMMA forward requires hidden divisible by 128");
  TORCH_CHECK(intermediate % olmo::bf16_mega_moe::kernels::kSm100TileContractN == 0,
              "standard EP UMMA forward requires intermediate divisible by 128");

  std::vector<at::Tensor> packed =
      rowwise_bf16_mega_moe_standard_ep_dispatch_pack_inputs_debug(
          source_input,
          route_expert_indices);
  at::Tensor workspace = packed[0];
  at::Tensor global_counts = packed[1];
  at::Tensor global_offsets = packed[2];
  at::Tensor packed_route = packed[3];
  at::Tensor packed_input = packed[4];
  std::vector<at::Tensor> umma_metadata =
      build_grouped_gemm_metadata(
          route_expert_indices,
          olmo::bf16_mega_moe::kernels::kStandardNumTotalExperts,
          olmo::bf16_mega_moe::kernels::kSm100TileContractM);
  at::Tensor umma_expert_counts = umma_metadata[0];
  at::Tensor umma_token_offsets = umma_metadata[1];
  at::Tensor umma_tile_offsets = umma_metadata[3];

  auto bf16_options =
      at::TensorOptions().device(source_input.device()).dtype(at::kBFloat16);
  auto int_options =
      at::TensorOptions().device(source_input.device()).dtype(at::kInt);
  const int64_t padded_rows = num_route_slots +
      olmo::bf16_mega_moe::kernels::kStandardNumTotalExperts *
          (olmo::bf16_mega_moe::kernels::kSm100TileContractM - 1);
  auto packed_input_padded = at::zeros({padded_rows, hidden}, bf16_options);
  auto w1_up_padded = at::zeros({padded_rows, intermediate}, bf16_options);
  auto w1_gate_padded = at::zeros({padded_rows, intermediate}, bf16_options);
  auto h_padded = at::zeros({padded_rows, intermediate}, bf16_options);
  auto packed_expert_out_padded = at::zeros({padded_rows, hidden}, bf16_options);
  auto h = at::zeros({num_route_slots, intermediate}, bf16_options);
  auto packed_expert_out = at::zeros({num_route_slots, hidden}, bf16_options);
  auto barrier_state = at::zeros({2}, int_options);

  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int64_t copy_threads = 256;
  const int64_t packed_input_values = num_route_slots * hidden;
  const int64_t packed_input_blocks = std::max<int64_t>(
      1,
      std::min<int64_t>(at::ceil_div(packed_input_values, copy_threads), 4096));
  olmo::bf16_mega_moe::kernels::copy_bf16_rows_kernel<<<
      static_cast<unsigned int>(packed_input_blocks),
      static_cast<unsigned int>(copy_threads),
      0,
      stream.stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(packed_input.data_ptr<at::BFloat16>()),
      num_route_slots,
      hidden,
      reinterpret_cast<__nv_bfloat16*>(packed_input_padded.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  constexpr int64_t swizzle_bytes =
      olmo::bf16_mega_moe::kernels::kSm100TileContractK *
      static_cast<int64_t>(sizeof(__nv_bfloat16));
  const olmo::bf16_mega_moe::Bf16TensorMap2D packed_input_map =
      olmo::bf16_mega_moe::make_bf16_tma_2d_desc(
          packed_input_padded,
          olmo::bf16_mega_moe::kernels::kSm100TileContractK,
          olmo::bf16_mega_moe::kernels::kSm100TileContractM,
          swizzle_bytes);
  const olmo::bf16_mega_moe::Bf16TensorMap2D h_map =
      olmo::bf16_mega_moe::make_bf16_tma_2d_desc(
          h_padded,
          olmo::bf16_mega_moe::kernels::kSm100TileContractK,
          olmo::bf16_mega_moe::kernels::kSm100TileContractM,
          swizzle_bytes);
  const olmo::bf16_mega_moe::Bf16TensorMap2D up_gate_weight_map =
      olmo::bf16_mega_moe::make_bf16_tma_2d_desc(
          up_gate_weight.view(
              {olmo::bf16_mega_moe::kernels::kStandardNumTotalExperts * 2 * intermediate,
               hidden}),
          olmo::bf16_mega_moe::kernels::kSm100TileContractK,
          olmo::bf16_mega_moe::kernels::kSm100TileContractN,
          swizzle_bytes);
  const olmo::bf16_mega_moe::Bf16TensorMap2D down_weight_map =
      olmo::bf16_mega_moe::make_bf16_tma_2d_desc(
          down_weight.view(
              {olmo::bf16_mega_moe::kernels::kStandardNumTotalExperts * intermediate,
               hidden}),
          olmo::bf16_mega_moe::kernels::kSm100TileContractK,
          olmo::bf16_mega_moe::kernels::kSm100TileContractK,
          swizzle_bytes);

  auto make_tensor_map_storage = [&](const olmo::bf16_mega_moe::Bf16TensorMap2D& tensor_map) {
    auto storage = at::empty(
        {static_cast<int64_t>(sizeof(CUtensorMap))},
        at::TensorOptions().device(source_input.device()).dtype(at::kByte));
    C10_CUDA_CHECK(cudaMemcpyAsync(
        storage.data_ptr(),
        &tensor_map.map,
        sizeof(CUtensorMap),
        cudaMemcpyHostToDevice,
        stream.stream()));
    return storage;
  };
  auto packed_input_map_storage = make_tensor_map_storage(packed_input_map);
  auto h_map_storage = make_tensor_map_storage(h_map);
  auto up_gate_weight_map_storage = make_tensor_map_storage(up_gate_weight_map);
  auto down_weight_map_storage = make_tensor_map_storage(down_weight_map);

  const int64_t max_m_tiles_per_expert = at::ceil_div(
      num_route_slots,
      static_cast<int64_t>(olmo::bf16_mega_moe::kernels::kSm100TileContractM));
  const int64_t w1_n_tiles = at::ceil_div(
      intermediate,
      static_cast<int64_t>(olmo::bf16_mega_moe::kernels::kSm100TileContractN));
  const int64_t w2_n_tiles = at::ceil_div(
      hidden,
      static_cast<int64_t>(olmo::bf16_mega_moe::kernels::kSm100TileContractN));
  const int64_t w1_tasks =
      olmo::bf16_mega_moe::kernels::kStandardNumTotalExperts *
      max_m_tiles_per_expert *
      w1_n_tiles;
  const int64_t w2_tasks =
      olmo::bf16_mega_moe::kernels::kStandardNumTotalExperts *
      max_m_tiles_per_expert *
      w2_n_tiles;
  const int64_t w1_blocks = std::max<int64_t>(
      1,
      std::min<int64_t>(w1_tasks, 4096));
  const int64_t w2_blocks = std::max<int64_t>(
      1,
      std::min<int64_t>(w2_tasks, 4096));

  olmo::bf16_mega_moe::kernels::grouped_w1_umma_linear_kernel<<<
      static_cast<unsigned int>(w1_blocks),
      olmo::bf16_mega_moe::kernels::kSm100TileContractM,
      0,
      stream.stream()>>>(
      reinterpret_cast<const CUtensorMap*>(packed_input_map_storage.data_ptr()),
      reinterpret_cast<const CUtensorMap*>(up_gate_weight_map_storage.data_ptr()),
      umma_expert_counts.data_ptr<int64_t>(),
      umma_token_offsets.data_ptr<int64_t>(),
      umma_tile_offsets.data_ptr<int64_t>(),
      olmo::bf16_mega_moe::kernels::kStandardNumTotalExperts,
      w1_tasks,
      hidden,
      intermediate,
      /*gate_offset=*/0,
      reinterpret_cast<__nv_bfloat16*>(w1_up_padded.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  olmo::bf16_mega_moe::kernels::grouped_w1_umma_linear_kernel<<<
      static_cast<unsigned int>(w1_blocks),
      olmo::bf16_mega_moe::kernels::kSm100TileContractM,
      0,
      stream.stream()>>>(
      reinterpret_cast<const CUtensorMap*>(packed_input_map_storage.data_ptr()),
      reinterpret_cast<const CUtensorMap*>(up_gate_weight_map_storage.data_ptr()),
      umma_expert_counts.data_ptr<int64_t>(),
      umma_token_offsets.data_ptr<int64_t>(),
      umma_tile_offsets.data_ptr<int64_t>(),
      olmo::bf16_mega_moe::kernels::kStandardNumTotalExperts,
      w1_tasks,
      hidden,
      intermediate,
      /*gate_offset=*/intermediate,
      reinterpret_cast<__nv_bfloat16*>(w1_gate_padded.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t h_values = num_route_slots * intermediate;
  const int64_t h_blocks = std::max<int64_t>(
      1,
      std::min<int64_t>(at::ceil_div(h_values, copy_threads), 4096));
  olmo::bf16_mega_moe::kernels::swiglu_forward_split_kernel<<<
      static_cast<unsigned int>(h_blocks),
      static_cast<unsigned int>(copy_threads),
      0,
      stream.stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(w1_up_padded.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(w1_gate_padded.data_ptr<at::BFloat16>()),
      num_route_slots,
      intermediate,
      reinterpret_cast<__nv_bfloat16*>(h_padded.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  olmo::bf16_mega_moe::kernels::copy_bf16_rows_kernel<<<
      static_cast<unsigned int>(h_blocks),
      static_cast<unsigned int>(copy_threads),
      0,
      stream.stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(h_padded.data_ptr<at::BFloat16>()),
      num_route_slots,
      intermediate,
      reinterpret_cast<__nv_bfloat16*>(h.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  olmo::bf16_mega_moe::kernels::grouped_w2_umma_linear_kernel<<<
      static_cast<unsigned int>(w2_blocks),
      olmo::bf16_mega_moe::kernels::kSm100TileContractM,
      0,
      stream.stream()>>>(
      reinterpret_cast<const CUtensorMap*>(h_map_storage.data_ptr()),
      reinterpret_cast<const CUtensorMap*>(down_weight_map_storage.data_ptr()),
      umma_expert_counts.data_ptr<int64_t>(),
      umma_token_offsets.data_ptr<int64_t>(),
      umma_tile_offsets.data_ptr<int64_t>(),
      olmo::bf16_mega_moe::kernels::kStandardNumTotalExperts,
      w2_tasks,
      intermediate,
      hidden,
      reinterpret_cast<__nv_bfloat16*>(packed_expert_out_padded.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t packed_out_values = num_route_slots * hidden;
  const int64_t packed_out_blocks = std::max<int64_t>(
      1,
      std::min<int64_t>(at::ceil_div(packed_out_values, copy_threads), 4096));
  olmo::bf16_mega_moe::kernels::copy_bf16_rows_kernel<<<
      static_cast<unsigned int>(packed_out_blocks),
      static_cast<unsigned int>(copy_threads),
      0,
      stream.stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(packed_expert_out_padded.data_ptr<at::BFloat16>()),
      num_route_slots,
      hidden,
      reinterpret_cast<__nv_bfloat16*>(packed_expert_out.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  std::vector<at::Tensor> combined =
      rowwise_bf16_mega_moe_combine_debug(packed_expert_out, packed_route, probs);
  return {
      workspace,
      global_counts,
      global_offsets,
      packed_route,
      packed_input,
      h,
      packed_expert_out,
      combined[0],
      combined[1],
      barrier_state,
      packed[5],
      packed[6],
      packed[7],
      packed[8],
      packed_input_padded,
      w1_up_padded,
      w1_gate_padded,
      h_padded,
      packed_expert_out_padded,
  };
}

std::vector<at::Tensor> rowwise_bf16_mega_moe_local_umma_compute_debug(
    at::Tensor& packed_input,
    at::Tensor& expert_counts,
    at::Tensor& up_gate_weight,
    at::Tensor& down_weight) {
  TORCH_CHECK(packed_input.is_cuda(), "packed_input must be CUDA");
  TORCH_CHECK(packed_input.scalar_type() == at::kBFloat16, "packed_input must be BF16");
  TORCH_CHECK(packed_input.dim() == 2, "packed_input must be rank-2 [rows, hidden]");
  TORCH_CHECK(packed_input.is_contiguous(), "packed_input must be contiguous");
  TORCH_CHECK(expert_counts.is_cuda(), "expert_counts must be CUDA");
  TORCH_CHECK(expert_counts.scalar_type() == at::kLong, "expert_counts must be int64");
  TORCH_CHECK(expert_counts.dim() == 1, "expert_counts must be rank-1 [num_local_experts]");
  TORCH_CHECK(expert_counts.is_contiguous(), "expert_counts must be contiguous");
  TORCH_CHECK(up_gate_weight.is_cuda(), "up_gate_weight must be CUDA");
  TORCH_CHECK(up_gate_weight.scalar_type() == at::kBFloat16, "up_gate_weight must be BF16");
  TORCH_CHECK(up_gate_weight.dim() == 3,
              "up_gate_weight must be rank-3 [local_experts, 2*intermediate, hidden]");
  TORCH_CHECK(up_gate_weight.is_contiguous(), "up_gate_weight must be contiguous");
  TORCH_CHECK(down_weight.is_cuda(), "down_weight must be CUDA");
  TORCH_CHECK(down_weight.scalar_type() == at::kBFloat16, "down_weight must be BF16");
  TORCH_CHECK(down_weight.dim() == 3,
              "down_weight must be rank-3 [local_experts, intermediate, hidden]");
  TORCH_CHECK(down_weight.is_contiguous(), "down_weight must be contiguous");
  TORCH_CHECK(expert_counts.get_device() == packed_input.get_device(),
              "expert_counts must be on the same device as packed_input");
  TORCH_CHECK(up_gate_weight.get_device() == packed_input.get_device(),
              "up_gate_weight must be on the same device as packed_input");
  TORCH_CHECK(down_weight.get_device() == packed_input.get_device(),
              "down_weight must be on the same device as packed_input");

  const int64_t num_rows = packed_input.size(0);
  const int64_t hidden = packed_input.size(1);
  const int64_t num_local_experts = expert_counts.numel();
  const int64_t intermediate = down_weight.size(1);
  TORCH_CHECK(num_local_experts > 0, "num_local_experts must be > 0");
  TORCH_CHECK(up_gate_weight.size(0) == num_local_experts,
              "up_gate_weight expert dimension must match expert_counts");
  TORCH_CHECK(down_weight.size(0) == num_local_experts,
              "down_weight expert dimension must match expert_counts");
  TORCH_CHECK(up_gate_weight.size(1) == 2 * intermediate,
              "up_gate_weight must have 2*intermediate rows");
  TORCH_CHECK(up_gate_weight.size(2) == hidden,
              "up_gate_weight hidden dimension mismatch");
  TORCH_CHECK(down_weight.size(2) == hidden,
              "down_weight hidden dimension mismatch");
  TORCH_CHECK(hidden >= olmo::bf16_mega_moe::kernels::kSm100TileContractN,
              "local UMMA compute requires hidden >= 128");
  TORCH_CHECK(intermediate >= olmo::bf16_mega_moe::kernels::kSm100TileContractN,
              "local UMMA compute requires intermediate >= 128");
  TORCH_CHECK(hidden % olmo::bf16_mega_moe::kernels::kSm100TileContractN == 0,
              "local UMMA compute requires hidden divisible by 128");
  TORCH_CHECK(intermediate % olmo::bf16_mega_moe::kernels::kSm100TileContractN == 0,
              "local UMMA compute requires intermediate divisible by 128");

  std::vector<at::Tensor> metadata =
      build_grouped_gemm_metadata_from_counts(
          expert_counts,
          olmo::bf16_mega_moe::kernels::kSm100TileContractM);
  at::Tensor token_offsets = metadata[1];
  at::Tensor tile_offsets = metadata[3];

  auto bf16_options =
      at::TensorOptions().device(packed_input.device()).dtype(at::kBFloat16);
  const int64_t padded_rows = num_rows +
      num_local_experts *
          (olmo::bf16_mega_moe::kernels::kSm100TileContractM - 1);
  auto packed_input_padded = at::zeros({padded_rows, hidden}, bf16_options);
  auto w1_up_padded = at::zeros({padded_rows, intermediate}, bf16_options);
  auto w1_gate_padded = at::zeros({padded_rows, intermediate}, bf16_options);
  auto h_padded = at::zeros({padded_rows, intermediate}, bf16_options);
  auto packed_expert_out_padded = at::zeros({padded_rows, hidden}, bf16_options);
  auto h = at::zeros({num_rows, intermediate}, bf16_options);
  auto packed_expert_out = at::zeros({num_rows, hidden}, bf16_options);

  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int64_t copy_threads = 256;
  const int64_t packed_input_values = num_rows * hidden;
  const int64_t packed_input_blocks = std::max<int64_t>(
      1,
      std::min<int64_t>(at::ceil_div(packed_input_values, copy_threads), 4096));
  olmo::bf16_mega_moe::kernels::copy_bf16_rows_kernel<<<
      static_cast<unsigned int>(packed_input_blocks),
      static_cast<unsigned int>(copy_threads),
      0,
      stream.stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(packed_input.data_ptr<at::BFloat16>()),
      num_rows,
      hidden,
      reinterpret_cast<__nv_bfloat16*>(packed_input_padded.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  constexpr int64_t swizzle_bytes =
      olmo::bf16_mega_moe::kernels::kSm100TileContractK *
      static_cast<int64_t>(sizeof(__nv_bfloat16));
  const olmo::bf16_mega_moe::Bf16TensorMap2D packed_input_map =
      olmo::bf16_mega_moe::make_bf16_tma_2d_desc(
          packed_input_padded,
          olmo::bf16_mega_moe::kernels::kSm100TileContractK,
          olmo::bf16_mega_moe::kernels::kSm100TileContractM,
          swizzle_bytes);
  const olmo::bf16_mega_moe::Bf16TensorMap2D h_map =
      olmo::bf16_mega_moe::make_bf16_tma_2d_desc(
          h_padded,
          olmo::bf16_mega_moe::kernels::kSm100TileContractK,
          olmo::bf16_mega_moe::kernels::kSm100TileContractM,
          swizzle_bytes);
  const olmo::bf16_mega_moe::Bf16TensorMap2D up_gate_weight_map =
      olmo::bf16_mega_moe::make_bf16_tma_2d_desc(
          up_gate_weight.view({num_local_experts * 2 * intermediate, hidden}),
          olmo::bf16_mega_moe::kernels::kSm100TileContractK,
          olmo::bf16_mega_moe::kernels::kSm100TileContractN,
          swizzle_bytes);
  const olmo::bf16_mega_moe::Bf16TensorMap2D down_weight_map =
      olmo::bf16_mega_moe::make_bf16_tma_2d_desc(
          down_weight.view({num_local_experts * intermediate, hidden}),
          olmo::bf16_mega_moe::kernels::kSm100TileContractK,
          olmo::bf16_mega_moe::kernels::kSm100TileContractK,
          swizzle_bytes);

  auto make_tensor_map_storage = [&](const olmo::bf16_mega_moe::Bf16TensorMap2D& tensor_map) {
    auto storage = at::empty(
        {static_cast<int64_t>(sizeof(CUtensorMap))},
        at::TensorOptions().device(packed_input.device()).dtype(at::kByte));
    C10_CUDA_CHECK(cudaMemcpyAsync(
        storage.data_ptr(),
        &tensor_map.map,
        sizeof(CUtensorMap),
        cudaMemcpyHostToDevice,
        stream.stream()));
    return storage;
  };
  auto packed_input_map_storage = make_tensor_map_storage(packed_input_map);
  auto h_map_storage = make_tensor_map_storage(h_map);
  auto up_gate_weight_map_storage = make_tensor_map_storage(up_gate_weight_map);
  auto down_weight_map_storage = make_tensor_map_storage(down_weight_map);

  const int64_t max_m_tiles_per_expert = at::ceil_div(
      num_rows,
      static_cast<int64_t>(olmo::bf16_mega_moe::kernels::kSm100TileContractM));
  const int64_t w1_n_tiles = at::ceil_div(
      intermediate,
      static_cast<int64_t>(olmo::bf16_mega_moe::kernels::kSm100TileContractN));
  const int64_t w2_n_tiles = at::ceil_div(
      hidden,
      static_cast<int64_t>(olmo::bf16_mega_moe::kernels::kSm100TileContractN));
  const int64_t w1_tasks = num_local_experts * max_m_tiles_per_expert * w1_n_tiles;
  const int64_t w2_tasks = num_local_experts * max_m_tiles_per_expert * w2_n_tiles;
  const int64_t w1_blocks = std::max<int64_t>(
      1,
      std::min<int64_t>(w1_tasks, 4096));
  const int64_t w2_blocks = std::max<int64_t>(
      1,
      std::min<int64_t>(w2_tasks, 4096));

  olmo::bf16_mega_moe::kernels::grouped_w1_umma_linear_kernel<<<
      static_cast<unsigned int>(w1_blocks),
      olmo::bf16_mega_moe::kernels::kSm100TileContractM,
      0,
      stream.stream()>>>(
      reinterpret_cast<const CUtensorMap*>(packed_input_map_storage.data_ptr()),
      reinterpret_cast<const CUtensorMap*>(up_gate_weight_map_storage.data_ptr()),
      expert_counts.data_ptr<int64_t>(),
      token_offsets.data_ptr<int64_t>(),
      tile_offsets.data_ptr<int64_t>(),
      num_local_experts,
      w1_tasks,
      hidden,
      intermediate,
      /*gate_offset=*/0,
      reinterpret_cast<__nv_bfloat16*>(w1_up_padded.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  olmo::bf16_mega_moe::kernels::grouped_w1_umma_linear_kernel<<<
      static_cast<unsigned int>(w1_blocks),
      olmo::bf16_mega_moe::kernels::kSm100TileContractM,
      0,
      stream.stream()>>>(
      reinterpret_cast<const CUtensorMap*>(packed_input_map_storage.data_ptr()),
      reinterpret_cast<const CUtensorMap*>(up_gate_weight_map_storage.data_ptr()),
      expert_counts.data_ptr<int64_t>(),
      token_offsets.data_ptr<int64_t>(),
      tile_offsets.data_ptr<int64_t>(),
      num_local_experts,
      w1_tasks,
      hidden,
      intermediate,
      /*gate_offset=*/intermediate,
      reinterpret_cast<__nv_bfloat16*>(w1_gate_padded.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t h_values = num_rows * intermediate;
  const int64_t h_blocks = std::max<int64_t>(
      1,
      std::min<int64_t>(at::ceil_div(h_values, copy_threads), 4096));
  olmo::bf16_mega_moe::kernels::swiglu_forward_split_kernel<<<
      static_cast<unsigned int>(h_blocks),
      static_cast<unsigned int>(copy_threads),
      0,
      stream.stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(w1_up_padded.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(w1_gate_padded.data_ptr<at::BFloat16>()),
      num_rows,
      intermediate,
      reinterpret_cast<__nv_bfloat16*>(h_padded.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  olmo::bf16_mega_moe::kernels::copy_bf16_rows_kernel<<<
      static_cast<unsigned int>(h_blocks),
      static_cast<unsigned int>(copy_threads),
      0,
      stream.stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(h_padded.data_ptr<at::BFloat16>()),
      num_rows,
      intermediate,
      reinterpret_cast<__nv_bfloat16*>(h.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  olmo::bf16_mega_moe::kernels::grouped_w2_umma_linear_kernel<<<
      static_cast<unsigned int>(w2_blocks),
      olmo::bf16_mega_moe::kernels::kSm100TileContractM,
      0,
      stream.stream()>>>(
      reinterpret_cast<const CUtensorMap*>(h_map_storage.data_ptr()),
      reinterpret_cast<const CUtensorMap*>(down_weight_map_storage.data_ptr()),
      expert_counts.data_ptr<int64_t>(),
      token_offsets.data_ptr<int64_t>(),
      tile_offsets.data_ptr<int64_t>(),
      num_local_experts,
      w2_tasks,
      intermediate,
      hidden,
      reinterpret_cast<__nv_bfloat16*>(packed_expert_out_padded.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int64_t packed_out_values = num_rows * hidden;
  const int64_t packed_out_blocks = std::max<int64_t>(
      1,
      std::min<int64_t>(at::ceil_div(packed_out_values, copy_threads), 4096));
  olmo::bf16_mega_moe::kernels::copy_bf16_rows_kernel<<<
      static_cast<unsigned int>(packed_out_blocks),
      static_cast<unsigned int>(copy_threads),
      0,
      stream.stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(packed_expert_out_padded.data_ptr<at::BFloat16>()),
      num_rows,
      hidden,
      reinterpret_cast<__nv_bfloat16*>(packed_expert_out.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  metadata.push_back(h);
  metadata.push_back(packed_expert_out);
  metadata.push_back(packed_input_padded);
  metadata.push_back(w1_up_padded);
  metadata.push_back(w1_gate_padded);
  metadata.push_back(h_padded);
  metadata.push_back(packed_expert_out_padded);
  return metadata;
}

at::Tensor rowwise_bf16_mega_moe_local_umma_compute(
    at::Tensor& packed_input,
    at::Tensor& expert_counts,
    at::Tensor& up_gate_weight,
    at::Tensor& down_weight) {
  std::vector<at::Tensor> debug_outputs =
      rowwise_bf16_mega_moe_local_umma_compute_debug(
          packed_input,
          expert_counts,
          up_gate_weight,
          down_weight);
  return debug_outputs[6];
}

std::vector<at::Tensor> rowwise_bf16_mega_moe_standard_ep_full_forward_megakernel_debug(
    at::Tensor& source_input,
    at::Tensor& route_expert_indices,
    at::Tensor& probs,
    at::Tensor& up_gate_weight,
    at::Tensor& down_weight) {
  TORCH_CHECK(source_input.is_cuda(), "source_input must be CUDA");
  TORCH_CHECK(source_input.scalar_type() == at::kBFloat16, "source_input must be BF16");
  TORCH_CHECK(source_input.dim() == 2, "source_input must be rank-2 [tokens, hidden]");
  TORCH_CHECK(source_input.is_contiguous(), "source_input must be contiguous");
  check_route_expert_indices(route_expert_indices);
  TORCH_CHECK(route_expert_indices.scalar_type() == at::kLong,
              "standard EP full forward megakernel debug currently requires int64 route indices");
  TORCH_CHECK(probs.is_cuda(), "probs must be CUDA");
  TORCH_CHECK(probs.scalar_type() == at::kFloat, "probs must be FP32");
  TORCH_CHECK(probs.dim() == 2, "probs must be rank-2 [tokens, top_k]");
  TORCH_CHECK(probs.is_contiguous(), "probs must be contiguous");
  TORCH_CHECK(up_gate_weight.is_cuda(), "up_gate_weight must be CUDA");
  TORCH_CHECK(up_gate_weight.scalar_type() == at::kBFloat16, "up_gate_weight must be BF16");
  TORCH_CHECK(up_gate_weight.dim() == 3,
              "up_gate_weight must be rank-3 [32 experts, 2*intermediate, hidden]");
  TORCH_CHECK(up_gate_weight.is_contiguous(), "up_gate_weight must be contiguous");
  TORCH_CHECK(down_weight.is_cuda(), "down_weight must be CUDA");
  TORCH_CHECK(down_weight.scalar_type() == at::kBFloat16, "down_weight must be BF16");
  TORCH_CHECK(down_weight.dim() == 3,
              "down_weight must be rank-3 [32 experts, intermediate, hidden]");
  TORCH_CHECK(down_weight.is_contiguous(), "down_weight must be contiguous");
  TORCH_CHECK(route_expert_indices.get_device() == source_input.get_device(),
              "route_expert_indices must be on the same device as source_input");
  TORCH_CHECK(probs.get_device() == source_input.get_device(),
              "probs must be on the same device as source_input");
  TORCH_CHECK(up_gate_weight.get_device() == source_input.get_device(),
              "up_gate_weight must be on the same device as source_input");
  TORCH_CHECK(down_weight.get_device() == source_input.get_device(),
              "down_weight must be on the same device as source_input");
  TORCH_CHECK(route_expert_indices.size(0) == source_input.size(0),
              "route_expert_indices token dimension mismatch");
  TORCH_CHECK(probs.sizes() == route_expert_indices.sizes(),
              "probs must match route_expert_indices shape");
  TORCH_CHECK(
      route_expert_indices.size(1) == olmo::bf16_mega_moe::kernels::kStandardTopK,
      "standard EP full forward megakernel debug requires top_k=4");
  TORCH_CHECK(
      route_expert_indices.size(0) <=
          olmo::bf16_mega_moe::kernels::kStandardNumMaxTokensPerRank,
      "standard EP full forward megakernel debug token count exceeds 16384");
  TORCH_CHECK(
      up_gate_weight.size(0) == olmo::bf16_mega_moe::kernels::kStandardNumTotalExperts,
      "standard EP full forward megakernel debug requires 32 global experts");
  TORCH_CHECK(
      down_weight.size(0) == olmo::bf16_mega_moe::kernels::kStandardNumTotalExperts,
      "standard EP full forward megakernel debug requires 32 global experts");

  const int64_t num_tokens = source_input.size(0);
  const int64_t num_route_slots = route_expert_indices.numel();
  const int64_t hidden = source_input.size(1);
  const int64_t intermediate = down_weight.size(1);
  TORCH_CHECK(up_gate_weight.size(1) == 2 * intermediate,
              "up_gate_weight must have 2*intermediate rows");
  TORCH_CHECK(up_gate_weight.size(2) == hidden,
              "up_gate_weight hidden dimension mismatch");
  TORCH_CHECK(down_weight.size(2) == hidden,
              "down_weight hidden dimension mismatch");
  TORCH_CHECK(hidden % olmo::bf16_mega_moe::kernels::kWmmaK == 0,
              "standard EP full forward megakernel debug requires hidden divisible by 16");
  TORCH_CHECK(intermediate % olmo::bf16_mega_moe::kernels::kWmmaK == 0,
              "standard EP full forward megakernel debug requires intermediate divisible by 16");

  const uint64_t workspace_stride_bytes =
      olmo::bf16_mega_moe::kernels::standard_ep_workspace_stride_bytes(hidden);
  auto byte_options =
      at::TensorOptions().device(source_input.device()).dtype(at::kByte);
  auto long_options =
      at::TensorOptions().device(source_input.device()).dtype(at::kLong);
  auto bf16_options =
      at::TensorOptions().device(source_input.device()).dtype(at::kBFloat16);
  auto int_options =
      at::TensorOptions().device(source_input.device()).dtype(at::kInt);

  auto workspace = at::zeros(
      {static_cast<int64_t>(
          workspace_stride_bytes * olmo::bf16_mega_moe::kernels::kStandardNumRanks)},
      byte_options);
  auto rank_workspace_bases = at::zeros(
      {olmo::bf16_mega_moe::kernels::kStandardNumRanks},
      long_options);
  auto global_counts = at::zeros(
      {olmo::bf16_mega_moe::kernels::kStandardNumTotalExperts},
      long_options);
  auto global_offsets = at::zeros(
      {olmo::bf16_mega_moe::kernels::kStandardNumTotalExperts + 1},
      long_options);
  auto expert_cursors = at::zeros(
      {olmo::bf16_mega_moe::kernels::kStandardNumTotalExperts},
      long_options);
  auto packed_route = at::full({num_route_slots}, -1, long_options);
  auto route_to_slot = at::full({num_route_slots}, -1, long_options);
  auto packed_input = at::zeros({num_route_slots, hidden}, bf16_options);
  auto h = at::zeros({num_route_slots, intermediate}, bf16_options);
  auto packed_expert_out = at::zeros({num_route_slots, hidden}, bf16_options);
  auto gathered_out = at::zeros(
      {num_tokens, olmo::bf16_mega_moe::kernels::kStandardTopK, hidden},
      bf16_options);
  auto out = at::zeros({num_tokens, hidden}, bf16_options);
  auto recv_counts = at::zeros(
      {olmo::bf16_mega_moe::kernels::kStandardNumRanks,
       olmo::bf16_mega_moe::kernels::kStandardNumLocalExperts},
      long_options);
  auto recv_ready_counts = at::zeros(
      {olmo::bf16_mega_moe::kernels::kStandardNumRanks,
       olmo::bf16_mega_moe::kernels::kStandardNumLocalExperts},
      long_options);
  auto src_token_topk_indices = at::full(
      {olmo::bf16_mega_moe::kernels::kStandardNumRanks,
       olmo::bf16_mega_moe::kernels::kStandardNumLocalExperts,
       num_route_slots},
      -1,
      long_options);
  auto barrier_state = at::zeros({2}, int_options);

  auto stream = at::cuda::getCurrentCUDAStream();
  olmo::bf16_mega_moe::kernels::standard_ep_fill_workspace_base_ptrs_kernel<<<
      1,
      olmo::bf16_mega_moe::kernels::kStandardNumRanks,
      0,
      stream.stream()>>>(
      workspace.data_ptr(),
      workspace_stride_bytes,
      reinterpret_cast<uint64_t*>(rank_workspace_bases.data_ptr<int64_t>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  olmo::bf16_mega_moe::kernels::standard_ep_full_forward_megakernel_debug_kernel<<<
      olmo::bf16_mega_moe::kernels::kStandardSchedulerSms,
      32,
      0,
      stream.stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(source_input.data_ptr<at::BFloat16>()),
      route_expert_indices.data_ptr<int64_t>(),
      probs.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(up_gate_weight.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(down_weight.data_ptr<at::BFloat16>()),
      num_tokens,
      hidden,
      intermediate,
      num_route_slots,
      num_route_slots,
      reinterpret_cast<const uint64_t*>(rank_workspace_bases.data_ptr<int64_t>()),
      0,
      false,
      false,
      global_counts.data_ptr<int64_t>(),
      global_offsets.data_ptr<int64_t>(),
      expert_cursors.data_ptr<int64_t>(),
      packed_route.data_ptr<int64_t>(),
      route_to_slot.data_ptr<int64_t>(),
      reinterpret_cast<__nv_bfloat16*>(packed_input.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(h.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(packed_expert_out.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(gathered_out.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      recv_counts.data_ptr<int64_t>(),
      recv_ready_counts.data_ptr<int64_t>(),
      src_token_topk_indices.data_ptr<int64_t>(),
      reinterpret_cast<uint32_t*>(barrier_state.data_ptr<int>()),
      false,
      3,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {
      workspace,
      rank_workspace_bases,
      global_counts,
      global_offsets,
      packed_route,
      route_to_slot,
      packed_input,
      h,
      packed_expert_out,
      gathered_out,
      out,
      recv_counts,
      recv_ready_counts,
      src_token_topk_indices,
      barrier_state,
  };
}

std::vector<at::Tensor> rowwise_bf16_mega_moe_standard_ep_full_forward_megakernel(
    at::Tensor& source_input,
    at::Tensor& route_expert_indices,
    at::Tensor& probs,
    at::Tensor& up_gate_weight,
    at::Tensor& down_weight) {
  std::vector<at::Tensor> debug_outputs =
      rowwise_bf16_mega_moe_standard_ep_full_forward_megakernel_debug(
          source_input,
          route_expert_indices,
          probs,
          up_gate_weight,
          down_weight);
  return {debug_outputs[9], debug_outputs[10]};
}

std::vector<int64_t> rowwise_bf16_mega_moe_standard_ep_workspace_config(
    int64_t num_tokens,
    int64_t hidden,
    int64_t intermediate) {
  TORCH_CHECK(num_tokens > 0, "num_tokens must be > 0");
  TORCH_CHECK(
      num_tokens <= olmo::bf16_mega_moe::kernels::kStandardNumMaxTokensPerRank,
      "standard EP workspace supports at most 16384 tokens");
  TORCH_CHECK(hidden > 0 && intermediate > 0, "hidden/intermediate must be > 0");
  TORCH_CHECK(hidden % olmo::bf16_mega_moe::kernels::kWmmaK == 0,
              "hidden must be divisible by 16");
  TORCH_CHECK(intermediate % olmo::bf16_mega_moe::kernels::kWmmaK == 0,
              "intermediate must be divisible by 16");
  const int64_t workspace_stride_bytes = static_cast<int64_t>(
      olmo::bf16_mega_moe::kernels::standard_ep_workspace_stride_bytes(hidden));
  const int64_t num_route_slots =
      num_tokens * static_cast<int64_t>(olmo::bf16_mega_moe::kernels::kStandardTopK);
  const int64_t local_packed_capacity = static_cast<int64_t>(
      olmo::bf16_mega_moe::kernels::standard_ep_local_packed_capacity(num_tokens));
  return {
      workspace_stride_bytes *
          static_cast<int64_t>(olmo::bf16_mega_moe::kernels::kStandardNumRanks),
      workspace_stride_bytes,
      num_route_slots,
      local_packed_capacity,
      static_cast<int64_t>(olmo::bf16_mega_moe::kernels::kStandardNumRanks),
      static_cast<int64_t>(olmo::bf16_mega_moe::kernels::kStandardNumTotalExperts),
      static_cast<int64_t>(olmo::bf16_mega_moe::kernels::kStandardNumLocalExperts),
      static_cast<int64_t>(olmo::bf16_mega_moe::kernels::kStandardTopK),
      2,
      num_route_slots * hidden,
      num_route_slots * intermediate,
  };
}

void rowwise_bf16_mega_moe_standard_ep_forward_persistent_workspace(
    at::Tensor& source_input,
    at::Tensor& gathered_out,
    at::Tensor& out,
    at::Tensor& route_expert_indices,
    at::Tensor& probs,
    at::Tensor& up_gate_weight,
    at::Tensor& down_weight,
    at::Tensor& workspace,
    at::Tensor& rank_workspace_bases,
    at::Tensor& global_counts,
    at::Tensor& global_offsets,
    at::Tensor& expert_cursors,
    at::Tensor& packed_route,
    at::Tensor& route_to_slot,
    at::Tensor& packed_input,
    at::Tensor& h,
    at::Tensor& packed_expert_out,
    at::Tensor& barrier_state,
    int64_t caller_rank_idx,
    bool use_peer_workspace_bases,
    bool enable_cross_rank_barriers,
    bool rank_local_expert_owner,
    bool use_nvshmem_world_collective,
    const std::optional<at::Tensor>& w1_up,
    const std::optional<at::Tensor>& w1_gate,
    bool use_umma_compute) {
  TORCH_CHECK(source_input.is_cuda(), "source_input must be CUDA");
  const c10::cuda::CUDAGuard device_guard(source_input.device());
  TORCH_CHECK(source_input.scalar_type() == at::kBFloat16, "source_input must be BF16");
  TORCH_CHECK(source_input.dim() == 2, "source_input must be rank-2 [tokens, hidden]");
  TORCH_CHECK(source_input.is_contiguous(), "source_input must be contiguous");
  check_route_expert_indices(route_expert_indices);
  TORCH_CHECK(route_expert_indices.scalar_type() == at::kLong,
              "standard EP forward persistent currently requires int64 route indices");
  TORCH_CHECK(probs.is_cuda(), "probs must be CUDA");
  TORCH_CHECK(probs.scalar_type() == at::kFloat, "probs must be FP32");
  TORCH_CHECK(probs.dim() == 2, "probs must be rank-2 [tokens, top_k]");
  TORCH_CHECK(probs.is_contiguous(), "probs must be contiguous");
  TORCH_CHECK(up_gate_weight.is_cuda(), "up_gate_weight must be CUDA");
  TORCH_CHECK(up_gate_weight.scalar_type() == at::kBFloat16, "up_gate_weight must be BF16");
  TORCH_CHECK(up_gate_weight.dim() == 3,
              "up_gate_weight must be rank-3 [32 experts, 2*intermediate, hidden]");
  TORCH_CHECK(up_gate_weight.is_contiguous(), "up_gate_weight must be contiguous");
  TORCH_CHECK(down_weight.is_cuda(), "down_weight must be CUDA");
  TORCH_CHECK(down_weight.scalar_type() == at::kBFloat16, "down_weight must be BF16");
  TORCH_CHECK(down_weight.dim() == 3,
              "down_weight must be rank-3 [32 experts, intermediate, hidden]");
  TORCH_CHECK(down_weight.is_contiguous(), "down_weight must be contiguous");

  auto check_same_device = [&](const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.get_device() == source_input.get_device(),
                name,
                " must be on the same device as source_input");
  };
  auto check_long_1d = [&](const at::Tensor& tensor, const char* name, int64_t len) {
    check_same_device(tensor, name);
    TORCH_CHECK(tensor.scalar_type() == at::kLong, name, " must be int64");
    TORCH_CHECK(tensor.dim() == 1, name, " must be rank-1");
    TORCH_CHECK(tensor.numel() == len, name, " length mismatch");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  };
  auto check_bf16_2d = [&](const at::Tensor& tensor,
                           const char* name,
                           int64_t rows,
                           int64_t cols) {
    check_same_device(tensor, name);
    TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, name, " must be BF16");
    TORCH_CHECK(tensor.dim() == 2, name, " must be rank-2");
    TORCH_CHECK(tensor.size(0) == rows, name, " row dimension mismatch");
    TORCH_CHECK(tensor.size(1) == cols, name, " column dimension mismatch");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  };

  check_same_device(route_expert_indices, "route_expert_indices");
  check_same_device(probs, "probs");
  check_same_device(up_gate_weight, "up_gate_weight");
  check_same_device(down_weight, "down_weight");
  TORCH_CHECK(route_expert_indices.size(0) == source_input.size(0),
              "route_expert_indices token dimension mismatch");
  TORCH_CHECK(probs.sizes() == route_expert_indices.sizes(),
              "probs must match route_expert_indices shape");
  TORCH_CHECK(
      route_expert_indices.size(1) == olmo::bf16_mega_moe::kernels::kStandardTopK,
      "standard EP forward persistent requires top_k=4");
  TORCH_CHECK(
      route_expert_indices.size(0) <=
          olmo::bf16_mega_moe::kernels::kStandardNumMaxTokensPerRank,
      "standard EP forward persistent token count exceeds 16384");
  const int64_t allowed_local_experts =
      static_cast<int64_t>(olmo::bf16_mega_moe::kernels::kStandardNumLocalExperts);
  const int64_t allowed_global_experts =
      static_cast<int64_t>(olmo::bf16_mega_moe::kernels::kStandardNumTotalExperts);
  if (rank_local_expert_owner) {
    TORCH_CHECK(
        up_gate_weight.size(0) == allowed_local_experts ||
            up_gate_weight.size(0) == allowed_global_experts,
        "standard EP local-owner forward requires either 8 local experts or 32 global experts");
    TORCH_CHECK(
        down_weight.size(0) == allowed_local_experts ||
            down_weight.size(0) == allowed_global_experts,
        "standard EP local-owner forward requires either 8 local experts or 32 global experts");
    TORCH_CHECK(
        up_gate_weight.size(0) == down_weight.size(0),
        "standard EP local-owner forward requires up_gate_weight and "
        "down_weight to use the same expert-sharding convention");
  } else {
    TORCH_CHECK(
        up_gate_weight.size(0) == allowed_global_experts,
        "standard EP forward persistent requires 32 global experts");
    TORCH_CHECK(
        down_weight.size(0) == allowed_global_experts,
        "standard EP forward persistent requires 32 global experts");
  }

  const int64_t num_tokens = source_input.size(0);
  const int64_t num_route_slots = route_expert_indices.numel();
  const int64_t hidden = source_input.size(1);
  const int64_t intermediate = down_weight.size(1);
  TORCH_CHECK(up_gate_weight.size(1) == 2 * intermediate,
              "up_gate_weight must have 2*intermediate rows");
  TORCH_CHECK(up_gate_weight.size(2) == hidden,
              "up_gate_weight hidden dimension mismatch");
  TORCH_CHECK(down_weight.size(2) == hidden,
              "down_weight hidden dimension mismatch");
  TORCH_CHECK(hidden % olmo::bf16_mega_moe::kernels::kWmmaK == 0,
              "standard EP forward persistent requires hidden divisible by 16");
  TORCH_CHECK(intermediate % olmo::bf16_mega_moe::kernels::kWmmaK == 0,
              "standard EP forward persistent requires intermediate divisible by 16");
  TORCH_CHECK(
      caller_rank_idx >= 0 &&
          caller_rank_idx < olmo::bf16_mega_moe::kernels::kStandardNumRanks,
      "caller_rank_idx must be in [0, 4)");

  check_same_device(gathered_out, "gathered_out");
  TORCH_CHECK(gathered_out.scalar_type() == at::kBFloat16, "gathered_out must be BF16");
  TORCH_CHECK(gathered_out.dim() == 3, "gathered_out must be rank-3 [tokens, top_k, hidden]");
  TORCH_CHECK(gathered_out.size(0) == num_tokens, "gathered_out token dimension mismatch");
  TORCH_CHECK(gathered_out.size(1) == route_expert_indices.size(1),
              "gathered_out top_k dimension mismatch");
  TORCH_CHECK(gathered_out.size(2) == hidden, "gathered_out hidden dimension mismatch");
  TORCH_CHECK(gathered_out.is_contiguous(), "gathered_out must be contiguous");
  check_bf16_2d(out, "out", num_tokens, hidden);

  const std::vector<int64_t> cfg =
      rowwise_bf16_mega_moe_standard_ep_workspace_config(
          num_tokens,
          hidden,
          intermediate);
  const int64_t workspace_bytes = cfg[0];
  const int64_t workspace_stride_bytes = cfg[1];
  const int64_t local_packed_capacity = cfg[3];
  int64_t packed_capacity =
      rank_local_expert_owner ? local_packed_capacity : num_route_slots;
  if (use_umma_compute && !rank_local_expert_owner) {
    packed_capacity += allowed_global_experts *
        (static_cast<int64_t>(olmo::bf16_mega_moe::kernels::kSm100TileContractM) - 1);
  }
  TORCH_CHECK(workspace.is_cuda(), "workspace must be CUDA");
  TORCH_CHECK(workspace.get_device() == source_input.get_device(),
              "workspace must be on the same device as source_input");
  TORCH_CHECK(workspace.scalar_type() == at::kByte, "workspace must be uint8");
  TORCH_CHECK(workspace.dim() == 1, "workspace must be rank-1 byte tensor");
  const int64_t required_workspace_bytes =
      use_peer_workspace_bases ? workspace_stride_bytes : workspace_bytes;
  TORCH_CHECK(
      workspace.numel() >= required_workspace_bytes,
      "workspace is too small: need at least ",
      required_workspace_bytes,
      " bytes, got ",
      workspace.numel());
  TORCH_CHECK(workspace.is_contiguous(), "workspace must be contiguous");
  if (use_nvshmem_world_collective) {
    TORCH_CHECK(
        use_peer_workspace_bases,
        "standard EP NVSHMEM collective launch requires caller-provided peer workspace bases");
    TORCH_CHECK(
        enable_cross_rank_barriers,
        "standard EP NVSHMEM collective launch requires cross-rank barriers");
    TORCH_CHECK(
        rank_local_expert_owner,
        "standard EP NVSHMEM collective launch requires rank-local expert ownership");
    const int world_pes = nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD);
    TORCH_CHECK(
        world_pes == static_cast<int>(olmo::bf16_mega_moe::kernels::kStandardNumRanks),
        "standard EP NVSHMEM collective launch currently requires NVSHMEM_TEAM_WORLD size ",
        olmo::bf16_mega_moe::kernels::kStandardNumRanks,
        ", got ",
        world_pes);
  } else if (enable_cross_rank_barriers) {
    TORCH_CHECK(
        use_peer_workspace_bases,
        "standard EP peer-workspace cross-rank barriers require caller-provided "
        "peer workspace bases");
    TORCH_CHECK(
        rank_local_expert_owner,
        "standard EP peer-workspace cross-rank barriers require rank-local "
        "expert ownership");
  }
  check_long_1d(
      rank_workspace_bases,
      "rank_workspace_bases",
      olmo::bf16_mega_moe::kernels::kStandardNumRanks);
  check_long_1d(
      global_counts,
      "global_counts",
      olmo::bf16_mega_moe::kernels::kStandardNumTotalExperts);
  check_long_1d(
      global_offsets,
      "global_offsets",
      olmo::bf16_mega_moe::kernels::kStandardNumTotalExperts + 1);
  check_long_1d(
      expert_cursors,
      "expert_cursors",
      olmo::bf16_mega_moe::kernels::kStandardNumTotalExperts);
  check_long_1d(packed_route, "packed_route", packed_capacity);
  check_long_1d(route_to_slot, "route_to_slot", num_route_slots);
  check_bf16_2d(packed_input, "packed_input", packed_capacity, hidden);
  check_bf16_2d(h, "h", packed_capacity, intermediate);
  check_bf16_2d(packed_expert_out, "packed_expert_out", packed_capacity, hidden);
  if (use_umma_compute) {
    TORCH_CHECK(
        hidden % olmo::bf16_mega_moe::kernels::kSm100TileContractN == 0,
        "standard EP UMMA forward requires hidden divisible by 128");
    TORCH_CHECK(
        intermediate % olmo::bf16_mega_moe::kernels::kSm100TileContractN == 0,
        "standard EP UMMA forward requires intermediate divisible by 128");
    TORCH_CHECK(w1_up.has_value(), "standard EP UMMA forward requires w1_up scratch");
    TORCH_CHECK(w1_gate.has_value(), "standard EP UMMA forward requires w1_gate scratch");
    check_bf16_2d(w1_up.value(), "w1_up", packed_capacity, intermediate);
    check_bf16_2d(w1_gate.value(), "w1_gate", packed_capacity, intermediate);
  }
  check_same_device(barrier_state, "barrier_state");
  TORCH_CHECK(barrier_state.scalar_type() == at::kInt, "barrier_state must be int32");
  TORCH_CHECK(barrier_state.dim() == 1, "barrier_state must be rank-1");
  TORCH_CHECK(barrier_state.numel() >= 2, "barrier_state must have at least 2 elements");
  TORCH_CHECK(barrier_state.is_contiguous(), "barrier_state must be contiguous");

  auto stream = at::cuda::getCurrentCUDAStream();
  if (enable_cross_rank_barriers) {
    const int pre_clear_barrier_status =
        nvshmemx_barrier_on_stream(NVSHMEM_TEAM_WORLD, stream.stream());
    TORCH_CHECK(
        pre_clear_barrier_status == 0,
        "nvshmemx_barrier_on_stream (standard EP pre-clear) failed with status ",
        pre_clear_barrier_status);
  }
  C10_CUDA_CHECK(cudaMemsetAsync(
      workspace.data_ptr(),
      0,
      static_cast<size_t>(required_workspace_bytes),
      stream.stream()));
  C10_CUDA_CHECK(cudaMemsetAsync(
      barrier_state.data_ptr<int>(),
      0,
      2 * sizeof(int),
      stream.stream()));

  if (!use_peer_workspace_bases) {
    olmo::bf16_mega_moe::kernels::standard_ep_fill_workspace_base_ptrs_kernel<<<
        1,
        olmo::bf16_mega_moe::kernels::kStandardNumRanks,
        0,
        stream.stream()>>>(
        workspace.data_ptr(),
        static_cast<uint64_t>(workspace_stride_bytes),
        reinterpret_cast<uint64_t*>(rank_workspace_bases.data_ptr<int64_t>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  const __nv_bfloat16* up_gate_weight_ptr =
      reinterpret_cast<const __nv_bfloat16*>(up_gate_weight.data_ptr<at::BFloat16>());
  const __nv_bfloat16* down_weight_ptr =
      reinterpret_cast<const __nv_bfloat16*>(down_weight.data_ptr<at::BFloat16>());
  at::Tensor up_gate_weight_for_map = up_gate_weight;
  at::Tensor down_weight_for_map = down_weight;
  if (rank_local_expert_owner && up_gate_weight.size(0) == allowed_global_experts) {
    const int64_t expert_base = caller_rank_idx * allowed_local_experts;
    up_gate_weight_ptr += expert_base * 2 * intermediate * hidden;
    down_weight_ptr += expert_base * intermediate * hidden;
    up_gate_weight_for_map = up_gate_weight.narrow(0, expert_base, allowed_local_experts);
    down_weight_for_map = down_weight.narrow(0, expert_base, allowed_local_experts);
  }

  int64_t umma_debug_phase_limit_arg = 3;
  if (const char* phase_limit_env = std::getenv("OLMO_BF16_MEGA_UMMA_PHASE_LIMIT")) {
    umma_debug_phase_limit_arg = std::strtoll(phase_limit_env, nullptr, 10);
  }

  std::optional<at::Tensor> packed_input_map_storage;
  std::optional<at::Tensor> h_map_storage;
  std::optional<at::Tensor> up_gate_weight_map_storage;
  std::optional<at::Tensor> down_weight_map_storage;
  const CUtensorMap* packed_input_map_ptr = nullptr;
  const CUtensorMap* h_map_ptr = nullptr;
  const CUtensorMap* up_gate_weight_map_ptr = nullptr;
  const CUtensorMap* down_weight_map_ptr = nullptr;
  if (use_umma_compute && umma_debug_phase_limit_arg > 0) {
    constexpr int64_t swizzle_bytes =
        olmo::bf16_mega_moe::kernels::kSm100TileContractK *
        static_cast<int64_t>(sizeof(__nv_bfloat16));
    const olmo::bf16_mega_moe::Bf16TensorMap2D packed_input_map =
        olmo::bf16_mega_moe::make_bf16_tma_2d_desc(
            packed_input,
            olmo::bf16_mega_moe::kernels::kSm100TileContractK,
            olmo::bf16_mega_moe::kernels::kSm100TileContractM,
            swizzle_bytes);
    const olmo::bf16_mega_moe::Bf16TensorMap2D h_map =
        olmo::bf16_mega_moe::make_bf16_tma_2d_desc(
            h,
            olmo::bf16_mega_moe::kernels::kSm100TileContractK,
            olmo::bf16_mega_moe::kernels::kSm100TileContractM,
            swizzle_bytes);
    const int64_t mapped_experts = up_gate_weight_for_map.size(0);
    const olmo::bf16_mega_moe::Bf16TensorMap2D up_gate_weight_map =
        olmo::bf16_mega_moe::make_bf16_tma_2d_desc(
            up_gate_weight_for_map.view({mapped_experts * 2 * intermediate, hidden}),
            olmo::bf16_mega_moe::kernels::kSm100TileContractK,
            olmo::bf16_mega_moe::kernels::kSm100TileContractN,
            swizzle_bytes);
    const olmo::bf16_mega_moe::Bf16TensorMap2D down_weight_map =
        olmo::bf16_mega_moe::make_bf16_tma_2d_desc(
            down_weight_for_map.view({mapped_experts * intermediate, hidden}),
            olmo::bf16_mega_moe::kernels::kSm100TileContractK,
            olmo::bf16_mega_moe::kernels::kSm100TileContractK,
            swizzle_bytes);
    auto make_tensor_map_storage =
        [&](const olmo::bf16_mega_moe::Bf16TensorMap2D& tensor_map) {
          auto storage = at::empty(
              {static_cast<int64_t>(sizeof(CUtensorMap))},
              at::TensorOptions().device(source_input.device()).dtype(at::kByte));
          C10_CUDA_CHECK(cudaMemcpyAsync(
              storage.data_ptr(),
              &tensor_map.map,
              sizeof(CUtensorMap),
              cudaMemcpyHostToDevice,
              stream.stream()));
          return storage;
        };
    packed_input_map_storage = make_tensor_map_storage(packed_input_map);
    h_map_storage = make_tensor_map_storage(h_map);
    up_gate_weight_map_storage = make_tensor_map_storage(up_gate_weight_map);
    down_weight_map_storage = make_tensor_map_storage(down_weight_map);
    packed_input_map_ptr =
        reinterpret_cast<const CUtensorMap*>(packed_input_map_storage->data_ptr());
    h_map_ptr = reinterpret_cast<const CUtensorMap*>(h_map_storage->data_ptr());
    up_gate_weight_map_ptr =
        reinterpret_cast<const CUtensorMap*>(up_gate_weight_map_storage->data_ptr());
    down_weight_map_ptr =
        reinterpret_cast<const CUtensorMap*>(down_weight_map_storage->data_ptr());
  }

  const __nv_bfloat16* source_input_ptr =
      reinterpret_cast<const __nv_bfloat16*>(source_input.data_ptr<at::BFloat16>());
  const int64_t* route_expert_indices_ptr = route_expert_indices.data_ptr<int64_t>();
  const float* probs_ptr = probs.data_ptr<float>();
  const uint64_t* rank_workspace_bases_ptr =
      reinterpret_cast<const uint64_t*>(rank_workspace_bases.data_ptr<int64_t>());
  int64_t* global_counts_ptr = global_counts.data_ptr<int64_t>();
  int64_t* global_offsets_ptr = global_offsets.data_ptr<int64_t>();
  int64_t* expert_cursors_ptr = expert_cursors.data_ptr<int64_t>();
  int64_t* packed_route_ptr = packed_route.data_ptr<int64_t>();
  int64_t* route_to_slot_ptr = route_to_slot.data_ptr<int64_t>();
  __nv_bfloat16* packed_input_ptr =
      reinterpret_cast<__nv_bfloat16*>(packed_input.data_ptr<at::BFloat16>());
  __nv_bfloat16* h_ptr =
      reinterpret_cast<__nv_bfloat16*>(h.data_ptr<at::BFloat16>());
  __nv_bfloat16* packed_expert_out_ptr =
      reinterpret_cast<__nv_bfloat16*>(packed_expert_out.data_ptr<at::BFloat16>());
  __nv_bfloat16* gathered_out_ptr =
      reinterpret_cast<__nv_bfloat16*>(gathered_out.data_ptr<at::BFloat16>());
  __nv_bfloat16* out_ptr =
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
  int64_t* recv_counts_ptr = nullptr;
  int64_t* recv_ready_counts_ptr = nullptr;
  int64_t* src_token_topk_indices_ptr = nullptr;
  uint32_t* barrier_state_ptr = reinterpret_cast<uint32_t*>(barrier_state.data_ptr<int>());
  int64_t num_tokens_arg = num_tokens;
  int64_t hidden_arg = hidden;
  int64_t intermediate_arg = intermediate;
  int64_t num_route_slots_arg = num_route_slots;
  int64_t packed_capacity_arg = packed_capacity;
  int64_t caller_rank_idx_arg = caller_rank_idx;
  bool enable_cross_rank_barriers_arg = enable_cross_rank_barriers;
  bool rank_local_expert_owner_arg = rank_local_expert_owner;
  bool use_umma_compute_arg = use_umma_compute;
  __nv_bfloat16* w1_up_ptr = use_umma_compute
      ? reinterpret_cast<__nv_bfloat16*>(w1_up.value().data_ptr<at::BFloat16>())
      : nullptr;
  __nv_bfloat16* w1_gate_ptr = use_umma_compute
      ? reinterpret_cast<__nv_bfloat16*>(w1_gate.value().data_ptr<at::BFloat16>())
      : nullptr;
  const bool force_umma_block32 =
      std::getenv("OLMO_BF16_MEGA_UMMA_FORCE_BLOCK32") != nullptr;
  const dim3 kernel_block(
      use_umma_compute && !force_umma_block32
          ? olmo::bf16_mega_moe::kernels::kStandardUmmaBlockThreads
          : 32);

  if (use_nvshmem_world_collective) {
    maybe_init_wave_nvshmem_cumodule(reinterpret_cast<const void*>(
        olmo::bf16_mega_moe::kernels::standard_ep_full_forward_megakernel_debug_kernel));
    const int pre_barrier_status =
        nvshmemx_barrier_on_stream(NVSHMEM_TEAM_WORLD, stream.stream());
    TORCH_CHECK(
        pre_barrier_status == 0,
        "nvshmemx_barrier_on_stream (standard EP pre) failed with status ",
        pre_barrier_status);
    void* args[] = {
        &source_input_ptr,
        &route_expert_indices_ptr,
        &probs_ptr,
        &up_gate_weight_ptr,
        &down_weight_ptr,
        &num_tokens_arg,
        &hidden_arg,
        &intermediate_arg,
        &num_route_slots_arg,
        &packed_capacity_arg,
        &rank_workspace_bases_ptr,
        &caller_rank_idx_arg,
        &enable_cross_rank_barriers_arg,
        &rank_local_expert_owner_arg,
        &global_counts_ptr,
        &global_offsets_ptr,
        &expert_cursors_ptr,
        &packed_route_ptr,
        &route_to_slot_ptr,
        &packed_input_ptr,
        &h_ptr,
        &packed_expert_out_ptr,
        &gathered_out_ptr,
        &out_ptr,
        &recv_counts_ptr,
        &recv_ready_counts_ptr,
        &src_token_topk_indices_ptr,
        &barrier_state_ptr,
        &use_umma_compute_arg,
        &umma_debug_phase_limit_arg,
        &packed_input_map_ptr,
        &h_map_ptr,
        &up_gate_weight_map_ptr,
        &down_weight_map_ptr,
        &w1_up_ptr,
        &w1_gate_ptr};
    nvshmemx_collective_launch(
        reinterpret_cast<const void*>(
            olmo::bf16_mega_moe::kernels::standard_ep_full_forward_megakernel_debug_kernel),
        dim3(olmo::bf16_mega_moe::kernels::kStandardSchedulerSms),
        kernel_block,
        args,
        0,
        stream);
  } else {
    olmo::bf16_mega_moe::kernels::standard_ep_full_forward_megakernel_debug_kernel<<<
        olmo::bf16_mega_moe::kernels::kStandardSchedulerSms,
        kernel_block,
        0,
        stream.stream()>>>(
        source_input_ptr,
        route_expert_indices_ptr,
        probs_ptr,
        up_gate_weight_ptr,
        down_weight_ptr,
        num_tokens,
        hidden,
        intermediate,
        num_route_slots,
        packed_capacity,
        rank_workspace_bases_ptr,
        caller_rank_idx,
        enable_cross_rank_barriers,
        rank_local_expert_owner,
        global_counts_ptr,
        global_offsets_ptr,
        expert_cursors_ptr,
        packed_route_ptr,
        route_to_slot_ptr,
        packed_input_ptr,
        h_ptr,
        packed_expert_out_ptr,
        gathered_out_ptr,
        out_ptr,
        recv_counts_ptr,
        recv_ready_counts_ptr,
        src_token_topk_indices_ptr,
        barrier_state_ptr,
        use_umma_compute,
        umma_debug_phase_limit_arg,
        packed_input_map_ptr,
        h_map_ptr,
        up_gate_weight_map_ptr,
        down_weight_map_ptr,
        w1_up_ptr,
        w1_gate_ptr);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rowwise_bf16_mega_moe_standard_ep_forward_persistent(
    at::Tensor& source_input,
    at::Tensor& gathered_out,
    at::Tensor& out,
    at::Tensor& route_expert_indices,
    at::Tensor& probs,
    at::Tensor& up_gate_weight,
    at::Tensor& down_weight) {
  TORCH_CHECK(source_input.is_cuda(), "source_input must be CUDA");
  TORCH_CHECK(source_input.scalar_type() == at::kBFloat16, "source_input must be BF16");
  TORCH_CHECK(source_input.dim() == 2, "source_input must be rank-2 [tokens, hidden]");
  TORCH_CHECK(source_input.is_contiguous(), "source_input must be contiguous");
  check_route_expert_indices(route_expert_indices);
  TORCH_CHECK(route_expert_indices.scalar_type() == at::kLong,
              "standard EP forward persistent currently requires int64 route indices");
  TORCH_CHECK(probs.is_cuda(), "probs must be CUDA");
  TORCH_CHECK(probs.scalar_type() == at::kFloat, "probs must be FP32");
  TORCH_CHECK(probs.dim() == 2, "probs must be rank-2 [tokens, top_k]");
  TORCH_CHECK(probs.is_contiguous(), "probs must be contiguous");
  TORCH_CHECK(up_gate_weight.is_cuda(), "up_gate_weight must be CUDA");
  TORCH_CHECK(up_gate_weight.scalar_type() == at::kBFloat16, "up_gate_weight must be BF16");
  TORCH_CHECK(up_gate_weight.dim() == 3,
              "up_gate_weight must be rank-3 [32 experts, 2*intermediate, hidden]");
  TORCH_CHECK(up_gate_weight.is_contiguous(), "up_gate_weight must be contiguous");
  TORCH_CHECK(down_weight.is_cuda(), "down_weight must be CUDA");
  TORCH_CHECK(down_weight.scalar_type() == at::kBFloat16, "down_weight must be BF16");
  TORCH_CHECK(down_weight.dim() == 3,
              "down_weight must be rank-3 [32 experts, intermediate, hidden]");
  TORCH_CHECK(down_weight.is_contiguous(), "down_weight must be contiguous");
  TORCH_CHECK(route_expert_indices.get_device() == source_input.get_device(),
              "route_expert_indices must be on the same device as source_input");
  TORCH_CHECK(probs.get_device() == source_input.get_device(),
              "probs must be on the same device as source_input");
  TORCH_CHECK(up_gate_weight.get_device() == source_input.get_device(),
              "up_gate_weight must be on the same device as source_input");
  TORCH_CHECK(down_weight.get_device() == source_input.get_device(),
              "down_weight must be on the same device as source_input");
  TORCH_CHECK(route_expert_indices.size(0) == source_input.size(0),
              "route_expert_indices token dimension mismatch");
  TORCH_CHECK(probs.sizes() == route_expert_indices.sizes(),
              "probs must match route_expert_indices shape");
  TORCH_CHECK(
      route_expert_indices.size(1) == olmo::bf16_mega_moe::kernels::kStandardTopK,
      "standard EP forward persistent requires top_k=4");
  TORCH_CHECK(
      route_expert_indices.size(0) <=
          olmo::bf16_mega_moe::kernels::kStandardNumMaxTokensPerRank,
      "standard EP forward persistent token count exceeds 16384");
  TORCH_CHECK(
      up_gate_weight.size(0) == olmo::bf16_mega_moe::kernels::kStandardNumTotalExperts,
      "standard EP forward persistent requires 32 global experts");
  TORCH_CHECK(
      down_weight.size(0) == olmo::bf16_mega_moe::kernels::kStandardNumTotalExperts,
      "standard EP forward persistent requires 32 global experts");

  const int64_t num_tokens = source_input.size(0);
  const int64_t num_route_slots = route_expert_indices.numel();
  const int64_t hidden = source_input.size(1);
  const int64_t intermediate = down_weight.size(1);
  TORCH_CHECK(up_gate_weight.size(1) == 2 * intermediate,
              "up_gate_weight must have 2*intermediate rows");
  TORCH_CHECK(up_gate_weight.size(2) == hidden,
              "up_gate_weight hidden dimension mismatch");
  TORCH_CHECK(down_weight.size(2) == hidden,
              "down_weight hidden dimension mismatch");
  TORCH_CHECK(hidden % olmo::bf16_mega_moe::kernels::kWmmaK == 0,
              "standard EP forward persistent requires hidden divisible by 16");
  TORCH_CHECK(intermediate % olmo::bf16_mega_moe::kernels::kWmmaK == 0,
              "standard EP forward persistent requires intermediate divisible by 16");

  TORCH_CHECK(gathered_out.is_cuda(), "gathered_out must be CUDA");
  TORCH_CHECK(gathered_out.scalar_type() == at::kBFloat16, "gathered_out must be BF16");
  TORCH_CHECK(gathered_out.dim() == 3, "gathered_out must be rank-3 [tokens, top_k, hidden]");
  TORCH_CHECK(gathered_out.is_contiguous(), "gathered_out must be contiguous");
  TORCH_CHECK(out.is_cuda(), "out must be CUDA");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16, "out must be BF16");
  TORCH_CHECK(out.dim() == 2, "out must be rank-2 [tokens, hidden]");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(gathered_out.get_device() == source_input.get_device(),
              "gathered_out must be on the same device as source_input");
  TORCH_CHECK(out.get_device() == source_input.get_device(),
              "out must be on the same device as source_input");
  TORCH_CHECK(gathered_out.size(0) == source_input.size(0),
              "gathered_out token dimension mismatch");
  TORCH_CHECK(gathered_out.size(1) == route_expert_indices.size(1),
              "gathered_out top_k dimension mismatch");
  TORCH_CHECK(gathered_out.size(2) == source_input.size(1),
              "gathered_out hidden dimension mismatch");
  TORCH_CHECK(out.size(0) == source_input.size(0), "out token dimension mismatch");
  TORCH_CHECK(out.size(1) == source_input.size(1), "out hidden dimension mismatch");

  const uint64_t workspace_stride_bytes =
      olmo::bf16_mega_moe::kernels::standard_ep_workspace_stride_bytes(source_input.size(1));
  auto byte_options =
      at::TensorOptions().device(source_input.device()).dtype(at::kByte);
  auto long_options =
      at::TensorOptions().device(source_input.device()).dtype(at::kLong);
  auto bf16_options =
      at::TensorOptions().device(source_input.device()).dtype(at::kBFloat16);
  auto int_options =
      at::TensorOptions().device(source_input.device()).dtype(at::kInt);

  auto workspace = at::zeros(
      {static_cast<int64_t>(
          workspace_stride_bytes * olmo::bf16_mega_moe::kernels::kStandardNumRanks)},
      byte_options);
  auto rank_workspace_bases = at::empty(
      {olmo::bf16_mega_moe::kernels::kStandardNumRanks},
      long_options);
  auto global_counts = at::empty(
      {olmo::bf16_mega_moe::kernels::kStandardNumTotalExperts},
      long_options);
  auto global_offsets = at::empty(
      {olmo::bf16_mega_moe::kernels::kStandardNumTotalExperts + 1},
      long_options);
  auto expert_cursors = at::empty(
      {olmo::bf16_mega_moe::kernels::kStandardNumTotalExperts},
      long_options);
  auto packed_route = at::empty({num_route_slots}, long_options);
  auto route_to_slot = at::empty({num_route_slots}, long_options);
  auto packed_input = at::empty({num_route_slots, hidden}, bf16_options);
  auto h = at::empty({num_route_slots, intermediate}, bf16_options);
  auto packed_expert_out = at::empty({num_route_slots, hidden}, bf16_options);
  auto barrier_state = at::zeros({2}, int_options);

  auto stream = at::cuda::getCurrentCUDAStream();
  olmo::bf16_mega_moe::kernels::standard_ep_fill_workspace_base_ptrs_kernel<<<
      1,
      olmo::bf16_mega_moe::kernels::kStandardNumRanks,
      0,
      stream.stream()>>>(
      workspace.data_ptr(),
      workspace_stride_bytes,
      reinterpret_cast<uint64_t*>(rank_workspace_bases.data_ptr<int64_t>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  olmo::bf16_mega_moe::kernels::standard_ep_full_forward_megakernel_debug_kernel<<<
      olmo::bf16_mega_moe::kernels::kStandardSchedulerSms,
      32,
      0,
      stream.stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(source_input.data_ptr<at::BFloat16>()),
      route_expert_indices.data_ptr<int64_t>(),
      probs.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(up_gate_weight.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(down_weight.data_ptr<at::BFloat16>()),
      num_tokens,
      hidden,
      intermediate,
      num_route_slots,
      num_route_slots,
      reinterpret_cast<const uint64_t*>(rank_workspace_bases.data_ptr<int64_t>()),
      0,
      false,
      false,
      global_counts.data_ptr<int64_t>(),
      global_offsets.data_ptr<int64_t>(),
      expert_cursors.data_ptr<int64_t>(),
      packed_route.data_ptr<int64_t>(),
      route_to_slot.data_ptr<int64_t>(),
      reinterpret_cast<__nv_bfloat16*>(packed_input.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(h.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(packed_expert_out.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(gathered_out.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      nullptr,
      nullptr,
      nullptr,
      reinterpret_cast<uint32_t*>(barrier_state.data_ptr<int>()),
      false,
      3,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rowwise_bf16_mega_moe_forward_persistent(
    at::Tensor& source_input,
    at::Tensor& gathered_out,
    at::Tensor& out,
    at::Tensor& route_dst_ranks,
    at::Tensor& route_dst_rows,
    at::Tensor& route_expert_indices,
    at::Tensor& probs,
    at::Tensor& up_gate_weight,
    at::Tensor& down_weight,
    at::Tensor& expert_offsets,
    const std::string& group_name,
    const std::optional<at::Tensor>& route_done_counts,
    const std::optional<at::Tensor>& symm_probs,
    bool pre_barrier,
    bool post_barrier) {
  (void)group_name;
  (void)pre_barrier;
  (void)post_barrier;
  if (source_input.is_cuda()) {
    const c10::cuda::CUDAGuard device_guard(source_input.device());
    const int64_t num_sms = olmo::bf16_mega_moe::current_device_sm_count();
    const int64_t num_ranks = 1;
    const int64_t num_total_experts = up_gate_weight.dim() > 0 ? up_gate_weight.size(0) : 0;
    const int64_t num_max_tokens_per_rank =
        source_input.dim() > 0 ? source_input.size(0) : 0;
    const auto problem = olmo::bf16_mega_moe::make_forward_problem(
        source_input,
        gathered_out,
        out,
        route_dst_ranks,
        route_dst_rows,
        route_expert_indices,
        probs,
        up_gate_weight,
        down_weight,
        expert_offsets,
        num_ranks,
        num_total_experts,
        num_max_tokens_per_rank,
        num_sms);
    TORCH_CHECK(
        !route_done_counts.has_value(),
        "rowwise_bf16_mega_moe_forward_persistent local CUDA forward does not "
        "support route_done_counts yet");
    TORCH_CHECK(
        !symm_probs.has_value(),
        "rowwise_bf16_mega_moe_forward_persistent local CUDA forward does not "
        "support symmetric probs yet");
    TORCH_CHECK(
        route_expert_indices.scalar_type() == at::kLong,
        "rowwise_bf16_mega_moe_forward_persistent local CUDA forward currently "
        "requires int64 route_expert_indices");
    TORCH_CHECK(
        problem.hidden >= olmo::bf16_mega_moe::kernels::kSm100TileContractN,
        "rowwise_bf16_mega_moe_forward_persistent local CUDA forward requires "
        "hidden >= 128");
    TORCH_CHECK(
        problem.intermediate >= olmo::bf16_mega_moe::kernels::kSm100TileContractN,
        "rowwise_bf16_mega_moe_forward_persistent local CUDA forward requires "
        "intermediate >= 128");
    TORCH_CHECK(
        problem.hidden % olmo::bf16_mega_moe::kernels::kSm100TileContractN == 0,
        "rowwise_bf16_mega_moe_forward_persistent local CUDA forward requires "
        "hidden divisible by 128");
    TORCH_CHECK(
        problem.intermediate % olmo::bf16_mega_moe::kernels::kSm100TileContractN == 0,
        "rowwise_bf16_mega_moe_forward_persistent local CUDA forward requires "
        "intermediate divisible by 128");

    std::vector<at::Tensor> packed =
        rowwise_bf16_mega_moe_route_pack_inputs_debug(
            source_input,
            route_expert_indices,
            probs,
            problem.num_local_experts);
    at::Tensor expert_counts = count_routes(
        route_expert_indices,
        problem.num_local_experts);
    std::vector<at::Tensor> computed =
        rowwise_bf16_mega_moe_local_umma_compute_debug(
            packed[2],
            expert_counts,
            up_gate_weight,
            down_weight);
    std::vector<at::Tensor> combined =
        rowwise_bf16_mega_moe_combine_debug(
            computed[6],
            packed[1],
            probs);
    gathered_out.copy_(combined[0]);
    out.copy_(combined[1]);
    return;
  }
  olmo::bf16_mega_moe::fail_unimplemented_forward_body();
}

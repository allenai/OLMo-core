#include <ATen/ceil_div.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nvshmem_team_manager.hpp>

#include <algorithm>
#include <cub/cub.cuh>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// NVSHMEM minimum SM arch
#define _NVSHMEM_MIN_SM_ARCH 700

// If CUDA_ARCH is less than sm_70, or on sm_110, skip NVSHMEM device APIs
#define _NVSHMEM_DEVICELIB_SUPPORTED 1
#if defined(__CUDA_ARCH__)
#if (__CUDA_ARCH__ < _NVSHMEM_MIN_SM_ARCH) || (__CUDA_ARCH__ == 1100)
#undef _NVSHMEM_DEVICELIB_SUPPORTED
#endif
#endif

// Some NVSHMEM device APIs do not compile on older SM archs
#ifndef _NVSHMEM_DEVICELIB_SUPPORTED
#define NVSHMEM_HOSTLIB_ONLY
#endif

#include <nvshmem.h>
#include <nvshmemx.h>

#define THREADS_PER_BLOCK 512
#define WARP_SIZE 32
#define ROWWISE_THREADS_PER_BLOCK 512
#define ROWWISE_WARPS_PER_BLOCK (ROWWISE_THREADS_PER_BLOCK / WARP_SIZE)
#define ROWWISE_COMBINE_FUSED_THREADS_PER_BLOCK 256
#define ROWWISE_COMBINE_FUSED_WARPS_PER_BLOCK \
  (ROWWISE_COMBINE_FUSED_THREADS_PER_BLOCK / WARP_SIZE)
#define ROWWISE_COMBINE_FUSED_VECS_PER_THREAD 16
#define ROWWISE_WEIGHTED_PUT_ELEMS_PER_THREAD 32

namespace {

struct OlmoSymmGroupInfo {
  int world_size = 0;
  int* rank_to_pe_dev = nullptr;
  std::vector<int> rank_to_pe_host;
};

struct OlmoSymmState {
  bool initialized = false;
  int rank = -1;
  int world_size = -1;
  int device_idx = -1;
  std::mutex mutex;
  std::vector<void*> allocations;
  std::unordered_map<std::string, OlmoSymmGroupInfo> groups;
};

OlmoSymmState& olmo_symm_state() {
  static OlmoSymmState state;
  return state;
}

void olmo_maybe_initialize_env_vars() {
  const char* nccl_socket_if_name = std::getenv("NCCL_SOCKET_IFNAME");
  const char* nccl_hca_list = std::getenv("NCCL_IB_HCA");
  const char* nccl_ib_gid_index = std::getenv("NCCL_IB_GID_INDEX");

  if (std::getenv("NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME") == nullptr &&
      nccl_socket_if_name != nullptr) {
    setenv("NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME", nccl_socket_if_name, 0);
  }
  if (std::getenv("NVSHMEM_HCA_LIST") == nullptr && nccl_hca_list != nullptr) {
    setenv("NVSHMEM_ENABLE_NIC_PE_MAPPING", "1", 0);
    setenv("NVSHMEM_HCA_LIST", nccl_hca_list, 0);
  }
  if (std::getenv("NVSHMEM_IB_GID_INDEX") == nullptr &&
      nccl_ib_gid_index != nullptr) {
    setenv("NVSHMEM_IB_GID_INDEX", nccl_ib_gid_index, 0);
  }
}

OlmoSymmGroupInfo* olmo_symm_find_group(const std::string& group_name) {
  auto& state = olmo_symm_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  auto it = state.groups.find(group_name);
  if (it == state.groups.end()) {
    return nullptr;
  }
  return &it->second;
}

int olmo_symm_group_local_rank(const OlmoSymmGroupInfo& group, int global_rank) {
  for (size_t rank_idx = 0; rank_idx < group.rank_to_pe_host.size(); ++rank_idx) {
    if (group.rank_to_pe_host[rank_idx] == global_rank) {
      return static_cast<int>(rank_idx);
    }
  }
  TORCH_CHECK(
      false,
      "Current NVSHMEM PE ",
      global_rank,
      " is not a member of the requested OLMo symmetric-memory group");
  return -1;
}

__device__ __forceinline__ int olmo_route_npes(
    nvshmem_team_t team,
    const int* rank_to_pe,
    int group_size) {
  return rank_to_pe == nullptr ? nvshmem_team_n_pes(team) : group_size;
}

__device__ __forceinline__ int olmo_route_peer_global(
    nvshmem_team_t team,
    const int* rank_to_pe,
    int peer) {
  return rank_to_pe == nullptr
      ? nvshmem_team_translate_pe(team, peer, NVSHMEM_TEAM_WORLD)
      : rank_to_pe[peer];
}

__device__ __forceinline__ int64_t olmo_atomic_add_i64(
    int64_t* ptr,
    int64_t value) {
  return static_cast<int64_t>(atomicAdd(
      reinterpret_cast<unsigned long long*>(ptr),
      static_cast<unsigned long long>(value)));
}

__device__ __forceinline__ uint64_t olmo_load_acquire_sys_u64(
    const uint64_t* ptr) {
  uint64_t ret;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
  return ret;
}

__device__ int64_t prefixSum(int64_t* odata, int64_t* idata, int n) {
  using BlockScanT =
      cub::BlockScan<int64_t, THREADS_PER_BLOCK, cub::BLOCK_SCAN_WARP_SCANS>;
  __shared__ typename BlockScanT::TempStorage temp_storage;

  CUDA_KERNEL_ASSERT(n <= THREADS_PER_BLOCK);

  int tid = threadIdx.x;
  int64_t thread_data = (tid < n) ? idata[tid] : 0;

  int64_t block_aggregate;
  BlockScanT(temp_storage).ExclusiveSum(thread_data, thread_data, block_aggregate);

  odata[tid] = thread_data;
  return block_aggregate;
}

template <int NUM_WARPS>
__device__ int64_t prefixSum_warp(int64_t* odata, int64_t* idata, int n) {
  CUDA_KERNEL_ASSERT(n <= WARP_SIZE);

  using WarpScan = cub::WarpScan<int64_t>;
  __shared__ typename WarpScan::TempStorage temp_storage[NUM_WARPS];

  int warp_id = threadIdx.x / WARP_SIZE;
  if (warp_id >= NUM_WARPS) {
    return 0;
  }

  int tid = threadIdx.x % WARP_SIZE;
  int64_t thread_data = (tid < n) ? idata[tid] : 0;

  int64_t warp_aggregate;
  WarpScan(temp_storage[warp_id]).ExclusiveSum(
      thread_data, thread_data, warp_aggregate);

  // Fix OOB write when n < WARP_SIZE.
  if (tid < n) {
    odata[tid] = thread_data;
  }
  return warp_aggregate;
}

__global__ void waitSignalPeersKernel(
    uint64_t* signals,
    int64_t signal_row,
    int64_t group_size,
    uint64_t generation) {
  int64_t peer = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (peer >= group_size) {
    return;
  }
  uint64_t* slot =
      signals + static_cast<size_t>(signal_row) * static_cast<size_t>(group_size) +
      static_cast<size_t>(peer);
  while (olmo_load_acquire_sys_u64(slot) < generation) {
    __nanosleep(100);
  }
}

int resolve_num_blocks(int world_size, int ne, int64_t requested_nblocks) {
  if (requested_nblocks > 0) {
    TORCH_CHECK(
        requested_nblocks <= std::numeric_limits<int>::max(),
        "nblocks is too large");
    return static_cast<int>(requested_nblocks);
  }
  return 0;
}

int resolve_num_blocks_auto(
    size_t input_size_bytes,
    int world_size,
    int ne,
    bool intra_node) {
  constexpr size_t chunk_size = 16 * THREADS_PER_BLOCK * 8;
  int nsplits = world_size * ne;
  TORCH_CHECK(nsplits > 0, "nsplits must be > 0");

  int num_blocks = at::ceil_div(input_size_bytes, chunk_size);
  num_blocks = std::max(num_blocks, nsplits);
  num_blocks = at::round_up(num_blocks, nsplits);
  int max_blocks = intra_node ? 256 : 64;
  return std::min(num_blocks, max_blocks);
}

int resolve_num_blocks_rowwise(
    int64_t num_routes,
    int64_t requested_nblocks,
    bool intra_node) {
  if (requested_nblocks > 0) {
    TORCH_CHECK(
        requested_nblocks <= std::numeric_limits<int>::max(),
        "nblocks is too large");
    return static_cast<int>(requested_nblocks);
  }
  if (num_routes <= 0) {
    return 1;
  }

  auto* props = at::cuda::getCurrentDeviceProperties();
  int sm_count = std::max(props->multiProcessorCount, 1);
  int target_blocks = sm_count * 4;
  int max_blocks = intra_node ? 2048 : 512;
  int64_t capped = std::min<int64_t>(num_routes, max_blocks);
  return std::max<int>(1, static_cast<int>(std::min<int64_t>(target_blocks, capped)));
}

int64_t resolve_num_row_blocks_fused(int64_t num_out_rows, int num_blocks) {
  static int factor = []() {
    constexpr int kDefault = 16;
    constexpr int kMin = 1;
    constexpr int kMax = 64;
    const char* env = std::getenv("OLMO_ROWWISE_COMBINE_FUSED_ROW_BLOCK_FACTOR");
    if (env == nullptr || env[0] == '\0') {
      return kDefault;
    }
    char* end = nullptr;
    long parsed = std::strtol(env, &end, 10);
    if (end == env || *end != '\0') {
      return kDefault;
    }
    parsed = std::max<long>(parsed, kMin);
    parsed = std::min<long>(parsed, kMax);
    return static_cast<int>(parsed);
  }();

  int64_t row_blocks = std::min<int64_t>(
      num_out_rows, static_cast<int64_t>(num_blocks) * factor);
  return std::max<int64_t>(row_blocks, 1);
}

struct CollectiveLaunchKey {
  uintptr_t func = 0;
  unsigned int block_x = 0;
  unsigned int block_y = 0;
  unsigned int block_z = 0;
  size_t shared_mem = 0;
  int device_idx = -1;

  bool operator==(const CollectiveLaunchKey& other) const {
    return func == other.func && block_x == other.block_x &&
        block_y == other.block_y && block_z == other.block_z &&
        shared_mem == other.shared_mem && device_idx == other.device_idx;
  }
};

struct CollectiveLaunchKeyHash {
  size_t operator()(const CollectiveLaunchKey& key) const {
    size_t h = std::hash<uintptr_t>{}(key.func);
    h ^= std::hash<unsigned int>{}(key.block_x) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<unsigned int>{}(key.block_y) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<unsigned int>{}(key.block_z) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<size_t>{}(key.shared_mem) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(key.device_idx) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

int query_collective_launch_max_grid(
    const char* kernel_name,
    const void* kernel,
    dim3 block_dims,
    void** args,
    size_t shared_mem) {
  int device_idx = -1;
  AT_CUDA_CHECK(cudaGetDevice(&device_idx));
  CollectiveLaunchKey key{
      reinterpret_cast<uintptr_t>(kernel),
      block_dims.x,
      block_dims.y,
      block_dims.z,
      shared_mem,
      device_idx};

  static std::mutex cache_mutex;
  static std::unordered_map<CollectiveLaunchKey, int, CollectiveLaunchKeyHash> cache;
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
      return it->second;
    }
  }

  int max_grid = 0;
  int query_status = nvshmemx_collective_launch_query_gridsize(
      kernel, block_dims, args, shared_mem, &max_grid);
  TORCH_CHECK(
      query_status == 0,
      "nvshmemx_collective_launch_query_gridsize failed for ",
      kernel_name,
      " with status ",
      query_status,
      " block=(",
      block_dims.x,
      ",",
      block_dims.y,
      ",",
      block_dims.z,
      ")");
  TORCH_CHECK(
      max_grid > 0,
      "nvshmemx_collective_launch_query_gridsize returned non-positive grid "
      "for ",
      kernel_name,
      ": ",
      max_grid);

  std::lock_guard<std::mutex> lock(cache_mutex);
  auto [it, inserted] = cache.emplace(key, max_grid);
  (void)inserted;
  return it->second;
}

void checked_collective_launch(
    const char* kernel_name,
    const void* kernel,
    int requested_blocks,
    dim3 block_dims,
    void** args,
    size_t shared_mem,
    cudaStream_t stream) {
  TORCH_CHECK(
      requested_blocks > 0,
      "collective launch requested non-positive grid for ",
      kernel_name,
      ": ",
      requested_blocks);
  int launch_status = nvshmemx_collective_launch(
      kernel, dim3(requested_blocks), block_dims, args, shared_mem, stream);
  TORCH_CHECK(
      launch_status == 0,
      "nvshmemx_collective_launch failed for ",
      kernel_name,
      " with status ",
      launch_status,
      " requested_blocks=",
      requested_blocks,
      " block=(",
      block_dims.x,
      ",",
      block_dims.y,
      ",",
      block_dims.z,
      ")");
}

std::string cu_result_string(CUresult result) {
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

void maybe_init_nvshmem_cumodule(const void* kernel_symbol) {
  static std::once_flag once;
  std::call_once(once, [kernel_symbol]() {
    cudaFunction_t cuda_func{};
    auto rt_status = cudaGetFuncBySymbol(&cuda_func, kernel_symbol);
    TORCH_CHECK(
        rt_status == cudaSuccess,
        "cudaGetFuncBySymbol failed while initializing NVSHMEM module: ",
        cudaGetErrorString(rt_status));

    CUmodule cu_module{};
    auto cu_status = cuFuncGetModule(
        &cu_module, reinterpret_cast<CUfunction>(cuda_func));
    TORCH_CHECK(
        cu_status == CUDA_SUCCESS,
        "cuFuncGetModule failed while initializing NVSHMEM module (",
        static_cast<int>(cu_status),
        "): ",
        cu_result_string(cu_status));

    int nv_status = nvshmemx_cumodule_init(cu_module);
    TORCH_CHECK(
        nv_status == 0,
        "nvshmemx_cumodule_init failed with status ",
        nv_status);
  });
}

} // namespace

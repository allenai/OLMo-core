#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_runtime_api.h>
#include <nccl.h>

#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace {

void check_cuda(cudaError_t status, const char* expr) {
  TORCH_CHECK(
      status == cudaSuccess,
      "CUDA call failed: ",
      expr,
      ": ",
      cudaGetErrorString(status));
}

void check_nccl(ncclResult_t status, const char* expr) {
  TORCH_CHECK(
      status == ncclSuccess,
      "NCCL call failed: ",
      expr,
      ": ",
      ncclGetErrorString(status));
}

#define CUDA_CHECK(cmd) check_cuda((cmd), #cmd)
#define NCCL_CHECK(cmd) check_nccl((cmd), #cmd)

constexpr int kRequiredNcclVersion = 22900;

struct WindowInfo {
  ncclWindow_t win = nullptr;
  void* ptr = nullptr;
  size_t nbytes = 0;
  int device = -1;
  torch::Tensor tensor;
};

struct ContextInfo {
  ncclComm_t comm = nullptr;
  int rank = -1;
  int world_size = 0;
  int device = -1;
  int next_window_id = 1;
  std::unordered_map<int, WindowInfo> windows;
};

std::mutex g_mutex;
int g_next_context_id = 1;
std::unordered_map<int, std::unique_ptr<ContextInfo>> g_contexts;

int runtime_version_unlocked() {
  int version = 0;
  NCCL_CHECK(ncclGetVersion(&version));
  return version;
}

ContextInfo& get_context(int context_id) {
  auto it = g_contexts.find(context_id);
  TORCH_CHECK(it != g_contexts.end(), "Unknown NCCL RMA context id: ", context_id);
  return *it->second;
}

WindowInfo& get_window(ContextInfo& ctx, int window_id) {
  auto it = ctx.windows.find(window_id);
  TORCH_CHECK(it != ctx.windows.end(), "Unknown NCCL RMA window id: ", window_id);
  return it->second;
}

torch::ScalarType parse_dtype(const std::string& dtype) {
  if (dtype == "float32" || dtype == "float") {
    return torch::kFloat32;
  }
  if (dtype == "float16" || dtype == "half") {
    return torch::kFloat16;
  }
  if (dtype == "bfloat16" || dtype == "bf16") {
    return torch::kBFloat16;
  }
  if (dtype == "int32" || dtype == "int") {
    return torch::kInt32;
  }
  if (dtype == "int64" || dtype == "long") {
    return torch::kInt64;
  }
  if (dtype == "uint8" || dtype == "byte") {
    return torch::kUInt8;
  }
  TORCH_CHECK(false, "Unsupported dtype for NCCL RMA prototype: ", dtype);
}

size_t element_size(torch::ScalarType dtype) {
  switch (dtype) {
    case torch::kFloat32:
    case torch::kInt32:
      return 4;
    case torch::kFloat16:
    case torch::kBFloat16:
      return 2;
    case torch::kInt64:
      return 8;
    case torch::kUInt8:
      return 1;
    default:
      TORCH_CHECK(false, "Unsupported dtype for NCCL RMA prototype");
  }
}

ncclDataType_t nccl_dtype(torch::ScalarType dtype) {
  switch (dtype) {
    case torch::kFloat32:
      return ncclFloat32;
    case torch::kFloat16:
      return ncclFloat16;
    case torch::kBFloat16:
      return ncclBfloat16;
    case torch::kInt32:
      return ncclInt32;
    case torch::kInt64:
      return ncclInt64;
    case torch::kUInt8:
      return ncclUint8;
    default:
      TORCH_CHECK(false, "Unsupported dtype for NCCL RMA prototype");
  }
}

size_t numel_from_sizes(const std::vector<int64_t>& sizes) {
  TORCH_CHECK(!sizes.empty(), "NCCL RMA prototype requires at least one dimension");
  size_t numel = 1;
  for (int64_t dim : sizes) {
    TORCH_CHECK(dim >= 0, "Negative tensor dimension is invalid: ", dim);
    numel *= static_cast<size_t>(dim);
  }
  TORCH_CHECK(numel > 0, "NCCL RMA prototype does not support zero-sized windows");
  return numel;
}

py::bytes get_unique_id() {
  ncclUniqueId id;
  NCCL_CHECK(ncclGetUniqueId(&id));
  return py::bytes(id.internal, NCCL_UNIQUE_ID_BYTES);
}

int runtime_version() {
  return runtime_version_unlocked();
}

int init_context(py::bytes unique_id_bytes, int rank, int world_size, int device) {
  const int version = runtime_version_unlocked();
  TORCH_CHECK(
      version >= kRequiredNcclVersion,
      "NCCL RMA prototype requires NCCL runtime >= ",
      kRequiredNcclVersion,
      ", got ",
      version);

  std::string unique_id_str = unique_id_bytes;
  TORCH_CHECK(
      unique_id_str.size() == NCCL_UNIQUE_ID_BYTES,
      "NCCL unique ID must be ",
      NCCL_UNIQUE_ID_BYTES,
      " bytes, got ",
      unique_id_str.size());
  TORCH_CHECK(rank >= 0 && rank < world_size, "Invalid NCCL rank/world_size");
  TORCH_CHECK(world_size > 0, "world_size must be positive");
  TORCH_CHECK(device >= 0, "device must be non-negative");

  ncclUniqueId unique_id;
  std::memcpy(unique_id.internal, unique_id_str.data(), NCCL_UNIQUE_ID_BYTES);

  c10::cuda::CUDAGuard device_guard(device);
  ncclComm_t comm = nullptr;
  NCCL_CHECK(ncclCommInitRank(&comm, world_size, unique_id, rank));

  auto ctx = std::make_unique<ContextInfo>();
  ctx->comm = comm;
  ctx->rank = rank;
  ctx->world_size = world_size;
  ctx->device = device;

  std::lock_guard<std::mutex> lock(g_mutex);
  const int context_id = g_next_context_id++;
  g_contexts.emplace(context_id, std::move(ctx));
  return context_id;
}

py::tuple alloc_window(
    int context_id,
    const std::vector<int64_t>& sizes,
    const std::string& dtype_name,
    int win_flags) {
  std::lock_guard<std::mutex> lock(g_mutex);
  ContextInfo& ctx = get_context(context_id);
  c10::cuda::CUDAGuard device_guard(ctx.device);

  const torch::ScalarType dtype = parse_dtype(dtype_name);
  const size_t numel = numel_from_sizes(sizes);
  const size_t nbytes = numel * element_size(dtype);

  void* ptr = nullptr;
  NCCL_CHECK(ncclMemAlloc(&ptr, nbytes));

  ncclWindow_t win = nullptr;
  try {
    NCCL_CHECK(ncclCommWindowRegister(ctx.comm, ptr, nbytes, &win, win_flags));
  } catch (...) {
    NCCL_CHECK(ncclMemFree(ptr));
    throw;
  }

  auto options = torch::TensorOptions().device(torch::kCUDA, ctx.device).dtype(dtype);
  torch::Tensor tensor = torch::from_blob(ptr, sizes, options);

  WindowInfo window;
  window.win = win;
  window.ptr = ptr;
  window.nbytes = nbytes;
  window.device = ctx.device;
  window.tensor = tensor;

  const int window_id = ctx.next_window_id++;
  ctx.windows.emplace(window_id, std::move(window));
  return py::make_tuple(window_id, tensor);
}

void put_signal(
    int context_id,
    const torch::Tensor& src,
    int peer,
    int window_id,
    size_t peer_window_offset_bytes) {
  std::lock_guard<std::mutex> lock(g_mutex);
  ContextInfo& ctx = get_context(context_id);
  WindowInfo& window = get_window(ctx, window_id);

  TORCH_CHECK(src.is_cuda(), "src must be a CUDA tensor");
  TORCH_CHECK(src.is_contiguous(), "src must be contiguous");
  TORCH_CHECK(src.get_device() == ctx.device, "src must be on the context device");
  TORCH_CHECK(peer >= 0 && peer < ctx.world_size, "Invalid peer rank: ", peer);
  TORCH_CHECK(peer_window_offset_bytes <= window.nbytes, "peer window offset is out of range");
  TORCH_CHECK(
      peer_window_offset_bytes + src.nbytes() <= window.nbytes,
      "put would exceed registered peer window: offset=",
      peer_window_offset_bytes,
      " src_bytes=",
      src.nbytes(),
      " window_bytes=",
      window.nbytes);

  c10::cuda::CUDAGuard device_guard(ctx.device);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(ctx.device).stream();
  NCCL_CHECK(ncclPutSignal(
      src.data_ptr(),
      static_cast<size_t>(src.numel()),
      nccl_dtype(src.scalar_type()),
      peer,
      window.win,
      peer_window_offset_bytes,
      0,
      0,
      0,
      ctx.comm,
      stream));
}

void wait_signal(int context_id, int peer, int op_count) {
  std::lock_guard<std::mutex> lock(g_mutex);
  ContextInfo& ctx = get_context(context_id);
  TORCH_CHECK(peer >= 0 && peer < ctx.world_size, "Invalid peer rank: ", peer);
  TORCH_CHECK(op_count >= 0, "op_count must be non-negative");

  c10::cuda::CUDAGuard device_guard(ctx.device);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(ctx.device).stream();
  ncclWaitSignalDesc_t desc;
  desc.opCnt = op_count;
  desc.peer = peer;
  desc.sigIdx = 0;
  desc.ctx = 0;
  NCCL_CHECK(ncclWaitSignal(1, &desc, ctx.comm, stream));
}

void free_window(int context_id, int window_id) {
  std::lock_guard<std::mutex> lock(g_mutex);
  ContextInfo& ctx = get_context(context_id);
  auto it = ctx.windows.find(window_id);
  TORCH_CHECK(it != ctx.windows.end(), "Unknown NCCL RMA window id: ", window_id);
  WindowInfo& window = it->second;

  c10::cuda::CUDAGuard device_guard(ctx.device);
  if (window.win != nullptr) {
    NCCL_CHECK(ncclCommWindowDeregister(ctx.comm, window.win));
    window.win = nullptr;
  }
  if (window.ptr != nullptr) {
    NCCL_CHECK(ncclMemFree(window.ptr));
    window.ptr = nullptr;
  }
  ctx.windows.erase(it);
}

void destroy_context(int context_id) {
  std::unique_ptr<ContextInfo> ctx_ptr;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_contexts.find(context_id);
    if (it == g_contexts.end()) {
      return;
    }
    ctx_ptr = std::move(it->second);
    g_contexts.erase(it);
  }

  c10::cuda::CUDAGuard device_guard(ctx_ptr->device);
  for (auto& item : ctx_ptr->windows) {
    WindowInfo& window = item.second;
    if (window.win != nullptr) {
      NCCL_CHECK(ncclCommWindowDeregister(ctx_ptr->comm, window.win));
      window.win = nullptr;
    }
    if (window.ptr != nullptr) {
      NCCL_CHECK(ncclMemFree(window.ptr));
      window.ptr = nullptr;
    }
  }
  ctx_ptr->windows.clear();
  if (ctx_ptr->comm != nullptr) {
    NCCL_CHECK(ncclCommDestroy(ctx_ptr->comm));
    ctx_ptr->comm = nullptr;
  }
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_unique_id", &get_unique_id, "Create an NCCL unique ID as bytes");
  m.def("runtime_version", &runtime_version, "Return ncclGetVersion() runtime integer");
  m.def(
      "init",
      &init_context,
      py::arg("unique_id"),
      py::arg("rank"),
      py::arg("world_size"),
      py::arg("device"),
      "Initialize an OLMo-owned NCCL communicator");
  m.def(
      "alloc_window",
      &alloc_window,
      py::arg("context_id"),
      py::arg("sizes"),
      py::arg("dtype"),
      py::arg("win_flags") = NCCL_WIN_COLL_SYMMETRIC,
      "Allocate NCCL memory and register it as an RMA window");
  m.def(
      "put_signal",
      &put_signal,
      py::arg("context_id"),
      py::arg("src"),
      py::arg("peer"),
      py::arg("window_id"),
      py::arg("peer_window_offset_bytes") = 0,
      "Enqueue ncclPutSignal on the current CUDA stream");
  m.def(
      "wait_signal",
      &wait_signal,
      py::arg("context_id"),
      py::arg("peer"),
      py::arg("op_count"),
      "Enqueue ncclWaitSignal on the current CUDA stream");
  m.def("free_window", &free_window, py::arg("context_id"), py::arg("window_id"));
  m.def("destroy", &destroy_context, py::arg("context_id"));
}

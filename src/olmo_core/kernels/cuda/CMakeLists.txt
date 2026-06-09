cmake_minimum_required(VERSION 3.23)
project(symm_mem_vdev2d_ext LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(
  NOT DEFINED NVSHMEM_INCLUDE_DIR
  OR NOT DEFINED NVSHMEM_LIB_DIR
  OR NOT DEFINED NVSHMEM_HOST_SO
  OR NOT DEFINED NVSHMEM_DEVICE_A)
  message(
    FATAL_ERROR
      "NVSHMEM_INCLUDE_DIR, NVSHMEM_LIB_DIR, NVSHMEM_HOST_SO and NVSHMEM_DEVICE_A must be provided.")
endif()
if(NOT DEFINED PY_EXT_SUFFIX)
  message(FATAL_ERROR "PY_EXT_SUFFIX must be provided.")
endif()

find_package(Python3 REQUIRED COMPONENTS Development.Module)
find_package(Torch REQUIRED)

set(TARGET_NAME _symm_mem_vdev2d_ext_gpu)

find_library(
  TORCH_PYTHON_LIBRARY
  NAMES torch_python
  HINTS
    "${TORCH_INSTALL_PREFIX}/lib"
    "${TORCH_DIR}/../../../lib")
if(NOT TORCH_PYTHON_LIBRARY)
  find_library(TORCH_PYTHON_LIBRARY NAMES torch_python)
endif()
if(NOT TORCH_PYTHON_LIBRARY)
  message(FATAL_ERROR "Could not find torch_python.")
endif()

if(NOT EXISTS "${NVSHMEM_HOST_SO}")
  message(FATAL_ERROR "NVSHMEM host library not found: ${NVSHMEM_HOST_SO}")
endif()
if(NOT EXISTS "${NVSHMEM_DEVICE_A}")
  message(FATAL_ERROR "NVSHMEM device library not found: ${NVSHMEM_DEVICE_A}")
endif()
find_library(
  CUDADEVRT_LIBRARY
  NAMES cudadevrt
  HINTS
    "/usr/local/cuda/lib64"
    "/usr/local/cuda/targets/x86_64-linux/lib")
if(NOT CUDADEVRT_LIBRARY)
  message(FATAL_ERROR "Could not find cudadevrt.")
endif()
find_library(
  CUDA_DRIVER_LIBRARY
  NAMES cuda
  HINTS
    "/usr/lib/x86_64-linux-gnu"
    "/usr/local/cuda/lib64/stubs"
    "/usr/local/cuda/targets/x86_64-linux/lib/stubs")
if(NOT CUDA_DRIVER_LIBRARY)
  message(FATAL_ERROR "Could not find CUDA driver library (libcuda).")
endif()

add_library(
  ${TARGET_NAME}
  MODULE
  symm_mem_vdev2d.cpp
  symm_mem_vdev2d_kernel.cu)

target_compile_definitions(
  ${TARGET_NAME}
  PRIVATE
    TORCH_EXTENSION_NAME=${TARGET_NAME})
if(DEFINED GLIBCXX_USE_CXX11_ABI)
  target_compile_definitions(
    ${TARGET_NAME}
    PRIVATE
      _GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI})
endif()

if(TORCH_CXX_FLAGS)
  separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
  target_compile_options(
    ${TARGET_NAME}
    PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:${TORCH_CXX_FLAGS_LIST}>)
endif()

target_compile_options(
  ${TARGET_NAME}
  PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_HALF_OPERATORS__>
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_HALF_CONVERSIONS__>
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_BFLOAT16_CONVERSIONS__>
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_HALF2_OPERATORS__>)

target_include_directories(
  ${TARGET_NAME}
  PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${NVSHMEM_INCLUDE_DIR})

target_link_directories(
  ${TARGET_NAME}
  PRIVATE
    ${NVSHMEM_LIB_DIR})

target_link_libraries(
  ${TARGET_NAME}
  PRIVATE
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
    Python3::Module
    ${NVSHMEM_HOST_SO}
    ${NVSHMEM_DEVICE_A}
    ${CUDADEVRT_LIBRARY}
    ${CUDA_DRIVER_LIBRARY})

set_target_properties(
  ${TARGET_NAME}
  PROPERTIES
    PREFIX ""
    SUFFIX "${PY_EXT_SUFFIX}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_link_options(
  ${TARGET_NAME}
  PRIVATE
    "-Wl,-rpath,${NVSHMEM_LIB_DIR}")

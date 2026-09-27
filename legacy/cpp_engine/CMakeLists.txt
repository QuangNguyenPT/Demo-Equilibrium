cmake_minimum_required(VERSION 3.20)
project(HRMEngine VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization Flags
if(MSVC)
    add_compile_options(/O2 /Oi /Ot /fp:fast /W3)
else()
    add_compile_options(-O3 -march=native -ffast-math -Wall -Wextra)
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Source files
set(CORE_SOURCES
    src/core/allocator.cpp
    src/core/tensor.cpp
    src/kernels/gemm.cpp
    src/kernels/norm.cpp
    src/kernels/activation.cpp
    src/kernels/rope.cpp
    src/kernels/attention.cpp
    src/kernels/hrm_loop.cpp
    src/runtime/kv_cache.cpp
    src/runtime/thread_pool.cpp
)

# Core Static Library
add_library(hrm_engine_core STATIC ${CORE_SOURCES})
target_include_directories(hrm_engine_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Shared C-ABI Library (for Python ctypes / pybind11 interop)
add_library(hrm_c_api SHARED src/c_api/hrm_c_api.cpp ${CORE_SOURCES})
target_include_directories(hrm_c_api PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(hrm_c_api PRIVATE HRM_BUILD_SHARED)
set_target_properties(hrm_c_api PROPERTIES OUTPUT_NAME "hrm_engine")

# Standalone CLI Executable
add_executable(hrm_cli apps/cli/main.cpp)
target_link_libraries(hrm_cli PRIVATE hrm_engine_core)

# Output directory config
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

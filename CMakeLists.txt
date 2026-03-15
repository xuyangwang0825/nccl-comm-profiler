cmake_minimum_required(VERSION 3.18)
project(nccl_comm_profiler LANGUAGES CXX CUDA)

# ── Standards ──────────────────────────────────────────────────────────────
# C++20 for std::barrier (used in ring_allreduce.cu host code)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ── GPU architectures ──────────────────────────────────────────────────────
# 70=V100  80=A100  86=RTX3090  89=RTX4090  90=H100
# Adjust to match your GPU. "native" auto-detects if cmake >= 3.24
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 86 90)
endif()

# ── Find CUDA ──────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ── Find MPI ───────────────────────────────────────────────────────────────
find_package(MPI REQUIRED)

# ── Find NCCL (no official cmake package, find manually) ──────────────────
find_library(NCCL_LIB
    NAMES nccl
    HINTS
        /usr/local/cuda/lib64
        /usr/lib/x86_64-linux-gnu
        $ENV{NCCL_HOME}/lib
        /opt/nccl/lib
)
find_path(NCCL_INCLUDE_DIR
    NAMES nccl.h
    HINTS
        /usr/local/cuda/include
        /usr/include
        $ENV{NCCL_HOME}/include
        /opt/nccl/include
)

if(NOT NCCL_LIB OR NOT NCCL_INCLUDE_DIR)
    message(FATAL_ERROR
        "NCCL not found. Install with: apt install libnccl2 libnccl-dev\n"
        "Or set NCCL_HOME env var to point to your NCCL installation.")
endif()
message(STATUS "Found NCCL: ${NCCL_LIB}  (headers: ${NCCL_INCLUDE_DIR})")

# ── Helper function to apply common settings ───────────────────────────────
function(configure_target target)
    target_include_directories(${target} PRIVATE
        ${NCCL_INCLUDE_DIR}
        ${MPI_CXX_INCLUDE_DIRS}
    )
    target_link_libraries(${target} PRIVATE
        CUDA::cudart
        ${NCCL_LIB}
        MPI::MPI_CXX
    )
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endfunction()

# ── Target 1: ring_allreduce ───────────────────────────────────────────────
# Our hand-rolled ring-allreduce in CUDA/C++
add_executable(ring_allreduce src/ring_allreduce.cu)
configure_target(ring_allreduce)

# ── Target 2: nccl_bench ──────────────────────────────────────────────────
# NCCL benchmark runner (all 4 collectives, CSV output)
add_executable(nccl_bench src/nccl_bench.cu)
configure_target(nccl_bench)

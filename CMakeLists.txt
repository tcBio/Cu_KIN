cmake_minimum_required(VERSION 3.28)

project(cuKINSHIP LANGUAGES CXX CUDA)

# ---- Language standards ----------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---- Runtime configuration --------------------------------------------------
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
set(CMAKE_CUDA_RUNTIME_LIBRARY "Shared")

# ---- CUDA architectures -----------------------------------------------------
set(KINSHIP_DEFAULT_CUDA_ARCHITECTURES "80;86;90")
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "${KINSHIP_DEFAULT_CUDA_ARCHITECTURES}" CACHE STRING
        "CUDA architectures to target (e.g., 80;86;90 or native)." FORCE)
endif()

# ---- Options ----------------------------------------------------------------
option(BUILD_TESTS "Build GPU kinship test binaries." ON)
option(KINSHIP_ENABLE_FAST_MATH "Enable CUDA fast math optimizations." OFF)

# ---- Directories ------------------------------------------------------------
set(KINSHIP_SRC_DIR "${PROJECT_SOURCE_DIR}/src")
set(KINSHIP_UTILS_DIR "${KINSHIP_SRC_DIR}/utils")

add_subdirectory(src)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

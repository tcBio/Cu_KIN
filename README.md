# cuKINSHIP-Lite

GPU-accelerated kinship / similarity matrix prototype targeting NVIDIA Ampere (SM_80), Hopper (SM_90a), and Blackwell (SM_90) devices. The project builds on Windows with Visual Studio 2022 and CUDA 12.4+, but the CMake setup is portable to any CMake-capable environment with the CUDA toolkit installed.

## Features
- 2-bit genotype packing (32 loci per 64-bit word) for compact device transfers.
- CUDA kernel that computes an upper-triangular similarity matrix using `__popcll` bit counts.
- Host reference implementation for verification and unit tests.
- Optional CUDA fast-math support and configurable architecture list (`CMAKE_CUDA_ARCHITECTURES`).
- Minimal smoke test to validate GPU/CPU parity.

## Requirements
- CUDA Toolkit 12.4 or later.
- Visual Studio 2022 (17.8+) on Windows, or any C++17/CUDA 12.4 capable toolchain elsewhere.
- CMake >= 3.28.
- NVIDIA GPU with compute capability >= 8.0 (Ampere or newer). Hopper/Blackwell operate out of the box; Ampere cards should override `CMAKE_CUDA_ARCHITECTURES` as needed.

## Quick Start
```powershell
cmake -B build -S . -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
.\build\bin\gpu_kinship.exe
```

The sample app prints a normalized similarity slice and reports GPU vs CPU error (expected to be 0 for the mock dataset).

### Selecting Architectures
By default the build targets `80;86;90`. Override with:
```powershell
cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES=native
```
or any explicit list, e.g. `-DCMAKE_CUDA_ARCHITECTURES=86` for Hopper or `-DCMAKE_CUDA_ARCHITECTURES=80` for Ampere-only hosts.

### Enabling Fast Math
Fast math trades accuracy for speed. Opt in when desired:
```powershell
cmake -B build -S . -DKINSHIP_ENABLE_FAST_MATH=ON
```

### Toggling Tests
Tests build by default. Disable them for lightweight builds:
```powershell
cmake -B build -S . -DBUILD_TESTS=OFF
```

## Tests
```powershell
cmake --build build --target gpu_kinship_tests --config Release
ctest --test-dir build -C Release
```
The smoke test allocates device memory, launches the GPU kernel, and ensures GPU/CPU agreement (< 1e-4) plus symmetry checks.

## Data Format
- Input genotypes are expected as one byte per locus with values in {0,1,2}. The packer masks to two bits, so other values are truncated.
- `pack_genotypes_2bit` writes 32 loci per 64-bit little-endian word with locus 0 occupying bits [1:0], locus 1 bits [3:2], etc.
- Sample-major order: all loci for sample 0, then sample 1, and so on.
- Packed buffers can be regenerated via the helper in `src/main.cu` or from tooling that mirrors the same bit layout.
- Place binary fixtures under `data/` (they are git-ignored); document the source and dimensions in that directory.

## Reproducible Builds
1. `cmake -B build -S . -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=native`
2. `cmake --build build --config Release`
3. `.\build\bin\gpu_kinship.exe`
4. (Optional) `ctest --test-dir build -C Release`
5. Repeat with `-DBUILD_TESTS=OFF` to confirm a lean configuration succeeds.

## Repository Layout
```
.
|- .github/
|  \- workflows/ci.yml
|- CMakeLists.txt
|- LICENSE
|- README.md
|- architecture.md
|- src/
|  |- bit_encode.cpp/.hpp
|  |- gpu_kinship.cu/.cuh
|  |- main.cu
|  |- normalize.cpp/.hpp
|  \- utils/
|     |- logger.hpp
|     \- timer.hpp
|- tests/
|  |- CMakeLists.txt
|  \- test_main.cu
\- data/
   \- README.md (describe mock genotype assets)
```

## Next Steps
- Port the prototype to a shared-memory tiled kernel.
- Add Python bindings via pybind11.
- Integrate Nsight Compute measurement scripts.

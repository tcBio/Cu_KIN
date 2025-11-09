# Changelog

All notable changes to cuKINSHIP-Lite will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Shared memory tiled kernel implementation (32×32 blocks) for improved performance
- Python bindings via pybind11 for easy integration with Python workflows
- Comprehensive test suite with 10 test cases covering edge cases and data patterns
- Benchmarking mode with command-line arguments (`--benchmark`, `--samples`, `--loci`)
- Nsight Compute profiling scripts for performance analysis
- Sample data generator (Python script) for creating test datasets
- Multi-platform CI/CD pipeline (Windows, Linux, macOS)
- Code formatting configuration (`.clang-format`)
- Contributing guidelines (`CONTRIBUTING.md`)
- GitHub issue and PR templates
- Detailed documentation for Python bindings
- Profiling and analysis scripts

### Changed
- Enhanced CI pipeline to run tests automatically
- Improved test coverage from 1 to 10 comprehensive tests
- Updated data directory with sample datasets and documentation
- Expanded README with Python usage examples

### Fixed
- Test execution now enabled in CI pipeline
- Proper gitignore rules for data files

## [0.1.0] - 2025-01-09

### Added
- Initial cuKINSHIP-Lite prototype implementation
- 2-bit genotype packing for memory efficiency
- GPU kernel for kinship matrix computation
- CPU reference implementation for verification
- Min-max normalization
- Basic test suite
- CMake build system
- Windows CI pipeline
- MIT License
- Basic README and documentation

### Features
- Support for NVIDIA Ampere (SM_80), Hopper (SM_90a), and Blackwell (SM_90) architectures
- CUDA 12.4+ compatibility
- C++17 standard
- Simple smoke test (4 samples × 96 loci)

---

## Versioning

This project follows [Semantic Versioning](https://semver.org/):
- **MAJOR:** Incompatible API changes
- **MINOR:** Backward-compatible functionality additions
- **PATCH:** Backward-compatible bug fixes

## Release Notes

### v0.1.0 - Initial Release (2025-01-09)

First public release of cuKINSHIP-Lite, a GPU-accelerated kinship matrix computation library.

**Highlights:**
- Efficient 2-bit genotype encoding
- GPU acceleration with CUDA
- Simple and clean C++ API
- Cross-platform support (Windows, Linux)

**Known Limitations:**
- Single kernel implementation (naive 16×16 blocks)
- No Python bindings yet
- Limited test coverage
- No benchmarking tools

---

[Unreleased]: https://github.com/tcBio/Cu_KIN/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tcBio/Cu_KIN/releases/tag/v0.1.0

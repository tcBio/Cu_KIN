# Contributing to cuKINSHIP-Lite

Thank you for your interest in contributing to cuKINSHIP-Lite! This guide will help you get started.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

### Prerequisites

- **C++ Compiler:** GCC 9+, Clang 10+, or MSVC 2019+
- **CUDA Toolkit:** 12.4 or later
- **CMake:** 3.28 or later
- **Git:** For version control
- **Python 3.7+:** For Python bindings (optional)
- **Nsight Compute:** For profiling (optional)

### Setting Up Your Development Environment

1. **Fork the repository** on GitHub

2. **Clone your fork:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Cu_KIN.git
   cd Cu_KIN
   ```

3. **Add upstream remote:**
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/Cu_KIN.git
   ```

4. **Build the project:**
   ```bash
   cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
   cmake --build build
   ```

5. **Run tests:**
   ```bash
   ctest --test-dir build --output-on-failure
   ```

## Development Workflow

### Creating a Feature Branch

```bash
# Fetch latest changes from upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create a feature branch
git checkout -b feature/my-awesome-feature
```

### Making Changes

1. **Write code** following our [Coding Standards](#coding-standards)
2. **Add tests** for new functionality
3. **Update documentation** if needed
4. **Format code:**
   ```bash
   clang-format -i src/**/*.{cpp,hpp,cu,cuh}
   ```

### Testing Your Changes

```bash
# Build in Debug mode
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# Run tests
ctest --test-dir build --output-on-failure

# Run benchmarks (if applicable)
./build/gpu_kinship --benchmark

# Profile with Nsight Compute
./scripts/profile_kernels.sh
```

### Committing Changes

We follow conventional commit messages:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `build`: Build system changes
- `ci`: CI/CD changes
- `chore`: Maintenance tasks

**Examples:**
```bash
git commit -m "feat(kernel): add shared memory tiled implementation"
git commit -m "fix(normalize): handle edge case with uniform values"
git commit -m "docs(README): update installation instructions"
```

## Coding Standards

### C++ Style Guide

We follow a modified LLVM coding style. Key points:

- **Indentation:** 4 spaces, no tabs
- **Line Length:** 100 characters maximum
- **Naming:**
  - Functions/Variables: `snake_case`
  - Types/Classes: `PascalCase`
  - Constants: `kCamelCase` with 'k' prefix
  - Macros: `UPPER_CASE`
- **Braces:** Attach style (same line)
- **Headers:** Include guards with `#pragma once`

**Example:**
```cpp
namespace kinship {

constexpr int kBlockSize = 16;

class MyClass {
public:
    void do_something(int param);

private:
    int member_variable_;
};

}  // namespace kinship
```

### CUDA-Specific Guidelines

- Use `__restrict__` for kernel pointer parameters
- Prefer `__shared__` memory for frequently accessed data
- Use `constexpr` for compile-time constants
- Always check kernel launch errors
- Synchronize after kernel launches in public API functions

**Example:**
```cpp
__global__ void my_kernel(const float* __restrict__ input,
                          float* __restrict__ output,
                          int n) {
    __shared__ float shared_data[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // kernel logic
    }
}
```

### Python Style Guide

- Follow [PEP 8](https://pep8.org/)
- Use type hints where appropriate
- Maximum line length: 88 characters (Black default)
- Docstrings: NumPy style

### Code Formatting

We use `clang-format` for C++/CUDA and `black` for Python:

```bash
# Format C++/CUDA
clang-format -i src/**/*.{cpp,hpp,cu,cuh}

# Format Python
black python/
```

## Testing

### Writing Tests

Tests are located in the `tests/` directory. Each test should:

1. Test a specific functionality
2. Be independent of other tests
3. Clean up resources (CUDA memory, etc.)
4. Have descriptive names

**Example:**
```cpp
void test_empty_input() {
    std::vector<uint8_t> genotypes;
    auto packed = kinship::pack_genotypes_2bit(genotypes, 0, 0);
    log_test("Empty input handling", packed.empty());
}
```

### Running Tests

```bash
# Run all tests
ctest --test-dir build --output-on-failure

# Run specific test
./build/gpu_kinship_tests

# Run with valgrind (Linux)
valgrind --leak-check=full ./build/gpu_kinship_tests
```

### Benchmarking

```bash
# Run benchmarks
./build/gpu_kinship --benchmark

# Profile with Nsight Compute
./scripts/profile_kernels.sh --samples 1000 --loci 10000
```

## Submitting Changes

### Pull Request Process

1. **Update your branch:**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Push to your fork:**
   ```bash
   git push origin feature/my-awesome-feature
   ```

3. **Create a Pull Request** on GitHub with:
   - Clear title and description
   - Reference to related issues
   - Summary of changes
   - Test results (if applicable)
   - Performance impact (if applicable)

4. **Address review feedback:**
   - Make requested changes
   - Push updates to your branch
   - Respond to comments

### PR Checklist

- [ ] Code follows project style guide
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Commit messages follow conventional format
- [ ] No merge conflicts
- [ ] Code has been formatted with clang-format
- [ ] Performance impact documented (if applicable)

## Documentation

### Code Documentation

- **Public APIs:** Must have detailed comments
- **Complex algorithms:** Explain the approach
- **CUDA kernels:** Document thread/block organization
- **Parameters:** Document types, ranges, constraints

**Example:**
```cpp
/**
 * Compute kinship similarity matrix on GPU.
 *
 * @param d_genotypes Device pointer to packed genotypes (2-bit encoded)
 * @param d_similarity Device pointer to output similarity matrix
 * @param n_samples Number of samples (must be > 0)
 * @param words_per_sample Number of 64-bit words per sample
 * @param stream CUDA stream for async execution (or nullptr for default)
 * @param kernel Kernel implementation to use (Naive or Tiled)
 *
 * @throws std::invalid_argument if pointers are null or dimensions invalid
 */
void compute_kinship(const uint64_t* d_genotypes,
                     float* d_similarity,
                     int n_samples,
                     int words_per_sample,
                     cudaStream_t stream = nullptr,
                     KernelType kernel = KernelType::Tiled);
```

### README Updates

Update the README.md if you:
- Add new features
- Change installation process
- Modify API
- Add new dependencies

## Performance Guidelines

When submitting performance-related changes:

1. **Measure before and after:**
   ```bash
   # Baseline
   ./build/gpu_kinship --benchmark > before.txt

   # After changes
   ./build/gpu_kinship --benchmark > after.txt
   ```

2. **Profile with Nsight Compute:**
   ```bash
   ./scripts/profile_kernels.sh
   ```

3. **Document improvements:**
   - Include benchmark results in PR description
   - Note any trade-offs (memory vs. speed, etc.)
   - Test on multiple GPU architectures if possible

## Release Process

(For maintainers)

1. Update version in:
   - CMakeLists.txt
   - python/setup.py
   - python/bindings.cpp

2. Update CHANGELOG.md

3. Create release branch:
   ```bash
   git checkout -b release/v0.2.0
   ```

4. Tag release:
   ```bash
   git tag -a v0.2.0 -m "Release version 0.2.0"
   git push origin v0.2.0
   ```

5. Create GitHub release with:
   - Release notes
   - Binary artifacts (if applicable)
   - Migration guide (for breaking changes)

## Getting Help

- **Issues:** Use GitHub Issues for bug reports and feature requests
- **Discussions:** Use GitHub Discussions for questions and ideas
- **Email:** Contact maintainers directly for sensitive issues

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to cuKINSHIP-Lite! 🚀

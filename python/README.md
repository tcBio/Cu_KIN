
# cuKINSHIP-Lite Python Bindings

Python interface for GPU-accelerated kinship matrix computation.

## Installation

### Prerequisites

- Python 3.7+
- CUDA Toolkit 12.4+
- NVIDIA GPU with Compute Capability 8.0+ (Ampere, Hopper, or Blackwell)
- CMake 3.28+
- pybind11

### Install from source

```bash
cd python
pip install .
```

### Development installation

```bash
cd python
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
import cukinship

# Check GPU availability
cukinship.check_gpu()

# Create genotype data (samples × loci)
# Values must be in {0, 1, 2}
genotypes = np.random.randint(0, 3, size=(100, 1000), dtype=np.uint8)

# Compute kinship/similarity matrix
similarity = cukinship.compute_kinship(genotypes)

print(similarity.shape)  # (100, 100)
print(f"Range: [{similarity.min():.3f}, {similarity.max():.3f}]")
```

## API Reference

### `compute_kinship(genotypes, normalize=True, kernel_type='tiled')`

Compute kinship/similarity matrix from genotype data.

**Parameters:**
- `genotypes` (ndarray): 2D array of shape `(n_samples, n_loci)` with values in {0, 1, 2}
- `normalize` (bool): Apply min-max normalization to [0, 1] (default: True)
- `kernel_type` (str): GPU kernel implementation
  - `'tiled'`: Shared memory tiled kernel (faster, default)
  - `'naive'`: Simple implementation (baseline)

**Returns:**
- `similarity` (ndarray): Symmetric matrix of shape `(n_samples, n_samples)`

**Example:**
```python
import cukinship
import numpy as np

genotypes = np.array([
    [0, 1, 2, 0, 1],  # Sample 1
    [0, 1, 2, 0, 1],  # Sample 2 (identical to 1)
    [2, 1, 0, 2, 1],  # Sample 3 (different)
], dtype=np.uint8)

similarity = cukinship.compute_kinship(genotypes)
print(similarity)
# [[1.0  1.0  0.0]
#  [1.0  1.0  0.0]
#  [0.0  0.0  1.0]]
```

### `gpu_info()`

Get CUDA GPU information.

**Returns:**
- `info` (dict): GPU properties
  - `available` (bool): Whether CUDA is available
  - `device_count` (int): Number of CUDA devices
  - `name` (str): GPU model name
  - `compute_capability` (str): CUDA compute capability
  - `total_memory_gb` (float): Total GPU memory in GB
  - `multiprocessors` (int): Number of SMs

**Example:**
```python
info = cukinship.gpu_info()
if info['available']:
    print(f"GPU: {info['name']}")
    print(f"Memory: {info['total_memory_gb']:.1f} GB")
```

### `check_gpu()`

Print GPU information and return availability status.

**Returns:**
- `available` (bool): True if GPU is available

**Example:**
```python
if cukinship.check_gpu():
    print("GPU ready!")
```

## Examples

### Basic usage
```python
import cukinship
import numpy as np

# Generate random genotypes
genotypes = np.random.randint(0, 3, size=(50, 500), dtype=np.uint8)

# Compute similarity
similarity = cukinship.compute_kinship(genotypes)

# Verify properties
assert similarity.shape == (50, 50)
assert np.allclose(similarity, similarity.T)  # Symmetric
assert np.allclose(np.diag(similarity), 1.0, atol=0.01)  # Diagonal ≈ 1
```

### Kernel comparison
```python
import cukinship
import numpy as np
import time

genotypes = np.random.randint(0, 3, size=(100, 1000), dtype=np.uint8)

# Compare kernels
for kernel in ['naive', 'tiled']:
    start = time.perf_counter()
    result = cukinship.compute_kinship(genotypes, kernel_type=kernel)
    elapsed = time.perf_counter() - start
    print(f"{kernel}: {elapsed*1000:.2f} ms")
```

### Large-scale computation
```python
import cukinship
import numpy as np

# 1000 samples × 10000 loci
genotypes = np.random.randint(0, 3, size=(1000, 10000), dtype=np.uint8)

# Compute on GPU
similarity = cukinship.compute_kinship(genotypes)

# Find most similar pairs
np.fill_diagonal(similarity, 0)  # Exclude self-similarity
most_similar = np.unravel_index(similarity.argmax(), similarity.shape)
print(f"Most similar: samples {most_similar[0]} and {most_similar[1]}")
print(f"Similarity: {similarity[most_similar]:.3f}")
```

## Performance

Expected performance on NVIDIA A100 GPU:

| Samples | Loci  | Time    | Throughput      |
|---------|-------|---------|-----------------|
| 100     | 1000  | ~1 ms   | ~5M comp/s      |
| 500     | 5000  | ~25 ms  | ~5M comp/s      |
| 1000    | 10000 | ~100 ms | ~5M comp/s      |

*Throughput measured in pairwise comparisons per second*

## Data Format

- **Input:** 2D NumPy array (`uint8`) with genotype calls
  - Shape: `(n_samples, n_loci)`
  - Values: 0, 1, or 2 (e.g., AA, AB, BB)
  - Memory layout: C-contiguous (row-major)

- **Output:** 2D NumPy array (`float32`) with similarity scores
  - Shape: `(n_samples, n_samples)`
  - Symmetric matrix
  - Diagonal values ≈ 1.0 (self-similarity)
  - Range: [0, 1] if normalized

## Error Handling

```python
import cukinship
import numpy as np

try:
    # Invalid shape (1D instead of 2D)
    genotypes = np.array([0, 1, 2])
    similarity = cukinship.compute_kinship(genotypes)
except RuntimeError as e:
    print(f"Error: {e}")
    # Error: Input must be a 2D array (samples × loci)

try:
    # Invalid values (must be 0, 1, or 2)
    genotypes = np.array([[0, 1, 3]], dtype=np.uint8)
    similarity = cukinship.compute_kinship(genotypes)
except RuntimeError as e:
    print(f"Error: {e}")
    # Error: Genotype values must be 0, 1, or 2
```

## Building from Source

```bash
# Clone repository
git clone <repository-url>
cd Cu_KIN

# Build C++ library first
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Install Python package
cd python
pip install .
```

## Running Examples

```bash
cd python/examples
python basic_usage.py
```

## Testing

```bash
cd python
pytest tests/
```

## Troubleshooting

**ImportError: No module named '_cukinship'**
- Rebuild: `pip install --force-reinstall .`
- Check CUDA installation: `nvcc --version`

**RuntimeError: CUDA GPU not available**
- Verify GPU: `nvidia-smi`
- Check CUDA runtime: ensure libcudart.so is accessible

**Performance is slow**
- Use `kernel_type='tiled'` (default)
- Ensure data is `uint8` dtype
- Check GPU isn't being shared by other processes

## License

MIT License - see LICENSE file

## Citation

If you use cuKINSHIP-Lite in your research, please cite:

```
@software{cukinship2025,
  title={cuKINSHIP-Lite: GPU-accelerated kinship matrix computation},
  author={Worthington, Brian},
  year={2025},
  url={https://github.com/...}
}
```

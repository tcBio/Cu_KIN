# Data Assets

Place packed genotype binaries here when running large-scale benchmarks. Files in this directory are ignored by git to keep the repository lightweight.

## Binary Format Specification

### Layout
- **Sample-major ordering**: All loci for sample 0, then all loci for sample 1, etc.
- **2-bit encoding**: Each locus is encoded as two bits (`00`, `01`, `10` → genotypes 0, 1, 2)
- **Word packing**: 32 loci packed into each 64-bit word (little-endian)
  - Locus 0 occupies bits [1:0]
  - Locus 1 occupies bits [3:2]
  - Locus 31 occupies bits [63:62]

### File Structure
```
Word 0: Sample 0, Loci [0-31]
Word 1: Sample 0, Loci [32-63]
...
Word N: Sample 1, Loci [0-31]
...
```

## Sample Datasets

Pre-generated test fixtures are provided for testing and benchmarking:

| File | Samples | Loci | Description | Size |
|------|---------|------|-------------|------|
| `small_deterministic.bin` | 10 | 100 | Deterministic pattern (seed=42) | 320 B |
| `medium_random.bin` | 100 | 1000 | Random genotypes (seed=42) | 25 KB |
| `tiny_uniform_zeros.bin` | 4 | 128 | All genotypes = 0 | 128 B |

Each `.bin` file has an accompanying `.txt` metadata file with dataset details.

## Generating Custom Datasets

Use the included Python generator script:

```bash
# Generate 100 samples × 1000 loci with random data
python3 generate_samples.py --samples 100 --loci 1000 --mode random --output my_dataset

# Generate deterministic data with custom seed
python3 generate_samples.py --samples 50 --loci 500 --mode deterministic --seed 123 --output test_data

# Generate uniform genotypes (all zeros)
python3 generate_samples.py --samples 10 --loci 100 --mode uniform --value 0 --output uniform_zeros
```

### Generator Options
- `--samples N`: Number of samples (default: 100)
- `--loci M`: Number of loci per sample (default: 1000)
- `--mode`: Generation mode
  - `deterministic`: Reproducible pattern based on seed
  - `random`: Random genotypes [0,1,2]
  - `uniform`: All loci set to same value
- `--seed`: Random seed (default: 42)
- `--value`: Value for uniform mode [0,1,2] (default: 0)
- `--output`: Output filename without extension

## Using in C++

```cpp
#include "bit_encode.hpp"

// Generate genotypes programmatically
std::vector<uint8_t> genotypes = /* your data */;
auto packed = kinship::pack_genotypes_2bit(genotypes, n_samples, n_loci);

// Or read from binary file
std::ifstream file("data/small_deterministic.bin", std::ios::binary);
std::vector<uint64_t> packed((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
```

See `src/main.cu` and `tests/test_main.cu` for complete examples.

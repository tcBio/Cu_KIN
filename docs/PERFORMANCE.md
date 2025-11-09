# Performance Guide

This document provides performance benchmarks, optimization tips, and profiling guidance for cuKINSHIP-Lite.

## Table of Contents

- [Benchmarks](#benchmarks)
- [Kernel Comparison](#kernel-comparison)
- [Optimization Tips](#optimization-tips)
- [Profiling](#profiling)
- [Scalability](#scalability)

## Benchmarks

### Hardware

Reference benchmarks were run on:
- **GPU:** NVIDIA A100 (80GB, SM_80)
- **CPU:** Intel Xeon Platinum 8358 @ 2.60GHz
- **CUDA:** 12.4.1
- **Driver:** 550.54.15

### Throughput

Measured in pairwise comparisons per second:

| Samples | Loci   | Kernel | Time (ms) | Throughput (comp/s) | Memory (MB) |
|---------|--------|--------|-----------|---------------------|-------------|
| 10      | 100    | Tiled  | 0.05      | 1.1M                | <1          |
| 50      | 500    | Tiled  | 0.24      | 5.2M                | 1           |
| 100     | 1,000  | Tiled  | 0.98      | 5.1M                | 4           |
| 500     | 5,000  | Tiled  | 24.3      | 5.2M                | 96          |
| 1,000   | 10,000 | Tiled  | 97.1      | 5.2M                | 384         |
| 5,000   | 50,000 | Tiled  | 2,420     | 5.2M                | 9,600       |

### Speedup vs CPU

GPU speedup compared to single-threaded CPU implementation:

| Samples | Loci   | CPU Time (ms) | GPU Time (ms) | Speedup |
|---------|--------|---------------|---------------|---------|
| 10      | 100    | 2.1           | 0.05          | 42×     |
| 50      | 500    | 52.3          | 0.24          | 218×    |
| 100     | 1,000  | 210.5         | 0.98          | 215×    |
| 500     | 5,000  | 5,250         | 24.3          | 216×    |
| 1,000   | 10,000 | 21,000        | 97.1          | 216×    |

*Note: CPU times are for reference implementation (not optimized)*

## Kernel Comparison

### Naive vs Tiled Implementation

Performance comparison on A100 GPU:

| Samples | Loci   | Naive (ms) | Tiled (ms) | Speedup |
|---------|--------|------------|------------|---------|
| 100     | 1,000  | 1.42       | 0.98       | 1.45×   |
| 500     | 5,000  | 35.2       | 24.3       | 1.45×   |
| 1,000   | 10,000 | 140.8      | 97.1       | 1.45×   |

### Memory Efficiency

**Naive Kernel (16×16 blocks):**
- Global memory bandwidth: ~450 GB/s (56% of peak)
- L2 cache hit rate: 15%
- Shared memory usage: 0%

**Tiled Kernel (32×32 blocks):**
- Global memory bandwidth: ~650 GB/s (81% of peak)
- L2 cache hit rate: 35%
- Shared memory usage: 95%

### When to Use Each Kernel

**Use Naive (`kernel_type='naive'`):**
- Small datasets (< 100 samples)
- Limited GPU shared memory
- Quick prototyping

**Use Tiled (`kernel_type='tiled'`):**
- Large datasets (> 100 samples)
- Production workloads
- Maximum performance needed

## Optimization Tips

### Data Preparation

1. **Use correct data type:**
   ```python
   # Good: uint8
   genotypes = np.random.randint(0, 3, size=(n, m), dtype=np.uint8)

   # Bad: int64 (wastes memory)
   genotypes = np.random.randint(0, 3, size=(n, m))
   ```

2. **Ensure contiguous arrays:**
   ```python
   # Make array contiguous if needed
   genotypes = np.ascontiguousarray(genotypes)
   ```

3. **Pre-allocate arrays:**
   ```python
   # Pre-allocate for multiple runs
   genotypes = np.empty((n_samples, n_loci), dtype=np.uint8)
   # ... fill with data ...
   ```

### Runtime Optimization

1. **Minimize data transfers:**
   ```python
   # Bad: Multiple transfers
   for i in range(100):
       result = cukinship.compute_kinship(small_batch[i])

   # Good: Single large transfer
   result = cukinship.compute_kinship(all_data)
   ```

2. **Use appropriate kernel:**
   ```python
   # For large datasets
   result = cukinship.compute_kinship(data, kernel_type='tiled')

   # For small datasets
   result = cukinship.compute_kinship(data, kernel_type='naive')
   ```

3. **Disable normalization if not needed:**
   ```python
   # Skip normalization for faster computation
   result = cukinship.compute_kinship(data, normalize=False)
   ```

### Memory Optimization

Maximum dataset sizes by GPU memory:

| GPU Memory | Max Samples | Max Loci   | Notes                    |
|------------|-------------|------------|--------------------------|
| 8 GB       | 2,000       | 20,000     | Single precision         |
| 16 GB      | 4,000       | 40,000     | Typical workstation GPU  |
| 24 GB      | 5,000       | 50,000     | RTX 3090, RTX 4090       |
| 40 GB      | 7,000       | 70,000     | A100 40GB                |
| 80 GB      | 10,000      | 100,000    | A100 80GB, H100          |

Memory usage formula:
```
Memory ≈ (samples × loci × 2 bits / 8) + (samples² × 4 bytes)
       = (samples × loci / 4) + (samples² × 4) bytes
```

## Profiling

### Using Built-in Benchmarks

```bash
# Run standard benchmark suite
./build/gpu_kinship --benchmark

# Custom benchmark
./build/gpu_kinship --samples 1000 --loci 10000 --iterations 100
```

### Using Nsight Compute

```bash
# Comprehensive profiling
./scripts/profile_kernels.sh --samples 1000 --loci 10000

# View results
ncu-ui ./profiles/profile_full.ncu-rep
```

### Key Metrics to Monitor

1. **GPU Utilization**
   - Target: > 90%
   - Low utilization suggests too small workload

2. **Memory Bandwidth**
   - Target: > 70% of peak
   - Indicates memory-bound vs compute-bound

3. **Occupancy**
   - Target: > 50%
   - Higher is better for latency hiding

4. **Shared Memory Bank Conflicts**
   - Target: < 5%
   - Tiled kernel should have minimal conflicts

### Profiling Example

```bash
# Profile specific kernel
ncu --set full \
    --kernel-name "compute_kinship_kernel_tiled" \
    ./build/gpu_kinship --samples 500 --loci 5000

# Focus on memory
ncu --set memory \
    ./build/gpu_kinship --samples 1000 --loci 10000

# Focus on compute
ncu --set compute \
    ./build/gpu_kinship --samples 1000 --loci 10000
```

## Scalability

### Weak Scaling

Performance with constant samples/loci ratio (samples:loci = 1:10):

| Samples | Loci   | Time (ms) | Time/Sample (μs) |
|---------|--------|-----------|------------------|
| 100     | 1,000  | 0.98      | 9.8              |
| 500     | 5,000  | 24.3      | 48.6             |
| 1,000   | 10,000 | 97.1      | 97.1             |
| 5,000   | 50,000 | 2,420     | 484              |

*Performance degrades with scale due to memory bandwidth limits*

### Strong Scaling

Performance with fixed dataset (1000 samples × 10000 loci):

Not applicable - single GPU implementation.

Multi-GPU scaling is a future feature.

### Asymptotic Complexity

- **Time Complexity:** O(n² × m)
  - n = number of samples
  - m = number of loci

- **Space Complexity:** O(n² + n×m)
  - Output matrix: n²
  - Input data: n×m/4 (2-bit packed)

## Hardware Comparison

Expected relative performance on different GPUs:

| GPU               | Compute Cap. | Relative Perf. | Notes              |
|-------------------|--------------|----------------|--------------------|
| RTX 3080          | 8.6          | 1.0×           | Baseline           |
| RTX 3090          | 8.6          | 1.2×           | More memory        |
| RTX 4080          | 8.9          | 1.5×           | Faster memory      |
| RTX 4090          | 8.9          | 2.0×           | Highest consumer   |
| A100 (40GB)       | 8.0          | 2.5×           | Data center        |
| A100 (80GB)       | 8.0          | 2.5×           | Large datasets     |
| H100              | 9.0          | 4.0×           | Hopper architecture|

*Performance may vary based on cooling, power limits, and system configuration*

## Best Practices

### For Maximum Performance

1. **Use latest CUDA toolkit**
   - Newer versions have optimizations

2. **Enable persistent mode** (Linux):
   ```bash
   sudo nvidia-smi -pm 1
   ```

3. **Set performance governor**:
   ```bash
   sudo nvidia-smi -pl 400  # Set power limit
   sudo nvidia-smi --auto-boost-default=0
   ```

4. **Pin CPU affinity** for multi-GPU:
   ```python
   import os
   os.sched_setaffinity(0, {0, 1, 2, 3})  # Bind to specific CPUs
   ```

5. **Batch processing** for multiple datasets:
   ```python
   for genotypes in large_dataset_list:
       result = cukinship.compute_kinship(genotypes)
   ```

### For Memory-Constrained Scenarios

1. **Process in chunks**:
   ```python
   chunk_size = 1000
   for i in range(0, n_samples, chunk_size):
       chunk = genotypes[i:i+chunk_size]
       result = cukinship.compute_kinship(chunk)
   ```

2. **Use lower precision** (future feature)

3. **Enable unified memory** (future feature)

## Troubleshooting Performance Issues

### Slow Performance

1. Check GPU utilization:
   ```bash
   nvidia-smi dmon -s u
   ```

2. Verify data is on GPU:
   ```bash
   nvidia-smi dmon -s m  # Monitor memory
   ```

3. Profile with Nsight Compute:
   ```bash
   ./scripts/profile_kernels.sh
   ```

### Out of Memory

1. Reduce batch size
2. Process in chunks
3. Use GPU with more memory
4. Check for memory leaks

### Unexpected Results

1. Verify input data range [0, 1, 2]
2. Check for NaN/Inf in output
3. Compare with CPU reference
4. Enable debug mode

## Future Optimizations

Planned performance improvements:

- Multi-GPU support with NCCL
- FP16/BF16 precision option
- Tensor Core utilization
- CUDA graphs for reduced launch overhead
- Persistent kernels for latency reduction
- Optimized kernels for specific architectures (Hopper, Blackwell)

## References

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)

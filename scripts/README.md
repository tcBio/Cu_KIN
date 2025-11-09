# Profiling and Analysis Scripts

This directory contains scripts for profiling and analyzing cuKINSHIP-Lite performance using NVIDIA Nsight Compute.

## Prerequisites

- NVIDIA Nsight Compute (`ncu`) installed and in PATH
- cuKINSHIP-Lite built in `./build` directory
- CUDA-capable GPU

## Scripts

### profile_kernels.sh

Comprehensive profiling of GPU kernels with Nsight Compute.

```bash
# Basic usage
./scripts/profile_kernels.sh

# Custom configuration
./scripts/profile_kernels.sh --samples 500 --loci 5000 --output ./my_profiles

# Options:
#   --samples N     Number of samples (default: 100)
#   --loci M        Number of loci (default: 1000)
#   --output DIR    Output directory (default: ./profiles)
#   --build-dir DIR Build directory (default: ./build)
```

**Outputs:**
- `profile_full.ncu-rep` - Complete metrics set
- `profile_naive.ncu-rep` - Naive kernel profile
- `profile_tiled.ncu-rep` - Tiled kernel profile
- `profile_memory.ncu-rep` - Memory-focused metrics
- `profile_compute.ncu-rep` - Compute-focused metrics

**View results:**
```bash
# Interactive UI
ncu-ui ./profiles/profile_full.ncu-rep

# CLI report
ncu --import ./profiles/profile_full.ncu-rep
```

### compare_kernels.sh

Compare naive vs tiled kernel implementations across multiple dataset sizes.

```bash
./scripts/compare_kernels.sh
```

**Outputs:**
- CSV files in `./profiles/comparison/`
- One file per configuration tested

## Profiling Workflow

1. **Build the project:**
   ```bash
   cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
   cmake --build build --config Release
   ```

2. **Run comprehensive profiling:**
   ```bash
   ./scripts/profile_kernels.sh --samples 1000 --loci 10000
   ```

3. **Compare kernel implementations:**
   ```bash
   ./scripts/compare_kernels.sh
   ```

4. **Analyze results:**
   ```bash
   # Open in Nsight Compute UI
   ncu-ui ./profiles/profile_full.ncu-rep

   # Or generate text report
   ncu --import ./profiles/profile_full.ncu-rep --page details
   ```

## Key Metrics to Examine

### Memory Performance
- **Global Load/Store Efficiency** - How efficiently global memory is accessed
- **Shared Memory Utilization** - Tiled kernel should show high shared memory usage
- **L1/L2 Cache Hit Rates** - Cache effectiveness
- **Memory Throughput** - GB/s achieved vs theoretical peak

### Compute Performance
- **SM Efficiency** - Streaming Multiprocessor utilization
- **Occupancy** - Active warps per SM
- **IPC (Instructions Per Cycle)** - Instruction throughput
- **Warp Execution Efficiency** - Branch divergence impact

### Kernel-Specific Insights
- **Naive kernel (16×16 blocks):**
  - Lower shared memory usage
  - More global memory transactions
  - Simpler execution pattern

- **Tiled kernel (32×32 blocks with shared memory):**
  - High shared memory utilization
  - Reduced global memory traffic
  - Better cache locality

## Troubleshooting

**"ncu: command not found"**
- Install NVIDIA Nsight Compute or add to PATH:
  ```bash
  export PATH=$PATH:/usr/local/cuda/nsight-compute
  ```

**"Permission denied" errors**
- GPU profiling may require elevated privileges:
  ```bash
  sudo ./scripts/profile_kernels.sh
  ```

**Profiling on remote/headless systems**
- Profile and copy reports to local machine:
  ```bash
  scp remote:Cu_KIN/profiles/*.ncu-rep ./
  ncu-ui ./profile_full.ncu-rep
  ```

## Advanced Usage

### Profile specific kernel only
```bash
ncu --kernel-name compute_kinship_kernel_tiled \
    --set full \
    ./build/gpu_kinship --samples 100 --loci 1000
```

### Focus on roofline analysis
```bash
ncu --set roofline \
    --export ./profiles/roofline \
    ./build/gpu_kinship --benchmark
```

### Export metrics to CSV
```bash
ncu --csv \
    --metrics gpu__time_duration.avg,dram__throughput.avg.pct_of_peak \
    ./build/gpu_kinship --samples 100 --loci 1000 > metrics.csv
```

## References

- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [CUDA Profiling Best Practices](https://docs.nvidia.com/cuda/profiler-users-guide/)

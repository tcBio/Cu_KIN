#!/usr/bin/env python3
"""
Basic usage examples for cuKINSHIP-Lite Python bindings
"""

import numpy as np
import time


def example_basic():
    """Basic kinship matrix computation"""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    import cukinship

    # Check GPU availability
    if not cukinship.check_gpu():
        print("This example requires a CUDA-capable GPU")
        return

    # Create sample genotype data
    # 10 samples, 100 loci, values in {0, 1, 2}
    np.random.seed(42)
    genotypes = np.random.randint(0, 3, size=(10, 100), dtype=np.uint8)

    print(f"\nGenotype matrix shape: {genotypes.shape}")
    print(f"Values: min={genotypes.min()}, max={genotypes.max()}")

    # Compute kinship matrix
    similarity = cukinship.compute_kinship(genotypes)

    print(f"\nSimilarity matrix shape: {similarity.shape}")
    print(f"Values: min={similarity.min():.3f}, max={similarity.max():.3f}")

    # Verify properties
    print("\nMatrix properties:")
    print(f"  Symmetric: {np.allclose(similarity, similarity.T)}")
    print(f"  Diagonal ≈ 1: {np.allclose(np.diag(similarity), 1.0, atol=0.01)}")

    # Print 4x4 corner
    print("\nUpper-left 4×4 corner:")
    print(similarity[:4, :4])


def example_benchmark():
    """Benchmark performance on different dataset sizes"""
    print("\n" + "=" * 60)
    print("Example 2: Performance Benchmark")
    print("=" * 60)

    import cukinship

    configs = [
        (10, 100),
        (50, 500),
        (100, 1000),
        (500, 5000),
    ]

    print(f"\n{'Samples':>8} {'Loci':>8} {'Time (ms)':>12} {'Throughput (comp/s)':>20}")
    print("-" * 60)

    for n_samples, n_loci in configs:
        genotypes = np.random.randint(0, 3, size=(n_samples, n_loci), dtype=np.uint8)

        # Warmup
        _ = cukinship.compute_kinship(genotypes)

        # Benchmark
        start = time.perf_counter()
        result = cukinship.compute_kinship(genotypes)
        elapsed = (time.perf_counter() - start) * 1000  # ms

        # Compute throughput (pairwise comparisons per second)
        comparisons = n_samples * (n_samples + 1) // 2
        throughput = comparisons / (elapsed / 1000)

        print(f"{n_samples:>8} {n_loci:>8} {elapsed:>12.3f} {throughput:>20.0f}")


def example_kernel_comparison():
    """Compare naive vs tiled kernel implementations"""
    print("\n" + "=" * 60)
    print("Example 3: Kernel Comparison (Naive vs Tiled)")
    print("=" * 60)

    import cukinship

    n_samples, n_loci = 100, 1000
    genotypes = np.random.randint(0, 3, size=(n_samples, n_loci), dtype=np.uint8)

    # Test both kernels
    for kernel_type in ['naive', 'tiled']:
        # Warmup
        _ = cukinship.compute_kinship(genotypes, kernel_type=kernel_type)

        # Benchmark (average of 10 runs)
        times = []
        for _ in range(10):
            start = time.perf_counter()
            result = cukinship.compute_kinship(genotypes, kernel_type=kernel_type)
            times.append((time.perf_counter() - start) * 1000)

        mean_time = np.mean(times)
        std_time = np.std(times)

        print(f"\n{kernel_type.upper()} kernel:")
        print(f"  Time: {mean_time:.3f} ± {std_time:.3f} ms")
        print(f"  Min/Max: {min(times):.3f} / {max(times):.3f} ms")

    # Verify both give same results
    result_naive = cukinship.compute_kinship(genotypes, kernel_type='naive')
    result_tiled = cukinship.compute_kinship(genotypes, kernel_type='tiled')

    max_diff = np.abs(result_naive - result_tiled).max()
    print(f"\nMax difference between kernels: {max_diff:.2e}")
    print(f"Results match: {np.allclose(result_naive, result_tiled, atol=1e-4)}")


def example_normalization():
    """Demonstrate normalization effects"""
    print("\n" + "=" * 60)
    print("Example 4: Normalization")
    print("=" * 60)

    import cukinship

    genotypes = np.random.randint(0, 3, size=(5, 50), dtype=np.uint8)

    # Without normalization
    sim_raw = cukinship.compute_kinship(genotypes, normalize=False)
    print("\nWithout normalization:")
    print(f"  Range: [{sim_raw.min():.3f}, {sim_raw.max():.3f}]")
    print(sim_raw)

    # With normalization (default)
    sim_norm = cukinship.compute_kinship(genotypes, normalize=True)
    print("\nWith normalization:")
    print(f"  Range: [{sim_norm.min():.3f}, {sim_norm.max():.3f}]")
    print(sim_norm)


if __name__ == '__main__':
    try:
        example_basic()
        example_benchmark()
        example_kernel_comparison()
        example_normalization()
    except ImportError as e:
        print(f"Error: {e}")
        print("\nPlease install cukinship first:")
        print("  cd python && pip install .")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

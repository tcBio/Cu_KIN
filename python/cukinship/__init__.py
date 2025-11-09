"""
cuKINSHIP-Lite: GPU-accelerated kinship matrix computation

This package provides Python bindings for the cuKINSHIP-Lite library,
enabling fast kinship/similarity matrix computation on NVIDIA GPUs.

Basic usage:
    >>> import numpy as np
    >>> import cukinship
    >>>
    >>> # Create sample genotype data (samples × loci)
    >>> genotypes = np.random.randint(0, 3, size=(100, 1000), dtype=np.uint8)
    >>>
    >>> # Compute kinship matrix
    >>> similarity = cukinship.compute_kinship(genotypes)
    >>>
    >>> # Check GPU availability
    >>> info = cukinship.gpu_info()
    >>> print(f"GPU: {info['name']}")
"""

from ._cukinship import compute_kinship, gpu_info, __version__

__all__ = ['compute_kinship', 'gpu_info', '__version__']


def check_gpu():
    """
    Check if CUDA GPU is available and print information.

    Returns
    -------
    bool
        True if GPU is available, False otherwise.
    """
    info = gpu_info()
    if info['available']:
        print(f"✓ CUDA GPU available")
        print(f"  Name: {info['name']}")
        print(f"  Compute Capability: {info['compute_capability']}")
        print(f"  Memory: {info['total_memory_gb']:.1f} GB")
        print(f"  Multiprocessors: {info['multiprocessors']}")
        return True
    else:
        print(f"✗ CUDA GPU not available")
        if 'error' in info:
            print(f"  Error: {info['error']}")
        return False


# Convenience function
def kinship_matrix(genotypes, **kwargs):
    """
    Alias for compute_kinship for convenience.

    See compute_kinship for full documentation.
    """
    return compute_kinship(genotypes, **kwargs)

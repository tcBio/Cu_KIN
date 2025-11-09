#!/usr/bin/env python3
"""
Generate sample genotype datasets for cuKINSHIP-Lite testing and benchmarking.

This script creates binary files in the packed 2-bit format expected by cuKINSHIP:
- 2 bits per locus (values 0, 1, or 2)
- 32 loci per 64-bit word
- Sample-major ordering (all loci for sample 0, then sample 1, etc.)
"""

import argparse
import struct
import random
from pathlib import Path


def pack_genotypes_2bit(genotypes, n_samples, n_loci):
    """
    Pack genotype data into 2-bit representation.

    Args:
        genotypes: List of uint8 values (0, 1, or 2) in sample-major order
        n_samples: Number of samples
        n_loci: Number of loci per sample

    Returns:
        List of uint64 words containing packed genotypes
    """
    loci_per_word = 32
    words_per_sample = (n_loci + loci_per_word - 1) // loci_per_word
    total_words = n_samples * words_per_sample

    packed = [0] * total_words

    for sample in range(n_samples):
        for locus in range(n_loci):
            genotype = genotypes[sample * n_loci + locus]
            word_idx = sample * words_per_sample + locus // loci_per_word
            bit_offset = (locus % loci_per_word) * 2
            packed[word_idx] |= (genotype & 0x3) << bit_offset

    return packed


def generate_deterministic(n_samples, n_loci, seed=42):
    """Generate deterministic genotypes for reproducible testing."""
    genotypes = []
    for sample in range(n_samples):
        for locus in range(n_loci):
            genotypes.append((sample * 7 + locus + seed) % 3)
    return genotypes


def generate_random(n_samples, n_loci, seed=42):
    """Generate random genotypes with specified seed."""
    random.seed(seed)
    return [random.randint(0, 2) for _ in range(n_samples * n_loci)]


def generate_uniform(n_samples, n_loci, value=0):
    """Generate uniform genotypes (all same value)."""
    return [value] * (n_samples * n_loci)


def write_binary(filename, packed_data):
    """Write packed genotype data to binary file."""
    with open(filename, 'wb') as f:
        for word in packed_data:
            f.write(struct.pack('<Q', word))  # Little-endian uint64


def write_metadata(filename, n_samples, n_loci, description):
    """Write metadata file describing the dataset."""
    with open(filename, 'w') as f:
        f.write(f"Samples: {n_samples}\n")
        f.write(f"Loci: {n_loci}\n")
        f.write(f"Description: {description}\n")
        f.write(f"Format: 2-bit packed, sample-major, little-endian uint64\n")


def main():
    parser = argparse.ArgumentParser(description='Generate sample genotype datasets')
    parser.add_argument('--samples', type=int, default=100, help='Number of samples')
    parser.add_argument('--loci', type=int, default=1000, help='Number of loci')
    parser.add_argument('--mode', choices=['deterministic', 'random', 'uniform'],
                        default='deterministic', help='Generation mode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--value', type=int, default=0, choices=[0, 1, 2],
                        help='Value for uniform mode')
    parser.add_argument('--output', type=str, required=True, help='Output filename (without extension)')

    args = parser.parse_args()

    # Generate genotypes
    if args.mode == 'deterministic':
        genotypes = generate_deterministic(args.samples, args.loci, args.seed)
        description = f"Deterministic pattern with seed {args.seed}"
    elif args.mode == 'random':
        genotypes = generate_random(args.samples, args.loci, args.seed)
        description = f"Random with seed {args.seed}"
    else:  # uniform
        genotypes = generate_uniform(args.samples, args.loci, args.value)
        description = f"Uniform value {args.value}"

    # Pack and write
    packed = pack_genotypes_2bit(genotypes, args.samples, args.loci)

    output_path = Path(args.output)
    write_binary(f"{output_path}.bin", packed)
    write_metadata(f"{output_path}.txt", args.samples, args.loci, description)

    print(f"Generated {args.samples} samples × {args.loci} loci")
    print(f"Mode: {args.mode}")
    print(f"Written to: {output_path}.bin")
    print(f"Metadata: {output_path}.txt")
    print(f"File size: {len(packed) * 8} bytes")


if __name__ == '__main__':
    main()

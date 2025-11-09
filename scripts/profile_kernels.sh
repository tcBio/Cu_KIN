#!/bin/bash
# Nsight Compute profiling script for cuKINSHIP-Lite kernels
#
# Usage: ./scripts/profile_kernels.sh [OPTIONS]
#   --samples N    Number of samples (default: 100)
#   --loci M       Number of loci (default: 1000)
#   --output DIR   Output directory (default: ./profiles)

set -e

# Default parameters
SAMPLES=100
LOCI=1000
OUTPUT_DIR="./profiles"
BUILD_DIR="./build"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --samples)
            SAMPLES="$2"
            shift 2
            ;;
        --loci)
            LOCI="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--samples N] [--loci M] [--output DIR] [--build-dir DIR]"
            exit 1
            ;;
    esac
done

# Check if ncu is available
if ! command -v ncu &> /dev/null; then
    echo "Error: Nsight Compute (ncu) not found in PATH"
    echo "Please install NVIDIA Nsight Compute or add it to your PATH"
    exit 1
fi

# Check if binary exists
BINARY="${BUILD_DIR}/gpu_kinship"
if [ ! -f "$BINARY" ]; then
    echo "Error: Binary not found at $BINARY"
    echo "Please build the project first: cmake --build build --config Release"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "================================================"
echo "cuKINSHIP-Lite Nsight Compute Profiling"
echo "================================================"
echo "Samples:      $SAMPLES"
echo "Loci:         $LOCI"
echo "Output:       $OUTPUT_DIR"
echo "Binary:       $BINARY"
echo "================================================"
echo ""

# Profile with default metrics
echo "Running profile with default metrics..."
ncu --set full \
    --export "${OUTPUT_DIR}/profile_full" \
    --force-overwrite \
    "$BINARY" --samples "$SAMPLES" --loci "$LOCI"

# Profile specific kernels
echo ""
echo "Profiling compute_kinship_kernel_naive..."
ncu --set full \
    --kernel-name "compute_kinship_kernel_naive" \
    --export "${OUTPUT_DIR}/profile_naive" \
    --force-overwrite \
    "$BINARY" --samples "$SAMPLES" --loci "$LOCI"

echo ""
echo "Profiling compute_kinship_kernel_tiled..."
ncu --set full \
    --kernel-name "compute_kinship_kernel_tiled" \
    --export "${OUTPUT_DIR}/profile_tiled" \
    --force-overwrite \
    "$BINARY" --samples "$SAMPLES" --loci "$LOCI"

# Memory profiling
echo ""
echo "Running memory-focused profile..."
ncu --set memory \
    --export "${OUTPUT_DIR}/profile_memory" \
    --force-overwrite \
    "$BINARY" --samples "$SAMPLES" --loci "$LOCI"

# Compute profiling
echo ""
echo "Running compute-focused profile..."
ncu --set compute \
    --export "${OUTPUT_DIR}/profile_compute" \
    --force-overwrite \
    "$BINARY" --samples "$SAMPLES" --loci "$LOCI"

echo ""
echo "================================================"
echo "Profiling complete!"
echo "================================================"
echo "View results with:"
echo "  ncu-ui ${OUTPUT_DIR}/profile_full.ncu-rep"
echo ""
echo "Or generate CLI report:"
echo "  ncu --import ${OUTPUT_DIR}/profile_full.ncu-rep"
echo "================================================"

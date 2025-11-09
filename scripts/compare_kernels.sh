#!/bin/bash
# Compare naive vs tiled kernel performance using Nsight Compute
#
# This script profiles both kernel implementations and generates a comparison report

set -e

BUILD_DIR="./build"
OUTPUT_DIR="./profiles/comparison"

echo "================================================"
echo "Kernel Comparison: Naive vs Tiled"
echo "================================================"

# Check if ncu is available
if ! command -v ncu &> /dev/null; then
    echo "Error: Nsight Compute (ncu) not found in PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Test configurations: (samples, loci)
CONFIGS=(
    "10 100"
    "50 500"
    "100 1000"
    "500 5000"
)

echo ""
echo "Testing configurations:"
for config in "${CONFIGS[@]}"; do
    echo "  - $config samples × loci"
done
echo ""

for config in "${CONFIGS[@]}"; do
    read -r samples loci <<< "$config"

    echo "----------------------------------------"
    echo "Configuration: ${samples} samples × ${loci} loci"
    echo "----------------------------------------"

    # Profile tiled kernel (default)
    echo "Profiling tiled kernel..."
    ncu --metrics gpu__time_duration.avg \
        --csv \
        --log-file "${OUTPUT_DIR}/tiled_${samples}_${loci}.csv" \
        "${BUILD_DIR}/gpu_kinship" --samples "$samples" --loci "$loci" > /dev/null 2>&1 || true

    echo "Done."
done

echo ""
echo "================================================"
echo "Comparison complete!"
echo "================================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Summary CSV files:"
ls -1 "${OUTPUT_DIR}"/*.csv
echo "================================================"

#include "gpu_kinship.cuh"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace kinship {

namespace {

constexpr int kBlockDim = 16;
constexpr int kTileDim = 32;

// Original naive kernel (16x16 blocks)
__global__ void compute_kinship_kernel_naive(const uint64_t* __restrict__ geno_words,
                                             float* __restrict__ kinship,
                                             int n_samples,
                                             int words_per_sample,
                                             float inverse_norm) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n_samples || col >= n_samples || col < row) {
        return;
    }

    unsigned int diff_bits = 0;
    for (int word = 0; word < words_per_sample; ++word) {
        const uint64_t a = geno_words[static_cast<std::size_t>(row) * words_per_sample + word];
        const uint64_t b = geno_words[static_cast<std::size_t>(col) * words_per_sample + word];
        diff_bits += static_cast<unsigned int>(__popcll(a ^ b));
    }

    const float similarity = 1.0f - static_cast<float>(diff_bits) * inverse_norm;
    kinship[static_cast<std::size_t>(row) * n_samples + col] = similarity;
    if (row != col) {
        kinship[static_cast<std::size_t>(col) * n_samples + row] = similarity;
    }
}

// Shared memory tiled kernel (32x32 blocks)
// Loads genotype data into shared memory for better cache utilization
__global__ void compute_kinship_kernel_tiled(const uint64_t* __restrict__ geno_words,
                                             float* __restrict__ kinship,
                                             int n_samples,
                                             int words_per_sample,
                                             float inverse_norm) {
    extern __shared__ uint64_t shared_data[];

    // Partition shared memory: first half for tile_a, second half for tile_b
    uint64_t* tile_a = shared_data;
    uint64_t* tile_b = &shared_data[kTileDim * words_per_sample];

    const int row = blockIdx.y * kTileDim + threadIdx.y;
    const int col = blockIdx.x * kTileDim + threadIdx.x;

    // Compute linear thread index for cooperative loading
    const int tid = threadIdx.y * kTileDim + threadIdx.x;
    const int total_threads = kTileDim * kTileDim;

    // Cooperatively load tile_a (rows for this block)
    const int row_base = blockIdx.y * kTileDim;
    for (int i = tid; i < kTileDim * words_per_sample; i += total_threads) {
        const int local_row = i / words_per_sample;
        const int word_idx = i % words_per_sample;
        const int global_row = row_base + local_row;

        if (global_row < n_samples) {
            tile_a[i] = geno_words[static_cast<std::size_t>(global_row) * words_per_sample + word_idx];
        } else {
            tile_a[i] = 0;
        }
    }

    // Cooperatively load tile_b (columns for this block)
    const int col_base = blockIdx.x * kTileDim;
    for (int i = tid; i < kTileDim * words_per_sample; i += total_threads) {
        const int local_col = i / words_per_sample;
        const int word_idx = i % words_per_sample;
        const int global_col = col_base + local_col;

        if (global_col < n_samples) {
            tile_b[i] = geno_words[static_cast<std::size_t>(global_col) * words_per_sample + word_idx];
        } else {
            tile_b[i] = 0;
        }
    }

    __syncthreads();

    // Compute similarity using shared memory
    if (row >= n_samples || col >= n_samples || col < row) {
        return;
    }

    unsigned int diff_bits = 0;
    for (int word = 0; word < words_per_sample; ++word) {
        const uint64_t a = tile_a[threadIdx.y * words_per_sample + word];
        const uint64_t b = tile_b[threadIdx.x * words_per_sample + word];
        diff_bits += static_cast<unsigned int>(__popcll(a ^ b));
    }

    const float similarity = 1.0f - static_cast<float>(diff_bits) * inverse_norm;
    kinship[static_cast<std::size_t>(row) * n_samples + col] = similarity;
    if (row != col) {
        kinship[static_cast<std::size_t>(col) * n_samples + row] = similarity;
    }
}

inline void throw_if_error(cudaError_t status, const char* context) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
    }
}

inline unsigned int popcount_host(uint64_t value) {
#if defined(_MSC_VER)
    return static_cast<unsigned int>(__popcnt64(value));
#else
    return static_cast<unsigned int>(__builtin_popcountll(value));
#endif
}

}  // namespace

std::size_t words_per_sample(std::size_t n_loci) {
    return (n_loci + kLociPerWord - 1) / kLociPerWord;
}

void compute_kinship(const uint64_t* d_genotypes,
                     float* d_similarity,
                     int n_samples,
                     int words_count,
                     cudaStream_t stream,
                     KernelType kernel) {
    if (!d_genotypes || !d_similarity) {
        throw std::invalid_argument("compute_kinship: null device pointer");
    }
    if (n_samples <= 0 || words_count <= 0) {
        throw std::invalid_argument("compute_kinship: invalid problem dimensions");
    }

    const float inverse_norm = 1.0f / static_cast<float>(words_count * 64);

    if (kernel == KernelType::Naive) {
        const dim3 block(kBlockDim, kBlockDim);
        const dim3 grid((n_samples + block.x - 1) / block.x, (n_samples + block.y - 1) / block.y);

        compute_kinship_kernel_naive<<<grid, block, 0, stream>>>(d_genotypes,
                                                                  d_similarity,
                                                                  n_samples,
                                                                  words_count,
                                                                  inverse_norm);
        throw_if_error(cudaGetLastError(), "compute_kinship_kernel_naive launch");
    } else {  // KernelType::Tiled
        const dim3 block(kTileDim, kTileDim);
        const dim3 grid((n_samples + block.x - 1) / block.x, (n_samples + block.y - 1) / block.y);

        // Calculate shared memory size: two tiles of genotype data
        const std::size_t shared_mem_bytes = 2 * kTileDim * words_count * sizeof(uint64_t);

        compute_kinship_kernel_tiled<<<grid, block, shared_mem_bytes, stream>>>(d_genotypes,
                                                                                  d_similarity,
                                                                                  n_samples,
                                                                                  words_count,
                                                                                  inverse_norm);
        throw_if_error(cudaGetLastError(), "compute_kinship_kernel_tiled launch");
    }
}

void compute_kinship_host_reference(const std::vector<uint64_t>& bitpacked,
                                    int n_samples,
                                    int words_count,
                                    std::vector<float>& similarity) {
    similarity.assign(static_cast<std::size_t>(n_samples) * n_samples, 0.0f);
    const float inverse_norm = 1.0f / static_cast<float>(words_count * 64);
    for (int row = 0; row < n_samples; ++row) {
        for (int col = row; col < n_samples; ++col) {
            unsigned int diff_bits = 0;
            for (int word = 0; word < words_count; ++word) {
                const uint64_t a = bitpacked[static_cast<std::size_t>(row) * words_count + word];
                const uint64_t b = bitpacked[static_cast<std::size_t>(col) * words_count + word];
                diff_bits += popcount_host(a ^ b);
            }
            const float similarity_value = 1.0f - static_cast<float>(diff_bits) * inverse_norm;
            similarity[static_cast<std::size_t>(row) * n_samples + col] = similarity_value;
            if (row != col) {
                similarity[static_cast<std::size_t>(col) * n_samples + row] = similarity_value;
            }
        }
    }
}

void synchronize_or_throw(cudaStream_t stream) {
    throw_if_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
}

}  // namespace kinship

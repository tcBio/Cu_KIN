#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

namespace kinship {

constexpr std::size_t kLociPerWord = 32;  // 2-bit genotypes packed into 64-bit words.

std::size_t words_per_sample(std::size_t n_loci);

void compute_kinship(const uint64_t* d_genotypes,
                     float* d_similarity,
                     int n_samples,
                     int words_per_sample,
                     cudaStream_t stream = nullptr);

void compute_kinship_host_reference(const std::vector<uint64_t>& bitpacked,
                                    int n_samples,
                                    int words_per_sample,
                                    std::vector<float>& similarity);

void synchronize_or_throw(cudaStream_t stream);

}  // namespace kinship

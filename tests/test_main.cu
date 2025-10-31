#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

#include "bit_encode.hpp"
#include "gpu_kinship.cuh"

namespace {

std::vector<uint8_t> make_samples(std::size_t n_samples, std::size_t n_loci) {
    std::vector<uint8_t> genotypes(n_samples * n_loci);
    for (std::size_t sample = 0; sample < n_samples; ++sample) {
        for (std::size_t locus = 0; locus < n_loci; ++locus) {
            genotypes[sample * n_loci + locus] = static_cast<uint8_t>((sample * 7 + locus) % 3);
        }
    }
    return genotypes;
}

}  // namespace

int main() {
    constexpr std::size_t kSamples = 4;
    constexpr std::size_t kLoci = 96;

    auto genotypes = make_samples(kSamples, kLoci);
    auto packed = kinship::pack_genotypes_2bit(genotypes, kSamples, kLoci);
    const int words = static_cast<int>(kinship::words_per_sample(kLoci));

    uint64_t* d_genotypes = nullptr;
    float* d_similarity = nullptr;
    const std::size_t genotypes_bytes = packed.size() * sizeof(uint64_t);
    const std::size_t similarity_bytes = kSamples * kSamples * sizeof(float);

    cudaError_t status = cudaMalloc(&d_genotypes, genotypes_bytes);
    assert(status == cudaSuccess);
    status = cudaMalloc(&d_similarity, similarity_bytes);
    assert(status == cudaSuccess);

    status = cudaMemcpy(d_genotypes, packed.data(), genotypes_bytes, cudaMemcpyHostToDevice);
    assert(status == cudaSuccess);

    kinship::compute_kinship(d_genotypes, d_similarity, static_cast<int>(kSamples), words);
    kinship::synchronize_or_throw(nullptr);

    std::vector<float> gpu_result(kSamples * kSamples);
    status = cudaMemcpy(gpu_result.data(), d_similarity, similarity_bytes, cudaMemcpyDeviceToHost);
    assert(status == cudaSuccess);

    std::vector<float> reference;
    kinship::compute_kinship_host_reference(packed, static_cast<int>(kSamples), words, reference);

    float max_error = 0.0f;
    for (std::size_t idx = 0; idx < gpu_result.size(); ++idx) {
        max_error = std::max(max_error, std::fabs(gpu_result[idx] - reference[idx]));
    }

    for (std::size_t row = 0; row < kSamples; ++row) {
        assert(gpu_result[row * kSamples + row] >= 0.99f);
        for (std::size_t col = row; col < kSamples; ++col) {
            const float a = gpu_result[row * kSamples + col];
            const float b = gpu_result[col * kSamples + row];
            assert(std::fabs(a - b) < 1e-6f);
        }
    }

    assert(max_error < 1e-4f);

    cudaFree(d_genotypes);
    cudaFree(d_similarity);

    return 0;
}

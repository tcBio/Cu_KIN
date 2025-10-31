#include <cuda_runtime.h>

#include <algorithm>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "bit_encode.hpp"
#include "gpu_kinship.cuh"
#include "normalize.hpp"
#include "utils/logger.hpp"
#include "utils/timer.hpp"

namespace {

std::vector<std::uint8_t> generate_mock_genotypes(std::size_t n_samples, std::size_t n_loci) {
    std::vector<std::uint8_t> data(n_samples * n_loci);
    for (std::size_t sample = 0; sample < n_samples; ++sample) {
        for (std::size_t locus = 0; locus < n_loci; ++locus) {
            data[sample * n_loci + locus] = static_cast<std::uint8_t>((sample + locus) % 3);
        }
    }
    return data;
}

}  // namespace

struct cuda_deleter {
    void operator()(void* ptr) const noexcept {
        if (ptr) {
            cudaFree(ptr);
        }
    }
};

int main() {
    constexpr std::size_t kSamples = 8;
    constexpr std::size_t kLoci = 256;

    try {
        const auto genotypes = generate_mock_genotypes(kSamples, kLoci);
        auto packed = kinship::pack_genotypes_2bit(genotypes, kSamples, kLoci);
        const int words = static_cast<int>(kinship::words_per_sample(kLoci));

        const std::size_t genotypes_bytes = packed.size() * sizeof(uint64_t);
        const std::size_t similarity_bytes = kSamples * kSamples * sizeof(float);

        uint64_t* raw_genotypes = nullptr;
        float* raw_similarity = nullptr;

        if (cudaMalloc(&raw_genotypes, genotypes_bytes) != cudaSuccess ||
            cudaMalloc(&raw_similarity, similarity_bytes) != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed");
        }

        std::unique_ptr<uint64_t, cuda_deleter> d_genotypes(raw_genotypes, cuda_deleter{});
        std::unique_ptr<float, cuda_deleter> d_similarity(raw_similarity, cuda_deleter{});

        if (cudaMemcpy(d_genotypes.get(), packed.data(), genotypes_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy host->device failed");
        }

        kinship::gpu_timer timer;
        timer.start();
        kinship::compute_kinship(d_genotypes.get(), d_similarity.get(), static_cast<int>(kSamples), words);
        kinship::synchronize_or_throw(nullptr);
        timer.stop();

        std::vector<float> similarity(kSamples * kSamples);
        if (cudaMemcpy(similarity.data(), d_similarity.get(), similarity_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy device->host failed");
        }

        std::vector<float> reference;
        kinship::compute_kinship_host_reference(packed, static_cast<int>(kSamples), words, reference);

        float max_error = 0.0f;
        for (std::size_t idx = 0; idx < similarity.size(); ++idx) {
            max_error = std::max(max_error, std::abs(similarity[idx] - reference[idx]));
        }

        kinship::normalize_similarity(similarity);

        KINSHIP_LOG_INFO("GPU kernel completed");
        KINSHIP_LOG_INFO("Elapsed (ms): " + std::to_string(timer.elapsed_milliseconds()));
        KINSHIP_LOG_INFO("Maximum absolute error vs. CPU: " + std::to_string(max_error));

        std::cout << "Normalized kinship matrix (upper-left 4x4 block):\n";
        for (std::size_t row = 0; row < std::min<std::size_t>(4, kSamples); ++row) {
            for (std::size_t col = 0; col < std::min<std::size_t>(4, kSamples); ++col) {
                std::cout << std::fixed << std::setprecision(3)
                          << similarity[row * kSamples + col] << " ";
            }
            std::cout << '\n';
        }

        return max_error < 1e-4f ? 0 : 1;
    } catch (const std::exception& ex) {
        KINSHIP_LOG_ERROR(ex.what());
        return 1;
    }
}

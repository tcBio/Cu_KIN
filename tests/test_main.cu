#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "bit_encode.hpp"
#include "gpu_kinship.cuh"
#include "normalize.hpp"

namespace {

// Test result tracking
int tests_passed = 0;
int tests_failed = 0;

void log_test(const char* name, bool passed) {
    if (passed) {
        std::cout << "[PASS] " << name << "\n";
        ++tests_passed;
    } else {
        std::cerr << "[FAIL] " << name << "\n";
        ++tests_failed;
    }
}

std::vector<uint8_t> make_samples(std::size_t n_samples, std::size_t n_loci) {
    std::vector<uint8_t> genotypes(n_samples * n_loci);
    for (std::size_t sample = 0; sample < n_samples; ++sample) {
        for (std::size_t locus = 0; locus < n_loci; ++locus) {
            genotypes[sample * n_loci + locus] = static_cast<uint8_t>((sample * 7 + locus) % 3);
        }
    }
    return genotypes;
}

std::vector<uint8_t> make_uniform(std::size_t n_samples, std::size_t n_loci, uint8_t value) {
    return std::vector<uint8_t>(n_samples * n_loci, value);
}

std::vector<uint8_t> make_random(std::size_t n_samples, std::size_t n_loci, unsigned int seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint8_t> dist(0, 2);
    std::vector<uint8_t> genotypes(n_samples * n_loci);
    for (auto& g : genotypes) {
        g = dist(rng);
    }
    return genotypes;
}

bool verify_gpu_cpu_match(const std::vector<float>& gpu, const std::vector<float>& cpu, float tolerance) {
    if (gpu.size() != cpu.size()) return false;
    float max_error = 0.0f;
    for (std::size_t i = 0; i < gpu.size(); ++i) {
        max_error = std::max(max_error, std::fabs(gpu[i] - cpu[i]));
    }
    return max_error < tolerance;
}

bool verify_symmetry(const std::vector<float>& matrix, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i; j < n; ++j) {
            if (std::fabs(matrix[i * n + j] - matrix[j * n + i]) > 1e-6f) {
                return false;
            }
        }
    }
    return true;
}

bool verify_diagonal(const std::vector<float>& matrix, std::size_t n, float min_val) {
    for (std::size_t i = 0; i < n; ++i) {
        if (matrix[i * n + i] < min_val) {
            return false;
        }
    }
    return true;
}

// Test 1: Original smoke test
void test_basic_smoke() {
    constexpr std::size_t kSamples = 4;
    constexpr std::size_t kLoci = 96;

    auto genotypes = make_samples(kSamples, kLoci);
    auto packed = kinship::pack_genotypes_2bit(genotypes, kSamples, kLoci);
    const int words = static_cast<int>(kinship::words_per_sample(kLoci));

    uint64_t* d_genotypes = nullptr;
    float* d_similarity = nullptr;
    const std::size_t genotypes_bytes = packed.size() * sizeof(uint64_t);
    const std::size_t similarity_bytes = kSamples * kSamples * sizeof(float);

    cudaMalloc(&d_genotypes, genotypes_bytes);
    cudaMalloc(&d_similarity, similarity_bytes);
    cudaMemcpy(d_genotypes, packed.data(), genotypes_bytes, cudaMemcpyHostToDevice);

    kinship::compute_kinship(d_genotypes, d_similarity, static_cast<int>(kSamples), words);
    kinship::synchronize_or_throw(nullptr);

    std::vector<float> gpu_result(kSamples * kSamples);
    cudaMemcpy(gpu_result.data(), d_similarity, similarity_bytes, cudaMemcpyDeviceToHost);

    std::vector<float> reference;
    kinship::compute_kinship_host_reference(packed, static_cast<int>(kSamples), words, reference);

    bool passed = verify_gpu_cpu_match(gpu_result, reference, 1e-4f) &&
                  verify_symmetry(gpu_result, kSamples) &&
                  verify_diagonal(gpu_result, kSamples, 0.99f);

    cudaFree(d_genotypes);
    cudaFree(d_similarity);

    log_test("Basic smoke test (4 samples x 96 loci)", passed);
}

// Test 2: Single sample
void test_single_sample() {
    constexpr std::size_t kSamples = 1;
    constexpr std::size_t kLoci = 64;

    auto genotypes = make_samples(kSamples, kLoci);
    auto packed = kinship::pack_genotypes_2bit(genotypes, kSamples, kLoci);
    const int words = static_cast<int>(kinship::words_per_sample(kLoci));

    uint64_t* d_genotypes = nullptr;
    float* d_similarity = nullptr;
    cudaMalloc(&d_genotypes, packed.size() * sizeof(uint64_t));
    cudaMalloc(&d_similarity, kSamples * kSamples * sizeof(float));
    cudaMemcpy(d_genotypes, packed.data(), packed.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);

    kinship::compute_kinship(d_genotypes, d_similarity, static_cast<int>(kSamples), words);
    kinship::synchronize_or_throw(nullptr);

    std::vector<float> gpu_result(kSamples * kSamples);
    cudaMemcpy(gpu_result.data(), d_similarity, sizeof(float), cudaMemcpyDeviceToHost);

    bool passed = (gpu_result[0] >= 0.99f);

    cudaFree(d_genotypes);
    cudaFree(d_similarity);

    log_test("Single sample", passed);
}

// Test 3: All zeros genotype
void test_all_zeros() {
    constexpr std::size_t kSamples = 4;
    constexpr std::size_t kLoci = 128;

    auto genotypes = make_uniform(kSamples, kLoci, 0);
    auto packed = kinship::pack_genotypes_2bit(genotypes, kSamples, kLoci);
    const int words = static_cast<int>(kinship::words_per_sample(kLoci));

    uint64_t* d_genotypes = nullptr;
    float* d_similarity = nullptr;
    cudaMalloc(&d_genotypes, packed.size() * sizeof(uint64_t));
    cudaMalloc(&d_similarity, kSamples * kSamples * sizeof(float));
    cudaMemcpy(d_genotypes, packed.data(), packed.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);

    kinship::compute_kinship(d_genotypes, d_similarity, static_cast<int>(kSamples), words);
    kinship::synchronize_or_throw(nullptr);

    std::vector<float> gpu_result(kSamples * kSamples);
    cudaMemcpy(gpu_result.data(), d_similarity, kSamples * kSamples * sizeof(float), cudaMemcpyDeviceToHost);

    // All samples should be identical
    bool passed = true;
    for (std::size_t i = 0; i < kSamples * kSamples; ++i) {
        if (std::fabs(gpu_result[i] - 1.0f) > 1e-5f) {
            passed = false;
            break;
        }
    }

    cudaFree(d_genotypes);
    cudaFree(d_similarity);

    log_test("All zeros genotype", passed);
}

// Test 4: All twos genotype
void test_all_twos() {
    constexpr std::size_t kSamples = 4;
    constexpr std::size_t kLoci = 128;

    auto genotypes = make_uniform(kSamples, kLoci, 2);
    auto packed = kinship::pack_genotypes_2bit(genotypes, kSamples, kLoci);
    const int words = static_cast<int>(kinship::words_per_sample(kLoci));

    uint64_t* d_genotypes = nullptr;
    float* d_similarity = nullptr;
    cudaMalloc(&d_genotypes, packed.size() * sizeof(uint64_t));
    cudaMalloc(&d_similarity, kSamples * kSamples * sizeof(float));
    cudaMemcpy(d_genotypes, packed.data(), packed.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);

    kinship::compute_kinship(d_genotypes, d_similarity, static_cast<int>(kSamples), words);
    kinship::synchronize_or_throw(nullptr);

    std::vector<float> gpu_result(kSamples * kSamples);
    cudaMemcpy(gpu_result.data(), d_similarity, kSamples * kSamples * sizeof(float), cudaMemcpyDeviceToHost);

    // All samples should be identical
    bool passed = true;
    for (std::size_t i = 0; i < kSamples * kSamples; ++i) {
        if (std::fabs(gpu_result[i] - 1.0f) > 1e-5f) {
            passed = false;
            break;
        }
    }

    cudaFree(d_genotypes);
    cudaFree(d_similarity);

    log_test("All twos genotype", passed);
}

// Test 5: Random data
void test_random_data() {
    constexpr std::size_t kSamples = 8;
    constexpr std::size_t kLoci = 256;

    auto genotypes = make_random(kSamples, kLoci, 42);
    auto packed = kinship::pack_genotypes_2bit(genotypes, kSamples, kLoci);
    const int words = static_cast<int>(kinship::words_per_sample(kLoci));

    uint64_t* d_genotypes = nullptr;
    float* d_similarity = nullptr;
    cudaMalloc(&d_genotypes, packed.size() * sizeof(uint64_t));
    cudaMalloc(&d_similarity, kSamples * kSamples * sizeof(float));
    cudaMemcpy(d_genotypes, packed.data(), packed.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);

    kinship::compute_kinship(d_genotypes, d_similarity, static_cast<int>(kSamples), words);
    kinship::synchronize_or_throw(nullptr);

    std::vector<float> gpu_result(kSamples * kSamples);
    cudaMemcpy(gpu_result.data(), d_similarity, kSamples * kSamples * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> reference;
    kinship::compute_kinship_host_reference(packed, static_cast<int>(kSamples), words, reference);

    bool passed = verify_gpu_cpu_match(gpu_result, reference, 1e-4f) &&
                  verify_symmetry(gpu_result, kSamples) &&
                  verify_diagonal(gpu_result, kSamples, 0.99f);

    cudaFree(d_genotypes);
    cudaFree(d_similarity);

    log_test("Random data (8 samples x 256 loci)", passed);
}

// Test 6: Large dataset
void test_large_dataset() {
    constexpr std::size_t kSamples = 64;
    constexpr std::size_t kLoci = 512;

    auto genotypes = make_samples(kSamples, kLoci);
    auto packed = kinship::pack_genotypes_2bit(genotypes, kSamples, kLoci);
    const int words = static_cast<int>(kinship::words_per_sample(kLoci));

    uint64_t* d_genotypes = nullptr;
    float* d_similarity = nullptr;
    cudaMalloc(&d_genotypes, packed.size() * sizeof(uint64_t));
    cudaMalloc(&d_similarity, kSamples * kSamples * sizeof(float));
    cudaMemcpy(d_genotypes, packed.data(), packed.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);

    kinship::compute_kinship(d_genotypes, d_similarity, static_cast<int>(kSamples), words);
    kinship::synchronize_or_throw(nullptr);

    std::vector<float> gpu_result(kSamples * kSamples);
    cudaMemcpy(gpu_result.data(), d_similarity, kSamples * kSamples * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> reference;
    kinship::compute_kinship_host_reference(packed, static_cast<int>(kSamples), words, reference);

    bool passed = verify_gpu_cpu_match(gpu_result, reference, 1e-4f) &&
                  verify_symmetry(gpu_result, kSamples) &&
                  verify_diagonal(gpu_result, kSamples, 0.99f);

    cudaFree(d_genotypes);
    cudaFree(d_similarity);

    log_test("Large dataset (64 samples x 512 loci)", passed);
}

// Test 7: Non-power-of-2 dimensions
void test_non_power_of_2() {
    constexpr std::size_t kSamples = 13;
    constexpr std::size_t kLoci = 157;

    auto genotypes = make_samples(kSamples, kLoci);
    auto packed = kinship::pack_genotypes_2bit(genotypes, kSamples, kLoci);
    const int words = static_cast<int>(kinship::words_per_sample(kLoci));

    uint64_t* d_genotypes = nullptr;
    float* d_similarity = nullptr;
    cudaMalloc(&d_genotypes, packed.size() * sizeof(uint64_t));
    cudaMalloc(&d_similarity, kSamples * kSamples * sizeof(float));
    cudaMemcpy(d_genotypes, packed.data(), packed.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);

    kinship::compute_kinship(d_genotypes, d_similarity, static_cast<int>(kSamples), words);
    kinship::synchronize_or_throw(nullptr);

    std::vector<float> gpu_result(kSamples * kSamples);
    cudaMemcpy(gpu_result.data(), d_similarity, kSamples * kSamples * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> reference;
    kinship::compute_kinship_host_reference(packed, static_cast<int>(kSamples), words, reference);

    bool passed = verify_gpu_cpu_match(gpu_result, reference, 1e-4f) &&
                  verify_symmetry(gpu_result, kSamples) &&
                  verify_diagonal(gpu_result, kSamples, 0.99f);

    cudaFree(d_genotypes);
    cudaFree(d_similarity);

    log_test("Non-power-of-2 dimensions (13 samples x 157 loci)", passed);
}

// Test 8: Normalization
void test_normalization() {
    std::vector<float> matrix = {1.0f, 0.5f, 0.8f, 0.5f, 1.0f, 0.3f, 0.8f, 0.3f, 1.0f};
    kinship::normalize_minmax(matrix);

    // Check that values are in [0, 1]
    bool passed = true;
    for (float val : matrix) {
        if (val < -1e-6f || val > 1.0f + 1e-6f) {
            passed = false;
            break;
        }
    }

    log_test("Normalization min-max", passed);
}

// Test 9: Empty normalization
void test_empty_normalization() {
    std::vector<float> matrix;
    kinship::normalize_minmax(matrix);
    log_test("Empty normalization", matrix.empty());
}

// Test 10: Uniform value normalization
void test_uniform_normalization() {
    std::vector<float> matrix(9, 0.5f);
    kinship::normalize_minmax(matrix);

    bool passed = true;
    for (float val : matrix) {
        if (std::fabs(val - 0.0f) > 1e-6f) {
            passed = false;
            break;
        }
    }

    log_test("Uniform value normalization", passed);
}

}  // namespace

int main() {
    std::cout << "Running cuKINSHIP-Lite test suite...\n\n";

    test_basic_smoke();
    test_single_sample();
    test_all_zeros();
    test_all_twos();
    test_random_data();
    test_large_dataset();
    test_non_power_of_2();
    test_normalization();
    test_empty_normalization();
    test_uniform_normalization();

    std::cout << "\n========================================\n";
    std::cout << "Tests passed: " << tests_passed << "\n";
    std::cout << "Tests failed: " << tests_failed << "\n";
    std::cout << "========================================\n";

    return (tests_failed == 0) ? 0 : 1;
}

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
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

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [OPTIONS]\n"
              << "\nOptions:\n"
              << "  --benchmark         Run benchmark mode with multiple dataset sizes\n"
              << "  --samples N         Number of samples (default: 8)\n"
              << "  --loci M            Number of loci (default: 256)\n"
              << "  --iterations K      Number of iterations for benchmarking (default: 10)\n"
              << "  --help              Display this help message\n"
              << "\nExamples:\n"
              << "  " << prog_name << "                          # Run default smoke test\n"
              << "  " << prog_name << " --samples 100 --loci 1000  # Custom dataset size\n"
              << "  " << prog_name << " --benchmark                # Run full benchmark suite\n";
}

double compute_throughput_gbps(std::size_t n_samples, std::size_t n_loci, double time_ms) {
    // Input data: packed genotypes (2 bits per locus)
    const std::size_t words_per_sample = kinship::words_per_sample(n_loci);
    const std::size_t input_bytes = n_samples * words_per_sample * sizeof(uint64_t);

    // Output data: similarity matrix
    const std::size_t output_bytes = n_samples * n_samples * sizeof(float);

    // Total data transferred
    const std::size_t total_bytes = input_bytes + output_bytes;

    // GB/s (using 1e9 for decimal GB)
    return (total_bytes / 1e9) / (time_ms / 1000.0);
}

double compute_throughput_samples_per_sec(std::size_t n_samples, double time_ms) {
    // Samples per second (pairwise comparisons)
    const std::size_t comparisons = n_samples * (n_samples + 1) / 2;
    return comparisons / (time_ms / 1000.0);
}

void run_benchmark(std::size_t n_samples, std::size_t n_loci, std::size_t iterations) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Benchmark: " << n_samples << " samples × " << n_loci << " loci\n";
    std::cout << std::string(60, '=') << "\n";

    const auto genotypes = generate_mock_genotypes(n_samples, n_loci);
    auto packed = kinship::pack_genotypes_2bit(genotypes, n_samples, n_loci);
    const int words = static_cast<int>(kinship::words_per_sample(n_loci));

    const std::size_t genotypes_bytes = packed.size() * sizeof(uint64_t);
    const std::size_t similarity_bytes = n_samples * n_samples * sizeof(float);

    uint64_t* raw_genotypes = nullptr;
    float* raw_similarity = nullptr;

    if (cudaMalloc(&raw_genotypes, genotypes_bytes) != cudaSuccess ||
        cudaMalloc(&raw_similarity, similarity_bytes) != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed");
    }

    std::unique_ptr<uint64_t, cuda_deleter> d_genotypes(raw_genotypes);
    std::unique_ptr<float, cuda_deleter> d_similarity(raw_similarity);

    if (cudaMemcpy(d_genotypes.get(), packed.data(), genotypes_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy host->device failed");
    }

    std::vector<double> timings;
    timings.reserve(iterations);

    // Warmup run
    kinship::compute_kinship(d_genotypes.get(), d_similarity.get(), static_cast<int>(n_samples), words);
    kinship::synchronize_or_throw(nullptr);

    // Benchmark runs
    for (std::size_t i = 0; i < iterations; ++i) {
        kinship::gpu_timer timer;
        timer.start();
        kinship::compute_kinship(d_genotypes.get(), d_similarity.get(), static_cast<int>(n_samples), words);
        kinship::synchronize_or_throw(nullptr);
        timer.stop();
        timings.push_back(timer.elapsed_milliseconds());
    }

    // Verify correctness (once)
    std::vector<float> similarity(n_samples * n_samples);
    if (cudaMemcpy(similarity.data(), d_similarity.get(), similarity_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy device->host failed");
    }

    std::vector<float> reference;
    kinship::compute_kinship_host_reference(packed, static_cast<int>(n_samples), words, reference);

    float max_error = 0.0f;
    for (std::size_t idx = 0; idx < similarity.size(); ++idx) {
        max_error = std::max(max_error, std::abs(similarity[idx] - reference[idx]));
    }

    // Statistics
    const double mean = std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
    const double min = *std::min_element(timings.begin(), timings.end());
    const double max = *std::max_element(timings.begin(), timings.end());

    double variance = 0.0;
    for (double t : timings) {
        variance += (t - mean) * (t - mean);
    }
    const double stddev = std::sqrt(variance / timings.size());

    std::cout << "\nResults (" << iterations << " iterations):\n";
    std::cout << "  Time (ms):      " << std::fixed << std::setprecision(3)
              << min << " / " << mean << " / " << max
              << " (min/mean/max)\n";
    std::cout << "  Std dev (ms):   " << stddev << "\n";
    std::cout << "  Throughput:     " << std::setprecision(2)
              << compute_throughput_gbps(n_samples, n_loci, mean) << " GB/s\n";
    std::cout << "  Comparisons/s:  " << std::setprecision(0)
              << compute_throughput_samples_per_sec(n_samples, mean) << "\n";
    std::cout << "  Max error:      " << std::scientific << std::setprecision(2)
              << max_error << " (vs CPU)\n";
    std::cout << "  Correctness:    " << (max_error < 1e-4f ? "PASS" : "FAIL") << "\n";
}

struct cuda_deleter {
    void operator()(void* ptr) const noexcept {
        if (ptr) {
            cudaFree(ptr);
        }
    }
};

}  // namespace

int main(int argc, char* argv[]) {
    std::size_t kSamples = 8;
    std::size_t kLoci = 256;
    std::size_t kIterations = 10;
    bool benchmark_mode = false;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (std::strcmp(argv[i], "--benchmark") == 0) {
            benchmark_mode = true;
        } else if (std::strcmp(argv[i], "--samples") == 0 && i + 1 < argc) {
            kSamples = std::stoull(argv[++i]);
        } else if (std::strcmp(argv[i], "--loci") == 0 && i + 1 < argc) {
            kLoci = std::stoull(argv[++i]);
        } else if (std::strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            kIterations = std::stoull(argv[++i]);
        } else {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    try {
        if (benchmark_mode) {
            std::cout << "cuKINSHIP-Lite Benchmark Suite\n";
            std::cout << std::string(60, '=') << "\n";

            // Get GPU info
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            std::cout << "GPU: " << prop.name << "\n";
            std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
            std::cout << "Global Memory: " << (prop.totalGlobalMem / (1024 * 1024 * 1024)) << " GB\n";

            // Run benchmarks with different dataset sizes
            const std::vector<std::pair<std::size_t, std::size_t>> configs = {
                {10, 100},
                {50, 500},
                {100, 1000},
                {500, 5000},
                {1000, 10000}
            };

            for (const auto& [samples, loci] : configs) {
                run_benchmark(samples, loci, kIterations);
            }

            std::cout << "\n" << std::string(60, '=') << "\n";
            std::cout << "Benchmark suite completed\n";
            std::cout << std::string(60, '=') << "\n";

            return 0;
        }

        // Standard mode: single run with verification
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

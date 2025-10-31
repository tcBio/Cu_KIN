#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace kinship {

class gpu_timer {
public:
    explicit gpu_timer(cudaStream_t stream = nullptr) : stream_(stream) {
        throw_if_error(cudaEventCreate(&start_), "cudaEventCreate start");
        throw_if_error(cudaEventCreate(&stop_), "cudaEventCreate stop");
    }

    gpu_timer(const gpu_timer&) = delete;
    gpu_timer& operator=(const gpu_timer&) = delete;

    gpu_timer(gpu_timer&& other) noexcept
        : start_(other.start_), stop_(other.stop_), stream_(other.stream_), running_(other.running_) {
        other.start_ = nullptr;
        other.stop_ = nullptr;
        other.running_ = false;
    }

    gpu_timer& operator=(gpu_timer&& other) noexcept {
        if (this != &other) {
            cleanup();
            start_ = other.start_;
            stop_ = other.stop_;
            stream_ = other.stream_;
            running_ = other.running_;
            other.start_ = nullptr;
            other.stop_ = nullptr;
            other.running_ = false;
        }
        return *this;
    }

    ~gpu_timer() {
        cleanup();
    }

    void start() {
        running_ = true;
        throw_if_error(cudaEventRecord(start_, stream_), "cudaEventRecord start");
    }

    void stop() {
        throw_if_error(cudaEventRecord(stop_, stream_), "cudaEventRecord stop");
        throw_if_error(cudaEventSynchronize(stop_), "cudaEventSynchronize stop");
        running_ = false;
    }

    float elapsed_milliseconds() const {
        float ms = 0.0f;
        throw_if_error(cudaEventElapsedTime(&ms, start_, stop_), "cudaEventElapsedTime");
        return ms;
    }

private:
    static void throw_if_error(cudaError_t status, const char* context) {
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
        }
    }

    void cleanup() noexcept {
        if (start_) {
            cudaEventDestroy(start_);
        }
        if (stop_) {
            cudaEventDestroy(stop_);
        }
        start_ = nullptr;
        stop_ = nullptr;
        running_ = false;
    }

    cudaEvent_t start_{};
    cudaEvent_t stop_{};
    cudaStream_t stream_{};
    bool running_{false};
};

}  // namespace kinship

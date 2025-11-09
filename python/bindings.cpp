// Python bindings for cuKINSHIP-Lite using pybind11
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <vector>

#include "bit_encode.hpp"
#include "gpu_kinship.cuh"
#include "normalize.hpp"

namespace py = pybind11;

namespace {

// RAII wrapper for CUDA memory
struct CudaDeleter {
    void operator()(void* ptr) const {
        if (ptr) cudaFree(ptr);
    }
};

// Main computation function exposed to Python
py::array_t<float> compute_kinship_matrix(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> genotypes,
    bool normalize = true,
    const std::string& kernel_type = "tiled") {

    // Get buffer info
    py::buffer_info buf = genotypes.request();

    if (buf.ndim != 2) {
        throw std::runtime_error("Input must be a 2D array (samples × loci)");
    }

    const std::size_t n_samples = static_cast<std::size_t>(buf.shape[0]);
    const std::size_t n_loci = static_cast<std::size_t>(buf.shape[1]);

    if (n_samples == 0 || n_loci == 0) {
        throw std::runtime_error("Input array must have non-zero dimensions");
    }

    // Convert to flat vector
    const uint8_t* data_ptr = static_cast<uint8_t*>(buf.ptr);
    std::vector<uint8_t> geno_vec(data_ptr, data_ptr + n_samples * n_loci);

    // Validate genotype values (must be 0, 1, or 2)
    for (const auto& val : geno_vec) {
        if (val > 2) {
            throw std::runtime_error("Genotype values must be 0, 1, or 2");
        }
    }

    // Pack genotypes
    auto packed = kinship::pack_genotypes_2bit(geno_vec, n_samples, n_loci);
    const int words = static_cast<int>(kinship::words_per_sample(n_loci));

    // Allocate device memory
    const std::size_t genotypes_bytes = packed.size() * sizeof(uint64_t);
    const std::size_t similarity_bytes = n_samples * n_samples * sizeof(float);

    uint64_t* raw_genotypes = nullptr;
    float* raw_similarity = nullptr;

    cudaError_t status = cudaMalloc(&raw_genotypes, genotypes_bytes);
    if (status != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed for genotypes: " +
                                 std::string(cudaGetErrorString(status)));
    }

    status = cudaMalloc(&raw_similarity, similarity_bytes);
    if (status != cudaSuccess) {
        cudaFree(raw_genotypes);
        throw std::runtime_error("cudaMalloc failed for similarity: " +
                                 std::string(cudaGetErrorString(status)));
    }

    std::unique_ptr<uint64_t, CudaDeleter> d_genotypes(raw_genotypes);
    std::unique_ptr<float, CudaDeleter> d_similarity(raw_similarity);

    // Copy to device
    status = cudaMemcpy(d_genotypes.get(), packed.data(), genotypes_bytes,
                        cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy failed (host->device): " +
                                 std::string(cudaGetErrorString(status)));
    }

    // Select kernel type
    kinship::KernelType kernel = kinship::KernelType::Tiled;
    if (kernel_type == "naive") {
        kernel = kinship::KernelType::Naive;
    } else if (kernel_type != "tiled") {
        throw std::runtime_error("kernel_type must be 'naive' or 'tiled'");
    }

    // Compute kinship
    kinship::compute_kinship(d_genotypes.get(), d_similarity.get(),
                             static_cast<int>(n_samples), words, nullptr, kernel);
    kinship::synchronize_or_throw(nullptr);

    // Copy result back
    std::vector<float> similarity(n_samples * n_samples);
    status = cudaMemcpy(similarity.data(), d_similarity.get(), similarity_bytes,
                        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy failed (device->host): " +
                                 std::string(cudaGetErrorString(status)));
    }

    // Normalize if requested
    if (normalize) {
        kinship::normalize_similarity(similarity);
    }

    // Create numpy array
    py::array_t<float> result({n_samples, n_samples});
    py::buffer_info result_buf = result.request();
    float* result_ptr = static_cast<float*>(result_buf.ptr);

    std::copy(similarity.begin(), similarity.end(), result_ptr);

    return result;
}

// Get GPU information
py::dict get_gpu_info() {
    py::dict info;

    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);

    if (status != cudaSuccess) {
        info["available"] = false;
        info["error"] = cudaGetErrorString(status);
        return info;
    }

    info["available"] = true;
    info["device_count"] = device_count;

    if (device_count > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        info["name"] = std::string(prop.name);
        info["compute_capability"] = std::to_string(prop.major) + "." +
                                      std::to_string(prop.minor);
        info["total_memory_gb"] = prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0);
        info["multiprocessors"] = prop.multiProcessorCount;
    }

    return info;
}

}  // namespace

PYBIND11_MODULE(_cukinship, m) {
    m.doc() = "cuKINSHIP-Lite: GPU-accelerated kinship matrix computation";

    m.def("compute_kinship", &compute_kinship_matrix,
          py::arg("genotypes"),
          py::arg("normalize") = true,
          py::arg("kernel_type") = "tiled",
          R"pbdoc(
        Compute kinship/similarity matrix from genotype data.

        Parameters
        ----------
        genotypes : ndarray, shape (n_samples, n_loci)
            Genotype matrix with values in {0, 1, 2}.
            Each row is a sample, each column is a locus.
        normalize : bool, default=True
            Whether to apply min-max normalization to [0, 1].
        kernel_type : str, default='tiled'
            GPU kernel implementation: 'naive' or 'tiled'.
            'tiled' uses shared memory for better performance.

        Returns
        -------
        similarity : ndarray, shape (n_samples, n_samples)
            Symmetric similarity matrix.
            Values range from 0 (completely different) to 1 (identical).

        Examples
        --------
        >>> import numpy as np
        >>> import cukinship
        >>> genotypes = np.random.randint(0, 3, size=(10, 100), dtype=np.uint8)
        >>> similarity = cukinship.compute_kinship(genotypes)
        >>> similarity.shape
        (10, 10)
        >>> np.allclose(similarity, similarity.T)  # Check symmetry
        True
    )pbdoc");

    m.def("gpu_info", &get_gpu_info,
          R"pbdoc(
        Get information about available CUDA GPUs.

        Returns
        -------
        info : dict
            Dictionary with GPU information:
            - available: bool, whether CUDA is available
            - device_count: int, number of CUDA devices
            - name: str, GPU model name
            - compute_capability: str, CUDA compute capability
            - total_memory_gb: float, total GPU memory in GB
            - multiprocessors: int, number of streaming multiprocessors

        Examples
        --------
        >>> import cukinship
        >>> info = cukinship.gpu_info()
        >>> print(f"GPU: {info['name']}, Memory: {info['total_memory_gb']:.1f} GB")
    )pbdoc");

    m.attr("__version__") = "0.1.0";
}

#include "bit_encode.hpp"

#include <stdexcept>

#include "gpu_kinship.cuh"

namespace kinship {

std::vector<uint64_t> pack_genotypes_2bit(const std::vector<uint8_t>& genotypes,
                                          std::size_t n_samples,
                                          std::size_t n_loci) {
    if (genotypes.size() != n_samples * n_loci) {
        throw std::invalid_argument("pack_genotypes_2bit: input size mismatch");
    }

    const std::size_t words_per_sample_count = words_per_sample(n_loci);
    std::vector<uint64_t> packed(n_samples * words_per_sample_count, 0ULL);

    for (std::size_t sample = 0; sample < n_samples; ++sample) {
        for (std::size_t locus = 0; locus < n_loci; ++locus) {
            const std::uint8_t geno = genotypes[sample * n_loci + locus] & 0x3u;
            const std::size_t word_idx = locus / kLociPerWord;
            const std::size_t bit_offset = (locus % kLociPerWord) * 2;
            packed[sample * words_per_sample_count + word_idx] |= static_cast<std::uint64_t>(geno) << bit_offset;
        }
    }

    return packed;
}

}  // namespace kinship

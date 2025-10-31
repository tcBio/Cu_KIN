#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace kinship {

std::vector<uint64_t> pack_genotypes_2bit(const std::vector<uint8_t>& genotypes,
                                          std::size_t n_samples,
                                          std::size_t n_loci);

}  // namespace kinship

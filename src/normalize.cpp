#include "normalize.hpp"

#include <algorithm>
#include <cmath>

namespace kinship {

void normalize_similarity(std::vector<float>& kinship_matrix) {
    if (kinship_matrix.empty()) {
        return;
    }

    const auto [min_it, max_it] = std::minmax_element(kinship_matrix.begin(), kinship_matrix.end());
    const float min_val = *min_it;
    const float max_val = *max_it;

    if (std::fabs(max_val - min_val) < 1e-6f) {
        return;
    }

    const float scale = 1.0f / (max_val - min_val);
    for (float& entry : kinship_matrix) {
        entry = (entry - min_val) * scale;
    }
}

}  // namespace kinship

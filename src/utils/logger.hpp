#pragma once

#include <cstdio>
#include <string_view>

namespace kinship {

inline void log(std::string_view level, std::string_view message) {
    std::fprintf(stdout, "[%.*s] %.*s\n",
                 static_cast<int>(level.size()), level.data(),
                 static_cast<int>(message.size()), message.data());
}

}  // namespace kinship

#define KINSHIP_LOG_INFO(msg) ::kinship::log("INFO", (msg))
#define KINSHIP_LOG_ERROR(msg) ::kinship::log("ERROR", (msg))

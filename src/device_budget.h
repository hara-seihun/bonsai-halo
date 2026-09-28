#pragma once
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace halo {

inline size_t device_memory_limit() {
    const char * value = std::getenv("BONSAI_DEVICE_MEMORY_MAX_BYTES");
    if (!value || !*value) return std::numeric_limits<size_t>::max();
    errno = 0;
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno || end == value || *end || parsed > std::numeric_limits<size_t>::max())
        throw std::runtime_error("BONSAI_DEVICE_MEMORY_MAX_BYTES must be an integer byte count");
    return (size_t) parsed;
}

inline void admit_device_bytes(size_t current, size_t additional, const char * operation) {
    const size_t limit = device_memory_limit();
    if (additional > limit || current > limit - additional)
        throw std::runtime_error(std::string(operation) + " would exceed "
            + std::to_string(limit) + " tracked device bytes from a current "
            + std::to_string(current) + " bytes");
}

} // namespace halo

#pragma once
#include "gguf.h"
#include <string>
#include <vector>
#include <cstdint>

namespace halo {

// Repack one ternary GGUF tensor (PTQ1_0 or PQ2_0, [K, N] in ggml order) into HALO tiles.
// dst must hold halo_tensor_bytes(N, K). Uses nthreads worker threads.
void repack_tensor(const GgufTensor & t, uint8_t * dst, int nthreads);

// Self-test of the pack/peel pair over every 5-trit and 4-trit combination.
void halo_format_selftest();

struct HaloCacheEntry { std::string name; int64_t N, K; uint64_t offset, bytes; };

// Cache of all repacked ternary tensors for one GGUF file. The cache file is
// <gguf path>.halo; it is rebuilt when missing or stale (size/mtime of the source changed).
struct HaloCache {
    std::vector<HaloCacheEntry> entries;
    int fd = -1;
    size_t map_size = 0;
    const uint8_t * base = nullptr;
    ~HaloCache();
    // returns true if the cache was (re)built
    bool open_or_build(const Gguf & g, const std::string & gguf_path, int nthreads);
    const HaloCacheEntry & entry(const std::string & name) const;
    const uint8_t * data(const HaloCacheEntry & e) const { return base + e.offset; }
};

} // namespace halo

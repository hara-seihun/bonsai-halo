// Drafter weights: bf16 safetensors quantised to tiles and cached next to the safetensors file.
// Two coordinates, one cache machine: Q8 (32 rows x 128 int8 + 32 fp16 scales per block, the HALO
// tile/row mapping) and Q4 (the same rows and blocks with packed nibbles, kernels/q4_format.h).
// A cache file carries its coordinate in its suffix and magic, so both can sit beside one drafter
// and an A/B needs no rebuild.
#pragma once
#include <string>
#include <vector>
#include <map>
#include <cstdint>

namespace halo {

struct StTensor { std::string dtype; std::vector<int64_t> shape; size_t begin, end; };

struct Safetensors {
    std::map<std::string, StTensor> tensors;
    const uint8_t * base = nullptr; size_t map_size = 0; size_t data_off = 0;
    ~Safetensors();
    void open(const std::string & path);
    const StTensor & t(const std::string & name) const;
    const uint8_t * data(const StTensor & t) const { return base + data_off + t.begin; }
    std::vector<float> f32(const std::string & name) const; // bf16/f32 -> f32
};

size_t q8_tensor_bytes(int64_t N, int64_t K);
// dst holds q8_tensor_bytes(N, K); src is bf16 [N][K]
void quantize_q8_tiles(const uint16_t * src, int64_t N, int64_t K, uint8_t * dst, int nthreads);

// Q4 tiles, kernels/q4_format.h. Returns sum of squared quantisation error and sum of squares of
// the source, so a cache build can report what the coordinate costs.
struct QuantError { double err2 = 0, ref2 = 0; };
QuantError quantize_q4_tiles(const uint16_t * src, int64_t N, int64_t K, uint8_t * dst, int nthreads);

struct Q8Entry { std::string name; int64_t N, K; uint64_t offset, bytes; };
struct Q8Cache {
    std::vector<Q8Entry> entries;
    int fd = -1; size_t map_size = 0; const uint8_t * base = nullptr;
    int bits = 8;
    ~Q8Cache();
    // names: tensors to quantise (2D bf16); bits picks the coordinate (8 or 4). Rebuilds when
    // missing/stale. Independent cache files per coordinate.
    bool open_or_build(const Safetensors & st, const std::string & st_path, const std::vector<std::string> & names, int nthreads, int bits = 8);
    const Q8Entry & entry(const std::string & name) const;
    const uint8_t * data(const Q8Entry & e) const { return base + e.offset; }
};

} // namespace halo

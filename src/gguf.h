// Minimal GGUF v3 reader: metadata, tensor table, mmap of the data section.
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <variant>

namespace halo {

enum GgufType : uint32_t {
    GGUF_U8 = 0, GGUF_I8, GGUF_U16, GGUF_I16, GGUF_U32, GGUF_I32, GGUF_F32, GGUF_BOOL,
    GGUF_STRING, GGUF_ARRAY, GGUF_U64, GGUF_I64, GGUF_F64
};

// ggml tensor type ids we care about
enum GgmlType : uint32_t {
    GGML_F32 = 0, GGML_F16 = 1, GGML_BF16 = 30, GGML_Q1_0 = 41, GGML_PQ2_0 = 142, GGML_PTQ1_0 = 143
};

struct GgufValue {
    GgufType type;
    GgufType elem_type = GGUF_U8;           // for arrays
    std::vector<uint8_t> raw;               // scalars: raw bytes; arrays of scalars: packed
    std::vector<std::string> strings;       // string or array of strings
    int64_t as_int() const;
    double  as_float() const;
    const std::string & as_string() const { return strings.at(0); }
    std::vector<int64_t> as_int_array() const;
};

struct GgufTensor {
    std::string name;
    std::vector<int64_t> ne;   // ne[0] is the fastest (row length)
    GgmlType type;
    uint64_t offset;           // relative to data section
    size_t nbytes;
    const uint8_t * data = nullptr;
    int64_t rows() const { int64_t r = 1; for (size_t i = 1; i < ne.size(); i++) r *= ne[i]; return r; }
};

struct Gguf {
    std::map<std::string, GgufValue> kv;
    std::vector<GgufTensor> tensors;
    std::map<std::string, size_t> index;
    void * map_base = nullptr;
    size_t map_size = 0;
    uint64_t data_offset = 0;

    ~Gguf();
    void open(const std::string & path);
    const GgufTensor & tensor(const std::string & name) const;
    const GgufTensor * find(const std::string & name) const;
    bool has(const std::string & key) const { return kv.count(key) != 0; }
    const GgufValue & get(const std::string & key) const;
};

size_t ggml_type_row_bytes(GgmlType t, int64_t ne0);

} // namespace halo

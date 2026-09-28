#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace halo {

static size_t scalar_size(GgufType t) {
    switch (t) {
        case GGUF_U8: case GGUF_I8: case GGUF_BOOL: return 1;
        case GGUF_U16: case GGUF_I16: return 2;
        case GGUF_U32: case GGUF_I32: case GGUF_F32: return 4;
        case GGUF_U64: case GGUF_I64: case GGUF_F64: return 8;
        default: throw std::runtime_error("gguf: not a scalar type");
    }
}

int64_t GgufValue::as_int() const {
    switch (type) {
        case GGUF_U8:  return raw[0];
        case GGUF_I8:  return (int8_t) raw[0];
        case GGUF_BOOL: return raw[0] != 0;
        case GGUF_U16: { uint16_t v; memcpy(&v, raw.data(), 2); return v; }
        case GGUF_I16: { int16_t v; memcpy(&v, raw.data(), 2); return v; }
        case GGUF_U32: { uint32_t v; memcpy(&v, raw.data(), 4); return v; }
        case GGUF_I32: { int32_t v; memcpy(&v, raw.data(), 4); return v; }
        case GGUF_U64: { uint64_t v; memcpy(&v, raw.data(), 8); return (int64_t) v; }
        case GGUF_I64: { int64_t v; memcpy(&v, raw.data(), 8); return v; }
        default: throw std::runtime_error("gguf: value is not an integer");
    }
}

double GgufValue::as_float() const {
    if (type == GGUF_F32) { float v; memcpy(&v, raw.data(), 4); return v; }
    if (type == GGUF_F64) { double v; memcpy(&v, raw.data(), 8); return v; }
    return (double) as_int();
}

std::vector<int64_t> GgufValue::as_int_array() const {
    if (type != GGUF_ARRAY) throw std::runtime_error("gguf: not an array");
    size_t sz = scalar_size(elem_type);
    std::vector<int64_t> out(raw.size() / sz);
    for (size_t i = 0; i < out.size(); i++) {
        GgufValue tmp; tmp.type = elem_type; tmp.raw.assign(raw.begin() + i * sz, raw.begin() + (i + 1) * sz);
        out[i] = tmp.as_int();
    }
    return out;
}

size_t ggml_type_row_bytes(GgmlType t, int64_t ne0) {
    switch (t) {
        case GGML_F32: return 4 * ne0;
        case GGML_F16: case GGML_BF16: return 2 * ne0;
        case GGML_Q1_0: return (ne0 / 32) * 6;       // block 32: fp16 + 4 bytes
        case GGML_PQ2_0: return (ne0 / 128) * 34;    // block 128: fp16 + 32 bytes
        case GGML_PTQ1_0: return (ne0 / 128) * 28;   // block 128: 24 + 2 + fp16
        default: throw std::runtime_error("gguf: unsupported tensor type " + std::to_string(t));
    }
}

namespace {
struct Reader {
    const uint8_t * p; const uint8_t * end;
    template <typename T> T rd() { if (p + sizeof(T) > end) throw std::runtime_error("gguf: truncated"); T v; memcpy(&v, p, sizeof(T)); p += sizeof(T); return v; }
    std::string str() { uint64_t n = rd<uint64_t>(); if (p + n > end) throw std::runtime_error("gguf: truncated string"); std::string s((const char *) p, n); p += n; return s; }
    void bytes(std::vector<uint8_t> & out, size_t n) { if (p + n > end) throw std::runtime_error("gguf: truncated"); out.insert(out.end(), p, p + n); p += n; }
};
}

Gguf::~Gguf() { if (map_base) munmap(map_base, map_size); }

void Gguf::open(const std::string & path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path);
    struct stat st; fstat(fd, &st);
    map_size = st.st_size;
    map_base = mmap(nullptr, map_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map_base == MAP_FAILED) { map_base = nullptr; throw std::runtime_error("mmap failed for " + path); }
    madvise(map_base, map_size, MADV_SEQUENTIAL);

    Reader r { (const uint8_t *) map_base, (const uint8_t *) map_base + map_size };
    if (r.rd<uint32_t>() != 0x46554747) throw std::runtime_error("not a GGUF file: " + path);
    uint32_t version = r.rd<uint32_t>();
    if (version != 3 && version != 2) throw std::runtime_error("unsupported GGUF version");
    uint64_t n_tensors = r.rd<uint64_t>();
    uint64_t n_kv = r.rd<uint64_t>();

    for (uint64_t i = 0; i < n_kv; i++) {
        std::string key = r.str();
        GgufValue v; v.type = (GgufType) r.rd<uint32_t>();
        if (v.type == GGUF_STRING) {
            v.strings.push_back(r.str());
        } else if (v.type == GGUF_ARRAY) {
            v.elem_type = (GgufType) r.rd<uint32_t>();
            uint64_t n = r.rd<uint64_t>();
            if (v.elem_type == GGUF_STRING) { v.strings.reserve(n); for (uint64_t j = 0; j < n; j++) v.strings.push_back(r.str()); }
            else r.bytes(v.raw, n * scalar_size(v.elem_type));
        } else {
            r.bytes(v.raw, scalar_size(v.type));
        }
        kv.emplace(std::move(key), std::move(v));
    }

    tensors.resize(n_tensors);
    for (uint64_t i = 0; i < n_tensors; i++) {
        GgufTensor & t = tensors[i];
        t.name = r.str();
        uint32_t nd = r.rd<uint32_t>();
        t.ne.resize(nd);
        for (uint32_t d = 0; d < nd; d++) t.ne[d] = (int64_t) r.rd<uint64_t>();
        t.type = (GgmlType) r.rd<uint32_t>();
        t.offset = r.rd<uint64_t>();
        t.nbytes = ggml_type_row_bytes(t.type, t.ne[0]) * t.rows();
        index[t.name] = i;
    }
    uint64_t alignment = has("general.alignment") ? get("general.alignment").as_int() : 32;
    uint64_t pos = r.p - (const uint8_t *) map_base;
    data_offset = (pos + alignment - 1) / alignment * alignment;
    for (auto & t : tensors) {
        t.data = (const uint8_t *) map_base + data_offset + t.offset;
        if (t.data + t.nbytes > (const uint8_t *) map_base + map_size) throw std::runtime_error("gguf: tensor " + t.name + " exceeds file");
    }
}

const GgufValue & Gguf::get(const std::string & key) const {
    auto it = kv.find(key);
    if (it == kv.end()) throw std::runtime_error("gguf: missing key " + key);
    return it->second;
}

const GgufTensor * Gguf::find(const std::string & name) const {
    auto it = index.find(name);
    return it == index.end() ? nullptr : &tensors[it->second];
}

const GgufTensor & Gguf::tensor(const std::string & name) const {
    const GgufTensor * t = find(name);
    if (!t) throw std::runtime_error("gguf: missing tensor " + name);
    return *t;
}

} // namespace halo

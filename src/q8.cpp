#include "q8.h"
#include "halo_format.h"
#include "q4_format.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <thread>
#include <atomic>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace halo {

static float bf16f(uint16_t b) { uint32_t u = (uint32_t) b << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t f16bits(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000; int exp = (int) ((x >> 23) & 0xff) - 127 + 15; uint32_t mant = x & 0x7fffff;
    if (exp <= 0) return (uint16_t) sign;
    if (exp >= 31) return (uint16_t) (sign | 0x7c00);
    uint32_t h = sign | (exp << 10) | (mant >> 13);
    if (mant & 0x1000) h++;
    return (uint16_t) h;
}
// fp16 bits -> float, so a quantiser can score itself against the scale the kernel will read
static float halff(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f, mant = h & 0x3ff, u;
    if (exp == 0) {
        if (mant == 0) u = sign;
        else { int e = -1; uint32_t m = mant; do { m <<= 1; e++; } while (!(m & 0x400)); u = sign | ((uint32_t) (112 - e) << 23) | ((m & 0x3ff) << 13); }
    } else if (exp == 31) u = sign | 0x7f800000u | (mant << 13);
    else u = sign | ((exp + 112) << 23) | (mant << 13);
    float f; memcpy(&f, &u, 4); return f;
}

Safetensors::~Safetensors() { if (base) munmap((void *) base, map_size); }

void Safetensors::open(const std::string & path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path);
    struct stat st; fstat(fd, &st); map_size = st.st_size;
    base = (const uint8_t *) mmap(nullptr, map_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (base == MAP_FAILED) { base = nullptr; throw std::runtime_error("mmap failed: " + path); }
    uint64_t n; memcpy(&n, base, 8);
    std::string hdr((const char *) base + 8, n);
    data_off = 8 + n;
    // minimal JSON walk: "name":{"dtype":"BF16","shape":[a,b],"data_offsets":[x,y]}
    size_t p = 0;
    while (true) {
        size_t q = hdr.find("\"dtype\"", p);
        if (q == std::string::npos) break;
        // name: the last quoted string before the '{' preceding this dtype
        size_t brace = hdr.rfind('{', q);
        size_t nend = hdr.rfind('"', brace); size_t nbeg = hdr.rfind('"', nend - 1);
        std::string name = hdr.substr(nbeg + 1, nend - nbeg - 1);
        StTensor t;
        size_t d0 = hdr.find(':', q + 7); d0 = hdr.find('"', d0); size_t d1 = hdr.find('"', d0 + 1);
        t.dtype = hdr.substr(d0 + 1, d1 - d0 - 1);
        size_t s0 = hdr.find("\"shape\"", q); s0 = hdr.find('[', s0); size_t s1 = hdr.find(']', s0);
        { std::string sh = hdr.substr(s0 + 1, s1 - s0 - 1); size_t i = 0; while (i < sh.size()) { size_t j = sh.find(',', i); if (j == std::string::npos) j = sh.size(); if (j > i) t.shape.push_back(std::stoll(sh.substr(i, j - i))); i = j + 1; } }
        size_t o0 = hdr.find("\"data_offsets\"", q); o0 = hdr.find('[', o0); size_t o1 = hdr.find(']', o0);
        { std::string of = hdr.substr(o0 + 1, o1 - o0 - 1); size_t c = of.find(','); t.begin = std::stoull(of.substr(0, c)); t.end = std::stoull(of.substr(c + 1)); }
        tensors[name] = t;
        p = o1;
    }
}

const StTensor & Safetensors::t(const std::string & name) const {
    auto it = tensors.find(name);
    if (it == tensors.end()) throw std::runtime_error("safetensors: missing " + name);
    return it->second;
}

std::vector<float> Safetensors::f32(const std::string & name) const {
    const StTensor & tt = t(name);
    size_t n = 1; for (auto d : tt.shape) n *= d;
    std::vector<float> out(n);
    const uint8_t * d = data(tt);
    if (tt.dtype == "BF16") { const uint16_t * s = (const uint16_t *) d; for (size_t i = 0; i < n; i++) out[i] = bf16f(s[i]); }
    else if (tt.dtype == "F32") memcpy(out.data(), d, n * 4);
    else throw std::runtime_error("safetensors: unsupported dtype " + tt.dtype + " for " + name);
    return out;
}

size_t q8_tensor_bytes(int64_t N, int64_t K) { return (size_t) (N / TILE_ROWS) * (K / BLOCK) * (TILE_ROWS * 128 + TILE_ROWS * 2); }

void quantize_q8_tiles(const uint16_t * src, int64_t N, int64_t K, uint8_t * dst, int nthreads) {
    if (N % TILE_ROWS || K % BLOCK) throw std::runtime_error("q8: shape not tileable");
    const int64_t nb = K / BLOCK, ntiles = N / TILE_ROWS;
    const size_t BB = TILE_ROWS * 128 + TILE_ROWS * 2;
    std::atomic<int64_t> next{0};
    auto worker = [&]() {
        for (;;) {
            int64_t T = next.fetch_add(1);
            if (T >= ntiles) break;
            uint8_t * tile = dst + (size_t) T * nb * BB;
            for (int64_t b = 0; b < nb; b++) {
                uint8_t * run = tile + b * BB;
                for (int l = 0; l < TILE_ROWS; l++) {
                    const uint16_t * row = src + (size_t) (T * TILE_ROWS + l) * K + b * BLOCK;
                    float amax = 0.0f;
                    for (int i = 0; i < BLOCK; i++) amax = std::max(amax, std::fabs(bf16f(row[i])));
                    const float scale = amax / 127.0f, inv = amax > 0 ? 127.0f / amax : 0.0f;
                    int8_t * q = (int8_t *) (run + l * 128);
                    for (int i = 0; i < BLOCK; i++) { int v = (int) std::lround(bf16f(row[i]) * inv); q[i] = (int8_t) std::max(-127, std::min(127, v)); }
                    uint16_t sc = f16bits(scale);
                    memcpy(run + TILE_ROWS * 128 + l * 2, &sc, 2);
                }
            }
        }
    };
    std::vector<std::thread> th;
    for (int i = 0; i < nthreads; i++) th.emplace_back(worker);
    for (auto & x : th) x.join();
}

// Q4: one fp16 scale per (row, 128-block) and offset-binary codes in [1, 15].
//
// The scale is not amax/7. Four bits leave the largest element of a block worth 14% of its own
// magnitude in rounding error, and the block's own amax is a poor place to spend that: shrinking
// the step clips the few extremes and rounds everything else finer. The build searches sixteen
// steps between 0.53 and 1.00 of amax/7 and keeps the one with the least squared error over the
// block. It is a host-side cache cost paid once and it changes no kernel instruction.
static void quantize_q4_block(const float * w, uint8_t * bytes, uint16_t * scale_out, double & err2, double & ref2) {
    float amax = 0.0f;
    for (int i = 0; i < BLOCK; i++) { const float a = std::fabs(w[i]); if (a > amax) amax = a; }
    float best_s = 0.0f, best_e = 0.0f;
    if (amax > 0.0f) {
        bool first = true;
        for (int k = 0; k < 16; k++) {
            const float s = (amax / 7.0f) * (1.0f - 0.03125f * (float) k);
            const float inv = 1.0f / s;
            float e = 0.0f;
            for (int i = 0; i < BLOCK; i++) {
                int q = (int) std::lround(w[i] * inv);
                q = std::max(-7, std::min(7, q));
                const float d = w[i] - (float) q * s;
                e += d * d;
            }
            if (first || e < best_e) { best_e = e; best_s = s; first = false; }
        }
    }
    const float inv = best_s > 0.0f ? 1.0f / best_s : 0.0f;
    // Store through the fp16 scale the kernel will actually read, so the reported error is the
    // error the drafter runs with rather than the error of an exact-scale fiction.
    const uint16_t sc = f16bits(best_s);
    const float s_used = halff(sc);
    for (int i = 0; i < Q4_ROW_BYTES; i++) bytes[i] = 0;
    for (int e = 0; e < BLOCK; e++) {
        int q = (int) std::lround(w[e] * inv);
        q = std::max(-7, std::min(7, q));
        const int code = q + Q4_ZERO;
        uint8_t & B = bytes[q4_byte_of(e)];
        B = (uint8_t) (q4_high_of(e) ? ((B & 0x0f) | (code << 4)) : ((B & 0xf0) | code));
        const float d = w[e] - (float) q * s_used;
        err2 += (double) d * d; ref2 += (double) w[e] * w[e];
    }
    *scale_out = sc;
}

QuantError quantize_q4_tiles(const uint16_t * src, int64_t N, int64_t K, uint8_t * dst, int nthreads) {
    if (N % TILE_ROWS || K % BLOCK) throw std::runtime_error("q4: shape not tileable");
    const int64_t nb = K / BLOCK, ntiles = N / TILE_ROWS;
    std::atomic<int64_t> next{0};
    std::vector<QuantError> per(nthreads);
    auto worker = [&](int tid) {
        float w[BLOCK];
        for (;;) {
            int64_t T = next.fetch_add(1);
            if (T >= ntiles) break;
            uint8_t * tile = dst + (size_t) T * nb * Q4_TILE_BLOCK_BYTES;
            for (int64_t b = 0; b < nb; b++) {
                uint8_t * run = tile + b * Q4_TILE_BLOCK_BYTES;
                for (int l = 0; l < TILE_ROWS; l++) {
                    const uint16_t * row = src + (size_t) (T * TILE_ROWS + l) * K + b * BLOCK;
                    for (int i = 0; i < BLOCK; i++) w[i] = bf16f(row[i]);
                    uint16_t sc;
                    quantize_q4_block(w, run + q4_off_row(l), &sc, per[tid].err2, per[tid].ref2);
                    memcpy(run + q4_off_scale(l), &sc, 2);
                }
            }
        }
    };
    std::vector<std::thread> th;
    for (int i = 0; i < nthreads; i++) th.emplace_back(worker, i);
    for (auto & x : th) x.join();
    QuantError tot;
    for (auto & e : per) { tot.err2 += e.err2; tot.ref2 += e.ref2; }
    return tot;
}

Q8Cache::~Q8Cache() { if (base) munmap((void *) base, map_size); if (fd >= 0) close(fd); }

struct Q8Header { char magic[8]; uint64_t src_size; uint64_t src_mtime; uint64_t n_entries; };

bool Q8Cache::open_or_build(const Safetensors & st, const std::string & st_path, const std::vector<std::string> & names, int nthreads, int bits_) {
    struct stat sb; if (stat(st_path.c_str(), &sb)) throw std::runtime_error("stat failed: " + st_path);
    bits = bits_;
    if (bits != 8 && bits != 4) throw std::runtime_error("drafter cache: bits must be 8 or 4");
    const char * magic_want = bits == 8 ? "HALOQ8_1" : "HALOQ4_1";
    std::string cache_path = st_path + (bits == 8 ? ".q8" : ".q4");
    auto try_open = [&]() -> bool {
        fd = ::open(cache_path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat cs; fstat(fd, &cs);
        if ((size_t) cs.st_size < sizeof(Q8Header)) { close(fd); fd = -1; return false; }
        map_size = cs.st_size;
        base = (const uint8_t *) mmap(nullptr, map_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (base == MAP_FAILED) { base = nullptr; close(fd); fd = -1; return false; }
        Q8Header h; memcpy(&h, base, sizeof h);
        if (memcmp(h.magic, magic_want, 8) || h.src_size != (uint64_t) sb.st_size || h.src_mtime != (uint64_t) sb.st_mtime || h.n_entries != names.size()) {
            munmap((void *) base, map_size); base = nullptr; close(fd); fd = -1; return false;
        }
        const uint8_t * p = base + sizeof h;
        entries.clear();
        for (uint64_t i = 0; i < h.n_entries; i++) {
            uint32_t nl; memcpy(&nl, p, 4); p += 4;
            Q8Entry e; e.name.assign((const char *) p, nl); p += nl;
            memcpy(&e.N, p, 8); p += 8; memcpy(&e.K, p, 8); p += 8; memcpy(&e.offset, p, 8); p += 8; memcpy(&e.bytes, p, 8); p += 8;
            entries.push_back(e);
        }
        for (size_t i = 0; i < names.size(); i++) if (entries[i].name != names[i]) { munmap((void *) base, map_size); base = nullptr; close(fd); fd = -1; return false; }
        return true;
    };
    if (try_open()) return false;

    auto t0 = std::chrono::steady_clock::now();
    size_t index_bytes = sizeof(Q8Header);
    for (auto & n : names) index_bytes += 4 + n.size() + 32;
    size_t total = (index_bytes + 4095) / 4096 * 4096;
    entries.clear();
    for (auto & n : names) {
        const StTensor & t = st.t(n);
        if (t.shape.size() != 2 || t.dtype != "BF16") throw std::runtime_error("q8: " + n + " is not a 2D bf16 tensor");
        Q8Entry e; e.name = n; e.N = t.shape[0]; e.K = t.shape[1]; e.offset = total;
        e.bytes = bits == 8 ? q8_tensor_bytes(e.N, e.K) : q4_tensor_bytes(e.N, e.K);
        total += (e.bytes + 4095) / 4096 * 4096;
        entries.push_back(e);
    }
    std::string tmp = cache_path + ".tmp";
    int wfd = ::open(tmp.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (wfd < 0) throw std::runtime_error("cannot create " + tmp);
    if (ftruncate(wfd, total)) throw std::runtime_error("ftruncate failed");
    uint8_t * wbase = (uint8_t *) mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, wfd, 0);
    if (wbase == MAP_FAILED) throw std::runtime_error("mmap rw failed");
    Q8Header h; memcpy(h.magic, magic_want, 8); h.src_size = sb.st_size; h.src_mtime = sb.st_mtime; h.n_entries = names.size();
    uint8_t * p = wbase; memcpy(p, &h, sizeof h); p += sizeof h;
    for (auto & e : entries) {
        uint32_t nl = e.name.size(); memcpy(p, &nl, 4); p += 4; memcpy(p, e.name.data(), nl); p += nl;
        memcpy(p, &e.N, 8); p += 8; memcpy(p, &e.K, 8); p += 8; memcpy(p, &e.offset, 8); p += 8; memcpy(p, &e.bytes, 8); p += 8;
    }
    QuantError qe;
    for (size_t i = 0; i < names.size(); i++) {
        fprintf(stderr, "\rquantising q%d %zu/%zu %-60s", bits, i + 1, names.size(), names[i].c_str());
        const uint16_t * s = (const uint16_t *) st.data(st.t(names[i]));
        if (bits == 8) quantize_q8_tiles(s, entries[i].N, entries[i].K, wbase + entries[i].offset, nthreads);
        else { QuantError e = quantize_q4_tiles(s, entries[i].N, entries[i].K, wbase + entries[i].offset, nthreads); qe.err2 += e.err2; qe.ref2 += e.ref2; }
    }
    fprintf(stderr, "\n");
    msync(wbase, total, MS_SYNC); munmap(wbase, total); close(wfd);
    if (rename(tmp.c_str(), cache_path.c_str())) throw std::runtime_error("rename failed");
    fprintf(stderr, "q%d cache built: %s (%.2f GB) in %.1f s", bits, cache_path.c_str(), total / 1e9, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    if (bits == 4 && qe.ref2 > 0) fprintf(stderr, "; relative RMS weight error %.4f", std::sqrt(qe.err2 / qe.ref2));
    fprintf(stderr, "\n");
    if (!try_open()) throw std::runtime_error("q8 cache reopen failed");
    return true;
}

const Q8Entry & Q8Cache::entry(const std::string & name) const {
    for (auto & e : entries) if (e.name == name) return e;
    throw std::runtime_error("q8 cache: missing " + name);
}

} // namespace halo

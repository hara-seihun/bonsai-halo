#include "repack.h"
#include "halo_format.h"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <atomic>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace halo {

void halo_format_selftest() {
    uint8_t t[5], back[5];
    for (int q = 0; q < 243; q++) {
        int v = q; for (int n = 4; n >= 0; n--) { t[n] = v % 3; v /= 3; }
        unsigned b = pack5(t);
        for (int n = 0; n < 5; n++) { unsigned m = b * 3u; back[n] = m >> 8; b = m & 0xff; }
        if (memcmp(t, back, 5)) throw std::runtime_error("halo pack5/peel self-test failed");
    }
    for (int q = 0; q < 81; q++) {
        int v = q; for (int n = 3; n >= 0; n--) { t[n] = v % 3; v /= 3; }
        unsigned b = pack4(t);
        for (int n = 0; n < 4; n++) { unsigned m = b * 3u; back[n] = m >> 8; b = m & 0xff; }
        if (memcmp(t, back, 4)) throw std::runtime_error("halo pack4/peel self-test failed");
    }
    // block round trip
    uint8_t trit[128], qs[24], qh[2], out[128];
    for (int i = 0; i < 128; i++) trit[i] = (i * 7 + i / 5) % 3;
    encode_block(trit, qs, qh); decode_block(qs, qh, out);
    if (memcmp(trit, out, 128)) throw std::runtime_error("halo block round trip failed");
}

void repack_tensor(const GgufTensor & t, uint8_t * dst, int nthreads) {
    const int64_t K = t.ne[0], N = t.rows();
    if (K % BLOCK || N % TILE_ROWS) throw std::runtime_error("repack: bad shape for " + t.name);
    const int64_t nb = K / BLOCK, ntiles = N / TILE_ROWS;
    const size_t src_row = ggml_type_row_bytes(t.type, K);
    const size_t src_blk = t.type == GGML_PTQ1_0 ? 28 : 34;
    if (t.type != GGML_PTQ1_0 && t.type != GGML_PQ2_0) throw std::runtime_error("repack: not a ternary tensor: " + t.name);

    std::atomic<int64_t> next{0};
    auto worker = [&]() {
        uint8_t trit[128];
        for (;;) {
            int64_t T = next.fetch_add(1);
            if (T >= ntiles) break;
            uint8_t * tile = dst + (size_t) T * nb * TILE_BLOCK_BYTES;
            for (int64_t b = 0; b < nb; b++) {
                uint8_t * run = tile + b * TILE_BLOCK_BYTES;
                for (int l = 0; l < TILE_ROWS; l++) {
                    const uint8_t * blk = t.data + (size_t) (T * TILE_ROWS + l) * src_row + b * src_blk;
                    uint16_t sc;
                    if (t.type == GGML_PTQ1_0) decode_gguf_ptq1_0(blk, trit, &sc); else decode_gguf_pq2_0(blk, trit, &sc);
                    for (int i = 0; i < 128; i++) if (trit[i] > 2) throw std::runtime_error("repack: non-ternary code in " + t.name);
                    uint8_t qs[24], qh[2];
                    encode_block(trit, qs, qh);
                    memcpy(run + tile_off_qs_a(l), qs, 16);
                    memcpy(run + tile_off_qs_b(l), qs + 16, 8);
                    uint8_t * tail = run + tile_off_tail(l);
                    tail[0] = qh[0]; tail[1] = qh[1]; memcpy(tail + 2, &sc, 2);
                }
            }
        }
    };
    std::vector<std::thread> th;
    for (int i = 0; i < nthreads; i++) th.emplace_back(worker);
    for (auto & x : th) x.join();
}

HaloCache::~HaloCache() { if (base) munmap((void *) base, map_size); if (fd >= 0) close(fd); }

static bool is_ternary(const GgufTensor & t) { return t.type == GGML_PTQ1_0 || t.type == GGML_PQ2_0; }

// cache header: magic, source size, source mtime, n entries, then entries
struct CacheHeader { char magic[8]; uint64_t src_size; uint64_t src_mtime; uint64_t n_entries; uint64_t data_offset; };

bool HaloCache::open_or_build(const Gguf & g, const std::string & gguf_path, int nthreads) {
    struct stat st; if (stat(gguf_path.c_str(), &st)) throw std::runtime_error("stat failed: " + gguf_path);
    std::string cache_path = gguf_path + ".halo";

    auto try_open = [&]() -> bool {
        fd = ::open(cache_path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat cs; fstat(fd, &cs);
        if ((size_t) cs.st_size < sizeof(CacheHeader)) { close(fd); fd = -1; return false; }
        map_size = cs.st_size;
        base = (const uint8_t *) mmap(nullptr, map_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (base == MAP_FAILED) { base = nullptr; close(fd); fd = -1; return false; }
        CacheHeader h; memcpy(&h, base, sizeof h);
        if (memcmp(h.magic, "HALOCAC2", 8) || h.src_size != (uint64_t) st.st_size || h.src_mtime != (uint64_t) st.st_mtime) {
            munmap((void *) base, map_size); base = nullptr; close(fd); fd = -1; return false;
        }
        const uint8_t * p = base + sizeof h;
        entries.clear();
        for (uint64_t i = 0; i < h.n_entries; i++) {
            uint32_t nl; memcpy(&nl, p, 4); p += 4;
            HaloCacheEntry e; e.name.assign((const char *) p, nl); p += nl;
            memcpy(&e.N, p, 8); p += 8; memcpy(&e.K, p, 8); p += 8; memcpy(&e.offset, p, 8); p += 8; memcpy(&e.bytes, p, 8); p += 8;
            entries.push_back(e);
        }
        return true;
    };
    if (try_open()) return false;

    // build
    auto t0 = std::chrono::steady_clock::now();
    std::vector<const GgufTensor *> list;
    for (auto & t : g.tensors) if (is_ternary(t)) list.push_back(&t);
    size_t index_bytes = sizeof(CacheHeader);
    for (auto * t : list) index_bytes += 4 + t->name.size() + 32;
    size_t data_offset = (index_bytes + 4095) / 4096 * 4096;
    size_t total = data_offset;
    entries.clear();
    for (auto * t : list) {
        HaloCacheEntry e; e.name = t->name; e.K = t->ne[0]; e.N = t->rows(); e.offset = total; e.bytes = halo_tensor_bytes(e.N, e.K);
        total += (e.bytes + 4095) / 4096 * 4096;
        entries.push_back(e);
    }
    std::string tmp = cache_path + ".tmp";
    int wfd = ::open(tmp.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (wfd < 0) throw std::runtime_error("cannot create " + tmp);
    if (ftruncate(wfd, total)) throw std::runtime_error("ftruncate failed");
    uint8_t * wbase = (uint8_t *) mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, wfd, 0);
    if (wbase == MAP_FAILED) throw std::runtime_error("mmap rw failed");

    CacheHeader h; memcpy(h.magic, "HALOCAC2", 8); h.src_size = st.st_size; h.src_mtime = st.st_mtime; h.n_entries = entries.size(); h.data_offset = data_offset;
    uint8_t * p = wbase; memcpy(p, &h, sizeof h); p += sizeof h;
    for (auto & e : entries) {
        uint32_t nl = e.name.size(); memcpy(p, &nl, 4); p += 4; memcpy(p, e.name.data(), nl); p += nl;
        memcpy(p, &e.N, 8); p += 8; memcpy(p, &e.K, 8); p += 8; memcpy(p, &e.offset, 8); p += 8; memcpy(p, &e.bytes, 8); p += 8;
    }
    for (size_t i = 0; i < list.size(); i++) {
        fprintf(stderr, "\rrepacking %zu/%zu %-40s", i + 1, list.size(), list[i]->name.c_str());
        repack_tensor(*list[i], wbase + entries[i].offset, nthreads);
    }
    fprintf(stderr, "\n");
    msync(wbase, total, MS_SYNC);
    munmap(wbase, total);
    close(wfd);
    if (rename(tmp.c_str(), cache_path.c_str())) throw std::runtime_error("rename failed");
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "halo cache built: %s (%.2f GB) in %.1f s\n", cache_path.c_str(), total / 1e9, s);
    if (!try_open()) throw std::runtime_error("cache reopen failed");
    return true;
}

const HaloCacheEntry & HaloCache::entry(const std::string & name) const {
    for (auto & e : entries) if (e.name == name) return e;
    throw std::runtime_error("halo cache: missing " + name);
}

} // namespace halo

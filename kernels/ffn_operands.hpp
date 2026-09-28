// Operands for the batched FFN: compact ternary weight storage and the register expansions the
// three matrix instructions want.
//
// Provenance: the packing layout and the three expansions are the Kelana batched-FFN research at
// commit 1adc35e — research/ffn/batched/arithmetic/maps.hpp (two-bit codes, expand_i8, expand_i4)
// and research/ffn/batched/compact-scaled/scaled.hpp (scale_tables, expand_scaled, whose identity
// is carried as a proof over raw half bits in Kelana/ScaledTrit.lean). The lane-major word order
// and its one-slice lookahead are from tile-ownership/candidates/to_maps.hip at commit 15e0524.
// This file is the engine's own copy; nothing here depends on the research tree at build or run
// time. What is new is the device-side HALO decode below, so the engine never round-trips 4.3 GB
// of weights through the host to prepare them.
//
// ---- storage ----
//
// One 128-wide K block of 16 weight rows is 512 bytes: eight 64-byte slices (one WMMA K16 step),
// each holding 16 rows x one 32-bit word, each word 16 two-bit codes with element j at bits 2j.
//
//     code 0 -> weight  0        code 1 -> weight +1        code 3 -> weight -1
//
// Both nonzero codes set bit 0 and only -1 sets bit 1, which is what makes all three expansions a
// handful of byte permutes. Block b of 16-row tile t lives at (t * nblocks + b) * 512.
//
// The FP16 scale of a (row, block) is kept as raw bits in tile-major order, [tile][block][16 rows],
// so the 16 scales one wave needs for a block are 32 contiguous bytes. The scaled-FP16 expansion
// consumes those bits directly and never converts them to float. One scale image serves every
// weight image and every mode, because no layout below moves a scale.
//
// The 512 bytes come in two orders, LAYOUT_SLICE and LAYOUT_LANE below. They hold the same words
// for the same weights and differ only in how many loads a lane needs for a block.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstddef>

namespace halo {
namespace ffnb {

constexpr int KB           = 128;          // scale block along K
constexpr int SLICES       = KB / 16;      // WMMA K16 slices per block
constexpr int BLOCK_BYTES  = 512;          // 16 rows x 128 K of two-bit codes
constexpr int ROWS_PER_TILE = 16;          // logical rows a wave owns

// Bytes of a [rows][kdim] matrix in this storage, and of its tile-major scale image.
inline size_t codes_bytes(int rows, int kdim) {
    return (size_t) (rows / ROWS_PER_TILE) * (kdim / KB) * BLOCK_BYTES;
}
inline size_t scale_elems(int rows, int kdim) {
    return (size_t) (rows / ROWS_PER_TILE) * (kdim / KB) * ROWS_PER_TILE;
}

using half16 = _Float16 __attribute__((ext_vector_type(16)));
using uint8v = unsigned __attribute__((ext_vector_type(8)));
using int8v  = int __attribute__((ext_vector_type(8)));
using int4v  = int __attribute__((ext_vector_type(4)));
using int2v  = int __attribute__((ext_vector_type(2)));
using float8 = float __attribute__((ext_vector_type(8)));

// ---------------------------------------------------------------- HALO -> two-bit, on the device

// One HALO 896-byte block-run holds 32 rows; `lane` selects a row. The trit order inside the run is
// the packed-16 peel documented in src/halo_format.h, and this reproduces decode_block() straight
// into the code words, so no 128-byte trit array ever exists.
__host__ __device__ __forceinline__ unsigned halo_qs_byte(const uint4 & a, const uint2 & b, int i) {
    const unsigned d = i < 4 ? a.x : i < 8 ? a.y : i < 12 ? a.z : i < 16 ? a.w : i < 20 ? b.x : b.y;
    return (d >> (8 * (i & 3))) & 0xffu;
}
// Stored trit is 0,1,2 meaning -1,0,+1; the code is 3,0,1 in the same order.
__host__ __device__ __forceinline__ unsigned trit_code(unsigned t) { return t == 0u ? 3u : t - 1u; }

// Host-callable as well, so tools/ffn_pack_check.cpp exercises this exact code on the CPU.
__host__ __device__ __forceinline__ void halo_block_codes(uint4 qa, uint2 qb, unsigned tail, uint32_t w[8]) {
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] = 0;
#define PUT(e, t) w[(e) >> 4] |= trit_code(t) << (2 * ((e) & 15))
#pragma unroll
    for (int d = 0; d < 6; ++d)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int half = j >> 1, p = 2 * d + (j & 1);
            unsigned b = halo_qs_byte(qa, qb, 4 * d + j), t[5];
#pragma unroll
            for (int n = 0; n < 5; ++n) { const unsigned m = b * 3u; t[n] = m >> 8; b = m & 0xffu; }
            PUT(8 * p + 2 * half + 0, t[0]);
            PUT(8 * p + 2 * half + 1, t[1]);
            PUT(8 * p + 4 + 2 * half + 0, t[2]);
            PUT(8 * p + 4 + 2 * half + 1, t[3]);
            PUT(96 + 4 * d + j, t[4]);
        }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        unsigned b = (tail >> (8 * h)) & 0xffu, t[4];
#pragma unroll
        for (int n = 0; n < 4; ++n) { const unsigned m = b * 3u; t[n] = m >> 8; b = m & 0xffu; }
        PUT(120 + 2 * h, t[0]);
        PUT(121 + 2 * h, t[1]);
        PUT(124 + 2 * h, t[2]);
        PUT(125 + 2 * h, t[3]);
    }
#undef PUT
}

// ---------------------------------------------------------------- register expansions

// 16 codes -> 16 signed bytes for v_wmma_i32_16x16x16_iu8, four v_perm lookups from a byte table.
__device__ __forceinline__ int4v expand_i8(unsigned codes) {
    int4v r;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned v = (codes >> (8 * j)) & 0xffu;
        const unsigned t = (v | (v << 12)) & 0x000f000fu;
        const unsigned sel = (t | (t << 6)) & 0x03030303u;
        r[j] = __builtin_amdgcn_perm(0u, 0xff000100u, sel);   // byte 0:0, 1:+1, 3:-1
    }
    return r;
}

// 16 codes -> 16 int4 nibbles for v_wmma_i32_16x16x16_iu4. Each code lands in its own nibble; the
// two high bits of a nibble are set exactly when the code is 3, which is int4 -1.
__device__ __forceinline__ int2v expand_i4(unsigned codes) {
    int2v r;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const unsigned v = (codes >> (16 * j)) & 0xffffu;
        unsigned x = (v | (v << 8)) & 0x00ff00ffu;
        x = (x | (x << 4)) & 0x0f0f0f0fu;
        x = (x | (x << 2)) & 0x33333333u;
        const unsigned h = x & (x >> 1) & 0x11111111u;
        r[j] = int(x | (h << 2) | (h << 3));
    }
    return r;
}

// A ternary weight times an FP16 scale is exactly one of three bit patterns: zero, the scale, or the
// scale with bit 15 flipped. One lane owns one row for a whole 128-block, so its scale is one value
// for all 128 weights and the fold is byte selection from two tables built once per block:
//
//   tlo bytes: [0x00, s_lo, s_lo, s_lo]            index c gives the low byte of t*s
//   thi bytes: [0x00, s_hi, s_hi, s_hi ^ 0x80]     index c gives the high byte of t*s
//
// Index 2 is never selected. The XOR is the sign flip, correct whatever sign the stored scale has.
__device__ __forceinline__ void scale_tables(unsigned s16, unsigned & tlo, unsigned & thi) {
    const unsigned lo = s16 & 0xffu, hi = (s16 >> 8) & 0xffu;
    unsigned a = lo << 8;  a |= a << 8;  tlo = a | (a << 8);
    unsigned b = hi << 8;  b |= b << 8;  thi = (b & 0x00ffff00u) | ((hi ^ 0x80u) << 24);
}

// 16 codes -> 16 scaled FP16 weights, nine instructions per four weights. Selecting a half's low
// byte from tlo and its high byte from thi makes the high position's selector the code plus four,
// so one OR and one interleaving permute build a selector pair and one more permute finishes two
// halves.
__device__ __forceinline__ half16 expand_scaled(unsigned codes, unsigned tlo, unsigned thi) {
    uint8v r;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned v = (codes >> (8 * j)) & 0xffu;
        const unsigned t = v | (v << 12);
        const unsigned c = (t | (t << 6)) & 0x03030303u;      // four code bytes
        const unsigned cp = c | 0x04040404u;                  // the same codes, indexing thi
        r[2 * j + 0] = __builtin_amdgcn_perm(thi, tlo, __builtin_amdgcn_perm(cp, c, 0x05010400u));
        r[2 * j + 1] = __builtin_amdgcn_perm(thi, tlo, __builtin_amdgcn_perm(cp, c, 0x07030602u));
    }
    return __builtin_bit_cast(half16, r);
}

// The word one lane reads for (block, slice); `col` is its row within the 16-row tile.
__device__ __forceinline__ unsigned weight_word(const uint8_t * tile_base, int blk, int slice, int col) {
    return *(const unsigned *) (tile_base + (size_t) blk * BLOCK_BYTES + slice * 64 + col * 4);
}

// ---------------------------------------------------------------- weight word order
//
// LAYOUT_SLICE  [slice][row]: the deployed order. A lane's eight words are 64 bytes apart, so a
//               block costs eight dword loads.
// LAYOUT_LANE   [row][slice]: a lane's eight words are contiguous, so a block is two uint4 loads.
//               Same bytes, same image size, same weights, same numbers out of the matrix
//               instruction; only the address arithmetic and the load count change.
enum WordLayout { LAYOUT_SLICE = 0, LAYOUT_LANE = 1 };

// Where row `col` of a block keeps its words, in either order.
__host__ __device__ __forceinline__ size_t lane_base_off(int layout, int col) {
    return (size_t) col * (layout == LAYOUT_LANE ? 32 : 4);
}

// One lane's eight words for a 128-block, in the form its layout allows.
//
// LAYOUT_SLICE fetches one slice ahead. Loaded in the slice that consumes them, the two matrices'
// words issue behind that slice's sixteen fragment loads and the operand expansion then waits on
// the whole outstanding queue, s_waitcnt vmcnt(0) rather than vmcnt(6), once per slice with the
// memory latency exposed. Kelana measured that as the down projection at 256 rows running 1.87 ms
// against 1.35 with an identical grid and instruction mix. The last fetch wraps to slice 0, which
// is cached and never used.
//
// LAYOUT_LANE reads the whole block up front as two uint4, which needs the slice loop unrolled so
// the word index is a constant.
template <int LAY> struct BlockWords {
    unsigned w[LAY == LAYOUT_LANE ? 8 : 1];
    // A wave that carries several weight row tiles holds one of these per tile, so the array form
    // needs a constructor that reads nothing; every element is assigned before it is taken.
    BlockWords() = default;
    __device__ __forceinline__ explicit BlockWords(const uint8_t * p) {
        if constexpr (LAY == LAYOUT_LANE) {
            const uint4 lo = *(const uint4 *) p, hi = *(const uint4 *) (p + 16);
            w[0] = lo.x; w[1] = lo.y; w[2] = lo.z; w[3] = lo.w;
            w[4] = hi.x; w[5] = hi.y; w[6] = hi.z; w[7] = hi.w;
        } else w[0] = *(const unsigned *) p;
    }
    __device__ __forceinline__ unsigned take(const uint8_t * p, int s) {
        if constexpr (LAY == LAYOUT_LANE) return w[s];
        const unsigned cur = w[0];
        w[0] = *(const unsigned *) (p + ((s + 1) & (SLICES - 1)) * 64);
        return cur;
    }
};

__device__ __forceinline__ float half_bits_to_float(unsigned bits) {
    const unsigned short u = (unsigned short) bits;
    _Float16 h;
    __builtin_memcpy(&h, &u, 2);
    return (float) h;
}

} // namespace ffnb
} // namespace halo

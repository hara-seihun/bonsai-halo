// Ternary operand maps for an IU8 matrix consumer, and the stored coordinates they read.
//
// A HALO block gives one lane 26 bytes holding 128 trits, and `v_wmma_i32_16x16x16_iu8` wants them
// as 32 dwords of four unsigned trit codes, one code per byte lane. That map is the whole of the
// decode work in the vocabulary head: 220 of the 600 issue slots its block loop spends, measured in
// docs/wide-head.md, and the head reads its 278 MB image at 24 GB/s at 128 rows, so the slots are
// what it waits for rather than the bytes.
//
// Three arms live here. They produce *the same 32 dwords*, byte for byte, so every consumer keeps
// its accumulator seed, its matrix instructions, its reduction order and its output bits; only the
// instructions that build the operand, and for arm 2 the bytes it reads, differ.
//
//   0  the deployed five-trit peel. A byte pair is multiplied by three in two 16-bit lanes, the
//      product is shifted down by eight, its low bytes are the next remainder, and a later
//      `v_lshl_or_b32` moves the trit into the byte lane the operand wants.
//
//   1  the same image and the same bytes, with the shift and the or deleted. The trit is ALREADY a
//      byte of the product: `m = r * 3` puts `(b_first * 3) >> 8` in byte 1 and
//      `(b_second * 3) >> 8` in byte 3, both a bare trit because `b * 3 <= 765`. So the peel's
//      `>> 8` only moves a value that is already in a byte lane, and one `v_perm_b32` with a
//      constant selector gathers four trits from two products straight into the operand dword.
//      Per source dword that is 26 ops for 20 trits against 36; per 128-trit block, 168 against 232.
//
//   2  a stored coordinate that skips the arithmetic: 4 trits per byte in 2-bit fields, spread so
//      that `(w >> 2k) & 0x03030303` IS operand dword k. 56 ops per block, and 34 bytes per
//      (row, block) against 28 - the same trade the sequence projections took for their IU8 operand
//      (`expand_i8_spread`, docs/sequence-projection-operands.md). docs/wide-head.md priced two-bit
//      codes at "2.25 VALU per weight, worse than the peel" from `expand_i8`, which reads the FFN's
//      *unspread* word and pays six ops to move each code into its byte lane. The spread layout
//      does not pay them, and nothing above the storage cares which of the two it is.
//
// The selector semantics used here are the measured gfx1151 table in
// kelana research/ffn/batched/dense-consumer/results/perm-semantics.json: for
// `__builtin_amdgcn_perm(a, b, sel)`, selector bytes 0..3 name bytes of `b` and 4..7 name bytes of
// `a`. Only 0..7 appear below, so the sign-replication entries do not arise.
//
// Everything except the `HeadOperand` traits is portable, so `kernels/head_op_check.cpp` runs these
// exact maps on the CPU against `halo::decode_block` with no GPU.
#pragma once
#include <cstdint>
#include <cstddef>
#include "halo_format.h"
#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#endif

// Both passes of a HIP translation unit need these, and a host-only build (the pack check) needs
// them too, so the maps carry both attributes wherever hipcc defines them.
#ifdef __HIPCC__
#define HX_FN __host__ __device__ __forceinline__
#else
#define HX_FN inline
#endif

namespace halo {

// ---------------------------------------------------------------------------- primitives

constexpr unsigned HX_LOW = 0x00ff00ffu;
// [b.1, a.1, b.3, a.3]: step n's trit from `lo`, step n+1's from `hi`, first source byte then second.
constexpr unsigned HX_SEL = 0x07030501u;

#if defined(__HIP_DEVICE_COMPILE__)
typedef unsigned short hx_us2 __attribute__((ext_vector_type(2)));
// One base-three peel step on a byte pair held in the low byte of each 16-bit lane. The product is
// the whole state: byte 1 and byte 3 hold the two trits, bytes 0 and 2 the next remainder.
__device__ __forceinline__ unsigned hx_mul3(unsigned r) {
    return __builtin_bit_cast(unsigned, __builtin_bit_cast(hx_us2, r) * (hx_us2){ 3, 3 });
}
__device__ __forceinline__ unsigned hx_shr8(unsigned r) {
    return __builtin_bit_cast(unsigned, __builtin_bit_cast(hx_us2, r) >> (hx_us2){ 8, 8 });
}
__device__ __forceinline__ unsigned hx_gather(unsigned hi, unsigned lo) {
    return __builtin_amdgcn_perm(hi, lo, HX_SEL);
}
#else
inline unsigned hx_mul3(unsigned r) {
    return (unsigned) (unsigned short) ((r & 0xffffu) * 3u)
         | ((unsigned) (unsigned short) (((r >> 16) & 0xffffu) * 3u) << 16);
}
inline unsigned hx_shr8(unsigned r) { return ((r & 0xffffu) >> 8) | (((r >> 16) & 0xffffu) >> 8 << 16); }
inline unsigned hx_gather(unsigned hi, unsigned lo) {
    unsigned out = 0;
    for (int j = 0; j < 4; j++) {
        const unsigned s = (HX_SEL >> (8 * j)) & 0xffu;
        const unsigned v = s < 4 ? (lo >> (8 * s)) & 0xffu : (hi >> (8 * (s - 4))) & 0xffu;
        out |= v << (8 * j);
    }
    return out;
}
#endif

// ---------------------------------------------------------------------------- arm 0: the peel

HX_FN unsigned hx_peel(unsigned & r) {
    const unsigned m = hx_mul3(r);
    r = m & HX_LOW;
    return hx_shr8(m);
}
HX_FN unsigned hx_peel_last(unsigned r) { return hx_shr8(hx_mul3(r)); }

// The 26 stored bytes of one row's block, as six dwords and the tail dword, into 32 operand dwords.
HX_FN void hx_expand_peel(const unsigned dw[6], unsigned tail, unsigned tr[32]) {
#pragma unroll
    for (int d = 0; d < 6; d++) {
        unsigned P0 = dw[d] & HX_LOW, P1 = (dw[d] >> 8) & HX_LOW;
        const unsigned t0 = hx_peel(P0), t1 = hx_peel(P0), t2 = hx_peel(P0), t3 = hx_peel(P0), t4 = hx_peel_last(P0);
        const unsigned u0 = hx_peel(P1), u1 = hx_peel(P1), u2 = hx_peel(P1), u3 = hx_peel(P1), u4 = hx_peel_last(P1);
        tr[4 * d] = t0 | (t1 << 8); tr[4 * d + 1] = t2 | (t3 << 8);
        tr[4 * d + 2] = u0 | (u1 << 8); tr[4 * d + 3] = u2 | (u3 << 8);
        tr[24 + d] = t4 | (u4 << 8);
    }
    unsigned P = (tail & 0xffu) | ((tail << 8) & 0xff0000u);
    const unsigned h0 = hx_peel(P), h1 = hx_peel(P), h2 = hx_peel(P), h3 = hx_peel_last(P);
    tr[30] = h0 | (h1 << 8); tr[31] = h2 | (h3 << 8);
}

// ---------------------------------------------------------------------------- arm 1: perm gather

HX_FN void hx_expand_perm(const unsigned dw[6], unsigned tail, unsigned tr[32]) {
#pragma unroll
    for (int d = 0; d < 6; d++) {
        // Five products per byte pair, each carrying its trit in bytes 1 and 3.
        const unsigned m0 = hx_mul3(dw[d] & HX_LOW);
        const unsigned m1 = hx_mul3(m0 & HX_LOW);
        const unsigned m2 = hx_mul3(m1 & HX_LOW);
        const unsigned m3 = hx_mul3(m2 & HX_LOW);
        const unsigned m4 = hx_mul3(m3 & HX_LOW);
        const unsigned n0 = hx_mul3((dw[d] >> 8) & HX_LOW);
        const unsigned n1 = hx_mul3(n0 & HX_LOW);
        const unsigned n2 = hx_mul3(n1 & HX_LOW);
        const unsigned n3 = hx_mul3(n2 & HX_LOW);
        const unsigned n4 = hx_mul3(n3 & HX_LOW);
        tr[4 * d] = hx_gather(m1, m0); tr[4 * d + 1] = hx_gather(m3, m2);
        tr[4 * d + 2] = hx_gather(n1, n0); tr[4 * d + 3] = hx_gather(n3, n2);
        tr[24 + d] = hx_gather(n4, m4);   // the fifth trit of each byte, paired across the halves
    }
    const unsigned P = (tail & 0xffu) | ((tail << 8) & 0xff0000u);
    const unsigned h0 = hx_mul3(P);
    const unsigned h1 = hx_mul3(h0 & HX_LOW);
    const unsigned h2 = hx_mul3(h1 & HX_LOW);
    const unsigned h3 = hx_mul3(h2 & HX_LOW);
    tr[30] = hx_gather(h1, h0); tr[31] = hx_gather(h3, h2);
}

// ---------------------------------------------------------------------------- arm 2: spread codes
//
// Per (32-row tile, 128-block), 1088 bytes in three lane-interleaved sections so every load is
// naturally aligned and coalesced across the 32 lanes of a wave:
//
//   S0  offset    0   32 lanes x 16 B   lane reads uint4     operand dwords  0..15
//   S1  offset  512   32 lanes x 16 B   lane reads uint4     operand dwords 16..31
//   T   offset 1024   32 lanes x  4 B   lane reads unsigned  the HALO tail dword, scale included
//
// The tail dword is copied verbatim rather than trimmed to its fp16 scale: the two `qh` bytes it
// also carries cost two bytes a row and keep the scale extraction, its byte offset and the store
// that feeds the scale table exactly what they are on the deployed arm.
constexpr int SPREAD_S1_OFF      = 512;
constexpr int SPREAD_T_OFF       = 1024;
constexpr int SPREAD_BLOCK_BYTES = 1088;
HX_FN size_t spread_off_a(int lane) { return (size_t) lane * 16; }
HX_FN size_t spread_off_b(int lane) { return SPREAD_S1_OFF + (size_t) lane * 16; }
HX_FN size_t spread_off_t(int lane) { return SPREAD_T_OFF + (size_t) lane * 4; }
inline size_t spread_tensor_bytes(int64_t N, int64_t K) {
    return (size_t) (N / TILE_ROWS) * (K / BLOCK) * SPREAD_BLOCK_BYTES;
}

// Operand dwords -> the stored word. Trit `p` of operand dword `4j + k` lands at bit `8p + 2k` of
// word `j`, which is what makes the expansion one shift and one mask.
HX_FN void spread_pack(const unsigned tr[32], unsigned w[8]) {
#pragma unroll
    for (int j = 0; j < 8; j++) {
        unsigned word = 0;
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const unsigned t = tr[4 * j + k];
#pragma unroll
            for (int p = 0; p < 4; p++) word |= ((t >> (8 * p)) & 3u) << (8 * p + 2 * k);
        }
        w[j] = word;
    }
}
HX_FN void spread_expand(const unsigned w[8], unsigned tr[32]) {
#pragma unroll
    for (int j = 0; j < 8; j++)
#pragma unroll
        for (int k = 0; k < 4; k++) tr[4 * j + k] = (w[j] >> (2 * k)) & 0x03030303u;
}

#ifdef __HIPCC__

// ---------------------------------------------------------------------------- the three arms
//
// A consumer holds `Regs` across its block lookahead, calls `expand` where it peeled, and reads the
// block's fp16 scale bits from `scale`. `BLOCK_BYTES` is the stride of one (tile, block) run and
// `load` takes the same run pointer the deployed kernel computes.

template <int OP> struct HeadOperand;

// arm 0 -- the deployed peel, kept as the in-process control.
template <> struct HeadOperand<0> {
    static constexpr int BLOCK_BYTES = TILE_BLOCK_BYTES;
    struct Regs { uint4 qa; uint2 qb; unsigned tail; };
    __device__ __forceinline__ static Regs load(const uint8_t * __restrict__ run, int lane) {
        Regs r;
        r.qa = *(const uint4 *) (run + tile_off_qs_a(lane));
        r.qb = *(const uint2 *) (run + tile_off_qs_b(lane));
        r.tail = *(const unsigned *) (run + tile_off_tail(lane));
        return r;
    }
    __device__ __forceinline__ static unsigned scale(const Regs & r) { return r.tail >> 16; }
    __device__ __forceinline__ static void expand(const Regs & r, unsigned tr[32]) {
        const unsigned dw[6] = { r.qa.x, r.qa.y, r.qa.z, r.qa.w, r.qb.x, r.qb.y };
        hx_expand_peel(dw, r.tail, tr);
    }
};

// arm 1 -- same image, same bytes, same operand dwords; the peel's shift and or are gone.
template <> struct HeadOperand<1> {
    static constexpr int BLOCK_BYTES = TILE_BLOCK_BYTES;
    using Regs = HeadOperand<0>::Regs;
    __device__ __forceinline__ static Regs load(const uint8_t * __restrict__ run, int lane) {
        return HeadOperand<0>::load(run, lane);
    }
    __device__ __forceinline__ static unsigned scale(const Regs & r) { return r.tail >> 16; }
    __device__ __forceinline__ static void expand(const Regs & r, unsigned tr[32]) {
        const unsigned dw[6] = { r.qa.x, r.qa.y, r.qa.z, r.qa.w, r.qb.x, r.qb.y };
        hx_expand_perm(dw, r.tail, tr);
    }
};

// arm 2 -- the spread two-bit coordinate.
template <> struct HeadOperand<2> {
    static constexpr int BLOCK_BYTES = SPREAD_BLOCK_BYTES;
    struct Regs { uint4 wa; uint4 wb; unsigned tail; };
    __device__ __forceinline__ static Regs load(const uint8_t * __restrict__ run, int lane) {
        Regs r;
        r.wa = *(const uint4 *) (run + spread_off_a(lane));
        r.wb = *(const uint4 *) (run + spread_off_b(lane));
        r.tail = *(const unsigned *) (run + spread_off_t(lane));
        return r;
    }
    __device__ __forceinline__ static unsigned scale(const Regs & r) { return r.tail >> 16; }
    __device__ __forceinline__ static void expand(const Regs & r, unsigned tr[32]) {
        const unsigned w[8] = { r.wa.x, r.wa.y, r.wa.z, r.wa.w, r.wb.x, r.wb.y, r.wb.z, r.wb.w };
        spread_expand(w, tr);
    }
};

#endif  // __HIPCC__

} // namespace halo

// Four-bit operands built straight out of stored labels, for the optimized A4 mode.
//
// The IU4 matrix instruction wants sixteen ternary nibbles per K16 slice. Both storages here hand
// v_perm_b32 a selector byte that already names the operand byte, so no individual weight is ever
// reconstructed:
//
//   pair codes   a nine-valued two-trit code per nibble, 2.000 bits per weight, lane-major so a
//                lane's whole 128-block is two uint4 loads. Expansion is a mask, a shift and two
//                permutes per dword.
//   dense five   the stored HALO five-trit byte itself, 26 bytes per (row, 128-block), 1.625 bits
//                per weight. Two multiplies and two shifts per byte pair lift a byte into two
//                nine-valued codes plus one trit, and a code is the permute selector.
//
// Both produce exactly the weights the two-bit image produces, so a projection built on either is
// bit-identical to the deployed IU4 map (module mode 2) for the same activations.
//
// Provenance: Kelana research/ffn/batched/dense-consumer at commit 15e0524 (dense5.hpp and
// candidates/dense_maps.hip). This is the engine's own copy; the device-side packers below are new,
// so preparation reads the HALO image on the device and the host never sees a decoded weight.
//
// ---- the code alphabet ----
//
// A nine-valued code names a weight pair: code = 3 * mu^-1(w_u) + mu^-1(w_v), with
//
//     mu(0) = -1,   mu(1) = +1,   mu(2) = 0
//
// and (w_u, w_v) the lower and upper K slot of one operand byte. The alphabet is not arbitrary.
// v_perm_b32 offers eight dynamic source bytes for selectors 0..7, and selector 8 replicates the
// sign bit of source byte 1, which is argB byte 1, which is code 1's operand byte. Under this mu,
// code 1 is (-1, +1) = 0x1F, whose top bit is clear, so selector 8 delivers the 0x00 that code 8
// = (0, 0) needs. The natural mu(t) = t - 1 puts 0xFF there and corrupts one pair in nine. The
// selector table was measured on gfx1151 over all 256 selector values in
// dense-consumer/results/perm-semantics.json, not quoted from documentation.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstddef>
#include "ffn_operands.hpp"

namespace halo {
namespace ffnb {

// ---------------------------------------------------------------- the nine-code operand table
//
//   0 -> 0xFF (-1,-1)   1 -> 0x1F (-1,+1)   2 -> 0x0F (-1, 0)
//   3 -> 0xF1 (+1,-1)   4 -> 0x11 (+1,+1)   5 -> 0x01 (+1, 0)
//   6 -> 0xF0 ( 0,-1)   7 -> 0x10 ( 0,+1)   8 -> 0x00 ( 0, 0), the sign replicate of code 1
// Low nibble is the lower K slot.
constexpr unsigned PERM_S1 = 0xF10F1FFFu;   // bytes 0..3 = codes 0..3
constexpr unsigned PERM_S0 = 0x10F00111u;   // bytes 0..3 = codes 4..7

constexpr int PAIR_BLOCK_BYTES  = 512;      // 16 rows x 128 K of nine-valued codes, lane-major
constexpr int DENSE_BLOCK_BYTES = 416;      // 16 rows x 128 K of five-trit bytes, three sections
constexpr int DENSE_S1_OFF      = 256;
constexpr int DENSE_S2_OFF      = 384;

// ---------------------------------------------------------------- stored two-bit code -> label
//
// The two-bit image's code is 0 for weight 0, 1 for +1, 3 for -1 (ffn_operands.hpp). mu^-1 of the
// same weight is 2, 1, 0. One code word holds sixteen codes, element j at bits 2j.
__host__ __device__ __forceinline__ unsigned code_at(const uint32_t w[8], int e) {
    return (w[e >> 4] >> (2 * (e & 15))) & 3u;
}
__host__ __device__ __forceinline__ unsigned code_mu(unsigned c) { return c == 0u ? 2u : (c == 1u ? 1u : 0u); }
__host__ __device__ __forceinline__ unsigned pair_code(const uint32_t w[8], int eu, int ev) {
    return 3u * code_mu(code_at(w, eu)) + code_mu(code_at(w, ev));
}

// ---------------------------------------------------------------- pair-code storage
//
// Slice s of a row becomes one dword: byte i carries the code of K slots (2i, 2i+1) in its low
// nibble and of (8 + 2i, 9 + 2i) in its high nibble, which is the pair the two expansion permutes
// want. Lane-major: a row's eight dwords are 32 contiguous bytes.
__host__ __device__ __forceinline__ void pack_pair_block(const uint32_t w[8], uint32_t out[8]) {
#pragma unroll
    for (int s = 0; s < SLICES; ++s) {
        uint32_t word = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            word |= pair_code(w, s * 16 + 2 * i, s * 16 + 2 * i + 1) << (8 * i);
            word |= pair_code(w, s * 16 + 8 + 2 * i, s * 16 + 9 + 2 * i) << (8 * i + 4);
        }
        out[s] = word;
    }
}

// ---------------------------------------------------------------- dense five-trit storage
//
// Per (16-row tile, 128-block), 416 bytes in three lane-interleaved sections so every load is
// naturally aligned and coalesced across the sixteen lanes of a wave half:
//
//   S0  offset   0   16 lanes x 16 B   lane reads uint4    bytes 0..15   five trits each
//   S1  offset 256   16 lanes x  8 B   lane reads uint2    bytes 16..23  five trits each
//   S2  offset 384   16 lanes x  2 B   lane reads ushort   bytes 24,25   four trits each
//
// 24*5 + 2*4 = 128 trits in 26 bytes. Each four-byte group carries one whole K16 slice in its A
// and B codes and lends its four c trits to a later slice:
//
//   slice i, K slots  0.. 7  <-  A(b0) A(b2) B(b0) B(b2)
//   slice i, K slots  8..15  <-  A(b1) A(b3) B(b1) B(b3)
//   slice 6 <- c of S0 dwords 0..3, paired as 3*c(b0)+c(b2) and 3*c(b1)+c(b3)
//   slice 7 <- c of S1 dwords 0..1, then A,B of bytes 24 and 25

// K slot inside a slice that byte j of a dense dword feeds; which_code 0 = A, 1 = B.
__host__ __device__ __forceinline__ int dense_slot(int j, int which_code) {
    return (j & 1) * 8 + which_code * 4 + (j >> 1) * 2;
}
// K slot inside slice 6 or 7 that the c trit of byte j of dense dword d feeds.
__host__ __device__ __forceinline__ int dense_cslot(int d, int j) {
    return 2 * (2 * (d & 3) + (j & 1)) + (j >> 1);
}

// b = ceil(256 * q / 243) for the radix-3 number q over five trits, t0 most significant. This is
// halo::pack5 of src/halo_format.h as a device function; that host inline stays the definition of
// the stored HALO format and this must keep matching it.
__host__ __device__ __forceinline__ unsigned dense_pack5(unsigned t0, unsigned t1, unsigned t2,
                                                         unsigned t3, unsigned t4) {
    const unsigned q = ((((t0 * 3u + t1) * 3u + t2) * 3u + t3) * 3u + t4);
    return (q * 256u + 242u) / 243u;
}
__host__ __device__ __forceinline__ unsigned dense_byte(unsigned A, unsigned B, unsigned c) {
    return dense_pack5(A / 3u, A % 3u, B / 3u, B % 3u, c);
}

// The 26 bytes of one row's 128-block, in section order: 0..15 = S0, 16..23 = S1, 24..25 = S2.
__host__ __device__ __forceinline__ void pack_dense_block(const uint32_t w[8], uint8_t out[26]) {
#pragma unroll
    for (int d = 0; d < 6; ++d)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int base = d * 16, cbase = (d < 4 ? 6 : 7) * 16;
            const int sa = dense_slot(j, 0), sb = dense_slot(j, 1);
            const unsigned A = pair_code(w, base + sa, base + sa + 1);
            const unsigned B = pair_code(w, base + sb, base + sb + 1);
            const unsigned c = code_mu(code_at(w, cbase + dense_cslot(d, j)));
            out[d * 4 + j] = (uint8_t) dense_byte(A, B, c);
        }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int base = 7 * 16;
        const unsigned A = pair_code(w, base + 8 + j * 2, base + 9 + j * 2);
        const unsigned B = pair_code(w, base + 12 + j * 2, base + 13 + j * 2);
        out[24 + j] = (uint8_t) dense_byte(A, B, 0u);
    }
}

// ---------------------------------------------------------------- register expansions

using ushort2v = unsigned short __attribute__((ext_vector_type(2)));

__device__ __forceinline__ unsigned perm3(unsigned a, unsigned b, unsigned sel) {
    return __builtin_amdgcn_perm(a, b, sel);
}
// Four selector codes in the four bytes of `sel` become eight int4 ternary nibbles.
__device__ __forceinline__ unsigned expand_codes(unsigned sel) { return perm3(PERM_S0, PERM_S1, sel); }

// One pair-code dword -> sixteen int4 nibbles: low nibbles are K 0..7, high nibbles K 8..15.
__device__ __forceinline__ int2v expand_pair(unsigned d) {
    return int2v{ int(expand_codes(d & 0x0f0f0f0fu)), int(expand_codes((d >> 4) & 0x0f0f0f0fu)) };
}

// The top-peel identity (b * 3^k) >> 8 = sum_{i<k} t_i 3^(k-1-i), two trits at a time, on a byte
// pair at once. Each 16-bit lane holds one byte, and 9 * 255 = 2295 stays inside it.
//
// Nothing shifts the peeled code down into its lane's low byte, because nothing needs it there.
// Every consumer of these three words is a v_perm_b32 that names a source byte, and the code
// already sits in byte 1 and byte 3 of the product: the shift is spelled in the selector instead,
// PEEL_HI rather than PEEL_LO. That is three v_pk_lshrrev_b16 of the eight slots a peel used to
// take, 76 of the 895 issue slots of a 32-row gate/up block.
struct PeelPair { ushort2v a, b, c; };       // codes in bytes 1 and 3 of each word

// Selector bytes for a gather that reads two peeled words: bytes 1 and 3 of the low word, then
// bytes 1 and 3 of the high word. The old layout, codes in bytes 0 and 2, was 0x06040200.
constexpr unsigned PEEL_HI = 0x07050301u;

__device__ __forceinline__ PeelPair peel_pair(ushort2v b) {
    const ushort2v nine = {9, 9}, three = {3, 3}, mask = {0xff, 0xff};
    PeelPair r;
    r.a = b * nine;
    r.b = (r.a & mask) * nine;
    r.c = (r.b & mask) * three;
    return r;
}
// The two S2 bytes carry four trits, so their fifth-trit step is not taken at all.
__device__ __forceinline__ PeelPair peel_pair_ab(ushort2v b) {
    const ushort2v nine = {9, 9}, mask = {0xff, 0xff};
    PeelPair r;
    r.a = b * nine;
    r.b = (r.a & mask) * nine;
    r.c = ushort2v{0, 0};
    return r;
}

// One dense dword: four five-trit bytes -> one whole K16 slice plus four c trits.
struct DenseDword {
    PeelPair p0;   // bytes 0 and 2 of the source dword
    PeelPair p1;   // bytes 1 and 3
};

__device__ __forceinline__ DenseDword peel_dword(unsigned d) {
    DenseDword r;
    r.p0 = peel_pair(__builtin_bit_cast(ushort2v, d & 0x00ff00ffu));
    r.p1 = peel_pair(__builtin_bit_cast(ushort2v, (d >> 8) & 0x00ff00ffu));
    return r;
}

// Two c trits per source byte pair fold into one two-trit code without leaving the byte lanes.
// Every byte of g is at most 2, so 3*g cannot carry across a byte and neither can the add.
__device__ __forceinline__ unsigned cc_of(const DenseDword & r) {
    const unsigned g = perm3(__builtin_bit_cast(unsigned, r.p1.c), __builtin_bit_cast(unsigned, r.p0.c), PEEL_HI);
    return g * 3u + (g >> 8);        // bytes 0 and 2 hold 3*c + c'
}

// The gathered K16 slice this dense dword owns, and its c contribution.
__device__ __forceinline__ int2v dense_slice(unsigned src, unsigned & cc) {
    const DenseDword r = peel_dword(src);
    cc = cc_of(r);
    const unsigned s0 = perm3(__builtin_bit_cast(unsigned, r.p0.b), __builtin_bit_cast(unsigned, r.p0.a), PEEL_HI);
    const unsigned s1 = perm3(__builtin_bit_cast(unsigned, r.p1.b), __builtin_bit_cast(unsigned, r.p1.a), PEEL_HI);
    return int2v{ int(expand_codes(s0)), int(expand_codes(s1)) };
}

// Four cc dwords (codes in bytes 0 and 2) become one K16 slice.
__device__ __forceinline__ int2v dense_slice_cc(unsigned c0, unsigned c1, unsigned c2, unsigned c3) {
    return int2v{ int(expand_codes(perm3(c1, c0, 0x06040200u))),
                  int(expand_codes(perm3(c3, c2, 0x06040200u))) };
}

// Slice 7: K 0..7 from the c trits of the two S1 dwords, K 8..15 from the two four-trit S2 bytes.
__device__ __forceinline__ int2v dense_slice7(unsigned cc4, unsigned cc5, unsigned s2) {
    const ushort2v bb = __builtin_bit_cast(ushort2v, (s2 & 0xffu) | ((s2 & 0xff00u) << 8));
    const PeelPair p = peel_pair_ab(bb);
    const unsigned tail = perm3(__builtin_bit_cast(unsigned, p.b), __builtin_bit_cast(unsigned, p.a), PEEL_HI);
    return int2v{ int(expand_codes(perm3(cc5, cc4, 0x06040200u))), int(expand_codes(tail)) };
}

__device__ __forceinline__ uint4 dense_load0(const uint8_t * blk, int col) { return *(const uint4 *) (blk + col * 16); }
__device__ __forceinline__ uint2 dense_load1(const uint8_t * blk, int col) { return *(const uint2 *) (blk + DENSE_S1_OFF + col * 8); }
__device__ __forceinline__ unsigned dense_load2(const uint8_t * blk, int col) { return *(const unsigned short *) (blk + DENSE_S2_OFF + col * 2); }

} // namespace ffnb
} // namespace halo

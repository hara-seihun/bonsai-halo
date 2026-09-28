// The deployed ternary matvec's weight operand, as nibbles for v_dot8_i32_iu4.
//
// `mv_rows_t` spends about 271 issue slots per 128-block lifting five trits out of a byte and
// laying them out as IU8 operand bytes, against 32 `v_dot4_i32_iu8` per row. That expansion is the
// largest single item in the eight-row verify pass, and cutting a tenth of it (the two-trit
// palette in device.hpp) measured null. This cuts it roughly in half, by changing what the operand
// is rather than how it is built.
//
// ---- why nibbles are cheaper than bytes ----
//
// A five-trit byte's digits come out two at a time: (b * 9) >> 8 is 3*t0 + t1, and re-multiplying
// the low byte gives 3*t2 + t3, then 3 gives t4. That nine-valued code is a whole operand BYTE
// when the operand is nibbles, so one `v_perm_b32` over a constant table turns four codes into
// eight ternary nibbles -- eight weights per instruction. As IU8 bytes the same four codes need a
// table lookup per byte and two more permutes to interleave them, which is the 271 slots.
//
// The A4 FFN already runs this machinery (kernels/ffn_a4_operands.hpp); what is new here is that
// the DEPLOYED path can use it without a four-bit activation, and that the HALO element order is
// already the right one.
//
// ---- the element order is already right ----
//
// src/halo_format.h fixes trit order by the peel: for byte pair p = 2d+w of qs dword d,
//   e = 8p + {0,1,2,3} <- t0(first), t1(first), t0(second), t1(second)
//   e = 8p + {4,5,6,7} <- t2(first), t3(first), t2(second), t3(second)
// so the four codes {A(first), A(second), B(first), B(second)} of one byte pair are exactly the
// four consecutive element pairs of ONE eight-element group -- one dot8 operand, in order. The
// fifth trits (e = 96 + 4d + j) and the two qh bytes fill the last four operand dwords the same
// way. Operand dword i covers elements 8i .. 8i+7 with nibble j = element 8i+j, for i in [0,16).
//
// ---- int8 activations keep the full MAC rate ----
//
// IU4 wants four-bit activations and this path has eight-bit ones. Split them instead:
// a = 16*a_hi + a_lo with a_lo the low nibble read unsigned and a_hi the high nibble read signed,
// so sum_k w_k a_k = 16 * sum_k w_k a_hi_k + sum_k w_k a_lo_k. Two dot8 per eight elements retires
// the same four useful MACs per instruction one dot4 per four elements does, and every term is an
// exact int32, so the block's integer sum -- and therefore the float that reaches `fmaf` -- is bit
// for bit the one the byte operand produces. The two halves are independent accumulator chains by
// construction, which is the split docs/mv-dot4-body.md had to introduce by hand.
//
// ---- the relabel that makes v_perm free ----
//
// v_perm_b32 offers eight dynamic table bytes for selectors 0..7 and a nine-valued code needs
// nine. Selector 8 hands back the sign replication of source byte 1 (byte 1 of `b`), which is
// 0x00 whenever that byte's top bit is clear. Code 8 is the trit pair (2,2), so store the operand
// under the relabel v = 2 - t: code 8's operand byte becomes (0,0) = 0x00, every other byte is at
// most 0x22 and none of them can turn selector 8 into 0xFF. The dot then computes
//   V = sum_k (2 - t_k) a_k = 2 * xsum - sum_k t_k a_k,
// so the block's deployed accumulator `sum t a - xsum` is `xsum - V`: the same integer, one
// subtraction, in the other direction.
//
// kernels/mv_nibble_check.cpp runs THIS source on the host against src/halo_format.h's own decode
// over every byte value and random blocks. The host arm of `nib_perm` reproduces the gfx1151
// selector rule the A4 FFN measured; the device arm is the instruction.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

namespace halo {

typedef unsigned short nib2 __attribute__((ext_vector_type(2)));

// Operand bytes of codes 0..7 under v = 2 - t, low nibble = the earlier element.
//   code = 3*t_u + t_v, byte = (2 - t_u) | ((2 - t_v) << 4)
//   0 -> 0x22   1 -> 0x12   2 -> 0x02   3 -> 0x21
//   4 -> 0x11   5 -> 0x01   6 -> 0x20   7 -> 0x10   8 -> 0x00 (selector 8, not a table byte)
constexpr unsigned NIB_T1 = 0x21021222u;   // bytes 0..3 = codes 0..3, the `b` argument
constexpr unsigned NIB_T0 = 0x10200111u;   // bytes 0..3 = codes 4..7, the `a` argument

// Gather the four codes of one byte pair: A(first), A(second), B(first), B(second). The codes sit
// in bytes 1 and 3 of the peel words because nothing shifts them down -- the selector says where.
constexpr unsigned NIB_AB = 0x07050301u;
// Gather the four fifth trits of one qs dword: t4(P0.first), t4(P1.first), t4(P0.second), t4(P1.second),
// which are elements 96+4d+0 .. 96+4d+3 in order.
constexpr unsigned NIB_C = 0x07030501u;
// Gather two folded code words (codes in bytes 0 and 2) into one selector.
constexpr unsigned NIB_CC = 0x06040200u;

__host__ __device__ __forceinline__ unsigned nib_perm(unsigned a, unsigned b, unsigned sel) {
#ifdef __HIP_DEVICE_COMPILE__
    return __builtin_amdgcn_perm(a, b, sel);
#else
    unsigned d = 0;
    for (int i = 0; i < 4; i++) {
        const unsigned s = (sel >> (8 * i)) & 0xffu;
        unsigned byte;
        if (s <= 3) byte = (b >> (8 * s)) & 0xffu;
        else if (s <= 7) byte = (a >> (8 * (s - 4))) & 0xffu;
        else if (s == 8) byte = ((b >> 8) & 0x80u) ? 0xffu : 0x00u;   // sign of source byte 1
        else if (s == 13) byte = 0xffu;
        else byte = 0x00u;
        d |= byte << (8 * i);
    }
    return d;
#endif
}

// Four codes in the four bytes of `sel` -> eight ternary nibbles, relabelled v = 2 - t.
__host__ __device__ __forceinline__ unsigned nib_codes(unsigned sel) { return nib_perm(NIB_T0, NIB_T1, sel); }

// Codes of a byte pair held as two 16-bit lanes, in bytes 1 and 3 of each word.
// 9 * 255 = 2295 stays inside a 16-bit lane, so the whole pair peels in one packed multiply.
struct NibPeel { nib2 a, b, c; };

__host__ __device__ __forceinline__ NibPeel nib_peel(nib2 p) {
    const nib2 nine = {9, 9}, three = {3, 3}, mask = {0xff, 0xff};
    NibPeel r;
    r.a = p * nine;                    // 3*t0 + t1
    r.b = (r.a & mask) * nine;         // 3*t2 + t3
    r.c = (r.b & mask) * three;        // t4
    return r;
}
// The two qh bytes carry four trits, so their fifth step is never taken.
__host__ __device__ __forceinline__ NibPeel nib_peel_ab(nib2 p) {
    const nib2 nine = {9, 9}, mask = {0xff, 0xff};
    NibPeel r;
    r.a = p * nine;
    r.b = (r.a & mask) * nine;
    r.c = nib2{0, 0};
    return r;
}

__host__ __device__ __forceinline__ unsigned nib_word(nib2 v) { return __builtin_bit_cast(unsigned, v); }
__host__ __device__ __forceinline__ nib2 nib_pair(unsigned v) { return __builtin_bit_cast(nib2, v); }

// One qs dword: sixteen elements as two operand dwords, plus the two fifth-trit codes of its four
// bytes folded into bytes 0 and 2 of `cc`. Every byte of the fold is at most 2, so 3*g cannot
// carry out of its lane and neither can the add.
__host__ __device__ __forceinline__ void nib_dword(unsigned dw, unsigned & w0, unsigned & w1, unsigned & cc) {
    const NibPeel r0 = nib_peel(nib_pair(dw & 0x00ff00ffu));          // bytes 0 and 2
    const NibPeel r1 = nib_peel(nib_pair((dw >> 8) & 0x00ff00ffu));   // bytes 1 and 3
    w0 = nib_codes(nib_perm(nib_word(r0.b), nib_word(r0.a), NIB_AB));
    w1 = nib_codes(nib_perm(nib_word(r1.b), nib_word(r1.a), NIB_AB));
    const unsigned g = nib_perm(nib_word(r1.c), nib_word(r0.c), NIB_C);
    cc = g * 3u + (g >> 8);
}

// One HALO block for one lane -> sixteen IU4 operand dwords; operand i covers elements 8i..8i+7.
__host__ __device__ __forceinline__ void nib_block(const uint4 & qa, const uint2 & qb, unsigned tail, unsigned w[16]) {
    const unsigned dw[6] = { qa.x, qa.y, qa.z, qa.w, qb.x, qb.y };
    unsigned cc[6];
#pragma unroll
    for (int d = 0; d < 6; d++) nib_dword(dw[d], w[2 * d], w[2 * d + 1], cc[d]);
#pragma unroll
    for (int m = 0; m < 3; m++) w[12 + m] = nib_codes(nib_perm(cc[2 * m + 1], cc[2 * m], NIB_CC));
    const NibPeel t = nib_peel_ab(nib_pair((tail & 0xffu) | ((tail << 8) & 0xff0000u)));
    w[15] = nib_codes(nib_perm(nib_word(t.b), nib_word(t.a), NIB_AB));
}

} // namespace halo

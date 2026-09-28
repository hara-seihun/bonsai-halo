// The two-bit weight word the wide sequence projections store, in the order their IU8 operand
// wants to read it.
//
// `expand_i8` in ffn_operands.hpp turns sixteen two-bit codes into sixteen signed bytes. The
// matrix instruction wants one code per byte lane, and the deployed word keeps code k at bits 2k,
// so four codes share a byte and every output dword costs a spread: two shift/or pairs and two
// masks before the byte permute that actually selects the weight. Sixteen issue slots per K16
// slice, of which four are the permutes and twelve are moving bits into byte lanes.
//
// Nothing forces that word order. A code's bit position inside the stored word is a free choice of
// the packer, because only the consumer reads it. Put the code of K slot k at
//
//     bit 8 * (k & 3) + 2 * (k >> 2)
//
// and the spread is already done: selector dword j is (word >> 2j) & 0x03030303, whose byte i is
// exactly the code of K slot 4j + i. Eleven slots per slice instead of sixteen, still 2.000 bits
// per weight, still eight dwords per (row, 128-block), and the K order the B operand sees does not
// move, so the products and their summation order are the deployed ones bit for bit.
//
// The scaled-FP16 control route keeps the deployed order: `expand_scaled` reads the same four
// codes per byte and its own spread is fused into the selector pair it needs anyway.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include "ffn_operands.hpp"
#include "ffn_a4_operands.hpp"

namespace halo {
namespace ffnb {

// Which alphabet a stage's stored code word is written in. The image carries exactly one of them
// at a time, the three are the same 2.000 bits per weight in the same 512 bytes per (row tile,
// 128-block), and `reorder_codes` moves between any two through the deployed word.
enum SeqCodeOrder { SEQ_ORD_SPREAD = 0, SEQ_ORD_NIBBLE = 1, SEQ_ORD_PAIR = 2 };

// Deployed word (code k at bits 2k) -> spread word. A pure bit permutation; every code keeps its
// K slot. Host-callable so the repack path can be checked on the CPU.
__host__ __device__ __forceinline__ uint32_t spread_codes(uint32_t w) {
    uint32_t out = 0;
#pragma unroll
    for (int k = 0; k < 16; ++k)
        out |= ((w >> (2 * k)) & 3u) << (8 * (k & 3) + 2 * (k >> 2));
    return out;
}

// The inverse, for a host check of the packer.
__host__ __device__ __forceinline__ uint32_t unspread_codes(uint32_t d) {
    uint32_t out = 0;
#pragma unroll
    for (int k = 0; k < 16; ++k)
        out |= ((d >> (8 * (k & 3) + 2 * (k >> 2))) & 3u) << (2 * k);
    return out;
}

// The same freedom, spent on the four-bit instruction instead.
//
// v_wmma_i32_16x16x16_iu4 wants sixteen signed nibbles in two dwords. A nibble's bottom two bits
// are exactly a stored code, so put the code of K slot k at
//
//     bit 4 * (k & 7) + 2 * (k >> 3)
//
// and `w & 0x33333333` is the first operand dword's codes already in their nibbles, `(w >> 2) &
// 0x33333333` the second's. What is left is a 2 -> 4 bit sign extension, and the codes make that
// three instructions rather than the six a general one costs: bit 1 of a nibble is set exactly on
// code 3, the only negative weight, so ORing that bit back in at positions 2 and 3 turns 3 into
// 0xf and leaves 0 and 1 alone. Both ORs fuse with their shift into v_lshl_or_b32.
//
// Nine slots per K16 slice against the eight-bit operand's eleven, for half the matrix issue and
// half the activation bytes. The K order inside a 128-block is permuted relative to the eight-bit
// operand and that is exact: the block's accumulation is int32, and the activation operand is
// written in the same order by the prep.
__host__ __device__ __forceinline__ uint32_t nibble_codes(uint32_t w) {
    uint32_t out = 0;
#pragma unroll
    for (int k = 0; k < 16; ++k)
        out |= ((w >> (2 * k)) & 3u) << (4 * (k & 7) + 2 * (k >> 3));
    return out;
}

// The inverse, for a host check of the packer and for the order switch.
__host__ __device__ __forceinline__ uint32_t unnibble_codes(uint32_t d) {
    uint32_t out = 0;
#pragma unroll
    for (int k = 0; k < 16; ++k)
        out |= ((d >> (4 * (k & 7) + 2 * (k >> 3))) & 3u) << (2 * k);
    return out;
}

// Codes already in nibble bottoms -> signed int4 nibbles.
//
// Bits 2 and 3 of the nibble are both exactly bit 1 of the code, so each is one `v_lshl_or_b32`
// against a mask of the previous step. Written as `x | (h << 1) | (h << 2)` from a common `h` the
// compiler folds the two shifted ORs into `h * 6` and emits a quarter-rate `v_mul_lo_u32`; taking
// the second mask from the first result instead keeps both steps full rate and costs one AND.
__host__ __device__ __forceinline__ unsigned sext_nibbles(unsigned x) {
    const unsigned a = x | ((x & 0x22222222u) << 1);   // bit 2 of a nibble is now code bit 1
    return a | ((a & 0x44444444u) << 1);               // 3 -> 0xf, the int4 -1; 0 and 1 unchanged
}

// Sixteen nibble-ordered codes -> sixteen signed nibbles for v_wmma_i32_16x16x16_iu4.
__host__ __device__ __forceinline__ int2v expand_i4_nib(unsigned w) {
    int2v r;
    r[0] = int(sext_nibbles(w & 0x33333333u));
    r[1] = int(sext_nibbles((w >> 2) & 0x33333333u));
    return r;
}

// Sixteen spread codes -> sixteen signed bytes for v_wmma_i32_16x16x16_iu8, one byte permute per
// output dword against the same table `expand_i8` uses: selector 0 -> 0, 1 -> +1, 3 -> -1.
__device__ __forceinline__ int4v expand_i8_spread(unsigned d) {
    int4v r;
#pragma unroll
    for (int j = 0; j < 4; ++j)
        r[j] = __builtin_amdgcn_perm(0u, 0xff000100u, (d >> (2 * j)) & 0x03030303u);
    return r;
}

// ------------------------------------------------------ the same operand out of a coarser alphabet
//
// The nibble order above stores one weight per code and pays for the 2 -> 4 bit sign extension in
// the block loop: three instructions per operand dword plus the mask that isolates the codes, which
// the deployed 128-row body spends 96 of its 325 work slots on. Nothing about the sixteen weights
// requires them to be stored one per code.
//
// `ffn_a4_operands.hpp` already carries the alphabet that removes the arithmetic, for the A4 FFN's
// own weight image: a NINE-VALUED code names a WEIGHT PAIR, `3 * mu^-1(w_u) + mu^-1(w_v)` with
// mu(0) = -1, mu(1) = +1, mu(2) = 0, and one `v_perm_b32` against its two-word table turns four of
// those codes straight into four operand bytes - eight signed nibbles - because a code IS the
// permute's source-byte selector. Nine values fit a nibble, so two codes fit a byte and four
// weights fit a byte: 2.000 bits per weight, the same 512 bytes per (row tile, 128-block) this
// image already stores, the same eight dwords per (row, 128-block), the same two `global_load_b128`
// per lane-block.
//
// Put the code of the weight pair (K slot 2i, K slot 2i+1) in byte i's low nibble and the pair
// (K slot 8+2i, 9+2i) in byte i's high nibble, and
//
//     expand_codes(d & 0x0f0f0f0f)        is operand dword 0
//     expand_codes((d >> 4) & 0x0f0f0f0f) is operand dword 1
//
// with exactly the nibbles `expand_i4_nib` produces for the same weights, in the same positions:
// the permute table's low nibble is the lower K slot, so operand nibble 2i of dword 0 is K slot 2i
// and nibble 2i+1 is K slot 2i+1, which is the map the nibble order builds by hand. Five
// instructions per K16 slice against twelve, no register, no byte, and the K order the B operand
// sees does not move - so the products, their order and the int32 block sum are the deployed ones.

// Deployed word (code k at bits 2k) -> pair-code word. A relabeling of which bit carries which
// weight, like the two orders above, and reversible for the same reason.
__host__ __device__ __forceinline__ uint32_t pair_codes(uint32_t w) {
    uint32_t out = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t lo = 3u * code_mu((w >> (2 * (2 * i))) & 3u) + code_mu((w >> (2 * (2 * i + 1))) & 3u);
        const uint32_t hi = 3u * code_mu((w >> (2 * (8 + 2 * i))) & 3u) + code_mu((w >> (2 * (9 + 2 * i))) & 3u);
        out |= lo << (8 * i);
        out |= hi << (8 * i + 4);
    }
    return out;
}

// mu^-1 back to the stored two-bit code: weight 0 is code 0, +1 is code 1, -1 is code 3.
__host__ __device__ __forceinline__ uint32_t code_unmu(uint32_t v) { return v == 0u ? 3u : (v == 1u ? 1u : 0u); }

// The inverse, for a host check of the packer and for the order switch.
__host__ __device__ __forceinline__ uint32_t unpair_codes(uint32_t d) {
    uint32_t out = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t lo = (d >> (8 * i)) & 0x0fu, hi = (d >> (8 * i + 4)) & 0x0fu;
        out |= code_unmu(lo / 3u) << (2 * (2 * i));
        out |= code_unmu(lo % 3u) << (2 * (2 * i + 1));
        out |= code_unmu(hi / 3u) << (2 * (8 + 2 * i));
        out |= code_unmu(hi % 3u) << (2 * (9 + 2 * i));
    }
    return out;
}

// Sixteen pair-ordered codes -> sixteen signed nibbles for v_wmma_i32_16x16x16_iu4. Host-callable
// so `kernels/seq_op_check.cpp` proves the map against the nibble order with no GPU.
// On the device this is `expand_codes` from ffn_a4_operands.hpp, one `v_perm_b32`. On the host it
// is the same table read the way the measured gfx1151 selector semantics read it: selector bytes
// 0..3 name bytes of argB, 4..7 name bytes of argA, and 8 replicates the sign of argB byte 1, which
// is code 1's operand byte 0x1F and therefore the 0x00 that code 8 = (0, 0) needs.
__host__ __device__ __forceinline__ unsigned seq_expand_codes(unsigned sel) {
#if defined(__HIP_DEVICE_COMPILE__)
    return expand_codes(sel);
#else
    unsigned out = 0;
    for (int j = 0; j < 4; ++j) {
        const unsigned c = (sel >> (8 * j)) & 0xffu;
        const unsigned v = c < 4u ? (PERM_S1 >> (8 * c)) & 0xffu
                         : c < 8u ? (PERM_S0 >> (8 * (c - 4u))) & 0xffu
                                  : 0x00u;
        out |= v << (8 * j);
    }
    return out;
#endif
}
__host__ __device__ __forceinline__ int2v expand_i4_pair(unsigned d) {
    return int2v{ int(seq_expand_codes(d & 0x0f0f0f0fu)), int(seq_expand_codes((d >> 4) & 0x0f0f0f0fu)) };
}

// Any stored order back to the deployed word, and out to any stored order. `reorder_codes` is the
// only caller and it composes them, so an image can move between any two alphabets in one pass.
__host__ __device__ __forceinline__ uint32_t seq_to_deployed(uint32_t d, int ord) {
    return ord == SEQ_ORD_NIBBLE ? unnibble_codes(d) : ord == SEQ_ORD_PAIR ? unpair_codes(d) : unspread_codes(d);
}
__host__ __device__ __forceinline__ uint32_t seq_from_deployed(uint32_t w, int ord) {
    return ord == SEQ_ORD_NIBBLE ? nibble_codes(w) : ord == SEQ_ORD_PAIR ? pair_codes(w) : spread_codes(w);
}

} // namespace ffnb
} // namespace halo

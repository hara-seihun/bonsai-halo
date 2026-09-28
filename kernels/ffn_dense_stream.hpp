// Keeping the dense five-trit weight stream in flight across a block boundary.
//
// A wave in the optimized projection owns one 16-row weight tile and walks its 40 or 136 blocks in
// order. Each block's operand bytes are three loads per matrix, and every one of them is consumed
// by the expansion a few instructions later, so the compiler's schedule for a block is: issue the
// loads, wait, expand, issue sixteen or thirty-two matrix instructions, drain. Nothing in that loop
// asks for the *next* block's bytes, so a wave pays one memory round trip per block that only other
// resident waves can cover.
//
// That is affordable where a stage fills the machine and fatal where it does not. At 32 rows the
// down projection launches DH/16 = 160 waves onto 80 SIMD32 units, two per unit, and measures twice
// its own block-loop issue model; gate/up launches 1088 and measures 1.27x. This cursor holds the
// next block's bytes in registers and hands out the current ones, so a wave always has a block in
// flight and the wait it pays has already been overlapped with a whole block of expansion and
// matrix work.
//
// The bytes are the image's bytes and the expansion is the same expansion: this changes when a load
// is issued, not what it returns, so every output bit is unchanged.
//
// The cost is fourteen VGPRs held across the block body. Only shapes with register headroom should
// take it - the dense arm allocates 200 of the 256 that keep seven waves per SIMD32 resident.
#pragma once
#include <hip/hip_runtime.h>
#include "ffn_a4_operands.hpp"

namespace halo {
namespace ffnb {

// One (row tile, 128-block) of the dense five-trit image, as one lane holds it.
struct DenseBlockWords {
    uint4 a0, b0;        // S0: sixteen bytes, five trits each, for both matrices
    uint2 a1, b1;        // S1: eight bytes
    unsigned a2, b2;     // S2: the two four-trit tail bytes
};

// DEPTH blocks are in flight: the one being handed out and DEPTH-1 already requested. One block of
// lookahead covers about 1150 cycles of issue, which is what a block of this map costs, so a stage
// whose waves cannot cover the rest of a memory round trip between them wants two.
//
// Three is measured and rejected, at both shapes this arm serves: the third block in flight costs
// 29 registers and takes the block from seven waves per SIMD32 to six, and the wave slot is worth
// more than the depth. 32-row prefill FFN phase 29.53 ms at depth two against 30.48 at depth
// three, normalised against the phases the cursor cannot touch; the 32-stream generation step
// agrees. docs/ffn-dense-loads.md has the panel. The weight stream is not what this arm waits
// for once the block's own B fragments are allowed to move (`DenseSched` in ffn_batch.hip).
//
// MATS = 1 is the down projection's one-matrix-per-wave ownership (k_proj_opt in ffn_batch.hip):
// the wave holds one weight stream, so the cursor requests three loads a block instead of six and
// costs half the registers at the same depth.
//
// ON = false compiles to nothing, so a map that does not read this image pays no register for it.
template <bool ON, int DEPTH = 2, int MATS = 2>
struct DenseStream {
    DenseBlockWords w[DEPTH];
    const uint8_t * ap;
    const uint8_t * bp;
    int col;
    // Bytes between a tile's block b and block b+1, which is one run of the image's order times
    // the block size: DENSE_BLOCK_BYTES on the deployed tile-major image and ntiles times that
    // where the image stores block-major (ffn_batch.hip, FfnRunOrder). Wave-uniform either way.
    int stepb;

    __device__ __forceinline__ DenseStream(const uint8_t * a, const uint8_t * b, int col_, int nb,
                                           int stepb_ = DENSE_BLOCK_BYTES)
        : ap(a), bp(b), col(col_), stepb(stepb_) {
#pragma unroll
        for (int i = 0; i < DEPTH; ++i) { fill(w[i]); advance(i + 1 < nb); }
    }

    __device__ __forceinline__ void fill(DenseBlockWords & d) const {
        d.a0 = dense_load0(ap, col);
        d.a1 = dense_load1(ap, col);
        d.a2 = dense_load2(ap, col);
        if constexpr (MATS == 2) {
            d.b0 = dense_load0(bp, col);
            d.b1 = dense_load1(bp, col);
            d.b2 = dense_load2(bp, col);
        }
    }

    // Wave-uniform and false past the last block, so the cursor never leaves its tile's image; a
    // repeated address is a cache hit rather than a branch in the block loop.
    __device__ __forceinline__ void advance(bool more) {
        const int step = more ? stepb : 0;
        ap += step; bp += step;
    }

    // Hand out the oldest block in flight and request one more.
    __device__ __forceinline__ DenseBlockWords take(bool more) {
        const DenseBlockWords cur = w[0];
#pragma unroll
        for (int i = 0; i + 1 < DEPTH; ++i) w[i] = w[i + 1];
        fill(w[DEPTH - 1]);
        advance(more);
        return cur;
    }
};

template <int DEPTH, int MATS>
struct DenseStream<false, DEPTH, MATS> {
    __device__ __forceinline__ DenseStream(const uint8_t *, const uint8_t *, int, int,
                                           int = DENSE_BLOCK_BYTES) {}
};

// ---------------------------------------------------------------- the block's scalar operands
//
// A block spends four kinds of operand and three of them were already streamed a block ahead: the
// weight bytes through the cursor above, the activation fragments through the LDS stage. The
// fourth kind is two words wide and was left in the block that spends it - and on gfx11 that is
// what the block's whole schedule ends up waiting for.
//
//   the weight scale   one `global_load_d16` per matrix, converted and broadcast to the eight row
//                      positions by `ds_bpermute`, which is the FIRST thing the block body does.
//   the token scale    one `global_load_b32` per token tile, multiplied into the drain's `fmaf`,
//                      which is the LAST thing the block body does.
//
// Compiled, the deployed gate/up block issued the scale pair at slot 20 of 625 and waited for it at
// slot 47, then issued a token scale at slot 488 and waited at 493, and another at 548 waited at
// 551. Three memory round trips a block with 3 to 27 instruction slots of cover each, on a phase
// whose weight stream is already covered: the wave stalls in the same block that asked.
//
// This cursor gives those two loads the cover the other two operands have. Block b's scalars are
// requested at the top of block b-1, held in `TT + 2` registers, and rotated into place after the
// drain has spent them - so the wait the rotation forces lands at the END of a block for a request
// made at its top, and the value a block reads is already in a register. Nothing about the value
// moves: the same words, the same `half_bits_to_float`, the same `__shfl` positions and the same
// `fmaf` chain, in the same order.
//
// Past the last block the address is clamped rather than branched, exactly as `DenseStream` clamps
// its own: a repeated address is a cache hit, and a wave-uniform branch in this loop is not free.
//
// The rotation is a register copy that depends on a load, which is the shape `docs/ffn-decode-
// schedule.md` measured as a LOSS on the weight cursor. It is the right shape here for the reason
// that one was wrong: the weight cursor wants its request to stay in flight ACROSS the wait, and
// these two want their request to be waited for as late as possible in the block that issued it.
template <bool ON, int TT, bool TWOMAT, bool CS>
struct ScalarStream {
    const uint16_t * ap;      // this tile's weight-scale run, block b+1
    const uint16_t * bp;
    const float * cp;         // the activation scale image, block b+1, this token group
    int col;
    int sstep, cstep;         // elements between one block's run and the next
    unsigned aw, bw;          // the words block b+1 will spend
    float cs[TT];

    __device__ __forceinline__ ScalarStream(const uint16_t * a, const uint16_t * b, const float * c,
                                            int col_, int sstep_, int cstep_)
        : ap(a), bp(b), cp(c), col(col_), sstep(sstep_), cstep(cstep_) { load(); }

    // Request the run the cursor currently names. A map that does not spend an activation scale
    // (the FP16 arm keeps its scale in the operand) does not request one.
    __device__ __forceinline__ void load() {
        aw = ap[(unsigned) col];
        if constexpr (TWOMAT) bw = bp[(unsigned) col];
        if constexpr (CS)
#pragma unroll
            for (int t = 0; t < TT; ++t) cs[t] = cp[(unsigned) (t * 16 + col)];
    }

    // Step to block b+1 and request it. Wave-uniform and clamped at the last block.
    __device__ __forceinline__ void step(bool more) {
        const int s = more ? sstep : 0, c = more ? cstep : 0;
        ap += s; bp += s; cp += c;
        load();
    }
};

template <int TT, bool TWOMAT, bool CS>
struct ScalarStream<false, TT, TWOMAT, CS> {
    unsigned aw, bw;
    float cs[TT];
    __device__ __forceinline__ ScalarStream(const uint16_t *, const uint16_t *, const float *,
                                            int, int, int) {}
    __device__ __forceinline__ void step(bool) {}
};

} // namespace ffnb
} // namespace halo

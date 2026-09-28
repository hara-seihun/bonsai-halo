// How the sequence projection reaches the weight stream, and when.
//
// A wave in `project` owns one 16-row weight tile and walks its `nb` 128-blocks in order. Each
// block starts with two dependent memory reads and nothing to cover them: thirty-two bytes of
// two-bit codes as a pair of `global_load_b128`, and the block's sixteen-bit weight scale, whose
// eight `ds_bpermute` broadcasts cannot start until it lands. Only then does the expansion run and
// the block's sixteen to thirty-two matrix instructions issue. Nothing in that loop asks for block
// `b + 1`, so every wave pays one memory round trip per block, and the only thing that can cover
// it is another resident wave.
//
// Whether that is affordable is a property of the launch, not of the kernel, and the two stages of
// this kernel land on opposite sides of it. `kernels/ffn_dense_stream.hpp` reached the same
// conclusion from the FFN's side - "affordable where a stage fills the machine and fatal where it
// does not" - and bought 13.705 to 9.835 ms on a down projection launching 160 waves.
//
// Two instruments in this repository predicted the split before this header existed. The per-phase
// clock fit in `docs/clock-power.md` puts the output projection at 59.6% clock-proportional
// against the input stage's 83.5%: two fifths of the output stage is time that does not respond to
// the shader clock, which is what a memory stall looks like and not what issue looks like. And
// `docs/sequence-fragment-reuse.md` measured the removal of the slice barrier as a null on the
// input stage and -4 to -6% on the output one, for the same reason - lookahead can only pay where
// there is clock-independent time to recover.
//
// Three arms, each adding one thing to the one before, so a panel can say which half pays:
//
//   SEQ_OP_LANE    the deployed path. Each lane holds its own byte pointer into the weight image,
//                  so every one of those three loads carries a 64-bit VGPR address pair, and the
//                  load is issued at the head of the block that consumes it.
//   SEQ_OP_BASE    the same schedule, reached from a wave-uniform base with an unsigned lane
//                  offset. `afc9679` gave the activation fragment this form and left the weight
//                  side alone; the emitted block still shows two `global_load_b128` and one
//                  `global_load_d16_b16` on VGPR pairs against thirty-two fragment loads on a
//                  scalar base.
// A third arm held one 128-block of the weight stream in registers so the wait a wave pays would
// already have been overlapped with a whole block of expansion and matrix work. It lost at every
// shape measured - a third of the base arm's gain at 128 rows and +22% at 32 - and is not here.
// docs/sequence-weight-address.md has its numbers and why lookahead is the wrong prescription for
// a block that already carries thirty-two outstanding fragment loads.
//
// None of them changes what is read. Same bytes, same expansion, same K order, same summation
// order, so every output bit is the deployed one.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

namespace halo {
namespace ffnb {

enum SeqOperandPath { SEQ_OP_LANE = 1, SEQ_OP_BASE = 2 };

// One (row tile, 128-block) of the spread two-bit image as one lane holds it, with the block's
// weight scale. Nine VGPRs per row tile per block in flight.
struct SeqBlock {
    unsigned w[8];
    unsigned scale;
};

__device__ __forceinline__ void seq_read_block(SeqBlock & d, const uint8_t * code, const uint16_t * scale) {
    const uint4 lo = *(const uint4 *) code, hi = *(const uint4 *) (code + 16);
    d.w[0] = lo.x; d.w[1] = lo.y; d.w[2] = lo.z; d.w[3] = lo.w;
    d.w[4] = hi.x; d.w[5] = hi.y; d.w[6] = hi.z; d.w[7] = hi.w;
    d.scale = scale[0];
}

// The deployed arm holds nothing and does nothing: `project` keeps the deployed operand path
// written out verbatim in its own `if constexpr` branch, so the default instantiation emits the
// kernel canonical main emits rather than a re-expression of it. A control built inside a new loop
// nest is not a control - docs/sequence-projection-operands.md paid for that lesson once already.
// The last two arguments are the run index's block step in bytes and in scale halves: 512 and 16
// in the deployed tile-major image, ntiles*512 and ntiles*16 when the image is block-major.
// kernels/sequence_batch.hip carries the order and docs/weight-stream-order.md why it matters.
template <int OP, int RT>
struct SeqStream {
    __device__ __forceinline__ SeqStream(const uint8_t *, const uint16_t *, unsigned, unsigned,
                                         size_t, size_t, unsigned, unsigned) {}
    __device__ __forceinline__ void take(SeqBlock (&)[RT], int, int) {}
};

// Wave-uniform base, advancing unsigned offset. The base stays where the tile starts and the
// offset carries both the lane's column and the block walk, which is the form the activation
// fragment already compiles to; advancing the base instead leaves the load on a VGPR address pair,
// measured and rejected before this arm took its present shape.
template <int RT>
struct SeqStream<SEQ_OP_BASE, RT> {
    const uint8_t * cb;
    const uint16_t * sb;
    unsigned coff, soff, cstride, sstride, cblk, sblk;

    __device__ __forceinline__ SeqStream(const uint8_t * code, const uint16_t * scale,
                                         unsigned code_lane, unsigned scale_lane,
                                         size_t code_stride, size_t scale_stride,
                                         unsigned code_block, unsigned scale_block)
        : cb(code), sb(scale), coff(code_lane), soff(scale_lane),
          cstride((unsigned) code_stride), sstride((unsigned) scale_stride),
          cblk(code_block), sblk(scale_block) {}

    // The offset carries the block walk as a running add, the way the activation fragment's `bo`
    // does two loops below. Rebuilding it from the block index instead costs a shift and an add
    // per block, which is under a percent of a four-token-tile block and about two percent of a
    // one-tile one - measured as a regression at 32 and 64 rows before this took its present form.
    __device__ __forceinline__ void take(SeqBlock (& out)[RT], int, int) {
#pragma unroll
        for (int u = 0; u < RT; ++u)
            seq_read_block(out[u], cb + (coff + u * cstride), sb + (soff + u * sstride));
        coff += cblk; soff += sblk;
    }
};

} // namespace ffnb
} // namespace halo

#pragma once
#include "halo_kernels.h"
#include <hip/hip_runtime.h>

namespace halo {

// Wide input prep for the sequence path.
//
// The deployed route runs the layer's input norm, Hadamard, quantiser and alpha/beta projection
// inside the persistent kernel, once per eight-row slice, so a 128-row pass launches it sixteen
// times per layer. Every unit of that work is independent across rows, so this module runs the
// same units for the whole batch in one pair of ordinary launches.
//
// The arithmetic is the deployed arithmetic: the same 256-thread reduction over D for the norm
// scalar, the same per-thread fmaf chain for alpha/beta, the same warp Hadamard and the same
// round-to-nearest-even codes into the same wide operand. Only the number of launches, the unit
// ownership and the point at which the norm scalar is computed move.
struct SeqPrepArgs {
    const float * x;             // [rows][D] batch residual
    const float * norm_w;        // layer input norm weight, read by the alpha/beta projection
    const float * norm_s;        // norm weight * sign vector: the quantiser's kvec
    const unsigned short * ab_w; // [2*HV][D] alpha/beta projection; null on attention layers
    unsigned * seq_q;            // wide WMMA operand base (FwdParams::sequence_q)
    float * seq_scales;          // [row][NB_D]
    float * ab_out;              // [row][2*HV], row 0 base
    float * ninv;                // [row] norm scalar
    int rows, npad;
};

// HALO_WIDE_PREP=0 selects the deployed per-slice route as the measurement control.
bool sequence_prep_wide_enabled();
void launch_sequence_prep(const SeqPrepArgs & a, hipStream_t st);

// How many bits of activation each sequence projection reads, and in which operand.
//
//   SEQ_QUANT_A8   127 levels, sixteen signed bytes per k-group, v_wmma_i32_16x16x16_iu8.
//                  The deployed map.
//   SEQ_QUANT_A4E  7 levels, still stored as bytes and still multiplied by the IU8 instruction.
//                  Arithmetically it is the A4 map: every product, every int32 accumulator and
//                  every FP32 drain is the one SEQ_QUANT_A4 computes, so its logits are the
//                  four-bit map's logits to the bit. It exists to measure the quality of that map
//                  without its kernel, and to prove the kernel afterwards.
//   SEQ_QUANT_A4   7 levels in sixteen nibbles, v_wmma_i32_16x16x16_iu4 at half the issue.
//
// The two projections carry the coordinate independently, because they are quantised in different
// kernels: the input operand comes from the wide prep, the output operand from `resident_output`
// for a recurrent layer and from the wide attention combine for an attention one. Levels 0..2 are
// the input stage alone and mean exactly what they meant when they were measured
// (docs/sequence-input-a4.md); `-out` is the output stage alone and `-both` is the whole route.
//
// Everything but A8 changes the numbers the model computes. Process default from HALO_SEQ_QUANT
// (a8/8, a4e, a4/4, a4e-out, a4-out, a4e-both, a4-both), and `a8` restores the exact operand on
// both stages.
//
// A coordinate the process was *told* is a contract: a route that cannot carry it fails. The
// default is a preference, so a route that cannot carry it falls back to eight bits and says so
// once. `sequence_quant_explicit` is what separates them.
enum { SEQ_QUANT_A8 = 0, SEQ_QUANT_A4E = 1, SEQ_QUANT_A4 = 2,
       SEQ_QUANT_A4E_OUT = 3, SEQ_QUANT_A4_OUT = 4,
       SEQ_QUANT_A4E_BOTH = 5, SEQ_QUANT_A4_BOTH = 6, SEQ_QUANT_MAX = 6 };
int sequence_quant();
void sequence_set_quant(int q);
const char * sequence_quant_name(int q);
// True when HALO_SEQ_QUANT named this coordinate or a caller set it, rather than the process
// taking its default. A measurement arm that cannot run has to fail rather than quietly measure a
// different one.
bool sequence_quant_explicit();
// Move the process coordinate without marking it told, which is how a route that cannot carry the
// default drops one stage back to eight bits and leaves a later measurement arm free to fail.
void sequence_quant_fall_back_to(int q);
// Seven levels rather than 127; nibble is the narrower operand on top of that, which only changes
// the storage and the matrix instruction and never a product.
inline bool sequence_quant_in_seven(int q) { return q == SEQ_QUANT_A4E || q == SEQ_QUANT_A4 || q >= SEQ_QUANT_A4E_BOTH; }
inline bool sequence_quant_in_nibble(int q) { return q == SEQ_QUANT_A4 || q == SEQ_QUANT_A4_BOTH; }
inline bool sequence_quant_out_seven(int q) { return q >= SEQ_QUANT_A4E_OUT; }
inline bool sequence_quant_out_nibble(int q) { return q == SEQ_QUANT_A4_OUT || q == SEQ_QUANT_A4_BOTH; }
// True while the input projection reads four-bit activations, in either operand.
inline bool sequence_quant_four_bit(int q) { return sequence_quant_in_seven(q); }
// The same coordinate with one stage returned to the exact eight-bit operand. A pass whose
// operand came from a route with no seven-level store reads 127-level bytes whatever the
// coordinate says, so dropping that stage is what makes the stored numbers readable again.
inline int sequence_quant_drop_out(int q) {
    return q == SEQ_QUANT_A4E_OUT || q == SEQ_QUANT_A4_OUT ? SEQ_QUANT_A8
         : q == SEQ_QUANT_A4E_BOTH ? SEQ_QUANT_A4E
         : q == SEQ_QUANT_A4_BOTH ? SEQ_QUANT_A4 : q;
}
inline int sequence_quant_drop_in(int q) {
    return q == SEQ_QUANT_A4E || q == SEQ_QUANT_A4 ? SEQ_QUANT_A8
         : q == SEQ_QUANT_A4E_BOTH ? SEQ_QUANT_A4E_OUT
         : q == SEQ_QUANT_A4_BOTH ? SEQ_QUANT_A4_OUT : q;
}

}

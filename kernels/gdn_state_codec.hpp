#pragma once
// The recurrent state's storage coordinate.
//
// A 32-stream generation step loads and stores 302 MB of gated-delta state per sequence, once per
// recurrent layer, and docs/decode-map.md measures that round trip at 224 GB/s against the 242 GB/s
// this device reaches at all. That phase is 38% of the step and it is not waiting on arithmetic,
// occupancy or scheduling, so the only two levers on it are fewer round trips and fewer bytes per
// element. This header is the second one: it holds the state in a 16- or 8-bit coordinate and
// decodes it into the same registers the recurrence already uses.
//
// What is preserved exactly. A wave owns one whole state row: its 32 lanes hold columns
// `lane + 32*s` for s in 0..3, and both consumers of a row (the k and q dot products) are reduced
// across the wave by `warp_sum`. Every packed coordinate keeps that (lane, s) -> column map, so the
// recurrence runs the same summation tree over the same operands in the same order. The only
// numerical difference between a packed arm and the fp32 arm is the rounding of the value that
// crosses memory. What moves instead is the column order *in memory*: column c is stored at
// `(c & 31) * 4 + (c >> 5)`, which puts one lane's four columns in eight contiguous bytes and makes
// a row one 256-byte wave transaction instead of four 64-byte ones.
//
// Why a shared exponent rather than fp16. The row's two consumers are dot products against L2
// normalised k and q, so the error that matters is absolute, summed over 128 columns, against a
// signal of the row's own scale. fp16 spends five bits on an exponent range a normalised state row
// does not use and keeps eleven bits of *relative* precision on elements that contribute nothing.
// One fp32 scale per row spends all sixteen bits on mantissa and is about 4.5x more accurate in the
// readout at the same byte count. The scale is wave-uniform, so it costs one scalar load per row.
//
// Where the scale lives. The scales sit immediately above the packed values in the same
// (slot, layer) region, and the region is exactly as long as the format needs - `gdn_fmt_*` in
// halo_kernels.h owns that arithmetic and `gdn_region_floats()` is the stride the engine allocates.
// Nothing that copies, zeroes, snapshots or rolls back a slot has to know the format, because all
// of them move whole regions: a zeroed region still decodes to a zero state in every format.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "halo_kernels.h"
#include "device.hpp"

namespace halo {

static_assert(gdn_fmt_defer_off(GDN_STATE_I8) + (size_t) HV * gdn_fmt_defer_head(GDN_STATE_I8)
                  <= gdn_fmt_region_floats(GDN_STATE_I8), "an int8 region holds its own scales and triples");
static_assert(gdn_fmt_defer_head(GDN_STATE_I8) == GDN_DEFER_HEAD, "the packed pending block is the one it shipped with");
static_assert(GDN_DEFER_EXACT_MAX * 2 <= GDN_DEFER_MAX, "both coordinates stage through the same arrays");

// ---------------------------------------------------------------------------------------------
// Deferred commit: the other half of the same lever.
//
// Halving the bytes per element halves both directions of a round trip the step makes twice. Only
// one of those two is forced. Both consumers of a state row contract all 128 of its columns, so a
// step cannot produce its output without reading every element; the write-back is there only
// because the next step wants to find the state somewhere. What one step actually does to a head is
// add one rank-1 term and scale what was there:
//
//     S_C = (prod_t g_t) B  +  sum_i (prod_{t>i} g_t) a_i (x) k_i
//
// for a base `B` last written to memory, the token's decay `g`, the per-row delta `a` the delta
// rule produced and the token's normalised key `k`. C steps of a stream are C such triples, 1 kB
// per head against the 32 kB row image they modify, so keeping them in the free upper half of the
// same (slot, layer) region and rebuilding that sum at load lets the full write happen once per C
// steps. Traffic per step goes from 2 to 1 + 1/C plus the triples.
//
// The rebuild is element-wise - one fma per element per pending term, no reduction, nothing added
// to the token loop - so the recurrence inside a step still runs on exactly the registers and in
// exactly the order it runs on today. Two things change numerically. The decayed base and the
// pending terms are summed in a different order than the step-by-step chain sums them, and in a
// packed coordinate the state now crosses memory once per C steps rather than every step, which
// rounds it C times less often. The second one runs the opposite way to the first.
//
// AN EXACT COORDINATE DOES NOT HAVE TO ACCEPT EITHER OF THEM, and that is what the second rebuild
// form below is for. `gdn_token` advances an element as `m = fmaf(d, k, m * g)` - one rounded
// multiply, then one fused multiply-add - and an fp32 store is exact, so replaying the pending
// triples in CHRONOLOGICAL order with that same pair of operations reproduces the value the eager
// path would have written, element for element. Everything downstream follows by induction: the
// k-side contraction sees the same state, so the delta is the same, so the next term is the same.
// The suffix-product form is algebraically equal and cheaper by one multiply per element per term,
// and it is the right one where the values are rounded anyway; the chain form is the one that can
// be published as bit-identical. docs/gdn-defer-exact.md has the panel and the hashes.
//
// Where the triples live. A packed region reserves room for them above the scales - four control
// floats (count, entry, flush, spare) and then one block per head - so every reset, slot copy,
// snapshot and rollback already moves them with the state they belong to, and a zeroed region is a
// valid empty pending list over a zero base in every format. fp32 keeps no room for them and so
// cannot defer, and does not need to: it is the coordinate this is composed with rather than an
// alternative to it.
//
// The offsets take the format because the values below them do. `gdn_fmt_defer_ctl` and
// `gdn_fmt_defer_off` are constexpr, so inside the fp32 body and inside every branch that has
// already switched on the format they are literals, the same ones they were when the layout was
// fixed at half a region.
__device__ __forceinline__ const float * gdn_defer_head(int fmt, const float * region, int h) {
    return region + gdn_fmt_defer_off(fmt) + (size_t) h * gdn_fmt_defer_head(fmt);
}
__device__ __forceinline__ float * gdn_defer_head(int fmt, float * region, int h) {
    return region + gdn_fmt_defer_off(fmt) + (size_t) h * gdn_fmt_defer_head(fmt);
}
// The delta columns of one pending slot, one per rounding class: the token loop writes this row's
// value as it produces it, from the lane of each class that owns it.
__device__ __forceinline__ float * gdn_defer_deltas(int fmt, float * region, int h, int slot) {
    return gdn_defer_head(fmt, region, h) + gdn_fmt_defer_cap(fmt)
         + (size_t) slot * gdn_fmt_defer_classes(fmt) * SS;
}

// The halves of a pending term that belong to the head rather than to a row: the token's key, which
// every row of the head multiplied by its own delta, and the decay every later term carries. One
// wave of one row group writes them; the deltas are already in place from the token loop.
__device__ __forceinline__
void gdn_defer_append_head(int fmt, float * region, int h, int slot, float g, const float kk[4], int lane) {
    float * H = gdn_defer_head(fmt, region, h);
    float * K = H + (size_t) gdn_fmt_defer_cap(fmt) * (1 + (size_t) gdn_fmt_defer_classes(fmt) * SS)
              + (size_t) slot * SS;
    K[lane] = kk[0]; K[lane + 32] = kk[1]; K[lane + 64] = kk[2]; K[lane + 96] = kk[3];
    if (lane == 0) H[slot] = g;
}

// ---------------------------------------------------------------------------------------------
// Where an element lives, which is a free parameter and the only thing the `i8s`/`i8w` arms move.
//
// A packed head is 128 rows x 32 lane-quads. The deployed order stores row `j` as one contiguous
// 128-byte run, so a wave - which owns rows `part*RPU + wave + 8r` for r in 0..R - issues R loads
// of 128 bytes each, 8*128 = 1024 bytes apart. Nothing outside this header sees that order, and
// int8 stopped being byte-bound, so the order is worth choosing for the request shape instead:
//
//     group(j) = (j >> 5) * 8 + (j & 7)      // which 8-strided row family, 0..31
//     slot(j)  = (j >> 3) & 3                // which of its four rows, 0..3
//     quad     = group * 128 + lane * 4 + slot
//
// One lane's four rows become four adjacent quads - 16 contiguous bytes at int8 - so the AMDGPU
// load/store merger folds a wave's R accesses into one `dwordx4` per lane, and the wave's whole
// 4-row footprint is one contiguous 512-byte run. A head stays one contiguous 16 kB block and the
// map is a bijection on (row, lane, s), so this is a permutation of addresses and nothing else.
//
// It is a pure function of (head, row, lane) with no SPLIT, R or RPU in it, which is what lets the
// other decompositions of this recurrence call it: SPLIT 4 reads a whole run per wave with one
// dwordx4, SPLIT 8 half a run with a dwordx2, SPLIT 16 a quarter with a dword, and a SPLIT 2 wave
// two whole runs. What it costs is the converse: a single row is no longer contiguous, it is 32
// four-byte chunks at stride 16, so a reader that wants one row's columns in one instruction wants
// the deployed order and should say so here rather than carry a second expression.
//
// `(lane, s) -> column c = lane + 32s` is untouched in every arm. Both consumers of a row reduce
// across the wave with `warp_sum`, whose butterfly leaves two distinct bit patterns per wave (the
// quad-permute pair agrees within a quad, the row-rotate pair sums quad families j and j+2
// together), and `out_o` reads lane 0. Moving a column between lanes would have to reproduce that;
// moving an address cannot disturb it.
__device__ __forceinline__ bool gdn_fmt_is_i8(int fmt) { return gdn_fmt_is_eight_bit(fmt); }
// Grouped runs are the accepted coordinate; the two int8 controls keep the row-major one.
__device__ __forceinline__ bool gdn_fmt_grouped(int fmt) { return fmt == GDN_STATE_I8; }
// Quad index of (row j, lane) inside a packed head, split into the part that varies with the wave
// and the part that varies with the row. A quad is one lane's four columns: 4 bytes at int8, 8 at
// int16.
//
// The split is what makes the merge happen, and it is worth stating because the obvious one-
// expression version measures as a null. Written as one 32-bit index the four rows of a wave differ
// by an integer constant *inside* the index, so the backend emits `v_or_b32 4, voff` and hands each
// load its own vector address - four addresses the load/store merger cannot relate, four
// `global_load_b32`. Split so the row contributes a separate constant added to a `size_t` base, the
// same four loads become one base plus immediate offsets 0/4/8/12, which is the form
// `SILoadStoreOptimizer` folds into one `global_load_b128`. Same addresses either way.
__device__ __forceinline__ size_t gdn_pack_run(int fmt, int j, int lane) {
    if (!gdn_fmt_grouped(fmt)) return (size_t) (j * (SS / 4) + lane);
    return (size_t) (((j >> 5) * 8 + (j & 7)) * 128 + lane * 4);
}
__device__ __forceinline__ int gdn_pack_slot(int fmt, int j) {
    return gdn_fmt_grouped(fmt) ? (j >> 3) & 3 : 0;
}
// Float index of row `j`'s shared exponent, permuted the same way so a wave's R scales are adjacent
// too. Wave-uniform in every arm.
__device__ __forceinline__ size_t gdn_pack_scale(int fmt, int h, int j) {
    if (!gdn_fmt_grouped(fmt)) return gdn_fmt_scale_off(fmt) + (size_t) h * SS + j;
    return gdn_fmt_scale_off(fmt) + (size_t) h * SS + (size_t) (((j >> 5) * 8 + (j & 7)) * 4)
         + (size_t) ((j >> 3) & 3);
}

__device__ __forceinline__ float gdn_half_to_float(unsigned short u) {
    __half h; __builtin_memcpy(&h, &u, sizeof h); return __half2float(h);
}
__device__ __forceinline__ unsigned short gdn_float_to_half(float v) {
    const __half h = __float2half_rn(v); unsigned short u; __builtin_memcpy(&u, &h, sizeof u); return u;
}

// The four values one lane owns in one state row. Passing this by value rather than writing into
// the caller's `m[R][4]` is not a style choice: a reference to that array is an address, and the
// deployed single-token kernel allocates 97 VGPRs instead of 96 with one taken - which is exactly
// the 16-waves-per-SIMD32 cliff, on a route that never packs anything.
struct GdnQuad { float v[4]; };

// A float broadcast from a lane index every lane agrees on.
//
// The deferred rebuild used this to pull one row's delta out of a coalesced column load. It now
// reads that value from the staging in LDS instead, so nothing calls it - but the reason it exists
// outlives the caller and belongs beside the pending updates it was written for.
// `__builtin_amdgcn_readlane` takes an integer, and C will convert a float operand's VALUE rather
// than move its bits, silently, at any optimisation level, with no warning. The first deferred
// rebuild did exactly that and every pending update became zero: step time did not move, the
// residual hash of the step that reads no pending updates was still bit-identical to the published
// one, the model wrote fluent text, and teacher-forced NLL went 2.406 to 2.671. Only the quality
// panel said anything. If you reach for a cross-lane broadcast of a float here, cast the bits.

// Both accessors take the head's fp32 base (the pointer the deployed path already computes) and,
// for the packed coordinates, the (slot, layer) region base and the head index. When the caller
// compiles for fp32 the format is a constant, the packed arms fold away and only `M` stays live.
__device__ __forceinline__
GdnQuad gdn_state_load_row(int fmt, const float * M, const float * region, int h, int j, int lane) {
    GdnQuad q;
    if (fmt == GDN_STATE_F32 || fmt == GDN_STATE_F32_PK) {
        #pragma unroll
        for (int s = 0; s < 4; s++) q.v[s] = M[j * SS + lane + 32 * s];
        return q;
    }
    const float sc = region[gdn_pack_scale(fmt, h, j)];
    if (gdn_fmt_is_i8(fmt)) {
        const char4 w = (((const char4 *) ((const char *) region + (size_t) h * (SS * SS)))
                         + gdn_pack_run(fmt, j, lane))[gdn_pack_slot(fmt, j)];
        q.v[0] = sc * (float) w.x; q.v[1] = sc * (float) w.y;
        q.v[2] = sc * (float) w.z; q.v[3] = sc * (float) w.w;
        return q;
    }
    const ushort4 w = ((const ushort4 *) ((const unsigned short *) region + (size_t) h * (SS * SS)))[j * (SS / 4) + lane];
    if (fmt == GDN_STATE_F16) {
        q.v[0] = gdn_half_to_float(w.x); q.v[1] = gdn_half_to_float(w.y);
        q.v[2] = gdn_half_to_float(w.z); q.v[3] = gdn_half_to_float(w.w);
    } else {
        q.v[0] = sc * (float) (short) w.x; q.v[1] = sc * (float) (short) w.y;
        q.v[2] = sc * (float) (short) w.z; q.v[3] = sc * (float) (short) w.w;
    }
    return q;
}

// The effective state row: the committed base carried forward by the decay accumulated since the
// commit, plus what each pending step added to this row. Newest term first, base last, and `pend`
// is wave-uniform, so the whole loop is the same trip count for every lane.
// Copies a head's pending keys into the block's LDS and computes the weight each pending term
// carries now: w_i = prod_{t>i} g_t, with the base's own factor in `w[pend]`. Every thread of the
// block participates in the keys; one computes the weights, because the product is serial and eight
// long. The caller owns the barrier that follows.
//
// The weights are what makes the rebuild parallel. Computed inside the per-row loop they put a
// dependent multiply between every pending term and the next, and the first device panel paid for
// it: the loop issued a global load and a multiply that each iteration had to wait for, 8.3 ms per
// step at three pending terms - as much as writing the whole state.
//
// The delta columns are staged with them, and that is not cosmetic. Read in place the rebuild's
// term loop does `A[i*SS + part*RPU + lane]` and then four `v_readlane` on the result, so every
// pending term puts a full global round trip on the loop's critical path with nothing to cover it:
// the loop's trip count is a runtime value, so the loads cannot all be hoisted. Staged here the
// same bytes arrive once, coalesced, behind the barrier this function already needs, and the term
// loop reads a wave-uniform LDS address instead. Same values, same order, same fma - bit-identical.
__device__ __forceinline__
void gdn_defer_stage(int fmt, const float * region, int h, int pend, float * keys, float * w, int tid, int nthread,
                     float * dels, int part, int rpu) {
    const int CL = gdn_fmt_defer_classes(fmt);
    const float * H = gdn_defer_head(fmt, region, h);
    const float * K = H + (size_t) gdn_fmt_defer_cap(fmt) * (1 + (size_t) CL * SS);
    const float * A = H + gdn_fmt_defer_cap(fmt);
    for (int i = tid; i < pend * SS; i += nthread) keys[i] = K[i];
    // This unit's rows of every pending term, one run per delta class. Staged as [term][class][row]
    // so the rebuild's read is a wave-uniform LDS address per class.
    for (int i = tid; i < pend * rpu * CL; i += nthread) {
        const int t = i / (rpu * CL), c = (i / rpu) % CL, j = i % rpu;
        dels[i] = A[((size_t) t * CL + c) * SS + part * rpu + j];
    }
    if (tid == 0) {
        if (gdn_fmt_is_exact(fmt)) {
            // The chain form spends each token's own decay in its own turn, so the base arrives
            // unweighted and `w` carries the raw decays rather than their suffix products.
            for (int i = 0; i < pend; ++i) w[i] = H[i];
            w[GDN_DEFER_MAX] = 1.0f;
        } else {
            float acc = 1.0f;
            for (int i = pend - 1; i >= 0; --i) { w[i] = acc; acc *= H[i]; }
            w[GDN_DEFER_MAX] = acc;   // what the committed base is worth now
        }
    }
}

// One wave's rows of one head, rebuilt from the committed base and the staged pending terms.
//
// The loop runs over terms rather than rows, so a term's delta column is one coalesced load for the
// whole wave - the wave owns rows `part*RPU + wave + 8r` and lane `wave + 8r` of that load holds
// row r's value - and its key is four LDS reads shared by every row. Nothing in the body depends on
// the previous iteration.
template <int R, int RPU>
__device__ __forceinline__
void gdn_defer_rebuild(int fmt, const float * M, const float * region, const float * keys, const float * w,
                       int h, int part, int lane, int wave, int pend, float (&m)[R][4],
                       const float * dels) {
    const bool exact = gdn_fmt_is_exact(fmt);
    const float base = pend > 0 && !exact ? w[GDN_DEFER_MAX] : 1.0f;
    #pragma unroll
    for (int r = 0; r < R; r++) {
        const GdnQuad q = gdn_state_load_row(fmt, M, region, h, part * RPU + wave + 8 * r, lane);
        #pragma unroll
        for (int s = 0; s < 4; s++) m[r][s] = exact ? q.v[s] : base * q.v[s];
    }
    if (pend <= 0) return;
    if (exact) {
        // One token's effect on this element, in the order the token loop applied it, oldest term
        // first, and with the delta rounding this lane's own columns were updated with. R * 4 of
        // these chains are live per lane, so the serial dependence inside one element costs no
        // issue rate.
        const int cls = (lane >> 2) & 1;
        for (int i = 0; i < pend; ++i) {
            const float gi = w[i];
            const float * kv = keys + (size_t) i * SS, * dv = dels + (size_t) (2 * i + cls) * RPU;
            const float k0 = kv[lane], k1 = kv[lane + 32], k2 = kv[lane + 64], k3 = kv[lane + 96];
            #pragma unroll
            for (int r = 0; r < R; r++) {
                const float d = dv[wave + 8 * r];
                m[r][0] = fmaf(d, k0, m[r][0] * gi); m[r][1] = fmaf(d, k1, m[r][1] * gi);
                m[r][2] = fmaf(d, k2, m[r][2] * gi); m[r][3] = fmaf(d, k3, m[r][3] * gi);
            }
        }
        return;
    }
    for (int i = pend - 1; i >= 0; --i) {
        const float wi = w[i];
        const float * kv = keys + (size_t) i * SS, * dv = dels + (size_t) i * RPU;
        const float k0 = kv[lane], k1 = kv[lane + 32], k2 = kv[lane + 64], k3 = kv[lane + 96];
        #pragma unroll
        for (int r = 0; r < R; r++) {
            const float c = wi * dv[wave + 8 * r];
            m[r][0] = fmaf(c, k0, m[r][0]); m[r][1] = fmaf(c, k1, m[r][1]);
            m[r][2] = fmaf(c, k2, m[r][2]); m[r][3] = fmaf(c, k3, m[r][3]);
        }
    }
}

// Stores the wave's rows and leaves `m` holding exactly what a later load will read back. The
// deployed path commits mid-loop and then keeps walking the same registers; without the write-back
// a replayed token would continue from a value no reload can reproduce, and the two routes would
// disagree about a state they are both supposed to own.
__device__ __forceinline__
GdnQuad gdn_state_store_row(int fmt, float * M, float * region, int h, int j, int lane, GdnQuad q) {
    if (fmt == GDN_STATE_F32 || fmt == GDN_STATE_F32_PK) {
        #pragma unroll
        for (int s = 0; s < 4; s++) M[j * SS + lane + 32 * s] = q.v[s];
        return q;
    }
    if (fmt == GDN_STATE_F16) {
        ushort4 w;
        w.x = gdn_float_to_half(q.v[0]); w.y = gdn_float_to_half(q.v[1]);
        w.z = gdn_float_to_half(q.v[2]); w.w = gdn_float_to_half(q.v[3]);
        ((ushort4 *) ((unsigned short *) region + (size_t) h * (SS * SS)))[j * (SS / 4) + lane] = w;
        q.v[0] = gdn_half_to_float(w.x); q.v[1] = gdn_half_to_float(w.y);
        q.v[2] = gdn_half_to_float(w.z); q.v[3] = gdn_half_to_float(w.w);
        return q;
    }
    // Shared exponent. The row's magnitude is one `warp_max` away because the wave holds the whole
    // row, and the scale a reader will multiply by is the one written here, so the registers and a
    // reload agree bit for bit.
    const float qmax = gdn_fmt_is_i8(fmt) ? 127.0f : 32767.0f;
    float a = fmaxf(fmaxf(fabsf(q.v[0]), fabsf(q.v[1])), fmaxf(fabsf(q.v[2]), fabsf(q.v[3])));
    a = warp_max(a);
    const float scale = a * (1.0f / qmax), inv = a > 0.0f ? qmax / a : 0.0f;
    int c[4];
    #pragma unroll
    for (int s = 0; s < 4; s++) {
        const int e = __float2int_rn(q.v[s] * inv);
        c[s] = e > (int) qmax ? (int) qmax : e < -(int) qmax ? -(int) qmax : e;
    }
    if (gdn_fmt_is_i8(fmt)) {
        char4 w; w.x = (char) c[0]; w.y = (char) c[1]; w.z = (char) c[2]; w.w = (char) c[3];
        (((char4 *) ((char *) region + (size_t) h * (SS * SS)))
         + gdn_pack_run(fmt, j, lane))[gdn_pack_slot(fmt, j)] = w;
    } else {
        ushort4 w;
        w.x = (unsigned short) (short) c[0]; w.y = (unsigned short) (short) c[1];
        w.z = (unsigned short) (short) c[2]; w.w = (unsigned short) (short) c[3];
        ((ushort4 *) ((unsigned short *) region + (size_t) h * (SS * SS)))[j * (SS / 4) + lane] = w;
    }
    // The scale is wave-uniform, so `lane == 0` is a divergent store, a divergent store is a branch,
    // and a branch ends a basic block: the caller's `for r` store loop becomes R blocks that the
    // scheduler cannot mix, each covering its own five-deep `warp_max` butterfly with nothing. That
    // is the defect docs/gdn-token-reduce.md removed from the token loop for 24.5% of its issue
    // slots, still here on the store side, and it is paid once per unit - which is most of a unit at
    // a generation shape, where a unit walks one token. Every lane writing the same wave-uniform
    // value to the same address is one instruction with full exec, no branch, and the same byte.
    if (fmt != GDN_STATE_I8C) region[gdn_pack_scale(fmt, h, j)] = scale;
    else if (lane == 0) region[gdn_fmt_scale_off(fmt) + (size_t) h * SS + j] = scale;
    #pragma unroll
    for (int s = 0; s < 4; s++) q.v[s] = scale * (float) c[s];
    return q;
}

} // namespace halo

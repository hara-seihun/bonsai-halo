// Reductions for the attention score unit, rescheduled without changing a bit.
//
// `docs/long-context-attention.md` priced the score unit's key loop at 64 slots of `v_fmac_f32`
// against 47 of cross-lane DPP reduction and 63 of `s_delay_alu`, and left a lane-owns-a-key
// rewrite as a *reassociation* that would need its own quality panel. That framing is one step too
// pessimistic, and this header is where the cheaper half of it lives.
//
// `warp_sum` is `quad_perm(1,0,3,2)`, `quad_perm(2,3,0,1)`, `row_ror:4`, `row_ror:8`,
// `permlanex16`. Read at lane 0 - the only lane whose value the score loop stores - those five
// steps are a balanced binary tree over the 32 per-lane FMA chains, with the quads of each
// 16-lane row paired 0-3 and 2-1 rather than 0-1 and 2-3, because `row_ror:N` reads lane
// ((n mod 16) - N) mod 16. See docs/attn-score-schedule.md; three engineers here derived the
// index-ordered version from the mnemonic and confirmed each other before the direction was
// checked against the inclusive-scan idiom.
//
// So the reduction's *shape* is a fixed tree over a fixed leaf order, not an artefact of the lane
// layout. Nothing in this header depends on knowing that order: every function below REPLAYS the
// same cross-lane ops in a different sequence over independent values, which is exact whatever the
// ops mean. Reconstructing the tree in one lane is the other kind of argument, and it needs the
// device. What the deployed shape costs is not the tree, it is walking eight of them one after
// another: five dependent DPP steps per row, eight rows, and the compiler filling the gaps with
// `s_delay_alu`.
//
// `warp_sum_n` walks the eight trees breadth first - level one for all eight rows, then level two -
// so every dependent step has seven independent ones behind it. Same ops per value, same order per
// value, no reassociation to defend.
//
// `block_sum_n` / `block_max_n` do the same for the block reduction the softmax runs. The deployed
// unit calls `block_sum_256`/`block_max_256` once per row, and each of those carries two
// `__syncthreads`: a TT=8 unit spends **thirty-two barriers** reducing 8 x 128 floats. Batched,
// the same trees cost four. The per-row tree is untouched: `warp_sum`, then wave `w`'s result into
// `red[w]`, then the eight wave results summed in wave order.
#pragma once
#include <hip/hip_runtime.h>
#include "device.hpp"

// How many of the score unit's rows are reduced together in the key loop. 1 is the deployed
// sequential form. Wider hides the five-deep DPP chain behind independent work and costs live
// registers in the one loop that sets this kernel's VGPR count.
#ifndef ATTN_RED_W
#define ATTN_RED_W 8
#endif

namespace halo {
namespace {

// R independent wave reductions, one level at a time. dppf/xrow16 and their order are `warp_sum`'s.
template <int R>
__device__ __forceinline__ void warp_sum_n(float (&v)[R]) {
    #pragma unroll
    for (int i = 0; i < R; i++) v[i] += dppf<0xB1>(v[i]);
    #pragma unroll
    for (int i = 0; i < R; i++) v[i] += dppf<0x4E>(v[i]);
    #pragma unroll
    for (int i = 0; i < R; i++) v[i] += dppf<0x124>(v[i]);
    #pragma unroll
    for (int i = 0; i < R; i++) v[i] += dppf<0x128>(v[i]);
    #pragma unroll
    for (int i = 0; i < R; i++) v[i] += xrow16(v[i]);
}

template <int R>
__device__ __forceinline__ void warp_max_n(float (&v)[R]) {
    #pragma unroll
    for (int i = 0; i < R; i++) v[i] = fmaxf(v[i], dppf<0xB1>(v[i]));
    #pragma unroll
    for (int i = 0; i < R; i++) v[i] = fmaxf(v[i], dppf<0x4E>(v[i]));
    #pragma unroll
    for (int i = 0; i < R; i++) v[i] = fmaxf(v[i], dppf<0x124>(v[i]));
    #pragma unroll
    for (int i = 0; i < R; i++) v[i] = fmaxf(v[i], dppf<0x128>(v[i]));
    #pragma unroll
    for (int i = 0; i < R; i++) v[i] = fmaxf(v[i], xrow16(v[i]));
}

// R block reductions over NWv waves in two barriers instead of 2R. `red` is [NWv][R] scratch the
// caller owns; the leading barrier is `block_sum_256`'s, and it is what lets `red` be reused.
template <int R, int NWv>
__device__ __forceinline__ void block_sum_n(float (&v)[R], float * red) {
    warp_sum_n<R>(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) {
        #pragma unroll
        for (int i = 0; i < R; i++) red[w * R + i] = v[i];
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < R; i++) {
        float r = red[i];
        #pragma unroll
        for (int w2 = 1; w2 < NWv; w2++) r += red[w2 * R + i];
        v[i] = r;
    }
}

template <int R, int NWv>
__device__ __forceinline__ void block_max_n(float (&v)[R], float * red) {
    warp_max_n<R>(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) {
        #pragma unroll
        for (int i = 0; i < R; i++) red[w * R + i] = v[i];
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < R; i++) {
        float r = red[i];
        #pragma unroll
        for (int w2 = 1; w2 < NWv; w2++) r = fmaxf(r, red[w2 * R + i]);
        v[i] = r;
    }
}

// The key block one lane holds: DPL halves, DPL == 8 for the target geometry and 4 for the
// drafter's. Returned by value so the lookahead can hold it across the dot products without the
// compiler putting it in scratch.
template <int DPL>
__device__ __forceinline__ uint4 attn_kload(const __half * p) {
    uint4 r;
    if constexpr (DPL == 8) {
        r = *(const uint4 *) p;
    } else {
        const uint2 t = *(const uint2 *) p;
        r.x = t.x; r.y = t.y; r.z = 0u; r.w = 0u;
    }
    return r;
}

template <int DPL>
__device__ __forceinline__ void attn_kexpand(const uint4 & raw, float (&kf)[DPL]) {
    const __half2 * h2 = (const __half2 *) &raw;
    #pragma unroll
    for (int e = 0; e < DPL / 2; e++) { float2 f = __half22float2(h2[e]); kf[2 * e] = f.x; kf[2 * e + 1] = f.y; }
}

} // namespace
} // namespace halo

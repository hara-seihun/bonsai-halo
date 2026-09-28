// Phase machinery shared by the persistent kernels: device barrier, dynamic units, prep, matvec,
// target-geometry attention, embedding, argmax.
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "halo_kernels.h"
#include "halo_expand.hpp"
#include "device.hpp"
#include "attn_score.hpp"
#include "attn_leaf.hpp"
#include <algorithm>

namespace halo {
namespace {

constexpr int NT = 256;
constexpr int NW = NT / 32;
constexpr int LDS_FLOATS = 2304 + 64; // 9.5 KB: attention scores, top-k candidates, and the WMMA reduction ([7][8][32] twice) + scale tables


struct Ctx {
    unsigned * bar; unsigned * work; unsigned long long * prof;
    unsigned gen, phase; bool dirty;
    float * lds; float * red; int * s_unit;
};

__device__ __forceinline__ void grid_sync(Ctx & c) {
    __syncthreads();
    if (c.dirty) __threadfence();
    c.dirty = false;
    c.gen++;
    if (threadIdx.x == 0) {
        const unsigned target = c.gen * gridDim.x;
        __hip_atomic_fetch_add(c.bar, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        while (__hip_atomic_load(c.bar, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < target) __builtin_amdgcn_s_sleep(2);
    }
    __syncthreads();
    __threadfence();
    __builtin_amdgcn_s_dcache_inv();
    if (c.prof && blockIdx.x == 0 && threadIdx.x == 0) c.prof[c.gen] = wall_clock64();
}

// THE UNIT DEAL'S TWO BARRIERS ARE MEMORY FENCES, AND THEY DO NOT NEED TO BE.
//
// `__syncthreads()` is a seq_cst fence over every address space, which on gfx1151 the memory
// legalizer compiles to `s_waitcnt vmcnt(0) lgkmcnt(0); s_barrier`: every outstanding vector load
// and store of every wave is drained at it. A `ph_matvec` unit boundary carries three of those (the
// two here and the drain's) and a phase boundary two more, about 150,000 of them in one drafted
// verify pass - which is why no lookahead in this engine has ever survived a unit edge, whatever
// depth it was given.
//
// These two barriers protect ONE LDS WORD, `*c.s_unit`. Units inside a phase are independent by
// construction - that is what lets them be dealt dynamically - and every global write a unit makes
// is published by the `grid_sync` that ends the phase, which keeps its full fence. So the ordering
// this needs is LDS ordering, and the LDS-scoped fence says exactly that: `s_waitcnt lgkmcnt(0);
// s_barrier`, with the weight stream left in flight.
//
// Nothing else moves: same atomic, same counter, same slot, same dealing, same arithmetic,
// same addresses, same output bits. `-DHALO_UNIT_BARRIER_LDS=0` compiles the predecessor as the
// control. docs/phase-ramp-prefetch.md holds the panel and the compiled listings.
#ifndef HALO_UNIT_BARRIER_LDS
#define HALO_UNIT_BARRIER_LDS 1
#endif
__device__ __forceinline__ void unit_barrier() {
#if HALO_UNIT_BARRIER_LDS
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
#else
    __syncthreads();
#endif
}

// THE CONVOY, AND THE ONLY THING MEASURED TO BREAK IT.
//
// `grid_sync` releases every workgroup at the same instant, so a phase opens with all of them at
// the same offset inside their own units, asking the memory system for their first weight block
// together. They de-phase only as far as dynamic dealing lets them drift, which is why the same
// unit body reads 142 GB/s over one grid round, 161 over three and 211 over seventy-eight
// (`bench/coop_cost --mode 2 --tt 8 --pw 5`, and the engine's own `mv_lm_head` 208 against
// `mv_ssm_out` 128). Three things that should have fixed a cold pipeline are nulls in that probe -
// warming the next phase's first block across the barrier, dealing the next unit before the drain
// and warming it there, and LDS-scoping the unit barriers so a request can survive them - and the
// one that works is a per-workgroup DELAY: +8.0% and +7.8% at sixteen sleeps a step and +10.6% at
// sixty-four, palindrome ordered, on a 320-unit phase. Rotating which block range a wave takes,
// which de-phases the addresses without de-phasing the time, is a null. It is the arrival times
// that collide, not the addresses.
//
// TWO THINGS BOUND THE SIZE OF IT, and both were measured rather than chosen. A stagger is only
// absorbed where dealing has units left to move, so a phase that does not fill the grid twice gets
// none. And the spread that pays is about ONE UNIT WIDE: 64 steps is the best cell of the engine
// ladder at eight rows (16 is -1.2% of the verify pass, 64 is -8.1%, 192 gives 4.8% of it back),
// and 64 steps is 61k clocks against a 20 us unit. A one-row unit is eight times cheaper, and the
// same absolute stagger costs plain decode 1.0%, so the magnitude rides on the row count and lands
// at about a unit in both routes.
//
// It changes no address, no value and no order: the same workgroup computes the same unit from the
// same bytes, later. `-DHALO_PHASE_STAGGER=0` compiles the predecessor.
// docs/phase-ramp-prefetch.md holds the panel, the failed arms and the unit timeline.
#ifndef HALO_PHASE_STAGGER
#define HALO_PHASE_STAGGER 64
#endif
template <int ROWS>
__device__ __forceinline__ void phase_stagger(int total) {
    if constexpr (HALO_PHASE_STAGGER > 0) {
        constexpr int D = HALO_PHASE_STAGGER * ROWS / 8 > 0 ? HALO_PHASE_STAGGER * ROWS / 8 : 1;
        if (total < 2 * (int) gridDim.x) return;
        for (int i = (int) (blockIdx.x & 15u) * D; i > 0; i--) __builtin_amdgcn_s_sleep(1);
    }
    (void) total;
}

__device__ __forceinline__ int next_unit(Ctx & c, int total) {
    unit_barrier();
    if (threadIdx.x == 0) *c.s_unit = (int) (gridDim.x + atomicAdd(&c.work[c.phase], 1u));
    unit_barrier();
    const int u = *c.s_unit;
    return u < total ? u : -1;
}
__device__ __forceinline__ void end_phase(Ctx & c) { c.phase++; }

// Zero n floats with every workgroup, before the barrier that precedes a K-split matvec (KS > 1
// accumulates its parts with atomics and needs a zeroed or residual-holding output).
__device__ __forceinline__ void zero_floats(Ctx & c, float * p, size_t n) {
    for (size_t i = (size_t) blockIdx.x * NT + threadIdx.x; i < n; i += (size_t) gridDim.x * NT) p[i] = 0.0f;
    c.dirty = true;
}

// Same, for a span short enough to index with 32 bits: the store then takes the saddr form and
// costs one address register, which matters when it sits between two register-heavy phases.
__device__ __forceinline__ void zero_span(Ctx & c, float * __restrict__ p, int n) {
    for (int i = (int) (blockIdx.x * NT + threadIdx.x); i < n; i += (int) (gridDim.x * NT)) p[i] = 0.0f;
    c.dirty = true;
}

// The first key chunk a query at `pos` can see under a sliding window of `window` keys, and the
// one piece of arithmetic the producer of attention partials and the fold that consumes them have
// to agree on.
//
// `attn_window` was enforced inside the score body alone: a chunk every key of which the window
// excludes still got a unit, which read its keys and values, computed every dot product, and then
// masked the whole thing to zero. The DFlash2 drafter is this engine's only windowed attention
// (2048 keys over five layers) and it is the shape that shows what that costs - at an 11k context
// it was 88 chunks a head read to keep 17, 60% of the drafter's time, growing with the context
// rather than with its own window. A window of 0 returns chunk 0, so every target-model pass keeps
// the schedule and the bits it has always had. docs/drafter-window-units.md.
__device__ __host__ __forceinline__ int attn_chunk0(int pos, int window) {
    return window > 0 && pos >= window ? (pos - window + 1) / ACHUNK : 0;
}

// Scalar-cache pointer to the quantised activations of this phase. The launder keeps the compiler
// from treating the (constant address space) loads as invariant across barriers.
__device__ __forceinline__ cptr xptr(const int8_t * p) { asm volatile("" : "+s"(p)); return (cptr) p; }
__device__ __forceinline__ cfptr sptr(const float * p) { asm volatile("" : "+s"(p)); return (cfptr) p; }
__device__ __forceinline__ cptr iptr(const int * p) { asm volatile("" : "+s"(p)); return (cptr) p; }

// ---------------------------------------------------------------------------------------------
// Prep (norm / elementwise / Hadamard / int8 quant), one 1024-chunk of one row per workgroup.

struct PrepR {
    const float * x; const float * x2; const float * kvec; const float * norm_w;
    int n; int flags; float eps;
    int8_t * xq; float * xs; int * xsum;              // [row][n], [row][n/128]
    float * out_f32;                                   // [row][n]
    const AttnPartial * partials; const RowInfo * rows;
    float * cap; int cap_idx;                          // capture: cap[row][NCAP][D] slot cap_idx
    float * store_normed;                              // [row][n]: x * inv * norm_w
    float * ninv;                                      // [row]: the norm scalar, written by chunk 0
    const unsigned short * ab_w; float * ab_out; int ab_rows; // side work: ab_out[row][r] = sum_i ab_w[r][i] * x[row][i] * norm_w[i]
    int xq_stride, xq_off;                             // quantised output row stride and column offset (elements); 0 = n, 0
    int comb_heads, comb_hd, comb_chunks;              // ATTN_COMBINE geometry (defaults NH, HD, AMAX_CHUNKS); comb_chunks < 0: all chunks of pos
    int comb_window;                                   // sliding window the partials were produced under; 0 = none. Must be the `attn_window` of the launch that wrote them: `ph_attn` deals no unit for a chunk the window excludes, so those partials hold whatever the last step left.
    int part_stride;                                   // AttnPartial entries per (row, head); 0 = AMAX_CHUNKS
    // Direct wide operand for the sequence path (FwdParams::sequence_*). Read only by the DIRECT
    // instantiation of prep_chunk_r; null seq_q keeps the row-major xq/xs/xsum stores. The wide
    // destination is absolute in the batch, so it ignores xq_stride/xq_off, which the sequence
    // parts leave at their defaults.
    unsigned * seq_q; float * seq_scales; int seq_npad, seq_offset;
};

// Binds the wide destination when this instantiation owns it, so a body compiled without it keeps
// the exact code it has today.
template <bool DIRECT>
__device__ __forceinline__ void bind_seq_dest(PrepR & p, const FwdParams & P) {
    if constexpr (DIRECT) { p.seq_q = P.sequence_q; p.seq_scales = P.sequence_scales; p.seq_npad = P.sequence_npad; p.seq_offset = P.sequence_offset; }
}

// QL is the quantiser's level count, and NIB selects the nibble operand the four-bit sequence
// projection reads. QL = 127, NIB = false is the deployed instantiation and compiles to the code
// it has always compiled to: the level is a literal in both arms and only the wide store moves.
template <bool DIRECT = false, int QL = 127, bool NIB = false>
__device__ __forceinline__ void prep_chunk_r(const PrepR & a, int row, int chunk, float inv, float * s, int nrows_seq_ignored) {
    static_assert(QL == 127 || QL == 7, "the sequence quantiser has an eight-bit and a four-bit level");
    static_assert(!NIB || (QL == 7 && DIRECT), "the nibble operand is the four-bit wide destination");
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int e0 = wave * 128 + lane * 4;
    const int base = chunk * 1024 + e0;
    const size_t rb = (size_t) row * a.n;
    const float4 k = a.kvec ? *(const float4 *) (a.kvec + base) : make_float4(1.f, 1.f, 1.f, 1.f);
    float4 v;
    if (a.flags & PREP_GDN_NORM) {
        const int q = base >> 7, r = q % 3, kh = q / 3, hd0 = base & 127;
        const int src = 128 * (kh + 16 * r) + hd0;
        v = *(const float4 *) (a.x + rb + src);
        const float4 z = *(const float4 *) (a.x2 + rb + src);
        float ss = warp_sum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
        const float hinv = rsqrtf(ss / 128.0f + a.eps);
        v.x = v.x * hinv * k.x * silu(z.x); v.y = v.y * hinv * k.y * silu(z.y); v.z = v.z * hinv * k.z * silu(z.z); v.w = v.w * hinv * k.w * silu(z.w);
    } else if (a.flags & PREP_ATTN_COMBINE) {
        const int hd = a.comb_hd ? a.comb_hd : HD, nheads = a.comb_heads ? a.comb_heads : NH;
        const int hq = base / hd, d0 = base % hd;
        const int nchunks = a.comb_chunks > 0 ? a.comb_chunks : a.rows[row].pos / ACHUNK + 1;
        // A windowed query sees no key below `pos - window + 1`, so a chunk under `attn_chunk0` is
        // empty for this row: its unit wrote `m = -inf, l = 0, acc = 0`, which contributes
        // `exp(-inf - M) = 0` to the denominator and `fmaf(0, 0, s) = s` to the accumulator. The
        // fold starting above them drops exactly those terms - same M, same den, same bits - and it
        // is what lets `ph_attn` stop writing them. Read from the window and not from `rows`,
        // because a fold with no window must not touch a row index it does not own.
        const int cc0 = a.comb_window > 0 ? attn_chunk0(a.rows[row].pos, a.comb_window) : 0;
        const AttnPartial * part = a.partials + ((size_t) row * nheads + hq) * (a.part_stride ? a.part_stride : AMAX_CHUNKS);
        const float4 gate = a.x2 ? *(const float4 *) (a.x2 + (size_t) row * Q_OUT + hq * 2 * HD + HD + d0) : make_float4(0.f, 0.f, 0.f, 0.f);
        float M = -INFINITY;
        for (int cc = cc0; cc < nchunks; cc++) M = fmaxf(M, part[cc].m);
        float den = 0.0f;
        v = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int cc = cc0; cc < nchunks; cc++) {
            const float w = __expf(part[cc].m - M);
            den = fmaf(w, part[cc].l, den);
            const float4 t = *(const float4 *) (part[cc].acc + d0);
            v.x = fmaf(w, t.x, v.x); v.y = fmaf(w, t.y, v.y); v.z = fmaf(w, t.z, v.z); v.w = fmaf(w, t.w, v.w);
        }
        const float rden = 1.0f / den;
        if (a.x2) { v.x = v.x * rden * sigmoid(gate.x) * k.x; v.y = v.y * rden * sigmoid(gate.y) * k.y; v.z = v.z * rden * sigmoid(gate.z) * k.z; v.w = v.w * rden * sigmoid(gate.w) * k.w; }
        else { v.x *= rden * k.x; v.y *= rden * k.y; v.z *= rden * k.z; v.w *= rden * k.w; }
    } else if (a.flags & PREP_SILU_MUL) {
        v = *(const float4 *) (a.x + rb + base);
        const float4 u = *(const float4 *) (a.x2 + rb + base);
        v.x = silu(v.x) * u.x * k.x; v.y = silu(v.y) * u.y * k.y; v.z = silu(v.z) * u.z * k.z; v.w = silu(v.w) * u.w * k.w;
    } else {
        v = *(const float4 *) (a.x + rb + base);
        if (a.cap) *(float4 *) (a.cap + ((size_t) row * NCAP + a.cap_idx) * D + base) = v;
        if (a.store_normed) {
            const float4 nw = *(const float4 *) (a.norm_w + base);
            *(float4 *) (a.store_normed + rb + base) = make_float4(v.x * inv * nw.x, v.y * inv * nw.y, v.z * inv * nw.z, v.w * inv * nw.w);
        }
        if (a.flags & (PREP_NORM | PREP_SIGN)) {
            const float sc = (a.flags & PREP_NORM) ? inv : 1.0f;
            v.x *= sc * k.x; v.y *= sc * k.y; v.z *= sc * k.z; v.w *= sc * k.w;
        }
    }
    if (a.flags & PREP_HADAMARD) {
        { const float p0 = v.x + v.y, p1 = v.x - v.y, p2 = v.z + v.w, p3 = v.z - v.w; v.x = p0 + p2; v.y = p1 + p3; v.z = p0 - p2; v.w = p1 - p3; }
        #pragma unroll
        for (int bit = 1; bit < 32; bit <<= 1) {
            const bool upper = (lane & bit) != 0;
            const float ox = __shfl_xor(v.x, bit, 32), oy = __shfl_xor(v.y, bit, 32), oz = __shfl_xor(v.z, bit, 32), ow = __shfl_xor(v.w, bit, 32);
            v.x = upper ? ox - v.x : v.x + ox; v.y = upper ? oy - v.y : v.y + oy; v.z = upper ? oz - v.z : v.z + oz; v.w = upper ? ow - v.w : v.w + ow;
        }
        *(float4 *) (s + e0) = v;
        __syncthreads();
        if (tid < 128) {
            float u[8];
            #pragma unroll
            for (int m = 0; m < 8; m++) u[m] = s[tid + 128 * m];
            #pragma unroll
            for (int len = 1; len < 8; len <<= 1) {
                #pragma unroll
                for (int m = 0; m < 8; m++) if ((m & len) == 0) { const float x0 = u[m], x1 = u[m + len]; u[m] = x0 + x1; u[m + len] = x0 - x1; }
            }
            #pragma unroll
            for (int m = 0; m < 8; m++) s[tid + 128 * m] = u[m] * 0.03125f;
        }
        __syncthreads();
        v = *(const float4 *) (s + e0);
    }
    if (a.flags & PREP_SIGN_AFTER) { v.x *= k.x; v.y *= k.y; v.z *= k.z; v.w *= k.w; }
    if (a.flags & PREP_STORE_F32) *(float4 *) (a.out_f32 + rb + base) = v;
    if (a.flags & PREP_QUANT) {
        float amax = warp_max(fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
        const float iscale = amax > 0.0f ? (float) QL / amax : 0.0f;
        const int q0 = __float2int_rn(v.x * iscale), q1 = __float2int_rn(v.y * iscale), q2 = __float2int_rn(v.z * iscale), q3 = __float2int_rn(v.w * iscale);
        const unsigned word = (unsigned) (q0 & 0xff) | ((unsigned) (q1 & 0xff) << 8) | ((unsigned) (q2 & 0xff) << 16) | ((unsigned) (q3 & 0xff) << 24);
        if constexpr (DIRECT) if (a.seq_q) {
            // Wide WMMA operand: the 16 elements [base & ~15, +16) of this row are one k-group,
            // and consecutive token columns of a group are one word apart inside it. The lanes of
            // a 4-lane quad hold that group, so the quad writes its 16 bytes contiguously.
            //
            // NIB writes the same k-group as sixteen nibbles for v_wmma_i32_16x16x16_iu4: the
            // quad's four codes are one 16-bit half of one operand dword, so the quad still writes
            // its group contiguously, in eight bytes instead of sixteen. K slot k lands in nibble
            // k & 7 of dword k >> 3, which is the order the weight word is packed in; any k order
            // inside a 128-block is exact, because the block's int32 accumulation is.
            if constexpr (NIB) {
                const unsigned half4 = (unsigned) (q0 & 0xf) | ((unsigned) (q1 & 0xf) << 4) |
                                       ((unsigned) (q2 & 0xf) << 8) | ((unsigned) (q3 & 0xf) << 12);
                ((unsigned short *) a.seq_q)[(size_t) ((base >> 4) * a.seq_npad + a.seq_offset + row) * 4 + ((base >> 2) & 3)] = (unsigned short) half4;
            } else
                a.seq_q[(size_t) ((base >> 4) * a.seq_npad + a.seq_offset + row) * 4 + ((base >> 2) & 3)] = word;
            if (lane == 0) a.seq_scales[(size_t) (a.seq_offset + row) * (a.n >> 7) + (base >> 7)] = amax / (float) QL;
            return;   // no row-major intermediate, and the signed wide projection has no xsum term
        }
        const int sum = warp_sum_i(q0 + q1 + q2 + q3);
        const int qstride = a.xq_stride ? a.xq_stride : a.n;
        ((unsigned *) (a.xq + (size_t) row * qstride + a.xq_off))[base >> 2] = word;
        if (lane == 0) { const int blk = base >> 7; const size_t so = (size_t) row * (qstride / 128) + a.xq_off / 128; a.xs[so + blk] = amax / 127.0f; a.xsum[so + blk] = sum; }
    }
}

// units = nrows x chunks; unit u -> (row = u / nchunks, chunk = u % nchunks)
// DIRECT lets the quantiser write the wide operand of the sequence path; see PrepR::seq_q.
// SINGLE gives every workgroup exactly one unit from its block index, for an ordinary launch whose
// grid is the unit count instead of a persistent cooperative grid.
// QL and NIB are prep_chunk_r's quantiser level and operand width, passed through so a caller that
// writes a wide operand can write the four-bit one. <127, false> is the deployed instantiation.
template <bool DIRECT = false, bool SINGLE = false, int QL = 127, bool NIB = false>
__device__ __forceinline__ void ph_prep(Ctx & c, const PrepR & a, int nrows) {
    const int nchunks = a.n / 1024, nprep = nrows * nchunks, total = nprep + a.ab_rows;
    for (int u = blockIdx.x; u >= 0 && u < total; u = SINGLE ? -1 : next_unit(c, total)) {
        c.dirty = true;
        if (u >= nprep) {
            // side work: one gate-projection row against every activation row
            const int r = u - nprep;
            const unsigned short * w = a.ab_w + (size_t) r * D;
            float acc[RMAX];
            #pragma unroll
            for (int i = 0; i < RMAX; i++) acc[i] = 0.0f;
            for (int i = threadIdx.x * 4; i < D; i += NT * 4) {
                const uint2 w4 = *(const uint2 *) (w + i);
                const float4 n4 = *(const float4 *) (a.norm_w + i);
                const float w0 = bf16_to_f32(w4.x & 0xffff) * n4.x, w1 = bf16_to_f32(w4.x >> 16) * n4.y, w2 = bf16_to_f32(w4.y & 0xffff) * n4.z, w3 = bf16_to_f32(w4.y >> 16) * n4.w;
                #pragma unroll
                for (int rr = 0; rr < RMAX; rr++) {
                    if (rr < nrows) { const float4 x4 = *(const float4 *) (a.x + (size_t) rr * D + i); acc[rr] = fmaf(w0, x4.x, fmaf(w1, x4.y, fmaf(w2, x4.z, fmaf(w3, x4.w, acc[rr])))); }
                }
            }
            #pragma unroll
            for (int rr = 0; rr < RMAX; rr++) {
                if (rr < nrows) { const float t = block_sum_256(acc[rr], c.red); if (threadIdx.x == 0) a.ab_out[rr * a.ab_rows + r] = t; }
            }
            continue;
        }
        const int row = u / nchunks, chunk = u - row * nchunks;
        float inv = 1.0f;
        if (a.flags & PREP_NORM) {
            const float * x = a.x + (size_t) row * a.n;
            float ss = 0.0f;
            for (int i = threadIdx.x * 4; i < a.n; i += NT * 4) { float4 t = *(const float4 *) (x + i); ss = fmaf(t.x, t.x, ss); ss = fmaf(t.y, t.y, ss); ss = fmaf(t.z, t.z, ss); ss = fmaf(t.w, t.w, ss); }
            ss = block_sum_256(ss, c.red);
            inv = rsqrtf(ss / (float) a.n + a.eps);
            if (a.ninv && chunk == 0 && threadIdx.x == 0) a.ninv[row] = inv;
        }
        prep_chunk_r<DIRECT, QL, NIB>(a, row, chunk, inv, c.lds, 0);
        __syncthreads();
    }
    end_phase(c);
}

// ---------------------------------------------------------------------------------------------
// Matvec over TT rows. One lane = one output row of the weight, 8 waves split K, KS parts split K
// across units (parts accumulate with atomics). Activations come through the scalar cache.

struct MvSegR { const uint8_t * w; float * out; int ntiles; int add; };
struct MvR { MvSegR seg[3]; int nseg; int total_tiles; };

template <int NB, int KS> struct Geom {
    static constexpr bool SPECIAL = (NB == 136 && KS == 2); // 64 + 72 so every wave count is integral
    static constexpr int PART = NB / KS;
    static constexpr int PW_A = SPECIAL ? 8 : PART / NW;
    static constexpr int PW_B = SPECIAL ? 9 : PART / NW;
    static constexpr int A_BLOCKS = PW_A * NW;
    static_assert(SPECIAL ? (PW_A + PW_B) * NW == NB : PART * KS == NB && PW_A * NW == PART, "block split");
};

// xq/xs/xsum: [row][K] with row strides K (bytes) and K/128
// SM > 0 selects the single-token operand map for TT == 1 HALO weights; SM >= 3 also orders the
// units so consecutive ones are consecutive in the weight stream.
template <int NB, int KS, int TT, bool Q8, int SM = 0, bool ORDERED = false>
__device__ __forceinline__ void ph_matvec(Ctx & c, const MvR & m, const int8_t * xq_g, const float * xs, const int * xsum, int nrows) {
    using G = Geom<NB, KS>;
    constexpr int BB = Q8 ? Q8_TILE_BLOCK_BYTES : TILE_BLOCK_BYTES;
    const int lane = threadIdx.x & 31, wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    const int total = m.total_tiles * (ORDERED ? 1 : KS);
    cptr xq = xptr(xq_g);
    cfptr xsc = sptr(xs); cptr xsumc = iptr(xsum);
    // [NW - 1][TT][32]. Wave 0 keeps its own partials in registers and never stages them, so the
    // [8][TT][32] form reserved 32 * TT floats that nothing ever wrote - 1 kB of the 9.5 kB block
    // that decides how many of these workgroups fit on a WGP. The drain below reads the same
    // partials in the same left-to-right order; only the address moves.
    // docs/rows-lds-occupancy.md.
    float * red = c.lds; // [NW - 1][TT][32]
    phase_stagger<TT>(total);
    for (int u = blockIdx.x; u >= 0 && u < total; u = next_unit(c, total)) {
        c.dirty = true;
        // One workgroup owns the tile for every K part, so its stores follow part order.
        // Other callers retain their independent-unit schedule.
        for (int ordered_part = 0; ordered_part < (ORDERED ? KS : 1); ordered_part++) {
        constexpr bool CONTIG = SM >= 3 && KS > 1;   // units consecutive in the weight stream
        const int part = ORDERED ? ordered_part : CONTIG ? u - (u / KS) * KS : u / m.total_tiles;
        int tile = ORDERED ? u : CONTIG ? u / KS : u - part * m.total_tiles;
        MvSegR seg = m.seg[0];
        if (m.nseg > 1 && tile >= seg.ntiles) { tile -= seg.ntiles; seg = m.seg[1]; if (m.nseg > 2 && tile >= seg.ntiles) { tile -= seg.ntiles; seg = m.seg[2]; } }
        const int pw = (part == 0 || !G::SPECIAL) ? G::PW_A : G::PW_B;
        const int wb = (part == 0 ? 0 : (G::SPECIAL ? G::A_BLOCKS : part * G::PART)) + wave * pw;
        const uint8_t * run = seg.w + ((size_t) tile * NB + wb) * BB;
        float y[TT];
        #pragma unroll
        for (int r = 0; r < TT; r++) y[r] = 0.0f;
        if constexpr (SM > 0 && TT == 1 && !Q8) {
            if (part == 0 || !G::SPECIAL) mv_rows_pal<G::PW_A>(run, lane, xq + wb * 32, xsc + wb, xsumc + wb, y[0]);
            else                          mv_rows_pal<G::PW_B>(run, lane, xq + wb * 32, xsc + wb, xsumc + wb, y[0]);
        } else {
        if (part == 0 || !G::SPECIAL) mv_rows_t<G::PW_A, TT, Q8>(run, lane, xq + wb * 32, NB * 32, xsc + wb, xsumc + wb, NB, nrows, y);
        else                                 mv_rows_t<G::PW_B, TT, Q8>(run, lane, xq + wb * 32, NB * 32, xsc + wb, xsumc + wb, NB, nrows, y);
        }
        // The drain, spread over the waves that computed it, inside the NW-1 slots this block has.
        // The deployed form had wave 0 keep its own partial in registers and sum the other seven
        // itself: at eight rows that is 56 `ds_read_b32` and 8 output writes in ONE wave while seven
        // wait at the barrier below, once per unit, about 150,000 units in a verify pass.
        // `bench/coop_cost --drain` prices it at 4.4% of a 1088-unit phase and 8.9% of a 320-unit
        // K-split one (docs/matvec-drain.md).
        //
        // Row `r` belongs to wave `r` now. Eight partials need seven slots because a row's owner
        // reads its own from a register, so the owner's slot is free for that row and wave 0 writes
        // there: slot `r-1` holds wave 0's partial for row `r`, and wave `w > 0` skips the one row
        // it owns. Every (slot, row) still has exactly one writer and the sum is still
        // `y_0[r] + y_1[r] + ... + y_7[r]` left to right, so no output bit moves - the owner splices
        // its register in at position `wave` instead of reading a slot. `TT == 1` has one row to own
        // and keeps the deployed form, which makes single-token decode untouched code.
        constexpr bool SPREAD = TT > 1;
        if (wave > 0) {
            #pragma unroll
            for (int r = 0; r < TT; r++)
                if (!SPREAD || r != wave) red[((wave - 1) * TT + r) * 32 + lane] = y[r];
        } else if (SPREAD) {
            #pragma unroll
            for (int r = 1; r < TT; r++) red[((r - 1) * TT + r) * 32 + lane] = y[r];
        }
        __syncthreads();
        if (!SPREAD ? wave == 0 : wave < TT) {
            const int N = seg.ntiles * 32;
            #pragma unroll
            for (int r = 0; r < TT; r++) {
                if ((!SPREAD || r == wave) && r < nrows) {
                    // wave 0's partial: its own register at row 0, the owner's freed slot above it
                    float v = (!SPREAD || r == 0) ? y[r] : red[((r - 1) * TT + r) * 32 + lane];
                    #pragma unroll
                    for (int w = 1; w < NW; w++)
                        v += (SPREAD && w == r) ? y[r] : red[((w - 1) * TT + r) * 32 + lane];
                    float * out = seg.out + (size_t) r * N + (size_t) tile * 32 + lane;
                    if constexpr (KS > 1 && !ORDERED) atomicAdd(out, v);
                    else if (KS > 1 || seg.add) *out += v;
                    else *out = v;
                }
            }
        }
        __syncthreads();
        }
    }
    end_phase(c);
}

// ---------------------------------------------------------------------------------------------
// Multi-row matvec through WMMA (int8 16x16x16, wave32). The 32-row tile is consumed as two 16-row
// matrices: A1 = the lanes as loaded, A2 = the lane halves swapped. The hardware reads C row 2r from
// the lower lane half and row 2r+1 from the upper half, so with A1 lane l<16 reg r holds tile row 2r
// and lane l>=16 holds row 2r+17; with A2 the rows are 2r+16 and 2r+1. Column = lane % 16 = activation
// row. Activations come straight from L2: the distinct 16-byte addresses of a fragment load coalesce.
//
// COLS is how many of those sixteen columns carry a distinct activation row. Below sixteen rows the
// fragment repeats rows 0..COLS-1 and the epilogue drops the copies, so the matrix work of a slice is
// the same at COLS = 8 and COLS = 16 and only the number of rows it serves changes. Every consumer
// that can present sixteen rows should: the weights, the K order, the per-block int32 accumulation
// and the FP32 scale chain are identical, so only which lane holds a column moves.
//
// GROUPS is how many sixteen-column groups one unit serves, so a unit covers 16 * GROUPS activation
// rows. Sixteen columns is where the matrix instruction stops paying for itself and the *weight*
// side starts: a block's 896 bytes are loaded once, peeled out of radix-3 once into `tr[32]`, and
// swapped into its second fragment once, whatever the row count. Only the accumulators and the
// activation fragments are per group. So a unit of G groups reads the weight stream once for 16 * G
// rows instead of G times, and the caller runs G times fewer slices over the same batch. Every
// output element sees the same weights in the same K order through the same per-block int32
// accumulation and the same FP32 fold, so the value a row gets does not depend on which group
// carried it.

// How many rows the scalar-fed dot4 body keeps before the WMMA body takes over, by weight
// representation. The two bodies are numerically identical - both accumulate one exact int32 per
// 128-wide block, subtract the same activation sum and fold it with the same `fmaf(acc, wscale *
// xscale, y)` in the same block and the same wave order - so this is a pure schedule choice. It is
// not the same choice for both representations; docs/matvec-crossover.md measures it.
//
//   Ternary HALO tiles: one peel produces tr[32] for a whole block and every row reuses it, so the
//   dot4 body pays the expansion once and then spends 32 sudot4 per row. The WMMA body pays the
//   same peel and issues 16 wmma_iu8 for sixteen columns when a deployed pass has eight rows to
//   put in them. What decides it is not the issue count: at a fixed grid the dot4 body is slower.
//   It is that dot4 holds k_forward_rows<8> in 136 VGPRs instead of 217, which takes the kernel
//   from 6 waves per SIMD32 to 10 and the deployed cooperative grid from 60 workgroups to 100,
//   and the eight-row pass wants that grid badly.
//
//   Q8 drafter tiles: eight times the weight bytes per block and no peel at all, so the body is
//   memory-bound and the schedule is about hiding load latency rather than saving issue slots.
//   The WMMA body's wider operand loads win there, and forcing dot4 on the drafter costs 5%.
#ifndef MV_DOT4_MAX
#define MV_DOT4_MAX 8
#endif
#ifndef MV_DOT4_MAX_Q8
#define MV_DOT4_MAX_Q8 4
#endif
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
__device__ __forceinline__ int swap16(int v) { return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xfedcba98u, false, false); }

// WHERE THIS BLOCK'S WEIGHT REQUEST IS DRAINED, AND WHY MOVING IT IS NOT WORTH A TURN.
// gfx11 gives a wave one in-order `vmcnt` counter, so `s_waitcnt vmcnt(k)` waits for every request
// issued before the one it wants. This block requests the next block's 896 weight bytes first and
// its own operands after, so the accumulator seed's wait for `xsum` drains all three weight loads
// at slot 265 of a 476-slot block - `tools/vmcnt_cover.py` prints that directly, and the cover it
// reports is 264 slots with zero matrix instructions in it. That is the shape
// docs/ffn-decode-schedule.md found binding in `k_proj_opt`, and IT IS NOT BINDING HERE: pinning
// the weight cursor so every request hits cache (`-DHALO_MVW_PIN=1`, below) takes the phase down
// only 8% while the block grows 9% of its instructions, and requesting the operands before the
// weights - built, bit-identical, measured - is +0.55% normalised. The phase is issue-bound, the
// weight round trip is under a tenth of it, and 61% of the block is the matrix instructions
// themselves at the 21.9 issue slots `bench/wmma_cost` measures for `v_wmma_i32_16x16x16_iu8`.
// docs/ffn-slice-issue-order.md carries the panels and the losing arm as a patch.
// ORD 1 REQUESTS THE FIRST OPERAND OF EVERY GROUP AFTER THE FIRST AT THE TOP OF THE BLOCK.
// `tools/vmcnt_cover.py` says which request this body actually stalls on, and it is not the weight
// cursor: group 0's fragments get 260 to 288 slots of cover because the peel stands in front of
// them, and GROUP 1's get 34 to 47 with zero matrix instructions inside, because they are requested
// inside the loop that consumes them. Only the group's FIRST fragment matters - once its two matrix
// instructions are issued, every later fragment of that group is covered by the ones before it, and
// the drain table shows exactly that (34 slots on the first, then 1 to 10 matrix instructions of
// cover, each worth 21.9 slots). So this arm hoists one `int4` and the two scale words per extra
// group - six VGPRs at two groups - and buys them the whole peel plus sixteen matrix instructions.
// Same bytes, same addresses, same order of use, same arithmetic: bit-identical by construction.
template <int NBLK, bool Q8, int COLS = 8, int GROUPS = 1, int ORD = 0>
__device__ __forceinline__ void mvw_rows(const uint8_t * __restrict__ run, int lane, const int8_t * __restrict__ xrow, int xstride_unused,
                                         const float * xs, const int * xsum, int xsstride, float * scl,
                                         float (&y1)[GROUPS][8], float (&y2)[GROUPS][8]) {
    constexpr int PRE = (ORD && GROUPS > 1) ? GROUPS - 1 : 0;
    constexpr int BB = Q8 ? Q8_TILE_BLOCK_BYTES : TILE_BLOCK_BYTES;
    // Diagnostic builds, numerically invalid by construction and never selected at runtime:
    // -DHALO_MVW_PIN=1 holds the weight cursor on block 0 so every weight request hits cache,
    // -DHALO_MVW_PIN=2 does the same to the activation fragments, 3 pins both. The phase
    // difference is what that stream's round trip costs, which bounds any schedule change against
    // it. Read the block census of the pinned build before believing its number: pinning removes
    // an address cursor, and on this kernel the pinned block is 519 instructions against 476, so
    // the arm pays for its own diagnosis. A pinned build must never produce tokens.
#ifndef HALO_MVW_PIN
#define HALO_MVW_PIN 0
#endif
    constexpr bool PIN_W = HALO_MVW_PIN == 1 || HALO_MVW_PIN == 3;
    constexpr bool PIN_B = HALO_MVW_PIN >= 2;
    const int half = lane >> 4, col = lane & 15;
    uint4 qa[Q8 ? 8 : 1]; uint2 qb = make_uint2(0, 0); unsigned tail = 0;
    if (Q8) {
        #pragma unroll
        for (int i = 0; i < 8; i++) qa[i] = *(const uint4 *) (run + lane * 128 + i * 16);
        tail = *(const unsigned short *) (run + TILE_ROWS * 128 + lane * 2);
    } else {
        qa[0] = *(const uint4 *) (run + tile_off_qs_a(lane));
        qb = *(const uint2 *) (run + tile_off_qs_b(lane));
        tail = *(const unsigned *) (run + tile_off_tail(lane));
    }
    #pragma unroll 1
    for (int b = 0; b < NBLK; b++) {
        int4 pf[PRE ? PRE : 1]; float pfs[PRE ? PRE : 1]; int pfc[PRE ? PRE : 1];
        if constexpr (PRE) {
            #pragma unroll
            for (int g = 1; g < GROUPS; g++) {
                const int8_t * xpg = xrow + (PIN_B ? 0 : b * 128) + g * (16 * 128 * xsstride);
                const int arow = g * 16 + (col & (COLS - 1));
                pf[g - 1] = *(const int4 *) xpg;
                pfc[g - 1] = Q8 ? 0 : xsum[arow * xsstride + b];
                pfs[g - 1] = xs[arow * xsstride + b];
            }
        }
        uint4 nqa[Q8 ? 8 : 1]; uint2 nqb = qb; unsigned ntail = tail;
        #pragma unroll
        for (int i = 0; i < (Q8 ? 8 : 1); i++) nqa[i] = qa[i];
        if (b + 1 < NBLK) {
            const uint8_t * nrun = run + (PIN_W ? 0 : (b + 1) * BB);
            if (Q8) {
                #pragma unroll
                for (int i = 0; i < 8; i++) nqa[i] = *(const uint4 *) (nrun + lane * 128 + i * 16);
                ntail = *(const unsigned short *) (nrun + TILE_ROWS * 128 + lane * 2);
            } else {
                nqa[0] = *(const uint4 *) (nrun + tile_off_qs_a(lane));
                nqb = *(const uint2 *) (nrun + tile_off_qs_b(lane));
                ntail = *(const unsigned *) (nrun + tile_off_tail(lane));
            }
        }
        unsigned tr[32];
        float wscale;
        if (Q8) {
            #pragma unroll
            for (int i = 0; i < 8; i++) { tr[4 * i] = qa[i].x; tr[4 * i + 1] = qa[i].y; tr[4 * i + 2] = qa[i].z; tr[4 * i + 3] = qa[i].w; }
            wscale = __half2float(__ushort_as_half((unsigned short) tail));
        } else {
            const unsigned dw[6] = { qa[0].x, qa[0].y, qa[0].z, qa[0].w, qb.x, qb.y };
            // The trit is already a byte of the radix-3 product - `b * 3 <= 765`, so `(b * 3) >> 8`
            // sits in byte 1 and byte 3 of `r * 3` - and one `v_perm_b32` gathers four of them from
            // two products straight into the operand dword. The peel this replaces shifted each
            // product down and then moved the result back into a byte lane with `v_lshl_or_b32`:
            // 232 operations per 128-trit block against 168, same image, same bytes, same `tr[32]`.
            // The block loop goes 561 to 476 instructions at 238 VGPR either way, and
            // docs/mv-peel-gather.md measures what that is worth. `hx_expand_peel` in the same
            // header is the map this used to write out, kept as the readable statement of it.
            hx_expand_perm(dw, tail, tr);
            wscale = __half2float(__ushort_as_half((unsigned short) (tail >> 16)));
        }
        // scale table: [0][m] = even row 2m, [1][m] = odd row 2m+1
        scl[(lane & 1) * 16 + (lane >> 1)] = wscale;
        // A1: half0 -> even rows 2r ([0][r]), half1 -> odd 2r+17 ([1][8+r]); A2: half0 -> even 2r+16 ([0][8+r]), half1 -> odd 2r+1 ([1][r])
        const float4 * s1p = (const float4 *) (scl + half * 16 + 8 * half);
        const float4 * s2p = (const float4 *) (scl + half * 16 + 8 * (half ^ 1));
        const int8_t * xb = xrow + (PIN_B ? 0 : b * 128);
        // Group outer, K slices inner. The block's weights stay in `tr` across every group; only the
        // int32 accumulator pair and the activation fragment are live per group, which is what keeps
        // this at the occupancy the one-group shape has. Running K outer and groups inner instead
        // holds GROUPS accumulator pairs at once and costs 256 VGPR, a wave slot and a spill for the
        // 32 half swaps it would save out of a block of about 1500 cycles.
        #pragma unroll
        for (int g = 0; g < GROUPS; g++) {
            // Without this the scheduler hoists every group's activation fragments above the first
            // matrix instruction, 32 VGPRs a group, and the shape lands at 256 registers with a
            // spill. Fencing the groups keeps one group's operands in flight at a time.
            if constexpr (GROUPS > 1) if (g) __builtin_amdgcn_sched_barrier(0);
            // Seed both accumulators with the negated activation block sum instead of subtracting it
            // from every output element in the drain. The A operand carries unsigned trit codes, so
            // every output owes one -sum(x); the seed costs the same eight `v_mov` the zero cost and
            // removes sixteen integer subtracts per group per block. Integer addition is associative
            // and wraps identically, so the drained int32 is the same word. The wide head already
            // does this (docs/wide-head.md). Here it also takes `k_forward_rows<8>`, the drafted
            // verify pass, from 217 VGPR to 216.
            const int arow = g * 16 + (col & (COLS - 1));
            const int xsc = (PRE && g) ? pfc[PRE ? g - 1 : 0] : (Q8 ? 0 : xsum[arow * xsstride + b]);
            v8i C1 = {-xsc, -xsc, -xsc, -xsc, -xsc, -xsc, -xsc, -xsc}, C2 = C1;
            const int8_t * xbg = xb + g * (16 * 128 * xsstride);
            #pragma unroll
            for (int kb = 0; kb < 8; kb++) {
                const v4i A = { (int) tr[4 * kb], (int) tr[4 * kb + 1], (int) tr[4 * kb + 2], (int) tr[4 * kb + 3] };
                const v4i As = { swap16(A.x), swap16(A.y), swap16(A.z), swap16(A.w) };
                const int4 bx = (PRE && g && kb == 0) ? pf[PRE ? g - 1 : 0] : *(const int4 *) (xbg + kb * 16);
                const v4i B = { bx.x, bx.y, bx.z, bx.w };
                C1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(Q8, A, true, B, C1, false);
                C2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(Q8, As, true, B, C2, false);
            }
            // Read inside the group so the sixteen scale floats are not live across both groups;
            // the table is this wave's own LDS and four b128 loads a group are cheaper than the
            // wave slot they cost.
            const float4 s1a = s1p[0], s1b = s1p[1], s2a = s2p[0], s2b = s2p[1];
            const float s1[8] = { s1a.x, s1a.y, s1a.z, s1a.w, s1b.x, s1b.y, s1b.z, s1b.w }, s2[8] = { s2a.x, s2a.y, s2a.z, s2a.w, s2b.x, s2b.y, s2b.z, s2b.w };
            const float xsb = (PRE && g) ? pfs[PRE ? g - 1 : 0] : xs[arow * xsstride + b];
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                y1[g][r] = fmaf((float) C1[r], s1[r] * xsb, y1[g][r]);
                y2[g][r] = fmaf((float) C2[r], s2[r] * xsb, y2[g][r]);
            }
        }
        #pragma unroll
        for (int i = 0; i < (Q8 ? 8 : 1); i++) qa[i] = nqa[i];
        qb = nqb; tail = ntail;
    }
}

// Same unit decomposition as ph_matvec; every unit produces up to COLS * GROUPS activation columns.
// ARM selects what this unit does with the two things its block loop does not control: where the
// wave's streams live, and who sums the partials at the end. Arm 0 is the deployed body,
// instruction for instruction. Arm 1 is the drain spread over the waves that computed it. Arms
// 2..4 are FOOTPRINT ABLATIONS - numerically invalid by construction, compiled only under
// `-DHALO_MVW_ABL=1`, and never reachable in a build that produces tokens.
//
// The ablations exist because this phase has a 33% term that is neither its counted instructions
// nor its weight round trip (docs/ffn-slice-issue-order.md), and the one candidate nobody could
// price is the activation stream: a 128-row pass reads 68.4 GB of fragments against 15.0 GB of
// weights, and the traffic law says that number does not depend on the row count, which is why
// halving the weight stream was a null. `-DHALO_MVW_PIN=2` was the obvious instrument and it is
// confounded - pinning the cursor constant-folds an address and the block goes 476 instructions to
// 518. THIS ONE IS NOT: dropping `wb` from a base pointer computed once per unit leaves the block
// loop's instruction stream alone (same loads, same lines per load, same cursor arithmetic) and
// only collapses WHICH lines the wave touches - every wave of every workgroup reads the same
// `pw` blocks, so the stream is served out of L0 instead of L1/L2. The difference is that stream's
// delivery cost and nothing else.
enum MvwArm { MVW_DEPLOYED = 0, MVW_SPREAD = 1, MVW_ABL_ACT = 2, MVW_ABL_W = 3, MVW_ABL_BOTH = 4,
              MVW_ABL_NODRAIN = 5, MVW_DBUF = 6, MVW_FLAT = 7 };
// Arms 6 and 7 need two reduction buffers instead of one, and arm 7 gives each buffer an eighth
// slot. The array is declared inside each `k_ffn_slice` instantiation, so this costs the deployed
// arm no LDS byte and no workgroup of grid.
constexpr int mvw_red_buffers(int ARM) { return (ARM == MVW_DBUF || ARM == MVW_FLAT) ? 2 : 1; }
constexpr int mvw_red_slots(int ARM) { return ARM == MVW_FLAT ? NW : NW - 1; }

// WHAT THIS LOOP COSTS, READ BY THE KERNEL ITSELF AND NOT CONVERTED THROUGH A CLOCK.
// Every phase model here turns a census into milliseconds through an assumed frequency, and this
// box delivers between 2.6 and 2.9 GHz depending on what else holds the socket
// (docs/power-budget.md). `-DHALO_SLICE_CYCLES=1` reads `SHADER_CYCLES` around the block loop and
// around the whole unit and prints, for one wave of one workgroup, cycles per block, cycles per
// matrix instruction and the unit's tail. It is a diagnostic build: the reads change no value but
// they do perturb scheduling, so it is never selected and never a published time.
// docs/ffn-matrix-rate.md carries what it measured.
#ifndef HALO_SLICE_CYCLES
#define HALO_SLICE_CYCLES 0
#endif
#if HALO_SLICE_CYCLES
__device__ __forceinline__ unsigned halo_shader_cycles() { return __builtin_amdgcn_s_getreg((29) | (19 << 11)); }
__device__ int halo_slice_prints = 0;
#endif

template <int NB, int KS, bool Q8, int COLS = 8, int GROUPS = 1, int ORD = 0, int ARM = 0, bool ORDERED = false>
__device__ __forceinline__ void ph_matvec_w(Ctx & c, const MvR & m, const int8_t * xq, const float * xs, const int * xsum, int nrows) {
    using G = Geom<NB, KS>;
    constexpr int BB = Q8 ? Q8_TILE_BLOCK_BYTES : TILE_BLOCK_BYTES;
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, half = lane >> 4, col = lane & 15;
    const int total = m.total_tiles * (ORDERED ? 1 : KS);
    constexpr int RBUF = mvw_red_buffers(ARM), RSTRIDE = mvw_red_slots(ARM) * 8 * 32;
    float * red = c.lds;                 // [RBUF][7 waves][8][32]
    float * scl = c.lds + RBUF * RSTRIDE + wave * 32; // per-wave scale table
#if HALO_SLICE_CYCLES
    unsigned long long cyc_blk = 0, cyc_unit = 0; int nblk = 0, nunit = 0; unsigned pw_seen = 0;
#endif
    for (int u = blockIdx.x; u >= 0 && u < total; u = next_unit(c, total)) {
        c.dirty = true;
#if HALO_SLICE_CYCLES
        const unsigned cyc_u0 = halo_shader_cycles();
#endif
        for (int ordered_part = 0; ordered_part < (ORDERED ? KS : 1); ordered_part++) {
        const int part = ORDERED ? ordered_part : u / m.total_tiles;
        int tile = ORDERED ? u : u - part * m.total_tiles;
        MvSegR seg = m.seg[0];
        if (m.nseg > 1 && tile >= seg.ntiles) { tile -= seg.ntiles; seg = m.seg[1]; if (m.nseg > 2 && tile >= seg.ntiles) { tile -= seg.ntiles; seg = m.seg[2]; } }
        const int pw = (part == 0 || !G::SPECIAL) ? G::PW_A : G::PW_B;
        const int wb = (part == 0 ? 0 : (G::SPECIAL ? G::A_BLOCKS : part * G::PART)) + wave * pw;
        // The ablations take the wave's own block offset out of one base pointer or the other. The
        // arithmetic below and the whole block loop are unchanged; only the address it starts from
        // moves, so the eight waves of every workgroup collapse onto one wave's worth of lines.
        constexpr int WB_W = (ARM == MVW_ABL_W || ARM == MVW_ABL_BOTH) ? 0 : 1;
        constexpr int WB_X = (ARM == MVW_ABL_ACT || ARM == MVW_ABL_BOTH) ? 0 : 1;
        const uint8_t * run = seg.w + ((size_t) tile * NB + wb * WB_W) * BB;
        const int8_t * xrow = xq + (size_t) (col & (COLS - 1)) * NB * 128 + (wb * WB_X) * 128;
        float y1[GROUPS][8], y2[GROUPS][8];
        #pragma unroll
        for (int g = 0; g < GROUPS; g++)
            #pragma unroll
            for (int r = 0; r < 8; r++) { y1[g][r] = 0.0f; y2[g][r] = 0.0f; }
#if HALO_SLICE_CYCLES
        const unsigned cyc_b0 = halo_shader_cycles();
#endif
        if (part == 0 || !G::SPECIAL) mvw_rows<G::PW_A, Q8, COLS, GROUPS, ORD>(run, lane, xrow, 0, xs + wb, xsum + wb, NB, scl, y1, y2);
        else                          mvw_rows<G::PW_B, Q8, COLS, GROUPS, ORD>(run, lane, xrow, 0, xs + wb, xsum + wb, NB, scl, y1, y2);
#if HALO_SLICE_CYCLES
        cyc_blk += (halo_shader_cycles() - cyc_b0) & 0xfffffu; nblk += pw; pw_seen = pw;
#endif
        const int N = seg.ntiles * 32;
        // One reduction buffer, 2 * GROUPS sequential passes through it.
        //
        // THE DEPLOYED DRAIN PUTS FOUR SERIAL PASSES ON ONE WAVE. Each pass has the other seven
        // waves publish eight floats and wave 0 read 56 of them, add 56 and write 8 outputs, between
        // two barriers - and a `gate/up` unit is only five blocks of work per wave, so a 128-row
        // pass runs that sequence 360,000 times. `86551bc8` measured the same shape in `ph_matvec`
        // at 4.4% of a 1088-unit phase and shipped the fix an hour before this; that body has ONE
        // pass per unit and this one has 2 * GROUPS.
        //
        // Slot `r` belongs to wave `r` now. Eight partials fit seven slots because a slot's owner
        // reads its own partial from a register and wave 0 writes into the slot that frees: slot
        // `r-1` carries wave 0's partial for `r`, and wave `w > 0` skips the one `r` it owns. Every
        // (slot, r) still has exactly one writer, and the sum stays `y_0[r] + y_1[r] + ... + y_7[r]`
        // left to right with the owner splicing its register in at position `wave`, so no output bit
        // moves. `NW` is 8 and there are 8 slots, so every wave owns exactly one.
        // ARM 6 GIVES THE BUFFER A SECOND COPY AND HALVES THE BARRIERS. The deployed loop needs two
        // barriers a pass because pass p+1 writes the same floats pass p is reading, so the second
        // one is a write-after-read guard and nothing else. With two buffers pass p+1 publishes into
        // the copy pass p is not reading, so its publish rides in the same barrier interval as pass
        // p's reduce and only the read-after-write barrier is left: 2 * GROUPS barriers a unit
        // instead of 4 * GROUPS. Pass p and pass p+2 share a buffer and still have a barrier between
        // them. It is LDS, not registers, so the block loop's allocation cannot move - which is what
        // separates it from arm 1, where two extra VGPRs bought 54 `s_delay_alu` in the block loop
        // and cost 15% of the phase.
        // ARM 7 IS THE ONE THAT SHIPS, AND IT IS BOTH HALVES AT ONCE WITHOUT A REGISTER TO ITS NAME.
        // Arm 1 owns eight slots between seven by having a row's owner splice its own partial in
        // from a register, which saves an LDS slot and costs two VGPRs - and two VGPRs in this body
        // bought 54 `s_delay_alu` inside the block loop and 15% of the phase. Arm 7 gives the buffer
        // its eighth slot instead: every wave publishes all eight of its partials and reads all
        // eight back, so `y` dies at the publish, no wave is special, and the reduce is eight ways
        // parallel. The sum is still `y_0[r] + y_1[r] + ... + y_7[r]` left to right and the drained
        // float is the same word. It carries arm 6's second buffer with it, so a unit spends
        // `2 * GROUPS + 1` barriers instead of `4 * GROUPS`.
        constexpr bool SPREAD = ARM == MVW_SPREAD, FLAT = ARM == MVW_FLAT;
        constexpr bool DBUF = ARM == MVW_DBUF || FLAT;
        if constexpr (ARM == MVW_FLAT) {
            #pragma unroll
            for (int pass = 0; pass < 2 * GROUPS; pass++) {
                const int g = pass >> 1;
                float * y = (pass & 1) ? y2[g] : y1[g];
                float * rb = red + (pass & 1) * RSTRIDE;
                #pragma unroll
                for (int r = 0; r < 8; r++) rb[(wave * 8 + r) * 32 + lane] = y[r];
                __syncthreads();
                if (g * 16 + col < nrows) {
                    float v = rb[wave * 32 + lane];
                    #pragma unroll
                    for (int w = 1; w < NW; w++) v += rb[(w * 8 + wave) * 32 + lane];
                    const int row = (pass & 1) ? (half ? 2 * wave + 1 : 2 * wave + 16) : (half ? 2 * wave + 17 : 2 * wave);
                    float * o = seg.out + (size_t) (g * 16 + col) * N + (size_t) tile * 32 + row;
                    if constexpr (KS > 1 && !ORDERED) atomicAdd(o, v);
                    else if (KS > 1 || seg.add) *o += v;
                    else *o = v;
                }
            }
            __syncthreads();
        } else if constexpr (ARM != MVW_ABL_NODRAIN) {
        #pragma unroll
        for (int pass = 0; pass < 2 * GROUPS; pass++) {
            const int g = pass >> 1;
            float * y = (pass & 1) ? y2[g] : y1[g];
            float * rb = red + (DBUF ? (pass & 1) * RSTRIDE : 0);
            if (wave > 0) {
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    if (!SPREAD || r != wave) rb[((wave - 1) * 8 + r) * 32 + lane] = y[r];
            } else if (SPREAD) {
                #pragma unroll
                for (int r = 1; r < 8; r++) rb[((r - 1) * 8 + r) * 32 + lane] = y[r];
            }
            __syncthreads();
            if ((SPREAD || wave == 0) && g * 16 + col < nrows) {
                #pragma unroll
                for (int r = 0; r < 8; r++) if (!SPREAD || r == wave) {
                    // wave 0's partial: its own register at r == 0, the owner's freed slot above it
                    float v = (!SPREAD || r == 0) ? y[r] : rb[((r - 1) * 8 + r) * 32 + lane];
                    #pragma unroll
                    for (int w = 1; w < NW; w++)
                        v += (SPREAD && w == r) ? y[r] : rb[((w - 1) * 8 + r) * 32 + lane];
                    const int row = (pass & 1) ? (half ? 2 * r + 1 : 2 * r + 16) : (half ? 2 * r + 17 : 2 * r);
                    float * o = seg.out + (size_t) (g * 16 + col) * N + (size_t) tile * 32 + row;
                    if constexpr (KS > 1 && !ORDERED) atomicAdd(o, v);
                    else if (KS > 1 || seg.add) *o += v;
                    else *o = v;
                }
            }
            if (!DBUF) __syncthreads();
        }
        if (DBUF) __syncthreads();
        } else {
            // The epilogue ablation: no publish, no reduce, no barrier, no store, and the
            // accumulators kept alive so the block loop is not dead code. What it removes is the
            // whole unit boundary, which is the last term of this phase that is neither its bytes
            // nor its counted block instructions. Numerically invalid by construction.
            float s = 0;
            #pragma unroll
            for (int g = 0; g < GROUPS; g++)
                #pragma unroll
                for (int r = 0; r < 8; r++) s += y1[g][r] + y2[g][r];
            if (s == 12345.678f && wave == 0 && lane == 0) seg.out[tile * 32] = s;
        }
        }
#if HALO_SLICE_CYCLES
        cyc_unit += (halo_shader_cycles() - cyc_u0) & 0xfffffu; nunit++;
#endif
    }
#if HALO_SLICE_CYCLES
    // One wave of one workgroup over every unit it took. `tail` is the publish, the reduce and
    // their barriers; `next_unit` runs between units and is in neither figure.
    if (blockIdx.x == 0 && threadIdx.x == 0 && nblk > 0 && atomicAdd(&halo_slice_prints, 1) < 12)
        printf("slice-cycles arm=%d nb=%d groups=%d pw=%u units=%d blocks=%d block_cyc=%.0f unit_cyc=%.0f per_block=%.1f per_wmma=%.2f tail=%.1f\n",
               ARM, NB, GROUPS, pw_seen, nunit, nblk, (double) cyc_blk, (double) cyc_unit,
               (double) cyc_blk / nblk, (double) cyc_blk / ((double) nblk * 16.0 * GROUPS),
               (double) (cyc_unit - cyc_blk) / nunit);
#endif
    end_phase(c);
}

// The eight-row matrix body, reached only when this build asks for it. It needs `Geom`, `MvR`,
// `Ctx` and the unit deal above, so it is included here rather than at the top of the file.
#ifndef HALO_MV_WMMA8
#define HALO_MV_WMMA8 0
#endif
#if HALO_MV_WMMA8
#include "mv_wmma8.hpp"
#endif

// Dispatch: the scalar-fed dot4 kernel up to this representation's row limit, then the WMMA
// kernel, which fills all sixteen matrix columns once the caller has sixteen rows to give it and
// takes a second column group above that rather than a second pass over the weights.
//
// `HALO_MV_WMMA8=1` sends the eight-row ternary shape to `kernels/mv_wmma8.hpp` instead of the
// dot4 body: same image, same blocks, same K order, same per-block `int32`, same `fmaf` fold, and
// the register live set of a block cut to what the deployed grid can afford.
// docs/mv-wmma-eight.md. The default build is the dot4 body, instruction for instruction.
template <int NB, int KS, int TT, bool Q8, int SM = 0, int ORD = 0, int ARM = 0, bool ORDERED = false>
__device__ __forceinline__ void ph_matvec_auto(Ctx & c, const MvR & m, const int8_t * xq, const float * xs, const int * xsum, int nrows) {
    constexpr int DOT4_MAX = Q8 ? MV_DOT4_MAX_Q8 : MV_DOT4_MAX;
    constexpr int MV_COLS = TT > 8 ? 16 : 8;
    constexpr int MV_GROUPS = TT > 16 ? (TT + 15) / 16 : 1;
    if constexpr (TT <= DOT4_MAX) ph_matvec<NB, KS, TT, Q8, SM, ORDERED>(c, m, xq, xs, xsum, nrows);
    else ph_matvec_w<NB, KS, Q8, MV_COLS, MV_GROUPS, ORD, ARM, ORDERED>(c, m, xq, xs, xsum, nrows);
}

// The same dispatch for the persistent kernel's own matvec phases. It exists so that the eight-row
// matrix body reaches `k_forward_rows` and nothing else: `k_ffn_slice` has a five-to-eight-row
// width of its own (`slice_wi`), and a body selected inside `ph_matvec_auto` would change that
// kernel's registers and grid as a side effect of a persistent-route experiment.
template <int NB, int KS, int TT, bool Q8, int SM = 0, int ORD = 0, int ARM = 0, bool ORDERED = true>
__device__ __forceinline__ void ph_matvec_pk(Ctx & c, const MvR & m, const int8_t * xq, const float * xs, const int * xsum, int nrows) {
#if HALO_MV_WMMA8
    if constexpr ((!ORDERED || KS == 1) && !Q8 && TT > 4 && TT <= 8 && SM < 3) { ph_matvec_m<NB, KS, TT>(c, m, xq, xs, xsum, nrows); return; }
#endif
    ph_matvec_auto<NB, KS, TT, Q8, SM, ORD, ARM, ORDERED>(c, m, xq, xs, xsum, nrows);
}

// ---------------------------------------------------------------------------------------------
// Attention. attn_pre: per (row, head): q/k norm + RoPE, K/V append. attn: per (seq, kv head, chunk).

__device__ __forceinline__ void rope_pair(float pos, int d, float & lo, float & hi) {
    const float theta = pos * __powf(1e7f, -(float) d / (float) (NROT / 2));
    float sn, cs; __sincosf(theta, &sn, &cs);
    const float a = lo, b = hi;
    lo = a * cs - b * sn; hi = a * sn + b * cs;
}

// WIDE reads the pass's whole row array instead of the eight-row slice inside FwdParams, and gives
// every workgroup one unit from its block index. The unit body does not change, so a row's rope,
// its cache write and its scaled q are the same bits from either launch shape.
template <bool WIDE = false>
__device__ __forceinline__ void ph_attn_pre(Ctx & c, const FwdParams & P, const float * q_norm, const float * k_norm, int kv_slot,
                                            const RowInfo * rw = nullptr, int nrows_w = 0) {
    const int total = (WIDE ? nrows_w : P.nrows) * (NH + NKV);
    const int d = threadIdx.x;
    for (int u = blockIdx.x; u >= 0 && u < total; u = WIDE ? -1 : next_unit(c, total)) {
        c.dirty = true;
        const int row = u / (NH + NKV), hh = u - row * (NH + NKV);
        RowInfo R;
        if constexpr (WIDE) R = rw[row]; else R = P.rows[row];
        const bool is_q = hh < NH;
        float v = is_q ? P.qfull[(size_t) row * Q_OUT + hh * 2 * HD + d] : P.kbuf[(size_t) row * KV_OUT + (hh - NH) * HD + d];
        const float ss = block_sum_256(v * v, c.red);
        v = v * rsqrtf(ss / (float) HD + NORM_EPS) * (is_q ? q_norm[d] : k_norm[d]);
        c.lds[d] = v;
        __syncthreads();
        float outv = v;
        if (d < NROT) {
            const int i = d < NROT / 2 ? d : d - NROT / 2;
            float lo = c.lds[i], hi = c.lds[i + NROT / 2];
            rope_pair((float) R.pos, i, lo, hi);
            outv = d < NROT / 2 ? lo : hi;
        }
        if (is_q) {
            P.qrot[(size_t) row * ATTN_OUT + hh * HD + d] = outv * 0.0625f;
        } else {
            const int hk = hh - NH;
            __half * kc = P.kcache + (size_t) (R.seq * KV_SLOTS + kv_slot) * ((size_t) NKV * P.context * HD) + (size_t) hk * P.context * HD;
            __half * vc = P.vcache + (size_t) (R.seq * KV_SLOTS + kv_slot) * ((size_t) NKV * P.context * HD) + (size_t) hk * P.context * HD;
            kc[P.k_tiled ? k_leaf_off(R.pos, d, HD) : (size_t) R.pos * HD + d] = __float2half(outv);
            vc[(size_t) R.pos * HD + d] = __float2half(P.vbuf[(size_t) row * KV_OUT + hk * HD + d]);
        }
        __syncthreads();
    }
    end_phase(c);
}

// unit = (seq, q head, chunk); scores[TT][ACHUNK] in LDS. K/V chunks are shared by the heads of a
// KV group through L2. Geometry: NHq query heads, NKVh KV heads, HDh head dim (256 or 128).
// qrot rows are [NHq * HDh] pre-scaled; the cache slot layout is [head][MAXCTX][HDh].
// One (row group, q head, key chunk) unit. A group is TT or fewer consecutive rows of one
// sequence, which is exactly what an eight-row slice handed this phase; a wide launch hands it the
// same groups from one grid. PB/pstride/prow0 place the group's partials, and the deployed path
// passes the pass's own array with its compile-time stride and no row offset.
// One lane owns one key and the cross-lane reduction is gone. This is the deployed body with its
// key loop replaced and everything below it - the softmax, the value loop, the partial - left
// exactly as it was; docs/attn-score-lane.md owns the coordinate, the relabeled tree and the
// panel. It is a separate function rather than an arm of the two above because a third body
// inlined into k_attn_wide costs the whole kernel 43 VGPRs and six wave slots, and because the
// rescheduled body's medicine (breadth-first DPP trees) is for a reduction this one does not have.
//
// Which kernels compile it is `ATTN_KM_LEAF`, decided by the launcher. It used to be reached
// through a runtime branch on `P.k_tiled` from inside `attn_chunk_unit`, which made every caller -
// including `k_forward_rows`, where this phase is inlined - allocate the union of this body and
// the row-major one. That cost the persistent kernel 206 VGPRs against 135 and three of its five
// workgroups per WGP.
//
// HG folds a KV group's query heads into this unit, and it is the axis this body can afford that
// the row-major one cannot. There the query is per lane - qr[TT][DPL], 64 VGPRs at eight rows -
// so a second head is 64 more registers and HG was restricted to one-row groups. Here the query is
// WAVE-UNIFORM and arrives on the scalar unit, so a folded head costs one score register per row
// plus its share of the tree. A key chunk's bytes are then charged to HG times as many dot
// products, and the value loop shares one `vc` load across the fold because V is per KV head.
// HB is how many of those heads one leaf walk carries: HB = HG reads each leaf once and holds
// HB * RB tree partials, HB = 1 walks the leaves once per head and holds RB, re-reading a leaf the
// same wave read microseconds earlier. docs/attn-reuse-fold.md prices both.
//
// PROBE is an ablation, not an arm: bit 1 pins the key chunk's offset and bit 2 the value chunk's,
// with an opaque device zero, so the unit issues THE SAME instructions against a working set every
// wave on the device already has resident. It answers "how much of this phase is the memory system"
// with no model of the memory system, and its output is wrong by construction. Only a
// -DHALO_ATTN_PROBE build compiles it; the default build has no instantiation with PROBE != 0.
template <int TT, int NHq, int NKVh, int HDh, bool QINV = true, int HG = 1, int HB = 1, int PROBE = 0>
__device__ __forceinline__ void attn_chunk_unit_leaf(Ctx & c, const FwdParams & P, const RowInfo * RB,
        const SeqCtl S, int hq0, int chunk, int kv_slot, AttnPartial * PB, int pstride, int prow0) {
    constexpr int GQA = NHq / NKVh;
    static_assert(HG >= 1 && HB >= 1 && HG % HB == 0, "a leaf walk carries a whole share of the fold");
    static_assert(GQA % HG == 0, "a unit's query heads must be a whole share of one KV group");
    static_assert(HG * TT * ACHUNK <= LDS_FLOATS, "score array does not fit the shared block");
    float * sc = c.lds; // [HG][TT][ACHUNK]
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    {
        const int hk = hq0 / GQA;
        const int slot = RB[S.row0].seq;
        const int start = chunk * ACHUNK;
        const int pos_last = RB[S.row0 + S.nrows - 1].pos;
        const int count = min(ACHUNK, pos_last + 1 - start);
        // grid.z is 1 in every launch that reaches this body, so this is zero on the device and the
        // compiler cannot prove it. Keeping the multiply means the address arithmetic, the request
        // count and the instruction sequence are the ones the deployed arm issues.
        const int pin = PROBE ? (int) __builtin_amdgcn_readfirstlane(blockIdx.z) : 1;
        const int kstart = (PROBE & 1) ? start * pin : start;
        const int vstart = (PROBE & 2) ? start * pin : start;
        const __half * kc = P.kcache + (size_t) (slot * KV_SLOTS + kv_slot) * ((size_t) NKV * P.context * HD) + ((size_t) hk * P.context + kstart) * HDh;
        const __half * vc = P.vcache + (size_t) (slot * KV_SLOTS + kv_slot) * ((size_t) NKV * P.context * HD) + ((size_t) hk * P.context + vstart) * HDh;
        {
            {
                // One lane owns one key and sums its whole dot product in its own register. The
                // wave owns a 32-key tile and half of the 32 leaves, so two waves meet at the
                // tree's root through `sc` - which is exactly where permlanex16 joins the two rows
                // of sixteen lanes. Same products, same eight-deep fmaf chains from 0.0f, same
                // combination order including the {0,3,2,1} quad relabeling `leaf_of_pos` carries,
                // so the score is the same float. What it does not do is spend 47 slots of DPP and
                // a five-deep dependent chain per key to move partial sums between lanes.
                constexpr int NLEAF = HDh / KLEAF, WLEAF = NLEAF / 2;
                // The wave index is wave-uniform and the compiler cannot know that from
                // threadIdx.x. Naming it here is what puts the query on the scalar unit: without
                // it every q value arrives as a vector load, 256 of them per chunk against 16 key
                // loads, and the wave carries their addresses and results in VGPRs.
                const int wu = __builtin_amdgcn_readfirstlane(wave);
                const int tile = wu >> 1, half2 = wu & 1, p = tile * KTILE + lane;
                const bool live = tile * KTILE < count;
                float hs[HG * TT];
                if (live) {
                    const __half * kb = kc + (size_t) tile * (KTILE * HDh)
                                      + (size_t) (half2 * WLEAF) * (KTILE * KLEAF) + lane * KLEAF;
                    typedef const float * __restrict__ qptr_inv;
                    typedef const float * qptr_any;
                    using QP = typename std::conditional<QINV, qptr_inv, qptr_any>::type;
                    QP qrow[HG * TT];
                    #pragma unroll
                    for (int g = 0; g < HG; g++)
                    #pragma unroll
                    for (int i = 0; i < TT; i++)
                        qrow[g * TT + i] = P.qrot + (size_t) (S.row0 + min(i, S.nrows - 1)) * NHq * HDh
                                + (hq0 + g) * HDh + half2 * (WLEAF * KLEAF);
                    // The tree holds one live float per output per level, so wide groups walk the
                    // leaves once per block of eight rows rather than spilling. At TT <= 8 this is
                    // one pass and the key loads are read exactly once.
                    constexpr int RB = TT < 8 ? TT : 8;
                    constexpr int NR = HB * RB;
                    #pragma unroll
                    for (int hb = 0; hb < HG / HB; hb++)
                    #pragma unroll
                    for (int rb = 0; rb < TT / RB; rb++) {
                        // One leaf: sixteen bytes of this lane's own key, and the query values
                        // every lane of the wave shares, which is what lets them reach the fma as
                        // scalars. HB heads read that one leaf.
                        auto leaf = [&](auto J, float * out) {
                            constexpr int jj = decltype(J)::value;
                            const uint4 raw = *(const uint4 *) (kb + (size_t) jj * (KTILE * KLEAF));
                            const __half2 * h2 = (const __half2 *) &raw;
                            float kf[KLEAF];
                            #pragma unroll
                            for (int e = 0; e < KLEAF / 2; e++) { float2 f = __half22float2(h2[e]); kf[2 * e] = f.x; kf[2 * e + 1] = f.y; }
                            #pragma unroll
                            for (int gb = 0; gb < HB; gb++)
                            #pragma unroll
                            for (int i = 0; i < RB; i++) {
                                const QP qp = qrow[(hb * HB + gb) * TT + rb * RB + i] + jj * KLEAF;
                                float s = 0.0f;
                                #pragma unroll
                                for (int e = 0; e < KLEAF; e++) s = fmaf(qp[e], kf[e], s);
                                out[gb * RB + i] = s;
                            }
                        };
                        float part[NR];
                        LeafTree<WLEAF, NR>::run(leaf, part);
                        #pragma unroll
                        for (int gb = 0; gb < HB; gb++)
                        #pragma unroll
                        for (int i = 0; i < RB; i++) hs[(hb * HB + gb) * TT + rb * RB + i] = part[gb * RB + i];
                        // Row blocks are independent, so the scheduler would happily interleave
                        // them and hold two trees live at once. One pass at a time is the point.
                        if constexpr ((HG / HB) * (TT / RB) > 1) __builtin_amdgcn_sched_barrier(0);
                    }
                    if (half2 && p < count) {
                        #pragma unroll
                        for (int k = 0; k < HG * TT; k++) sc[k * ACHUNK + p] = hs[k];
                    }
                }
                __syncthreads();
                if (live && !half2 && p < count) {
                    #pragma unroll
                    for (int k = 0; k < HG * TT; k++) sc[k * ACHUNK + p] = hs[k] + sc[k * ACHUNK + p];
                }
            }
            __syncthreads();
            // A fold's m and l go to their partial as soon as the block reduction has them. Holding
            // HG * TT of each across the value loop is 2 * HG * TT registers for two numbers a
            // single lane stores, and at HG = 2 that alone is a wave granule. The unfolded arm
            // keeps the arrays so its codegen is the one the published register table was read on.
            constexpr bool EARLY_ML = HG > 1;
            constexpr int MLN = EARLY_ML ? 1 : HG * TT;
            float mx[MLN], lsum[MLN];
            #pragma unroll
            for (int k = 0; k < HG * TT; k++) {
                const int i = k % TT;
                const int qpos = RB[S.row0 + min(i, S.nrows - 1)].pos;
                int lo = max(0, P.attn_key_min - start);
                if (P.attn_window) lo = max(lo, qpos - P.attn_window + 1 - start);
                const int vis = P.attn_noncausal ? count : min(count, qpos + 1 - start);
                float m = -INFINITY;
                for (int p = tid + lo; p < vis; p += NT) m = fmaxf(m, sc[k * ACHUNK + p]);
                m = block_max_256(m, c.red);
                float l = 0.0f;
                for (int p = tid; p < count; p += NT) { float e = (p < vis && p >= lo) ? __expf(sc[k * ACHUNK + p] - m) : 0.0f; sc[k * ACHUNK + p] = e; l += e; }
                l = block_sum_256(l, c.red);
                if constexpr (EARLY_ML) {
                    if (tid == 0 && i < S.nrows) {
                        AttnPartial * pml = PB + (((size_t) (S.row0 + i - prow0)) * NHq + (hq0 + k / TT)) * pstride + chunk;
                        pml->m = m; pml->l = l;
                    }
                } else { mx[k] = m; lsum[k] = l; }
            }
            __syncthreads();
            float acc[HG * TT];
            #pragma unroll
            for (int k = 0; k < HG * TT; k++) acc[k] = 0.0f;
            if (tid < HDh) {
                // One value element, HG * TT accumulators: the fold's heads share a KV head, so
                // this loop reads the chunk's values once whatever HG is.
                #pragma unroll 1
                for (int p = 0; p < count; p++) {
                    const float vv = __half2float(vc[(size_t) p * HDh + tid]);
                    #pragma unroll
                    for (int k = 0; k < HG * TT; k++) acc[k] = fmaf(sc[k * ACHUNK + p], vv, acc[k]);
                }
            }
            #pragma unroll
            for (int k = 0; k < HG * TT; k++) {
                const int g = k / TT, i = k % TT;
                if (i < S.nrows) {
                    AttnPartial * part = PB + (((size_t) (S.row0 + i - prow0)) * NHq + (hq0 + g)) * pstride + chunk;
                    if (tid < HDh) part->acc[tid] = acc[k];
                    if constexpr (!EARLY_ML) if (tid == 0) { part->m = mx[k]; part->l = lsum[k]; }
                }
            }
            __syncthreads();
        }
    }
}


// SCHED 0 is the deployed body kept verbatim as the exactness control; SCHED 1 is the rescheduled
// one in `attn_chunk_unit_fast` below. Both compute the same floats - see `attn_score.hpp` for why
// the reduction tree is a fixed object rather than a property of the lane layout.
// HG folds a KV group's query heads into one unit. A key chunk costs the same bytes whoever reads
// it; HG and TT are the two axes that decide how many dot products are charged to them, and the two
// shapes this engine runs have one each. A prompt pass amortises a chunk over eight neighbouring
// rows. A generation step cannot - its rows are different sequences reading different caches - and
// it has six query heads per KV group instead. Nothing crosses the head axis: separate query,
// separate scores, separate softmax, separate partial, so this is a relabeling of which workgroup
// owns which (row, head) and HG = 1 compiles to the code that ran before the parameter existed.
//
// The registers make them alternatives rather than two knobs: a unit holds HG * TT * DPL query
// registers live across the key loop, so eight rows by six heads is 384 VGPRs and cannot be built
// while one row by six heads is 48, under the 64 qr[8][8] costs today.
template <int TT, int NHq, int NKVh, int HDh, int HG = 1>
__device__ __forceinline__ void attn_chunk_unit_ref(Ctx & c, const FwdParams & P, const RowInfo * RB,
        const SeqCtl S, int hq0, int chunk, int kv_slot, AttnPartial * PB, int pstride, int prow0) {
    constexpr int GQA = NHq / NKVh, DPL = HDh / 32;
    static_assert(GQA % HG == 0, "a unit's query heads must be a whole share of one KV group");
    static_assert(HG * TT * ACHUNK <= LDS_FLOATS, "score array does not fit the shared block");
    float * sc = c.lds; // [HG][TT][ACHUNK]
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    {
        const int hq = hq0, hk = hq / GQA;
        const int slot = RB[S.row0].seq;
        const int start = chunk * ACHUNK;
        const int pos_last = RB[S.row0 + S.nrows - 1].pos;
        const int count = min(ACHUNK, pos_last + 1 - start);
        // The head's base, without the chunk: `k_pos_off` places the key, so this body reads a
        // leaf-interleaved cache as readily as a row-major one. Same sixteen bytes of the same
        // key's leaf, same expansion, same fma chain - only the address moves. Only the target
        // geometry's cache is ever interleaved, so the drafter's HDh folds this to a constant.
        const bool ktil = (HDh == HD) && P.k_tiled;
        const __half * kh = P.kcache + (size_t) (slot * KV_SLOTS + kv_slot) * ((size_t) NKV * P.context * HD) + (size_t) hk * P.context * HDh;
        const __half * vc = P.vcache + (size_t) (slot * KV_SLOTS + kv_slot) * ((size_t) NKV * P.context * HD) + ((size_t) hk * P.context + start) * HDh;
        {
            float qr[HG][TT][DPL];
            #pragma unroll
            for (int g = 0; g < HG; g++)
            #pragma unroll
            for (int i = 0; i < TT; i++) {
                const int row = S.row0 + min(i, S.nrows - 1);
                const float * qp = P.qrot + (size_t) row * NHq * HDh + (hq + g) * HDh + lane * DPL;
                #pragma unroll
                for (int e = 0; e < DPL; e += 4) { const float4 a = *(const float4 *) (qp + e); qr[g][i][e] = a.x; qr[g][i][e + 1] = a.y; qr[g][i][e + 2] = a.z; qr[g][i][e + 3] = a.w; }
            }
            #pragma unroll 1
            for (int p = wave; p < count; p += NW) {
                float kf[DPL];
                const __half * kp = kh + k_pos_off(ktil, start + p, lane * DPL, HDh);
                if (DPL == 8) {
                    const uint4 raw = *(const uint4 *) kp;
                    const __half2 * h2 = (const __half2 *) &raw;
                    #pragma unroll
                    for (int e = 0; e < 4; e++) { float2 f = __half22float2(h2[e]); kf[2 * e] = f.x; kf[2 * e + 1] = f.y; }
                } else {
                    const uint2 raw = *(const uint2 *) kp;
                    const __half2 * h2 = (const __half2 *) &raw;
                    #pragma unroll
                    for (int e = 0; e < 2; e++) { float2 f = __half22float2(h2[e]); kf[2 * e] = f.x; kf[2 * e + 1] = f.y; }
                }
                #pragma unroll
                for (int g = 0; g < HG; g++)
                #pragma unroll
                for (int i = 0; i < TT; i++) {
                    float dd = 0.0f;
                    #pragma unroll
                    for (int e = 0; e < DPL; e++) dd = fmaf(qr[g][i][e], kf[e], dd);
                    dd = warp_sum(dd);
                    if (lane == 0) sc[(g * TT + i) * ACHUNK + p] = dd;
                }
            }
            __syncthreads();
            float mx[HG][TT], lsum[HG][TT];
            #pragma unroll
            for (int g = 0; g < HG; g++)
            #pragma unroll
            for (int i = 0; i < TT; i++) {
                float * scg = sc + (g * TT + i) * ACHUNK;
                const int qpos = RB[S.row0 + min(i, S.nrows - 1)].pos;
                int lo = max(0, P.attn_key_min - start);
                if (P.attn_window) lo = max(lo, qpos - P.attn_window + 1 - start);
                const int vis = P.attn_noncausal ? count : min(count, qpos + 1 - start);
                float m = -INFINITY;
                for (int p = tid + lo; p < vis; p += NT) m = fmaxf(m, scg[p]);
                m = block_max_256(m, c.red);
                float l = 0.0f;
                for (int p = tid; p < count; p += NT) { float e = (p < vis && p >= lo) ? __expf(scg[p] - m) : 0.0f; scg[p] = e; l += e; }
                l = block_sum_256(l, c.red);
                mx[g][i] = m; lsum[g][i] = l;
            }
            __syncthreads();
            float acc[HG][TT];
            #pragma unroll
            for (int g = 0; g < HG; g++)
            #pragma unroll
            for (int i = 0; i < TT; i++) acc[g][i] = 0.0f;
            if (tid < HDh) {
                #pragma unroll 1
                for (int p = 0; p < count; p++) {
                    const float vv = __half2float(vc[(size_t) p * HDh + tid]);
                    #pragma unroll
                    for (int g = 0; g < HG; g++)
                    #pragma unroll
                    for (int i = 0; i < TT; i++) acc[g][i] = fmaf(sc[(g * TT + i) * ACHUNK + p], vv, acc[g][i]);
                }
            }
            #pragma unroll
            for (int g = 0; g < HG; g++)
            #pragma unroll
            for (int i = 0; i < TT; i++) {
                if (i < S.nrows) {
                    AttnPartial * part = PB + (((size_t) (S.row0 + i - prow0)) * NHq + hq + g) * pstride + chunk;
                    if (tid < HDh) part->acc[tid] = acc[g][i];
                    if (tid == 0) { part->m = mx[g][i]; part->l = lsum[g][i]; }
                }
            }
            __syncthreads();
        }
    }
}

// Same unit, same floats, four schedule changes. Each one is exact by construction:
//
//  1. The key block for `p + NW` is issued right after the current block is expanded into
//     registers, so its round trip is covered by this key's dot products instead of a `vmcnt(0)`
//     at the top of the next iteration. Same loads, same order, same addresses.
//  2. The eight per-row trees are walked breadth first (`warp_sum_n`), so each dependent DPP step
//     has seven independent ones behind it. `s_delay_alu` was 29% of this block.
//  3. The softmax's sixteen block reductions become two batched ones: 32 barriers to 4. The
//     per-row tree is `block_sum_256`'s, wave order and all.
//  4. The value loop reads four keys' scores per row with one `ds_read_b128` instead of four
//     `ds_read_b32`, and issues the four value loads together. Accumulation stays in ascending
//     key order, which is what fixes the bits.
template <int TT, int NHq, int NKVh, int HDh>
__device__ __forceinline__ void attn_chunk_unit_fast(Ctx & c, const FwdParams & P, const RowInfo * RB,
        const SeqCtl S, int hq, int chunk, int kv_slot, AttnPartial * PB, int pstride, int prow0) {
    constexpr int GQA = NHq / NKVh, DPL = HDh / 32;
    static_assert(TT * ACHUNK + NW * TT <= LDS_FLOATS, "scores plus the batched reduction scratch must fit");
    float * sc = c.lds;                  // [TT][ACHUNK]
    float * rd = c.lds + TT * ACHUNK;    // [NW][TT], the batched block reduction's wave results
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int hk = hq / GQA;
    const int slot = RB[S.row0].seq;
    const int start = chunk * ACHUNK;
    const int pos_last = RB[S.row0 + S.nrows - 1].pos;
    const int count = min(ACHUNK, pos_last + 1 - start);
    // See `attn_chunk_unit_ref`: the coordinate is an address, so the block cursor below keeps its
    // schedule - the same load issued at the same point of the same loop - in either form.
    const bool ktil = (HDh == HD) && P.k_tiled;
    const __half * kh = P.kcache + (size_t) (slot * KV_SLOTS + kv_slot) * ((size_t) NKV * P.context * HD) + (size_t) hk * P.context * HDh;
    const __half * vc = P.vcache + (size_t) (slot * KV_SLOTS + kv_slot) * ((size_t) NKV * P.context * HD) + ((size_t) hk * P.context + start) * HDh;

    float qr[TT][DPL];
    #pragma unroll
    for (int i = 0; i < TT; i++) {
        const int row = S.row0 + min(i, S.nrows - 1);
        const float * qp = P.qrot + (size_t) row * NHq * HDh + hq * HDh + lane * DPL;
        #pragma unroll
        for (int e = 0; e < DPL; e += 4) { const float4 a = *(const float4 *) (qp + e); qr[i][e] = a.x; qr[i][e + 1] = a.y; qr[i][e + 2] = a.z; qr[i][e + 3] = a.w; }
    }

    uint4 raw;
    if (wave < count) raw = attn_kload<DPL>(kh + k_pos_off(ktil, start + wave, lane * DPL, HDh));
    #pragma unroll 1
    for (int p = wave; p < count; p += NW) {
        float kf[DPL];
        attn_kexpand<DPL>(raw, kf);
        const int pn = p + NW;
        if (pn < count) raw = attn_kload<DPL>(kh + k_pos_off(ktil, start + pn, lane * DPL, HDh));
        // ATTN_RED_W rows reduced together. The score loop is the kernel's VGPR peak (qr alone is
        // TT*DPL = 64 registers at the target geometry), and this unit is one of the few where
        // wave slots are the measured medicine, so the interleave width is a register decision
        // rather than a free one: see docs/attn-score-schedule.md for the census that set it.
        #pragma unroll
        for (int i0 = 0; i0 < TT; i0 += ATTN_RED_W) {
            float dd[ATTN_RED_W];
            #pragma unroll
            for (int i = 0; i < ATTN_RED_W; i++) {
                float d = 0.0f;
                #pragma unroll
                for (int e = 0; e < DPL; e++) d = fmaf(qr[i0 + i][e], kf[e], d);
                dd[i] = d;
            }
            warp_sum_n<ATTN_RED_W>(dd);
            if (lane == 0) {
                #pragma unroll
                for (int i = 0; i < ATTN_RED_W; i++) sc[(i0 + i) * ACHUNK + p] = dd[i];
            }
        }
    }
    __syncthreads();

    float mx[TT];
    #pragma unroll
    for (int i = 0; i < TT; i++) {
        const int qpos = RB[S.row0 + min(i, S.nrows - 1)].pos;
        int lo = max(0, P.attn_key_min - start);
        if (P.attn_window) lo = max(lo, qpos - P.attn_window + 1 - start);
        const int vis = P.attn_noncausal ? count : min(count, qpos + 1 - start);
        float m = -INFINITY;
        for (int p = tid + lo; p < vis; p += NT) m = fmaxf(m, sc[i * ACHUNK + p]);
        mx[i] = m;
    }
    block_max_n<TT, NW>(mx, rd);
    float lsum[TT];
    #pragma unroll
    for (int i = 0; i < TT; i++) {
        const int qpos = RB[S.row0 + min(i, S.nrows - 1)].pos;
        int lo = max(0, P.attn_key_min - start);
        if (P.attn_window) lo = max(lo, qpos - P.attn_window + 1 - start);
        const int vis = P.attn_noncausal ? count : min(count, qpos + 1 - start);
        float l = 0.0f;
        for (int p = tid; p < count; p += NT) { float e = (p < vis && p >= lo) ? __expf(sc[i * ACHUNK + p] - mx[i]) : 0.0f; sc[i * ACHUNK + p] = e; l += e; }
        lsum[i] = l;
    }
    block_sum_n<TT, NW>(lsum, rd);
    __syncthreads();

    float acc[TT];
    #pragma unroll
    for (int i = 0; i < TT; i++) acc[i] = 0.0f;
    if (tid < HDh) {
        // ATTN_VAL_W keys per block. One `ds_read_b128` replaces four `ds_read_b32` per row, and
        // the value loads issue together - but the compiler hoists every row's score read, so the
        // width is bought with registers in a kernel where wave slots are the medicine.
        int p = 0;
        #pragma unroll 1
        for (; p < count; p++) {
            const float vv = __half2float(vc[(size_t) p * HDh + tid]);
            #pragma unroll
            for (int i = 0; i < TT; i++) acc[i] = fmaf(sc[i * ACHUNK + p], vv, acc[i]);
        }
    }
    #pragma unroll
    for (int i = 0; i < TT; i++) {
        if (i < S.nrows) {
            AttnPartial * part = PB + (((size_t) (S.row0 + i - prow0)) * NHq + hq) * pstride + chunk;
            if (tid < HDh) part->acc[tid] = acc[i];
            if (tid == 0) { part->m = mx[i]; part->l = lsum[i]; }
        }
    }
    __syncthreads();
}

// 0: the deployed body, kept verbatim as the exactness and timing control.
// 1: rescheduled, one key per value block - occupancy-neutral at 93 VGPR against the control's 92.
// A four-key value block was arm 2 and is a measured null: -28% of the unit's issue slots against
// arm 0 and the same time as arm 1. See docs/attn-score-schedule.md - the value loop is where this
// unit now spends itself and it is not issue-bound, which is the next engineer's starting point.
// HG > 1 always takes the reference body. The rescheduled one has its own key cursor, its own
// batched block reduction and its own value block, none of which have been read against a folded
// head axis; the fold is worth 2.6x on the shape it serves and the schedule is worth a few percent,
// so the honest composition is a separate measurement rather than an untested instantiation.
//
// KM says which score body this instantiation compiles, and it is a LAUNCH decision, not a unit
// one. The leaf coordinate is a third body with no cross-lane reduction at all
// (docs/attn-score-lane.md), but reaching it through a runtime branch on `P.k_tiled` made every
// caller allocate the union of two full bodies: k_attn_wide<8> 92 VGPR / 16 waves became 130 / 10
// and k_forward_rows 135 / 10 became 206 / 7, which is three workgroups per WGP against five. So
// the branch moves up to the launcher, each kernel carries one body, and both go back to their own
// register counts.
//
// ATTN_KM_ROW is correct in BOTH coordinates - the row-major body places its key through
// `k_pos_off` - so a caller that does not want the lane-per-key algorithm does not have to compile
// it to stay correct when the cache is interleaved. That is what lets the persistent kernel hold
// one body without constraining the process setting. ATTN_KM_DYN reproduces the fused arm exactly
// and is kept as the in-process control for what the delivery costs; HG > 1 has no leaf body to
// choose between, for the same reason SCHED does not.
template <int TT, int NHq, int NKVh, int HDh, int SCHED = 0, int HG = 1, bool QINV = true, int KM = ATTN_KM_ROW, int HB = 1, int PROBE = 0>
__device__ __forceinline__ void attn_chunk_unit(Ctx & c, const FwdParams & P, const RowInfo * RB,
        const SeqCtl S, int hq, int chunk, int kv_slot, AttnPartial * PB, int pstride, int prow0) {
    if constexpr (KM != ATTN_KM_ROW && HDh == HD) {
        if constexpr (KM == ATTN_KM_LEAF) {
            attn_chunk_unit_leaf<TT, NHq, NKVh, HDh, QINV, HG, HB, PROBE>(c, P, RB, S, hq, chunk, kv_slot, PB, pstride, prow0);
            return;
        } else if constexpr (HG == 1) if (P.k_tiled) {   // ATTN_KM_DYN, the control: both bodies, one kernel
            attn_chunk_unit_leaf<TT, NHq, NKVh, HDh, QINV>(c, P, RB, S, hq, chunk, kv_slot, PB, pstride, prow0);
            return;
        }
    }
    if constexpr (SCHED == 0 || HG > 1)
        attn_chunk_unit_ref<TT, NHq, NKVh, HDh, HG>(c, P, RB, S, hq, chunk, kv_slot, PB, pstride, prow0);
    else
        attn_chunk_unit_fast<TT, NHq, NKVh, HDh>(c, P, RB, S, hq, chunk, kv_slot, PB, pstride, prow0);
}

// A group carries between one and TT rows, and the unit above is written for TT of them: it loads
// TT query rows, runs TT dot products per key, TT block reductions over the score array and TT
// fmacs per value, then keeps the first `S.nrows` of them. Prompt passes hand it full groups, so
// that cost is invisible in every prefill panel this lane has taken. Generation hands it one-row
// groups - a 32-stream step is slices of eight sequences of one row, single-stream decode and the
// drafted verify's context rows are the same shape - and there the unit does TT times the work it
// keeps, with the redundant q rows read from one address the compiler cannot fold because
// `min(i, nrows-1)` is a runtime clamp.
//
// Dispatching on the group's own row count is exact rather than approximate. The narrow
// instantiation is the same source with the row loop taken fewer times: row i sees the same query,
// the same key chunk in the same order, the same fmac chain and warp_sum, the same block_max_256 /
// block_sum_256 over the same 256 threads, and writes the same partial. Nothing is reassociated,
// because nothing was ever summed across the row axis.
//
// The ladder stops at TT because a group is never wider, and it is a ladder rather than a single
// narrow case so a batched drafted step (`--decode-tokens`, two or four rows per sequence) lands on
// a shape that fits it. P.attn_narrow == 0 restores the wide-only dispatch as the in-process
// control.
//
// The ladder is four bodies inlined into one kernel, so KM multiplies through it: a fused KM here
// compiles EIGHT score bodies into the caller. That is the other half of what the persistent
// kernel was paying, and it is why the leaf arm measured 130 VGPR / 10 waves even after the
// coordinate branch moved to the launcher - four copies of one body is the same union as two
// copies of two. ATTN_KM_LEAF therefore takes the group width alone: 94 VGPR and SIXTEEN waves,
// which is the register count docs/attn-score-lane.md's -13.5% panel was taken at.
//
// What that costs is bounded and named. A pass whose groups are all one row never arrives here -
// `launch_attn_wide` sends it to the folded-head unit, which has no leaf body - so the ladder's
// absence is visible only on MIXED group widths in the leaf coordinate (a drafted step with two
// or four rows a sequence), where the unit walks TT rows and keeps the group's own. The leaf
// coordinate is not a serving default, and giving it the ladder back is a launcher-level TT=1
// instantiation rather than a second body in this one.
template <int TT, int NHq, int NKVh, int HDh, int SCHED = 0, bool QINV = true, int KM = ATTN_KM_ROW,
          int HG = 1, int HB = 1, int PROBE = 0>
__device__ __forceinline__ void attn_chunk_group(Ctx & c, const FwdParams & P, const RowInfo * RB,
        const SeqCtl S, int hq, int chunk, int kv_slot, AttnPartial * PB, int pstride, int prow0) {
    if constexpr (KM != ATTN_KM_LEAF) if (P.attn_narrow) {
        if constexpr (TT > 1) if (S.nrows <= 1) {
            attn_chunk_unit<1, NHq, NKVh, HDh, SCHED, 1, QINV, KM>(c, P, RB, S, hq, chunk, kv_slot, PB, pstride, prow0); return; }
        if constexpr (TT > 2) if (S.nrows <= 2) {
            attn_chunk_unit<2, NHq, NKVh, HDh, SCHED, 1, QINV, KM>(c, P, RB, S, hq, chunk, kv_slot, PB, pstride, prow0); return; }
        if constexpr (TT > 4) if (S.nrows <= 4) {
            attn_chunk_unit<4, NHq, NKVh, HDh, SCHED, 1, QINV, KM>(c, P, RB, S, hq, chunk, kv_slot, PB, pstride, prow0); return; }
    }
    attn_chunk_unit<TT, NHq, NKVh, HDh, SCHED, HG, QINV, KM, HB, PROBE>(c, P, RB, S, hq, chunk, kv_slot, PB, pstride, prow0);
}


// The deployed persistent-kernel phase: units = (sequence, q head, chunk) over one slice.
//
// SCHED defaults to 0 here on purpose. This body is inlined into `k_forward_rows`, whose VGPR count
// sets `rows_grid_size` for every wide pass, and three engineers are measuring exactly that number
// this hour. The rescheduled body is 92 VGPR against the deployed 92 as a standalone kernel, so it
// is very likely free here too - but that is the persistent kernel's measurement to take, not this
// one's. The narrow shapes this phase serves are held separately.
//
// HALO_PK_ATTN_KM is the score body this phase compiles into the persistent kernel, and it is a
// REGISTER decision before it is an attention one. The row-major body caches the query per lane -
// qr[TT][DPL], 64 VGPRs at eight rows - and that array is this kernel's high-water mark, which sets
// `rows_grid_size` and therefore the cooperative grid of every wide pass. The lane-per-key body
// streams the same query per leaf instead of caching it, so it holds pointers where the other holds
// values. It requires the leaf K coordinate (`P.k_tiled`), which `attn_coord` selects at runtime and
// which docs/long-context-serve.md measured as bit-identical on this route.
// docs/pk-register-peak.md owns the trade.
#ifndef HALO_PK_ATTN_KM
#define HALO_PK_ATTN_KM ATTN_KM_ROW
#endif
#ifndef HALO_PK_ATTN_QINV
#define HALO_PK_ATTN_QINV 0
#endif
// HALO_PK_ATTN_RG is how many rows of the slice one score unit carries, and it is the register
// knob this kernel has been missing. `qr[TT][DPL]` is 64 VGPRs at eight rows and it is the whole
// reason `k_forward_rows<8>` cannot hold the sixteen-wave budget: at `amdgpu_waves_per_eu(13)` the
// deployed body spills 33 registers and the same body with the attention phase deleted spills none.
// A narrower unit holds `qr[RG][DPL]`, re-reads its key chunk once per group, and doubles or
// quadruples the unit count. Zero restores the deployed unit list instruction for instruction.
//
// BIT-IDENTICAL BY CONSTRUCTION: a row's score, softmax and value accumulation read only that
// row's query and the same keys in the same order, and its partial is written to its own
// `(row, head, chunk)` slot. Splitting the rows of a unit changes which workgroup computes which
// row, not what any row computes. A group's key count comes from its own last row, so an earlier
// group reads fewer keys - exactly the keys its rows mask off anyway.
#ifndef HALO_PK_ATTN_RG
#define HALO_PK_ATTN_RG 0
#endif
template <int TT, int NHq, int NKVh, int HDh, int SCHED = 0>
__device__ __forceinline__ void ph_attn(Ctx & c, const FwdParams & P, int kv_slot) {
    constexpr int GQA = NHq / NKVh;
    // Only the target model's attention geometry builds the wide-head unit. The DFlash2 drafter's
    // own layers reach this phase with block groups of DF_BLOCK rows, so they would never take the
    // branch and would only pay for its code; the MTP head shares the target geometry and is the
    // single-stream generation shape, so it does take it.
    constexpr bool WIDE_HEAD_FITS = TT > 1 && NHq == NH && NKVh == NKV && HDh == HD
                                 && GQA * ACHUNK <= LDS_FLOATS;
    // The head width is a property of the whole launch, not of one group, because it sets how many
    // units there are. The test is read from P.seqs, so nothing outside this phase has to agree
    // with it, and MAXSEQ is 8.
    bool one_row = true;
    for (int si = 0; si < P.nseq; si++) if (P.seqs[si].nrows != 1) one_row = false;
    const bool wide_head = WIDE_HEAD_FITS && one_row && P.attn_hg > 1;
    const int HU = wide_head ? NHq / GQA : NHq;   // head units per (sequence, chunk)
    // A chunk the window excludes is not a unit. The rows of a group share one unit, so the lowest
    // chunk the group can reach is the one its LOWEST-positioned row can reach, which is its first:
    // `P.rows` is in position order within a sequence. A later row's own floor is higher and its
    // fold drops the difference as the empty partials they are. The floor is read from the group's
    // first row and from no other row - a phase's `seqs` may describe rows this launch has not
    // filled, and walking the group to take a minimum reads them.
    // With no window this is chunk 0 in every group and the unit list is exactly what it was.
    const int win = P.attn_window_deal ? P.attn_window : 0;
    // The row-grouped unit list: (sequence, row group, head, chunk), behind HALO_PK_ATTN_RG. A
    // group's chunk count is its own last row's and its floor its own first row's, so the groups of
    // a slice differ by at most one chunk at each end.
    constexpr int RG = (HALO_PK_ATTN_RG > 0 && HALO_PK_ATTN_RG < TT) ? HALO_PK_ATTN_RG : TT;
    if constexpr (RG < TT) {
        int total = 0;
        for (int si = 0; si < P.nseq; si++) { const SeqCtl S = P.seqs[si];
            for (int r0 = 0; r0 < S.nrows; r0 += RG)
                total += HU * (P.rows[S.row0 + min(r0 + RG, (int) S.nrows) - 1].pos / ACHUNK + 1
                               - attn_chunk0(P.rows[S.row0 + r0].pos, win)); }
        for (int u = blockIdx.x; u >= 0 && u < total; u = next_unit(c, total)) {
            c.dirty = true;
            int si = 0, r0 = 0, rem = u;
            for (;;) {
                const SeqCtl S = P.seqs[si];
                const int nr = min(r0 + RG, (int) S.nrows) - r0;
                const int n = HU * (P.rows[S.row0 + r0 + nr - 1].pos / ACHUNK + 1
                                    - attn_chunk0(P.rows[S.row0 + r0].pos, win));
                if (rem < n) break;
                rem -= n; r0 += RG;
                if (r0 >= S.nrows) { si++; r0 = 0; }
            }
            SeqCtl G = P.seqs[si];
            const int nr = min(r0 + RG, (int) G.nrows) - r0;
            G.row0 += r0; G.nrows = nr;
            const int c0 = attn_chunk0(P.rows[G.row0].pos, win);
            if constexpr (WIDE_HEAD_FITS) if (wide_head) {
                attn_chunk_unit<1, NHq, NKVh, HDh, SCHED, GQA, false>(c, P, P.rows, G, (rem % HU) * GQA,
                                                       c0 + rem / HU, kv_slot, P.partials, AMAX_CHUNKS, 0);
                continue;
            }
            attn_chunk_group<RG, NHq, NKVh, HDh, SCHED, HALO_PK_ATTN_QINV != 0, HALO_PK_ATTN_KM>(c, P, P.rows, G,
                                                   rem % HU, c0 + rem / HU, kv_slot, P.partials, AMAX_CHUNKS, 0);
        }
        end_phase(c);
        return;
    }
    int total = 0;
    for (int si = 0; si < P.nseq; si++) { const SeqCtl S = P.seqs[si];
        total += HU * (P.rows[S.row0 + S.nrows - 1].pos / ACHUNK + 1 - attn_chunk0(P.rows[S.row0].pos, win)); }
    for (int u = blockIdx.x; u >= 0 && u < total; u = next_unit(c, total)) {
        c.dirty = true;
        int si = 0, rem = u, n = 0;
        for (;; si++) { const SeqCtl S = P.seqs[si];
            n = HU * (P.rows[S.row0 + S.nrows - 1].pos / ACHUNK + 1 - attn_chunk0(P.rows[S.row0].pos, win));
            if (rem < n) break; rem -= n; }
        const int c0 = attn_chunk0(P.rows[P.seqs[si].row0].pos, win);
        if constexpr (WIDE_HEAD_FITS) if (wide_head) {
            attn_chunk_unit<1, NHq, NKVh, HDh, SCHED, GQA, false>(c, P, P.rows, P.seqs[si], (rem % HU) * GQA, c0 + rem / HU,
                                                           kv_slot, P.partials, AMAX_CHUNKS, 0);
            continue;
        }
        // QINV=false: this kernel writes qrot in ph_attn_pre and a grid sync does not
        // invalidate the scalar cache, so the query must be read through the vector path.
        // ATTN_KM_ROW: one score body. The persistent kernel's register count is the cooperative
        // grid of every wide pass, and the lane-per-key body is a wide-launch optimization that
        // this kernel would only ever pay for - it stays correct in the leaf coordinate through
        // `k_pos_off`. docs/attn-score-lane.md holds what the fused form cost here, and
        // docs/pk-register-peak.md what it costs this kernel: 214 VGPRs against 149, inlined.
        attn_chunk_group<TT, NHq, NKVh, HDh, SCHED, HALO_PK_ATTN_QINV != 0, HALO_PK_ATTN_KM>(c, P, P.rows, P.seqs[si], rem % HU, c0 + rem / HU, kv_slot,
                                                    P.partials, AMAX_CHUNKS, 0);
    }
    end_phase(c);
}

// ---------------------------------------------------------------------------------------------

__device__ __forceinline__ void embed_row(const uint8_t * __restrict__ w, int r, float * __restrict__ out) {
    const int b = threadIdx.x;
    if (b >= NB_D) return;
    const int tile = r >> 5, lane = r & 31;
    const uint8_t * run = w + ((size_t) tile * NB_D + b) * TILE_BLOCK_BYTES;
    const uint4 qa = *(const uint4 *) (run + tile_off_qs_a(lane));
    const uint2 qb = *(const uint2 *) (run + tile_off_qs_b(lane));
    const unsigned tail = *(const unsigned *) (run + tile_off_tail(lane));
    const float scale = __half2float(__ushort_as_half((unsigned short) (tail >> 16)));
    float * o = out + b * 128;
    #pragma unroll 1
    for (int d = 0; d < 6; d++) {
        const unsigned dwd = d == 0 ? qa.x : d == 1 ? qa.y : d == 2 ? qa.z : d == 3 ? qa.w : d == 4 ? qb.x : qb.y;
        #pragma unroll 1
        for (int j = 0; j < 4; j++) {
            const int w2 = j & 1, half = j >> 1, p = 2 * d + w2;
            unsigned v = (dwd >> (8 * j)) & 0xffu, m;
            m = v * 3u; o[8 * p + 2 * half + 0]     = (float) ((int) (m >> 8) - 1) * scale; v = m & 0xffu;
            m = v * 3u; o[8 * p + 2 * half + 1]     = (float) ((int) (m >> 8) - 1) * scale; v = m & 0xffu;
            m = v * 3u; o[8 * p + 4 + 2 * half + 0] = (float) ((int) (m >> 8) - 1) * scale; v = m & 0xffu;
            m = v * 3u; o[8 * p + 4 + 2 * half + 1] = (float) ((int) (m >> 8) - 1) * scale; v = m & 0xffu;
            m = v * 3u; o[96 + 4 * d + j]           = (float) ((int) (m >> 8) - 1) * scale;
        }
    }
    #pragma unroll
    for (int h = 0; h < 2; h++) {
        unsigned v = (tail >> (8 * h)) & 0xffu, m;
        m = v * 3u; o[120 + 2 * h] = (float) ((int) (m >> 8) - 1) * scale; v = m & 0xffu;
        m = v * 3u; o[121 + 2 * h] = (float) ((int) (m >> 8) - 1) * scale; v = m & 0xffu;
        m = v * 3u; o[124 + 2 * h] = (float) ((int) (m >> 8) - 1) * scale; v = m & 0xffu;
        m = v * 3u; o[125 + 2 * h] = (float) ((int) (m >> 8) - 1) * scale;
    }
}

constexpr int AMAX_SLICES = (VOCAB + 1023) / 1024;
__device__ __forceinline__ void argmax_slice(const float * logits, int lo, int hi, float * vals, int * idxs, int u, float * lds) {
    float best = -INFINITY; int bi = 0;
    for (int i = lo + threadIdx.x; i < hi; i += NT) { float v = logits[i]; if (v > best) { best = v; bi = i; } }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) { float ob = __shfl_xor(best, o, 32); int oi = __shfl_xor(bi, o, 32); if (ob > best || (ob == best && oi < bi)) { best = ob; bi = oi; } }
    float * sm = lds; int * si = (int *) (lds + 8);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    if (l == 0) { sm[w] = best; si[w] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) { for (int i = 1; i < NW; i++) if (sm[i] > best || (sm[i] == best && si[i] < bi)) { best = sm[i]; bi = si[i]; } vals[u] = best; idxs[u] = bi; }
    __syncthreads();
}
__device__ __forceinline__ void ph_argmax_slices(Ctx & c, const FwdParams & P) {
    const int total = P.nrows * AMAX_SLICES;
    for (int u = blockIdx.x; u >= 0 && u < total; u = next_unit(c, total)) {
        c.dirty = true;
        const int row = u / AMAX_SLICES, sl = u - row * AMAX_SLICES;
        argmax_slice(P.logits + (size_t) row * VOCAB, sl * 1024, min((sl + 1) * 1024, VOCAB), P.amax_val + row * AMAX_SLICES, P.amax_idx + row * AMAX_SLICES, sl, c.lds);
    }
    end_phase(c);
}
__device__ __forceinline__ void argmax_final(const float * vals, const int * idxs, int * out, float * lds) {
    float best = -INFINITY; int bi = 0;
    for (int i = threadIdx.x; i < AMAX_SLICES; i += NT) if (vals[i] > best || (vals[i] == best && idxs[i] < bi)) { best = vals[i]; bi = idxs[i]; }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) { float ob = __shfl_xor(best, o, 32); int oi = __shfl_xor(bi, o, 32); if (ob > best || (ob == best && oi < bi)) { best = ob; bi = oi; } }
    float * sm = lds; int * si = (int *) (lds + 8);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    if (l == 0) { sm[w] = best; si[w] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) { for (int i = 1; i < NW; i++) if (sm[i] > best || (sm[i] == best && si[i] < bi)) { best = sm[i]; bi = si[i]; } out[0] = bi; }
}

} // namespace

// Grid for a cooperative persistent kernel: the occupancy API's count per WGP, held to what the
// register file actually allows.
//
// gfx11 wave32 allocates VGPRs in granules of 24 out of 1536 per SIMD32, and the occupancy API
// rounds to 8. That is why it reported one workgroup too many for a 253-VGPR kernel: 253 rounds to
// 264, which is five waves, not the six that 256 would allow. The first repair here was a 16
// register margin against a full file, and it corrects that case for the wrong reason - so it also
// vetoes every kernel that fits the file EXACTLY, and those are real: 192 x 8 and 96 x 16 are both
// 1536 and both schedulable. A 192-VGPR drafter body was launched on three workgroups per WGP
// while the API, the compiler's own occupancy remark and the granule arithmetic all said four
// (docs/drafter-occupancy.md).
//
// Rounding the way the hardware rounds reproduces every grid this engine launches today - 135 VGPR
// -> 100 workgroups, 120 -> 120, 239 -> 60, 247 -> 40 - and differs only at the full-file points.
//
// BOTH TERMS ARE NECESSARY AND THAT IS MEASURED NOW RATHER THAN ASSUMED. `bench/occ_resident`
// counts blocks that are resident at the same time - no grid sync, so it cannot hang - and the
// device follows this granule-24 arithmetic at every register count and every LDS block it was run
// on, while `hipOccupancyMaxActiveBlocksPerMultiprocessor` is wrong in BOTH directions: at 116
// VGPRs it says five blocks and the device holds six, at 121 it says six and the device holds five,
// at 180 it says three and the device holds four.
//
// So the minimum is not conservatism, it is the only correct answer available here, for two
// different reasons pulling opposite ways. Where the API over-counts, launching its number
// deadlocks: 121 to 144 VGPRs really is five blocks and a cooperative grid of six would wait
// forever at the first `grid_sync`. Where the API UNDER-counts, its number is still a hard ceiling,
// because `hipLaunchCooperativeKernel` validates the grid against its own estimate and refuses with
// "too many blocks in cooperative launch" - measured, on this kernel, at 120 workgroups for a body
// the hardware demonstrably holds 120 of. A block the API does not believe in cannot be launched
// however real it is.
//
// WHAT THAT COSTS, AND WHERE THE GRID ACTUALLY COMES FROM: the API and the device agree at 96
// registers and below (eight blocks) and at 105 to 112 (six), and disagree at 113 to 120, which is
// exactly where `waves_per_eu(12)` lands the eight-row persistent kernel. Its grid is therefore set
// by an estimator's arithmetic rather than by the machine, and the way to move it is to move the
// allocation into a window where the two agree - see `rows_occ_rule` in kernels/halo_rows.hip and
// the measured ladder in docs/rows-grid-occupancy.md.
//
// The LDS term was wrong for an unrelated reason and is fixed here. `sharedMemPerBlock` is 65536
// because that is the most ONE workgroup may allocate; the WGP has 128 kB and hands it to as many
// blocks as fit. Measured: eight blocks of a 9508-byte kernel are co-resident, which is 76 kB and
// impossible under the 64 kB reading that `docs/rows-grid-occupancy.md` used to cap this kernel at
// six blocks however few registers it used. LDS does not bind any kernel in this engine.
static inline int coop_grid(const void * fn) {
    int per = 0;
    (void) hipOccupancyMaxActiveBlocksPerMultiprocessor(&per, fn, NT, 0);
    hipFuncAttributes attr{};
    (void) hipFuncGetAttributes(&attr, fn);
    const int vg = (attr.numRegs + 23) / 24 * 24;          // hardware VGPR granule
    int waves = vg > 0 ? 1536 / vg : 16;                   // resident waves per SIMD32
    if (waves > 16) waves = 16;                            // the SIMD32 has sixteen wave slots
    int blocks = waves / (NW / 4);                         // a 256-thread workgroup is 2 of them per SIMD32
    const size_t lds = attr.sharedSizeBytes;
    if (lds > 0) {                                         // 128 kB per WGP, whatever one block may ask for
        const int by_lds = (int) (131072 / lds);
        if (by_lds < blocks) blocks = by_lds;
    }
    const int api = per;
    if (blocks > 0 && per > blocks) per = blocks;
    hipDeviceProp_t prop; (void) hipGetDeviceProperties(&prop, 0);
    // What the runtime and this correction each said, with the count the previous 8-register
    // rounding produced beside it: a register change that does not move the grid is
    // indistinguishable from one that does until you can see both numbers. Either environment
    // name prints it.
    if (getenv("HALO_ROWS_GRID") || getenv("HALO_COOP_GRID_PRINT")) {
        fprintf(stderr, "[coop_grid] numRegs %d (granule %d, %d waves, %d blocks/WGP on the device), lds %zu, api %d -> grid %d%s\n",
                attr.numRegs, vg, waves, blocks, lds, api,
                std::max(1, per) * prop.multiProcessorCount,
                api < blocks ? "  [API under-counts: the device holds more and the launcher will not take them]" : "");
    }
    return std::max(1, per) * prop.multiProcessorCount;
}

} // namespace halo

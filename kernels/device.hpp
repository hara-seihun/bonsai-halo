// Device-side building blocks shared by the standalone kernels and the persistent forward kernel.
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "halo_kernels.h"
#include "halo_expand.hpp"

// Which ternary operand map the matvec bodies build their 32 IU8 dwords with. Both arms live in
// kernels/halo_expand.hpp and produce the same dwords byte for byte, proved against
// `halo::decode_block` by `make kernels/head_op_check`; 1 is the perm gather 9681dcc7 landed in the
// FFN slice, 0 the peel this body carried. docs/mv-peel-gather.md, docs/rows-lds-occupancy.md.
//
// It is the DEFAULT of a template parameter rather than a plain `#if`, so one process can hold both
// arms and a panel can alternate them under one clock: `bench/mvsched` walks peel, gather and an
// expansion-free ablation in palindrome order, which is how the price of an issue slot in this loop
// was measured. Nothing in the runtime instantiates anything but the default.
// docs/mv-peel-rows.md.
#ifndef HALO_MV_EXPAND
#define HALO_MV_EXPAND 1
#endif

namespace halo {

// ---------------------------------------------------------------------------------------------
// helpers

typedef unsigned short us2 __attribute__((ext_vector_type(2)));

// Peel one trit from each byte of a (byte, byte) pair held in the low bytes of two u16 halves.
// Returns dword [t(first), 0, t(second), 0]; r keeps the remainders.
__device__ __forceinline__ unsigned peel(unsigned & r) {
    us2 m = __builtin_bit_cast(us2, r) * (us2){3, 3};
    us2 t = m >> 8;
    r = __builtin_bit_cast(unsigned, m) & 0x00ff00ffu;
    return __builtin_bit_cast(unsigned, t);
}
__device__ __forceinline__ unsigned peel_last(unsigned r) {
    us2 m = __builtin_bit_cast(us2, r) * (us2){3, 3};
    return __builtin_bit_cast(unsigned, m >> 8);
}
__device__ __forceinline__ int dot4(unsigned trits, int x, int acc) {
    return __builtin_amdgcn_sudot4(false, (int) trits, true, x, acc, false);
}

// Wave reductions through DPP lane shuffles (row_ror / quad_perm / permlanex16), 2.7x faster than
// the ds_bpermute path __shfl_xor takes on gfx11. Every lane receives the result.
template <int CTRL> __device__ __forceinline__ float dppf(float v) { return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), CTRL, 0xf, 0xf, true)); }
__device__ __forceinline__ float xrow16(float v) { return __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(v), __float_as_int(v), 0x76543210u, 0xfedcba98u, false, false)); }
__device__ __forceinline__ float warp_sum(float v) {
    v += dppf<0xB1>(v);   // quad_perm(1,0,3,2)
    v += dppf<0x4E>(v);   // quad_perm(2,3,0,1)
    v += dppf<0x124>(v);  // row_ror 4
    v += dppf<0x128>(v);  // row_ror 8
    return v + xrow16(v);
}
__device__ __forceinline__ float warp_max(float v) {
    v = fmaxf(v, dppf<0xB1>(v));
    v = fmaxf(v, dppf<0x4E>(v));
    v = fmaxf(v, dppf<0x124>(v));
    v = fmaxf(v, dppf<0x128>(v));
    return fmaxf(v, xrow16(v));
}
__device__ __forceinline__ int warp_sum_i(int v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    return v;
}
// block reduce over 256 threads (8 waves); every thread receives the result
__device__ __forceinline__ float block_sum_256(float v, float * red) {
    v = warp_sum(v);
    int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    float r = red[0];
    #pragma unroll
    for (int i = 1; i < 8; i++) r += red[i];
    return r;
}
__device__ __forceinline__ float block_max_256(float v, float * red) {
    v = warp_max(v);
    int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    float r = red[0];
    #pragma unroll
    for (int i = 1; i < 8; i++) r = fmaxf(r, red[i]);
    return r;
}
__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }
__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float bf16_to_f32(unsigned short b) { return __uint_as_float(((unsigned) b) << 16); }

// ---------------------------------------------------------------------------------------------
// Ternary matvec. One lane = one output row, one wave = one tile of 32 rows, WAVES waves split K.

// Dot product of one lane's row over blocks [0, nblk) of a tile run, against the quantised input.
// run points at the first 896 B block-run this wave handles; xq/xs/xsum are already offset to match.
#ifndef MV_ROWS_ATTR
#define MV_ROWS_ATTR __forceinline__
#endif
// One block's bytes for one lane, loadable ahead of time.
struct BlockRegs { uint4 qa; uint2 qb; unsigned tail; };
__device__ __forceinline__ BlockRegs load_block(const uint8_t * __restrict__ run, int lane) {
    BlockRegs r;
    r.qa = *(const uint4 *) (run + tile_off_qs_a(lane));
    r.qb = *(const uint2 *) (run + tile_off_qs_b(lane));
    r.tail = *(const unsigned *) (run + tile_off_tail(lane));
    return r;
}

// EXP is which expansion builds the operand dwords: 0 the peel, 1 the perm gather. Both write the
// same 32 dwords, so this selects an instruction sequence and not a numerical map, and every body
// below takes it as a defaulted template parameter so a bench can hold both at once.
template <int EXP>
__device__ __forceinline__ void mv_expand_tr(const unsigned dw[6], unsigned tail, unsigned tr[32]) {
#ifdef HALO_MV_EXPAND_PROBE
    // EXP = 2 is an ABLATION, not an arm: it hands the dot products the stored bytes themselves, so
    // the block's whole expansion disappears and its answer is wrong by construction. It exists to
    // price what an issue slot in this loop is worth - the two real arms differ by 64 slots and the
    // ladder needs the 168-slot end of it. Only `bench/mvsched` defines the macro; no runtime
    // translation unit compiles this branch. docs/mv-peel-rows.md.
    if constexpr (EXP == 2) {
        #pragma unroll
        for (int i = 0; i < 32; i++) tr[i] = dw[i % 6] & 0x02020202u;
        (void) tail;
        return;
    }
#endif
    if constexpr (EXP == 1) hx_expand_perm(dw, tail, tr);
    else                    hx_expand_peel(dw, tail, tr);
}

template <int NBLK, int UNROLL = 2, typename XQ, typename XS, typename XSUM, int EXP = HALO_MV_EXPAND>
__device__ MV_ROWS_ATTR float mv_rows_pre(const uint8_t * __restrict__ run, int lane, XQ xq, XS xs, XSUM xsum, BlockRegs first) {
    float y = 0.0f;
    uint4 qa = first.qa; uint2 qb = first.qb; unsigned tail = first.tail;
    #pragma unroll UNROLL
    for (int b = 0; b < NBLK; b++) {
        uint4 nqa = qa; uint2 nqb = qb; unsigned ntail = tail;
        if (b + 1 < NBLK) {
            const uint8_t * nrun = run + (b + 1) * TILE_BLOCK_BYTES;
            nqa = *(const uint4 *) (nrun + tile_off_qs_a(lane));
            nqb = *(const uint2 *) (nrun + tile_off_qs_b(lane));
            ntail = *(const unsigned *) (nrun + tile_off_tail(lane));
        }
        XQ xb = xq + b * 32;
        const unsigned dw[6] = { qa.x, qa.y, qa.z, qa.w, qb.x, qb.y };
        unsigned tr[32];
        mv_expand_tr<EXP>(dw, tail, tr);
        int acc = 0;
        #pragma unroll
        for (int d = 0; d < 6; d++) {
            acc = dot4(tr[4 * d + 0], xb[4 * d + 0], acc);
            acc = dot4(tr[4 * d + 1], xb[4 * d + 1], acc);
            acc = dot4(tr[4 * d + 2], xb[4 * d + 2], acc);
            acc = dot4(tr[4 * d + 3], xb[4 * d + 3], acc);
            acc = dot4(tr[24 + d], xb[24 + d], acc);
        }
        acc = dot4(tr[30], xb[30], acc);
        acc = dot4(tr[31], xb[31], acc);
        acc -= xsum[b];
        y = fmaf((float) acc, __half2float(__ushort_as_half((unsigned short) (tail >> 16))) * xs[b], y);
        qa = nqa; qb = nqb; tail = ntail;
    }
    return y;
}

template <int NBLK, int UNROLL = 2, typename XQ, typename XS, typename XSUM>
__device__ __forceinline__ float mv_rows(const uint8_t * __restrict__ run, int lane, XQ xq, XS xs, XSUM xsum) {
    return mv_rows_pre<NBLK, UNROLL>(run, lane, xq, xs, xsum, load_block(run, lane));
}

// ---------------------------------------------------------------------------------------------
// Prep: norm / elementwise / sign / Hadamard / int8 quant.

// One 1024-element chunk of the prep pipeline, executed by a 256-thread workgroup.
__device__ __forceinline__ void prep_chunk(const PrepArgs & a, int chunk, float * s, float * red) {
    const int tid = threadIdx.x;
    const int base = chunk * 1024;

    float inv = 1.0f;
    if (a.flags & PREP_NORM) {
        float ss = 0.0f;
        for (int i = tid; i < a.n; i += 256) { float v = a.x[i]; ss = fmaf(v, v, ss); }
        ss = block_sum_256(ss, red);
        inv = rsqrtf(ss / (float) a.n + a.eps);
    }

    // GDN gated norm: per source head rsqrt(mean(o^2) + eps); wave w handles grouped slot q = 8 chunk + w
    if (a.flags & PREP_GDN_NORM) {
        const int wave = tid >> 5, lane = tid & 31;
        const int q = (base >> 7) + wave, r = q % 3, kh = q / 3, h = kh + 16 * r;
        const float * oh = a.x + h * 128;
        float ss = 0.0f;
        #pragma unroll
        for (int k = 0; k < 4; k++) { float v = oh[lane + 32 * k]; ss = fmaf(v, v, ss); }
        ss = warp_sum(ss);
        if (lane == 0) red[wave] = rsqrtf(ss / 128.0f + a.eps);
        __syncthreads();
    }
    // attention combine: chunk covers heads 4 chunk .. 4 chunk + 3
    int nchunks = 0;
    if (a.flags & PREP_ATTN_COMBINE) nchunks = a.pos[0] / ATTN_CHUNK + 1;

    #pragma unroll
    for (int k = 0; k < 4; k++) {
        int i = tid + k * 256;
        int g = base + i;
        float v;
        if (a.flags & PREP_GDN_NORM) {
            int hd = g & 127, q = g >> 7, r = q % 3, kh = q / 3;
            int src = hd + 128 * (kh + 16 * r);
            v = a.x[src] * red[q & 7] * a.norm_w[hd] * silu(a.x2[src]);
        } else if (a.flags & PREP_ATTN_COMBINE) {
            const int hq = g >> 8, d = g & 255;
            const AttnPartial * part = a.partials + (size_t) hq * ATTN_MAX_CHUNKS;
            float M = -INFINITY;
            for (int c = 0; c < nchunks; c++) M = fmaxf(M, part[c].m);
            float num = 0.0f, den = 0.0f;
            for (int c = 0; c < nchunks; c++) { const float w = __expf(part[c].m - M); num = fmaf(w, part[c].acc[d], num); den = fmaf(w, part[c].l, den); }
            v = num / den * sigmoid(a.x2[hq * 2 * HD + HD + d]);
        } else if (a.flags & PREP_PERMUTE_GDN) {
            // grouped position g = hd + 128 (r + 3 kh)  <-  tiled source hd + 128 (kh + 16 r)
            int hd = g & 127, q = g >> 7, r = q % 3, kh = q / 3;
            v = a.x[hd + 128 * (kh + 16 * r)];
        } else {
            v = a.x[g];
        }
        if (a.flags & PREP_NORM) {
            v = v * inv * a.norm_w[g];
            if (a.out_norm) a.out_norm[g] = v;
        }
        if (a.flags & PREP_SILU_MUL) v = silu(v) * a.x2[g];
        if (a.flags & PREP_SIGN) v *= a.signs[g];
        s[i] = v;
    }
    __syncthreads();

    if (a.flags & PREP_HADAMARD) {
        #pragma unroll
        for (int len = 1; len < 1024; len <<= 1) {
            #pragma unroll
            for (int k = 0; k < 2; k++) {
                int pi = tid + k * 256;
                int i = ((pi / len) * 2 * len) + (pi % len);
                int j = i + len;
                float x0 = s[i], x1 = s[j];
                s[i] = x0 + x1; s[j] = x0 - x1;
            }
            __syncthreads();
        }
        // 1/sqrt(1024)
        #pragma unroll
        for (int k = 0; k < 4; k++) s[tid + k * 256] *= 0.03125f;
        __syncthreads();
    }

    if (a.flags & PREP_SIGN_AFTER) {
        #pragma unroll
        for (int k = 0; k < 4; k++) { int i = tid + k * 256; s[i] *= a.signs[base + i]; }
        __syncthreads();
    }

    if (a.flags & PREP_STORE_F32) {
        #pragma unroll
        for (int k = 0; k < 4; k++) { int i = tid + k * 256; a.out_f32[base + i] = s[i]; }
    }

    if (a.flags & PREP_QUANT) {
        // wave w quantises block w of this 1024 chunk; lane holds 4 consecutive elements
        const int wave = tid >> 5, lane = tid & 31;
        const int off = wave * 128 + lane * 4;
        float v0 = s[off], v1 = s[off + 1], v2 = s[off + 2], v3 = s[off + 3];
        float amax = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
        amax = warp_max(amax);
        float scale = amax / 127.0f;
        float iscale = amax > 0.0f ? 127.0f / amax : 0.0f;
        int q0 = __float2int_rn(v0 * iscale), q1 = __float2int_rn(v1 * iscale), q2 = __float2int_rn(v2 * iscale), q3 = __float2int_rn(v3 * iscale);
        int sum = warp_sum_i(q0 + q1 + q2 + q3);
        unsigned packed = (unsigned) (q0 & 0xff) | ((unsigned) (q1 & 0xff) << 8) | ((unsigned) (q2 & 0xff) << 16) | ((unsigned) (q3 & 0xff) << 24);
        ((unsigned *) a.xq)[(base + off) >> 2] = packed;
        if (lane == 0) {
            int blk = (base >> 7) + wave;
            a.xs[blk] = scale;
            a.xsum[blk] = sum;
        }
    }
}


typedef const __attribute__((address_space(4))) int * cptr;
typedef const __attribute__((address_space(4))) float * cfptr;

// ---------------------------------------------------------------------------------------------
// Single-token operand map: HALO five-trit bytes straight to byte palettes.
//
// The stored byte is b = ceil(256 q / 243) for q = 81 t0 + 27 t1 + 9 t2 + 3 t3 + t4. One packed
// multiply chain lifts out the two nine-valued two-trit codes and the leftover trit without ever
// forming t0..t3 as separate values:
//
//   A = (b*9) >> 8 = 3 t0 + t1      r  = (b*9) & 255
//   B = (r*9) >> 8 = 3 t2 + t3      r2 = (r*9) & 255
//   C = (r2*3) >> 8 = t4
//
// A and B are v_perm_b32 selectors: a code *is* an index into an eight-byte palette. Nine codes
// need a ninth source byte, and selector 8 gives the sign replication of source byte 1, which is
// 0x00 for every palette here. That fixes the palette to carry the complement 2 - t, so the dot
// returns sum (2 - t_i) x_i = 2 sum(x_i) - sum t_i x_i. The block correction absorbs it exactly:
// the deployed path forms acc - xsum, this one forms xsum - acc. Both are integer, so the float
// input to the block fma is bit-identical and the FP32 schedule is untouched.
//
// Per 128-weight block: 199 decode ops against 264 for the peel, same 32 dot4 and same 3 loads.
// Selector semantics for gfx1151 are the measured table in
// kelana research/ffn/batched/dense-consumer/results/perm-semantics.json.

__device__ __forceinline__ unsigned vperm(unsigned a, unsigned b, unsigned s) { return __builtin_amdgcn_perm(a, b, s); }

// palette bytes, index = two-trit code, value = complement of that trit
// first trit  2 - code/3: 2 2 2 1 1 1 0 0 (0)     second trit 2 - code%3: 2 1 0 2 1 0 2 1 (0)
constexpr unsigned PAL0_LO = 0x01020202u, PAL0_HI = 0x00000101u;  // source bytes 0..3 / 4..7
constexpr unsigned PAL1_LO = 0x02000102u, PAL1_HI = 0x01020001u;
constexpr unsigned PALC_LO = 0x00000102u, PALC_HI = 0x00000000u;  // leftover trit, 2 - t
// selector constants: source byte 0..3 = arg b, 4..7 = arg a
constexpr unsigned SEL_PAIR = 0x06040200u;  // [b.0, b.2, a.0, a.2]
constexpr unsigned SEL_LO   = 0x05010400u;  // [b.0, a.0, b.1, a.1]
constexpr unsigned SEL_HI   = 0x07030602u;  // [b.2, a.2, b.3, a.3]
constexpr unsigned SEL_C    = 0x06020400u;  // [b.0, a.0, b.2, a.2]

// two-trit codes of the byte pair held as (u16, u16) in H
__device__ __forceinline__ void pal_codes(unsigned H, unsigned & A, unsigned & B, unsigned & C) {
    us2 m = __builtin_bit_cast(us2, H) * (us2){9, 9};
    A = __builtin_bit_cast(unsigned, m >> 8);
    m = __builtin_bit_cast(us2, __builtin_bit_cast(unsigned, m) & 0x00ff00ffu) * (us2){9, 9};
    B = __builtin_bit_cast(unsigned, m >> 8);
    m = __builtin_bit_cast(us2, __builtin_bit_cast(unsigned, m) & 0x00ff00ffu) * (us2){3, 3};
    C = __builtin_bit_cast(unsigned, m >> 8);
}
__device__ __forceinline__ void pal_codes2(unsigned H, unsigned & A, unsigned & B) {
    us2 m = __builtin_bit_cast(us2, H) * (us2){9, 9};
    A = __builtin_bit_cast(unsigned, m >> 8);
    m = __builtin_bit_cast(us2, __builtin_bit_cast(unsigned, m) & 0x00ff00ffu) * (us2){9, 9};
    B = __builtin_bit_cast(unsigned, m >> 8);
}
// K0 holds the codes of the even byte pair, K1 of the odd one; out lo/hi are the two operand
// dwords [w0(first), w1(first), w0(second), w1(second)] of each pair.
__device__ __forceinline__ void pal_expand(unsigned K0, unsigned K1, unsigned & lo, unsigned & hi) {
    const unsigned sel = vperm(K1, K0, SEL_PAIR);
    const unsigned q0 = vperm(PAL0_HI, PAL0_LO, sel);
    const unsigned q1 = vperm(PAL1_HI, PAL1_LO, sel);
    lo = vperm(q1, q0, SEL_LO);
    hi = vperm(q1, q0, SEL_HI);
}

// One HALO block for one lane -> 32 complemented IU8 operand dwords, element order unchanged.
__device__ __forceinline__ void pal_block(const uint4 & qa, const uint2 & qb, unsigned tail, unsigned tr[32]) {
    const unsigned dw[6] = { qa.x, qa.y, qa.z, qa.w, qb.x, qb.y };
    #pragma unroll
    for (int d = 0; d < 6; d++) {
        unsigned A0, B0, C0, A1, B1, C1;
        pal_codes(dw[d] & 0x00ff00ffu, A0, B0, C0);
        pal_codes((dw[d] >> 8) & 0x00ff00ffu, A1, B1, C1);
        pal_expand(A0, A1, tr[4 * d + 0], tr[4 * d + 2]);
        pal_expand(B0, B1, tr[4 * d + 1], tr[4 * d + 3]);
        tr[24 + d] = vperm(PALC_HI, PALC_LO, vperm(C1, C0, SEL_C));
    }
    unsigned A, B;
    pal_codes2((tail & 0xffu) | ((tail << 8) & 0xff0000u), A, B);
    const unsigned sel = vperm(B, A, SEL_PAIR);
    const unsigned q0 = vperm(PAL0_HI, PAL0_LO, sel), q1 = vperm(PAL1_HI, PAL1_LO, sel);
    tr[30] = vperm(q1, q0, SEL_LO);
    tr[31] = vperm(q1, q0, SEL_HI);
}

// Single-row dot over NBLK blocks. Same loads, same one-block-deep prefetch and same FP32
// accumulation order as mv_rows_t<NBLK, 1, false>; only the operand map differs.
template <int NBLK>
__device__ __forceinline__ void mv_rows_pal(const uint8_t * __restrict__ run, int lane, cptr xq, cfptr xs, cptr xsum, float & y) {
    uint4 qa = *(const uint4 *) (run + tile_off_qs_a(lane));
    uint2 qb = *(const uint2 *) (run + tile_off_qs_b(lane));
    unsigned tail = *(const unsigned *) (run + tile_off_tail(lane));
    #pragma unroll 1
    for (int b = 0; b < NBLK; b++) {
        uint4 nqa = qa; uint2 nqb = qb; unsigned ntail = tail;
        if (b + 1 < NBLK) {
            const uint8_t * nrun = run + (b + 1) * TILE_BLOCK_BYTES;
            nqa = *(const uint4 *) (nrun + tile_off_qs_a(lane));
            nqb = *(const uint2 *) (nrun + tile_off_qs_b(lane));
            ntail = *(const unsigned *) (nrun + tile_off_tail(lane));
        }
        unsigned tr[32];
        pal_block(qa, qb, tail, tr);
        int acc = 0;
        #pragma unroll
        for (int i = 0; i < 32; i++) acc = __builtin_amdgcn_sudot4(false, (int) tr[i], true, xq[b * 32 + i], acc, false);
        const float wscale = __half2float(__ushort_as_half((unsigned short) (tail >> 16)));
        y = fmaf((float) (xsum[b] - acc), wscale * xs[b], y);
        qa = nqa; qb = nqb; tail = ntail;
    }
}


// dot of one lane's weight row over NBLK blocks against TT activation rows
//
// ACC is how many independent int32 accumulator chains a row's 32 `v_dot4` are split across. At
// one chain the compiler must fence every pair of them with `s_delay_alu VALU_DEP_1`, because each
// dot4 reads the result the previous one wrote: at TT = 8 that is 128 stall hints per 128-K block,
// and the block carries more scheduling slots than a third of its work. Four chains remove 90 of
// them for seven work slots and no register.
//
// This re-associates nothing the machine can observe. A row's dot is a sum of NBLK * 32 exact
// int32 terms each bounded by 4 * 2 * 127, so every summation order yields the same integer, the
// same `acc - xsum`, and the same bits into `fmaf(acc, wscale * xs, y)`. Measured: 248,320
// full-vocabulary logit floats byte for byte equal. docs/mv-dot4-body.md.
//
// One row is a different machine: it has 232 slots of trit expansion to interleave with its 32
// dot4, so no chain ever stalls and it runs at 240 GB/s of a 242 GB/s roof. TT = 1 keeps one
// accumulator, and the deployed single-token kernel is untouched code.
#ifndef HALO_MV_ACC
#define HALO_MV_ACC 4       // the measured knee; two chains still stall, eight cost slots
#endif

// RB is how many rows share one scalar wait, and it is the axis that separates the eight-row pass
// from the one-row pass.
//
// EVERY SCALAR WAIT ON gfx11 IS `s_waitcnt lgkmcnt(0)`. SMEM returns out of order, so the counter
// cannot be waited part way: one wait drains every scalar load the wave has outstanding. The
// deployed row loop asks for row r+1's thirty-two activation dwords and then waits for row r a few
// instructions later, so the request it just issued is drained by the wait it is already paying and
// the double buffer buys nothing. Its second wait is the drain scalars, `xs` and `xsum`, loaded
// inside the row. Counted in the assembly of `mv_body_isa.hip`: a TT = 8 block carries **sixteen**
// `lgkmcnt(0)` waits, a TT = 1 block carries **one**, and both stream the same 896 weight bytes.
// That is the whole distance between 127 GB/s and 200 on one image - not the trit expansion, not
// the issue census, not the stride.
//
// So a row batch requests RB rows of activations and their two drain scalars together, waits once,
// and then retires RB * 32 dot products whose RB * ACC accumulator chains are independent by
// construction. `TT / RB` waits a block instead of `2 * TT`, and the first batch is requested above
// the trit expansion so its round trip has 229 slots of cover that no later batch can get.
//
// WHAT DOES NOT MOVE: the bytes, the blocks, the K order, the weight image, the dot4 operands and
// the per-block `fmaf(acc - xsum, wscale * xs, y)`. A row's int32 sum is exact and two's-complement
// addition is associative, so regrouping the chains cannot change it - the same argument ACC is
// already shipped on. Rows past `nrows` read row `nrows - 1`'s scalars, in bounds and discarded,
// exactly as the deployed loop already clamps its activation row.
#ifndef HALO_MV_RB
#define HALO_MV_RB 2        // the measured knee: docs/mv-scalar-waits.md
#endif
// Depth - more than one weight block in flight - is a MEASURED NEGATIVE here and is not a
// parameter: two blocks was a null and three cost 2.9% at eight rows, 8% at four. The weight loads
// are vector loads on `vmcnt`, which is partially waitable, so the wave was never short of
// outstanding weight bytes. bench/mvsched carries both arms; docs/mv-scalar-waits.md has the table.

// One block of HALO five-trit bytes for one lane, and its expansion to 32 IU8 operand dwords.
struct MvBlk { uint4 qa; uint2 qb; unsigned tail; };
__device__ __forceinline__ MvBlk mv_load_blk(const uint8_t * __restrict__ p, int lane) {
    MvBlk r;
    r.qa = *(const uint4 *) (p + tile_off_qs_a(lane));
    r.qb = *(const uint2 *) (p + tile_off_qs_b(lane));
    r.tail = *(const unsigned *) (p + tile_off_tail(lane));
    return r;
}
template <int EXP = HALO_MV_EXPAND>
__device__ __forceinline__ float mv_expand(const MvBlk & w, unsigned tr[32]) {
    const unsigned dw[6] = { w.qa.x, w.qa.y, w.qa.z, w.qa.w, w.qb.x, w.qb.y };
    mv_expand_tr<EXP>(dw, w.tail, tr);
    return __half2float(__ushort_as_half((unsigned short) (w.tail >> 16)));
}

// Row-batched schedule: `TT / RB` scalar waits a block instead of `2 * TT`.
template <int NBLK, int TT, int ACC, int RB, int EXP = HALO_MV_EXPAND>
__device__ __forceinline__ void mv_rows_batched(const uint8_t * __restrict__ run, int lane, cptr xq, int xstride,
                                                cfptr xs, cptr xsum, int xsstride, int nrows, float y[TT]) {
    constexpr int BB = TILE_BLOCK_BYTES;
    MvBlk cur = mv_load_blk(run, lane);
    #pragma unroll 1
    for (int b = 0; b < NBLK; b++) {
        MvBlk nxt = cur;
        if (b + 1 < NBLK) nxt = mv_load_blk(run + (b + 1) * BB, lane);
        // One batch of scalars lives in these registers for its whole turn; the next batch writes
        // the same ones, which is what keeps its requests behind the wait they must follow.
        int xa[RB][32]; float sc[RB]; int sm[RB];
        #pragma unroll
        for (int j = 0; j < RB; j++) {
            const int rr = min(j, nrows - 1);
            cptr xr = xq + rr * xstride + b * 32;
            #pragma unroll
            for (int i = 0; i < 32; i++) xa[j][i] = xr[i];
            sc[j] = xs[rr * xsstride + b];
            sm[j] = xsum[rr * xsstride + b];
        }
        __builtin_amdgcn_sched_barrier(0);
        unsigned tr[32];
        const float wscale = mv_expand<EXP>(cur, tr);
        #pragma unroll
        for (int base = 0; base < TT; base += RB) {
            if (base > 0) {
                #pragma unroll
                for (int j = 0; j < RB; j++) {
                    const int rr = min(base + j, nrows - 1);
                    cptr xr = xq + rr * xstride + b * 32;
                    #pragma unroll
                    for (int i = 0; i < 32; i++) xa[j][i] = xr[i];
                    sc[j] = xs[rr * xsstride + b];
                    sm[j] = xsum[rr * xsstride + b];
                }
                __builtin_amdgcn_sched_barrier(0);
            }
            #pragma unroll
            for (int j = 0; j < RB; j++) {
                if (base + j < nrows) {
                    int part[ACC];
                    #pragma unroll
                    for (int q = 0; q < ACC; q++) part[q] = 0;
                    #pragma unroll
                    for (int i = 0; i < 32; i++) part[i % ACC] = __builtin_amdgcn_sudot4(false, (int) tr[i], true, xa[j][i], part[i % ACC], false);
                    int acc = part[0];
                    #pragma unroll
                    for (int q = 1; q < ACC; q++) acc += part[q];
                    y[base + j] = fmaf((float) (acc - sm[j]), wscale * sc[j], y[base + j]);
                }
            }
            __builtin_amdgcn_sched_barrier(0);
        }
        cur = nxt;
    }
}

// The deployed body: one weight block of lookahead, one activation row of lookahead, and two
// scalar drains a row. `TT = 1` pays one drain a block and runs at its bandwidth roof, and the
// drafter's `Q8` image has its own block bytes; both keep this body instruction for instruction.
template <int NBLK, int TT, bool Q8, int ACC, int EXP = HALO_MV_EXPAND>
__device__ __forceinline__ void mv_rows_deployed(const uint8_t * __restrict__ run, int lane, cptr xq, int xstride, cfptr xs, cptr xsum, int xsstride, int nrows, float y[TT]) {
    constexpr int BB = Q8 ? Q8_TILE_BLOCK_BYTES : TILE_BLOCK_BYTES;
    uint4 qa[Q8 ? 8 : 1]; uint2 qb; unsigned tail = 0;
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
        uint4 nqa[Q8 ? 8 : 1]; uint2 nqb = qb; unsigned ntail = tail;
        #pragma unroll
        for (int i = 0; i < (Q8 ? 8 : 1); i++) nqa[i] = qa[i];
        if (b + 1 < NBLK) {
            const uint8_t * nrun = run + (b + 1) * BB;
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
            mv_expand_tr<EXP>(dw, tail, tr);
            wscale = __half2float(__ushort_as_half((unsigned short) (tail >> 16)));
        }
        // rows: the next row's activations are requested (scalar cache) before the current row's dot
        // products; the scheduling barrier keeps the compiler from hoisting every row's loads to the
        // top, which does not fit the SGPR file and serialises the loads behind full waits
        {
            int xa[32], xn[32];
            #pragma unroll
            for (int i = 0; i < 32; i++) xa[i] = xq[b * 32 + i];
            #pragma unroll
            for (int r = 0; r < TT; r++) {
                if (r + 1 < TT) {
                    cptr xr = xq + min(r + 1, nrows - 1) * xstride + b * 32;
                    #pragma unroll
                    for (int i = 0; i < 32; i++) xn[i] = xr[i];
                }
                if (r < nrows) {
                    int part[ACC];
                    #pragma unroll
                    for (int j = 0; j < ACC; j++) part[j] = 0;
                    #pragma unroll
                    for (int i = 0; i < 32; i++) part[i % ACC] = __builtin_amdgcn_sudot4(Q8, (int) tr[i], true, xa[i], part[i % ACC], false);
                    int acc = part[0];
                    #pragma unroll
                    for (int j = 1; j < ACC; j++) acc += part[j];
                    if (!Q8) acc -= xsum[r * xsstride + b];
                    y[r] = fmaf((float) acc, wscale * xs[r * xsstride + b], y[r]);
                }
                if (TT > 1) __builtin_amdgcn_sched_barrier(0);
                #pragma unroll
                for (int i = 0; i < 32; i++) xa[i] = xn[i];
            }
        }
        #pragma unroll
        for (int i = 0; i < (Q8 ? 8 : 1); i++) qa[i] = nqa[i];
        qb = nqb; tail = ntail;
    }
}

// Which body a shape takes. A multi-row ternary pass takes the row batch; `TT = 1` pays one scalar
// drain a block already and the drafter's Q8 image has its own block bytes, so both keep the
// deployed body, instruction for instruction. `HALO_MV_SCHED=0` restores the deployed schedule
// everywhere and is the control arm both measurement tools walk.
#ifndef HALO_MV_SCHED
#define HALO_MV_SCHED 1
#endif
template <int NBLK, int TT, bool Q8, int ACC = (TT > 1 ? HALO_MV_ACC : 1), int MVS = HALO_MV_SCHED,
          int RB = HALO_MV_RB, int EXP = HALO_MV_EXPAND>
__device__ __forceinline__ void mv_rows_t(const uint8_t * __restrict__ run, int lane, cptr xq, int xstride, cfptr xs, cptr xsum, int xsstride, int nrows, float y[TT]) {
    if constexpr (!Q8 && TT > 1 && MVS == 1)
        mv_rows_batched<NBLK, TT, ACC, (RB < TT ? RB : TT), EXP>(run, lane, xq, xstride, xs, xsum, xsstride, nrows, y);
    else
        mv_rows_deployed<NBLK, TT, Q8, ACC, EXP>(run, lane, xq, xstride, xs, xsum, xsstride, nrows, y);
}

} // namespace halo

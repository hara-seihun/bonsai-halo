// Matvec bodies for the Q4 drafter coordinate. Included by kernels/halo_draft.hip only.
//
// These mirror `ph_matvec` / `ph_matvec_w` in phases.hpp block for block: same unit decomposition,
// same K order, same one exact int32 per 128-wide block, same `fmaf(acc, wscale * xscale, y)` fold
// in the same wave order. Three things differ and all three are the coordinate:
//
//   1. a block run is Q4_TILE_BLOCK_BYTES, and a lane loads 64 bytes instead of 128;
//   2. the operand words come out of the packed nibbles with an AND and a shifted AND instead of
//      being the loaded bytes themselves;
//   3. the weight codes are offset binary, so the block's int32 gets 8 * xsum subtracted where the
//      ternary path subtracts xsum and the Q8 path subtracts nothing.
//
// They live outside phases.hpp because that file is the deployed target model's hot loop and three
// engineers are usually inside it. When Q4 settles, the tidy form is one `WFMT` template parameter
// on the existing bodies rather than this copy; see docs/drafter-q4.md.
#pragma once
#include "q4_format.h"

namespace halo {

// Where the scalar-fed dot4 body hands over to WMMA for Q4 tiles. MV_DOT4_MAX_Q8 was fitted on the
// Q8 drafter; a Q4 block is half the bytes with an unpack, so the crossover is its own question.
#ifndef MV_DOT4_MAX_Q4
#define MV_DOT4_MAX_Q4 4
#endif

// One lane's 64 packed bytes -> the 32 operand words the dot bodies consume, tr[m] carrying the
// four codes of K 4m..4m+3 in byte order. Two VALU per four weights, amortised over every
// activation row of the block.
__device__ __forceinline__ void q4_expand(const uint4 (&qa)[4], unsigned (&tr)[32]) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const unsigned dw[4] = { qa[i].x, qa[i].y, qa[i].z, qa[i].w };
        #pragma unroll
        for (int m = 0; m < 4; m++) {
            tr[8 * i + 2 * m]     = dw[m] & 0x0f0f0f0fu;
            tr[8 * i + 2 * m + 1] = (dw[m] >> 4) & 0x0f0f0f0fu;
        }
    }
}

// dot of one lane's weight row over NBLK blocks against TT activation rows
template <int NBLK, int TT>
__device__ __forceinline__ void mv_rows_q4(const uint8_t * __restrict__ run, int lane, cptr xq, int xstride,
                                           cfptr xs, cptr xsum, int xsstride, int nrows, float y[TT]) {
    uint4 qa[4]; unsigned short tail;
    #pragma unroll
    for (int i = 0; i < 4; i++) qa[i] = *(const uint4 *) (run + q4_off_row(lane) + i * 16);
    tail = *(const unsigned short *) (run + q4_off_scale(lane));
    #pragma unroll 1
    for (int b = 0; b < NBLK; b++) {
        uint4 nqa[4]; unsigned short ntail = tail;
        #pragma unroll
        for (int i = 0; i < 4; i++) nqa[i] = qa[i];
        if (b + 1 < NBLK) {
            const uint8_t * nrun = run + (b + 1) * Q4_TILE_BLOCK_BYTES;
            #pragma unroll
            for (int i = 0; i < 4; i++) nqa[i] = *(const uint4 *) (nrun + q4_off_row(lane) + i * 16);
            ntail = *(const unsigned short *) (nrun + q4_off_scale(lane));
        }
        unsigned tr[32];
        q4_expand(qa, tr);
        const float wscale = __half2float(__ushort_as_half(tail));
        // rows: the next row's activations are requested (scalar cache) before the current row's dot
        // products, exactly as mv_rows_t does; the barrier keeps the compiler from hoisting them all
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
                    int acc = 0;
                    #pragma unroll
                    for (int i = 0; i < 32; i++) acc = __builtin_amdgcn_sudot4(false, (int) tr[i], true, xa[i], acc, false);
                    acc -= Q4_ZERO * xsum[r * xsstride + b];
                    y[r] = fmaf((float) acc, wscale * xs[r * xsstride + b], y[r]);
                }
                if (TT > 1) __builtin_amdgcn_sched_barrier(0);
                #pragma unroll
                for (int i = 0; i < 32; i++) xa[i] = xn[i];
            }
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) qa[i] = nqa[i];
        tail = ntail;
    }
}

// WMMA form; the A fragment is unsigned because the codes are offset binary.
template <int NBLK, int COLS = 8>
__device__ __forceinline__ void mvw_rows_q4(const uint8_t * __restrict__ run, int lane, const int8_t * __restrict__ xrow,
                                            const float * xs, const int * xsum, int xsstride, float * scl, float y1[8], float y2[8]) {
    const int half = lane >> 4, col = lane & 15;
    uint4 qa[4]; unsigned short tail;
    #pragma unroll
    for (int i = 0; i < 4; i++) qa[i] = *(const uint4 *) (run + q4_off_row(lane) + i * 16);
    tail = *(const unsigned short *) (run + q4_off_scale(lane));
    #pragma unroll 1
    for (int b = 0; b < NBLK; b++) {
        uint4 nqa[4]; unsigned short ntail = tail;
        #pragma unroll
        for (int i = 0; i < 4; i++) nqa[i] = qa[i];
        if (b + 1 < NBLK) {
            const uint8_t * nrun = run + (b + 1) * Q4_TILE_BLOCK_BYTES;
            #pragma unroll
            for (int i = 0; i < 4; i++) nqa[i] = *(const uint4 *) (nrun + q4_off_row(lane) + i * 16);
            ntail = *(const unsigned short *) (nrun + q4_off_scale(lane));
        }
        unsigned tr[32];
        q4_expand(qa, tr);
        const float wscale = __half2float(__ushort_as_half(tail));
        // scale table: [0][m] = even row 2m, [1][m] = odd row 2m+1
        scl[(lane & 1) * 16 + (lane >> 1)] = wscale;
        v8i C1 = {0, 0, 0, 0, 0, 0, 0, 0}, C2 = {0, 0, 0, 0, 0, 0, 0, 0};
        const int8_t * xb = xrow + b * 128;
        #pragma unroll
        for (int kb = 0; kb < 8; kb++) {
            const v4i A = { (int) tr[4 * kb], (int) tr[4 * kb + 1], (int) tr[4 * kb + 2], (int) tr[4 * kb + 3] };
            const v4i As = { swap16(A.x), swap16(A.y), swap16(A.z), swap16(A.w) };
            const int4 bx = *(const int4 *) (xb + kb * 16);
            const v4i B = { bx.x, bx.y, bx.z, bx.w };
            C1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, A, true, B, C1, false);
            C2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, As, true, B, C2, false);
        }
        const int xsc = Q4_ZERO * xsum[(col & (COLS - 1)) * xsstride + b];
        const float xsb = xs[(col & (COLS - 1)) * xsstride + b];
        const float4 * s1p = (const float4 *) (scl + half * 16 + 8 * half);
        const float4 * s2p = (const float4 *) (scl + half * 16 + 8 * (half ^ 1));
        const float4 s1a = s1p[0], s1b = s1p[1], s2a = s2p[0], s2b = s2p[1];
        const float s1[8] = { s1a.x, s1a.y, s1a.z, s1a.w, s1b.x, s1b.y, s1b.z, s1b.w }, s2[8] = { s2a.x, s2a.y, s2a.z, s2a.w, s2b.x, s2b.y, s2b.z, s2b.w };
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            y1[r] = fmaf((float) (C1[r] - xsc), s1[r] * xsb, y1[r]);
            y2[r] = fmaf((float) (C2[r] - xsc), s2[r] * xsb, y2[r]);
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) qa[i] = nqa[i];
        tail = ntail;
    }
}

template <int NB, int KS, int TT>
__device__ __forceinline__ void ph_matvec_q4(Ctx & c, const MvR & m, const int8_t * xq_g, const float * xs, const int * xsum, int nrows) {
    using G = Geom<NB, KS>;
    const int lane = threadIdx.x & 31, wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    const int total = m.total_tiles * KS;
    cptr xq = xptr(xq_g);
    cfptr xsc = sptr(xs); cptr xsumc = iptr(xsum);
    float * red = c.lds; // [8][TT][32]
    for (int u = blockIdx.x; u >= 0 && u < total; u = next_unit(c, total)) {
        c.dirty = true;
        const int part = u / m.total_tiles;
        int tile = u - part * m.total_tiles;
        MvSegR seg = m.seg[0];
        if (m.nseg > 1 && tile >= seg.ntiles) { tile -= seg.ntiles; seg = m.seg[1]; if (m.nseg > 2 && tile >= seg.ntiles) { tile -= seg.ntiles; seg = m.seg[2]; } }
        const int pw = (part == 0 || !G::SPECIAL) ? G::PW_A : G::PW_B;
        const int wb = (part == 0 ? 0 : (G::SPECIAL ? G::A_BLOCKS : part * G::PART)) + wave * pw;
        const uint8_t * run = seg.w + ((size_t) tile * NB + wb) * Q4_TILE_BLOCK_BYTES;
        float y[TT];
        #pragma unroll
        for (int r = 0; r < TT; r++) y[r] = 0.0f;
        if (part == 0 || !G::SPECIAL) mv_rows_q4<G::PW_A, TT>(run, lane, xq + wb * 32, NB * 32, xsc + wb, xsumc + wb, NB, nrows, y);
        else                          mv_rows_q4<G::PW_B, TT>(run, lane, xq + wb * 32, NB * 32, xsc + wb, xsumc + wb, NB, nrows, y);
        if (wave > 0) {
            #pragma unroll
            for (int r = 0; r < TT; r++) red[(wave * TT + r) * 32 + lane] = y[r];
        }
        __syncthreads();
        if (wave == 0) {
            const int N = seg.ntiles * 32;
            #pragma unroll
            for (int r = 0; r < TT; r++) {
                if (r < nrows) {
                    float v = y[r];
                    #pragma unroll
                    for (int w = 1; w < NW; w++) v += red[(w * TT + r) * 32 + lane];
                    float * out = seg.out + (size_t) r * N + (size_t) tile * 32 + lane;
                    if (KS > 1) atomicAdd(out, v);
                    else if (seg.add) *out += v;
                    else *out = v;
                }
            }
        }
        __syncthreads();
    }
    end_phase(c);
}

template <int NB, int KS, int COLS = 8>
__device__ __forceinline__ void ph_matvec_w_q4(Ctx & c, const MvR & m, const int8_t * xq, const float * xs, const int * xsum, int nrows) {
    using G = Geom<NB, KS>;
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, col = lane & 15, half = lane >> 4;
    const int total = m.total_tiles * KS;
    float * red = c.lds;                 // [7 waves][8][32], used twice
    float * scl = c.lds + (NW - 1) * 8 * 32 + wave * 32; // per-wave scale table
    for (int u = blockIdx.x; u >= 0 && u < total; u = next_unit(c, total)) {
        c.dirty = true;
        const int part = u / m.total_tiles;
        int tile = u - part * m.total_tiles;
        MvSegR seg = m.seg[0];
        if (m.nseg > 1 && tile >= seg.ntiles) { tile -= seg.ntiles; seg = m.seg[1]; if (m.nseg > 2 && tile >= seg.ntiles) { tile -= seg.ntiles; seg = m.seg[2]; } }
        const int pw = (part == 0 || !G::SPECIAL) ? G::PW_A : G::PW_B;
        const int wb = (part == 0 ? 0 : (G::SPECIAL ? G::A_BLOCKS : part * G::PART)) + wave * pw;
        const uint8_t * run = seg.w + ((size_t) tile * NB + wb) * Q4_TILE_BLOCK_BYTES;
        const int8_t * xrow = xq + (size_t) (col & (COLS - 1)) * NB * 128 + wb * 128;
        float y1[8], y2[8];
        #pragma unroll
        for (int r = 0; r < 8; r++) { y1[r] = 0.0f; y2[r] = 0.0f; }
        if (part == 0 || !G::SPECIAL) mvw_rows_q4<G::PW_A, COLS>(run, lane, xrow, xs + wb, xsum + wb, NB, scl, y1, y2);
        else                          mvw_rows_q4<G::PW_B, COLS>(run, lane, xrow, xs + wb, xsum + wb, NB, scl, y1, y2);
        const int N = seg.ntiles * 32;
        #pragma unroll
        for (int pass = 0; pass < 2; pass++) {
            float * y = pass ? y2 : y1;
            if (wave > 0) {
                #pragma unroll
                for (int r = 0; r < 8; r++) red[((wave - 1) * 8 + r) * 32 + lane] = y[r];
            }
            __syncthreads();
            if (wave == 0 && col < nrows) {
                #pragma unroll
                for (int r = 0; r < 8; r++) {
                    float v = y[r];
                    #pragma unroll
                    for (int w = 1; w < NW; w++) v += red[((w - 1) * 8 + r) * 32 + lane];
                    const int row = pass ? (half ? 2 * r + 1 : 2 * r + 16) : (half ? 2 * r + 17 : 2 * r);
                    float * o = seg.out + (size_t) col * N + (size_t) tile * 32 + row;
                    if (KS > 1) atomicAdd(o, v);
                    else if (seg.add) *o += v;
                    else *o = v;
                }
            }
            __syncthreads();
        }
    }
    end_phase(c);
}

template <int NB, int KS, int TT>
__device__ __forceinline__ void ph_matvec_auto_q4(Ctx & c, const MvR & m, const int8_t * xq, const float * xs, const int * xsum, int nrows) {
    if (TT <= MV_DOT4_MAX_Q4) ph_matvec_q4<NB, KS, TT>(c, m, xq, xs, xsum, nrows);
    else                      ph_matvec_w_q4<NB, KS, (TT > 8 ? 16 : 8)>(c, m, xq, xs, xsum, nrows);
}

// The drafter's weight matvecs, by coordinate. Q8 keeps the deployed dispatch untouched.
template <int NB, int KS, int TT, bool Q4>
__device__ __forceinline__ void ph_matvec_draft(Ctx & c, const MvR & m, const int8_t * xq, const float * xs, const int * xsum, int nrows) {
    if constexpr (Q4) ph_matvec_auto_q4<NB, KS, TT>(c, m, xq, xs, xsum, nrows);
    else              ph_matvec_auto<NB, KS, TT, true>(c, m, xq, xs, xsum, nrows);
}

} // namespace halo

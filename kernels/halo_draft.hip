// Drafter kernels: the Qwen3.8 MTP head and the DFlash2 block drafter, both persistent cooperative
// kernels built from the shared phase machinery. Drafter weights are Q8 or (DFlash2 only) Q4 tiles;
// the LM head is the target's ternary head and never moves.
//
// The drafter is the one place in this engine where a coarser weight coordinate is free of the
// output contract: the target verifies every drafted token, so the drafted greedy stream is
// identical whatever the drafter computes, and only the accepted count can move.
// docs/drafter-q4.md measures that count.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include "halo_kernels.h"
#include "device.hpp"
#include "phases.hpp"
#include "q4_tiles.hpp"

namespace halo {

namespace {

constexpr int NB_CAT = 2 * D / BLOCK; // 80: eh_proj input

#define SYNC() do { grid_sync(c); if (P.debug_stop && (int) c.gen >= P.debug_stop) return; } while (0)

// One MTP pass over nrows rows: row i pairs hidden h_in[i] with token rows[i].token at position rows[i].pos.
template <int TT>
__global__ void __launch_bounds__(NT) k_mtp(MtpParams M) {
    __shared__ __attribute__((aligned(16))) float lds[LDS_FLOATS];
    __shared__ float red[8];
    __shared__ int s_unit;
    const FwdParams & P = M.P;
    const MtpW & w = M.w;
    Ctx c { P.bar, P.work, P.prof, 0, 0, true, lds, red, &s_unit };
    const int nrows = P.nrows;
    if (P.prof && blockIdx.x == 0 && threadIdx.x == 0) P.prof[0] = wall_clock64();

    // token embedding (Hadamard-restored) -> P.tmp rows
    if ((int) blockIdx.x < nrows) embed_row(P.tok_embd, P.rows[blockIdx.x].token, P.x + (size_t) blockIdx.x * D);
    SYNC();
    { PrepR p{}; p.x = P.x; p.kvec = P.signs5120; p.n = D; p.flags = PREP_HADAMARD | PREP_SIGN_AFTER | PREP_STORE_F32; p.out_f32 = P.tmp; ph_prep(c, p, nrows); }
    SYNC();
    // [enorm(e) | hnorm(h)] -> int8 rows of width 2D
    { PrepR p{}; p.x = P.tmp; p.kvec = w.enorm; p.eps = NORM_EPS; p.n = D; p.flags = PREP_NORM | PREP_QUANT; p.xq = P.xq; p.xs = P.xs; p.xsum = P.xsum; p.xq_stride = 2 * D; p.xq_off = 0; ph_prep(c, p, nrows); }
    { PrepR p{}; p.x = M.h_in; p.kvec = w.hnorm; p.eps = NORM_EPS; p.n = D; p.flags = PREP_NORM | PREP_QUANT; p.xq = P.xq; p.xs = P.xs; p.xsum = P.xsum; p.xq_stride = 2 * D; p.xq_off = D; ph_prep(c, p, nrows); }
    zero_floats(c, P.x, (size_t) nrows * D);
    SYNC();
    { MvR m{}; m.nseg = 1; m.seg[0] = { w.fc, P.x, D / 32, 0 }; m.total_tiles = D / 32; ph_matvec_auto<NB_CAT, 2, TT, true>(c, m, P.xq, P.xs, P.xsum, nrows); }
    SYNC();
    // decoder block: full attention with q gate, then SwiGLU
    { PrepR p{}; p.x = P.x; p.kvec = w.in_norm; p.eps = NORM_EPS; p.n = D; p.flags = PREP_NORM | PREP_QUANT; p.xq = P.xq; p.xs = P.xs; p.xsum = P.xsum; ph_prep(c, p, nrows); }
    SYNC();
    { MvR m{}; m.nseg = 3; m.seg[0] = { w.q, P.qfull, Q_OUT / 32, 0 }; m.seg[1] = { w.k, P.kbuf, KV_OUT / 32, 0 }; m.seg[2] = { w.v, P.vbuf, KV_OUT / 32, 0 };
      m.total_tiles = (Q_OUT + 2 * KV_OUT) / 32; ph_matvec_auto<NB_D, 1, TT, true>(c, m, P.xq, P.xs, P.xsum, nrows); }
    SYNC();
    ph_attn_pre(c, P, w.q_norm, w.k_norm, NATTN);
    SYNC();
    ph_attn<TT, NH, NKV, HD>(c, P, NATTN);
    SYNC();
    { PrepR p{}; p.x2 = P.qfull; p.partials = P.partials; p.rows = P.rows; p.n = ATTN_OUT; p.flags = PREP_ATTN_COMBINE | PREP_QUANT; p.xq = P.xq; p.xs = P.xs; p.xsum = P.xsum; ph_prep(c, p, nrows); }
    SYNC();
    { MvR m{}; m.nseg = 1; m.seg[0] = { w.o, P.x, D / 32, 1 }; m.total_tiles = D / 32; ph_matvec_auto<NB_V, 2, TT, true>(c, m, P.xq, P.xs, P.xsum, nrows); }
    SYNC();
    { PrepR p{}; p.x = P.x; p.kvec = w.post_norm; p.eps = NORM_EPS; p.n = D; p.flags = PREP_NORM | PREP_QUANT; p.xq = P.xq; p.xs = P.xs; p.xsum = P.xsum; ph_prep(c, p, nrows); }
    SYNC();
    { MvR m{}; m.nseg = 2; m.seg[0] = { w.gate, P.gu, FF / 32, 0 }; m.seg[1] = { w.up, P.gu + (size_t) FMAX * FF, FF / 32, 0 }; m.total_tiles = 2 * FF / 32;
      ph_matvec_auto<NB_D, 1, TT, true>(c, m, P.xq, P.xs, P.xsum, nrows); }
    SYNC();
    { PrepR p{}; p.x = P.gu; p.x2 = P.gu + (size_t) FMAX * FF; p.n = FF; p.flags = PREP_SILU_MUL | PREP_QUANT; p.xq = P.xq; p.xs = P.xs; p.xsum = P.xsum; ph_prep(c, p, nrows); }
    SYNC();
    { MvR m{}; m.nseg = 1; m.seg[0] = { w.down, P.x, D / 32, 1 }; m.total_tiles = D / 32; ph_matvec_auto<NB_FF, 2, TT, true>(c, m, P.xq, P.xs, P.xsum, nrows); }
    SYNC();
    // final norm: the normed hidden is the next chain input and the LM head input (sign, Hadamard)
    { PrepR p{}; p.x = P.x; p.kvec = w.out_norm_s; p.norm_w = w.out_norm; p.eps = NORM_EPS; p.n = D; p.flags = PREP_NORM | PREP_HADAMARD | PREP_QUANT;
      p.xq = P.xq; p.xs = P.xs; p.xsum = P.xsum; p.store_normed = M.h_out; ph_prep(c, p, nrows); }
    SYNC();
    { MvR m{}; m.nseg = 1; m.seg[0] = { P.output, P.logits, VOCAB / 32, 0 }; m.total_tiles = VOCAB / 32; ph_matvec_auto<NB_D, 1, TT, false>(c, m, P.xq, P.xs, P.xsum, nrows); }
    SYNC();
    ph_argmax_slices(c, P);
    SYNC();
    if ((int) blockIdx.x < nrows) argmax_final(P.amax_val + blockIdx.x * AMAX_SLICES, P.amax_idx + blockIdx.x * AMAX_SLICES, P.argmax_out + blockIdx.x, c.lds);
}

// ---------------------------------------------------------------------------------------------
// DFlash2

constexpr int NB_5D = NCAP * D / BLOCK;  // 200
constexpr int NB_DQ = DF_Q / BLOCK;      // 32
constexpr int DF_GROUPS = D / 16;        // 320
static_assert(LDS_FLOATS >= 2 * DF_TOPCAP + 16, "top-k scratch must fit");

// Grouped dynamic causal conv over the block rows (kernel size 2, group 16):
// out[t][e] = sum_off (base[off][e] + dyn[t][which*640 + off*320 + e/16]) * src[t-off][e], src[-1] = 0.
// mode 0: store to out; mode 1: h[t][e] += value.
__device__ __forceinline__ void ph_dconv(Ctx & c, const float * src, const float * dyn, int which, const float * base, float * out, float * h, int nrows) {
    const int total = nrows * (D / 1024);
    for (int u = blockIdx.x; u >= 0 && u < total; u = next_unit(c, total)) {
        c.dirty = true;
        const int row = u / (D / 1024), chunk = u - row * (D / 1024);
        const int e = chunk * 1024 + threadIdx.x * 4, g = e >> 4;
        const float4 x0 = *(const float4 *) (src + (size_t) row * D + e);
        const float4 x1 = row > 0 ? *(const float4 *) (src + (size_t) (row - 1) * D + e) : make_float4(0.f, 0.f, 0.f, 0.f);
        const float d0 = dyn[(size_t) row * DF_KPROJ + which * 2 * DF_GROUPS + g];
        const float d1 = dyn[(size_t) row * DF_KPROJ + which * 2 * DF_GROUPS + DF_GROUPS + g];
        const float4 b0 = *(const float4 *) (base + e), b1 = *(const float4 *) (base + D + e);
        float4 v;
        v.x = (b0.x + d0) * x0.x + (b1.x + d1) * x1.x; v.y = (b0.y + d0) * x0.y + (b1.y + d1) * x1.y;
        v.z = (b0.z + d0) * x0.z + (b1.z + d1) * x1.z; v.w = (b0.w + d0) * x0.w + (b1.w + d1) * x1.w;
        if (h) { float4 * hp = (float4 *) (h + (size_t) row * D + e); float4 t = *hp; t.x += v.x; t.y += v.y; t.z += v.z; t.w += v.w; *hp = t; }
        else *(float4 *) (out + (size_t) row * D + e) = v;
    }
    end_phase(c);
}

// q/k norm + full-dim RoPE (theta 1e7, pairs (i, i+64)) for the drafter's 128-dim heads; K/V appended
// to the cache slot at the row's position. With do_q == 0 only K/V of `rows` are produced (ingest).
__device__ __forceinline__ void ph_dattn_pre(Ctx & c, const FwdParams & P, const RowInfo * rows, int nrows, const float * q_norm, const float * k_norm, int kv_slot, int do_q) {
    const int per_row = (do_q ? DF_NH : 0) + DF_NKV;
    const int total = nrows * per_row;
    const int d = threadIdx.x;
    for (int u = blockIdx.x; u >= 0 && u < total; u = next_unit(c, total)) {
        c.dirty = true;
        const int row = u / per_row, hh = u - row * per_row;
        const RowInfo R = rows[row];
        const bool is_q = do_q && hh < DF_NH;
        const int hk = is_q ? 0 : hh - (do_q ? DF_NH : 0);
        float v = 0.0f;
        if (d < DF_HD) v = is_q ? P.qfull[(size_t) row * DF_Q + hh * DF_HD + d] : P.kbuf[(size_t) row * DF_KV + hk * DF_HD + d];
        const float ss = block_sum_256(v * v, c.red);
        v = v * rsqrtf(ss / (float) DF_HD + NORM_EPS) * (d < DF_HD ? (is_q ? q_norm[d] : k_norm[d]) : 0.0f);
        c.lds[d] = v;
        __syncthreads();
        if (d < DF_HD) {
            const int i = d < DF_HD / 2 ? d : d - DF_HD / 2;
            const float theta = (float) R.pos * __powf(1e7f, -(float) i / (float) (DF_HD / 2));
            float sn, cs; __sincosf(theta, &sn, &cs);
            const float lo = c.lds[i], hi = c.lds[i + DF_HD / 2];
            const float outv = d < DF_HD / 2 ? lo * cs - hi * sn : lo * sn + hi * cs;
            if (is_q) {
                P.qrot[(size_t) row * DF_Q + hh * DF_HD + d] = outv * 0.08838834764831845f; // 1/sqrt(128)
            } else {
                __half * kc = P.kcache + (size_t) (R.seq * KV_SLOTS + kv_slot) * ((size_t) NKV * P.context * HD) + (size_t) hk * P.context * DF_HD;
                __half * vc = P.vcache + (size_t) (R.seq * KV_SLOTS + kv_slot) * ((size_t) NKV * P.context * HD) + (size_t) hk * P.context * DF_HD;
                kc[(size_t) R.pos * DF_HD + d] = __float2half(outv);
                vc[(size_t) R.pos * DF_HD + d] = __float2half(P.vbuf[(size_t) row * DF_KV + hk * DF_HD + d]);
            }
        }
        __syncthreads();
    }
    end_phase(c);
}

// ---- top-k (16) per logits row by threshold: slice maxima -> 16th largest as a bound -> collect -> select
// repeated block argmax over v[0..n) in LDS; removes each winner. scratch: 16 free LDS floats.
__device__ __forceinline__ void extract_top(float * v, int * ix, int n, int k, float * outv, int * outi, float * scratch) {
    for (int r = 0; r < k; r++) {
        float best = -INFINITY; int bi = -1;
        for (int i = threadIdx.x; i < n; i += NT) if (v[i] > best || (v[i] == best && ix[i] < bi)) { best = v[i]; bi = ix[i]; }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) { float ob = __shfl_xor(best, o, 32); int oi = __shfl_xor(bi, o, 32); if (ob > best || (ob == best && oi < bi)) { best = ob; bi = oi; } }
        __syncthreads();
        float * sm = scratch; int * si = (int *) (scratch + 8);
        if ((threadIdx.x & 31) == 0) { sm[threadIdx.x >> 5] = best; si[threadIdx.x >> 5] = bi; }
        __syncthreads();
        best = sm[0]; bi = si[0];
        for (int w = 1; w < NW; w++) if (sm[w] > best || (sm[w] == best && si[w] < bi)) { best = sm[w]; bi = si[w]; }
        if (threadIdx.x == 0) { outv[r] = best; outi[r] = bi; }
        for (int i = threadIdx.x; i < n; i += NT) if (ix[i] == bi) v[i] = -INFINITY;
        __syncthreads();
    }
}

__device__ __forceinline__ void ph_slice_max(Ctx & c, const float * logits, int row0, int nrows, float * vals) {
    const int total = nrows * AMAX_SLICES;
    for (int u = blockIdx.x; u >= 0 && u < total; u = next_unit(c, total)) {
        c.dirty = true;
        const int row = row0 + u / AMAX_SLICES, sl = u % AMAX_SLICES;
        const float * lg = logits + (size_t) row * VOCAB;
        float m = -INFINITY;
        for (int i = sl * 1024 + threadIdx.x; i < min((sl + 1) * 1024, VOCAB); i += NT) m = fmaxf(m, lg[i]);
        m = block_max_256(m, c.red);
        if (threadIdx.x == 0) vals[row * AMAX_SLICES + sl] = m;
    }
    end_phase(c);
}
__device__ __forceinline__ void ph_threshold(Ctx & c, const float * vals, int row0, int nrows, float * thr, int * cand_n) {
    for (int u = blockIdx.x; u >= 0 && u < nrows; u = next_unit(c, nrows)) {
        c.dirty = true;
        const int row = row0 + u;
        float * v = c.lds; int * ix = (int *) (c.lds + AMAX_SLICES);
        for (int i = threadIdx.x; i < AMAX_SLICES; i += NT) { v[i] = vals[row * AMAX_SLICES + i]; ix[i] = i; }
        __syncthreads();
        float ov[DF_TOPK]; int oi[DF_TOPK];
        extract_top(v, ix, AMAX_SLICES, DF_TOPK, c.lds + 2 * AMAX_SLICES, (int *) (c.lds + 2 * AMAX_SLICES + DF_TOPK), c.lds + 2 * AMAX_SLICES + 2 * DF_TOPK);
        if (threadIdx.x == 0) { thr[row] = c.lds[2 * AMAX_SLICES + DF_TOPK - 1]; cand_n[row] = 0; }
        (void) ov; (void) oi;
        __syncthreads();
    }
    end_phase(c);
}
__device__ __forceinline__ void ph_collect(Ctx & c, const float * logits, int row0, int nrows, const float * thr, float * cand_v, int * cand_i, int * cand_n) {
    const int total = nrows * AMAX_SLICES;
    for (int u = blockIdx.x; u >= 0 && u < total; u = next_unit(c, total)) {
        c.dirty = true;
        const int row = row0 + u / AMAX_SLICES, sl = u % AMAX_SLICES;
        const float * lg = logits + (size_t) row * VOCAB;
        const float t = thr[row];
        for (int i = sl * 1024 + threadIdx.x; i < min((sl + 1) * 1024, VOCAB); i += NT) {
            const float v = lg[i];
            if (v >= t) { const int k = atomicAdd(cand_n + row, 1); if (k < DF_TOPCAP) { cand_v[(size_t) row * DF_TOPCAP + k] = v; cand_i[(size_t) row * DF_TOPCAP + k] = i; } }
        }
    }
    end_phase(c);
}
__device__ __forceinline__ void ph_select_top(Ctx & c, int row0, int nrows, const float * cand_v, const int * cand_i, const int * cand_n, float * topv, int * topi) {
    for (int u = blockIdx.x; u >= 0 && u < nrows; u = next_unit(c, nrows)) {
        c.dirty = true;
        const int row = row0 + u;
        const int n = min(cand_n[row], DF_TOPCAP);
        float * v = c.lds; int * ix = (int *) (c.lds + DF_TOPCAP);
        for (int i = threadIdx.x; i < n; i += NT) { v[i] = cand_v[(size_t) row * DF_TOPCAP + i]; ix[i] = cand_i[(size_t) row * DF_TOPCAP + i]; }
        __syncthreads();
        extract_top(v, ix, n, DF_TOPK, topv + row * DF_TOPK, topi + row * DF_TOPK, c.lds + 2 * DF_TOPCAP);
        __syncthreads();
    }
    end_phase(c);
}

// Candidate selector, one workgroup: chain through block positions 1..7 with the low-rank pairwise term.
__device__ __forceinline__ void selector(const DflashParams & M, float * lds, float * red) {
    __shared__ int prev;
    __shared__ float part[NW][DF_TOPK];
    const int t = threadIdx.x, lane = t & 31, wave = t >> 5;
    if (t == 0) prev = M.anchor;
    __syncthreads();
    for (int pos = 1; pos < DF_BLOCK; pos++) {
        const int pred = prev;
        const float pr = bf16_to_f32(M.w.pred_cb[(size_t) pred * DF_RANK + t]) * M.hproj_out[(size_t) pos * DF_RANK + t];
        float sc[DF_TOPK];
        #pragma unroll
        for (int k = 0; k < DF_TOPK; k++) {
            const int cand = M.topi[pos * DF_TOPK + k];
            float s = pr * bf16_to_f32(M.w.succ_cb[(size_t) cand * DF_RANK + t]);
            s = warp_sum(s);
            sc[k] = s;
        }
        if (lane == 0) {
            #pragma unroll
            for (int k = 0; k < DF_TOPK; k++) part[wave][k] = sc[k];
        }
        __syncthreads();
        if (t == 0) {
            float best = -INFINITY; int bk = 0;
            for (int k = 0; k < DF_TOPK; k++) {
                float s = M.topv[pos * DF_TOPK + k];
                for (int w = 0; w < NW; w++) s += part[w][k];
                if (s > best) { best = s; bk = k; }
            }
            prev = M.topi[pos * DF_TOPK + bk];
            M.draft_out[pos - 1] = prev;
        }
        __syncthreads();
    }
    (void) lds; (void) red;
}

#undef SYNC
#define SYNC() do { grid_sync(c); if (P.debug_stop && (int) c.gen >= P.debug_stop) return; } while (0)

// TC: ingest rows instantiation (context rows), block is always DF_BLOCK rows.
// Q4: the drafter's weight coordinate. Everything else in the body - phases, order, activation
// quantiser, attention, the ternary LM head - is identical between the two instantiations.
// This body holds 239 VGPRs at six waves per SIMD32, three workgroups per WGP and a grid of 60,
// and MORE WAVES ARE MEASURED NOT TO HELP IT: `amdgpu_waves_per_eu(8)` fits it in 192 registers
// for three spilled VGPRs and gets the fourth workgroup, and the drafted step does not move
// (6.471 ms per step at 60 against 6.483 at 80). Its time is flat from 40 workgroups upward and
// only 20 costs anything. Do not re-derive it from docs/matvec-crossover.md's 21% grid range on
// the eight-row target pass - that pass is a different shape. docs/drafter-occupancy.md.
template <int TC, bool Q4>
__global__ void __launch_bounds__(NT) k_dflash(DflashParams M) {
    __shared__ __attribute__((aligned(16))) float lds[LDS_FLOATS];
    __shared__ float red[8];
    __shared__ int s_unit;
    const FwdParams & P = M.P;
    const DflashW & w = M.w;
    Ctx c { P.bar, P.work, P.prof, 0, 0, true, lds, red, &s_unit };
    if (P.prof && blockIdx.x == 0 && threadIdx.x == 0) P.prof[0] = wall_clock64();
    constexpr int TB = DF_BLOCK;

    // ---- ingest: captured target features -> fc -> hidden_norm -> per-layer K/V into the drafter cache
    if (M.n_ctx > 0) {
        const int nc = M.n_ctx;
        { PrepR p{}; p.x = M.hcap; p.n = NCAP * D; p.flags = PREP_QUANT; p.xq = P.xq; p.xs = P.xs; p.xsum = P.xsum; ph_prep(c, p, nc); }
        zero_floats(c, M.th, (size_t) nc * D);
        SYNC();
        { MvR m{}; m.nseg = 1; m.seg[0] = { w.fc, M.th, D / 32, 0 }; m.total_tiles = D / 32; ph_matvec_draft<NB_5D, 5, TC, Q4>(c, m, P.xq, P.xs, P.xsum, nc); }
        SYNC();
        // th = rms(fc out) * hidden_norm, stored f32 and quantised for the K/V projections
        { PrepR p{}; p.x = M.th; p.kvec = w.hidden_norm; p.eps = NORM_EPS; p.n = D; p.flags = PREP_NORM | PREP_STORE_F32 | PREP_QUANT; p.out_f32 = M.th; p.xq = P.xq; p.xs = P.xs; p.xsum = P.xsum; ph_prep(c, p, nc); }
        SYNC();
        for (int l = 0; l < DF_LAYERS; l++) {
            const DflashLayerW & L = w.layer[l];
            { MvR m{}; m.nseg = 2; m.seg[0] = { L.k, P.kbuf, DF_KV / 32, 0 }; m.seg[1] = { L.v, P.vbuf, DF_KV / 32, 0 }; m.total_tiles = 2 * DF_KV / 32; ph_matvec_draft<NB_D, 1, TC, Q4>(c, m, P.xq, P.xs, P.xsum, nc); }
            SYNC();
            ph_dattn_pre(c, P, M.ctx_rows, nc, L.q_norm, L.k_norm, NATTN + 1 + l, 0);
            SYNC();
        }
    }
    if (!M.do_block) return;

    // ---- block draft: rows = [anchor, mask x 7] at positions start .. start + 7
    if ((int) blockIdx.x < TB) embed_row(P.tok_embd, P.rows[blockIdx.x].token, P.tmp + (size_t) blockIdx.x * D);
    SYNC();
    { PrepR p{}; p.x = P.tmp; p.kvec = P.signs5120; p.n = D; p.flags = PREP_HADAMARD | PREP_SIGN_AFTER | PREP_STORE_F32; p.out_f32 = P.x; ph_prep(c, p, TB); }
    SYNC();
    // keep the restored embeddings (th is free after the ingest) for diagnostics
    for (int i = blockIdx.x * NT + threadIdx.x; i < TB * D; i += gridDim.x * NT) M.th[i] = P.x[i];
    c.dirty = true;
    SYNC();
    const int nl = P.debug_layers > 0 ? P.debug_layers : DF_LAYERS;
    for (int l = 0; l < nl; l++) {
        const DflashLayerW & L = w.layer[l];
        // attention branch
        { PrepR p{}; p.x = P.x; p.kvec = L.in_norm; p.eps = NORM_EPS; p.n = D; p.flags = PREP_NORM | PREP_STORE_F32 | PREP_QUANT; p.out_f32 = M.xn; p.xq = P.xq; p.xs = P.xs; p.xsum = P.xsum; ph_prep(c, p, TB); }
        zero_floats(c, M.dyn, (size_t) TB * DF_KPROJ);
        SYNC();
        { MvR m{}; m.nseg = 1; m.seg[0] = { L.akp, M.dyn, DF_KPROJ / 32, 0 }; m.total_tiles = DF_KPROJ / 32; ph_matvec_draft<NB_D, 5, TB, Q4>(c, m, P.xq, P.xs, P.xsum, TB); }
        SYNC();
        ph_dconv(c, M.xn, M.dyn, 0, L.abase, M.cbuf, nullptr, TB);
        SYNC();
        { PrepR p{}; p.x = M.cbuf; p.n = D; p.flags = PREP_QUANT; p.xq = P.xq; p.xs = P.xs; p.xsum = P.xsum; ph_prep(c, p, TB); }
        SYNC();
        { MvR m{}; m.nseg = 3; m.seg[0] = { L.q, P.qfull, DF_Q / 32, 0 }; m.seg[1] = { L.k, P.kbuf, DF_KV / 32, 0 }; m.seg[2] = { L.v, P.vbuf, DF_KV / 32, 0 };
          m.total_tiles = (DF_Q + 2 * DF_KV) / 32; ph_matvec_draft<NB_D, 1, TB, Q4>(c, m, P.xq, P.xs, P.xsum, TB); }
        SYNC();
        ph_dattn_pre(c, P, P.rows, TB, L.q_norm, L.k_norm, NATTN + 1 + l, 1);
        SYNC();
        ph_attn<TB, DF_NH, DF_NKV, DF_HD>(c, P, NATTN + 1 + l);
        SYNC();
        // The drafter's layers are this engine's only windowed attention, so this is the only fold
        // that has to know it: `ph_attn` deals no unit for a chunk the window excludes, and
        // `comb_window` is what tells the fold not to read the partials those units never wrote.
        { PrepR p{}; p.partials = P.partials; p.rows = P.rows; p.n = DF_Q; p.comb_heads = DF_NH; p.comb_hd = DF_HD; p.comb_window = P.attn_window_deal ? P.attn_window : 0; p.flags = PREP_ATTN_COMBINE | PREP_QUANT; p.xq = P.xq; p.xs = P.xs; p.xsum = P.xsum; ph_prep(c, p, TB); }
        zero_floats(c, M.tmpo, (size_t) TB * D);
        SYNC();
        { MvR m{}; m.nseg = 1; m.seg[0] = { L.o, M.tmpo, D / 32, 0 }; m.total_tiles = D / 32; ph_matvec_draft<NB_DQ, 2, TB, Q4>(c, m, P.xq, P.xs, P.xsum, TB); }
        SYNC();
        ph_dconv(c, M.tmpo, M.dyn, 1, L.abase, nullptr, P.x, TB);
        SYNC();
        // mlp branch
        { PrepR p{}; p.x = P.x; p.kvec = L.post_norm; p.eps = NORM_EPS; p.n = D; p.flags = PREP_NORM | PREP_STORE_F32 | PREP_QUANT; p.out_f32 = M.xn; p.xq = P.xq; p.xs = P.xs; p.xsum = P.xsum; ph_prep(c, p, TB); }
        zero_floats(c, M.dyn, (size_t) TB * DF_KPROJ);
        SYNC();
        { MvR m{}; m.nseg = 1; m.seg[0] = { L.mkp, M.dyn, DF_KPROJ / 32, 0 }; m.total_tiles = DF_KPROJ / 32; ph_matvec_draft<NB_D, 5, TB, Q4>(c, m, P.xq, P.xs, P.xsum, TB); }
        SYNC();
        ph_dconv(c, M.xn, M.dyn, 0, L.mbase, M.cbuf, nullptr, TB);
        SYNC();
        { PrepR p{}; p.x = M.cbuf; p.n = D; p.flags = PREP_QUANT; p.xq = P.xq; p.xs = P.xs; p.xsum = P.xsum; ph_prep(c, p, TB); }
        SYNC();
        { MvR m{}; m.nseg = 2; m.seg[0] = { L.gate, P.gu, FF / 32, 0 }; m.seg[1] = { L.up, P.gu + (size_t) FMAX * FF, FF / 32, 0 }; m.total_tiles = 2 * FF / 32; ph_matvec_draft<NB_D, 1, TB, Q4>(c, m, P.xq, P.xs, P.xsum, TB); }
        SYNC();
        { PrepR p{}; p.x = P.gu; p.x2 = P.gu + (size_t) FMAX * FF; p.n = FF; p.flags = PREP_SILU_MUL | PREP_QUANT; p.xq = P.xq; p.xs = P.xs; p.xsum = P.xsum; ph_prep(c, p, TB); }
        zero_floats(c, M.tmpo, (size_t) TB * D);
        SYNC();
        { MvR m{}; m.nseg = 1; m.seg[0] = { L.down, M.tmpo, D / 32, 0 }; m.total_tiles = D / 32; ph_matvec_draft<NB_FF, 2, TB, Q4>(c, m, P.xq, P.xs, P.xsum, TB); }
        SYNC();
        ph_dconv(c, M.tmpo, M.dyn, 1, L.mbase, nullptr, P.x, TB);
        SYNC();
    }
    // final norm -> hidden projection (selector) and LM head (sign + Hadamard) for rows 1..7
    { PrepR p{}; p.x = P.x; p.kvec = w.norm; p.eps = NORM_EPS; p.n = D; p.flags = PREP_NORM | PREP_QUANT; p.xq = P.xq; p.xs = P.xs; p.xsum = P.xsum; ph_prep(c, p, TB); }
    SYNC();
    { MvR m{}; m.nseg = 1; m.seg[0] = { w.hproj, M.hproj_out, DF_RANK / 32, 0 }; m.total_tiles = DF_RANK / 32; ph_matvec_draft<NB_D, 1, TB, Q4>(c, m, P.xq, P.xs, P.xsum, TB); }
    SYNC();
    { PrepR p{}; p.x = P.x; p.kvec = w.norm_s; p.eps = NORM_EPS; p.n = D; p.flags = PREP_NORM | PREP_HADAMARD | PREP_QUANT; p.xq = P.xq; p.xs = P.xs; p.xsum = P.xsum; ph_prep(c, p, TB); }
    SYNC();
    { MvR m{}; m.nseg = 1; m.seg[0] = { P.output, P.logits, VOCAB / 32, 0 }; m.total_tiles = VOCAB / 32; ph_matvec_auto<NB_D, 1, TB, false>(c, m, P.xq, P.xs, P.xsum, TB); }
    SYNC();
    ph_slice_max(c, P.logits, 1, TB - 1, P.amax_val);
    SYNC();
    ph_threshold(c, P.amax_val, 1, TB - 1, M.thr, M.cand_n);
    SYNC();
    ph_collect(c, P.logits, 1, TB - 1, M.thr, M.cand_v, M.cand_i, M.cand_n);
    SYNC();
    ph_select_top(c, 1, TB - 1, M.cand_v, M.cand_i, M.cand_n, M.topv, M.topi);
    SYNC();
    if (blockIdx.x == 0) selector(M, lds, red);
    // The selector runs on one workgroup after the last barrier, so nothing closes it and it is the
    // only phase a timeline cannot see. `P.prof` is uniform over the grid and null on every
    // deployed step, so this barrier exists only while the drafter is being profiled.
    if (P.prof) SYNC();
}


int mtp_grid() { return std::min({ coop_grid((const void *) k_mtp<1>), coop_grid((const void *) k_mtp<4>), coop_grid((const void *) k_mtp<8>) }); }
// Each coordinate takes the minimum over its own row instantiations. The two arms have different
// register footprints, and a shared minimum would hand one arm's occupancy to the other - the trap
// docs/matvec-crossover.md found in the deployed launcher.
template <bool Q4>
int dflash_grid() { return std::min({ coop_grid((const void *) k_dflash<1, Q4>), coop_grid((const void *) k_dflash<4, Q4>), coop_grid((const void *) k_dflash<8, Q4>) }); }

} // namespace

// Workgroups for the drafter, as an in-process arm. The drafter's own occupancy is the ceiling and
// the default; a smaller value pins a point on the grid curve, which is how docs/drafter-occupancy.md
// measured that curve to be flat above 40. --draft-grid walks it, HALO_DRAFT_GRID pins it.
static int df_grid_request = -1;   // -1 = unread, 0 = the build's own choice

void set_dflash_grid(int g) { df_grid_request = g < 0 ? 0 : g; }

void launch_dflash(const DflashParams & p, hipStream_t st) {
    static int grid8 = dflash_grid<false>(), grid4 = dflash_grid<true>();
    if (df_grid_request < 0) { const char * e = getenv("HALO_DRAFT_GRID"); df_grid_request = e ? atoi(e) : 0; }
    const int ceiling = p.q4 ? grid4 : grid8;
    const int grid = df_grid_request > 0 && df_grid_request <= ceiling ? df_grid_request : ceiling;
    // HALO_DRAFT_GRID_PRINT=1 says which grid a panel actually ran, once per coordinate.
    { static int shown[2] = {}; int & s = shown[p.q4];
      if (!s && getenv("HALO_DRAFT_GRID_PRINT")) { s = 1;
        fprintf(stderr, "dflash: %d workgroups (ceilings q8 %d, q4 %d)\n", grid, grid8, grid4); } }
    void * args[] = { (void *) &p };
    const void * fn = p.q4 ? (p.n_ctx <= 1 ? (const void *) k_dflash<1, true> : p.n_ctx <= 4 ? (const void *) k_dflash<4, true> : (const void *) k_dflash<8, true>)
                           : (p.n_ctx <= 1 ? (const void *) k_dflash<1, false> : p.n_ctx <= 4 ? (const void *) k_dflash<4, false> : (const void *) k_dflash<8, false>);
    hipError_t e = hipLaunchCooperativeKernel(fn, dim3(grid), dim3(NT), args, 0, st);
    if (e != hipSuccess) { fprintf(stderr, "dflash launch failed: %s\n", hipGetErrorString(e)); abort(); }
}

void launch_mtp(const MtpParams & p, hipStream_t st) {
    static int grid = mtp_grid();
    void * args[] = { (void *) &p };
    const void * fn = p.P.nrows <= 1 ? (const void *) k_mtp<1> : p.P.nrows <= 4 ? (const void *) k_mtp<4> : (const void *) k_mtp<8>;
    hipError_t e = hipLaunchCooperativeKernel(fn, dim3(grid), dim3(NT), args, 0, st);
    if (e != hipSuccess) { fprintf(stderr, "mtp launch failed: %s\n", hipGetErrorString(e)); abort(); }
}

} // namespace halo

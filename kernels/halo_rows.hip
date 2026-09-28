// Multi-row persistent forward kernel: one cooperative launch processes up to RMAX rows, which are
// speculative-verification blocks of one sequence and/or single tokens of several sequences.
//
// GDN state rollback is by replay: the pass caches each row's conv output and gates, leaves the
// committed state untouched, and the next pass first commits the accepted prefix of the previous
// block (n_replay tokens) from that cache before processing its own rows.
//
// Ordinary prompt ingestion accepts every row and pays for that replay twice over. FwdParams::
// commit_state selects the direct-commit path (CM = 1 below): identical arithmetic, operand order
// and outputs, but the conv ring is stored after this pass's last row and the GDN state after its
// last token, so the next pass needs no replay. Incoming n_replay is still honoured, so a pass can
// commit an uncommitted predecessor and its own rows together.
//
// k_forward_rows also serves one layer in two halves (PART 1 and 2) so a wider projection can own
// that layer's matvecs; see launch_sequence_part in halo_kernels.h. PART 0 compiles the deployed
// body unchanged.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include "halo_kernels.h"
#include "device.hpp"
#include "phases.hpp"
#include "gdn_state_codec.hpp"
#include "prep_batch.h"

namespace halo {

__constant__ LayerW r_layers[NLAYER];
// The recurrent state's storage coordinate, uniform over every kernel in one process. It is a
// property of the bytes already in P.gdn_state rather than of a launch, so it is a constant the
// setter uploads between passes instead of a field each caller would have to thread through.
__constant__ int r_gdn_fmt = GDN_STATE_F32;
// How many steps of a stream may share one state write-back. 1 commits every step, which is the
// path the packed coordinate shipped with; above that the resident state kernel keeps each step's
// rank-1 update beside the state and rebuilds it at load. Uniform over a process for the same
// reason the format is: it describes bytes that outlive a pass.
__constant__ int r_gdn_defer = 1;
// Floats between one (slot, layer) state region and the next. It describes what the engine
// allocated rather than what a launch does, so it travels the same way the format does - one
// constant, uploaded once, before any pass reads a region. A packed coordinate needs a third of an
// fp32 region and the engine now allocates only that, which is what lets 128 sequence slots exist:
// see gdn_fmt_region_floats in halo_kernels.h and docs/state-region-stride.md.
__constant__ int r_gdn_region = GDN_STATE_FLOATS;
// Which slots may hold pending rank-1 updates. Host state, because the host is what has to decide
// whether a pass on a route that cannot rebuild them must commit first; see gdn_defer_settle.
static int gdn_pend_slot[MAXSLOTS] = { 0 };

namespace {

// ---------------------------------------------------------------------------------------------
// GDN. gdn_pre: per (seq, group of 256 channels): commit the accepted tokens' pre-activations into
// the conv ring, run the conv + silu for this pass's rows, L2-normalise q and k per head, cache
// conv outputs and gates. ph_gdn: per (seq, head, quarter): replay the accepted tokens into the
// committed state, write it back, then run the rows from the fresh cache without committing.

// The decay gate of one token and head, from the alpha projection the prep phase already scaled.
// softplus goes through the full-precision log1pf, whose compensated arithmetic is about 180
// scalar-float instructions; that is why the only place this is allowed to run is once per
// (row, head). Both producers call this one expression so their bits cannot drift apart.
__device__ __forceinline__ float gdn_decay(float alpha, float dt, float a) {
    const float xg = alpha + dt;
    const float sp = xg > 20.0f ? xg : log1pf(__expf(xg));
    return __expf(a * sp);
}

// What the block token cache holds in its last 2 * HV columns.
//
// It held the raw alpha and beta projections, and every reader re-derived the gates from them. The
// sliced route reads that cache from `ph_gdn`, whose unit is (seq, head, row group), so one
// `gdn_decay` per (row, head) became one per (row, head, group) - and per WAVE of the group, since
// all eight waves of a unit walk the same token. At the deployed width that is 32 evaluations of a
// 180-instruction libm chain for every (row, head) of every layer.
//
// The resident route already evaluates its own rows' gates once per (row, head) in `resident_conv`
// (GATE_HOIST) and reads them back; this makes the block cache carry the same evaluated form, so
// the sliced route and every replay prefix get the same treatment. `gdn_load`'s PREGATE path is the
// reader. Same expression on the same inputs, stored as the fp32 it already was: bit-identical.
// HALO_GDN_BLK_GATE=0 restores the raw form in every producer and consumer at once.
//
// The resident route's own rule is the warning to read first (`GATE_HOIST_MIN_ROWS_PER_SEQ`):
// hoisting these 190 issue slots is worth nothing when the wave was issuing them while its state
// loads were in flight, and it measured 3.5% WORSE there at four rows per sequence. That rule takes
// the hoist from eight rows per sequence up, and a verify pass walks eight rows plus its accepted
// prefix - but it was fitted on the resident kernel, so the sliced route's own panel is what
// decides this. docs/gdn-sliced-gates.md.
#ifndef HALO_GDN_BLK_GATE
#define HALO_GDN_BLK_GATE 1
#endif
constexpr bool GDN_BLK_PREGATE = HALO_GDN_BLK_GATE != 0;

__device__ __forceinline__ float * blk_tok(const FwdParams & P, int seq, int parity, int gdn_idx, int slot) {
    return P.blk_cache + ((((size_t) seq * 2 + parity) * 48 + gdn_idx) * RMAX + slot) * BLK_TOKEN_FLOATS;
}

template <int CM>
__device__ __forceinline__ void ph_gdn_pre(Ctx & c, const FwdParams & P, const LayerW & L, int gdn_idx) {
    constexpr int GROUPS = CONV_CH / 256; // 40
    const int total = P.nseq * GROUPS;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    for (int u = blockIdx.x; u >= 0 && u < total; u = next_unit(c, total)) {
        c.dirty = true;
        const int si = u / GROUPS, g = u - si * GROUPS;
        const SeqCtl S = P.seqs[si];
        const int s = P.rows[S.row0].seq; // state slot
        const int ch = g * 256 + tid;
        float * ring = P.conv_ring + ((size_t) s * 48 + gdn_idx) * GDN_RING_FLOATS;
        float r0 = ring[ch], r1 = ring[CONV_CH + ch], r2 = ring[2 * CONV_CH + ch];
        // commit the previous block's accepted pre-activations
        for (int t = 0; t < S.n_replay; t++) {
            const float v = blk_tok(P, s, S.parity ^ 1, gdn_idx, t)[ch];
            r0 = r1; r1 = r2; r2 = v;
        }
        // Replay contract: persist the ring at the accepted prefix, because the next pass rebuilds
        // this pass's rows. Direct commit persists it once, after those rows, below.
        if (CM == 0) { ring[ch] = r0; ring[CONV_CH + ch] = r1; ring[2 * CONV_CH + ch] = r2; }
        const float4 w = *(const float4 *) (L.conv_w + ch * 4);
        const int chunk_kind = ch < 2 * KDIM ? 0 : 1; // q/k channels get L2-normalised per 128
        for (int i = 0; i < S.nrows; i++) {
            const int row = S.row0 + i;
            const float xc = P.big[(size_t) row * QKV_OUT + ch];
            float v = silu(w.x * r0 + w.y * r1 + w.z * r2 + w.w * xc);
            float * tok = blk_tok(P, s, S.parity, gdn_idx, i);
            tok[ch] = xc;
            if (chunk_kind == 0) {
                // 128-channel head = 4 waves; reduce within the half of the workgroup
                float ss = warp_sum(v * v);
                if (lane == 0) c.red[wave] = ss;
                __syncthreads();
                const int h0 = (wave >> 2) * 4;
                const float tot = c.red[h0] + c.red[h0 + 1] + c.red[h0 + 2] + c.red[h0 + 3];
                __syncthreads();
                v *= 1.0f / fmaxf(sqrtf(tot), NORM_EPS);
            }
            tok[CONV_CH + ch] = v;
            r0 = r1; r1 = r2; r2 = xc;
        }
        // The gate columns, one thread per (row, head), and one ROW per unit rather than all of
        // them in the unit that happens to be group 0. They belong to no channel group - the loop
        // above walks 256 channels and these are 2 * HV numbers for the whole row - and evaluating
        // them costs a `log1pf` each, so leaving them inside that serial row walk put eight of
        // those chains end to end in one workgroup while thirty-nine others had none, and the phase
        // ends when its slowest unit does. A pass carries at most RMAX rows and a layer deals 40
        // groups, so `g` selects a row and every row has exactly one writer.
        if (g < S.nrows && tid < 2 * HV) {
            // alpha for the first HV entries, the raw beta projection for the rest.
            const int row = S.row0 + g;
            const float ab = P.ab[(size_t) row * 2 * HV + tid] * P.ninv[row];
            blk_tok(P, s, S.parity, gdn_idx, g)[2 * CONV_CH + tid] = GDN_BLK_PREGATE
                ? (tid < HV ? gdn_decay(ab, L.ssm_dt[tid], L.ssm_a[tid]) : sigmoid(ab)) : ab;
        }
        // The rows are the committed prefix now; the same three pre-activations the next pass would
        // have shifted in from blk_cache are already in these registers.
        if (CM) { ring[ch] = r0; ring[CONV_CH + ch] = r1; ring[2 * CONV_CH + ch] = r2; }
    }
    end_phase(c);
}

// One token's operands for this lane: gates, k/q at lane + 32 s, v at this wave's R state rows.
// SPLIT units share a 128-row head, so a wave owns R = 16 / SPLIT of its unit's rows.
//
// PREGATE says the token record already holds the evaluated gates rather than the raw alpha and
// beta projections. The resident path evaluates them once per (row, head) in the convolution
// kernel; the sliced path and the replay cache keep the raw form, where each state unit re-derives
// them for every token it walks.
template <int R> struct GdnTokT { float g, beta, kk[4], qq[4], vj[R]; };
template <int SPLIT, int OFFSET = CONV_CH, bool PREGATE = false> __device__ __forceinline__ GdnTokT<16 / SPLIT> gdn_load(const LayerW & L, const float * tok, int h, int kh, int part, int lane, int wave) {
    constexpr int R = 16 / SPLIT, RPU = 128 / SPLIT;
    GdnTokT<R> t;
    if constexpr (PREGATE) {
        t.g = tok[OFFSET + CONV_CH + h];
        t.beta = tok[OFFSET + CONV_CH + HV + h];
    } else {
        t.g = gdn_decay(tok[OFFSET + CONV_CH + h], L.ssm_dt[h], L.ssm_a[h]);
        t.beta = sigmoid(tok[OFFSET + CONV_CH + HV + h]);
    }
    const float * q = tok + OFFSET + kh * 128, * k = tok + OFFSET + KDIM + kh * 128, * v = tok + OFFSET + 2 * KDIM + h * 128;
    #pragma unroll
    for (int s2 = 0; s2 < 4; s2++) { t.kk[s2] = k[lane + 32 * s2]; t.qq[s2] = q[lane + 32 * s2]; if (s2 < R) t.vj[s2] = v[part * RPU + wave + 8 * s2]; }
    #pragma unroll
    for (int r = 4; r < R; r++) t.vj[r] = v[part * RPU + wave + 8 * r];
    return t;
}
// state rows part*RPU + wave + 8r, columns lane + 32 s
//
// `pend_a` is the deferred-commit path's delta column: the wave that owns a row writes the value
// the rank-1 update is built from, which is the only quantity a later step cannot recompute from
// the token record. It is null on every other route and folds away with the store.
//
// THE R ROWS OF A UNIT ARE INDEPENDENT AND THE STORE PREDICATE USED TO FENCE THEM APART. A row's
// work is a decay, a k-dot, a 5-deep butterfly, a rank-1 update, a q-dot and a second 5-deep
// butterfly, and its tail `if (out_o && lane == 0)` is a divergent branch. Written one row at a
// time that branch terminates a basic block, so the emitted body was R blocks the scheduler could
// not mix: 54 instructions each at SPLIT = 4, of which 10 were cross-lane and 14 were `s_delay_alu`
// covering their latency with nothing. The rows are batched here instead - every row's k-dot, then
// every row's butterfly, then every row's update and q-dot, then every row's butterfly - so one
// block holds R independent chains and the machine covers each one's DPP latency with the others'
// arithmetic. Nothing is reassociated: each sum keeps its own products in its own order, the
// butterflies are the same `warp_sum` over the same values with full exec, and the stores are the
// same values to the same addresses. docs/gdn-token-reduce.md.
template <int SPLIT> __device__ __forceinline__ void gdn_token(const GdnTokT<16 / SPLIT> & t, int h, int pu, int lane, int wave, float m[16 / SPLIT][4], float * out_o, float * pend_a = nullptr, int pend_cls = 1) {
    constexpr int R = 16 / SPLIT, RPU = 128 / SPLIT;
    float d[R], op[R];
    #pragma unroll
    for (int r = 0; r < R; r++) {
        float part = 0.0f;
        #pragma unroll
        for (int s2 = 0; s2 < 4; s2++) { m[r][s2] *= t.g; part = fmaf(m[r][s2], t.kk[s2], part); }
        d[r] = part;
    }
    #pragma unroll
    for (int r = 0; r < R; r++) d[r] = (t.vj[r] - warp_sum(d[r])) * t.beta;
    #pragma unroll
    for (int r = 0; r < R; r++) {
        float o = 0.0f;
        #pragma unroll
        for (int s2 = 0; s2 < 4; s2++) { m[r][s2] = fmaf(d[r], t.kk[s2], m[r][s2]); o = fmaf(m[r][s2], t.qq[s2], o); }
        op[r] = o;
    }
    #pragma unroll
    for (int r = 0; r < R; r++) op[r] = warp_sum(op[r]);
    // `warp_sum` leaves the wave two roundings of this delta and the update above applied each to
    // the columns its own lanes own, so a term that has to replay exactly keeps both. One lane of
    // each class writes; the packed coordinate asks for one class and gets the value it always had.
    if (pend_a && (lane == 0 || (pend_cls > 1 && lane == 4))) {
        float * dst = pend_a + (lane ? (size_t) SS : 0);
        #pragma unroll
        for (int r = 0; r < R; r++) dst[pu * RPU + wave + 8 * r] = d[r];
    }
    if (out_o && lane == 0) {
        #pragma unroll
        for (int r = 0; r < R; r++) out_o[h * 128 + pu * RPU + wave + 8 * r] = op[r] * 0.08838834764831845f;
    }
}

// unit = (seq, head, one of SPLIT row groups of the state): replay the accepted tokens into the
// committed state, write it back, then run this pass's rows from the fresh cache. The next token's
// operands are loaded while the current one is processed. Every state row belongs to exactly one
// unit and nothing is reduced across units, so SPLIT changes only the work decomposition: at one
// row the deployed SPLIT = 4 leaves 192 units on a 160-workgroup grid, two rounds of makespan for
// 1.2 rounds of state traffic.
template <int SPLIT, int CM, bool PK>
__device__ __forceinline__ void ph_gdn(Ctx & c, const FwdParams & P, const LayerW & L, int gdn_idx) {
    // A persistent kernel allocates one register maximum over every phase it contains, so a runtime
    // format switch here costs the deployed single-token pass 96 VGPRs to 135 and six waves per
    // SIMD32 - on a route that never asks for a packed state. PK carries the answer from the
    // launcher instead, and the fp32 kernels compile to the code they have today.
    const int fmt = PK ? r_gdn_fmt : GDN_STATE_F32;
    constexpr int R = 16 / SPLIT, RPU = 128 / SPLIT, SHIFT = SPLIT == 16 ? 4 : SPLIT == 8 ? 3 : SPLIT == 4 ? 2 : 1;
    static_assert(1 << SHIFT == SPLIT, "SPLIT is 2, 4, 8 or 16");
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int total = P.nseq * HV * SPLIT;
    for (int u = blockIdx.x; u >= 0 && u < total; u = next_unit(c, total)) {
        c.dirty = true;
        const int si = u / (HV * SPLIT), rem = u - si * HV * SPLIT, h = rem >> SHIFT, part = rem & (SPLIT - 1), kh = h % HK;
        const SeqCtl S = P.seqs[si];
        const int s = P.rows[S.row0].seq;
        float * region = P.gdn_state + ((size_t) s * 48 + gdn_idx) * (size_t) r_gdn_region;
        float * M = region + (size_t) h * 128 * 128;
        const int ntok = S.n_replay + S.nrows;
        auto tok_of = [&](int t) { return t < S.n_replay ? blk_tok(P, s, S.parity ^ 1, gdn_idx, t) : blk_tok(P, s, S.parity, gdn_idx, t - S.n_replay); };
        GdnTokT<R> cur = gdn_load<SPLIT, CONV_CH, GDN_BLK_PREGATE>(L, tok_of(0), h, kh, part, lane, wave);
        float m[R][4];
        #pragma unroll
        for (int r = 0; r < R; r++) {
            const GdnQuad q = gdn_state_load_row(fmt, M, region, h, part * RPU + wave + 8 * r, lane);
            #pragma unroll
            for (int s2 = 0; s2 < 4; s2++) m[r][s2] = q.v[s2];
        }
        for (int t = 0; t < ntok; t++) {
            GdnTokT<R> nxt;
            if (t + 1 < ntok) nxt = gdn_load<SPLIT, CONV_CH, GDN_BLK_PREGATE>(L, tok_of(t + 1), h, kh, part, lane, wave);
            const bool replay = t < S.n_replay;
            gdn_token<SPLIT>(cur, h, part, lane, wave, m, replay ? nullptr : P.o + (size_t) (S.row0 + t - S.n_replay) * VDIM);
            if (CM == 0 && t + 1 == S.n_replay) {
                #pragma unroll
                for (int r = 0; r < R; r++) {
                    GdnQuad q;
                    #pragma unroll
                    for (int s2 = 0; s2 < 4; s2++) q.v[s2] = m[r][s2];
                    q = gdn_state_store_row(fmt, M, region, h, part * RPU + wave + 8 * r, lane, q);
                    #pragma unroll
                    for (int s2 = 0; s2 < 4; s2++) m[r][s2] = q.v[s2];
                }
            }
            if (t + 1 < ntok) cur = nxt;
        }
        // Direct commit: the registers now hold the state after the replayed prefix and this pass's
        // own rows, the exact value the next pass would have rebuilt by replaying them.
        if (CM) {
            #pragma unroll
            for (int r = 0; r < R; r++) {
                GdnQuad q;
                #pragma unroll
                for (int s2 = 0; s2 < 4; s2++) q.v[s2] = m[r][s2];
                q = gdn_state_store_row(fmt, M, region, h, part * RPU + wave + 8 * r, lane, q);
                #pragma unroll
                for (int s2 = 0; s2 < 4; s2++) m[r][s2] = q.v[s2];
        }
        }
    }
    end_phase(c);
}

constexpr int RESIDENT_TOKEN_FLOATS=CONV_CH+2*HV;
__device__ __forceinline__ float resident_history(const GdnResidentParams &a,
        const ResidentSeq &s,int gi,int t,int ch) {
    if(t>=0)return a.p.big[(size_t)(s.row0+t)*QKV_OUT+ch];
    if(t>=-s.n_replay)return blk_tok(a.p,s.slot,s.parity^1,gi,t+s.n_replay)[ch];
    return a.p.conv_ring[((size_t)s.slot*48+gi)*GDN_RING_FLOATS+
                        (3+t+s.n_replay)*CONV_CH+ch];
}
// Where the decay gate is evaluated.
//
//   GATE_LOOP   each state unit derives its own gates inside the token loop: what shipped, and the
//               control this is measured against.
//   GATE_HOIST  resident_conv evaluates them once per (row, head) and the loop reads them.
//
// Both run the same tokens through the same recurrence in the same order against the same gate
// values, so this is a placement choice and not a numerical one.
//
// A third arm, walking the replay prefix in its own loop so the raw-gate transform is not compiled
// into the loop that runs once per row, measured 1.5 ms worse and is not here. docs/resident-gdn.md
// records it: it took the row loop from 355 issue slots to 325 and the kernel from 6640 bytes to
// 7736, because the unrolled token body is then emitted twice.
enum {GATE_LOOP=0,GATE_HOIST=1};

template<int GATE,bool DEFER=false> __global__ __launch_bounds__(256) void resident_conv(GdnResidentParams a,int layer) {
    const auto &P=a.p;const auto &L=r_layers[layer];
    constexpr int GROUPS=CONV_CH/256;
    const int row=blockIdx.x/GROUPS,g=blockIdx.x%GROUPS,tid=threadIdx.x,lane=tid&31,wave=tid>>5;
    int si=0;while(row>=a.seqs[si].row0+a.seqs[si].nrows)++si;
    const ResidentSeq S=a.seqs[si];const int gi=layer-layer/4,ch=g*256+tid,i=row-S.row0;
    // Whether this pass commits, decided once per (sequence, layer) in the launch before the one
    // that reads it. Every block of the state kernel has to agree, and the count it agrees on is
    // the one thing in the region that a state block both reads and writes - so it is settled here
    // instead, where a single thread owns it and no reader is running.
    if constexpr(DEFER) {
        if(g==0&&tid==0&&row==S.row0) {
            float *region=P.gdn_state+((size_t)S.slot*48+gi)*(size_t)r_gdn_region;
            const size_t ctl=gdn_fmt_defer_ctl(r_gdn_fmt);
            const int ntok=S.n_replay+S.nrows;
            int pend=(int)region[ctl];
            if(pend<0||pend>=r_gdn_defer)pend=0;
            // A pass that walks more than one token per sequence is a prefill shape: it pays the
            // state round trip once for many tokens, so there is nothing to defer and the full
            // write leaves the region in the plain packed coordinate.
            const bool flush=ntok!=1||pend+1>=r_gdn_defer;
            region[ctl+1]=(float)pend;
            region[ctl+2]=flush?1.0f:0.0f;
            region[ctl]=flush?0.0f:(float)(pend+1);
        }
    }
    float r0=resident_history(a,S,gi,i-3,ch),r1=resident_history(a,S,gi,i-2,ch),
          r2=resident_history(a,S,gi,i-1,ch);
    const float4 w=*(const float4*)(L.conv_w+ch*4);
    __shared__ float red[8];
    {
        float xc=P.big[(size_t)row*QKV_OUT+ch];
        // Match the deployed contraction: y*r1 rounds before x*r0 is fused.
        float v=silu(fmaf(w.w,xc,fmaf(w.z,r2,fmaf(w.x,r0,__fmul_rn(w.y,r1)))));
        if(ch<2*KDIM) {
            float ss=warp_sum(v*v);if(lane==0)red[wave]=ss;
            __syncthreads();int h0=(wave>>2)*4;
            float tot=red[h0]+red[h0+1]+red[h0+2]+red[h0+3];
            __syncthreads();v*=1.0f/fmaxf(sqrtf(tot),NORM_EPS);
        }
        float *tok=a.tokens+(size_t)row*RESIDENT_TOKEN_FLOATS;tok[ch]=v;
        if(g==0&&tid<2*HV) {
            // alpha for the first HV entries, the raw beta projection for the rest.
            const float ab=P.ab[(size_t)row*2*HV+tid]*P.ninv[row];
            if constexpr(GATE>=GATE_HOIST)tok[CONV_CH+tid]=tid<HV?gdn_decay(ab,L.ssm_dt[tid],L.ssm_a[tid]):sigmoid(ab);
            else tok[CONV_CH+tid]=ab;
        }
    }
}
template<int SPLIT,int GATE,bool DEFER=false> __global__ __launch_bounds__(256) void resident_state(GdnResidentParams a,int layer) {
    const auto &P=a.p;const auto &L=r_layers[layer];
    constexpr int R=16/SPLIT,RPU=128/SPLIT;
    int si=blockIdx.x/(HV*SPLIT),h=blockIdx.x/SPLIT%HV,part=blockIdx.x%SPLIT;
    int lane=threadIdx.x&31,wave=threadIdx.x>>5,gi=layer-layer/4;
    ResidentSeq S=a.seqs[si];
    // The convolution has finished reading the entry ring. Its last three raw
    // inputs can now replace it without racing another token's convolution.
    int group=h*SPLIT+part;
    if(group<CONV_CH/256) {
        int ch=group*256+threadIdx.x;
        float r0=resident_history(a,S,gi,S.nrows-3,ch),
              r1=resident_history(a,S,gi,S.nrows-2,ch),
              r2=resident_history(a,S,gi,S.nrows-1,ch);
        float *ring=P.conv_ring+((size_t)S.slot*48+gi)*GDN_RING_FLOATS;
        ring[ch]=r0;ring[CONV_CH+ch]=r1;ring[2*CONV_CH+ch]=r2;
    }
    float *region=P.gdn_state+((size_t)S.slot*48+gi)*(size_t)r_gdn_region,*M=region+(size_t)h*128*128;
    int pend=0;bool commit=true;
    __shared__ float skeys[DEFER?GDN_DEFER_MAX*SS:1],swgt[DEFER?GDN_DEFER_MAX+1:1],
                     sdels[DEFER?GDN_DEFER_MAX*RPU:1];
    if constexpr(DEFER) {
        pend=(int)region[gdn_fmt_defer_ctl(r_gdn_fmt)+1];commit=region[gdn_fmt_defer_ctl(r_gdn_fmt)+2]!=0.0f;
        if(pend>0) {
            gdn_defer_stage(r_gdn_fmt,region,h,pend,skeys,swgt,threadIdx.x,256,sdels,part,RPU);
            __syncthreads();
        }
    }
    float m[R][4];
    if constexpr(DEFER) gdn_defer_rebuild<R,RPU>(r_gdn_fmt,M,region,skeys,swgt,h,part,lane,wave,pend,m,sdels);
    else
#pragma unroll
    for(int r=0;r<R;++r) {
        const GdnQuad q=gdn_state_load_row(r_gdn_fmt,M,region,h,part*RPU+wave+8*r,lane);
#pragma unroll
        for(int j=0;j<4;++j)m[r][j]=q.v[j];
    }
    // The replay prefix comes from the deployed block cache, whose gate columns carry the same
    // evaluated form this kernel's own rows hold (GDN_BLK_PREGATE); with the arm off both are raw.
    auto load_replay=[&](int t) {
        return gdn_load<SPLIT,CONV_CH,GDN_BLK_PREGATE>(L,blk_tok(P,S.slot,S.parity^1,gi,t),h,h%HK,part,lane,wave);
    };
    auto load_row=[&](int t) {
        return gdn_load<SPLIT,0,(GATE>=GATE_HOIST)>(L,a.tokens+(size_t)(S.row0+t)*RESIDENT_TOKEN_FLOATS,h,h%HK,part,lane,wave);
    };
    auto load=[&](int t){return t<S.n_replay?load_replay(t):load_row(t-S.n_replay);};
    const int ntok=S.n_replay+S.nrows;
    auto cur=load(0);
    // A deferred pass keeps its own rank-1 term instead of the state it produces, and the per-row
    // half of that term - the delta - exists only inside the token body. The decision is made here
    // rather than inside the loop: a pass that defers walks exactly one token, and the loop below is
    // the whole cost of a 128-token prefill pass. Carrying the test into it cost that pass 11.7%.
    if(DEFER&&!commit) {
        gdn_token<SPLIT>(cur,h,part,lane,wave,m,P.o+(size_t)S.row0*VDIM,gdn_defer_deltas(r_gdn_fmt,region,h,pend),
                         gdn_fmt_defer_classes(r_gdn_fmt));
        // Every row of this head multiplied the same key by its own delta, and every later term
        // carries the same decay, so those two are written once for the head.
        if(part==0&&wave==0)gdn_defer_append_head(r_gdn_fmt,region,h,pend,cur.g,cur.kk,lane);
        return;
    }
    for(int t=0;t<ntok;++t) {
        GdnTokT<R> next;if(t+1<ntok)next=load(t+1);
        gdn_token<SPLIT>(cur,h,part,lane,wave,m,t<S.n_replay?nullptr:P.o+(size_t)(S.row0+t-S.n_replay)*VDIM);
        if(t+1<ntok)cur=next;
    }
#pragma unroll
    for(int r=0;r<R;++r) {
        GdnQuad q;
#pragma unroll
        for(int j=0;j<4;++j)q.v[j]=m[r][j];
        gdn_state_store_row(r_gdn_fmt,M,region,h,part*RPU+wave+8*r,lane,q);
    }
}
// ---------------------------------------------------------------------------------------------
// THE COLUMN OWNERSHIP OF A STATE ROW. One lane owns 32 columns of ONE row; a wave owns 8 rows.
//
// The deployed layout gives a wave R = 16/SPLIT whole rows and spreads each row's 128 columns over
// all 32 lanes, four per lane. Both consumers of a row - the k-side contraction that builds the
// delta and the q-side contraction that builds the output - are therefore cross-wave reductions,
// and the wave pays TWO five-stage `warp_sum` butterflies PER ROW PER TOKEN. The census in
// docs/resident-gdn.md prices what that costs: 355 issue slots a token, of which ~50 are the
// butterflies and 66 are the arithmetic, and instruction issue explains 88.6% of the measured
// phase. docs/gdn-map/README.md reaches the same place from the algebra - "the cross-lane
// reductions the block form was supposed to remove belong to the state's lane layout, not to the
// recurrence" - ranks L = 4, R = 1 at 1.69x on its lane-op count, and parks it.
//
// Turn the ownership 90 degrees. Lane `l` owns row `j = part*64 + wave + 8*(l>>2)` and, inside it,
// the 32 columns of the deployed lanes `8p .. 8p+7` for `p = l&3`. The arithmetic per state element
// is unchanged - a decay multiply, a k-side fma, an update fma, a q-side fma - but a row's whole
// contraction now happens inside four lanes, so ONE pair of two-stage quad butterflies reduces all
// eight of the wave's rows at once. The arithmetic floor is 16 slots per row-token at any L; what
// moves is the overhead, which is divided by the 32/L rows a wave carries instead of multiplied by
// the R rows a lane carries.
//
// WHY IT IS BIT-IDENTICAL, WHICH THE LANE-OP COUNT COULD NOT SEE. `warp_sum` is not a broadcast of
// one value: it leaves exactly TWO distinct bit patterns in the wave. After `quad_perm(1,0,3,2)`
// and `quad_perm(2,3,0,1)` every lane of a DPP quad holds the same quad sum `Q_{H,q}` (the two
// operand orders differ only by commutativity). `row_ror 4` then `row_ror 8` give lane `l` the value
// `(Q_{H,j} + Q_{H,j-1}) + (Q_{H,j-2} + Q_{H,j-3})` with `j = (l>>2)&3` and the quad indices taken
// mod 4, and `j = 2` reproduces `j = 0` while `j = 3` reproduces `j = 1` - again by commutativity.
// `xrow16` preserves `j` and adds the other half, so:
//
// THE ROTATION GOES THE OTHER WAY THAN IT READS. `tools/gdn_warpsum_probe` measures each DPP stage
// as a lane permutation on device: `row_ror n` gives lane `m` the value from lane `(m - n) mod 16`,
// so class 0 - the one lane 0 holds - pairs quad `j` with quad `j-1` and a half contributes
// `(Q0 + Q3) + (Q1 + Q2)`, NOT `(Q0 + Q1) + (Q2 + Q3)`. Getting that backwards swaps the two
// classes, which is four ULP on about half the state and nothing else, and an offline emulation
// written from the same wrong reading will confirm itself. This arm measured exactly that before
// the probe existed. Use the probe, not the manual.
//
//   * the deployed kernel updates column `c` with the delta computed from class `((c&31)>>2)&1`,
//     two roundings of the same sum living side by side in one row;
//   * `out_o` is written by lane 0, so an output always carries the even class.
//
// This arm rebuilds both classes from the eight column groups one lane holds and applies each to
// exactly the columns that had it. Old lanes `8p..8p+3` are one quad and `8p+4..8p+7` the next, so
// `p` holds quads `(p>>1, 2*(p&1))` and `(p>>1, 2*(p&1)+1)`; the class of column `8p+i+32s` is
// `((8p+i)>>2)&1 = (i>>2)&1`, independent of `p`. Same products, same summation trees, same deltas
// on the same columns: the residual FNV-64 is the control, not an argument.
//
// fp32 only, and deliberately. A packed store derives the row's shared exponent with `warp_max`
// over the whole wave, which is one row in the deployed layout and eight here; the codec owns that
// expression and this arm does not reach into it. See docs/gdn-lane-layout.md.
struct GdnColSums { float even, odd; };

// The two classes of one row's contraction, from the eight column-group partials of one lane.
// `A` and `B` are this lane's two deployed quad sums, in the deployed tree order: sub-lane `p`
// holds `Q_{p>>1, 2(p&1)}` and `Q_{p>>1, 2(p&1)+1}`.
__device__ __forceinline__ GdnColSums gdn_cols_reduce(const float (&part)[8]) {
    const float A = (part[0] + part[1]) + (part[2] + part[3]);
    const float B = (part[4] + part[5]) + (part[6] + part[7]);
    // Class 0, which lanes with `(l>>2)&1 == 0` hold and every output reads: a half contributes
    // `(Q0 + Q3) + (Q1 + Q2)`, so the pairing crosses the sub-lane pair.
    float e = A + dppf<0xB1>(B);              // u_p = Q_even(p) + Q_odd(p^1)
    e += dppf<0xB1>(e);                       // u_p + u_{p^1}: this half's class-0 value
    e += dppf<0x4E>(e);                       // + the other half
    // Class 1: `(Q0 + Q1) + (Q2 + Q3)`, the pair sum inside each sub-lane.
    float o = A + B;
    o += dppf<0xB1>(o);
    o += dppf<0x4E>(o);
    return { e, o };
}
// Class 0 alone, which is the only one an output reads.
__device__ __forceinline__ float gdn_cols_reduce_even(const float (&part)[8]) {
    const float A = (part[0] + part[1]) + (part[2] + part[3]);
    const float B = (part[4] + part[5]) + (part[6] + part[7]);
    float e = A + dppf<0xB1>(B);
    e += dppf<0xB1>(e);
    e += dppf<0x4E>(e);
    return e;
}

// One token's operands for a column-owning lane: the head's gates, this lane's 32 key and query
// columns, and the single `v` element of the row it owns. `kk[8*s + i]` is key column `8p+i+32s`,
// so a float4 covers `i = 4g .. 4g+3` at one `s` and the whole vector is eight loads.
struct __align__(16) GdnColTok { float kk[32], qq[32]; float g, beta, vj; };

template <int OFFSET, bool PREGATE>
__device__ __forceinline__ GdnColTok gdn_load_cols(const LayerW & L, const float * tok, int h, int kh, int j, int p) {
    GdnColTok t;
    if constexpr (PREGATE) {
        t.g = tok[OFFSET + CONV_CH + h];
        t.beta = tok[OFFSET + CONV_CH + HV + h];
    } else {
        t.g = gdn_decay(tok[OFFSET + CONV_CH + h], L.ssm_dt[h], L.ssm_a[h]);
        t.beta = sigmoid(tok[OFFSET + CONV_CH + HV + h]);
    }
    const float * q = tok + OFFSET + kh * 128, * k = tok + OFFSET + KDIM + kh * 128,
                * v = tok + OFFSET + 2 * KDIM + h * 128;
    t.vj = v[j];
    #pragma unroll
    for (int s = 0; s < 4; s++)
        #pragma unroll
        for (int g = 0; g < 2; g++) {
            *(float4 *) (&t.kk[8 * s + 4 * g]) = *(const float4 *) (k + 32 * s + 8 * p + 4 * g);
            *(float4 *) (&t.qq[8 * s + 4 * g]) = *(const float4 *) (q + 32 * s + 8 * p + 4 * g);
        }
    return t;
}

// One token against the 32 state columns this lane owns. Element for element and in the same order
// as `gdn_token`: decay then the k-side product into this column group's own chain, the two delta
// classes, then the rank-1 update and the q-side product interleaved exactly as the deployed loop
// interleaves them. `out_o` is the row's own output slot and only the `p == 0` lane writes it,
// which is the deployed lane-0 store.
__device__ __forceinline__ void gdn_token_cols(const GdnColTok & t, float m[8][4], int p, float * out_o) {
    float d[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        float acc = 0.0f;
        #pragma unroll
        for (int s = 0; s < 4; s++) { m[i][s] *= t.g; acc = fmaf(m[i][s], t.kk[8 * s + i], acc); }
        d[i] = acc;
    }
    const GdnColSums ks = gdn_cols_reduce(d);
    // Column `8p+i` sits in deployed lane `8p+i`, whose class is `((8p+i)>>2)&1 = (i>>2)&1`.
    const float d0 = (t.vj - ks.even) * t.beta, d1 = (t.vj - ks.odd) * t.beta;
    float op[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const float dd = i < 4 ? d0 : d1;
        float o = 0.0f;
        #pragma unroll
        for (int s = 0; s < 4; s++) {
            m[i][s] = fmaf(dd, t.kk[8 * s + i], m[i][s]);
            o = fmaf(m[i][s], t.qq[8 * s + i], o);
        }
        op[i] = o;
    }
    const float oe = gdn_cols_reduce_even(op);
    if (out_o && p == 0) *out_o = oe * 0.08838834764831845f;
}

// The column-ownership unit: (seq, head, one of two 64-row halves of the state). Two units a head
// against the deployed four, eight waves a unit, eight rows a wave - the same row set a SPLIT = 2
// unit owns, so a storage order fitted to an eight-stride wave serves this arm unchanged. Nothing
// is reduced across units or across waves here either.
template <int GATE> __global__ __launch_bounds__(256) void resident_state_cols(GdnResidentParams a, int layer) {
    const auto & P = a.p; const auto & L = r_layers[layer];
    constexpr int NPART = 2, RPU = 128 / NPART;
    const int si = blockIdx.x / (HV * NPART), h = blockIdx.x / NPART % HV, part = blockIdx.x % NPART;
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, gi = layer - layer / 4;
    const int p = lane & 3, j = part * RPU + wave + 8 * (lane >> 2);
    ResidentSeq S = a.seqs[si];
    // Same ring commit as the deployed unit: the convolution has finished reading the entry ring,
    // so its last three raw inputs can replace it. Forty groups of 256 channels cover CONV_CH.
    const int group = h * NPART + part;
    if (group < CONV_CH / 256) {
        const int ch = group * 256 + threadIdx.x;
        float r0 = resident_history(a, S, gi, S.nrows - 3, ch),
              r1 = resident_history(a, S, gi, S.nrows - 2, ch),
              r2 = resident_history(a, S, gi, S.nrows - 1, ch);
        float * ring = P.conv_ring + ((size_t) S.slot * 48 + gi) * GDN_RING_FLOATS;
        ring[ch] = r0; ring[CONV_CH + ch] = r1; ring[2 * CONV_CH + ch] = r2;
    }
    float * region = P.gdn_state + ((size_t) S.slot * 48 + gi) * (size_t) r_gdn_region, * M = region + (size_t) h * 128 * 128;
    // Eight calls to the codec's own row accessor, one per column group, so the addresses and the
    // storage order stay the codec's and a reorder there needs no change here.
    float m[8][4];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const GdnQuad q = gdn_state_load_row(GDN_STATE_F32, M, region, h, j, 8 * p + i);
        #pragma unroll
        for (int s = 0; s < 4; ++s) m[i][s] = q.v[s];
    }
    const int ntok = S.n_replay + S.nrows;
    for (int t = 0; t < ntok; ++t) {
        const GdnColTok cur = t < S.n_replay
            ? gdn_load_cols<CONV_CH, GDN_BLK_PREGATE>(L, blk_tok(P, S.slot, S.parity ^ 1, gi, t), h, h % HK, j, p)
            : gdn_load_cols<0, (GATE >= GATE_HOIST)>(L, a.tokens + (size_t) (S.row0 + t - S.n_replay) * RESIDENT_TOKEN_FLOATS, h, h % HK, j, p);
        gdn_token_cols(cur, m, p, t < S.n_replay ? nullptr : P.o + (size_t) (S.row0 + t - S.n_replay) * VDIM + h * 128 + j);
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        GdnQuad q;
        #pragma unroll
        for (int s = 0; s < 4; ++s) q.v[s] = m[i][s];
        gdn_state_store_row(GDN_STATE_F32, M, region, h, j, 8 * p + i, q);
    }
}

// Rebuilds the committed state from its base and pending updates and writes it back, so a route
// that cannot read a deferred region can read this one. It is the same rebuild and the same store
// the resident kernel runs, which makes it idempotent: at zero pending it re-stores the values it
// just decoded, and in a packed coordinate re-rounding an already rounded value returns it.
template<int SPLIT> __global__ __launch_bounds__(256) void gdn_defer_flush_k(float *gdn_state,int slot,int layers) {
    constexpr int R=16/SPLIT,RPU=128/SPLIT;
    const int per=HV*SPLIT,gi=blockIdx.x/per,rem=blockIdx.x%per,h=rem/SPLIT,part=rem%SPLIT;
    if(gi>=layers)return;
    const int lane=threadIdx.x&31,wave=threadIdx.x>>5;
    float *region=gdn_state+((size_t)slot*48+gi)*(size_t)r_gdn_region,*M=region+(size_t)h*128*128;
    const int pend=(int)region[gdn_fmt_defer_ctl(r_gdn_fmt)];
    if(pend<=0||pend>GDN_DEFER_MAX)return;
    __shared__ float skeys[GDN_DEFER_MAX*SS],swgt[GDN_DEFER_MAX+1];
    __shared__ float sdels[GDN_DEFER_MAX*RPU];
    gdn_defer_stage(r_gdn_fmt,region,h,pend,skeys,swgt,threadIdx.x,256,sdels,part,RPU);
    __syncthreads();
    float m[R][4];
    gdn_defer_rebuild<R,RPU>(r_gdn_fmt,M,region,skeys,swgt,h,part,lane,wave,pend,m,sdels);
#pragma unroll
    for(int r=0;r<R;++r) {
        GdnQuad q;
#pragma unroll
        for(int j=0;j<4;++j)q.v[j]=m[r][j];
        gdn_state_store_row(r_gdn_fmt,M,region,h,part*RPU+wave+8*r,lane,q);
    }
}
__global__ void gdn_defer_clear_k(float *gdn_state,int slot,int layers) {
    const int gi=blockIdx.x*256+threadIdx.x;
    if(gi>=layers)return;
    float *region=gdn_state+((size_t)slot*48+gi)*(size_t)r_gdn_region;
    const size_t ctl=gdn_fmt_defer_ctl(r_gdn_fmt);
    region[ctl]=0.0f;region[ctl+1]=0.0f;region[ctl+2]=1.0f;
}
template<int GATE,bool DEFER> void launch_resident_arm(const GdnResidentParams &a,int layer,int split,dim3 conv,dim3 grid,hipStream_t stream) {
    resident_conv<GATE,DEFER><<<conv,256,0,stream>>>(a,layer);
    // SPLIT 1 is not here on purpose: it measured 16.102 ms against 15.400 at four and 14.513 at
    // two on a 128-row pass, so it is a loss, and four instantiations of a 173-VGPR kernel body is
    // 8 seconds of this translation unit. tools/gdn_occ_probe still instantiates it, which is where
    // the register ladder stays reproducible. docs/gdn-unit-width.md.
    if(split==2)resident_state<2,GATE,DEFER><<<grid,256,0,stream>>>(a,layer);
    else if(split==4)resident_state<4,GATE,DEFER><<<grid,256,0,stream>>>(a,layer);
    else if(split==8)resident_state<8,GATE,DEFER><<<grid,256,0,stream>>>(a,layer);
    else resident_state<16,GATE,DEFER><<<grid,256,0,stream>>>(a,layer);
}
// This is the quantiser of the sequence OUTPUT projection for the 48 recurrent layers: it writes
// the wide operand `project<part 2>` reads. QL and NIB carry the output stage's activation
// coordinate; <127,false> is the deployed instantiation and compiles to the code it always did.
// docs/sequence-output-a4.md.
template<int QL,bool NIB> __global__ __launch_bounds__(256) void resident_output(GdnResidentParams a,int layer) {
    const auto &P=a.p;const auto &L=r_layers[layer];
    int row=blockIdx.x/(VDIM/1024),chunk=blockIdx.x%(VDIM/1024);
    PrepR p{};p.x=P.o;p.x2=P.zbuf;p.kvec=L.ssm_norm_s;p.eps=NORM_EPS;p.n=VDIM;
    p.flags=PREP_GDN_NORM|PREP_HADAMARD|PREP_QUANT;
    p.seq_q=P.sequence_q;p.seq_scales=P.sequence_scales;p.seq_npad=P.sequence_npad;p.seq_offset=0;
    __shared__ float scratch[1024];
    prep_chunk_r<true,QL,NIB>(p,row,chunk,1.0f,scratch,a.nrows);
}

} // namespace

// Where the decay gate is evaluated. 1 evaluates it once per (row, head) in the convolution kernel
// and stores it; 0 re-derives it inside every state unit's token loop, which is the shape the
// resident path shipped with and the control this is measured against. Process default from
// HALO_GDN_PREGATE; gdn_resident_set_pregate() moves it between calls so both arms share one
// process and one clock.
// -1 lets the shape decide, which is the default; 0 or 1 pins one placement for measurement.
static int gdn_pregate = [] {
    const char *v=getenv("HALO_GDN_PREGATE");
    if(!v)return -1;
    const std::string value(v);
    if(value!="0"&&value!="1"){fprintf(stderr,"HALO_GDN_PREGATE must be 0 or 1\n");abort();}
    return atoi(v);
}();
void gdn_resident_set_pregate(int on){gdn_pregate=on<GATE_LOOP||on>GATE_HOIST?-1:on;}
int gdn_resident_pregate(){return gdn_pregate;}
bool gdn_blk_pregate(){return GDN_BLK_PREGATE;}

// How wide a state unit is, overriding what the batch was created with. The width owns no state -
// a row lives at the same address in its region whatever unit walks it, and nothing is reduced
// across units - so unlike a format or a defer depth this needs no synchronise, no reset and no
// re-prefill, and two arms can alternate inside one process and one clock. -1 leaves the batch's
// own HALO_GDN_SPLIT value alone. docs/gdn-unit-width.md.
static int gdn_split_req = -1;
void gdn_resident_set_split(int s){gdn_split_req=gdn_split_valid(s)?s:-1;}
int gdn_resident_split(){return gdn_split_req;}

// Which width a launch gets when nothing has pinned one. Mode 19 at a pinned clock, both arms
// interleaved in one process, `gdn-resident-core` against the phases the width cannot reach:
//
//   1 seq x 128 rows   15.400 -> 14.513 ms   -5.9%
//   1 seq x  32 rows    4.686 ->  4.490      -4.2%
//   4 seq x  32 tokens 14.890 -> 14.618      null (0.09403 -> 0.09381 of the untouched phases)
//   2 seq x  64 tokens 15.364 -> 16.094      +4.8%
//  32 seq x   8 tokens 63.883 -> 64.695      +1.3%
//  32 seq x   1 token  44.926 -> 45.791      +1.9%
//
// THE AXIS IS THE DEPTH OF THE GRID, NOT PREFILL AGAINST DECODE AND NOT TOKENS PER SEQUENCE. One
// sequence deals 48*SPLIT units, so SPLIT 4 asks 192 blocks of a machine that holds 160 - 1.2
// rounds of work taking two - while SPLIT 2 asks 96 and runs them once, and its wider wave pays
// the per-token fixed cost (the gate, the k and q fragments, the loop and its addressing) over
// eight rows instead of four. Two sequences already deal 384 units, deep enough that other waves
// cover that cost, and there the wider unit's lower occupancy (14 waves/SIMD32 against 16, 12 when
// deferred) and coarser rounding cost 1 to 5%. Every arm above is bit-identical - one residual
// FNV-64 per shape across the whole ladder - so this selects a schedule, never a numerical map.
// docs/gdn-unit-width.md.
int gdn_resident_split_for(int nseq){return nseq==1?2:4;}

// The state's storage coordinate, process-wide because it describes bytes that outlive a pass.
// kernels/gdn_state_codec.hpp holds the layout and why a shared exponent beats fp16 here.
static int gdn_fmt = [] {
    const char *v=getenv("HALO_GDN_STATE");
    if(!v)return (int)GDN_STATE_F32;
    const std::string s(v);
    if(s=="f32"||s=="0")return (int)GDN_STATE_F32;
    if(s=="f16"||s=="1")return (int)GDN_STATE_F16;
    if(s=="i16"||s=="2")return (int)GDN_STATE_I16;
    if(s=="i8"||s=="3")return (int)GDN_STATE_I8;
    if(s=="f32pk"||s=="4")return (int)GDN_STATE_F32_PK;
    if(s=="i8c"||s=="5")return (int)GDN_STATE_I8C;
    if(s=="i8s"||s=="6")return (int)GDN_STATE_I8S;
    fprintf(stderr,"HALO_GDN_STATE must be f32, f16, i16, i8, f32pk, i8c or i8s\n");abort();
    return (int)GDN_STATE_F32;
}();
static void gdn_fmt_upload() {
    const hipError_t e=hipMemcpyToSymbol(HIP_SYMBOL(r_gdn_fmt),&gdn_fmt,sizeof gdn_fmt);
    if(e!=hipSuccess){fprintf(stderr,"state format upload failed: %s\n",hipGetErrorString(e));abort();}
}

// The stride the engine allocates its state regions at, and the one every kernel addresses them
// with. A coordinate needs a third of an fp32 region at eight bits, and reserving the fp32 size for
// it cost 144 MiB per sequence slot - the largest single item in a slot and the reason 128 streams
// needed a 43 GiB budget. It is process-wide for the same reason the format is: a region's start is
// an address, and every live slot already holds state at whichever stride it was written under.
//
// Reserve before the engine allocates, freeze there, and refuse to move afterwards. The
// measurement tools sweep coordinates inside one engine, so they reserve every arm they will run
// and pay the widest one's memory; a process that runs one coordinate pays only that one's.
//
// IT IS NOT THE DEFAULT YET, AND THE REASON IS WRITTEN DOWN. The compact stride reproduces the
// full-stride bits on batched generation (32 streams, 32/32 stream token-identical) and on prompt
// ingestion (tail-logit and residual FNV-64 equal over 128-row passes), and it FAILS the mixed
// commit/replay acceptance at its first committing pass in every packed coordinate, where the
// full-stride build of the same source passes. docs/state-region-stride.md has the bisect: the
// trigger is the stride and not the format-derived offsets, and until that divergence is explained
// the engine allocates what it always allocated. HALO_GDN_REGION=packed selects the compact one.
static int gdn_region_packed = [] {
    const char *v=getenv("HALO_GDN_REGION");
    if(!v)return 0;
    const std::string s(v);
    if(s=="packed"||s=="1")return 1;
    if(s=="full"||s=="0")return 0;
    fprintf(stderr,"HALO_GDN_REGION must be full or packed\n");abort();
    return 0;
}();
// Whether this process wants somewhere to keep a pending write-back. Depth 0 runs the deferred
// kernels and commits every step, so it needs the room too; depth 1 is the eager path and does not.
// Constant-initialised, raised by the environment default below and by gdn_defer_reserve_depth,
// and read before the engine allocates - never during static initialisation.
static bool gdn_defer_room_req=false;
static bool gdn_defer_room_wanted(int depth){return depth==0||depth>=2;}
static size_t gdn_region_for(int fmt) {
    size_t want=gdn_region_packed?gdn_fmt_region_floats(fmt):(size_t)GDN_STATE_FLOATS;
    // A packed coordinate's values leave half or three quarters of a region free and the pending
    // list lives there. An exact coordinate's values fill the region exactly, so a deferring
    // process has to be given the room - about 13% of a (slot, layer) region, which is memory
    // nothing paid for until someone asked to defer. docs/gdn-defer-exact.md.
    if(gdn_fmt_is_exact(fmt)&&gdn_defer_room_req) {
        const size_t room=gdn_fmt_defer_room(fmt);
        if(room>want)want=room;
    }
    return want;
}
int gdn_region_compact(){return gdn_region_packed;}
static size_t gdn_region_res=0;
static bool gdn_region_frozen=false;
// Which coordinates this process has said it may run. The stride is one number for the whole
// process, so raising the defer reservation has to re-ask every coordinate already reserved rather
// than only the one in hand.
static unsigned gdn_fmt_res_mask=0;
static size_t gdn_region_want() {
    size_t want=gdn_region_for(gdn_fmt);
    for(int f=0;f<=GDN_STATE_FMT_MAX;++f)
        if(gdn_fmt_res_mask&(1u<<(unsigned)f)){const size_t w=gdn_region_for(f);if(w>want)want=w;}
    return want;
}
static void gdn_region_upload() {
    const int v=(int)(gdn_region_frozen?gdn_region_res:gdn_region_for(gdn_fmt));
    const hipError_t e=hipMemcpyToSymbol(HIP_SYMBOL(r_gdn_region),&v,sizeof v);
    if(e!=hipSuccess){fprintf(stderr,"state region stride upload failed: %s\n",hipGetErrorString(e));abort();}
}
void gdn_state_boot_format(int fmt) {
    if(fmt<0||fmt>GDN_STATE_FMT_MAX){fprintf(stderr,"invalid GDN state format %d\n",fmt);abort();}
    if(gdn_region_frozen){gdn_state_set_format(fmt);return;}   // regions exist: take the ordinary path
    // Before the engine allocates there is no live state and no launch, so this is host state and
    // the uploads happen with the rest when the device appears. It matters because the stride is
    // frozen from the format the process is IN: a panel that only ever runs int8 arms but starts in
    // fp32 would reserve fp32-sized regions for coordinates it never uses.
    gdn_fmt=fmt;gdn_fmt_res_mask|=1u<<(unsigned)fmt;
    const size_t want=gdn_region_want();
    if(want>gdn_region_res)gdn_region_res=want;
}
void gdn_state_reserve_format(int fmt) {
    if(fmt<0||fmt>GDN_STATE_FMT_MAX){fprintf(stderr,"invalid GDN state format %d\n",fmt);abort();}
    const size_t want=gdn_region_for(fmt);
    if(gdn_region_frozen&&want>gdn_region_res) {
        fprintf(stderr,"state format %s needs %zu floats per (slot, layer) region and this process "
                       "already allocated %zu; reserve it before the engine is built\n",
                gdn_state_format_name(fmt),want,gdn_region_res);
        abort();
    }
    gdn_fmt_res_mask|=1u<<(unsigned)fmt;
    const size_t all=gdn_region_want();
    if(all>gdn_region_res)gdn_region_res=all;
}
// The same contract for the write-back depth: say it before the regions exist, or be told.
void gdn_defer_reserve_depth(int depth) {
    if(depth<0||depth>GDN_DEFER_MAX){fprintf(stderr,"invalid GDN defer depth %d\n",depth);abort();}
    if(!gdn_defer_room_wanted(depth)||gdn_defer_room_req)return;
    gdn_defer_room_req=true;
    const size_t want=gdn_region_want();
    if(gdn_region_frozen&&want>gdn_region_res) {
        gdn_defer_room_req=false;
        fprintf(stderr,"a state write-back shared by %d steps keeps a pending list beside the state "
                       "and needs %zu floats per (slot, layer) region at this process's coordinates, "
                       "which already allocated %zu; reserve the depth before the engine is built\n",
                depth,want,gdn_region_res);
        abort();
    }
    if(want>gdn_region_res)gdn_region_res=want;
}
size_t gdn_region_floats() {
    if(!gdn_region_frozen) {
        const size_t mine=gdn_region_want();
        if(mine>gdn_region_res)gdn_region_res=mine;
        gdn_region_frozen=true;
        gdn_region_upload();
    }
    return gdn_region_res;
}
static void gdn_defer_upload();
const char *gdn_state_format_name(int fmt) {
    switch(fmt){case GDN_STATE_F16:return "f16";case GDN_STATE_I16:return "i16";case GDN_STATE_I8:return "i8";
                case GDN_STATE_F32:return "f32";case GDN_STATE_F32_PK:return "f32pk";
                case GDN_STATE_I8C:return "i8c";case GDN_STATE_I8S:return "i8s";default:return "?";}
}
int gdn_state_format(){return gdn_fmt;}
void gdn_state_set_format(int fmt) {
    if(fmt<0||fmt>GDN_STATE_FMT_MAX){fprintf(stderr,"invalid GDN state format %d\n",fmt);abort();}
    if(fmt==gdn_fmt)return;
    gdn_state_reserve_format(fmt);   // aborts if the regions this process allocated are too narrow
    // Every live slot holds the old coordinate, so no launch may still be reading it and the
    // caller has to reset or re-prefill. Kernels take the format from a constant, not an argument.
    const hipError_t e=hipDeviceSynchronize();
    if(e!=hipSuccess){fprintf(stderr,"state format switch failed: %s\n",hipGetErrorString(e));abort();}
    gdn_fmt=fmt;gdn_fmt_upload();gdn_defer_upload();
}

// How many steps share one state write-back. A packed head leaves three quarters or half of its
// (slot, layer) region unused, which is where the pending rank-1 updates live and why this axis
// was packed-only when it was built. An exact coordinate can run it too - it buys more there,
// because fp32 is the one phase in this engine already at its byte roof, and it is bit-identical
// there, because the chain rebuild in gdn_state_codec.hpp replays the pending triples in the
// order gdn_token applied them and an fp32 store loses nothing. What it costs is the region, so
// the room is reserved rather than assumed: a process says the depth it may run before the engine
// allocates. Both measurement tools record the requested and the effective depth.
static bool gdn_defer_told=false;   // whether a depth was asked for rather than defaulted
static int gdn_defer_req = [] {
    const char *v=getenv("HALO_GDN_DEFER");
    // The default. Two steps per write-back is where the measured ladder arrives - 2, 4 and 8 are
    // one number at 32 streams, because the write stops being half the traffic and the phase
    // becomes its read - and at an exact coordinate it changes no output bit. A packed coordinate
    // keeps 1 unless it is told otherwise, because there the deferral rounds differently from the
    // step-by-step chain and every published packed number was measured on the eager map.
    if(!v){gdn_defer_room_req=true;return 2;}
    const int d=atoi(v);
    if(d<0||d>GDN_DEFER_MAX){fprintf(stderr,"HALO_GDN_DEFER must be 0 (deferred kernels, nothing deferred) or 1..%d\n",GDN_DEFER_MAX);abort();}
    gdn_defer_told=true;
    if(gdn_defer_room_wanted(d))gdn_defer_room_req=true;   // before any allocation: no region exists yet
    return d;
}();
int gdn_defer_depth(){return gdn_defer_req;}
// An exact coordinate's pending term carries both delta roundings, so its list is capped where its
// measured ladder stopped paying. Every sample records the depth that ran beside the one asked for.
int gdn_defer_effective() {
    if(!gdn_defer_told&&!gdn_fmt_is_exact(gdn_fmt))return 1;   // the packed map nobody asked to change
    const int cap=gdn_fmt_defer_cap(gdn_fmt);
    return gdn_defer_req>cap?cap:gdn_defer_req;
}
// Depth 0 is the control that separates the kernels from the deferral, the way f32pk separates the
// packed kernels from the packed coordinate: the deferred instantiations run, with their extra
// registers and their rebuild path, and commit every step anyway. It must reproduce the plain
// packed arm's bits exactly, and whatever it costs in time is the instantiation rather than the
// change.
static bool gdn_defer_kernels(){const int d=gdn_defer_effective();return d==0||d>=2;}
static void gdn_defer_upload() {
    const int e0=gdn_defer_effective(),d=e0==0?1:e0;
    const hipError_t e=hipMemcpyToSymbol(HIP_SYMBOL(r_gdn_defer),&d,sizeof d);
    if(e!=hipSuccess){fprintf(stderr,"defer depth upload failed: %s\n",hipGetErrorString(e));abort();}
}
void gdn_defer_set_depth(int d) {
    if(d<0||d>GDN_DEFER_MAX){fprintf(stderr,"invalid GDN defer depth %d\n",d);abort();}
    const bool told=gdn_defer_told;
    gdn_defer_told=true;
    if(d==gdn_defer_req&&told)return;
    gdn_defer_reserve_depth(d);   // aborts if the regions this process allocated have no room for a pending list
    // Live slots carry pending updates whose count is interpreted against the old depth, so this
    // has the same contract as a format switch: synchronise, then reset or re-prefill.
    const hipError_t e=hipDeviceSynchronize();
    if(e!=hipSuccess){fprintf(stderr,"defer depth switch failed: %s\n",hipGetErrorString(e));abort();}
    gdn_defer_req=d;gdn_defer_upload();
}

// How many rows a sequence contributes is what decides this, not how many rows the pass has.
//
// A state unit loads its 64 kB of head state once and then walks that sequence's tokens. Hoisting
// the gate takes 190 issue slots out of each of those tokens, which is most of a long walk and
// nothing at all next to the state round trip on a short one. Worse than nothing on a short one:
// at four tokens per sequence the in-loop gate measures 52.823 ms against 54.673 for the hoist,
// because the wave loses instructions it was issuing while its state loads were in flight.
//
// Measured at 1, 4 and 128 rows per sequence. 5 to 127 is interpolation; both arms are correct
// everywhere and --gdn-pregate pins either one.
constexpr int GATE_HOIST_MIN_ROWS_PER_SEQ = 8;
int gdn_resident_gate_for(int nrows,int nseq) {
    if(gdn_pregate>=0)return gdn_pregate;
    return nseq>0&&nrows>=GATE_HOIST_MIN_ROWS_PER_SEQ*nseq?GATE_HOIST:GATE_LOOP;
}

// How wide a state unit is. A unit owns 128/SPLIT rows of one (sequence, head); each of the
// block's eight waves owns 16/SPLIT of them and holds their four columns per lane in registers.
// Every row belongs to exactly one unit and nothing is reduced across units or waves, so this is
// a work decomposition and nothing else: the same row runs the same decay, the same k-dot over the
// same 32 lanes, the same rank-1 update and the same q-dot in the same order whatever SPLIT is.
//
// The axis was capped at 4, 8 and 16 and only the losing half of it had ever been compiled.
// `docs/resident-gdn.md` measured 235.6 ms at 4 against 242.8 and 253.4 at 8 and 16 and named the
// reason: a unit re-reads the whole token's k and q however few rows it owns, so narrowing the
// unit multiplies the operand stream. Widening it divides that stream instead, and it is also
// what fixes a makespan: one sequence's 48 heads at SPLIT = 4 deal 192 units onto a 160-block
// grid, two rounds for 1.2 rounds of work.
//
// `gdn_split_valid` in halo_kernels.h is the contract both the launcher and the batch's env
// validation read. SPLIT must be a power of two so that 16/SPLIT rows per wave divides the block,
// and 1 is the widest a 256-thread block can express. docs/gdn-unit-width.md holds the ladder.
//
// The register cost of a wider unit, from tools/gdn_occ_probe on gfx1151, no spill in any arm:
// SPLIT 4 is 64 VGPR at 16 waves/SIMD32, SPLIT 2 is 100 (105 deferred) at 14 (12), SPLIT 1 is 173
// (194) at 8 (6). A 128-row pass deals 48*SPLIT units, so the grid at SPLIT 2 is 96 blocks against
// a 140-block capacity and the lower occupancy limit never binds; a 32-stream step deals 3072 and
// runs 22 rounds of them.
// Which lane layout the state units run. -1 lets the shape decide, which is the default; 0 pins the
// deployed row ownership and 1 pins the column ownership. docs/gdn-lane-layout.md.
//
// The rule is the same quantity the gate placement turns on, for the same reason: the column arm
// removes issue slots from a token, and a unit that walks one token is waiting for its state round
// trip rather than issuing. The two hard constraints are not tuning - a deferred pass keeps its
// rank-1 term through machinery that owns the deployed row decomposition, and a packed store
// derives its shared exponent with a whole-wave `warp_max` that means "this row" only in the
// deployed layout - so this arm runs on fp32 state and committed passes.
static int gdn_cols = [] {
    const char *v=getenv("HALO_GDN_COLS");
    if(!v)return -1;
    const std::string value(v);
    if(value!="0"&&value!="1"){fprintf(stderr,"HALO_GDN_COLS must be 0 or 1\n");abort();}
    return atoi(v);
}();
void gdn_resident_set_cols(int on){gdn_cols=on<0||on>1?-1:on;}
int gdn_resident_cols(){return gdn_cols;}
static void gdn_defer_flush_slot(float * gdn_state, int slot, hipStream_t st);
// The column unit is fp32 by construction. It used to be non-deferring as well, because it owns a
// state row across four lanes and has no rebuild, so it reads whatever is in the region rather
// than a base plus a pending sum. That is now a settle rather than an exclusion: the unit is only
// ever chosen for a pass carrying many rows per sequence, which commits its state anyway, so the
// flush it costs is one a prefill pass was going to pay at its own end.
bool gdn_resident_cols_available(){return gdn_fmt==GDN_STATE_F32;}
constexpr int COLS_MIN_ROWS_PER_SEQ = 8;
static bool gdn_resident_cols_for(int nrows,int nseq) {
    if(!gdn_resident_cols_available())return false;
    if(gdn_cols>=0)return gdn_cols!=0;
    return nseq>0&&nrows>=COLS_MIN_ROWS_PER_SEQ*nseq;
}

void launch_gdn_resident(const GdnResidentParams &a,int layer,int split,hipStream_t stream) {
    if(gdn_split_req>0)split=gdn_split_req;
    else if(split<1)split=gdn_resident_split_for(a.nseq);
    if(layer<0||layer>=NLAYER||layer%4==3||a.nseq<1||a.nseq>MAXSLOTS||a.nrows<1||a.nrows>PASSMAX||!a.p.commit_state||!a.p.sequence_q||!gdn_split_valid(split)) {
        fprintf(stderr,"invalid resident GDN geometry or contract\n");abort();
    }
    const dim3 conv(a.nrows*(CONV_CH/256)),grid(a.nseq*HV*split);
    const bool defer=gdn_defer_kernels(),cols=gdn_resident_cols_for(a.nrows,a.nseq);
    // The same rule the convolution kernel writes into the region, kept where the host can see it:
    // a route that cannot rebuild a deferred state asks this before it reads one.
    if(defer&&cols) {
        for(int i=0;i<a.nseq;++i) {
            const int slot=a.seqs[i].slot;
            if(slot>=0&&slot<MAXSLOTS&&gdn_pend_slot[slot])gdn_defer_flush_slot(a.p.gdn_state,slot,stream);
        }
    } else if(defer) {
        const int depth=gdn_defer_effective()<2?1:gdn_defer_effective();
        for(int i=0;i<a.nseq;++i) {
            const int slot=a.seqs[i].slot;
            if(slot<0||slot>=MAXSLOTS)continue;
            const int ntok=a.seqs[i].n_replay+a.seqs[i].nrows,pend=gdn_pend_slot[slot];
            gdn_pend_slot[slot]=ntok!=1||pend+1>=depth?0:pend+1;
        }
    }
    if(cols) {
        // The column arm's unit is 64 rows, so its grid is two units a head whatever `split` says.
        // `resident_conv` is the deployed non-deferred instantiation; only the state unit changes.
        const dim3 cgrid(a.nseq*HV*2);
        if(gdn_resident_gate_for(a.nrows,a.nseq)==GATE_LOOP) {
            resident_conv<GATE_LOOP,false><<<conv,256,0,stream>>>(a,layer);
            resident_state_cols<GATE_LOOP><<<cgrid,256,0,stream>>>(a,layer);
        } else {
            resident_conv<GATE_HOIST,false><<<conv,256,0,stream>>>(a,layer);
            resident_state_cols<GATE_HOIST><<<cgrid,256,0,stream>>>(a,layer);
        }
    } else if(gdn_resident_gate_for(a.nrows,a.nseq)==GATE_LOOP) {
        if(defer)launch_resident_arm<GATE_LOOP,true>(a,layer,split,conv,grid,stream);
        else launch_resident_arm<GATE_LOOP,false>(a,layer,split,conv,grid,stream);
    } else {
        if(defer)launch_resident_arm<GATE_HOIST,true>(a,layer,split,conv,grid,stream);
        else launch_resident_arm<GATE_HOIST,false>(a,layer,split,conv,grid,stream);
    }
    const int q=sequence_quant();
    const dim3 og(a.nrows*(VDIM/1024));
    if(!sequence_quant_out_seven(q))        resident_output<127,false><<<og,256,0,stream>>>(a,layer);
    else if(sequence_quant_out_nibble(q))   resident_output<7,true><<<og,256,0,stream>>>(a,layer);
    else                                    resident_output<7,false><<<og,256,0,stream>>>(a,layer);
}

#define SYNC() do { grid_sync(c); if (P.debug_stop && (int) c.gen >= P.debug_stop) return; } while (0)

// SM: single-token execution map, 0 = deployed path. See FwdParams::single_map.
// PART: 0 the whole pass, or one whole layer when `stage` selects it. 1 and 2 are the two halves
// of one layer around a projection the caller owns (launch_sequence_part). 3 is the head's input
// prep alone, for a vocabulary projection that owns every row of the batch.
// CM: 1 commits the conv ring and GDN state at the end of this pass (FwdParams::commit_state).
// OCC is the minimum waves per SIMD32 asked of the register allocator. 1 is no constraint, which
// is what this kernel was compiled with until the constraint existed, and PK_OCC_WIDE is the arm
// that buys the wider cooperative grid. gfx1151 allocates VGPRs to a wave32 in granules of 24 out
// of a 1536-register file, so the occupancy steps are 144 registers (10 waves), 120 (12), 96 (16),
// 72 (21): there is nothing between 135 and 120, and a kernel at 121 registers is a kernel at 10
// waves.
//
// The phase that sets the mark is the attention unit, whose qr[TT][HD/32] query cache is 64 VGPRs
// at eight rows: this kernel is 135 registers with it and 90 without. `HALO_ROWS_DROP` is how that
// was found, a compile-time bitmask that deletes one phase from the body so
// tools/kernel_resources.py can read the high-water mark with and without it, and
// kernels/pk_pressure.hip prices every phase on its own. docs/rows-grid-occupancy.md holds both
// tables, what the constraint costs, and how to re-fit it when the attention unit changes.
//
// The ask belongs to the body in the tree on the day it was fitted, and the body moves: 7eeb556
// gave the attention unit one score body and took the fp32 eight-row shape from 206 registers to
// 141, which is four blocks per WGP where 135 was five. HALO_ROWS_WPE re-asks the whole build, so
// the re-fit in docs/rows-grid-occupancy.md is one -D and one probe rather than an edit: 12 buys
// 120 registers and five blocks, 13 buys 96 and the six that LDS caps this kernel at.
#ifndef HALO_ROWS_WPE
#define HALO_ROWS_WPE 12
#endif
#define PK_OCC_WIDE HALO_ROWS_WPE
// The second wide budget, so the two asks are in one executable and `--pk-occ 1,2` interleaves them
// on one clock. 13 is the ask that reaches 96 registers, which is the next window where the
// occupancy API and the hardware agree - and agreement is what a cooperative launch needs, because
// the launcher refuses any grid above the API's estimate whatever the device would hold.
#ifndef HALO_ROWS_WPE2
#define HALO_ROWS_WPE2 13
#endif
#define PK_OCC_WIDE2 HALO_ROWS_WPE2

// Which budget a pass takes when nobody pins one. 1 is `waves_per_eu(12)`, the ask this rule has
// shipped since 7eeb556 moved the body to 141 registers; 2 is `waves_per_eu(13)`, which reaches 96
// and is the next window where the occupancy API and the hardware agree on the block count.
// docs/rows-grid-occupancy.md carries the panel that decides it.
#ifndef HALO_ROWS_OCC_RULE
#define HALO_ROWS_OCC_RULE 1
#endif


// THE ALLOCATOR'S LADDER IS NOT THE DEVICE'S, AND THE DEVICE'S IS NOT MONOTONIC. `waves_per_eu` is
// the only budget the front end will take as a template parameter, and the compiler answers it on
// its own granule of 24 out of 1536: the whole attribute reaches 144, 120, 96, 72 and nothing else.
// What the runtime co-schedules is a different function, measured to the register by
// tools/occ_surface at a 256-thread workgroup and any LDS block under 8192 bytes:
//
//   VGPRs  89..96  97..104  105..112  113..120  121..128  129..136  137..176  177..184
//   blocks    8        7         6         5         6         5         4         3
//   grid    160      140       120       100       120       100        80        60
//
// Both inversions are real and reproducible. 121 registers holds six blocks where 120 holds five,
// and 89 holds eight where 88 holds seven, so a register cut can cost a block and a register added
// can buy one. `waves_per_eu(12)` lands the eight-row body on exactly 120 - the last register of a
// five-block window with a six-block window starting one register above it.
//
// HALO_ROWS_VGPR asks the allocator for an exact ceiling instead, which is what reaches the windows
// `waves_per_eu` cannot name: 128 and 112 are six blocks, 104 is seven, 96 is eight. The attribute
// takes a literal, so this is a build axis rather than a template parameter, and
// tools/kernel_resources.py -D HALO_ROWS_PROBE=8 -D HALO_ROWS_VGPR=104 prices one in nine seconds.
// A ceiling the body cannot meet is paid for in scratch, which is why every candidate is censused
// for spills before it is built into an engine. docs/rows-grid-occupancy.md carries the table.
#ifndef HALO_ROWS_VGPR
#define HALO_ROWS_VGPR 0
#endif
#if HALO_ROWS_VGPR
#define ROWS_VGPR_CAP __attribute__((amdgpu_num_vgpr(HALO_ROWS_VGPR)))
#else
#define ROWS_VGPR_CAP
#endif

#ifndef HALO_ROWS_DROP
#define HALO_ROWS_DROP 0
#endif
// A BUILD THAT FITS A BOUNDED CALL, FOR THE SLICED ROUTE. This translation unit is about fifty
// instantiations of the largest kernel in the repository and device codegen for all of them has
// grown past 55 seconds on this box, which is longer than one agent tool call: `tools/split-build
// --step 0` cannot finish and neither can `make`. Every row count here is a *capacity*, not a
// contract - `nrows` is a runtime argument and every body guards on it - so the two intermediate
// widths can be served by the eight-row kernel while a measurement build is being cut, which drops
// roughly half the instantiations and brings the TU back inside a call. A trimmed build runs 2- and
// 4-row passes on the 8-row kernel: correct, and slower for those two shapes only. It is a build
// flag for measuring the 1- and 8-row route, not a default: `make HALO_ROWS_TRIM=1`.
#ifndef HALO_ROWS_TRIM
#define HALO_ROWS_TRIM 0
#endif
#if HALO_ROWS_TRIM
#define ROWS_TT2 8
#define ROWS_TT4 8
#else
#define ROWS_TT2 2
#define ROWS_TT4 4
#endif
#define DROP(b) ((HALO_ROWS_DROP & (b)) != 0)

// What this kernel's shared block actually has to hold, in the form `slice_lds_floats` already
// takes for `k_ffn_slice` below. `LDS_FLOATS` is the union of every phase in `phases.hpp` plus the
// drafter's top-k list, and a forward pass runs none of the widest terms: what happens here is
// prep's Hadamard staging, the dot4 matvec's cross-wave drain, attention's score array with its
// block reduction, the one-row wide-head unit, the rope staging and the argmax slice.
//
// THIS IS AN OCCUPANCY NUMBER, NOT A MEMORY ONE. The occupancy API hands out 65536 / bytes per
// block, so 9508 bytes is six workgroups per WGP and 7204 is nine, and six is exactly what held
// this kernel below the eight its register file allows at `waves_per_eu(13)`.
// docs/rows-lds-occupancy.md has the ladder.
constexpr int rows_lds_floats(int TT) {
    const int mv   = (NW - 1) * TT * 32;      // ph_matvec's drain stages waves 1..NW-1
    const int attn = TT * ACHUNK + NW * TT;   // attn_chunk_group's scores plus its block reduction
    const int head = (NH / NKV) * ACHUNK;     // ph_attn's one-row wide-head unit, HG = GQA, TT = 1
    const int prep = 1024;                    // prep_chunk_r's Hadamard staging, and the floor
    int n = mv > attn ? mv : attn;
    if (head > n) n = head;
    return n > prep ? n : prep;
}
// The asserts that used to protect this kernel compare against `LDS_FLOATS` inside `phases.hpp`,
// which no longer describes its block, so they are restated here for every width it launches.
#define ROWS_LDS_FITS(TT) \
    static_assert(rows_lds_floats(TT) >= (NW - 1) * (TT) * 32 && \
                  rows_lds_floats(TT) >= (TT) * ACHUNK + NW * (TT) && \
                  rows_lds_floats(TT) >= (NH / NKV) * ACHUNK && \
                  rows_lds_floats(TT) >= 1024 && \
                  rows_lds_floats(TT) >= NT && \
                  rows_lds_floats(TT) <= LDS_FLOATS, "forward block outgrew its phases")
ROWS_LDS_FITS(1); ROWS_LDS_FITS(2); ROWS_LDS_FITS(4); ROWS_LDS_FITS(8);

template <int TT, int SM = 0, int PART = 0, int CM = 0, bool PK = false, int OCC = 1>
__global__ void __launch_bounds__(NT) __attribute__((amdgpu_waves_per_eu(OCC))) ROWS_VGPR_CAP k_forward_rows(FwdParams P, int stage) {
    // Units per 128-row GDN head. The eight-row route deals `HV * SPLIT` of them on this kernel's
    // grid, so SPLIT decides both the makespan of the state phase and how many times a token's
    // gate, `k` and `q` are re-read: every wave of every unit reads the whole token whatever few
    // state rows it owns. At SPLIT 4 one sequence deals 192 units on a 100-workgroup grid - 1.92
    // rounds of work taking two - and at SPLIT 2 it deals 96, which is one round with the operand
    // stream halved. `gdn_resident_split_for` made the same choice for one sequence on the resident
    // route (docs/gdn-unit-width.md, -5.9% there); this is that result carried to the route the
    // served verify pass runs. Every state row keeps its own columns, its own lane and its own
    // order, so the width is a decomposition and not a numerical choice. docs/gdn-sliced-gates.md.
    //
    // IT IS TAKEN ONLY WHERE IT IS FREE, and two measurements set the gate.
    //
    // REGISTERS. A persistent kernel allocates one budget over every phase it contains, and a
    // SPLIT 2 unit holds twice the state rows per wave: `resident_state` goes 64 VGPR to 100 for
    // the same step. At TT = 8 that fits under the budget the matvec and attention phases already
    // set - the deployed `k_forward_rows<8, 0, 0, 0, false, 12>` is 120 VGPR with 8 spilled dwords
    // in both arms, cell for cell - while at TT = 1 it costs two occupancy steps (12 waves at 100
    // VGPR to 10 at 132) and the packed-state instantiation spills 21 dwords to 102.
    //
    // SEQUENCES. The width is worth -0.4% of a one-sequence eight-row step and +2.3% AGAINST an
    // eight-sequence one, which is the same interior optimum `gdn_resident_split_for` found on the
    // resident route: a unit re-reads the whole token however few rows it owns, so a wider unit
    // pays less of that per token, and a second sequence already deals enough units to fill the
    // grid without it. `CM == 0` is the gate because it is the one that means one sequence in this
    // engine: an eight-row pass that does NOT commit its state is a drafted verify pass or an
    // `Engine::forward` slice, both solo, while every multi-sequence eight-row step commits
    // (`forward_batch` routes 5..31 rows to mode 4). A multi-sequence uncommitted eight-row pass is
    // reachable only from `tools/batch_profile --modes 0 --decode-streams N`, and it keeps the
    // deployed width. docs/gdn-sliced-gates.md.
#ifndef HALO_GDN_PK_SPLIT
#define HALO_GDN_PK_SPLIT 2
#endif
    constexpr int GDN_SPLIT = SM >= 2 ? 16 : (TT == 8 && !PK && CM == 0 ? HALO_GDN_PK_SPLIT : 4);
    // SM 3 retiles every matvec so the unit count is far larger than the grid: the deployed
    // splits leave 512 and 448 unit phases on a 160-workgroup grid, which round up to a whole
    // extra round. Residual-accumulating segments (add = 1) already atomicAdd, so only the
    // add = 0 outputs need zeroing first.
    // THE EIGHT-ROW PASS DOES NOT WANT THE SAME RETILING, AND IT IS NOT A CLOSE CALL. The scaffolding
    // probe says it should: `bench/coop_cost --tt 8` reads 141.9 GB/s for a one-round phase, 162.4 at
    // four rounds, 184.3 at eleven and 204.2 at seventy-eight, and the shapes a wider split would give
    // these phases read 182 to 217. Built and measured, the same split is **+31% on the pass**: every
    // retiled phase is 12 to 46% slower and the vocabulary head, which keeps KS = 1, is the 1.1%
    // in-panel null. The unit boundary the probe does not model - eight waves through LDS, two
    // barriers and 256 `atomicAdd`s per unit - costs about 2.7 us of workgroup time, more than the
    // longer phase wins. Coarsening to KSV = KSFF = 1 is a null in the same panel, so the deployed
    // split sits on a plateau. It is also a numerical map rather than a schedule: one prompt of four
    // changed its greedy continuation. docs/drafted-serve-step.md has the surface, the panel and the
    // patch.
    constexpr int KS40 = SM >= 3 ? 5 : 1;    // 40-block matvecs: qkv+z, q/k/v, gate/up
    constexpr int KSV  = SM >= 3 ? 6 : 2;    // 48-block matvecs: ssm_out, o
    constexpr int KSFF = SM >= 3 ? 17 : 2;   // 136-block matvec: down
    // The wide operand belongs to the projection the caller owns, so only the part that feeds it
    // offers the quantiser that destination. PART 0 is compiled without it and keeps its code.
    constexpr bool DQ_IN = (PART == 1), DQ_OUT = (PART == 2);
    __shared__ __attribute__((aligned(16))) float lds[rows_lds_floats(TT)];
    __shared__ float red[8];
    __shared__ int s_unit;
    Ctx c { P.bar, P.work, P.prof, 0, 0, true, lds, red, &s_unit };
    const int nrows = P.nrows;
    if (P.prof && blockIdx.x == 0 && threadIdx.x == 0) P.prof[0] = wall_clock64();

    if constexpr (PART == 3) {
        // The deployed head's prep, writing the wide operand instead of the row-major triple.
        { PrepR p{}; p.x = P.x; p.kvec = P.output_norm_s; p.norm_w = P.output_norm; p.eps = NORM_EPS; p.n = D; p.flags = PREP_NORM | PREP_HADAMARD | PREP_QUANT;
          p.xq = P.xq; p.xs = P.xs; p.xsum = P.xsum; p.store_normed = P.hfinal; bind_seq_dest<true>(p, P); ph_prep<true>(c, p, nrows); }
        return;
    }

    if constexpr (PART == 0) {
    if (stage < 0) {
    if ((int) blockIdx.x < nrows) embed_row(P.tok_embd, P.rows[blockIdx.x].token, P.tmp + (size_t) blockIdx.x * D);
    SYNC();
    { PrepR p{}; p.x = P.tmp; p.kvec = P.signs5120; p.n = D; p.flags = PREP_HADAMARD | PREP_SIGN_AFTER | PREP_STORE_F32; p.out_f32 = P.x; ph_prep(c, p, nrows); }
    SYNC();

    }
    if (stage == -1) return;
    }
    int gdn_idx = stage >= 0 ? stage - stage / 4 : 0, cap_next = 0;
    const int * cap_layers = CAP_LAYERS;
    const int nlayer = stage >= 0 && stage < NLAYER ? stage + 1 : P.debug_layers > 0 ? P.debug_layers : NLAYER;
    for (int l = stage >= 0 ? stage : 0; l < nlayer; l++) {
        const LayerW & L = r_layers[l];
        int cap_idx = -1;
        if (P.hcap && cap_next < NCAP && l == cap_layers[cap_next] + 1) cap_idx = cap_next++;
        if (L.recurrent) {
            if constexpr (PART != 2) {
            { PrepR p{}; p.x = P.x; p.kvec = L.attn_norm_s; p.norm_w = L.attn_norm; p.eps = NORM_EPS; p.n = D; p.flags = PREP_NORM | PREP_HADAMARD | PREP_QUANT;
              p.xq = P.xq; p.xs = P.xs; p.xsum = P.xsum; p.cap = cap_idx >= 0 ? P.hcap : nullptr; p.cap_idx = cap_idx;
              p.ninv = P.ninv; p.ab_w = L.alpha_beta; p.ab_out = P.ab; p.ab_rows = 2 * HV; bind_seq_dest<DQ_IN>(p, P); ph_prep<DQ_IN>(c, p, nrows); }
            if (KS40 > 1) { zero_span(c, P.big, nrows * QKV_OUT); zero_span(c, P.zbuf, nrows * VDIM); }
            SYNC();
            }
            if constexpr (PART == 1) return;   // the caller owns qkv/z from here
            if constexpr (PART == 0) {
            { MvR m{}; m.nseg = 2; m.seg[0] = { L.qkv, P.big, QKV_OUT / 32, 0 }; m.seg[1] = { L.z, P.zbuf, VDIM / 32, 0 }; m.total_tiles = (QKV_OUT + VDIM) / 32;
              ph_matvec_pk<NB_D, KS40, TT, false, SM>(c, m, P.xq, P.xs, P.xsum, nrows); }
            SYNC();
            }
            if (!DROP(8)) {
            ph_gdn_pre<CM>(c, P, L, gdn_idx);
            SYNC();
            ph_gdn<GDN_SPLIT, CM, PK>(c, P, L, gdn_idx);
            SYNC();
            }
            { PrepR p{}; p.x = P.o; p.x2 = P.zbuf; p.kvec = L.ssm_norm_s; p.eps = NORM_EPS; p.n = VDIM; p.flags = PREP_GDN_NORM | PREP_HADAMARD | PREP_QUANT;
              p.xq = P.xq; p.xs = P.xs; p.xsum = P.xsum; bind_seq_dest<DQ_OUT>(p, P); ph_prep<DQ_OUT>(c, p, nrows); }
            SYNC();
            if constexpr (PART == 2) return;   // the caller owns ssm_out and the residual add
            if (!DROP(32)) {
            { MvR m{}; m.nseg = 1; m.seg[0] = { L.ssm_out, P.x, D / 32, 1 }; m.total_tiles = D / 32; ph_matvec_pk<NB_V, KSV, TT, false, SM>(c, m, P.xq, P.xs, P.xsum, nrows); }
            SYNC();
            }
            gdn_idx++;
        } else {
            if constexpr (PART != 2) {
            { PrepR p{}; p.x = P.x; p.kvec = L.attn_norm_s; p.eps = NORM_EPS; p.n = D; p.flags = PREP_NORM | PREP_HADAMARD | PREP_QUANT;
              p.xq = P.xq; p.xs = P.xs; p.xsum = P.xsum; p.cap = cap_idx >= 0 ? P.hcap : nullptr; p.cap_idx = cap_idx; bind_seq_dest<DQ_IN>(p, P); ph_prep<DQ_IN>(c, p, nrows); }
            if (KS40 > 1) { zero_span(c, P.qfull, nrows * Q_OUT); zero_span(c, P.kbuf, nrows * KV_OUT); zero_span(c, P.vbuf, nrows * KV_OUT); }
            SYNC();
            }
            if constexpr (PART == 1) return;   // the caller owns q/k/v from here
            if constexpr (PART == 0) {
            { MvR m{}; m.nseg = 3; m.seg[0] = { L.q, P.qfull, Q_OUT / 32, 0 }; m.seg[1] = { L.k, P.kbuf, KV_OUT / 32, 0 }; m.seg[2] = { L.v, P.vbuf, KV_OUT / 32, 0 };
              m.total_tiles = (Q_OUT + 2 * KV_OUT) / 32; ph_matvec_pk<NB_D, KS40, TT, false, SM>(c, m, P.xq, P.xs, P.xsum, nrows); }
            SYNC();
            }
            if (!DROP(4)) {
            ph_attn_pre(c, P, L.q_norm, L.k_norm, L.kv_slot);
            SYNC();
            ph_attn<TT, NH, NKV, HD>(c, P, L.kv_slot);
            SYNC();
            }
            { PrepR p{}; p.x2 = P.qfull; p.partials = P.partials; p.rows = P.rows; p.kvec = P.signs6144; p.n = ATTN_OUT;
              p.flags = PREP_ATTN_COMBINE | PREP_HADAMARD | PREP_QUANT; p.xq = P.xq; p.xs = P.xs; p.xsum = P.xsum; bind_seq_dest<DQ_OUT>(p, P); ph_prep<DQ_OUT>(c, p, nrows); }
            SYNC();
            if constexpr (PART == 2) return;   // the caller owns o and the residual add
            if (!DROP(32)) {
            { MvR m{}; m.nseg = 1; m.seg[0] = { L.o, P.x, D / 32, 1 }; m.total_tiles = D / 32; ph_matvec_pk<NB_V, KSV, TT, false, SM>(c, m, P.xq, P.xs, P.xsum, nrows); }
            SYNC();
            }
        }
        if constexpr (PART != 0) return;
        if (stage >= 0) return;
        { PrepR p{}; p.x = P.x; p.kvec = L.post_norm_s; p.eps = NORM_EPS; p.n = D; p.flags = PREP_NORM | PREP_HADAMARD | PREP_QUANT;
          p.xq = P.xq; p.xs = P.xs; p.xsum = P.xsum; ph_prep(c, p, nrows); }
        if (KS40 > 1) { zero_span(c, P.gu, nrows * FF); zero_span(c, P.gu + (size_t) FMAX * FF, nrows * FF); }
        SYNC();
        if (!DROP(128)) {
        { MvR m{}; m.nseg = 2; m.seg[0] = { L.gate, P.gu, FF / 32, 0 }; m.seg[1] = { L.up, P.gu + (size_t) FMAX * FF, FF / 32, 0 }; m.total_tiles = 2 * FF / 32;
          ph_matvec_pk<NB_D, KS40, TT, false, SM>(c, m, P.xq, P.xs, P.xsum, nrows); }
        SYNC();
        }
        { PrepR p{}; p.x = P.gu; p.x2 = P.gu + (size_t) FMAX * FF; p.kvec = P.signs17408; p.n = FF; p.flags = PREP_SILU_MUL | PREP_HADAMARD | PREP_QUANT;
          p.xq = P.xq; p.xs = P.xs; p.xsum = P.xsum; ph_prep(c, p, nrows); }
        SYNC();
        if (!DROP(64)) {
        { MvR m{}; m.nseg = 1; m.seg[0] = { L.down, P.x, D / 32, 1 }; m.total_tiles = D / 32; ph_matvec_pk<NB_FF, KSFF, TT, false, SM>(c, m, P.xq, P.xs, P.xsum, nrows); }
        SYNC();
        }
    }
    if (PART == 0 && P.with_logits && !DROP(1)) {
        { PrepR p{}; p.x = P.x; p.kvec = P.output_norm_s; p.norm_w = P.output_norm; p.eps = NORM_EPS; p.n = D; p.flags = PREP_NORM | PREP_HADAMARD | PREP_QUANT;
          p.xq = P.xq; p.xs = P.xs; p.xsum = P.xsum; p.store_normed = P.hfinal; ph_prep(c, p, nrows); }
        SYNC();
        { MvR m{}; m.nseg = 1; m.seg[0] = { P.output, P.logits, VOCAB / 32, 0 }; m.total_tiles = VOCAB / 32; ph_matvec_pk<NB_D, 1, TT, false, SM>(c, m, P.xq, P.xs, P.xsum, nrows); }
        SYNC();
        if (!DROP(256)) {
        ph_argmax_slices(c, P);
        SYNC();
        }
        if ((int) blockIdx.x < nrows) argmax_final(P.amax_val + blockIdx.x * AMAX_SLICES, P.amax_idx + blockIdx.x * AMAX_SLICES, P.argmax_out + blockIdx.x, c.lds);
    }
}

// The deployed FFN, one layer, as its own kernel. It reads no row metadata, no block cache and no
// recurrent state, so its row count is bounded by `FMAX` rather than by `RMAX`, and
// `ph_matvec_auto` turns that row count into column groups: sixteen rows fill the matrix
// instruction once, and every further sixteen take another group against the same peeled weight
// block. The body is here, above the probe guard, so that a probe build can instantiate it without
// the fifty `k_forward_rows` shapes below.
// OCC is the minimum waves per SIMD32 asked of the register allocator, and on this kernel it is the
// grid. A workgroup is 256 threads, eight waves over a WGP's four SIMDs, so `coop_grid` gives out
// whole workgroups: waves/SIMD32 of 5, 6, 7, 8 buy 2, 3, 3 and 4 workgroups per WGP, which is 40,
// 60, 60 and 80 blocks. The two-group body was fitted to 6 and the one-group body asked for
// nothing, so both have been running three workgroups per WGP. Asking for eight is what reaches
// four, and `tools/kernel_resources.py -D HALO_SLICE_PROBE=32 -D HALO_SLICE_WPE=8` prices it in
// three seconds: 238 VGPR at six waves, 205 at seven with nothing spilled, 192 at eight with 18
// spilled, and the spill is in the drain rather than the block loop.
//
// IT LOSES ANYWAY, AND NOT BECAUSE OF THE SPILL. The allocator pays for the fourth workgroup inside
// the block loop: 445 -> 518 instructions at one group with nothing spilled at all, 565 -> 614 at
// two, with the same 16 or 32 matrix instructions and the same 10 or 20 global loads. A third more
// blocks does not cover a sixth more issue, and the measured phase is 2.6% and 9.8% worse.
// docs/ffn-slice-grid.md has the panel; `-DHALO_SLICE_OCC_CONTROL=1` rebuilds both arms for a
// re-fit, which is worth doing after anything that shortens this loop.
#ifndef HALO_SLICE_WPE
#define HALO_SLICE_WPE 6
#endif
#ifndef HALO_SLICE_WPE_LOW
#define HALO_SLICE_WPE_LOW 1
#endif
// What this kernel's shared block actually has to hold. `LDS_FLOATS` is the union every phase in
// `phases.hpp` can ask for, and its largest term is an attention score array this kernel never
// runs: a slice is prep, matvec, prep, matvec and nothing else. `prep_chunk_r` stages 1024 floats,
// the dot4 matvec reduces through `[8][TT][32]` and the WMMA body through `[7][8][32]` plus a
// 32-float scale table per wave. THAT IS THE GRID FOR THE NARROW WIDTHS: their registers allow
// eight workgroups per WGP and the occupancy API hands out 65536 / bytes-per-block, so 9508 bytes
// is six workgroups and 8228 is seven.
constexpr int slice_lds_floats(int TT, int ARM = 0) {
    // arms 6 and 7 carry a second reduction buffer, arm 7 an eighth slot in each
    if ((ARM == MVW_DBUF || ARM == MVW_FLAT) && TT > MV_DOT4_MAX)
        return 2 * mvw_red_slots(ARM) * 8 * 32 + NW * 32;
    const int mv = TT > MV_DOT4_MAX ? (NW - 1) * 8 * 32 + NW * 32 : 8 * TT * 32;
    return mv > 1024 ? mv : 1024;   // prep_chunk_r's staging is the floor
}
static_assert(slice_lds_floats(FMAX) <= LDS_FLOATS && slice_lds_floats(1) <= LDS_FLOATS, "slice block outgrew the union");

template <int TT, int SM = 0, int OCC = (TT > 16 ? HALO_SLICE_WPE : HALO_SLICE_WPE_LOW), int MVORD = 0, int MVARM = 0>
__global__ void __launch_bounds__(NT) __attribute__((amdgpu_waves_per_eu(OCC))) k_ffn_slice(FwdParams P, int layer) {
    constexpr int KS40 = SM >= 3 ? 5 : 1;
    constexpr int KSFF = SM >= 3 ? 17 : 2;
    __shared__ __attribute__((aligned(16))) float lds[slice_lds_floats(TT, MVARM)];
    // A workgroup gets 65536 / bytes-per-block of the WGP, and this kernel's registers already cap
    // it at three; arm 6's 15392-byte block still allows four, so the second buffer costs no grid.
    static_assert(slice_lds_floats(TT, MVARM) * 4 + 64 <= 65536 / 3, "slice block outgrew three workgroups per WGP");
    __shared__ float red[8];
    __shared__ int s_unit;
    Ctx c { P.bar, P.work, nullptr, 0, 0, true, lds, red, &s_unit };
    const int nrows = P.nrows;
    const LayerW & L = r_layers[layer];
        { PrepR p{}; p.x = P.x; p.kvec = L.post_norm_s; p.eps = NORM_EPS; p.n = D; p.flags = PREP_NORM | PREP_HADAMARD | PREP_QUANT;
          p.xq = P.xq; p.xs = P.xs; p.xsum = P.xsum; ph_prep(c, p, nrows); }
        if (KS40 > 1) { zero_span(c, P.gu, nrows * FF); zero_span(c, P.gu + (size_t) FMAX * FF, nrows * FF); }
        SYNC();
        { MvR m{}; m.nseg = 2; m.seg[0] = { L.gate, P.gu, FF / 32, 0 }; m.seg[1] = { L.up, P.gu + (size_t) FMAX * FF, FF / 32, 0 }; m.total_tiles = 2 * FF / 32;
          ph_matvec_auto<NB_D, KS40, TT, false, SM, MVORD, MVARM, true>(c, m, P.xq, P.xs, P.xsum, nrows); }
        SYNC();
        { PrepR p{}; p.x = P.gu; p.x2 = P.gu + (size_t) FMAX * FF; p.kvec = P.signs17408; p.n = FF; p.flags = PREP_SILU_MUL | PREP_HADAMARD | PREP_QUANT;
          p.xq = P.xq; p.xs = P.xs; p.xsum = P.xsum; ph_prep(c, p, nrows); }
        SYNC();
        { MvR m{}; m.nseg = 1; m.seg[0] = { L.down, P.x, D / 32, 1 }; m.total_tiles = D / 32; ph_matvec_auto<NB_FF, KSFF, TT, false, SM, MVORD, MVARM, true>(c, m, P.xq, P.xs, P.xsum, nrows); }
        SYNC();
}

#if defined(HALO_ROWS_PROBE) || defined(HALO_SLICE_PROBE)
// Resource probe. `k_forward_rows` is what sets the cooperative grid of every wide pass, and
// register allocation over a body this large is the slowest thing the compiler does here: the whole
// translation unit is about fifty instantiations and a minute of hipcc, which is too slow to search
// against. `-DHALO_ROWS_PROBE=8 -DHALO_ROWS_PROBE_PK=0 -DHALO_ROWS_PROBE_OCC=12` compiles exactly
// one shape and stops, which tools/kernel_resources.py turns into a VGPR and occupancy reading in
// nine seconds and no GPU. Pair it with -DHALO_ROWS_DROP to attribute the mark to a phase.
//
// `-DHALO_SLICE_PROBE=32` does the same for one `k_ffn_slice` width, and that one is three seconds
// rather than a minute because it also cuts the `k_forward_rows` ladder. It is how the column
// groups, the wave budget and the shared-block size of this kernel are priced without a GPU:
// `tools/kernel_resources.py -D HALO_SLICE_PROBE=8 kernels/halo_rows.hip k_ffn_slice` reports the
// registers, the spills and the LDS bytes that decide the grid.
#ifndef HALO_ROWS_PROBE_PK
#define HALO_ROWS_PROBE_PK 0
#endif
#ifndef HALO_ROWS_PROBE_OCC
#define HALO_ROWS_PROBE_OCC 1
#endif
#ifdef HALO_SLICE_PROBE
// `-D HALO_SLICE_PROBE_ARM=1` prices the drain arm and `=2..4` the footprint ablations on the same
// three-second compile, which is how the register cost of a unit shape is known before a panel.
#ifndef HALO_SLICE_PROBE_ARM
#define HALO_SLICE_PROBE_ARM 0
#endif
template __global__ void k_ffn_slice<HALO_SLICE_PROBE, 0, (HALO_SLICE_PROBE > 16 ? HALO_SLICE_WPE : HALO_SLICE_WPE_LOW), 0, HALO_SLICE_PROBE_ARM>(FwdParams, int);
#else
template __global__ void k_forward_rows<HALO_ROWS_PROBE, 0, 0, 0, HALO_ROWS_PROBE_PK != 0, HALO_ROWS_PROBE_OCC>(FwdParams, int);
#endif
}   // namespace halo
#else

// Both budgets of a shape in one executable is what makes the A/B an in-process pair, and it is
// also two more instantiations of the largest body in this repository, which every engineer waits
// for on every build. So the default build carries exactly the arms it launches - the packed shape
// on the wide budget, the fp32 shape on the allocator's - and -DHALO_ROWS_OCC_CONTROL=1 adds the
// other arm of each so `--pk-occ 0,1` can interleave them. docs/rows-grid-occupancy.md says which
// numbers came from which build.
#ifndef HALO_ROWS_OCC_CONTROL
#define HALO_ROWS_OCC_CONTROL 0
#endif

// ti 0..2 are rows 1, 2 and 4; ti 3 is the five-to-eight row shape.
static const void * rows_fn_narrow3(int ti, bool commit, bool pk);

static const void * rows_fn_narrow(int ti, bool commit, bool pk) {
    if (ti == 3) return rows_fn_narrow3(3, commit, pk);
    if (pk) {
        static const void * f[2][3] = {
            { (const void *) k_forward_rows<1, 0, 0, 0, true>, (const void *) k_forward_rows<ROWS_TT2, 0, 0, 0, true>,
              (const void *) k_forward_rows<ROWS_TT4, 0, 0, 0, true> },
            { (const void *) k_forward_rows<1, 0, 0, 1, true>, (const void *) k_forward_rows<ROWS_TT2, 0, 0, 1, true>,
              (const void *) k_forward_rows<ROWS_TT4, 0, 0, 1, true> },
        };
        return f[commit][ti];
    }
    static const void * f[2][3] = {
        { (const void *) k_forward_rows<1>, (const void *) k_forward_rows<ROWS_TT2>, (const void *) k_forward_rows<ROWS_TT4> },
        { (const void *) k_forward_rows<1, 0, 0, 1>, (const void *) k_forward_rows<ROWS_TT2, 0, 0, 1>,
          (const void *) k_forward_rows<ROWS_TT4, 0, 0, 1> },
    };
    return f[commit][ti];
}

// The eight-row shape at the allocator's own budget, which is what every published number before
// docs/rows-grid-occupancy.md was measured on.
//
// The fp32 arm stays in EVERY build, unlike the packed one, and the two extra instantiations are
// bought deliberately. fp32 is the coordinate that ships, `--pk-occ 0,1` is the only instrument in
// the tree that can order two register budgets under one clock, and for fp32 it was a silent no-op:
// this function was reached for both arms, so the panel that should have caught 7eeb556's move read
// two identical cells (143.8 against 144.1 tok/s, verify 57.817 against 57.784 ms) and said the
// budget did not matter. A trap that answers is worse than a missing instrument. The packed
// coordinate keeps its control behind -DHALO_ROWS_OCC_CONTROL=1, where asking for a budget the
// build does not carry gets the route and HALO_ROWS_GRID says so.
static const void * rows_fn_narrow3(int ti, bool commit, bool pk) {
    (void) ti;
    if (pk) {
#if HALO_ROWS_OCC_CONTROL
        static const void * f[2] = { (const void *) k_forward_rows<8, 0, 0, 0, true>,
                                     (const void *) k_forward_rows<8, 0, 0, 1, true> };
        return f[commit];
#else
        static const void * f[2] = { (const void *) k_forward_rows<8, 0, 0, 0, true, PK_OCC_WIDE>,
                                     (const void *) k_forward_rows<8, 0, 0, 1, true, PK_OCC_WIDE> };
        return f[commit];
#endif
    }
    static const void * f[2] = { (const void *) k_forward_rows<8>, (const void *) k_forward_rows<8, 0, 0, 1> };
    return f[commit];
}

// TAKING A KERNEL'S ADDRESS INSIDE A HOST FUNCTION TEMPLATE DOES NOT EMIT IT. `rows_fn_wide<OCC>`
// is the only reference to the second budget's bodies, and a host template's body is not
// instantiated in the device pass, so a build that compiles and links cleanly loses the kernels and
// fails at the first launch with "Cannot find Symbol with name: _ZN4halo14k_forward_rowsILi8...".
// The first budget survives because `rows_fn_narrow3` - an ordinary function - names it too. These
// four explicit instantiations are what put the second budget's bodies in the device image, and
// they are the reason HALO_ROWS_OCC2 defaults off: they are four more compiles of the largest body
// in this repository on every engineer's build.
#ifndef HALO_ROWS_OCC2
#define HALO_ROWS_OCC2 0
#endif
#if HALO_ROWS_OCC2
template __global__ void k_forward_rows<8, 0, 0, 0, false, PK_OCC_WIDE2>(FwdParams, int);
template __global__ void k_forward_rows<8, 0, 0, 1, false, PK_OCC_WIDE2>(FwdParams, int);
template __global__ void k_forward_rows<8, 0, 0, 0, true, PK_OCC_WIDE2>(FwdParams, int);
template __global__ void k_forward_rows<8, 0, 0, 1, true, PK_OCC_WIDE2>(FwdParams, int);
#endif

// The occupancy arm. `pk_occ` picks which register budget the pass runs under, and only the shapes
// whose allocation crosses a whole block per WGP get a second instantiation. The ladder this rule
// was fitted against is superseded - see docs/rows-grid-occupancy.md for the co-residency the
// hardware actually gives, measured by bench/occ_resident, and for why the occupancy API's answer
// is a ceiling on what can be launched even where it is wrong.
template <int OCC>
static const void * rows_fn_wide(int ti, bool commit, bool pk) {
    // Only the eight-row shape gets a second budget, because it is the only one whose grid the
    // budget can change: rows 1 to 4 take ROWS_GRID_NARROW = 64 workgroups, which every
    // instantiation here already holds, and a second instantiation of a body this size costs about
    // eight seconds of every engineer's build.
    if (pk) {
        static const void * f[2] = { (const void *) k_forward_rows<8, 0, 0, 0, true, OCC>,
                                     (const void *) k_forward_rows<8, 0, 0, 1, true, OCC> };
        return ti == 3 ? f[commit] : rows_fn_narrow(ti, commit, true);
    }
    // fp32 is on the wide budget too, and it is a route rather than a control because the body
    // moved under it. When this arm was written the fp32 eight-row shape allocated 135 registers,
    // which is the same five blocks per WGP as 120, so the ask bought no grid and paid its spills
    // for nothing. 7eeb556's attention split left it at 141, which is four blocks, and the same ask
    // now buys a whole block back. docs/rows-grid-occupancy.md carries both panels.
    static const void * f[2] = { (const void *) k_forward_rows<8, 0, 0, 0, false, OCC>,
                                 (const void *) k_forward_rows<8, 0, 0, 1, false, OCC> };
    return ti == 3 ? f[commit] : rows_fn_narrow(ti, commit, false);
}

// The measured rule. Asking costs registers and buys workgroups, and what decides the answer is the
// register count of the eight-row body, which moves whenever a phase inside it does:
//
//   fp32 state   141 VGPR is FOUR blocks per WGP and 120 is five, so the ask buys a quarter more
//                grid. It bought nothing when the same kernel allocated 135, and this rule returned
//                0 for fp32 until 7eeb556 moved the body: the register count is the input, not the
//                coordinate. docs/rows-grid-occupancy.md carries the fp32 panel beside the packed
//                one and the re-fit procedure that produced it.
//   packed state 146 VGPR is four blocks and 120 is five. The ask buys a quarter more grid:
//                +20% deployed prompt ingestion, verify 57.5 -> 47.2..52.0 ms.
//
// Both coordinates now want the wide budget for the same reason, which is what the rule is for: it
// records a measurement about registers, so re-fit it with tools/rows_occ_probe whenever a phase
// body inside k_forward_rows changes, and do not read it as a property of the state format.
static int rows_occ_rule(bool pk) { (void) pk; return HALO_ROWS_OCC_RULE; }
static constexpr int ROWS_OCC_ARMS = 3;   // allocator, PK_OCC_WIDE, PK_OCC_WIDE2

static const void * rows_fn_occ(int ti, bool commit, bool pk, int occ) {
    if (occ < 0) occ = rows_occ_rule(pk);
#if HALO_ROWS_OCC2
    if (occ >= 2) return rows_fn_wide<PK_OCC_WIDE2>(ti, commit, pk);
#else
    if (occ >= 2) occ = 1;   // the build does not carry the second budget; HALO_ROWS_GRID says so
#endif
    return occ ? rows_fn_wide<PK_OCC_WIDE>(ti, commit, pk) : rows_fn_narrow(ti, commit, pk);
}

static int rows_ti(int nrows) { return nrows <= 1 ? 0 : nrows <= 2 ? 1 : nrows <= 4 ? 2 : 3; }

// The grid of the kernel a wide pass actually launches. This used to be a minimum over all four TT
// instantiations, which is the right answer only while the eight-row kernel is the worst of them:
// it always has been, and the narrow shapes are capped at ROWS_GRID_NARROW anyway, so the minimum
// was doing nothing except forcing every shape to hold every other shape's registers.
// A packed state is its own set of kernels and its own grid.
static int rows_grid_size(bool pk, int occ) {
    return coop_grid(rows_fn_occ(3, false, pk, occ));
}

// The direct-commit kernels are the deployed bodies with the ring and state stores moved to the end
// of the pass, so they are selected by row count in the same way.
static int commit_grid_size(bool pk, int occ) {
    return coop_grid(rows_fn_occ(3, true, pk, occ));
}

// Single-token maps are their own kernels, so each gets its own occupancy-derived grid and the
// deployed path keeps exactly the grid it has today.
static const void * rows_fn_packed(const FwdParams & p) {
    if (p.commit_state) return rows_fn_occ(rows_ti(p.nrows), true, true, p.pk_occ);
    if (p.nrows <= 1 && p.single_map > 0) {
        switch (p.single_map) {
            case 1: return (const void *) k_forward_rows<1, 1, 0, 0, true>;
            case 2: return (const void *) k_forward_rows<1, 2, 0, 0, true>;
            case 3: return (const void *) k_forward_rows<1, 3, 0, 0, true>;
            default: return (const void *) k_forward_rows<1, 0, 0, 0, true>;
        }
    }
    return rows_fn_occ(rows_ti(p.nrows), false, true, p.pk_occ);
}

// A deferred commit leaves a base plus its pending rank-1 updates in the state region, and only the
// resident kernels rebuild that sum. Every other reader - the persistent kernel's whole-pass route,
// a sliced recurrent layer, the sliced state route - would read a base several steps stale, so a
// pass on one of those routes commits the slots it is about to read first. A served workload mixes
// them freely: the batch drivers prefill a stream in narrow passes and then step it wide.
//
// Which slots hold pending updates is host state because the host is what has to decide, and the
// rule is the one the convolution kernel applies: a pass of one token per sequence appends, and
// anything else commits. A reset zeroes the region, which is an empty pending list in every format.
static bool gdn_any_pending() { for (int s = 0; s < MAXSLOTS; s++) if (gdn_pend_slot[s]) return true; return false; }

static void gdn_defer_flush_slot(float * gdn_state, int slot, hipStream_t st) {
    constexpr int LAYERS = 48, SPLIT = 4;
    gdn_defer_flush_k<SPLIT><<<dim3(LAYERS * HV * SPLIT), 256, 0, st>>>(gdn_state, slot, LAYERS);
    gdn_defer_clear_k<<<dim3((LAYERS + 255) / 256), 256, 0, st>>>(gdn_state, slot, LAYERS);
    gdn_pend_slot[slot] = 0;
}

// Commits every slot this pass is about to read on a route that cannot rebuild one.
static void gdn_defer_settle(const FwdParams & p, hipStream_t st) {
    if (gdn_defer_effective() < 2 || !gdn_any_pending()) return;
    for (int i = 0; i < p.nseq; i++) {
        const int slot = p.rows[p.seqs[i].row0].seq;
        if (slot >= 0 && slot < MAXSLOTS && gdn_pend_slot[slot]) gdn_defer_flush_slot(p.gdn_state, slot, st);
    }
}

// Commits one slot for a caller outside this file: the single-sequence reference route reads a
// state region directly rather than through the codec, so it has to see a settled one. It costs
// one no-op test when the slot has nothing pending, which is every process that does not defer.
void gdn_defer_flush_pending(float * gdn_state, int slot, hipStream_t st) {
    if (gdn_defer_effective() < 2 || slot < 0 || slot >= MAXSLOTS || !gdn_pend_slot[slot]) return;
    gdn_defer_flush_slot(gdn_state, slot, st);
}

// A reset or a slot copy moves the pending list with the region it lives in.
void gdn_defer_note_reset(int slot) { if (slot >= 0 && slot < MAXSLOTS) gdn_pend_slot[slot] = 0; }
void gdn_defer_note_copy(int dst, int src) {
    if (dst >= 0 && dst < MAXSLOTS && src >= 0 && src < MAXSLOTS) gdn_pend_slot[dst] = gdn_pend_slot[src];
}

static const void * rows_fn(const FwdParams & p) {
    if (gdn_state_format() != GDN_STATE_F32) return rows_fn_packed(p);
    if (p.commit_state)   // commit takes the deployed arithmetic map; single_map is a decode path
        return rows_fn_occ(rows_ti(p.nrows), true, false, p.pk_occ);
    if (p.nrows <= 1 && p.single_map > 0) {
        switch (p.single_map) {
            case 1: return (const void *) k_forward_rows<1, 1>;
            case 2: return (const void *) k_forward_rows<1, 2>;
            case 3: return (const void *) k_forward_rows<1, 3>;
            default: return (const void *) k_forward_rows<1>;   // map 4: deployed map, own grid
        }
    }
    return rows_fn_occ(rows_ti(p.nrows), false, false, p.pk_occ);
}
// The deployed path launches on the min grid over the TT instantiations, so the 217-VGPR TT 8
// kernel sets the workgroup count for every row count including one. Every single_map kernel is
// its own function and gets the grid its own occupancy allows, which is a second variable in any
// comparison against mode 0. single_map 4 isolates it: same kernel as mode 0, own grid.
// Below the occupancy limit the workgroup count is a free scheduling choice: a unit is a whole
// 32-row tile (and, where KS > 1, one K part of it) computed by one workgroup, so which workgroup
// runs it changes no operand, no order and no output bit. What it does change is how the phase's
// units divide: a phase of U units over G workgroups costs ceil(U/G) rounds, and this model's unit
// counts are all multiples of 64 (gate/up 1088, qkv+z 512, q/k/v 448, the three 5120-row matvecs
// 320, GDN state 192), so a grid that does not divide them wastes most of a round. G = 60 is the
// worst value in its neighbourhood: 92.8% of the bandwidth its own quantisation allows, against
// 98.1% at 40. See docs/single-stream.md.
// The narrow and wide row counts want different grids, and one number served both until the dot4
// crossover took k_forward_rows<8> from 217 VGPRs to 136: the eight-row kernel's occupancy was the
// minimum of the four, so it set the workgroup count for single-token decode as well, and the
// deployed grid was 60 because that kernel could hold no more.
//
// Eight rows want all of it. The drafted verify pass measures 62.8, 55.6, 52.6 and 49.8 ms at 60,
// 72, 84 and 100 workgroups (docs/matvec-crossover.md), a 21% range - fifty times the +-0.4% the
// same grid moves at one row. One row does not: it loses 1.6% on the same build when the ceiling
// rises to 100 and nothing holds it back.
//
// ROWS_GRID_NARROW keeps rows 1..4 off the wide ceiling. docs/single-stream.md measured 48 at
// +0.43% and 40 at -0.81% against 60 and left the value open; docs/matvec-phase-length.md closes
// it at **64**, which is the one point in that neighbourhood nobody had tried and the only one that
// divides the work. Every unit count on this path is a multiple of 64 - gate/up 1088, qkv+z 512,
// q/k/v 448, the three 5120-row matvecs 320, the GDN state 192 - so 64 workgroups deal every phase
// in whole rounds and 60 wastes a third of a round on the 320-unit phases.
//
// It is worth +0.25%, not the +7.7% that quantisation ceiling predicts: four paired in-process
// panels (HALO_BENCH_GRID=60,64) read +0.17, +0.37, +0.21 and +0.24%, with 66 of 105 tokens
// favouring 64. The gain is real and the model behind it is not - see the document. 80 and 100 lose
// 1.4 and 1.0% on the same instrument. Bit-exact: 248,320 logits at grid 60 and 64 hash identically
// (sha256 a1ba88f9...), which is what lets a serving default move on a 0.25% measurement at all.
static constexpr int ROWS_GRID_NARROW = 64;
static constexpr int ROWS_GRID_NARROW_MAX = 4;   // rows per pass that take the narrow grid

// Split K parts now run inside one workgroup per output tile in part order. A dynamic unit
// counter cannot guarantee their order, even at a grid smaller than the tile count. Retain the
// old grid budget until the ordered drain's throughput has been measured at wider grids.
static constexpr int ROWS_GRID_BUDGET = D / 32;

static int rows_grid_for(const FwdParams & p) {
    const int pk = gdn_state_format() != GDN_STATE_F32;
    const int occ = p.pk_occ < 0 ? rows_occ_rule(pk) : (p.pk_occ > 2 ? 2 : p.pk_occ);
    static int base[2][ROWS_OCC_ARMS] = {};
    static int own[2][SINGLE_MAP_MAX + 1] = {};
    static int commit[2][ROWS_OCC_ARMS] = {};
    if (!base[pk][occ]) {
        base[pk][occ] = rows_grid_size(pk, occ);
        // Keep the previous occupancy budget while comparing the ordered and split drains.
        if (base[pk][occ] >= ROWS_GRID_BUDGET) {
            hipDeviceProp_t prop{}; (void) hipGetDeviceProperties(&prop, 0);
            const int wgp = prop.multiProcessorCount > 0 ? prop.multiProcessorCount : 1;
            const int held = (ROWS_GRID_BUDGET - 1) / wgp * wgp;
            base[pk][occ] = held > 0 ? held : ROWS_GRID_BUDGET - 1;
        }
        // What the register budget actually bought, which no compiler remark can tell you: the
        // remark knows VGPRs and LDS, the runtime knows what this device will really co-schedule.
        if (getenv("HALO_ROWS_GRID")) {
            hipDeviceProp_t prop{}; (void) hipGetDeviceProperties(&prop, 0);
            const int wgp = prop.multiProcessorCount > 0 ? prop.multiProcessorCount : 1;
            fprintf(stderr, "[rows grid] %s state, budget %s: base %d workgroups (%d per WGP, %d waves/SIMD32), narrow %d, grid budget %d\n",
                    pk ? "packed" : "fp32", occ ? "wide" : "allocator", base[pk][occ],
                    base[pk][occ] / wgp, base[pk][occ] / wgp * NW / 4,
                    std::min(base[pk][occ], ROWS_GRID_NARROW), ROWS_GRID_BUDGET);
        }
    }
    if (p.grid_rows > 0 && p.grid_rows <= base[pk][occ]) return p.grid_rows;
    const int room = p.nrows <= ROWS_GRID_NARROW_MAX ? std::min(base[pk][occ], ROWS_GRID_NARROW) : base[pk][occ];
    // A commit pass decomposes its work exactly like the replay pass it replaces, so it keeps the
    // deployed grid unless its own occupancy is lower.
    if (p.commit_state) {
        if (!commit[pk][occ]) commit[pk][occ] = std::min(base[pk][occ], commit_grid_size(pk, occ));
        return std::min(room, commit[pk][occ]);
    }
    const int sm = p.nrows <= 1 ? std::min(std::max(p.single_map, 0), SINGLE_MAP_MAX) : 0;
    if (!sm) return room;
    if (!own[pk][sm]) own[pk][sm] = coop_grid(rows_fn(p));
    return std::min(room, own[pk][sm]);   // single_map kernels have one budget, so `occ` is not theirs
}

void set_layer_table_rows(const LayerW * host_layers) {
    hipError_t e = hipMemcpyToSymbol(HIP_SYMBOL(r_layers), host_layers, sizeof(LayerW) * NLAYER);
    if (e != hipSuccess) { fprintf(stderr, "layer table upload failed: %s\n", hipGetErrorString(e)); abort(); }
    gdn_fmt_upload();   // once the device exists, before any pass reads a state region
    gdn_defer_upload();
    gdn_region_upload();
}

void launch_forward_rows(const FwdParams & p, hipStream_t st) {
    gdn_defer_settle(p, st);
    int stage = -2;
    void * args[] = { (void *) &p, &stage };
    hipError_t e = hipLaunchCooperativeKernel(rows_fn(p), dim3(rows_grid_for(p)), dim3(NT), args, 0, st);
    if (e != hipSuccess) { fprintf(stderr, "cooperative launch failed: %s\n", hipGetErrorString(e)); abort(); }
}

void launch_forward_slice(const FwdParams & p, int stage, hipStream_t st) {
    // Layers 3, 7, 11 ... are attention; every other layer runs the recurrence in this body.
    if (stage >= 0 && stage < NLAYER && stage % 4 != 3) gdn_defer_settle(p, st);
    void * args[] = { (void *) &p, &stage };
    hipError_t e = hipLaunchCooperativeKernel(rows_fn(p), dim3(rows_grid_for(p)), dim3(NT), args, 0, st);
    if (e != hipSuccess) { fprintf(stderr, "slice launch failed: %s\n", hipGetErrorString(e)); abort(); }
}

// ---- one layer in two halves, around a projection the caller owns (see halo_kernels.h) ----
static const void * part_fn(int part, int ti, int cm, int pk) {
    // Only the state half reads or writes a state region; the input prep and the head prep are the
    // same kernels in every format.
    if (pk && part == SEQ_PART_STATE) {
        static const void * f[2][4] = {
            { (const void *) k_forward_rows<1, 0, 2, 0, true>, (const void *) k_forward_rows<ROWS_TT2, 0, 2, 0, true>,
              (const void *) k_forward_rows<ROWS_TT4, 0, 2, 0, true>, (const void *) k_forward_rows<8, 0, 2, 0, true> },
            { (const void *) k_forward_rows<1, 0, 2, 1, true>, (const void *) k_forward_rows<ROWS_TT2, 0, 2, 1, true>,
              (const void *) k_forward_rows<ROWS_TT4, 0, 2, 1, true>, (const void *) k_forward_rows<8, 0, 2, 1, true> },
        };
        return f[cm][ti];
    }
    if (part == SEQ_PART_PREP) {   // input prep only: no state, so commit_state cannot apply
        static const void * f[4] = { (const void *) k_forward_rows<1, 0, 1>, (const void *) k_forward_rows<ROWS_TT2, 0, 1>,
                                     (const void *) k_forward_rows<ROWS_TT4, 0, 1>, (const void *) k_forward_rows<8, 0, 1> };
        return f[ti];
    }
    if (part == SEQ_PART_HEAD) {   // output-head prep only, after the last layer
        static const void * f[4] = { (const void *) k_forward_rows<1, 0, 3>, (const void *) k_forward_rows<ROWS_TT2, 0, 3>,
                                     (const void *) k_forward_rows<ROWS_TT4, 0, 3>, (const void *) k_forward_rows<8, 0, 3> };
        return f[ti];
    }
    static const void * f[2][4] = {
        { (const void *) k_forward_rows<1, 0, 2, 0>, (const void *) k_forward_rows<ROWS_TT2, 0, 2, 0>,
          (const void *) k_forward_rows<ROWS_TT4, 0, 2, 0>, (const void *) k_forward_rows<8, 0, 2, 0> },
        { (const void *) k_forward_rows<1, 0, 2, 1>, (const void *) k_forward_rows<ROWS_TT2, 0, 2, 1>,
          (const void *) k_forward_rows<ROWS_TT4, 0, 2, 1>, (const void *) k_forward_rows<8, 0, 2, 1> },
    };
    return f[cm][ti];
}

void launch_sequence_part(const FwdParams & p, int layer, int part, hipStream_t st) {
    const bool head = part == SEQ_PART_HEAD;
    if ((head ? layer != NLAYER : (layer < 0 || layer >= NLAYER))
        || (part != SEQ_PART_PREP && part != SEQ_PART_STATE && !head)
        || p.nrows < 1 || p.nrows > RMAX || p.nseq < 1 || p.nseq > MAXSEQ) {
        fprintf(stderr, "sequence part outside its contract: layer %d part %d rows %d seqs %d\n", layer, part, p.nrows, p.nseq);
        abort();
    }
    if (part == SEQ_PART_STATE && layer % 4 != 3) gdn_defer_settle(p, st);
    const int ti = p.nrows <= 1 ? 0 : p.nrows <= 2 ? 1 : p.nrows <= 4 ? 2 : 3;
    const int cm = part == SEQ_PART_STATE && p.commit_state ? 1 : 0;
    const int pk = gdn_state_format() != GDN_STATE_F32;
    const void * fn = part_fn(part, ti, cm, pk);
    // Each part is a short single-layer body that shares its launch with nothing, so it takes the
    // whole grid its own registers allow rather than the minimum over the whole-model kernels.
    static int grid[SEQ_PART_HEAD + 1][4][2][2] = {};
    int & g = grid[part][ti][cm][pk];
    if (!g) g = coop_grid(fn);
    int stage = layer;
    void * args[] = { (void *) &p, &stage };
    hipError_t e = hipLaunchCooperativeKernel(fn, dim3(g), dim3(NT), args, 0, st);
    if (e != hipSuccess) { fprintf(stderr, "sequence part launch failed: %s\n", hipGetErrorString(e)); abort(); }
}

// The wide register budget: eight waves per SIMD32 is the fourth workgroup per WGP.
constexpr int FFN_OCC_WIDE = 8;
// Which budget the widths that carry matrix groups run under when nothing pins an arm. Arm 0 is
// what shipped - no ask at sixteen rows, six waves at thirty-two.
#ifndef HALO_SLICE_OCC_RULE
#define HALO_SLICE_OCC_RULE 0
#endif
constexpr int FFN_OCC_RULE = HALO_SLICE_OCC_RULE;
constexpr int FFN_OCC16[2] = { HALO_SLICE_WPE_LOW, FFN_OCC_WIDE };
constexpr int FFN_OCC32[2] = { HALO_SLICE_WPE, FFN_OCC_WIDE };
// The four-group width runs under the same budget as the two-group one, and it is not an
// assumption: at six waves per SIMD32 it allocates 240 VGPR against 238, with two spilled dwords
// and the same 8228-byte shared block, so it reaches the same three workgroups per WGP. It is
// built only under -DHALO_SLICE_WIDE=1, where `FMAX` is 64; in the default build `FMAX == FMID`
// and this names the same instantiation the deployed width already carries.
constexpr int FFN_OCC64[2] = { HALO_SLICE_WPE, FFN_OCC_WIDE };
#ifndef HALO_SLICE_OCC_CONTROL
#define HALO_SLICE_OCC_CONTROL 0
#endif

// Which body a row count takes. Above eight rows the 16-column matrix instruction carries a
// distinct row in every column, so the sixteen-row body does the same work for twice the rows, and
// above sixteen a second column group takes another sixteen against the same peeled weight block.
//
// Every rung is its own instantiation and its own grid, so widening the ladder cannot move a pass
// that does not use the new rung: a slice of 1..32 rows takes exactly the kernel it took before.
// Index 6 exists only in a `-DHALO_SLICE_WIDE=1` build, where `FMAX` is 64; otherwise `FMAX` is
// `FMID` and `launch_ffn_slice` has already rejected anything that could reach it.
static inline int slice_wi(int nrows) {
    return nrows <= 1 ? 0 : nrows <= 2 ? 1 : nrows <= 4 ? 2 : nrows <= 8 ? 3 : nrows <= 16 ? 4
         : nrows <= FMID ? 5 : 6;
}
constexpr int SLICE_WIDTHS = HALO_SLICE_WIDE ? 7 : 6;

// The register budget each width runs under. Arm 0 is what shipped; arm 1 asks the two bodies that
// carry matrix groups for eight waves per SIMD32, which is the fourth workgroup per WGP. The narrow
// bodies are the scalar-fed dot4 path at 73 VGPR and already sit at the occupancy API's ceiling, so
// there is nothing to ask them for and both arms select the same kernel.
// Which operand order the group-carrying widths run. `16259bba` built arm 1 - the first operand of
// every group after the first requested at the top of the block - against a cover table that says
// group 0's fragments get 260-288 slots and group 1's get 34-47. It is bit-identical by
// construction and both arms are in every build of those two widths, because the trade is a
// register against a stall and the answer differs with how much matrix work the block has to hide
// behind: 32 matrix instructions at two groups, 64 at four. docs/ffn-slice-rows.md carries the
// panel. The narrow widths carry one group and have nothing to hoist, so they select arm 0 always.
// It loses at both widths, so only -DHALO_MVW_ORDER_ARM=1 compiles it and the default build carries
// exactly the instantiations it carried before. `ffn_slice_order()` still reports what was asked,
// the way `ffn_slice_occ()` does without its own control flag, so a tool can walk the axis against
// any build and say what it got.
#ifndef HALO_MVW_ORDER_ARM
#define HALO_MVW_ORDER_ARM 0
#endif
static int g_mvw_order = [] { const char * v = getenv("HALO_MVW_ORDER"); return v ? (atoi(v) > 0 ? 1 : 0) : 0; }();
void ffn_slice_set_order(int ord) { g_mvw_order = ord > 0 ? 1 : 0; }
int ffn_slice_order() { return HALO_MVW_ORDER_ARM ? g_mvw_order : 0; }

// WHICH UNIT SHAPE THE GROUP-CARRYING WIDTHS RUN. Arm 0 is the deployed body. Arm 1 spreads the
// unit's drain over the eight waves that computed it instead of leaving four sequential passes on
// wave 0 - bit-identical by construction, the same change `86551bc8` shipped for `ph_matvec`, and
// it is in every build because the trade has no register in it. Arms 2..4 are the footprint
// ablations `ph_matvec_w` documents: they produce wrong numbers on purpose and only a
// `-DHALO_MVW_ABL=1` build carries them, so a binary that produces tokens cannot select one.
// `ffn_slice_arm()` reports what was asked whatever the build carries, so a tool can walk the axis
// and say what it got. docs/ffn-slice-activation.md.
#ifndef HALO_MVW_ABL
#define HALO_MVW_ABL 0
#endif
constexpr int MVW_ARM_MAX = 7;
// Which arms this build actually carries. The ablations compute wrong numbers, so a default build
// leaves them out and refuses to report an arm it is not running: a panel that asks for arm 3 on a
// binary without `-DHALO_MVW_ABL=1` gets arm 0 and a sample that says 0, rather than a deployed
// timing labelled as an ablation.
static inline bool mvw_arm_carried(int a) { return a == 0 || a == 1 || a == 6 || a == 7 || HALO_MVW_ABL; }
// Arm 7 is the default: bit-identical to arm 0 by construction and measured so, residual FNV-64
// 9171727463267762618 in every sample of both arms, at the same 60-workgroup grid.
// `HALO_MVW_ARM=0` restores the predecessor's drain for a control.
#ifndef HALO_MVW_ARM_DEFAULT
#define HALO_MVW_ARM_DEFAULT MVW_FLAT
#endif
static int g_mvw_arm = [] { const char * v = getenv("HALO_MVW_ARM"); const int n = v ? atoi(v) : HALO_MVW_ARM_DEFAULT;
                            return n >= 0 && n <= MVW_ARM_MAX && mvw_arm_carried(n) ? n : HALO_MVW_ARM_DEFAULT; }();
void ffn_slice_set_arm(int a) { g_mvw_arm = a >= 0 && a <= MVW_ARM_MAX && mvw_arm_carried(a) ? a : HALO_MVW_ARM_DEFAULT; }
int ffn_slice_arm() { return g_mvw_arm; }

template <int TT, int OCC, int ORD> static const void * slice_arm(int a) {
#if HALO_MVW_ABL
    switch (a) {
        case 1: return (const void *) k_ffn_slice<TT, 0, OCC, ORD, 1>;
        case 2: return (const void *) k_ffn_slice<TT, 0, OCC, ORD, 2>;
        case 3: return (const void *) k_ffn_slice<TT, 0, OCC, ORD, 3>;
        case 4: return (const void *) k_ffn_slice<TT, 0, OCC, ORD, 4>;
        case 5: return (const void *) k_ffn_slice<TT, 0, OCC, ORD, 5>;
        case 6: return (const void *) k_ffn_slice<TT, 0, OCC, ORD, 6>;
        case 7: return (const void *) k_ffn_slice<TT, 0, OCC, ORD, 7>;
        default: return (const void *) k_ffn_slice<TT, 0, OCC, ORD>;
    }
#else
    // Without the ablation build only the arms that compute the right answer exist.
    return a == 1 ? (const void *) k_ffn_slice<TT, 0, OCC, ORD, 1>
         : a == 6 ? (const void *) k_ffn_slice<TT, 0, OCC, ORD, 6>
         : a == 7 ? (const void *) k_ffn_slice<TT, 0, OCC, ORD, 7>
                  : (const void *) k_ffn_slice<TT, 0, OCC, ORD>;
#endif
}

template <int TT, int OCC> static const void * slice_ord(int ord, int a) {
#if HALO_MVW_ORDER_ARM
    return ord ? slice_arm<TT, OCC, 1>(a) : slice_arm<TT, OCC, 0>(a);
#else
    (void) ord; return slice_arm<TT, OCC, 0>(a);
#endif
}

static const void * slice_fn(int wi, int sm, int arm, int ord, int marm) {
    switch (wi) {
        // map 4 is the deployed body, and it selected the same kernel as map 0 to get a grid of its
        // own out of a launcher that shared one; now that every width has its own, the two are the
        // same launch.
        case 0: return sm == 1 ? (const void*) k_ffn_slice<1, 1> : sm == 2 ? (const void*) k_ffn_slice<1, 2>
                     : sm == 3 ? (const void*) k_ffn_slice<1, 3> : (const void*) k_ffn_slice<1>;
        case 1: return (const void*) k_ffn_slice<2>;
        case 2: return (const void*) k_ffn_slice<4>;
        case 3: return (const void*) k_ffn_slice<8>;
#if HALO_SLICE_OCC_CONTROL
        case 4: (void) ord; (void) marm; return arm ? (const void*) k_ffn_slice<16, 0, FFN_OCC16[1]> : (const void*) k_ffn_slice<16, 0, FFN_OCC16[0]>;
        case 5: (void) ord; (void) marm; return arm ? (const void*) k_ffn_slice<FMID, 0, FFN_OCC32[1]> : (const void*) k_ffn_slice<FMID, 0, FFN_OCC32[0]>;
        default: (void) ord; (void) marm; return arm ? (const void*) k_ffn_slice<FMAX, 0, FFN_OCC64[1]> : (const void*) k_ffn_slice<FMAX, 0, FFN_OCC64[0]>;
#else
        case 4: (void) arm; (void) ord; return slice_arm<16, FFN_OCC16[FFN_OCC_RULE], 0>(marm);
        case 5: (void) arm; return slice_ord<FMID, FFN_OCC32[FFN_OCC_RULE]>(ord, marm);
        default: (void) arm; return slice_ord<FMAX, FFN_OCC64[FFN_OCC_RULE]>(ord, marm);
#endif
    }
}

// Which budget this process runs. -1 takes the fitted rule; HALO_SLICE_OCC_ARM and ffn_slice_set_occ
// pin an arm so one build can measure both. Without -DHALO_SLICE_OCC_CONTROL=1 the build carries only
// the rule's arm and this selects nothing.
static int g_ffn_occ = [] { const char * v = getenv("HALO_SLICE_OCC_ARM"); return v ? (atoi(v) < 0 ? -1 : (atoi(v) ? 1 : 0)) : -1; }();
void ffn_slice_set_occ(int arm) { g_ffn_occ = arm < 0 ? -1 : (arm ? 1 : 0); }
int ffn_slice_occ() { return g_ffn_occ < 0 ? FFN_OCC_RULE : g_ffn_occ; }

// The predecessor's grid, kept as the control this change is measured against: one grid for the
// whole ladder, the minimum over every width. `HALO_SLICE_GRID=0` selects it, which is what makes
// the block count an in-process axis instead of a second build.
static const bool g_slice_grid_min = [] { const char * v = getenv("HALO_SLICE_GRID"); return v && atoi(v) == 0; }();

void launch_ffn_slice(const FwdParams & p, int layer, hipStream_t st) {
    void * args[] = {(void*) &p, &layer};
    if (p.nrows < 1 || p.nrows > FMAX) { fprintf(stderr, "ffn slice outside its contract: rows %d\n", p.nrows); abort(); }
    const int sm = p.nrows <= 1 ? std::min(std::max(p.single_map, 0), SINGLE_MAP_MAX) : 0;
    const int wi = slice_wi(p.nrows), arm = ffn_slice_occ(), ord = g_mvw_order, marm = g_mvw_arm;
    const void * fn = slice_fn(wi, sm, arm, ord, marm);
    // Every width is its own launch and shares its grid with nothing, so it takes the grid its own
    // registers allow rather than the minimum over the ladder - the same reasoning
    // `launch_sequence_part` above already applies to the sequence parts. Sharing one minimum cost
    // the narrow bodies most of their grid: the dot4 path is 73 VGPR and sixteen waves per SIMD32,
    // and it was running on the 60 workgroups the two-group body's 238 registers allow.
    static int wgrid[SLICE_WIDTHS][SINGLE_MAP_MAX + 1][2][2][MVW_ARM_MAX + 1] = {};
    int & g = wgrid[wi][sm][arm][ord][marm];
    if (!g) {
        g = coop_grid(fn);
        if (g_slice_grid_min && !sm)
            for (int w = 0; w < SLICE_WIDTHS; w++) g = std::min(g, coop_grid(slice_fn(w, 0, arm, ord, marm)));
    }
    // The cooperative grid is what a register change actually costs here, and a phase time alone
    // reads it as a slower schedule. The occupancy API reports workgroups per WGP and a workgroup's
    // eight waves sit two to a SIMD32, so one lost workgroup is a third of this kernel's waves:
    // the two-group body measured -6% at 40 workgroups and -17.5% at 60, same source.
    if (getenv("HALO_FFN_GRID")) {
        static int said[SLICE_WIDTHS][SINGLE_MAP_MAX + 1][MVW_ARM_MAX + 1] = {};
        if (!said[wi][sm][marm]) { said[wi][sm][marm] = 1; fprintf(stderr, "[ffn slice] rows<=%d map %d occ %d order %d arm %d: grid %d workgroups\n", p.nrows, sm, arm, ord, marm, g); }
    }
    hipError_t e = hipLaunchCooperativeKernel(fn, dim3(g), dim3(NT), args, 0, st);
    if (e != hipSuccess) { fprintf(stderr, "FFN slice launch failed: %s\n", hipGetErrorString(e)); abort(); }
}

} // namespace halo
#endif   // HALO_ROWS_PROBE

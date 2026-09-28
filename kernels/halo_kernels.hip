// All device kernels for Bonsai Halo. Shapes are compile-time constants of Bonsai 2 27B.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdlib>
#include "halo_kernels.h"
#include "device.hpp"
#include "attn_leaf.hpp"

namespace halo {

// The key cache's storage coordinate. A process default, because it describes what is already in
// the cache; the measurement tools walk it as a case axis and re-prefill between cases.
static int k_coord = -1;
int attn_coord() {
    if (k_coord < 0) { const char * v = getenv("HALO_ATTN_COORD"); k_coord = (v && *v) ? atoi(v) : 0; }
    return k_coord;
}
void attn_coord_set(int tiled) { k_coord = tiled ? 1 : 0; }

static_assert(TILE_BLOCK_BYTES == 896, "tile geometry");

// ---------------------------------------------------------------------------------------------
// Ternary matvec kernel: one tile per workgroup, WAVES waves split K.

template <int NB, int WAVES>
__global__ void __launch_bounds__(WAVES * 32) k_matvec(MvArgs a) {
    constexpr int PER_WAVE = NB / WAVES;
    static_assert(PER_WAVE * WAVES == NB, "waves must divide blocks");

    int tile = blockIdx.x, s = 0;
    while (s < a.nseg - 1 && tile >= a.seg[s].ntiles) { tile -= a.seg[s].ntiles; s++; }
    const MvSeg seg = a.seg[s];

    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const uint8_t * __restrict__ run = seg.w + ((size_t) tile * NB + wave * PER_WAVE) * TILE_BLOCK_BYTES;
    float y = mv_rows<PER_WAVE>(run, lane, (const int *) a.xq + wave * PER_WAVE * 32, a.xs + wave * PER_WAVE, a.xsum + wave * PER_WAVE);

    __shared__ float red[WAVES][32];
    if (wave > 0) red[wave][lane] = y;
    __syncthreads();
    if (wave == 0) {
        #pragma unroll
        for (int w = 1; w < WAVES; w++) y += red[w][lane];
        float * out = seg.out + (size_t) tile * 32 + lane;
        if (seg.add) *out += y; else *out = y;
    }
}

// ---------------------------------------------------------------------------------------------
// Prep kernel. Grid = n/1024, block = 256.

__global__ void __launch_bounds__(256) k_prep(PrepArgs a) {
    __shared__ float s[1024];
    __shared__ float red[8];
    prep_chunk(a, blockIdx.x, s, red);
}

// ---------------------------------------------------------------------------------------------
// Small bf16 matvec (alpha/beta gates): grid = rows, block = 256.

__global__ void __launch_bounds__(256) k_matvec_bf16(const unsigned short * __restrict__ w, const float * __restrict__ x, float * __restrict__ out, int K) {
    __shared__ float red[8];
    const unsigned short * row = w + (size_t) blockIdx.x * K;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < K; i += 256) acc = fmaf(bf16_to_f32(row[i]), x[i], acc);
    acc = block_sum_256(acc, red);
    if (threadIdx.x == 0) out[blockIdx.x] = acc;
}

// ---------------------------------------------------------------------------------------------
// GDN: conv1d over the 4-tap window + silu, ring-state update. Grid = CONV_CH/256.

__global__ void __launch_bounds__(256) k_gdn_conv(const float * __restrict__ x, const float * __restrict__ w, float * __restrict__ state, float * __restrict__ out) {
    int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= CONV_CH) return;
    float s0 = state[c], s1 = state[CONV_CH + c], s2 = state[2 * CONV_CH + c], xc = x[c];
    const float4 wk = *(const float4 *) (w + c * 4);
    float v = wk.x * s0 + wk.y * s1 + wk.z * s2 + wk.w * xc;
    out[c] = silu(v);
    state[c] = s1; state[CONV_CH + c] = s2; state[2 * CONV_CH + c] = xc;
}

// GDN recurrence for one head per workgroup. Grid = HV (48), block = 256.
__global__ void __launch_bounds__(256) k_gdn(GdnArgs a) {
    __shared__ float q[128], k[128], v[128], o[128];
    __shared__ float red[8];
    const int h = blockIdx.x, kh = h % HK;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;

    if (tid < 128) {
        q[tid] = a.conv_out[kh * 128 + tid];
        k[tid] = a.conv_out[KDIM + kh * 128 + tid];
        v[tid] = a.conv_out[2 * KDIM + h * 128 + tid];
    }
    __syncthreads();
    // L2 normalise q and k (scale = 1/max(norm, eps))
    {
        float qq = tid < 128 ? q[tid] * q[tid] : 0.0f;
        float kk = tid < 128 ? k[tid] * k[tid] : 0.0f;
        qq = block_sum_256(qq, red);
        kk = block_sum_256(kk, red);
        float qs = 1.0f / fmaxf(sqrtf(qq), a.eps);
        float ks = 1.0f / fmaxf(sqrtf(kk), a.eps);
        __syncthreads();
        if (tid < 128) { q[tid] *= qs; k[tid] *= ks; }
        __syncthreads();
    }
    const float xg = a.alpha[h] + a.dt_bias[h];
    const float sp = xg > 20.0f ? xg : log1pf(__expf(xg));
    const float g = __expf(a.ssm_a[h] * sp);
    const float beta = sigmoid(a.beta[h]);

    // thread owns columns i = lane + 32 s (s < 4) of rows j = wave + 8 r (r < 16)
    float * __restrict__ M = a.state + (size_t) h * 128 * 128;
    float m[16][4];
    float kk[4], qq[4];
    #pragma unroll
    for (int s2 = 0; s2 < 4; s2++) { kk[s2] = k[lane + 32 * s2]; qq[s2] = q[lane + 32 * s2]; }
    #pragma unroll
    for (int r = 0; r < 16; r++) {
        const int j = wave + 8 * r;
        #pragma unroll
        for (int s2 = 0; s2 < 4; s2++) m[r][s2] = M[j * 128 + lane + 32 * s2] * g;
        float part = 0.0f;
        #pragma unroll
        for (int s2 = 0; s2 < 4; s2++) part = fmaf(m[r][s2], kk[s2], part);
        part = warp_sum(part);
        const float delta = (v[j] - part) * beta;
        float op = 0.0f;
        #pragma unroll
        for (int s2 = 0; s2 < 4; s2++) { m[r][s2] = fmaf(delta, kk[s2], m[r][s2]); op = fmaf(m[r][s2], qq[s2], op); }
        op = warp_sum(op);
        if (lane == 0) o[j] = op * 0.08838834764831845f; // 1/sqrt(128)
        #pragma unroll
        for (int s2 = 0; s2 < 4; s2++) M[j * 128 + lane + 32 * s2] = m[r][s2];
    }
    __syncthreads();
    // gated RMSnorm: y = rmsnorm(o) * ssm_norm * silu(z)
    float oo = tid < 128 ? o[tid] * o[tid] : 0.0f;
    oo = block_sum_256(oo, red);
    const float inv = rsqrtf(oo / 128.0f + a.eps);
    if (tid < 128) {
        const int idx = h * 128 + tid;
        a.y[idx] = o[tid] * inv * a.ssm_norm[tid] * silu(a.z[idx]);
    }
}

// ---------------------------------------------------------------------------------------------
// Attention: q/k norm + RoPE + KV append. Grid = NH + NKV (28), block = 256.

__global__ void __launch_bounds__(256) k_attn_pre(AttnPreArgs a) {
    __shared__ float s[HD];
    __shared__ float red[8];
    const int d = threadIdx.x;
    const int pos = *a.pos;
    const bool is_q = blockIdx.x < NH;
    const int hq = blockIdx.x, hk = blockIdx.x - NH;
    float v = is_q ? a.q_full[hq * 2 * HD + d] : a.k[hk * HD + d];
    float ss = block_sum_256(v * v, red);
    const float inv = rsqrtf(ss / (float) HD + a.eps);
    v = v * inv * (is_q ? a.q_norm[d] : a.k_norm[d]);
    s[d] = v;
    __syncthreads();
    float outv = v;
    if (d < NROT) {
        const int i = d < NROT / 2 ? d : d - NROT / 2;
        const float theta = (float) pos * __powf(1e7f, -(float) i / (float) (NROT / 2));
        float c, sn; __sincosf(theta, &sn, &c);
        if (d < NROT / 2) outv = s[d] * c - s[d + NROT / 2] * sn;
        else              outv = s[d - NROT / 2] * sn + s[d] * c;
    }
    if (is_q) {
        a.q_out[hq * HD + d] = outv * 0.0625f; // 1/sqrt(256), where ph_attn_pre also applies it
    } else {
        __half * kc = a.kcache + (size_t) hk * a.context * HD;
        __half * vc = a.vcache + ((size_t) hk * a.context + pos) * HD;
        kc[a.k_tiled ? k_leaf_off(pos, d, HD) : (size_t) pos * HD + d] = __float2half(outv);
        vc[d] = __float2half(a.v[hk * HD + d]);
    }
}

// Flash-decoding partials. Grid = (NKV, MAXCTX/ATTN_CHUNK), block = 256.
__global__ void __launch_bounds__(256) k_attn_partial(AttnArgs a) {
    constexpr int GQA = NH / NKV; // 6
    __shared__ float sc[GQA][ATTN_CHUNK];
    __shared__ float red[8];
    const int hk = blockIdx.x, chunk = blockIdx.y;
    const int pos = *a.pos;
    const int start = chunk * ATTN_CHUNK;
    if (start > pos) return;
    const int count = min(ATTN_CHUNK, pos + 1 - start);
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;

    // One lane owns one key and one wave owns a 32-key tile, so a dot product never leaves the
    // register that accumulates it. The leaf-interleaved cache (halo_kernels.h) makes that load
    // coalesced and the six query heads wave-uniform; LeafTree reproduces the bits lane 0 of
    // warp_sum used to receive, including the {0,3,2,1} quad order row_ror gives it. The
    // 1/sqrt(256) scale now sits on the q_out store, where ph_attn_pre has always applied it.
    // The wave index is wave-uniform; naming it is what keeps the query on the scalar unit. This
    // kernel only reads `a.q`, which k_attn_pre wrote in an earlier launch, so a scalar load of it
    // is coherent here in a way it would not be inside a persistent kernel.
    if (a.k_tiled) {
        const int wu = __builtin_amdgcn_readfirstlane(wave);
        const int lp = wu * KTILE + lane;
        const __half * kb = a.kcache + (size_t) hk * a.context * HD
                          + k_leaf_off(start + wu * KTILE, 0, HD) + lane * KLEAF;
        const float * __restrict__ qrow[GQA];
        #pragma unroll
        for (int g = 0; g < GQA; g++) qrow[g] = a.q + (hk * GQA + g) * HD;
        if (lp < count) {
            float ds[GQA];
            auto leaf = [&](auto J, float * out) {
                constexpr int jj = decltype(J)::value;
                const uint4 raw = *(const uint4 *) (kb + (size_t) jj * (KTILE * KLEAF));
                const __half2 * h2 = (const __half2 *) &raw;
                float kf[KLEAF];
                #pragma unroll
                for (int e = 0; e < KLEAF / 2; e++) { float2 f = __half22float2(h2[e]); kf[2 * e] = f.x; kf[2 * e + 1] = f.y; }
                #pragma unroll
                for (int g = 0; g < GQA; g++) {
                    const float * __restrict__ qp = qrow[g] + jj * KLEAF;
                    float s = 0.0f;
                    #pragma unroll
                    for (int e = 0; e < KLEAF; e++) s = fmaf(qp[e], kf[e], s);
                    out[g] = s;
                }
            };
            LeafTree<HD / KLEAF, GQA>::run(leaf, ds);
            #pragma unroll
            for (int g = 0; g < GQA; g++) sc[g][lp] = ds[g];
        }
    } else {
        // The row-major control, unchanged except that its query arrives pre-scaled.
        float qr[GQA][8];
        #pragma unroll
        for (int g = 0; g < GQA; g++) {
            const float * qp = a.q + (hk * GQA + g) * HD + lane * 8;
            #pragma unroll
            for (int e = 0; e < 8; e++) qr[g][e] = qp[e];
        }
        const __half * kc = a.kcache + ((size_t) hk * a.context + start) * HD;
        for (int p = wave; p < count; p += 8) {
            const uint4 raw = *(const uint4 *) (kc + (size_t) p * HD + lane * 8);
            const __half2 * h2 = (const __half2 *) &raw;
            float kf[8];
            #pragma unroll
            for (int e = 0; e < 4; e++) { float2 f = __half22float2(h2[e]); kf[2 * e] = f.x; kf[2 * e + 1] = f.y; }
            #pragma unroll
            for (int g = 0; g < GQA; g++) {
                float d = 0.0f;
                #pragma unroll
                for (int e = 0; e < 8; e++) d = fmaf(qr[g][e], kf[e], d);
                d = warp_sum(d);
                if (lane == 0) sc[g][p] = d;
            }
        }
    }
    __syncthreads();
    // softmax stats per head
    float mx[GQA], lsum[GQA];
    #pragma unroll
    for (int g = 0; g < GQA; g++) {
        float m = -INFINITY;
        for (int p = tid; p < count; p += 256) m = fmaxf(m, sc[g][p]);
        m = block_max_256(m, red);
        float l = 0.0f;
        for (int p = tid; p < count; p += 256) { float e = __expf(sc[g][p] - m); sc[g][p] = e; l += e; }
        l = block_sum_256(l, red);
        mx[g] = m; lsum[g] = l;
    }
    __syncthreads();
    // P @ V: thread = dim
    const __half * vc = a.vcache + ((size_t) hk * a.context + start) * HD;
    float acc[GQA];
    #pragma unroll
    for (int g = 0; g < GQA; g++) acc[g] = 0.0f;
    for (int p = 0; p < count; p++) {
        const float vv = __half2float(vc[(size_t) p * HD + tid]);
        #pragma unroll
        for (int g = 0; g < GQA; g++) acc[g] = fmaf(sc[g][p], vv, acc[g]);
    }
    #pragma unroll
    for (int g = 0; g < GQA; g++) {
        const int hq = hk * GQA + g;
        AttnPartial * part = a.partials + (size_t) hq * ATTN_MAX_CHUNKS + chunk;
        part->acc[tid] = acc[g];
        if (tid == 0) { part->m = mx[g]; part->l = lsum[g]; }
    }
}

// Combine partials, apply output gate. Grid = NH, block = 256.
__global__ void __launch_bounds__(256) k_attn_combine(AttnArgs a) {
    const int hq = blockIdx.x, d = threadIdx.x;
    const int pos = *a.pos;
    const int nchunks = pos / ATTN_CHUNK + 1;
    const AttnPartial * part = a.partials + (size_t) hq * ATTN_MAX_CHUNKS;
    float M = -INFINITY;
    for (int c = 0; c < nchunks; c++) M = fmaxf(M, part[c].m);
    float num = 0.0f, den = 0.0f;
    for (int c = 0; c < nchunks; c++) {
        const float w = __expf(part[c].m - M);
        num = fmaf(w, part[c].acc[d], num);
        den = fmaf(w, part[c].l, den);
    }
    const float gate = a.q_full[hq * 2 * HD + HD + d];
    a.y[hq * HD + d] = num / den * sigmoid(gate);
}

// ---------------------------------------------------------------------------------------------
// Embedding: decode one HALO row of token_embd into fp32. Block = 64 (one thread per block of 128).

__global__ void __launch_bounds__(64) k_embed_row(const uint8_t * __restrict__ w, const int * __restrict__ tok_ring, const int * __restrict__ pos, float * __restrict__ out) {
    const int b = threadIdx.x;
    if (b >= NB_D) return;
    const int r = tok_ring[pos[0] & (TOK_RING - 1)];
    const int tile = r >> 5, lane = r & 31;
    const uint8_t * run = w + ((size_t) tile * NB_D + b) * TILE_BLOCK_BYTES;
    uint8_t qs[24];
    #pragma unroll
    for (int i = 0; i < 16; i++) qs[i] = run[tile_off_qs_a(lane) + i];
    #pragma unroll
    for (int i = 0; i < 8; i++) qs[16 + i] = run[tile_off_qs_b(lane) + i];
    const unsigned tail = *(const unsigned *) (run + tile_off_tail(lane));
    const unsigned short qh = (unsigned short) (tail & 0xffffu);
    const float scale = __half2float(__ushort_as_half((unsigned short) (tail >> 16)));
    float * o = out + b * 128;
    #pragma unroll
    for (int d = 0; d < 6; d++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const int w2 = j & 1, half = j >> 1, p = 2 * d + w2;
            unsigned v = qs[4 * d + j];
            int t[5];
            #pragma unroll
            for (int n = 0; n < 5; n++) { unsigned m = v * 3u; t[n] = (int) (m >> 8); v = m & 0xffu; }
            o[8 * p + 2 * half + 0] = (float) (t[0] - 1) * scale;
            o[8 * p + 2 * half + 1] = (float) (t[1] - 1) * scale;
            o[8 * p + 4 + 2 * half + 0] = (float) (t[2] - 1) * scale;
            o[8 * p + 4 + 2 * half + 1] = (float) (t[3] - 1) * scale;
            o[96 + 4 * d + j] = (float) (t[4] - 1) * scale;
        }
    }
    #pragma unroll
    for (int h = 0; h < 2; h++) {
        unsigned v = h == 0 ? (qh & 0xffu) : (qh >> 8);
        int t[4];
        #pragma unroll
        for (int n = 0; n < 4; n++) { unsigned m = v * 3u; t[n] = (int) (m >> 8); v = m & 0xffu; }
        o[120 + 2 * h] = (float) (t[0] - 1) * scale; o[121 + 2 * h] = (float) (t[1] - 1) * scale;
        o[124 + 2 * h] = (float) (t[2] - 1) * scale; o[125 + 2 * h] = (float) (t[3] - 1) * scale;
    }
}

// ---------------------------------------------------------------------------------------------
// Argmax over the logits. Block = 1024, grid = 1.

__global__ void __launch_bounds__(1024) k_argmax(const float * __restrict__ logits, int n, int * __restrict__ out, int * __restrict__ out2) {
    __shared__ float sm[32]; __shared__ int si[32];
    float best = -INFINITY; int bi = 0;
    for (int i = threadIdx.x; i < n; i += 1024) { float v = logits[i]; if (v > best) { best = v; bi = i; } }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float ob = __shfl_xor(best, o, 32); int oi = __shfl_xor(bi, o, 32);
        if (ob > best || (ob == best && oi < bi)) { best = ob; bi = oi; }
    }
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    if (l == 0) { sm[w] = best; si[w] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 1; i < 32; i++) if (sm[i] > best || (sm[i] == best && si[i] < bi)) { best = sm[i]; bi = si[i]; }
        out[0] = bi; if (out2) out2[0] = bi;
    }
}

__global__ void k_set_int(int * p, int v) { p[0] = v; }
__global__ void k_inc_int(int * p) { p[0] += 1; }

// ---------------------------------------------------------------------------------------------
// Host-side launchers

#define LAUNCH_MV(NB, WAVES) k_matvec<NB, WAVES><<<ntiles, WAVES * 32, 0, st>>>(a)

void launch_matvec(const MvArgs & a_in, int K, hipStream_t st) {
    MvArgs a = a_in;
    int ntiles = 0;
    for (int i = 0; i < a.nseg; i++) ntiles += a.seg[i].ntiles;
    a.total_tiles = ntiles;
    switch (K) {
        case 5120:  LAUNCH_MV(40, 4); break;   // 10 blocks per wave
        case 6144:  LAUNCH_MV(48, 8); break;   // 6 per wave
        case 17408: LAUNCH_MV(136, 8); break;  // 17 per wave
        default: abort();
    }
}

void launch_prep(const PrepArgs & a, hipStream_t st) { k_prep<<<a.n / 1024, 256, 0, st>>>(a); }
void launch_matvec_bf16(const unsigned short * w, const float * x, float * out, int rows, int K, hipStream_t st) { k_matvec_bf16<<<rows, 256, 0, st>>>(w, x, out, K); }
void launch_gdn_conv(const float * x, const float * w, float * state, float * out, hipStream_t st) { k_gdn_conv<<<CONV_CH / 256, 256, 0, st>>>(x, w, state, out); }
void launch_gdn(const GdnArgs & a, hipStream_t st) {
    // The v1 per-op path reads the state as plain fp32 rows and has no packed decoder.
    if (gdn_state_format() != GDN_STATE_F32 && gdn_state_format() != GDN_STATE_F32_PK) {
        fprintf(stderr, "the --v1 per-op path needs HALO_GDN_STATE=f32 (have %s)\n", gdn_state_format_name(gdn_state_format()));
        abort();
    }
    k_gdn<<<HV, 256, 0, st>>>(a);
}
void launch_attn_pre(const AttnPreArgs & a, hipStream_t st) { k_attn_pre<<<NH + NKV, 256, 0, st>>>(a); }
void launch_attn(const AttnArgs & a, hipStream_t st) {
    k_attn_partial<<<dim3(NKV, ATTN_MAX_CHUNKS), 256, 0, st>>>(a);
    k_attn_combine<<<NH, 256, 0, st>>>(a);
}
void launch_embed_row(const uint8_t * w, const int * tok_ring, const int * pos, float * out, hipStream_t st) { k_embed_row<<<1, 64, 0, st>>>(w, tok_ring, pos, out); }
void launch_argmax(const float * logits, int n, int * out, int * out2, hipStream_t st) { k_argmax<<<1, 1024, 0, st>>>(logits, n, out, out2); }
void launch_set_int(int * p, int v, hipStream_t st) { k_set_int<<<1, 1, 0, st>>>(p, v); }
void launch_inc_int(int * p, hipStream_t st) { k_inc_int<<<1, 1, 0, st>>>(p); }

} // namespace halo

// Per-phase register census for the persistent kernel. Not linked into the engine.
//
// `k_forward_rows` inlines every phase of a layer into one function, so the compiler allocates it
// the maximum any program point in it needs and the whole kernel's occupancy is set by the single
// worst phase. The resource report names the kernel, never the phase, and `docs/mv-dot4-body.md`
// left "forty registers of the eight-row kernel come from some other phase" as an open question
// precisely because nothing could attribute them.
//
// This does. Each kernel below carries exactly one phase of `phases.hpp`, at the persistent
// kernel's own `__launch_bounds__(NT)` and with the same LDS allocation, so its VGPR count is the
// pressure that phase contributes on its own. Phases living in `halo_rows.hip`'s anonymous
// namespace (`ph_gdn_pre`, `ph_gdn`) are not reachable from here and do not need to be: the
// `PART = 2` instantiation of `k_forward_rows` is exactly those two plus attention and the output
// prep, and `tools/kernel_resources.py` reads it directly.
//
//   tools/kernel_resources.py kernels/pk_pressure.hip
//
// Forty seconds, no GPU and no lock.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "halo_kernels.h"
#include "device.hpp"
#include "phases.hpp"

using namespace halo;

namespace {


__device__ __forceinline__ Ctx make_ctx(FwdParams & P, float * lds, float * red, int * s_unit) {
    return Ctx { P.bar, P.work, P.prof, 0, 0, true, lds, red, s_unit };
}

}  // namespace

#define PROBE_PROLOGUE                                                        \
    __shared__ __attribute__((aligned(16))) float lds[LDS_FLOATS];            \
    __shared__ float red[8];                                                  \
    __shared__ int s_unit;                                                    \
    Ctx c = make_ctx(P, lds, red, &s_unit);                                   \
    const int nrows = P.nrows;

// The four prep shapes a layer runs: the normed input, the GDN output norm, the attention combine
// and the SiLU-multiplied FFN interior. DIRECT selects the wide operand destination.
template <int FLAGS, bool DIRECT>
__global__ void __launch_bounds__(NT) k_probe_prep(FwdParams P) {
    PROBE_PROLOGUE
    PrepR p{};
    p.x = P.x; p.x2 = P.zbuf; p.kvec = P.signs5120; p.norm_w = P.output_norm; p.eps = NORM_EPS;
    p.n = D; p.flags = FLAGS; p.xq = P.xq; p.xs = P.xs; p.xsum = P.xsum;
    p.partials = P.partials; p.rows = P.rows; p.ninv = P.ninv; p.ab_out = P.ab; p.ab_rows = 2 * HV;
    bind_seq_dest<DIRECT>(p, P);
    ph_prep<DIRECT>(c, p, nrows);
}

// The three matvec geometries of a deployed layer, plus the vocabulary head, at each row count.
template <int NB, int KS, int TT>
__global__ void __launch_bounds__(NT) k_probe_matvec(FwdParams P) {
    PROBE_PROLOGUE
    MvR m{}; m.nseg = 1; m.seg[0] = { P.output, P.x, D / 32, 1 }; m.total_tiles = D / 32;
    ph_matvec<NB, KS, TT, false, 0>(c, m, P.xq, P.xs, P.xsum, nrows);
}

// Three segments is the qkv/z and q/k/v shape; the segment walk is per unit, so it is live across
// the body.
template <int NB, int KS, int TT>
__global__ void __launch_bounds__(NT) k_probe_matvec3(FwdParams P) {
    PROBE_PROLOGUE
    MvR m{}; m.nseg = 3;
    m.seg[0] = { P.output, P.qfull, Q_OUT / 32, 0 };
    m.seg[1] = { P.output, P.kbuf, KV_OUT / 32, 0 };
    m.seg[2] = { P.output, P.vbuf, KV_OUT / 32, 0 };
    m.total_tiles = (Q_OUT + 2 * KV_OUT) / 32;
    ph_matvec<NB, KS, TT, false, 0>(c, m, P.xq, P.xs, P.xsum, nrows);
}

template <int TT>
__global__ void __launch_bounds__(NT) k_probe_attn(FwdParams P) {
    PROBE_PROLOGUE
    ph_attn<TT, NH, NKV, HD>(c, P, 0);
}

__global__ void __launch_bounds__(NT) k_probe_attn_pre(FwdParams P) {
    PROBE_PROLOGUE
    ph_attn_pre(c, P, P.output_norm, P.output_norm, 0);
}

__global__ void __launch_bounds__(NT) k_probe_argmax(FwdParams P) {
    PROBE_PROLOGUE
    ph_argmax_slices(c, P);
    if ((int) blockIdx.x < nrows)
        argmax_final(P.amax_val + blockIdx.x * AMAX_SLICES, P.amax_idx + blockIdx.x * AMAX_SLICES,
                     P.argmax_out + blockIdx.x, c.lds);
}

__global__ void __launch_bounds__(NT) k_probe_embed(FwdParams P) {
    PROBE_PROLOGUE
    if ((int) blockIdx.x < nrows) embed_row(P.tok_embd, P.rows[blockIdx.x].token, P.tmp + (size_t) blockIdx.x * D);
    zero_span(c, P.big, nrows * QKV_OUT);
}

// Prep: input norm, GDN output norm, attention combine, FFN interior.
template __global__ void k_probe_prep<PREP_NORM | PREP_HADAMARD | PREP_QUANT, false>(FwdParams);
template __global__ void k_probe_prep<PREP_NORM | PREP_HADAMARD | PREP_QUANT, true>(FwdParams);
template __global__ void k_probe_prep<PREP_GDN_NORM | PREP_HADAMARD | PREP_QUANT, false>(FwdParams);
template __global__ void k_probe_prep<PREP_ATTN_COMBINE | PREP_HADAMARD | PREP_QUANT, false>(FwdParams);
template __global__ void k_probe_prep<PREP_SILU_MUL | PREP_HADAMARD | PREP_QUANT, false>(FwdParams);

// Matvecs: qkv/z + gate/up + head (NB_D, KS 1), ssm_out/o (NB_V, KS 2), down (NB_FF, KS 2).
template __global__ void k_probe_matvec<NB_D, 1, 1>(FwdParams);
template __global__ void k_probe_matvec<NB_D, 1, 4>(FwdParams);
template __global__ void k_probe_matvec<NB_D, 1, 8>(FwdParams);
template __global__ void k_probe_matvec3<NB_D, 1, 8>(FwdParams);
template __global__ void k_probe_matvec<NB_V, 2, 1>(FwdParams);
template __global__ void k_probe_matvec<NB_V, 2, 4>(FwdParams);
template __global__ void k_probe_matvec<NB_V, 2, 8>(FwdParams);
template __global__ void k_probe_matvec<NB_FF, 2, 1>(FwdParams);
template __global__ void k_probe_matvec<NB_FF, 2, 4>(FwdParams);
template __global__ void k_probe_matvec<NB_FF, 2, 8>(FwdParams);

template __global__ void k_probe_attn<1>(FwdParams);
template __global__ void k_probe_attn<8>(FwdParams);

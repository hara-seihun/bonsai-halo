#include "prep_batch.h"
#include "device.hpp"
#include "phases.hpp"
#include <stdexcept>
#include <string>

namespace halo {
namespace {

// The alpha/beta projection is 2*HV dot products of length D against every token row. Each
// workgroup keeps PREP_AB_TOK token rows and the norm weight in registers and walks PREP_AB_ROWS
// projection rows against them, so the token rows are read once per projection group instead of
// once per projection row.
constexpr int PREP_AB_ROWS = 16;
constexpr int PREP_AB_TOK  = 4;
constexpr int PREP_KI      = D / (NT * 4);   // i-steps one thread takes across D
static_assert(PREP_KI * NT * 4 == D, "the prep loop must cover D exactly");

// block_sum_256 for G values at once. Same warp reduction and same ordered sum over the eight
// wave partials, with one pair of barriers instead of one pair per value.
template <int G>
__device__ __forceinline__ void block_sum_256_multi(float v[G], float * red, float out[G]) {
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    #pragma unroll
    for (int g = 0; g < G; g++) v[g] = warp_sum(v[g]);
    __syncthreads();
    if (l == 0) {
        #pragma unroll
        for (int g = 0; g < G; g++) red[g * 8 + w] = v[g];
    }
    __syncthreads();
    #pragma unroll
    for (int g = 0; g < G; g++) {
        float r = red[g * 8];
        #pragma unroll
        for (int i = 1; i < 8; i++) r += red[g * 8 + i];
        out[g] = r;
    }
}

// Blocks [0, rows) own one token row's norm scalar; the rest own an (alpha/beta group, token
// group) pair. Both read only the residual, so they share one launch.
__global__ __launch_bounds__(NT) void prep_meta(SeqPrepArgs a, int ab_groups) {
    __shared__ float red[PREP_AB_TOK * 8];
    const int tid = threadIdx.x;
    if ((int) blockIdx.x < a.rows) {
        const float * x = a.x + (size_t) blockIdx.x * D;
        float ss = 0.0f;
        for (int i = tid * 4; i < D; i += NT * 4) {
            const float4 t = *(const float4 *) (x + i);
            ss = fmaf(t.x, t.x, ss); ss = fmaf(t.y, t.y, ss); ss = fmaf(t.z, t.z, ss); ss = fmaf(t.w, t.w, ss);
        }
        ss = block_sum_256(ss, red);
        if (tid == 0) a.ninv[blockIdx.x] = rsqrtf(ss / (float) D + NORM_EPS);
        return;
    }
    const int u = (int) blockIdx.x - a.rows;
    const int row0 = (u / ab_groups) * PREP_AB_TOK, r0 = (u % ab_groups) * PREP_AB_ROWS;
    float4 xv[PREP_AB_TOK][PREP_KI], nv[PREP_KI];
    #pragma unroll
    for (int k = 0; k < PREP_KI; k++) {
        const int i = tid * 4 + k * NT * 4;
        nv[k] = *(const float4 *) (a.norm_w + i);
        #pragma unroll
        for (int rr = 0; rr < PREP_AB_TOK; rr++)
            xv[rr][k] = row0 + rr < a.rows ? *(const float4 *) (a.x + (size_t) (row0 + rr) * D + i)
                                           : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    for (int rl = 0; rl < PREP_AB_ROWS; rl++) {
        const int r = r0 + rl;
        if (r >= 2 * HV) break;
        const unsigned short * w = a.ab_w + (size_t) r * D;
        float acc[PREP_AB_TOK];
        #pragma unroll
        for (int rr = 0; rr < PREP_AB_TOK; rr++) acc[rr] = 0.0f;
        #pragma unroll
        for (int k = 0; k < PREP_KI; k++) {
            const uint2 w4 = *(const uint2 *) (w + tid * 4 + k * NT * 4);
            const float w0 = bf16_to_f32(w4.x & 0xffff) * nv[k].x, w1 = bf16_to_f32(w4.x >> 16) * nv[k].y,
                        w2 = bf16_to_f32(w4.y & 0xffff) * nv[k].z, w3 = bf16_to_f32(w4.y >> 16) * nv[k].w;
            #pragma unroll
            for (int rr = 0; rr < PREP_AB_TOK; rr++)
                acc[rr] = fmaf(w0, xv[rr][k].x, fmaf(w1, xv[rr][k].y, fmaf(w2, xv[rr][k].z, fmaf(w3, xv[rr][k].w, acc[rr]))));
        }
        float out[PREP_AB_TOK];
        block_sum_256_multi<PREP_AB_TOK>(acc, red, out);
        if (tid == 0) {
            #pragma unroll
            for (int rr = 0; rr < PREP_AB_TOK; rr++)
                if (row0 + rr < a.rows) a.ab_out[(size_t) (row0 + rr) * (2 * HV) + r] = out[rr];
        }
    }
}

// One workgroup per (token row, 1024-element chunk): the deployed prep unit, reading the norm
// scalar the meta launch already produced instead of recomputing it five times per row.
template <int QL, bool NIB>
__global__ __launch_bounds__(NT) void prep_chunk(SeqPrepArgs a) {
    __shared__ __attribute__((aligned(16))) float lds[1024];
    PrepR p{};
    p.x = a.x; p.kvec = a.norm_s; p.n = D; p.eps = NORM_EPS;
    p.flags = PREP_NORM | PREP_HADAMARD | PREP_QUANT;
    p.seq_q = a.seq_q; p.seq_scales = a.seq_scales; p.seq_npad = a.npad; p.seq_offset = 0;
    prep_chunk_r<true, QL, NIB>(p, blockIdx.x, blockIdx.y, a.ninv[blockIdx.x], lds, 0);
}

} // namespace

// The process default for the sequence projections' activation operand.
//
// Four bits on BOTH projections. The two stages are quantised in different kernels and were
// measured a day apart - the input stage is 1.84x faster in four bits (docs/sequence-input-a4.md)
// and the output stage's phase is -40% (docs/sequence-output-a4.md) - but their quality only makes
// sense read together, and read together they do not add:
//
//   horizon, 1024 paired teacher-forced predictions, 32 streams x 256 single-token steps,
//   stream-clustered, mode 19, against the same route's eight-bit operand
//     input four bits alone   +0.00770 +- 0.00798 nats
//     output four bits alone  +0.00458 +- 0.00901
//     both                    +0.00397 +- 0.00940      <- less than either stage alone
//     the A4 FFN this engine already serves (mode 18 -> 19), same panel:  +0.01866 +- 0.00687
//
// So adding the output stage to the four-bit input default is -0.00373 +- 0.00934 nats: the point
// estimate moves the wide route *towards* the exact engine, 15 of 32 streams get worse, and
// teacher-forced top-1 goes 0.3975 -> 0.4033. The whole wide A4 route is +0.02263 from the
// eight-bit-FFN reference where today's default is +0.02636. An activation quantiser is memoryless
// - the horizon quarters change sign (+0.0224, -0.0118, +0.0250, -0.0198) rather than growing -
// which is why this reads differently from the packed recurrent state on the same instrument.
//
// `HALO_SEQ_QUANT=a4` restores the previous default exactly and `a8` the exact operand on both
// stages. docs/seq-out-a4-default.md owns the acceptance.
static constexpr int SEQ_QUANT_DEFAULT = SEQ_QUANT_A4_BOTH;

// Whether this process was told its coordinate or took the default. A told coordinate is a
// contract and a route that cannot carry it throws; the default falls back instead.
static bool seq_quant_told = false;

static int seq_quant = [] {
    const char * v = getenv("HALO_SEQ_QUANT");
    if (!v) return (int) SEQ_QUANT_DEFAULT;
    seq_quant_told = true;
    const std::string s(v);
    if (s == "a8" || s == "8" || s == "0") return (int) SEQ_QUANT_A8;
    if (s == "a4e" || s == "1") return (int) SEQ_QUANT_A4E;
    if (s == "a4" || s == "4" || s == "2") return (int) SEQ_QUANT_A4;
    if (s == "a4e-out" || s == "3") return (int) SEQ_QUANT_A4E_OUT;
    if (s == "a4-out" || s == "4-out") return (int) SEQ_QUANT_A4_OUT;
    if (s == "a4e-both" || s == "5") return (int) SEQ_QUANT_A4E_BOTH;
    if (s == "a4-both" || s == "6") return (int) SEQ_QUANT_A4_BOTH;
    throw std::runtime_error("HALO_SEQ_QUANT must be a8, a4e, a4, a4e-out, a4-out, a4e-both or a4-both");
}();
int sequence_quant() { return seq_quant; }
const char * sequence_quant_name(int q) {
    switch (q) { case SEQ_QUANT_A8: return "a8"; case SEQ_QUANT_A4E: return "a4e"; case SEQ_QUANT_A4: return "a4";
                 case SEQ_QUANT_A4E_OUT: return "a4e-out"; case SEQ_QUANT_A4_OUT: return "a4-out";
                 case SEQ_QUANT_A4E_BOTH: return "a4e-both"; case SEQ_QUANT_A4_BOTH: return "a4-both"; default: return "?"; }
}
void sequence_set_quant(int q) {
    if (q < 0 || q > SEQ_QUANT_MAX) throw std::runtime_error("invalid sequence activation precision");
    seq_quant = q;
    seq_quant_told = true;
}
bool sequence_quant_explicit() { return seq_quant_told; }
// The fallback path sets the coordinate without claiming the process asked for it, so a later
// measurement arm still fails loudly rather than measuring a stage it did not select.
void sequence_quant_fall_back_to(int q) { seq_quant = q; }

bool sequence_prep_wide_enabled() {
    static const int on = [] {
        const char * v = getenv("HALO_WIDE_PREP");
        if (!v) return 1;
        const std::string value(v);
        if (value != "0" && value != "1") throw std::runtime_error("HALO_WIDE_PREP must be 0 or 1");
        return value == "1" ? 1 : 0;
    }();
    return on != 0;
}

void launch_sequence_prep(const SeqPrepArgs & a, hipStream_t st) {
    if (a.rows < 1 || a.rows > PASSMAX || a.npad < a.rows || !a.x || !a.norm_s || !a.seq_q || !a.seq_scales || !a.ninv)
        throw std::runtime_error("invalid wide sequence prep shape");
    if (a.ab_w && (!a.norm_w || !a.ab_out)) throw std::runtime_error("alpha/beta prep without its operands");
    const int ab_groups = (2 * HV + PREP_AB_ROWS - 1) / PREP_AB_ROWS;
    const int blocks = a.rows + (a.ab_w ? ab_groups * ((a.rows + PREP_AB_TOK - 1) / PREP_AB_TOK) : 0);
    prep_meta<<<blocks, NT, 0, st>>>(a, ab_groups);
    const dim3 chunks(a.rows, D / 1024);
    if (!sequence_quant_in_seven(seq_quant))     prep_chunk<127, false><<<chunks, NT, 0, st>>>(a);
    else if (sequence_quant_in_nibble(seq_quant))prep_chunk<7, true><<<chunks, NT, 0, st>>>(a);
    else                                         prep_chunk<7, false><<<chunks, NT, 0, st>>>(a);
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) throw std::runtime_error(std::string("prep_batch: ") + hipGetErrorString(e));
}

}

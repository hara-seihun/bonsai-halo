// Batched FFN kernels. See ffn_batch.h for the contract, the modes and the numerical labelling.
//
// Four kernels per layer, all on the caller's stream:
//
//   1. post-attention RMS norm (engine epsilon), the layer's folded sign/norm vector, the 1024-point
//      Hadamard, and the block quantiser, written straight into WMMA B fragments — there is no
//      intermediate int8 buffer for a later kernel to re-read,
//   2. gate and up projection, with SiLU, the product and the hidden sign folded into the store,
//   3. the hidden Hadamard and quantiser into B fragments,
//   4. down projection accumulated onto the incoming residual.
//
// A wave owns 16 weight rows of two matrices and TT token tiles; the A operand is built once per
// (block, slice) and used TT times against both accumulator sets. The gate/up pair is the natural
// two matrices; the down matrix has no natural partner, so it is split at D/2 and the halves ride
// together, which only needs them to consume the same activation vector.
//
// The control modes 1, 2 and 3 run k_proj_int and k_proj_scaled, unchanged since they were
// measured. The optimized modes 6, 7 and 8 run k_proj_opt, which adds two levers over the same
// arithmetic: who owns a weight tile, and how many loads a lane's block costs. Nothing in it
// changes a product, a summation order or an activation.
//
// Provenance: Kelana research/ffn/batched at commit 1adc35e for the control kernels
// (arithmetic/candidates/arith_maps.hip and compact-scaled/candidates/cs_maps.hip) and at commit
// 15e0524 for the optimized ones (tile-ownership/candidates/to_maps.hip ownership and load
// schedule, dense-consumer/candidates/dense_maps.hip four-bit operand maps).
#include "ffn_batch.h"
#include "ffn_operands.hpp"
#include "ffn_a4_operands.hpp"
#include "ffn_b_stage.hpp"
#include "ffn_dense_stream.hpp"
#include "device.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace halo {

using namespace halo::ffnb;

namespace {

constexpr int NT = 256;    // producer threads per 1024-element chunk (8 waves, as block_sum_256 wants)
constexpr int NW = 4;      // waves per projection workgroup under per-wave tile ownership
constexpr int MAXTT = 8;   // widest token tile any mode uses; sets the fragment slack
constexpr int FRAG_SLACK = MAXTT * 16 * 32;   // one tile group of FP16 fragments, 4 KiB
constexpr int ROW_SLACK = MAXTT * 16;         // token columns a rounded-up tile group can reach
constexpr int WAVES_WANTED = 320;             // four waves per SIMD32 on this device's 80 units
constexpr int WAVES_WANTED_WIDE = 640;        // eight, which is what the thin down stage measures
constexpr int SMALL_ROWS = 32;                // at or below this a batch is one token tile group
constexpr int CURSOR_SLACK_BLOCKS = 2;        // blocks a weight image carries past its last tile

// ---------------------------------------------------------------- the stored image's run order
//
// A weight image is a flat array of RUNS, and a run is one (16-row tile, 128-block) pair: the
// codes one wave reads for that block and the sixteen FP16 scales beside them. Which run a pair
// lands on is `tile * tstep + blk * bstep`. Nothing else about the image moves with it - same
// bytes, same values, same order of use, the same loads in the same block loop - so every order
// produces the same bits and only the addresses the resident waves ask for at one instant change.
//
//   FFN_RUN_TILE (0)    tstep = nb,      bstep = 1        the deployed image. A wave's own blocks
//                                                         are contiguous and the hundreds of
//                                                         resident waves sit one stream stride
//                                                         (nb * block bytes) apart.
//   FFN_RUN_PAD  (1)    tstep = nb + 1,  bstep = 1        the same locality, with the stream
//                                                         stride's power-of-two factor broken by
//                                                         one run and 1/nb more bytes allocated.
//   FFN_RUN_BLOCK(2)    tstep = 1,       bstep = ntiles   every resident wave's block b inside one
//                                                         contiguous run.
//   FFN_RUN_AUTO (3)    padded where the stream stride is a multiple of 4096, tile-major where it
//                                                         is not. The default, and what the two
//                                                         measured images want: see below.
//
// The sequence projections took this axis first (kernels/sequence_batch.hip,
// docs/weight-stream-order.md): block-major was worth 19% of their phase at 32 rows with an
// unchanged instruction census. bench/wstream prices the mechanism with no model in it and reads
// 116-143 GB/s for streams whose stride carries a large power of two against 194-218 for the same
// bytes read where the other waves are reading. The FFN's deployed strides are 16,640 and 56,576
// bytes on the dense five-trit image, 20,480 and 69,632 on the pair-code and two-bit ones.
// Measured at 32 prefill rows, mode 19, both images forced, three orders in one process, with the
// five phases the change cannot reach as the control (`ffn` ms, normalised in brackets):
//
//   dense five-trit, stride 16,640 = 2^8 * 65    tile 31.97   pad 32.56 [1.00]   block 33.96 [1.06]
//   pair codes,      stride 20,480 = 2^12 * 5    tile 50.66   pad 43.57 [0.83]   block 43.30 [0.83]
//
// **A stride that is a multiple of 4096 costs this kernel about a sixth of its phase, and one run
// of padding recovers it.** Breaking the alignment while keeping a wave's own blocks contiguous is
// worth as much as moving every wave into one run, which says the term is the aliasing and not the
// distance between the waves; and where there is no aliasing to fix, block-major's loss of
// per-wave locality is what is left (+6%). The deployed 128-row pass, which reads the pair image
// on both stages, is -4.6% padded and -3.7% block-major.
enum FfnRunOrder { FFN_RUN_TILE = 0, FFN_RUN_PAD = 1, FFN_RUN_BLOCK = 2, FFN_RUN_AUTO = 3 };

constexpr int RUN_ALIAS_STRIDE = 4096;   // the alignment that costs, fitted on the two images above

struct RunSteps { int tstep, bstep; };

// `bb` is the image's block size, so the rule can see the stream stride a wave's neighbours stand
// at rather than the run count alone.
inline RunSteps run_steps(int order, int ntiles, int nb, int bb) {
    if (order == FFN_RUN_BLOCK) return { 1, ntiles };
    const bool pad = order == FFN_RUN_PAD
                  || (order == FFN_RUN_AUTO && (size_t) nb * bb % RUN_ALIAS_STRIDE == 0);
    return { pad ? nb + 1 : nb, 1 };
}

// Preparation is a capacity decision made against a device whose free memory this module does not
// own, so it reports and throws rather than aborting: a caller that asked for more images than fit
// gets to fall back to a smaller mode set with the state it already had still working.
void check(hipError_t e, const char * what) {
    if (e != hipSuccess)
        throw std::runtime_error(std::string("ffn_batch: ") + what + ": " + hipGetErrorString(e));
}

std::string gb(size_t bytes) {
    char t[32];
    snprintf(t, sizeof t, "%.3f GB", (double) bytes / 1e9);
    return t;
}

std::string mode_list(unsigned mask) {
    std::string s;
    for (int m = 0; m < 32; ++m)
        if (mask & (1u << m)) { if (!s.empty()) s += ","; s += std::to_string(m); }
    return s.empty() ? std::string("none") : s;
}

std::string mask_hex(unsigned mask) {
    char t[16];
    snprintf(t, sizeof t, "0x%x", mask);
    return t;
}

std::string image_list(unsigned images) {
    std::string s;
    for (int i = 0; i < FFN_IMAGE_COUNT; ++i)
        if (images & (1u << i)) { if (!s.empty()) s += ", "; s += ffn_batch_image_name(i); }
    return s.empty() ? std::string("none") : s;
}

// What the runtime will admit right now. It is the runtime's view of one shared pool, so another
// GPU client can take memory between this answer and the allocation; the failure path below has to
// stay correct whether or not this check passed.
bool device_memory(size_t & freeb, size_t & totalb) {
    freeb = totalb = 0;
    return hipMemGetInfo(&freeb, &totalb) == hipSuccess;
}

std::string device_memory_phrase() {
    size_t freeb = 0, totalb = 0;
    if (!device_memory(freeb, totalb)) return "hipMemGetInfo did not answer";
    return "hipMemGetInfo reports " + gb(freeb) + " free of " + gb(totalb);
}

// Every device allocation one create() or prepare() call makes. A failure part way through gives
// back exactly what that call took, so the state keeps serving the modes it already had and its
// byte accounting stays equal to what it actually holds.
struct Arena {
    std::vector<void *> taken;
    size_t bytes = 0;

    template <class T> void zeroed(T *& p, size_t elems) {
        const size_t n = elems * sizeof(T);
        void * q = nullptr;
        const hipError_t e = hipMalloc(&q, n);
        if (e != hipSuccess || !q) {
            (void) hipGetLastError();   // consume it, so the caller's next check is about its own work
            throw std::runtime_error("hipMalloc of " + gb(n) + " failed ("
                                     + (e != hipSuccess ? hipGetErrorString(e) : "null pointer") + ")");
        }
        taken.push_back(q);
        bytes += n;
        check(hipMemset(q, 0, n), "hipMemset");
        p = (T *) q;
    }
    void give_back() {
        for (void * q : taken) (void) hipFree(q);
        taken.clear();
        bytes = 0;
    }
};

template <int MODE> struct Traits {
    static constexpr bool four_bit = MODE == FFN_BATCH_IU4_A4;
    static constexpr bool integer  = MODE != FFN_BATCH_SCALED_A8;
    // Bytes per (k-slice, token) B fragment element.
    static constexpr size_t frag = MODE == FFN_BATCH_IU4_A4 ? 8 : MODE == FFN_BATCH_IU8_A8 ? 16 : 32;
    static constexpr float levels = four_bit ? 7.0f : 127.0f;
};

// ------------------------------------------------------------------ weight preparation

// Every weight image this module can hold. A repack pass fills the ones that are not resident yet
// and leaves the others alone, so a lazily prepared mode never rewrites what an earlier one built.
struct RepackDst {
    uint8_t * slice2 = nullptr;    // two-bit codes, [slice][row]
    uint8_t * lane2  = nullptr;    // two-bit codes, [row][slice]
    uint8_t * pairw  = nullptr;    // nine-valued pair codes, [row][slice]
    uint8_t * dense  = nullptr;    // five-trit HALO bytes, three lane-interleaved sections
    uint16_t * scales = nullptr;
};

// HALO block-run (32 rows x 128 K, 896 B) -> the requested images of two 16-row tiles.
// One thread per (32-row tile, block, row): it decodes the run's trits for its own row once and
// writes each image from that. Untimed; runs once per matrix per newly requested image set.
// `ts`/`bs` are the run steps of the 512-byte images and `tsd`/`bsd` the dense five-trit image's,
// because the order's rule reads the stream stride and those two block sizes do not share one.
// The scale image is tile-major in every arm: a run carries 32 bytes of scale, so its stream
// stride is 1,280 or 4,352 bytes and no order question reaches it.
__global__ __launch_bounds__(256) void k_repack(const uint8_t * __restrict__ src, RepackDst d,
        int nb, int tile0, int ntile32, int tstep, int bstep, int tstepd, int bstepd) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t) ntile32 * nb * 32;
    if (i >= total) return;
    const int lane = int(i & 31);
    const size_t rest = i >> 5;
    const int blk = int(rest % (size_t) nb), t = int(rest / (size_t) nb);

    const uint8_t * run = src + ((size_t) (tile0 + t) * nb + blk) * TILE_BLOCK_BYTES;
    const uint4 qa = *(const uint4 *) (run + tile_off_qs_a(lane));
    const uint2 qb = *(const uint2 *) (run + tile_off_qs_b(lane));
    const unsigned tail = *(const unsigned *) (run + tile_off_tail(lane));

    uint32_t w[8];
    halo_block_codes(qa, qb, tail, w);

    const int tile16 = 2 * t + (lane >> 4), r = lane & 15;
    // `tile16` is the destination's own row tile: `tile0` selects source rows only, and the two
    // halves of the down matrix are separate images that each start at their own tile zero.
    const size_t block = (size_t) tile16 * tstep + (size_t) blk * bstep;
    const size_t blockd = (size_t) tile16 * tstepd + (size_t) blk * bstepd;
    const size_t blocks = (size_t) tile16 * nb + blk;      // the scale image, always tile-major
    if (d.slice2) {
        uint8_t * q = d.slice2 + block * BLOCK_BYTES + r * 4;
#pragma unroll
        for (int s = 0; s < SLICES; ++s) *(uint32_t *) (q + s * 64) = w[s];
    }
    if (d.lane2) {
        uint8_t * q = d.lane2 + block * BLOCK_BYTES + r * 32;
#pragma unroll
        for (int s = 0; s < SLICES; ++s) *(uint32_t *) (q + s * 4) = w[s];
    }
    if (d.pairw) {
        uint32_t pc[8];
        pack_pair_block(w, pc);
        uint8_t * q = d.pairw + block * PAIR_BLOCK_BYTES + r * 32;
#pragma unroll
        for (int s = 0; s < SLICES; ++s) *(uint32_t *) (q + s * 4) = pc[s];
    }
    if (d.dense) {
        uint8_t db[26];
        pack_dense_block(w, db);
        uint8_t * q = d.dense + blockd * DENSE_BLOCK_BYTES;
#pragma unroll
        for (int j = 0; j < 16; ++j) q[r * 16 + j] = db[j];
#pragma unroll
        for (int j = 0; j < 8; ++j) q[DENSE_S1_OFF + r * 8 + j] = db[16 + j];
        q[DENSE_S2_OFF + r * 2] = db[24];
        q[DENSE_S2_OFF + r * 2 + 1] = db[25];
    }
    if (d.scales) d.scales[blocks * 16 + r] = (uint16_t) (tail >> 16);
}

// ------------------------------------------------------------------ activation producer

// One workgroup per (row, 1024-chunk). Same normalisation, folded sign vector and Hadamard order as
// the deployed prep_chunk_r, with the engine's NORM_EPS; only the quantiser's destination differs.
// Fragment (k-slice, token) holds the 16 K values of one token, which is what WMMA lane l%16 wants.
//
// The optimized modes run this kernel at the same instantiation as the control they match, so their
// activations are the same bits and not merely the same construction.
template <int MODE, bool NORM, bool HAS_SIGN>
__global__ __launch_bounds__(NT) void k_prep(const float * __restrict__ x, const float * __restrict__ kvec,
        void * __restrict__ Bout, float * __restrict__ cscale, int n, int Npad, int csstride) {
    __shared__ __attribute__((aligned(16))) float s[1024];
    __shared__ float red[NT / 32];
    const int nchunks = n / 1024, tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int row = blockIdx.x / nchunks, chunk = blockIdx.x - row * nchunks;
    const int e0 = wave * 128 + lane * 4, base = chunk * 1024 + e0;
    const size_t rb = (size_t) row * n;

    float inv = 1.0f;
    if constexpr (NORM) {
        float ss = 0.0f;
        for (int i = tid * 4; i < n; i += NT * 4) {
            const float4 t = *(const float4 *) (x + rb + i);
            ss = fmaf(t.x, t.x, ss); ss = fmaf(t.y, t.y, ss);
            ss = fmaf(t.z, t.z, ss); ss = fmaf(t.w, t.w, ss);
        }
        ss = block_sum_256(ss, red);
        inv = rsqrtf(ss / (float) n + NORM_EPS);
    }
    float4 v = *(const float4 *) (x + rb + base);
    if constexpr (HAS_SIGN) {
        const float4 k = *(const float4 *) (kvec + base);
        v.x *= inv * k.x; v.y *= inv * k.y; v.z *= inv * k.z; v.w *= inv * k.w;
    }
    { const float p0 = v.x + v.y, p1 = v.x - v.y, p2 = v.z + v.w, p3 = v.z - v.w;
      v.x = p0 + p2; v.y = p1 + p3; v.z = p0 - p2; v.w = p1 - p3; }
#pragma unroll
    for (int bit = 1; bit < 32; bit <<= 1) {
        const bool upper = (lane & bit) != 0;
        const float ox = __shfl_xor(v.x, bit, 32), oy = __shfl_xor(v.y, bit, 32);
        const float oz = __shfl_xor(v.z, bit, 32), ow = __shfl_xor(v.w, bit, 32);
        v.x = upper ? ox - v.x : v.x + ox; v.y = upper ? oy - v.y : v.y + oy;
        v.z = upper ? oz - v.z : v.z + oz; v.w = upper ? ow - v.w : v.w + ow;
    }
    *(float4 *) (s + e0) = v;
    __syncthreads();
    if (tid < 128) {
        float u[8];
#pragma unroll
        for (int m = 0; m < 8; ++m) u[m] = s[tid + 128 * m];
#pragma unroll
        for (int len = 1; len < 8; len <<= 1)
#pragma unroll
            for (int m = 0; m < 8; ++m) if ((m & len) == 0) {
                const float x0 = u[m], x1 = u[m + len]; u[m] = x0 + x1; u[m + len] = x0 - x1;
            }
#pragma unroll
        for (int m = 0; m < 8; ++m) s[tid + 128 * m] = u[m] * 0.03125f;
    }
    __syncthreads();
    v = *(const float4 *) (s + e0);

    // Deterministic symmetric quantiser over the 128-element block this wave owns.
    constexpr float L = Traits<MODE>::levels;
    const float amax = warp_max(fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    const float iscale = amax > 0.0f ? L / amax : 0.0f;
    int q[4] = { __float2int_rn(v.x * iscale), __float2int_rn(v.y * iscale),
                 __float2int_rn(v.z * iscale), __float2int_rn(v.w * iscale) };
#pragma unroll
    for (int j = 0; j < 4; ++j) q[j] = max(-int(L), min(int(L), q[j]));

    const int kslice = base >> 4, slot = base & 15;
    if constexpr (MODE == FFN_BATCH_IU4_A4) {
        uint8_t * dst = (uint8_t *) Bout + ((size_t) kslice * Npad + row) * 8 + slot / 2;
        dst[0] = uint8_t((q[0] & 0xf) | ((q[1] & 0xf) << 4));
        dst[1] = uint8_t((q[2] & 0xf) | ((q[3] & 0xf) << 4));
    } else if constexpr (MODE == FFN_BATCH_IU8_A8) {
        uint8_t * dst = (uint8_t *) Bout + ((size_t) kslice * Npad + row) * 16 + slot;
#pragma unroll
        for (int j = 0; j < 4; ++j) dst[j] = uint8_t(q[j] & 0xff);
    } else {
        // The eight-bit code carried back to a real value and rounded to FP16: the deployed input
        // semantics up to one rounding, with the token's block scale already folded in.
        const float scale = amax * (1.0f / 127.0f);
        _Float16 * dst = (_Float16 *) Bout + ((size_t) kslice * Npad + row) * 16 + slot;
#pragma unroll
        for (int j = 0; j < 4; ++j) dst[j] = (_Float16) ((float) q[j] * scale);
    }
    if constexpr (Traits<MODE>::integer)
        // Block-major: the projection's drain wants sixteen consecutive tokens of one block in one
        // load, so the token is the contiguous axis. See the operand cursors in k_proj_opt.
        if (lane == 0) cscale[(size_t) (base >> 7) * csstride + row] = amax / L;
}

// ------------------------------------------------------------------ control projections

// Integer modes. int32 accumulates one 128-block, then the epilogue drains it to FP32 against the
// weight's block scale and the token's block scale. FUSE folds SiLU, the product and the hidden
// sign into the store; otherwise the two row sets accumulate onto the incoming residual.
//
// Npad is the B fragment stride, Nloop the token axis this launch walks. They differ when the tile
// width does not divide the batch's tile count: the last step then reads fragment rows past the
// batch, which the allocation slack covers and the row guard discards.
template <int TT, bool A4, bool FUSE>
__global__ __launch_bounds__(NW * 32) void k_proj_int(
        const uint8_t * __restrict__ Wa, const uint8_t * __restrict__ Wb, const void * __restrict__ B,
        const uint16_t * __restrict__ sa, const uint16_t * __restrict__ sb,
        const float * __restrict__ cscale, const float * __restrict__ signs,
        const float * resid, float * out,
        int nb, int rows, int Npad, int Nloop, int outstride, int rowoff, int csstride,
        int tstep, int bstep) {
    using BF = typename std::conditional<A4, int2v, int4v>::type;
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int tile = blockIdx.x * NW + wave, col = lane & 15, half = lane >> 4;
    const int tg0 = blockIdx.y * TT * 16, tstride = gridDim.y * TT * 16;
    const uint8_t * abase = Wa + (size_t) tile * tstep * BLOCK_BYTES;
    const uint8_t * bbase = Wb + (size_t) tile * tstep * BLOCK_BYTES;

    for (int tg = tg0; tg < Nloop; tg += tstride) {
        float ya[TT][8], yb[TT][8];
#pragma unroll
        for (int t = 0; t < TT; ++t)
#pragma unroll
            for (int r = 0; r < 8; ++r) { ya[t][r] = 0.f; yb[t][r] = 0.f; }
        for (int blk = 0; blk < nb; ++blk) {
            // The 16 scales one wave needs for a block are 32 contiguous bytes, whichever run the
            // image's order puts this (tile, block) pair on.
            const int blkr = blk * bstep;
            const float asl = half_bits_to_float(sa[((size_t) tile * nb + blk) * 16 + col]);
            const float bsl = half_bits_to_float(sb[((size_t) tile * nb + blk) * 16 + col]);
            float asr[8], bsr[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) { asr[r] = __shfl(asl, 2 * r + half, 32); bsr[r] = __shfl(bsl, 2 * r + half, 32); }
            int8v aa[TT], ab[TT];
#pragma unroll
            for (int t = 0; t < TT; ++t) { aa[t] = int8v{}; ab[t] = int8v{}; }
#pragma unroll 1
            for (int s = 0; s < SLICES; ++s) {
                const unsigned wa = weight_word(abase, blkr, s, col), wb = weight_word(bbase, blkr, s, col);
#pragma unroll
                for (int t = 0; t < TT; ++t) {
                    const BF bf = ((const BF *) B)[(size_t) (blk * SLICES + s) * Npad + tg + t * 16 + col];
                    if constexpr (A4) {
                        aa[t] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, expand_i4(wa), true, bf, aa[t], false);
                        ab[t] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, expand_i4(wb), true, bf, ab[t], false);
                    } else {
                        aa[t] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, expand_i8(wa), true, bf, aa[t], false);
                        ab[t] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, expand_i8(wb), true, bf, ab[t], false);
                    }
                }
            }
#pragma unroll
            for (int t = 0; t < TT; ++t) {
                const float cs = cscale[(size_t) blk * csstride + tg + t * 16 + col];
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    ya[t][r] = __builtin_fmaf(float(aa[t][r]), asr[r] * cs, ya[t][r]);
                    yb[t][r] = __builtin_fmaf(float(ab[t][r]), bsr[r] * cs, yb[t][r]);
                }
            }
        }
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            const int token = tg + t * 16 + col;
            if (token >= rows) continue;
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int row = tile * 16 + 2 * r + half;
                const size_t o = (size_t) token * outstride + row;
                if constexpr (FUSE) out[o] = silu(ya[t][r]) * yb[t][r] * signs[row];
                else { out[o] = resid[o] + ya[t][r]; out[o + rowoff] = resid[o + rowoff] + yb[t][r]; }
            }
        }
    }
}

// Scaled-FP16 mode. The block scale is folded into the A operand while it is built, so there is no
// per-block epilogue at all: the FP32 accumulator holds the finished projection when the K loop
// ends. That is what lets this kernel carry eight token tiles where the integer maps, which need a
// second accumulator set, spill at eight.
template <int TT, bool FUSE>
__global__ __launch_bounds__(NW * 32) void k_proj_scaled(
        const uint8_t * __restrict__ Wa, const uint8_t * __restrict__ Wb, const _Float16 * __restrict__ B,
        const uint16_t * __restrict__ sa, const uint16_t * __restrict__ sb,
        const float * __restrict__ signs, const float * resid, float * out,
        int nb, int rows, int Npad, int Nloop, int outstride, int rowoff, int tstep, int bstep) {
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int tile = blockIdx.x * NW + wave, col = lane & 15, half = lane >> 4;
    const int tg0 = blockIdx.y * TT * 16, tstride = gridDim.y * TT * 16;
    const uint8_t * abase = Wa + (size_t) tile * tstep * BLOCK_BYTES;
    const uint8_t * bbase = Wb + (size_t) tile * tstep * BLOCK_BYTES;

    for (int tg = tg0; tg < Nloop; tg += tstride) {
        float8 ca[TT], cb[TT];
#pragma unroll
        for (int t = 0; t < TT; ++t) { ca[t] = float8{}; cb[t] = float8{}; }
        for (int blk = 0; blk < nb; ++blk) {
            const int blkr = blk * bstep;
            unsigned atlo, athi, btlo, bthi;
            scale_tables(sa[((size_t) tile * nb + blk) * 16 + col], atlo, athi);
            scale_tables(sb[((size_t) tile * nb + blk) * 16 + col], btlo, bthi);
#pragma unroll 1
            for (int s = 0; s < SLICES; ++s) {
                const half16 wa = expand_scaled(weight_word(abase, blkr, s, col), atlo, athi);
                const half16 wb = expand_scaled(weight_word(bbase, blkr, s, col), btlo, bthi);
#pragma unroll
                for (int t = 0; t < TT; ++t) {
                    const half16 bf = ((const half16 *) B)[(size_t) (blk * SLICES + s) * Npad + tg + t * 16 + col];
                    ca[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(wa, bf, ca[t]);
                    cb[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(wb, bf, cb[t]);
                }
            }
        }
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            const int token = tg + t * 16 + col;
            if (token >= rows) continue;
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int row = tile * 16 + 2 * r + half;
                const size_t o = (size_t) token * outstride + row;
                if constexpr (FUSE) out[o] = silu(ca[t][r]) * cb[t][r] * signs[row];
                else { out[o] = resid[o] + ca[t][r]; out[o + rowoff] = resid[o + rowoff] + cb[t][r]; }
            }
        }
    }
}

// ------------------------------------------------------------------ optimized projection
//
// One kernel for the three optimized modes. Against the control kernels above it changes two
// things and nothing else:
//
//   who owns a weight tile   SHARE = false is the control: a workgroup holds NW waves on NW
//                            different row tiles, and whatever the tile width cannot hold becomes
//                            a serial token group or a gridDim.y split, each of which re-reads the
//                            weight stream. SHARE = true gives one row tile to the whole
//                            workgroup and one token group to each of its WV waves, so the tokens
//                            covered per weight read grow by the wave count instead of by the
//                            accumulator count, and the row-tile grid grows by the wave count.
//                            It costs no registers and needs no reduction: an output element is
//                            still owned end to end by one wave.
//
//   how a block is loaded    WM_TWO with LAYOUT_LANE reads a lane's whole 128-block as two uint4
//                            instead of eight dwords; WM_PAIR always does; WM_DENSE reads 26 bytes
//                            in three pieces. LAYOUT_SLICE keeps the deployed word order with a
//                            one-slice lookahead.
//
// The A operand values, the accumulation, the scales, SiLU, the residual add and the output layout
// are the control's, so every output bit is the control's.
enum ProjOp { OP_IU8 = 0, OP_IU4 = 1, OP_F16 = 2 };
enum WeightMap { WM_TWO = 0, WM_PAIR = 1, WM_DENSE = 2 };

// Which of the block's operands are alive at once.
//
//   ORD_SLICE   the deployed order. A K16 slice is expanded once and immediately spent against
//               every token tile the wave holds, so the TT int32 accumulator pairs are all live
//               for the whole block. At four token tiles that is 64 of the kernel's 252 VGPRs, and
//               252 VGPRs is five waves per SIMD32.
//   ORD >= 1    the interchange, at ORD token tiles per accumulator group. The block's sixteen
//               fragments are expanded once into registers and then each group of ORD tiles runs
//               all eight slices and drains, so 2*ORD accumulator sets are live instead of 2*TT.
//               Each output element still accumulates slices 0..7 of a block in that order and
//               still drains once per block into the same ya/yb chain, so this is the deployed
//               arithmetic element for element, not a reassociation of it.
//
// The trade is wave slots against independent matrix chains. At TT=4 the deployed order holds
// eight chains (four token tiles times two matrices) at 252 VGPRs and five waves per SIMD32;
// ORD=2 holds four at 225 and six, ORD=1 two at 209 and seven. ORD=4 is the control that keeps
// all eight chains and changes only where the expansion happens, which is how the interchange and
// the chain count are told apart. docs/ffn-order.md measures all four.
enum ProjOrder { ORD_SLICE = 0 };

// How far the dense arm's block may reach for its memory, and what its scheduling barriers fence.
//
// The barriers in the dense block body are there to stop the compiler hoisting a whole block's
// radix-3 peel above the first matrix instruction: five peeled words per source dword would be
// live at once and the allocation runs past what the wave slots pay for. That is a VALU
// pressure problem, but `sched_barrier(0)` fences *every* class, so it also pinned the block's
// sixteen B-fragment loads to the slice pair that consumes them. The compiled loop issued four
// `global_load_b64` and waited on the first of them four instructions later, four times a block,
// with only the previous pair's four matrix instructions covering the latency.
//
// Mask 0x20 lets VMEM reads cross while still fencing VALU, so the peel stays where it was put and
// the loads may move: 46 waits per block become 39 and the loads arrive in clusters of eleven
// instead of four, for six registers and no wave slot. docs/ffn-dense-loads.md measures it.
//
// Read the distinction before copying this to another kernel: a barrier placed against VALU
// pressure should not fence memory, and a barrier placed against VMEM hoisting (the pair arm's, at
// four token tiles) must.
//
// `depth` is the weight cursor in ffn_dense_stream.hpp, which requests a block of operand bytes
// that many blocks before it hands them out. Three is measured and loses to two: it costs a wave
// slot (235 VGPR against 206, six waves per SIMD32 against seven) and returns less than the slot.
//
// Nothing here changes a value, an operand, an accumulation order or a store: every arm below
// issues the same instructions against the same bytes and only the schedule differs.
//
// Codes 4 and 5 are the DOWN stage's own depths and gate/up never sees them. Depth was one knob
// over two stages that are limited by opposite things: gate/up launches 1088 waves and is capped
// at seven per SIMD32 by its registers, so a third block in flight costs it a wave slot, while
// down launches 160 (320 split) onto 80 SIMD32 and never reaches its register ceiling at all - its
// resident set is the grid. A block in flight is free there and is the only lever it has on the
// memory latency it is waiting for. docs/ffn-down-waves.md.
struct DenseSched { unsigned bar; int depth; int probe; };
constexpr DenseSched dense_sched_of(int dls) {
    return dls == 1 ? DenseSched{ 0x20u, 2, 0 }
         : dls == 2 ? DenseSched{ 0x20u, 3, 0 }
         : dls == 3 ? DenseSched{ 0u, 3, 0 }
         : dls == 4 ? DenseSched{ 0x20u, 4, 0 }
         : dls == 5 ? DenseSched{ 0x20u, 6, 0 }
         // The ablation ladder (HALO_FFN_PROBE builds only). Same schedule as the selected arm,
         // same instructions, same request counts - only WHICH address a request names moves, so
         // the stream it walks collapses onto one block and hits in cache. These arms are
         // numerically invalid by construction and exist to say what the phase waits on.
         : dls == 8 ? DenseSched{ 0x20u, 2, 1 }   // activation fragments pinned to block 0
         : dls == 9 ? DenseSched{ 0x20u, 2, 2 }   // weight stream pinned to block 0
         : dls == 10 ? DenseSched{ 0x20u, 2, 3 }  // both pinned: the block loop's issue floor
         // 11 and 12 remove requests rather than move them. 11 gives every slice and token tile of
         // a block the same fragment, so the block's sixteen activation loads common-subexpression
         // into one: same bytes from the weight stream, same matrix work, fifteen fewer requests.
         // 12 drops the weight-scale broadcast's sixteen ds_bpermute.
         : dls == 11 ? DenseSched{ 0x20u, 2, 4 }
         : dls == 12 ? DenseSched{ 0x20u, 2, 8 }
         : dls == 13 ? DenseSched{ 0x20u, 2, 12 }
         : DenseSched{ 0u, 2, 0 };
}
// The ablation ladder is a build-time instrument, not a runtime one: at its default of 0 no probe
// instantiation is compiled and no line of the selected arm changes.
#ifndef HALO_FFN_PROBE
#define HALO_FFN_PROBE 0
#endif
constexpr int DLS_PROBE_LO = 8, DLS_PROBE_HI = 13;
} // namespace
// An opaque zero. A probe arm masks a block index or a cursor step with this, so the address stays
// loop-variant and the load stays inside the block loop; a literal 0 would let the compiler hoist
// the whole stream out and the arm would measure a kernel nobody runs. External linkage and a
// host-side write in create_ffn_batch keep it opaque to the device compiler.
__device__ int halo_ffn_probe_zero = 0;
namespace {
// The occupancy instrument for this kernel, and the answer it already gave.
//
// `-DHALO_FFN_WPE=8` asks the allocator for eight waves per SIMD32 instead of taking what it wants.
// It can deliver: the dense decode arm goes 206 VGPR to 191, the prefill pair arm 235 to 192 and
// the split down arm 120 to 118, with ZERO spills and zero scratch in all three - the allocator was
// spending slack, not holding liveness. And it is a **+54.7%** loss on the dense arm's phase, with
// the pair arm in the same build a -0.4% null because 194 registers do not reach the constraint.
// Those fifteen registers are instruction-level parallelism for the radix-3 peel, which no spill
// counter shows. Do not ask this kernel for occupancy; docs/ffn-decode-bytes.md has the panel.
//
// The default of 1 is codegen-identical to no attribute at all (254,282 instruction lines in the
// device listing, one differing line and it is the HIP compilation-unit hash), so this costs the
// served build nothing and lets the question be re-asked in one compile.
#ifndef HALO_FFN_WPE
#define HALO_FFN_WPE 1
#endif
// The activation stage's own wave budget, which is not the deployed body's. Staging costs this
// kernel about thirty registers, and unconstrained the allocator spends them and drops the block
// from seven waves per SIMD32 to six - which is worth more than the requests the stage removes.
// Asking for seven puts the allocation at 216 with nine spilled dwords instead. Only the staged
// instantiations carry this; every shipped body keeps exactly the attribute it had.
#ifndef HALO_FFN_BSTAGE_WPE
#define HALO_FFN_BSTAGE_WPE 7
#endif
constexpr int DLS_FENCED = 0;   // barriers fence everything, cursor two deep: the shape ca9d83a left
constexpr int DLS_LOADS  = 1;   // barriers let VMEM reads cross, cursor two deep: selected
constexpr int DLS_DN_MAX = 5;   // the deepest down-stage cursor the launcher carries

// MATS is how many of the projection's two weight matrices one wave owns.
//
// Two is the shape this kernel was written with, and for gate/up it is structural: the wave fuses
// SiLU(gate) * up in its own registers, so both matrices have to land in the same wave.
//
// For the down projection the two matrices are only the two row halves of one tensor. Pairing them
// is an arbitrary choice that HALVES the stage's wave count: DH/16 = 160 waves on this device's 80
// SIMD32 units, two per unit, which docs/ffn-decode-shape.md names as "latency with nothing to hide
// it behind". Every earlier repair bought waves by SHARING a row tile, and a sharing wave re-reads
// the whole weight stream - the trade the dense arm never takes.
//
// MATS = 1 buys the same waves for nothing: wave `own` takes row tile `own >> 1` of matrix
// `own & 1`, so 320 waves stream exactly the bytes 160 waves streamed. Only the B activation
// fragments are read by two waves instead of one, and those are a layer's 278 kB out of cache.
//
// Bit-identical by construction: an output element is still one wave's dot product over the same
// K order with the same per-block FP32 drain. Which wave owns it is not in the arithmetic.
template <int OP, int WM, int LAY, bool SHARE, int TT, int WV, bool FUSE, int ORD = ORD_SLICE,
          int DLS = DLS_FENCED, int MATS = 2, int BST = ffnb::BSTAGE_OFF, int SP = 1>
__global__ __launch_bounds__(WV * 32)
__attribute__((amdgpu_waves_per_eu(BST == ffnb::BSTAGE_OFF ? HALO_FFN_WPE : HALO_FFN_BSTAGE_WPE))) void k_proj_opt(
        const uint8_t * __restrict__ Wa, const uint8_t * __restrict__ Wb, const void * __restrict__ B,
        const uint16_t * __restrict__ sa, const uint16_t * __restrict__ sb,
        const float * __restrict__ cscale, const float * __restrict__ signs,
        const float * resid, float * out,
        int nb, int rows, int Npad, int Nloop, int outstride, int rowoff, int csstride,
        int tstep, int bstep) {
    using BF = typename std::conditional<OP == OP_F16, half16,
               typename std::conditional<OP == OP_IU4, int2v, int4v>::type>::type;
    using WOP = BF;                                        // A and B fragments have the same shape
    using ACC = typename std::conditional<OP == OP_F16, float8, int8v>::type;
    constexpr int WB = WM == WM_DENSE ? DENSE_BLOCK_BYTES : BLOCK_BYTES;
    constexpr int TGW = SHARE ? WV : 1;                    // token groups resident per workgroup

    // The activation stage, when this instantiation carries one. Its fragments are the same bytes
    // the block body used to load one at a time; see ffn_b_stage.hpp for what that is worth.
    using Stage = ffnb::BStage<BST, BF, SLICES, TT, WV>;
    __shared__ __attribute__((aligned(16))) BF stage_lds[BST == ffnb::BSTAGE_OFF ? 1 : Stage::BUFS * Stage::COPIES * Stage::ELEMS];
    const int lane = threadIdx.x & 31;
    // Every lane of a wave has the same threadIdx.x >> 5, but the uniformity analysis only knows
    // that threadIdx.x is divergent, so the wave index poisons everything derived from it: the
    // row tile, both weight bases and the token group. That is why every operand load in this
    // loop is addressed by a 64-bit VGPR pair instead of a scalar base and a 32-bit lane offset.
    // readfirstlane states the fact the hardware already guarantees and costs one instruction
    // outside the loop.
    const int wave = __builtin_amdgcn_readfirstlane(int(threadIdx.x) >> 5);
    const int col = lane & 15, half = lane >> 4;
    static_assert(MATS == 2 || !FUSE, "one matrix per wave needs an unfused stage: the fused arm reduces gate and up in the wave");
    const int own = __builtin_amdgcn_readfirstlane(SHARE ? int(blockIdx.x) : int(blockIdx.x) * WV + wave);
    const int tile = MATS == 2 ? own : (own >> 1);
    const int matsel = MATS == 2 ? 0 : (own & 1);     // 0 = this stage's Wa half, 1 = its Wb half
    const uint8_t * abase = (MATS == 2 || !matsel ? Wa : Wb) + (size_t) tile * tstep * WB;
    const uint8_t * bbase = Wb + (size_t) tile * tstep * WB;
    // One run of the image's order, in bytes: what a block step advances every weight cursor by.
    const int bstepb = __builtin_amdgcn_readfirstlane(bstep * WB);
    // The ablation ladder's two masks. Every shipped arm has PROBE == 0, where both fold to the
    // deployed expressions and no instruction moves; a probe arm collapses one of the two streams
    // onto its first block through an opaque zero, keeping the request count and losing the walk.
    // The ladder used to be dense-only, which left the two shapes that matter most unmeasurable:
    // the pair image has no cursor at all, and the deployed staged body was launched through a
    // fixed DLS_LOADS so no probe code ever reached it. Both now carry it; every shipped arm still
    // has PROBE == 0 and folds to the deployed expressions.
    constexpr int PROBE = dense_sched_of(DLS).probe;
    const int bmask = (PROBE & 1) ? halo_ffn_probe_zero : -1;
    const int wstepb = (PROBE & 2) ? (bstepb & halo_ffn_probe_zero) : bstepb;
    const int G = Nloop / (TT * 16), stride = int(gridDim.y) * TGW;

    for (int g = int(blockIdx.y) * TGW + (SHARE ? wave : 0); g < G; g += stride) {
        const int tg = g * TT * 16;
        // The dense arm streams its weight bytes ahead of use; see ffn_dense_stream.hpp. Every
        // other map compiles this to nothing.
        DenseStream<WM == WM_DENSE, dense_sched_of(DLS).depth, MATS> dense(abase, bbase, col, nb, wstepb);
        Stage bstage(stage_lds, (const BF *) B + tg, Npad, int(threadIdx.x), wave);
        if constexpr (BST != ffnb::BSTAGE_OFF) bstage.prime();
        // The block's two scalar operands, streamed a block ahead (ffn_dense_stream.hpp). The
        // cursor holds the words the NEXT block will spend; `awc/bwc/csc` are this block's, and the
        // rotation that fills them sits after the drain, so the wait it forces is a block old.
        ffnb::ScalarStream<SP != 0, TT, MATS == 2 || OP == OP_F16, OP != OP_F16> scal(
                (MATS == 2 || !matsel ? sa : sb) + (size_t) tile * nb * 16,
                sb + (size_t) tile * nb * 16, cscale + tg, col, 16, csstride);
        unsigned awc = 0, bwc = 0;
        float csc[TT];
        if constexpr (SP) {
            awc = scal.aw; bwc = scal.bw;
            if constexpr (OP != OP_F16)
#pragma unroll
                for (int t = 0; t < TT; ++t) csc[t] = scal.cs[t];
            scal.step(nb > 1);
        }
        ACC accA[TT], accB[TT];
        float ya[TT][8], yb[TT][8];
        if constexpr (OP == OP_F16) {
#pragma unroll
            for (int t = 0; t < TT; ++t) { accA[t] = ACC{}; accB[t] = ACC{}; }
        } else {
#pragma unroll
            for (int t = 0; t < TT; ++t)
#pragma unroll
                for (int r = 0; r < 8; ++r) { ya[t][r] = 0.f; yb[t][r] = 0.f; }
        }

        for (int blk = 0; blk < nb; ++blk) {
            // Operand cursors. Everything before the lane index is wave-uniform, so each of these
            // is a scalar base the hardware pairs with a 32-bit lane offset
            // (global_load_b64 v, v_off, s[base]). Folding the lane into a size_t index instead,
            // which is what this loop used to do, forces a 64-bit VGPR address pair per load and a
            // v_add_co pair to build each one; the lane offsets here are loop-invariant and the
            // bases advance by a constant per block.
            const BF * bblk = (const BF *) B + (size_t) ((blk & bmask) * SLICES) * Npad + tg;
            // The codes move with the image's run order; the scale image is tile-major in every
            // arm, because 32 bytes a run puts its stream stride nowhere near the aliasing zone.
            const int blkr = blk * bstep;          // this block's run index inside the wave's tile
            // The same collapse for every map that addresses its weights by the run index rather
            // than through DenseStream's cursor: the pair codes, the two-bit maps and FP16.
            const int blkw = (PROBE & 2) ? (blkr & halo_ffn_probe_zero) : blkr;
            const uint16_t * sab = (MATS == 2 || !matsel ? sa : sb) + ((size_t) tile * nb + blk) * 16;
            const uint16_t * sbb = sb + ((size_t) tile * nb + blk) * 16;
            const float * csb = cscale + (size_t) blk * csstride + tg;
            // Exactly the words the two expressions above name, from whichever schedule this
            // instantiation carries. Same value, same lane, same use.
            const unsigned awv = SP ? awc : (unsigned) sab[(unsigned) col];
            const unsigned bwv = (OP == OP_F16 || MATS == 2) ? (SP ? bwc : (unsigned) sbb[(unsigned) col]) : 0u;
#define OPT_CS(T) (SP ? csc[(T)] : csb[(unsigned) ((T) * 16 + col)])
            // The next block's fragments, requested a whole block before they are read - the same
            // cover the weight cursor gets, and the reason this stage is a schedule and not a copy.
            if constexpr (BST != ffnb::BSTAGE_OFF) if (blk + 1 < nb) bstage.fetch(blk + 1);
            unsigned atlo = 0, athi = 0, btlo = 0, bthi = 0;
            float asr[8], bsr[8];
            if constexpr (OP == OP_F16) {
                scale_tables((uint16_t) awv, atlo, athi);
                scale_tables((uint16_t) bwv, btlo, bthi);
            } else {
                const float asl = half_bits_to_float((uint16_t) awv);
#pragma unroll
                for (int r = 0; r < 8; ++r) asr[r] = (PROBE & 8) ? asl : __shfl(asl, 2 * r + half, 32);
                if constexpr (MATS == 2) {
                    const float bsl = half_bits_to_float((uint16_t) bwv);
#pragma unroll
                    for (int r = 0; r < 8; ++r) bsr[r] = (PROBE & 8) ? bsl : __shfl(bsl, 2 * r + half, 32);
                }
                if constexpr (ORD == ORD_SLICE)
#pragma unroll
                    for (int t = 0; t < TT; ++t) { accA[t] = ACC{}; if constexpr (MATS == 2) accB[t] = ACC{}; }
            }

            // One K16 slice against both matrices and every token tile the wave holds.
#define OPT_MAC(SL, WA_, WB_) { \
                const WOP wa_ = (WA_); \
                WOP wb_{}; \
                if constexpr (MATS == 2) wb_ = (WB_); \
                _Pragma("unroll") \
                for (int t = 0; t < TT; ++t) { \
                    const BF bf = BST != ffnb::BSTAGE_OFF ? bstage.read(blk, (SL), t, col) \
                                : (PROBE & 4) ? bblk[(unsigned) col] \
                                              : (bblk + (size_t) (SL) * Npad)[(unsigned) (t * 16 + col)]; \
                    if constexpr (OP == OP_F16) { \
                        accA[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(wa_, bf, accA[t]); \
                        if constexpr (MATS == 2) accB[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(wb_, bf, accB[t]); \
                    } else if constexpr (OP == OP_IU8) { \
                        accA[t] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, wa_, true, bf, accA[t], false); \
                        if constexpr (MATS == 2) accB[t] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, wb_, true, bf, accB[t], false); \
                    } else { \
                        accA[t] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, wa_, true, bf, accA[t], false); \
                        if constexpr (MATS == 2) accB[t] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, wb_, true, bf, accB[t], false); \
                    } \
                } }

            if constexpr (ORD >= 1) {
                // The whole block's A operand first: two uint4 per matrix, sixteen expansions, and
                // the sixteen fragments stay in registers across the token tiles that spend them.
                const uint8_t * ap = abase + (size_t) blkr * WB + (unsigned) col * 32;
                const uint8_t * bp = bbase + (size_t) blkr * WB + (unsigned) col * 32;
                const uint4 a0 = *(const uint4 *) ap, b0 = *(const uint4 *) bp;
                const uint4 a1 = *(const uint4 *) (ap + 16), b1 = *(const uint4 *) (bp + 16);
                WOP wf[SLICES], wg[SLICES];
                wf[0] = expand_pair(a0.x); wg[0] = expand_pair(b0.x);
                wf[1] = expand_pair(a0.y); wg[1] = expand_pair(b0.y);
                wf[2] = expand_pair(a0.z); wg[2] = expand_pair(b0.z);
                wf[3] = expand_pair(a0.w); wg[3] = expand_pair(b0.w);
                wf[4] = expand_pair(a1.x); wg[4] = expand_pair(b1.x);
                wf[5] = expand_pair(a1.y); wg[5] = expand_pair(b1.y);
                wf[6] = expand_pair(a1.z); wg[6] = expand_pair(b1.z);
                wf[7] = expand_pair(a1.w); wg[7] = expand_pair(b1.w);
                // ORD token tiles at a time, so the matrix pipe sees 2*ORD independent chains.
#pragma unroll
                for (int t0 = 0; t0 < TT; t0 += ORD) {
                    ACC pa[ORD], pb[ORD];
#pragma unroll
                    for (int u = 0; u < ORD; ++u) { pa[u] = ACC{}; pb[u] = ACC{}; }
#pragma unroll
                    for (int s = 0; s < SLICES; ++s)
#pragma unroll
                        for (int u = 0; u < ORD; ++u) {
                            const BF bf = (bblk + (size_t) s * Npad)[(unsigned) ((t0 + u) * 16 + col)];
                            pa[u] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, wf[s], true, bf, pa[u], false);
                            pb[u] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, wg[s], true, bf, pb[u], false);
                        }
#pragma unroll
                    for (int u = 0; u < ORD; ++u) {
                        const float cs = OPT_CS(t0 + u);
#pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            ya[t0 + u][r] = __builtin_fmaf(float(pa[u][r]), asr[r] * cs, ya[t0 + u][r]);
                            yb[t0 + u][r] = __builtin_fmaf(float(pb[u][r]), bsr[r] * cs, yb[t0 + u][r]);
                        }
                    }
                }
            } else if constexpr (WM == WM_DENSE) {
                // Slice order 0,1,2,3,6,4,5,7: slice 6 is finished by the fifth trits of the first
                // four dwords, and integer accumulation does not care in which order its terms
                // arrive. Two phases so the loaded bytes and the pending fifth-trit codes are not
                // all live at once.
                // A scheduling barrier after every second slice, for the same reason the pair arm
                // has them. Without one the compiler hoists the whole radix-3 peel of a block above
                // the first matrix instruction: five peeled words per source dword stay live at
                // once and the allocation runs past what the wave slots can pay for.
                //
                // The block's six weight requests are not here at all. The cursor issued them a
                // block earlier (ffn_dense_stream.hpp), so what the barriers fence is the peel,
                // and the wait this block pays was overlapped with the previous block's expansion
                // and matrix work rather than with the four slices between the two phases.
                constexpr unsigned DBAR = dense_sched_of(DLS).bar;
                const DenseBlockWords dw = dense.take(blk + 1 < nb);
                {
                    unsigned ca[4], cb[4]; (void) cb;
                    OPT_MAC(0, dense_slice(dw.a0.x, ca[0]), dense_slice(dw.b0.x, cb[0]))
                    OPT_MAC(1, dense_slice(dw.a0.y, ca[1]), dense_slice(dw.b0.y, cb[1])) __builtin_amdgcn_sched_barrier(DBAR);
                    OPT_MAC(2, dense_slice(dw.a0.z, ca[2]), dense_slice(dw.b0.z, cb[2]))
                    OPT_MAC(3, dense_slice(dw.a0.w, ca[3]), dense_slice(dw.b0.w, cb[3])) __builtin_amdgcn_sched_barrier(DBAR);
                    OPT_MAC(6, dense_slice_cc(ca[0], ca[1], ca[2], ca[3]),
                               dense_slice_cc(cb[0], cb[1], cb[2], cb[3]))
                }
                {
                    unsigned ca4, ca5, cb4, cb5;
                    OPT_MAC(4, dense_slice(dw.a1.x, ca4), dense_slice(dw.b1.x, cb4))
                    OPT_MAC(5, dense_slice(dw.a1.y, ca5), dense_slice(dw.b1.y, cb5)) __builtin_amdgcn_sched_barrier(DBAR);
                    OPT_MAC(7, dense_slice7(ca4, ca5, dw.a2), dense_slice7(cb4, cb5, dw.b2))
                }
            } else if constexpr (WM == WM_PAIR) {
                const uint8_t * ap = abase + (size_t) blkw * WB + col * 32;
                const uint8_t * bp = bbase + (size_t) blkw * WB + col * 32;
                // A scheduling barrier after every second slice, at four token tiles only. It was
                // fitted there, where without it the compiler hoists all thirty-two fragment loads
                // of the block above the first matrix instruction and spills 10-12 registers to
                // scratch; one slice of lookahead is enough to keep loads in flight (the deepest
                // wait in the compiled loop is vmcnt(12)) and leaves that shape spill-free at
                // 249-252 VGPR.
                //
                // At two token tiles the block holds sixteen fragment loads, not thirty-two, and
                // allocates 145 VGPR against 252, so there is nothing to spill and the same barrier
                // only forbids the lookahead. PAIR_SCHED keeps it exactly where it was measured.
#define PAIR_SCHED() do { if constexpr (TT >= 4) __builtin_amdgcn_sched_barrier(0); } while (0)
                const uint4 a0 = *(const uint4 *) ap, b0 = *(const uint4 *) bp;
                OPT_MAC(0, expand_pair(a0.x), expand_pair(b0.x))
                OPT_MAC(1, expand_pair(a0.y), expand_pair(b0.y)) PAIR_SCHED();
                OPT_MAC(2, expand_pair(a0.z), expand_pair(b0.z))
                OPT_MAC(3, expand_pair(a0.w), expand_pair(b0.w)) PAIR_SCHED();
                const uint4 a1 = *(const uint4 *) (ap + 16), b1 = *(const uint4 *) (bp + 16);
                OPT_MAC(4, expand_pair(a1.x), expand_pair(b1.x))
                OPT_MAC(5, expand_pair(a1.y), expand_pair(b1.y)) PAIR_SCHED();
                OPT_MAC(6, expand_pair(a1.z), expand_pair(b1.z)) PAIR_SCHED();
                OPT_MAC(7, expand_pair(a1.w), expand_pair(b1.w))
#undef PAIR_SCHED
            } else {
                const uint8_t * ap = abase + (size_t) blkw * WB + (unsigned) lane_base_off(LAY, col);
                const uint8_t * bp = bbase + (size_t) blkw * WB + (unsigned) lane_base_off(LAY, col);
                BlockWords<LAY> wa_w(ap), wb_w(bp);
                // The lane-major form needs the slice index constant so its two uint4 stay in
                // registers; the slice-major form keeps the deployed rolled loop with its
                // one-slice lookahead, which is what hides the load latency there.
#define OPT_SLICE(S) { \
                    const unsigned wa = wa_w.take(ap, (S)), wb = wb_w.take(bp, (S)); \
                    if constexpr (OP == OP_F16) { \
                        OPT_MAC((S), expand_scaled(wa, atlo, athi), expand_scaled(wb, btlo, bthi)) \
                    } else { \
                        OPT_MAC((S), expand_i8(wa), expand_i8(wb)) \
                    } }
                if constexpr (LAY == LAYOUT_LANE) {
#pragma unroll
                    for (int s = 0; s < SLICES; ++s) OPT_SLICE(s)
                } else {
#pragma unroll 1
                    for (int s = 0; s < SLICES; ++s) OPT_SLICE(s)
                }
#undef OPT_SLICE
            }
#undef OPT_MAC

            if constexpr (BST != ffnb::BSTAGE_OFF) {
                if (blk + 1 < nb) bstage.commit(blk + 1);
                bstage.sync();
            }
            if constexpr (OP != OP_F16 && ORD == ORD_SLICE) {
#pragma unroll
                for (int t = 0; t < TT; ++t) {
                    const float cs = OPT_CS(t);
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        ya[t][r] = __builtin_fmaf(float(accA[t][r]), asr[r] * cs, ya[t][r]);
                        if constexpr (MATS == 2)
                            yb[t][r] = __builtin_fmaf(float(accB[t][r]), bsr[r] * cs, yb[t][r]);
                    }
                }
            }
#undef OPT_CS
            // The rotation, after every use of this block's scalars: the wait it forces is for a
            // request made at the end of the previous block, and the request it makes has this
            // block's successor to land in.
            if constexpr (SP) {
                awc = scal.aw; bwc = scal.bw;
                if constexpr (OP != OP_F16)
#pragma unroll
                    for (int t = 0; t < TT; ++t) csc[t] = scal.cs[t];
                scal.step(blk + 2 < nb);
            }
        }

#pragma unroll
        for (int t = 0; t < TT; ++t) {
            const int token = tg + t * 16 + col;
            if (token >= rows) continue;
            // The residual add reads and writes the same buffer: `out` and `resid` are one pointer
            // at every call site, and neither can be __restrict__ for it. Written as sixteen
            // read-add-write triples the compiler must assume each store may land on the next
            // load's address, so it emits sixteen (global_load, s_waitcnt vmcnt(0), v_add_f32,
            // global_store) chains and pays a full memory round trip per output element. Reading
            // the whole group first leaves one wait for sixteen loads in flight. A lane reads
            // exactly the sixteen elements it writes and no other lane touches them, so the values,
            // the additions and their order are unchanged.
            const size_t moff = MATS == 2 ? 0 : (size_t) matsel * rowoff;
            float ra[8], rb[8];
            if constexpr (!FUSE) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const size_t o = (size_t) token * outstride + tile * 16 + 2 * r + half + moff;
                    ra[r] = resid[o]; if constexpr (MATS == 2) rb[r] = resid[o + rowoff];
                }
            }
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int row = tile * 16 + 2 * r + half;
                const size_t o = (size_t) token * outstride + row + moff;
                float va, vb{};
                if constexpr (OP == OP_F16) { va = accA[t][r]; if constexpr (MATS == 2) vb = accB[t][r]; }
                else { va = ya[t][r]; if constexpr (MATS == 2) vb = yb[t][r]; }
                if constexpr (FUSE) out[o] = silu(va) * vb * signs[row];
                else { out[o] = ra[r] + va; if constexpr (MATS == 2) out[o + rowoff] = rb[r] + vb; }
            }
        }
    }
}

// ------------------------------------------------------------------ state

constexpr int DH = D / 2;                  // the down matrix's two row halves
constexpr int NB_IN = D / KB;              // 40 scale blocks along the gate/up K
constexpr int NB_HID = FF / KB;            // 136 along the down K

// Blocks of a [rows][kdim] matrix, which is what every image's size is a multiple of.
inline size_t blocks_of(int rows, int kdim) {
    return (size_t) (rows / ROWS_PER_TILE) * (kdim / KB);
}
// Runs one matrix occupies under an order. Tile-major and block-major are permutations of the same
// set, so they hold exactly `blocks_of`; the padded order spends one extra run per row tile, which
// is 1/nb more bytes (2.5% of the gate/up image, 0.7% of the down one).
inline size_t image_block_bytes(int image) {
    return image == FFN_IMAGE_DENSE5 ? DENSE_BLOCK_BYTES : BLOCK_BYTES;
}
inline size_t image_runs(int order, int rows, int kdim, int bb) {
    const int ntiles = rows / ROWS_PER_TILE, nb = kdim / KB;
    const RunSteps s = run_steps(order, ntiles, nb, bb);
    return (size_t) (ntiles - 1) * s.tstep + (size_t) (nb - 1) * s.bstep + 1;
}
// One layer of one image: gate, up and the two down halves.
inline size_t image_layer_bytes(int image, int order) {
    const int bb = (int) image_block_bytes(image);
    return (2 * image_runs(order, FF, D, bb) + 2 * image_runs(order, DH, FF, bb)) * bb;
}

// One image's four matrices, laid out as four allocations of NLAYER layers each.
struct CodeImage {
    uint8_t * gate = nullptr, * up = nullptr, * dn_a = nullptr, * dn_b = nullptr;
    size_t bytes = 0;
    bool resident() const { return gate != nullptr; }
};

struct LayerSlab {
    const uint8_t * wg, * wu, * wd_a, * wd_b;
    const uint16_t * sg, * su, * sd_a, * sd_b;
};

} // namespace

struct FfnBatch {
    int max_rows = 0, npad_max = 0;
    // Token stride of the block-major activation scale arrays. Fixed for the module's lifetime so
    // a narrow pass addresses the same slots a wide one does and never reads a wider pass's
    // leftovers as its own slack.
    int cs_stride = 0;
    const LayerW * host_layers = nullptr;       // kept for lazily prepared modes
    const float * signs17408 = nullptr;
    const float * post_norm_s[NLAYER] = {};
    unsigned prepared = 0, options = 0;
    CodeImage img[FFN_IMAGE_COUNT];             // FFN_IMAGE_SCALES entry only carries its byte count
    uint16_t * sc_gate = nullptr, * sc_up = nullptr, * sc_dn_a = nullptr, * sc_dn_b = nullptr;
    void * Bin = nullptr, * Bhid = nullptr;     // FP16-sized, reused by every mode
    float * csin = nullptr, * cshid = nullptr;
    float * hidden = nullptr;
    size_t bytes = 0;
    int tt_gu_override = 0, tt_dn_override = 0;
    int w_gu_override = 0, w_dn_override = 0;
    int dn_mats = 0;           // 0/1 = one matrix per wave on the dense down stage, 2 = the pair
    int a4_sched = 2;   // SCHED_WIDE; see ffn_batch_set_a4_sched and mode 8 in run_ffn_batch.
    int a4_image = 0;   // 0 row-count rule, 1 dense five-trit, 2 pair codes; ffn_batch_set_a4_image.
    // Token tiles per accumulator group in the gate/up projection; see ProjOrder and
    // ffn_batch_set_a4_order. Two is selected: it is 489 issue slots per 128-block against the
    // slice-outer order's 508, at the same five waves per SIMD32. The down projection keeps the
    // slice-outer order: at two token tiles the interchange allocates more registers (158 against
    // 147) without changing its nine waves per SIMD32.
    int a4_order = 2;
    // Where a (row tile, K block) pair lands in the stored image; see FfnRunOrder and
    // ffn_batch_set_run_order. Every resident image and the shared scale image are written in it,
    // so changing it rewrites them all from the HALO source.
    int run_order = FFN_RUN_AUTO;
    // The order the images were SIZED and layer-strided for. Tile-major and block-major are
    // permutations of one run set and need the same bytes; the padded order needs one extra run
    // per row tile, so a process that wants to reach it says so before the images are allocated
    // (HALO_FFN_RUN_ORDER=1, or --ffn-run with a 1 in it, which the tools pass through).
    int run_alloc = FFN_RUN_AUTO;
    // The dense arm's load schedule; see DenseSched and ffn_batch_set_a4_dense. One is selected:
    // it takes 5.5% off the FFN phase of a 32-row pass and 3% off a 32-stream generation step,
    // bit-identically, and zero is the shape it replaced.
    int a4_dense = DLS_LOADS;
    // Where the block's activation fragments come from: 0 global loads (the shape this replaced),
    // 1 a workgroup-shared LDS stage, 2 a wave-private one. docs/ffn-b-operand.md.
    int b_stage = ffnb::BSTAGE_WAVE;
    // The block's scalar operands one block ahead of the block that spends them; see
    // ScalarStream in ffn_dense_stream.hpp and ffn_batch_set_spipe. One is selected.
    int spipe = 1;

    LayerSlab layer(int l, int image) const {
        const CodeImage & im = img[image];
        const size_t bb = image_block_bytes(image);
        const size_t cgu = image_runs(run_alloc, FF, D, (int) bb) * bb,
                     cdn = image_runs(run_alloc, DH, FF, (int) bb) * bb;
        const size_t sgu = scale_elems(FF, D), sdn = scale_elems(DH, FF);
        return { im.gate + (size_t) l * cgu, im.up + (size_t) l * cgu,
                 im.dn_a + (size_t) l * cdn, im.dn_b + (size_t) l * cdn,
                 sc_gate + (size_t) l * sgu, sc_up + (size_t) l * sgu,
                 sc_dn_a + (size_t) l * sdn, sc_dn_b + (size_t) l * sdn };
    }
};

namespace {

// Every layer of every image the four destinations name, written from the HALO source weights in
// `p->run_order`. One decode pass per matrix fills all of them, so a second image costs writes and
// not a second read of the source. Both the first preparation and a later change of run order go
// through here, so there is one description of what a stored image holds.
void write_images(FfnBatch * p, RepackDst gate, RepackDst up, RepackDst dn_a, RepackDst dn_b);

// Repack one HALO matrix, or one row-tile range of it, into every image `d` names.
void repack(const uint8_t * src, RepackDst d, int rows, int kdim, int row0, int order,
            hipStream_t stream = nullptr) {
    const int nb = kdim / KB, ntile32 = rows / 32, tile0 = row0 / 32;
    const RunSteps s = run_steps(order, rows / ROWS_PER_TILE, nb, BLOCK_BYTES);
    const RunSteps sd = run_steps(order, rows / ROWS_PER_TILE, nb, DENSE_BLOCK_BYTES);
    const size_t threads = (size_t) ntile32 * nb * 32;
    const int block = 256;
    const size_t grid = (threads + block - 1) / block;
    k_repack<<<dim3((unsigned) grid), dim3(block), 0, stream>>>(src, d, nb, tile0, ntile32,
                                                                s.tstep, s.bstep, sd.tstep, sd.bstep);
}

void write_images(FfnBatch * p, RepackDst gate, RepackDst up, RepackDst dn_a, RepackDst dn_b) {
    for (int l = 0; l < NLAYER; ++l) {
        const LayerW & L = p->host_layers[l];
        const auto at = [&](RepackDst d, int image, int rows, int kdim) {
            const size_t bb = image_block_bytes(image);
            const size_t off = (size_t) l * image_runs(p->run_alloc, rows, kdim, (int) bb) * bb;
            switch (image) {
                case FFN_IMAGE_SLICE2: if (d.slice2) d.slice2 += off; break;
                case FFN_IMAGE_LANE2:  if (d.lane2)  d.lane2  += off; break;
                case FFN_IMAGE_PAIR:   if (d.pairw)  d.pairw  += off; break;
                default:               if (d.dense)  d.dense  += off; break;
            }
            return d;
        };
        // The layer stride is the allocated order's run count for that image; where a (tile, block)
        // pair lands inside a layer is the selected order's business.
        const auto slab = [&](RepackDst d, int rows, int kdim, uint16_t * sc) {
            for (int image = FFN_IMAGE_SLICE2; image < FFN_IMAGE_COUNT; ++image) d = at(d, image, rows, kdim);
            if (d.scales) d.scales = sc + (size_t) l * blocks_of(rows, kdim) * ROWS_PER_TILE;
            return d;
        };
        repack(L.gate, slab(gate, FF, D, p->sc_gate), FF, D, 0, p->run_order);
        repack(L.up,   slab(up,   FF, D, p->sc_up),   FF, D, 0, p->run_order);
        // The down matrix splits at D/2, which is tile-aligned (2560 = 80 HALO row tiles).
        repack(L.down, slab(dn_a, DH, FF, p->sc_dn_a), DH, FF, 0, p->run_order);
        repack(L.down, slab(dn_b, DH, FF, p->sc_dn_b), DH, FF, DH, p->run_order);
    }
}

// Enough workgroups to fill the device without splitting a weight stream that is already spread
// over more workgroups than the machine has processors.
int group_split(int xgroups, int groups) {
    const int want = WAVES_WANTED / (xgroups > 0 ? xgroups : 1);
    return std::max(1, std::min(groups, want));
}

// Token tiles per wave, as measured by Kelana on this target. The integer modes carry two int32
// accumulator sets and spill past four; the scaled mode carries only the FP32 sums and takes eight
// where the grid still fills the device. Gate/up owns 272 workgroups before the token axis is split
// at all, so it always takes the widest tile the batch allows; down owns 40 and has to split the
// token axis, which re-reads its weights per split, so below 256 rows it prefers the splits.
//
// Four is a hard cap for the integer modes, not a tuning choice: eight token tiles there needs 256
// VGPRs and spills 120-128 of them, which the compiler confirms and Kelana measured as a 2.8x
// whole-FFN regression. The environment overrides exist so a sweep can re-fit the widths on another
// shape without a rebuild; they cannot reach the spilling shape.
int pick_tt(bool gate_up, bool integer, int Npad, int override_tt) {
    const int ntiles = std::max(1, Npad / 16);
    const int cap = std::min(integer ? 4 : MAXTT, ntiles);
    if (override_tt > 0) return std::max(1, std::min(override_tt, cap));
    if (integer) return std::min(Npad >= 64 ? 4 : Npad >= 32 ? 2 : 1, cap);
    return std::min(gate_up || Npad >= 256 ? MAXTT : 2, cap);
}

template <bool FUSE, int MODE>
void launch_proj(const FfnBatch * p, int Npad, int tiles, hipStream_t stream,
                 const uint8_t * Wa, const uint8_t * Wb, const void * B,
                 const uint16_t * sa, const uint16_t * sb, const float * cscale,
                 const float * signs, const float * resid, float * out,
                 int nb, int rows, int outstride, int rowoff, int csstride, int tstep, int bstep) {
    const bool integer = Traits<MODE>::integer;
    const int xg = tiles / NW;
    const int tt = pick_tt(FUSE, integer, Npad, FUSE ? p->tt_gu_override : p->tt_dn_override);
#define GRID(TT) const int nloop = (Npad + TT * 16 - 1) / (TT * 16) * (TT * 16); \
                 const dim3 grid(xg, group_split(xg, nloop / (TT * 16)))
#define GO_SCALED(TT) do { GRID(TT); \
        k_proj_scaled<TT, FUSE><<<grid, NW * 32, 0, stream>>>( \
            Wa, Wb, (const _Float16 *) B, sa, sb, signs, resid, out, nb, rows, Npad, nloop, outstride, \
            rowoff, tstep, bstep); \
    } while (0)
#define GO_INT(TT) do { GRID(TT); \
        k_proj_int<TT, Traits<MODE>::four_bit, FUSE><<<grid, NW * 32, 0, stream>>>( \
            Wa, Wb, B, sa, sb, cscale, signs, resid, out, nb, rows, Npad, nloop, outstride, rowoff, \
            csstride, tstep, bstep); \
    } while (0)
    if constexpr (MODE == FFN_BATCH_SCALED_A8) {
        if (tt >= 8)      GO_SCALED(8);
        else if (tt >= 4) GO_SCALED(4);
        else if (tt >= 2) GO_SCALED(2);
        else              GO_SCALED(1);
    } else {
        if (tt >= 4)      GO_INT(4);
        else if (tt >= 2) GO_INT(2);
        else              GO_INT(1);
    }
#undef GO_SCALED
#undef GO_INT
#undef GRID
}

// ------------------------------------------------------------------ optimized schedule
//
// A batch of `ntiles` 16-token tiles is spent three ways: TT tiles held in one wave's accumulators,
// W waves of a workgroup owning different token groups of the same weight tile, and whatever is
// left, walked serially or split across workgroups at the price of re-reading the weight stream.
// TT is operand reuse; W is grid parallelism that costs no registers and no re-read, because the W
// waves stream the same bytes together.
//
//   SCHED_TILE    W = 1 inside a workgroup of NW waves on NW different row tiles: the control.
//   SCHED_ADAPT   waves per shared row tile is the larger of what one pass over the batch needs at
//                 full width and what this stage's row-tile grid needs to fill the device; W = 1
//                 degenerates to the control. Kelana fitted this from a six-cell stage sweep and
//                 measured it at 32, 128 and 256 rows on two layers.
//   SCHED_WIDE    the same rule against eight waves per SIMD32 instead of four.
//
// Gate/up owns 1088 row tiles, so its grid is full without sharing and it keeps the control's
// mapping until the batch stops fitting in one pass at full width. The down projection owns 160,
// which is two workgroups per processor, so the control has to split its token axis and re-read
// its weights per split; sharing gives it 160 workgroups whose waves want the same weight rows.
//
// Four waves per SIMD32 turned out to be the wrong target for that thin stage. The integer maps
// carry two accumulator sets, so four token tiles allocate 249-252 VGPRs and the occupancy the
// hardware grants is five waves per SIMD32, not four; two tiles allocate 145-147 and get nine. At
// 128 rows the down stage measures 1.00x at the control, 1.10x at four waves per SIMD32 (W = 2,
// four tiles) and 1.21x at eight (W = 4, two tiles), where the extra wave slots cover the block's
// weight-load latency that the narrower resident set cannot. SCHED_WIDE is that target; it leaves
// the gate/up stage, whose 1088 row tiles already fill the device, at W = 2 and full width.
// docs/ffn-schedule.md holds the measurements.
enum Sched { SCHED_TILE = 0, SCHED_ADAPT = 1, SCHED_WIDE = 2 };

// Which of mode 8's two weight images a projection reads. Both carry the same weights; the dense
// five-trit byte is 1.625 bits per weight against the pair code's 2.000, and pays for it in the
// operand build. A4_IMAGE_RULE is the fitted row-count rule; the other two force one arm so a
// process can order them.
enum A4Image { A4_IMAGE_RULE = 0, A4_IMAGE_DENSE = 1, A4_IMAGE_PAIR = 2,
               A4_IMAGE_SPLIT = 3,      // dense gate/up, pair down
               A4_IMAGE_SPLIT_R = 4 };  // pair gate/up, dense down

// Row counts above which each stage of mode 8's rule leaves the dense five-trit image. The two
// stages cross at different widths because they are limited by different things, so one constant
// could not serve both. Fitted on the panel in docs/ffn-decode-shape.md.
// Measured at 32, 64 and 128 rows; 48 and 96 are interpolated, as mode 7's own rule is. What
// makes the two differ is how many times a stage walks its weight stream. The dense map's
// accumulator budget caps it at two token tiles, so above 32 rows gate/up reads its 2.85 GB twice
// and the pair map reads its 4.28 GB once - and gate/up loses. The down stage's pair arm is under
// the wide-wave target by then, four waves on one row tile at one token tile, so it reads its
// stream four times while the dense arm still reads it twice, and down wins the other way.
constexpr int A4_GU_PAIR_ROWS = 32;
constexpr int A4_DN_PAIR_ROWS = 64;

struct Shape { bool share; int w; int tt; };

int ceil_div(int a, int b) { return (a + b - 1) / b; }

// `cap` is the class's accumulator budget: 4 token tiles for the integer maps, 8 for scaled FP16.
// `tiles` is the projection's row-tile count, which is what decides whether its grid fills the
// device on its own.
//
// Both overrides exist so a sweep can re-fit this schedule on another shape without a rebuild, and
// so the two axes can be crossed. `override_w` is 1 for the control's ownership, 2 or 4 for that
// many waves sharing one row tile; the launcher carries no other wave count. Neither override can
// reach a shape the class's accumulator budget forbids.
Shape shape_for(int cap, int sched, bool gate_up, int Npad, int tiles, int override_tt, int override_w) {
    const int ntiles = std::max(1, Npad / 16);
    Shape s{};
    if (sched == SCHED_TILE) {
        const int want = cap >= MAXTT ? (gate_up || Npad >= 256 ? MAXTT : 2) : (Npad >= 64 ? 4 : 2);
        s = { false, NW, std::min(want, std::min(cap, ntiles)) };
    } else {
        const int target = sched == SCHED_WIDE ? WAVES_WANTED_WIDE : WAVES_WANTED;
        int w = std::min(NW, std::max(ceil_div(ntiles, cap), ceil_div(target, std::max(1, tiles))));
        if (w == 3) w = 2;    // only reachable at 129..192 rows; the launcher carries 1, 2 and 4
        s = w <= 1 ? Shape{ false, NW, std::min(cap, ntiles) }
                   : Shape{ true, w, std::max(1, std::min(cap, ceil_div(ntiles, w))) };
    }
    if (override_w > 0) {
        const int w = override_w >= 4 ? 4 : override_w >= 2 ? 2 : 1;
        s.share = w > 1;
        s.w = w > 1 ? w : NW;
        if (override_tt <= 0) s.tt = w > 1 ? std::max(1, std::min(cap, ceil_div(ntiles, w)))
                                           : std::min(cap, ntiles);
    }
    if (override_tt > 0) s.tt = std::max(1, std::min(override_tt, cap));
    return s;
}

struct ProjArgs {
    const uint8_t * Wa, * Wb;
    const void * B;
    const uint16_t * sa, * sb;
    const float * cscale, * signs, * resid;
    float * out;
    int nb, rows, Npad, outstride, rowoff, tiles, csstride;
    int tstep, bstep;      // the stored image's run order; see FfnRunOrder
};

// Only the pair-code IU4 arm carries the interchanged order: it is the one whose accumulator
// liveness costs occupancy, and its fragments are the cheap ones to hold. Every other
// instantiation keeps the deployed order and is not duplicated.
template <int OP, int WM, int LAY, bool SHARE, int TT, int WV, bool FUSE, int MATSC, int SP>
void launch_opt_sp(const ProjArgs & a, int Nloop, int ord, int dls, int bst, hipStream_t stream) {
    const int G = Nloop / (TT * 16);
    // One matrix per wave doubles the row-tile axis, so the same tiles come from twice the
    // workgroups. Weight bytes are unchanged: each wave streams one matrix instead of two.
    const int xg = (SHARE ? a.tiles : a.tiles / WV) * (MATSC == 1 ? 2 : 1);
    const int yg = group_split(xg, ceil_div(G, SHARE ? WV : 1));
    const dim3 grid(xg, yg);
#define LAUNCH_MATS(D) k_proj_opt<OP, WM, LAY, SHARE, TT, WV, FUSE, ORD_SLICE, D, MATSC, ffnb::BSTAGE_OFF, SP><<<grid, WV * 32, 0, stream>>>( \
        a.Wa, a.Wb, a.B, a.sa, a.sb, a.cscale, a.signs, a.resid, a.out, \
        a.nb, a.rows, a.Npad, Nloop, a.outstride, a.rowoff, a.csstride, a.tstep, a.bstep)
    // The activation stage replaces the block's sixteen fragment loads with one staging request and
    // ds_read, which is the measured -11.6% of this phase (docs/ffn-b-operand.md). It is built for
    // the arm that runs it: ternary codes, one token group per workgroup, the selected schedule.
#define LAUNCH_BST(S) k_proj_opt<OP, WM, LAY, SHARE, TT, WV, FUSE, ORD_SLICE, DLS_LOADS, MATSC, S, SP><<<grid, WV * 32, 0, stream>>>( \
        a.Wa, a.Wb, a.B, a.sa, a.sb, a.cscale, a.signs, a.resid, a.out, \
        a.nb, a.rows, a.Npad, Nloop, a.outstride, a.rowoff, a.csstride, a.tstep, a.bstep)
    if constexpr (OP == OP_IU4 && !SHARE && (WM == WM_DENSE || WM == WM_PAIR) && TT <= 2) {
#if HALO_FFN_PROBE
        // The staged body is the one the engine deploys at a generation width, and it used to be
        // unreachable from the ladder: this branch fires before the schedule dispatch below and
        // hardcodes DLS_LOADS, so `--ffn-dense 8..13` silently measured the deployed arm. These
        // instantiations exist only in a probe build.
#define LAUNCH_BSTP(D) k_proj_opt<OP, WM, LAY, SHARE, TT, WV, FUSE, ORD_SLICE, D, MATSC, ffnb::BSTAGE_WAVE, SP><<<grid, WV * 32, 0, stream>>>( \
        a.Wa, a.Wb, a.B, a.sa, a.sb, a.cscale, a.signs, a.resid, a.out, \
        a.nb, a.rows, a.Npad, Nloop, a.outstride, a.rowoff, a.csstride, a.tstep, a.bstep)
        if (bst == ffnb::BSTAGE_WAVE) {
            if (dls == 8)  { LAUNCH_BSTP(8);  return; }
            if (dls == 9)  { LAUNCH_BSTP(9);  return; }
            if (dls == 10) { LAUNCH_BSTP(10); return; }
            if (dls == 11) { LAUNCH_BSTP(11); return; }
            if (dls == 12) { LAUNCH_BSTP(12); return; }
            if (dls == 13) { LAUNCH_BSTP(13); return; }
        }
#undef LAUNCH_BSTP
#endif
        // The down stage's weight cursor, which the staged dispatch above used to swallow. `dls`
        // selects a depth for the unstaged body a hundred lines below, and this branch returned
        // before it with `DLS_LOADS` baked in, so from the moment the activation stage became the
        // default the deployed down stage ran depth two whatever anyone asked for - including the
        // depths docs/ffn-down-waves.md built for exactly this stage, which launches 320 waves onto
        // 80 SIMD32 and has a quarter of gate/up's cover for the same memory latency. Only the
        // depths the launcher already admits are instantiated, and only for the shape that has a
        // cursor at all; `dls` 0 and 1 fall through to the line they always took, so a default
        // build launches exactly the kernel it launched before.
#define LAUNCH_BSTD(D) k_proj_opt<OP, WM, LAY, SHARE, TT, WV, FUSE, ORD_SLICE, D, MATSC, ffnb::BSTAGE_WAVE, SP><<<grid, WV * 32, 0, stream>>>( \
        a.Wa, a.Wb, a.B, a.sa, a.sb, a.cscale, a.signs, a.resid, a.out, \
        a.nb, a.rows, a.Npad, Nloop, a.outstride, a.rowoff, a.csstride, a.tstep, a.bstep)
        if constexpr (WM == WM_DENSE && !FUSE) {
            if (bst == ffnb::BSTAGE_WAVE) {
                if (dls == 4) { LAUNCH_BSTD(4); return; }
                if (dls == 5) { LAUNCH_BSTD(5); return; }
            }
        }
#undef LAUNCH_BSTD
        if (bst == ffnb::BSTAGE_SHARED) { LAUNCH_BST(ffnb::BSTAGE_SHARED); return; }
        if (bst == ffnb::BSTAGE_WAVE)   { LAUNCH_BST(ffnb::BSTAGE_WAVE); return; }
    }
#undef LAUNCH_BST
    // Only the dense arm reads the block cursor and the peel barriers, so only it carries the load
    // schedule; every other map compiles one instantiation as before.
    if constexpr (OP == OP_IU4 && WM == WM_DENSE) {
        // The deep cursors exist for the down stage; gate/up is clamped before it gets here.
        if constexpr (!FUSE) {
            if (dls == 5) { LAUNCH_MATS(5); return; }
            if (dls == 4) { LAUNCH_MATS(4); return; }
        }
        if (dls == 1) { LAUNCH_MATS(1); return; }
        if (dls == 2) { LAUNCH_MATS(2); return; }
        if (dls == 3) { LAUNCH_MATS(3); return; }
#if HALO_FFN_PROBE
        if (dls == 8) { LAUNCH_MATS(8); return; }
        if (dls == 9) { LAUNCH_MATS(9); return; }
        if (dls == 10) { LAUNCH_MATS(10); return; }
        if (dls == 11) { LAUNCH_MATS(11); return; }
        if (dls == 12) { LAUNCH_MATS(12); return; }
        if (dls == 13) { LAUNCH_MATS(13); return; }
#endif
    }
#undef LAUNCH_MATS
#define LAUNCH(ORD) k_proj_opt<OP, WM, LAY, SHARE, TT, WV, FUSE, ORD, DLS_FENCED, MATSC, ffnb::BSTAGE_OFF, SP><<<grid, WV * 32, 0, stream>>>( \
        a.Wa, a.Wb, a.B, a.sa, a.sb, a.cscale, a.signs, a.resid, a.out, \
        a.nb, a.rows, a.Npad, Nloop, a.outstride, a.rowoff, a.csstride, a.tstep, a.bstep)
    if constexpr (OP == OP_IU4 && WM == WM_PAIR) {
        if (ord == 1) { LAUNCH(1); return; }
        if (ord == 2 && TT % 2 == 0) { LAUNCH(2); return; }
        if (ord == 4 && TT % 4 == 0) { LAUNCH(4); return; }
    }
    LAUNCH(ORD_SLICE);
#undef LAUNCH
}

// The scalar stream is an arm rather than a rewrite, so a panel can interleave it with the
// schedule it replaces in one process on one clock. Only the A4 maps carry both: they are the
// arms the block-pipeline panel measures, and a second instantiation of every FP16 shape would
// double this translation unit for a control nobody runs.
template <int OP, int WM, int LAY, bool SHARE, int TT, int WV, bool FUSE, int MATSC = 2>
void launch_opt_one(const ProjArgs & a, int Nloop, int ord, int dls, int bst, int sp,
                    hipStream_t stream) {
    if constexpr (OP == OP_IU4) {
        if (sp == 0) { launch_opt_sp<OP, WM, LAY, SHARE, TT, WV, FUSE, MATSC, 0>(a, Nloop, ord, dls, bst, stream); return; }
    }
    launch_opt_sp<OP, WM, LAY, SHARE, TT, WV, FUSE, MATSC, 1>(a, Nloop, ord, dls, bst, stream);
}

// The token axis is walked in whole tile groups, so a batch whose tile count the width does not
// divide is rounded up rather than given a narrower width. The extra columns are fragment slack,
// and the row guard discards their outputs.
template <int OP, int WM, int LAY, bool FUSE, int CAP, int MATSC = 2>
void launch_opt(const ProjArgs & a, const Shape & s, int Npad, int ord, int dls, int bst, int sp,
                hipStream_t stream) {
#define GO(SHARE, TTV, WV) do { \
        const int nloop = (Npad + TTV * 16 - 1) / (TTV * 16) * (TTV * 16); \
        launch_opt_one<OP, WM, LAY, SHARE, TTV, WV, FUSE, MATSC>(a, nloop, ord, dls, bst, sp, stream); \
        return; } while (0)
#define BY_TT(SHARE, WV) do { \
        if constexpr (CAP >= 8) { if (s.tt >= 8) GO(SHARE, 8, WV); } \
        if constexpr (CAP >= 4) { if (s.tt >= 4) GO(SHARE, 4, WV); } \
        if constexpr (CAP >= 2) { if (s.tt >= 2) GO(SHARE, 2, WV); } \
        GO(SHARE, 1, WV); } while (0)
    if (!s.share)         BY_TT(false, NW);
    else if (s.w >= 4)    BY_TT(true, 4);
    else                  BY_TT(true, 2);
#undef BY_TT
#undef GO
}

// ------------------------------------------------------------------ mode plumbing

// The images a mode reads. Mode 7 needs both two-bit orders because its small-batch schedule wants
// wide loads and its large-batch schedule wants the deployed word order; mode 8 holds a dense and
// a pair-code image unless an option restricts it to one.
unsigned images_for(int mode, unsigned options) {
    const unsigned scales = 1u << FFN_IMAGE_SCALES;
    switch (mode) {
        case FFN_BATCH_IU8_A8:
        case FFN_BATCH_IU4_A4:
        case FFN_BATCH_SCALED_A8:    return scales | (1u << FFN_IMAGE_SLICE2);
        case FFN_BATCH_IU8_A8_OPT:   return scales | (1u << FFN_IMAGE_LANE2);
        case FFN_BATCH_SCALED_A8_OPT:return scales | (1u << FFN_IMAGE_LANE2) | (1u << FFN_IMAGE_SLICE2);
        case FFN_BATCH_IU4_A4_OPT: {
            unsigned m = scales;
            if (!(options & FFN_BATCH_A4_DENSE_ONLY)) m |= 1u << FFN_IMAGE_PAIR;
            if (!(options & FFN_BATCH_A4_WIDE_ONLY))  m |= 1u << FFN_IMAGE_DENSE5;
            return m;
        }
        default: return 0;
    }
}

unsigned images_for_modes(unsigned modes, unsigned options) {
    unsigned m = 0;
    for (int mode = 0; mode < 32; ++mode)
        if (modes & (1u << mode)) m |= images_for(mode, options);
    return m;
}

template <int MODE>
void run_mode(FfnBatch * p, int layer, int rows, const float * input, float * output, hipStream_t stream) {
    const LayerSlab w = p->layer(layer, FFN_IMAGE_SLICE2);
    const int Npad = (rows + 15) / 16 * 16;

    k_prep<MODE, true, true><<<rows * (D / 1024), NT, 0, stream>>>(
        input, p->post_norm_s[layer], p->Bin, p->csin, D, Npad, p->cs_stride);
    const RunSteps rgu = run_steps(p->run_order, FF / ROWS_PER_TILE, NB_IN, BLOCK_BYTES);
    const RunSteps rdn = run_steps(p->run_order, DH / ROWS_PER_TILE, NB_HID, BLOCK_BYTES);
    launch_proj<true, MODE>(p, Npad, FF / 16, stream, w.wg, w.wu, p->Bin, w.sg, w.su,
        p->csin, p->signs17408, nullptr, p->hidden, NB_IN, rows, FF, 0, p->cs_stride,
        rgu.tstep, rgu.bstep);
    k_prep<MODE, false, false><<<rows * (FF / 1024), NT, 0, stream>>>(
        p->hidden, nullptr, p->Bhid, p->cshid, FF, Npad, p->cs_stride);
    launch_proj<false, MODE>(p, Npad, DH / 16, stream, w.wd_a, w.wd_b, p->Bhid, w.sd_a, w.sd_b,
        p->cshid, nullptr, input, output, NB_HID, rows, D, DH, p->cs_stride,
        rdn.tstep, rdn.bstep);
}

// One optimized mode's two projections. OP fixes the arithmetic and therefore which control this
// mode reproduces; IMAGE, WM, LAY and SCHED are the schedule, which changes no output bit.
// The class's accumulator budget in token tiles. Eight needs a second int32 accumulator set the
// integer maps do not have room for: Kelana measured that as a 2.8x whole-FFN regression with
// 120-128 registers spilled. Two is the dense map's ceiling here, not the class's: at four token
// tiles its pending fifth-trit codes push the allocation over 256 registers and spill. That shape
// is only reachable under FFN_BATCH_A4_DENSE_ONLY above 32 rows, and mode 8's own switch never
// asks for it.
template <int OP, int WM> struct TileCap {
    static constexpr int value = WM == WM_DENSE ? 2 : OP == OP_F16 ? MAXTT : 4;
};

// The two stages are separate calls because they do not want the same schedule and, for mode 8,
// not even the same weight image. Gate/up owns 1088 row tiles and fills the device; down owns
// DH/16 = 160 and runs two waves per SIMD32, so extra waves are worth far more there, and what
// they cost - every sharing wave rebuilds the same operands - is exactly what differs between the
// two images. `sched` is a plain argument: shape_for only ever read it as one, so per-stage
// ownership costs no extra kernel instantiation. docs/ffn-decode-shape.md.
template <int OP, int IMAGE, int WM, int LAY>
void run_gate_up(FfnBatch * p, int layer, int rows, int Npad, const float * input,
                 int sched, hipStream_t stream) {
    constexpr int PREP = OP == OP_F16 ? FFN_BATCH_SCALED_A8 : OP == OP_IU8 ? FFN_BATCH_IU8_A8 : FFN_BATCH_IU4_A4;
    constexpr int CAP = TileCap<OP, WM>::value;
    const LayerSlab w = p->layer(layer, IMAGE);
    k_prep<PREP, true, true><<<rows * (D / 1024), NT, 0, stream>>>(
        input, p->post_norm_s[layer], p->Bin, p->csin, D, Npad, p->cs_stride);
    const RunSteps r = run_steps(p->run_order, FF / ROWS_PER_TILE, NB_IN, (int) image_block_bytes(IMAGE));
    const ProjArgs a{ w.wg, w.wu, p->Bin, w.sg, w.su, p->csin, p->signs17408, nullptr,
                      p->hidden, NB_IN, rows, Npad, FF, 0, FF / 16, p->cs_stride, r.tstep, r.bstep };
    // Cursor depths 4 and 5 are the down stage's; gate/up keeps the depth its own occupancy pays
    // for, which is the selected one.
    launch_opt<OP, WM, LAY, true, CAP>(
        a, shape_for(CAP, sched, true, Npad, a.tiles, p->tt_gu_override, p->w_gu_override), Npad,
        p->a4_order, (p->a4_dense >= 4 && p->a4_dense < DLS_PROBE_LO) ? DLS_LOADS : p->a4_dense,
        p->b_stage, p->spipe, stream);
}

template <int OP, int IMAGE, int WM, int LAY>
void run_down(FfnBatch * p, int layer, int rows, int Npad, const float * input, float * output,
              int sched, hipStream_t stream) {
    constexpr int PREP = OP == OP_F16 ? FFN_BATCH_SCALED_A8 : OP == OP_IU8 ? FFN_BATCH_IU8_A8 : FFN_BATCH_IU4_A4;
    constexpr int CAP = TileCap<OP, WM>::value;
    const LayerSlab w = p->layer(layer, IMAGE);
    k_prep<PREP, false, false><<<rows * (FF / 1024), NT, 0, stream>>>(
        p->hidden, nullptr, p->Bhid, p->cshid, FF, Npad, p->cs_stride);
    const RunSteps r = run_steps(p->run_order, DH / ROWS_PER_TILE, NB_HID, (int) image_block_bytes(IMAGE));
    const ProjArgs a{ w.wd_a, w.wd_b, p->Bhid, w.sd_a, w.sd_b, p->cshid, nullptr, input,
                      output, NB_HID, rows, Npad, D, DH, DH / 16, p->cs_stride, r.tstep, r.bstep };
    // One matrix per wave is the dense arm's ownership. It is built only there: the pair arm buys
    // its waves by sharing a row tile, and the two-bit and FP16 maps keep the shape they were
    // measured on. Splitting halves the wave's accumulator budget as well as its weight stream,
    // which is what lets this stage hold FOUR token tiles where the paired dense arm holds two.
    constexpr bool CAN_SPLIT = OP == OP_IU4 && WM == WM_DENSE;
    if constexpr (CAN_SPLIT) {
        if (p->dn_mats != 2) {
            constexpr int CAPS = 4;
            launch_opt<OP, WM, LAY, false, CAPS, 1>(
                a, shape_for(CAPS, sched, false, Npad, a.tiles, p->tt_dn_override, p->w_dn_override),
                Npad, ORD_SLICE, p->a4_dense, p->b_stage, p->spipe, stream);
            return;
        }
    }
    launch_opt<OP, WM, LAY, false, CAP>(
        a, shape_for(CAP, sched, false, Npad, a.tiles, p->tt_dn_override, p->w_dn_override), Npad,
        ORD_SLICE, p->a4_dense, p->b_stage, p->spipe, stream);
}

template <int MODE, int OP, int IMAGE, int WM, int LAY, int SCHED>
void run_opt(FfnBatch * p, int layer, int rows, const float * input, float * output, hipStream_t stream) {
    const int Npad = (rows + 15) / 16 * 16;
    run_gate_up<OP, IMAGE, WM, LAY>(p, layer, rows, Npad, input, SCHED, stream);
    run_down<OP, IMAGE, WM, LAY>(p, layer, rows, Npad, input, output, SCHED, stream);
}

// Mode 8 dispatches its stages independently.
template <int LAY>
void run_a4_stages(FfnBatch * p, int layer, int rows, const float * input, float * output,
                   int img_gu, int img_dn, hipStream_t stream) {
    const int Npad = (rows + 15) / 16 * 16;
    if (img_gu == A4_IMAGE_DENSE)
        run_gate_up<OP_IU4, FFN_IMAGE_DENSE5, WM_DENSE, LAY>(p, layer, rows, Npad, input, SCHED_TILE, stream);
    else
        run_gate_up<OP_IU4, FFN_IMAGE_PAIR, WM_PAIR, LAY>(p, layer, rows, Npad, input, p->a4_sched, stream);
    if (img_dn == A4_IMAGE_DENSE)
        run_down<OP_IU4, FFN_IMAGE_DENSE5, WM_DENSE, LAY>(p, layer, rows, Npad, input, output, SCHED_TILE, stream);
    else
        run_down<OP_IU4, FFN_IMAGE_PAIR, WM_PAIR, LAY>(p, layer, rows, Npad, input, output, p->a4_sched, stream);
}

int env_int(const char * name) { const char * v = getenv(name); return v ? atoi(v) : 0; }
// The same for a knob whose unset value is NOT zero. `env_int` above reads an override whose
// "leave it alone" value happens to be 0 (a tile-count override, the down-matrix rule); a selected
// arm needs its own default back when the variable is absent, and reading the one-argument form
// into such a field silently ships the zero arm. That is what happened to the activation stage:
// `b_stage` defaulted to the wave stage in the struct and every process without HALO_FFN_BSTAGE
// set it to BSTAGE_OFF here, so the arm docs/ffn-b-operand.md measured and selected was not the
// arm the engine ran.
int env_int(const char * name, int dflt) { const char * v = getenv(name); return v ? atoi(v) : dflt; }

} // namespace

// ------------------------------------------------------------------ public interface

size_t ffn_batch_image_bytes(int image, int order) {
    if (image < 0 || image >= FFN_IMAGE_COUNT) return 0;
    if (order < FFN_RUN_TILE || order > FFN_RUN_BLOCK) order = FFN_RUN_TILE;
    if (image == FFN_IMAGE_SCALES)
        return (size_t) NLAYER * (2 * scale_elems(FF, D) + 2 * scale_elems(DH, FF)) * sizeof(uint16_t);
    return (size_t) NLAYER * image_layer_bytes(image, order);
}

size_t ffn_batch_workspace_bytes(int max_rows) {
    const int Npad = std::max(16, (max_rows + 15) / 16 * 16);
    return (size_t) (D / 16) * Npad * 32 + FRAG_SLACK
         + (size_t) (FF / 16) * Npad * 32 + FRAG_SLACK
         + ((size_t) (Npad + ROW_SLACK) * (NB_IN + NB_HID) + (size_t) Npad * FF) * sizeof(float);
}

size_t ffn_batch_resident_image_bytes(const FfnBatch * p, int image) {
    if (!p || image < 0 || image >= FFN_IMAGE_COUNT) return 0;
    return p->img[image].bytes;
}

size_t ffn_batch_weight_bytes(unsigned modes, int order) {
    const unsigned images = images_for_modes(modes & ~FFN_BATCH_OPTION_MASK, modes & FFN_BATCH_OPTION_MASK);
    size_t total = 0;
    for (int i = 0; i < FFN_IMAGE_COUNT; ++i) if (images & (1u << i)) total += ffn_batch_image_bytes(i, order);
    return total;
}

const char * ffn_batch_image_name(int image) {
    switch (image) {
        case FFN_IMAGE_SCALES: return "scales-fp16-tile-major";
        case FFN_IMAGE_SLICE2: return "two-bit-slice-major";
        case FFN_IMAGE_LANE2:  return "two-bit-lane-major";
        case FFN_IMAGE_PAIR:   return "pair-code-lane-major";
        case FFN_IMAGE_DENSE5: return "dense-five-trit";
        default:               return "?";
    }
}

const char * ffn_batch_mode_name(int mode) {
    switch (mode) {
        case FFN_BATCH_ENGINE:        return "engine";
        case FFN_BATCH_IU8_A8:        return "iu8-a8";
        case FFN_BATCH_IU4_A4:        return "iu4-a4";
        case FFN_BATCH_SCALED_A8:     return "scaled-f16-a8";
        case FFN_BATCH_IU8_A8_OPT:    return "iu8-a8-opt";
        case FFN_BATCH_SCALED_A8_OPT: return "scaled-f16-a8-opt";
        case FFN_BATCH_IU4_A4_OPT:    return "iu4-a4-opt";
        default:                      return "?";
    }
}

int ffn_batch_a4_sched(const FfnBatch * p) { return p ? p->a4_sched : 0; }
void ffn_batch_set_a4_sched(FfnBatch * p, int sched) {
    if (!p || sched < SCHED_TILE || sched > SCHED_WIDE)
        throw std::runtime_error("ffn_batch: row-tile ownership must be 0 (tile), 1 (adapt) or 2 (wide)");
    p->a4_sched = sched;
}

int ffn_batch_run_order(const FfnBatch * p) { return p ? p->run_order : FFN_RUN_TILE; }

const char * ffn_batch_run_order_name(int order) {
    switch (order) {
        case FFN_RUN_TILE:  return "tile-major";
        case FFN_RUN_PAD:   return "tile-major-padded";
        case FFN_RUN_BLOCK: return "block-major";
        case FFN_RUN_AUTO:  return "padded-where-aliased";
        default:            return "?";
    }
}

void ffn_batch_set_run_order(FfnBatch * p, int order, hipStream_t stream) {
    if (!p || order < FFN_RUN_TILE || order > FFN_RUN_AUTO)
        throw std::runtime_error("ffn_batch: run order must be 0 (tile-major), 1 (padded tile-major),"
                                 " 2 (block-major) or 3 (the stride rule)");
    if (order == p->run_order) return;
    if (image_runs(order, FF, D, BLOCK_BYTES) > image_runs(p->run_alloc, FF, D, BLOCK_BYTES)
        || image_runs(order, FF, D, DENSE_BLOCK_BYTES) > image_runs(p->run_alloc, FF, D, DENSE_BLOCK_BYTES))
        throw std::runtime_error("ffn_batch: the resident images were sized for the "
                                 + std::string(ffn_batch_run_order_name(p->run_alloc))
                                 + " order and the " + ffn_batch_run_order_name(order)
                                 + " one needs one more run per row tile. Set HALO_FFN_RUN_ORDER=1"
                                 " before the state is created to reach it.");
    const int previous = p->run_order;
    p->run_order = order;

    // Rewrite every resident image from the HALO source in the new order. The scale image is
    // shared by every mode and is written with the same run index, so it moves with them.
    RepackDst gate{}, up{}, dn_a{}, dn_b{};
    for (int image = FFN_IMAGE_SLICE2; image < FFN_IMAGE_COUNT; ++image) {
        const CodeImage & im = p->img[image];
        if (!im.resident()) continue;
        switch (image) {
            case FFN_IMAGE_SLICE2: gate.slice2 = im.gate; up.slice2 = im.up; dn_a.slice2 = im.dn_a; dn_b.slice2 = im.dn_b; break;
            case FFN_IMAGE_LANE2:  gate.lane2  = im.gate; up.lane2  = im.up; dn_a.lane2  = im.dn_a; dn_b.lane2  = im.dn_b; break;
            case FFN_IMAGE_PAIR:   gate.pairw  = im.gate; up.pairw  = im.up; dn_a.pairw  = im.dn_a; dn_b.pairw  = im.dn_b; break;
            default:               gate.dense  = im.gate; up.dense  = im.up; dn_a.dense  = im.dn_a; dn_b.dense  = im.dn_b; break;
        }
    }
    if (p->sc_gate) {
        gate.scales = p->sc_gate; up.scales = p->sc_up;
        dn_a.scales = p->sc_dn_a; dn_b.scales = p->sc_dn_b;
    }
    try {
        write_images(p, gate, up, dn_a, dn_b);
        check(hipDeviceSynchronize(), "weight image reorder");
    } catch (...) {
        p->run_order = previous;   // the images still hold whatever the failed pass left; say so
        throw;
    }
    (void) stream;
}

int ffn_batch_a4_image(const FfnBatch * p) { return p ? p->a4_image : 0; }
void ffn_batch_set_a4_image(FfnBatch * p, int image) {
    if (!p || image < A4_IMAGE_RULE || image > A4_IMAGE_SPLIT_R)
        throw std::runtime_error("ffn_batch: A4 image must be 0 (rule), 1 (dense), 2 (pair), "
                                 "3 (dense gate/up + pair down) or 4 (pair gate/up + dense down)");
    p->a4_image = image;
}

int ffn_batch_a4_dense(const FfnBatch * p) { return p ? p->a4_dense : 0; }
void ffn_batch_set_a4_dense(FfnBatch * p, int dls) {
    if (!p) return;
    const bool probe = HALO_FFN_PROBE && dls >= DLS_PROBE_LO && dls <= DLS_PROBE_HI;
    if ((dls < 0 || dls > DLS_DN_MAX) && !probe)
        throw std::runtime_error("ffn_batch_set_a4_dense: load schedule must be 0..5 (4 and 5 are the down stage's deep cursors)");
    // Say so when the shape cannot run what it is being asked for, instead of running something
    // else and reporting the request. `launch_opt_sp` dispatches the activation stage before the
    // schedule, so the staged body carries the depths it was instantiated with and nothing else:
    // the deployed cursor for 0 and 1, the down stage's deep cursors for 4 and 5, and the ablation
    // ladder in a probe build. Depth three has no staged instantiation. This used to be silent, and
    // from the moment the stage became the default at 21:30 on September 21 every `--ffn-dense`
    // panel above 1 was measuring the deployed arm and labelling it with the arm it asked for -
    // six ablation codes produced six different residual hashes and one unmoved phase, which reads
    // exactly like a clean null. docs/ffn-stream-ceiling.md has that retraction.
    if (p->b_stage != ffnb::BSTAGE_OFF && (dls == 2 || dls == 3))
        throw std::runtime_error("ffn_batch_set_a4_dense: the activation stage carries no depth-3 cursor, "
                                 "so this schedule cannot run with HALO_FFN_BSTAGE on; set the stage off first");
    p->a4_dense = dls;
}

int ffn_batch_b_stage(const FfnBatch * p) { return p ? p->b_stage : 0; }
void ffn_batch_set_b_stage(FfnBatch * p, int mode) {
    if (!p) return;
    if (mode < ffnb::BSTAGE_OFF || mode > ffnb::BSTAGE_WAVE)
        throw std::runtime_error("ffn_batch_set_b_stage: must be 0 (global loads), 1 (workgroup LDS stage) or 2 (wave LDS stage)");
    p->b_stage = mode;
}

int ffn_batch_spipe(const FfnBatch * p) { return p ? p->spipe : 0; }
void ffn_batch_set_spipe(FfnBatch * p, int mode) {
    if (!p) return;
    if (mode < 0 || mode > 1)
        throw std::runtime_error("ffn_batch_set_spipe: must be 0 (scalars read in the block that spends them) or 1 (a block ahead)");
    p->spipe = mode;
}

int ffn_batch_dn_mats(const FfnBatch * p) { return p ? p->dn_mats : 0; }
void ffn_batch_set_dn_mats(FfnBatch * p, int mats) {
    if (!p) return;
    if (mats < 0 || mats > 2)
        throw std::runtime_error("ffn_batch_set_dn_mats: must be 0 (rule), 1 (one matrix per wave) or 2 (the pair)");
    p->dn_mats = mats;
}

int ffn_batch_a4_order(const FfnBatch * p) { return p ? p->a4_order : 0; }
void ffn_batch_set_a4_order(FfnBatch * p, int order) {
    if (!p || (order != 0 && order != 1 && order != 2 && order != 4))
        throw std::runtime_error("ffn_batch: operand order must be 0 (slice-outer) or 1, 2, 4 token tiles per group");
    p->a4_order = order;
}

bool ffn_batch_supports(const FfnBatch * p, int mode) {
    if (!p || mode < 0 || mode >= 32) return false;
    return (p->prepared & (1u << mode)) != 0;
}

size_t ffn_batch_prepare_bytes(const FfnBatch * p, unsigned modes) {
    if (!p) return 0;
    const unsigned options = p->options;
    const unsigned have = images_for_modes(p->prepared, options);
    const unsigned need = images_for_modes(modes & ~FFN_BATCH_OPTION_MASK, options) & ~have;
    size_t total = 0;
    for (int i = 0; i < FFN_IMAGE_COUNT; ++i) if (need & (1u << i)) total += ffn_batch_image_bytes(i, p->run_alloc);
    return total;
}

size_t ffn_batch_prepare(FfnBatch * p, unsigned modes) {
    if (!p) return 0;
    modes &= ~FFN_BATCH_OPTION_MASK;
    unsigned want = 0;
    for (int mode = 0; mode < 32; ++mode)
        if ((modes & (1u << mode)) && images_for(mode, p->options)) want |= 1u << mode;
    if (!want) return 0;

    const unsigned have_images = images_for_modes(p->prepared, p->options);
    const unsigned need_images = images_for_modes(want, p->options) & ~have_images;
    if (!need_images) { p->prepared |= want; return 0; }

    size_t need = 0;
    for (int i = 0; i < FFN_IMAGE_COUNT; ++i) if (need_images & (1u << i)) need += ffn_batch_image_bytes(i, p->run_alloc);
    const std::string asked = "modes " + mode_list(want) + " (mask " + mask_hex(want) + ") need "
                            + gb(need) + " for images " + image_list(need_images);
    const std::string keeps = "this state still serves modes " + mode_list(p->prepared)
                            + " and holds " + gb(p->bytes);

    // Refuse a request the runtime already says will not fit, before taking any of it. The check is
    // advisory in both directions - another client can take memory after it passes, and the runtime
    // can be pessimistic - so the allocation path below still unwinds cleanly on its own.
    size_t freeb = 0, totalb = 0;
    if (device_memory(freeb, totalb) && need > freeb)
        throw std::runtime_error("ffn_batch: " + asked + "; hipMemGetInfo reports " + gb(freeb)
                                 + " free of " + gb(totalb) + ". Nothing was allocated; " + keeps
                                 + ". Prepare a smaller mode set, or the same one with"
                                   " FFN_BATCH_A4_WIDE_ONLY.");

    Arena arena;
    RepackDst gate{}, up{}, dn_a{}, dn_b{};
    try {
        const auto alloc_image = [&](int image, uint8_t *& g, uint8_t *& u, uint8_t *& a, uint8_t *& b) {
            if (!(need_images & (1u << image))) return;
            CodeImage & im = p->img[image];
            const size_t bb = image_block_bytes(image);
            // The block cursor (ffn_dense_stream.hpp) requests a tile's next block while it hands
            // out the current one, so the last blocks of the LAST row tile of an image are read
            // one or two blocks past its end. The values are never consumed - the loop has already
            // ended - but the address has to be mapped, and an image whose size lands exactly on a
            // page boundary faults. That is not hypothetical: it is what the pair-code version of
            // the cursor did, where 512 * nb * tiles is an exact multiple of the page size. Two
            // blocks of slack per image costs under a kilobyte and keeps the cursor's hot loop free
            // of a clamp, which measured eight instructions in a block loop of 587.
            // A block step is one run tile-major and `ntiles` runs block-major, so the cursor's
            // reach past the last block is widest in the block-major order. Every image carries
            // that much, so any order can be selected into an allocation this call made.
            const size_t sgu_slack = (size_t) CURSOR_SLACK_BLOCKS * (FF / ROWS_PER_TILE) * bb;
            const size_t sdn_slack = (size_t) CURSOR_SLACK_BLOCKS * (DH / ROWS_PER_TILE) * bb;
            const size_t cgu = (size_t) NLAYER * image_runs(p->run_alloc, FF, D, (int) bb) * bb + sgu_slack,
                         cdn = (size_t) NLAYER * image_runs(p->run_alloc, DH, FF, (int) bb) * bb + sdn_slack;
            arena.zeroed(im.gate, cgu);
            arena.zeroed(im.up,   cgu);
            arena.zeroed(im.dn_a, cdn);
            arena.zeroed(im.dn_b, cdn);
            im.bytes = 2 * cgu + 2 * cdn;
            g = im.gate; u = im.up; a = im.dn_a; b = im.dn_b;
        };
        alloc_image(FFN_IMAGE_SLICE2, gate.slice2, up.slice2, dn_a.slice2, dn_b.slice2);
        alloc_image(FFN_IMAGE_LANE2,  gate.lane2,  up.lane2,  dn_a.lane2,  dn_b.lane2);
        alloc_image(FFN_IMAGE_PAIR,   gate.pairw,  up.pairw,  dn_a.pairw,  dn_b.pairw);
        alloc_image(FFN_IMAGE_DENSE5, gate.dense,  up.dense,  dn_a.dense,  dn_b.dense);

        if (need_images & (1u << FFN_IMAGE_SCALES)) {
            const size_t sgu = (size_t) NLAYER * scale_elems(FF, D),
                         sdn = (size_t) NLAYER * scale_elems(DH, FF);
            arena.zeroed(p->sc_gate, sgu);
            arena.zeroed(p->sc_up,   sgu);
            arena.zeroed(p->sc_dn_a, sdn);
            arena.zeroed(p->sc_dn_b, sdn);
            p->img[FFN_IMAGE_SCALES].bytes = (2 * sgu + 2 * sdn) * sizeof(uint16_t);
            gate.scales = p->sc_gate; up.scales = p->sc_up;
            dn_a.scales = p->sc_dn_a; dn_b.scales = p->sc_dn_b;
        }

        write_images(p, gate, up, dn_a, dn_b);
        check(hipDeviceSynchronize(), "weight preparation");
    } catch (const std::exception & e) {
        // Undo exactly this call. Images the state already held keep their pointers and their
        // bytes, so `prepared`, ffn_batch_bytes() and every resident image stay true.
        for (int i = FFN_IMAGE_SLICE2; i < FFN_IMAGE_COUNT; ++i)
            if (need_images & (1u << i)) p->img[i] = CodeImage{};
        if (need_images & (1u << FFN_IMAGE_SCALES)) {
            p->sc_gate = p->sc_up = p->sc_dn_a = p->sc_dn_b = nullptr;
            p->img[FFN_IMAGE_SCALES].bytes = 0;
        }
        const std::string took = gb(arena.bytes);
        arena.give_back();
        throw std::runtime_error(std::string("ffn_batch: ") + e.what() + " while preparing " + asked
                                 + ". " + device_memory_phrase() + " now; gave back the " + took
                                 + " this call had taken, so " + keeps + ".");
    }
    p->bytes += arena.bytes;
    p->prepared |= want;
    return arena.bytes;
}

FfnBatch * create_ffn_batch(const LayerW * host_layers, const float * signs17408, int max_rows, unsigned modes) {
    auto * p = new FfnBatch{};
    p->max_rows = std::max(16, (max_rows + 15) / 16 * 16);
    p->npad_max = p->max_rows;
    p->cs_stride = p->npad_max + ROW_SLACK;
    p->host_layers = host_layers;
    p->signs17408 = signs17408;
    // Asking for both A4 restrictions at once asks for no A4 image at all; hold both instead.
    p->options = (modes & FFN_BATCH_OPTION_MASK) == FFN_BATCH_OPTION_MASK ? 0u : (modes & FFN_BATCH_OPTION_MASK);
    p->tt_gu_override = env_int("HALO_FFN_TT_GU");
    p->tt_dn_override = env_int("HALO_FFN_TT_DN");
    p->w_gu_override = env_int("HALO_FFN_W_GU");
    p->w_dn_override = env_int("HALO_FFN_W_DN");
    p->dn_mats = env_int("HALO_FFN_DN_MATS");
    if (p->dn_mats < 0 || p->dn_mats > 2) throw std::runtime_error("HALO_FFN_DN_MATS must be 0, 1 or 2");
    p->b_stage = env_int("HALO_FFN_BSTAGE", ffnb::BSTAGE_WAVE);
    if (p->b_stage < 0 || p->b_stage > 2) throw std::runtime_error("HALO_FFN_BSTAGE must be 0 (global), 1 (workgroup LDS) or 2 (wave LDS)");
    p->spipe = env_int("HALO_FFN_SPIPE", 1);
    if (p->spipe < 0 || p->spipe > 1) throw std::runtime_error("HALO_FFN_SPIPE must be 0 (scalars in the spending block) or 1 (a block ahead)");
    for (int w : { p->w_gu_override, p->w_dn_override })
        if (w && w != 1 && w != 2 && w != 4)
            throw std::runtime_error("ffn_batch: HALO_FFN_W_GU/HALO_FFN_W_DN must be 1, 2 or 4");
    if (const char * v = getenv("HALO_FFN_RUN_ORDER")) {
        const int order = atoi(v);
        if (order < FFN_RUN_TILE || order > FFN_RUN_AUTO)
            throw std::runtime_error("ffn_batch: HALO_FFN_RUN_ORDER must be 0 (tile-major), "
                                     "1 (padded tile-major), 2 (block-major) or 3 (the stride rule)");
        p->run_order = order;
        // Only the unconditional pad needs an extra run per row tile on EVERY image, so only a
        // process that names it allocates for it. The rule pads exactly the images it measured
        // faster and the other two orders fit inside whatever it takes.
        p->run_alloc = order == FFN_RUN_PAD ? FFN_RUN_PAD : FFN_RUN_AUTO;
    }
    if (const char * v = getenv("HALO_FFN_A4_IMAGE")) {
        const int img = atoi(v);
        if (img < A4_IMAGE_RULE || img > A4_IMAGE_SPLIT_R)
            throw std::runtime_error("ffn_batch: HALO_FFN_A4_IMAGE must be 0 (rule), 1 (dense), 2 (pair), 3 or 4 (split)");
        p->a4_image = img;
    }
    if (const char * v = getenv("HALO_FFN_A4_SCHED")) {
        const std::string value(v);
        if (value != "0" && value != "1" && value != "2")
            throw std::runtime_error("ffn_batch: HALO_FFN_A4_SCHED must be 0 (tile), 1 (adapt) or 2 (wide)");
        p->a4_sched = atoi(v);
    }
    if (const char * v = getenv("HALO_FFN_A4_ORDER")) {
        const std::string value(v);
        if (value != "0" && value != "1" && value != "2" && value != "4")
            throw std::runtime_error("ffn_batch: HALO_FFN_A4_ORDER must be 0 (slice-outer) or 1, 2, 4");
        p->a4_order = atoi(v);
    }
#if HALO_FFN_PROBE
    // Make the opaque zero a value the device compiler has seen written from outside the module.
    { const int z = 0; check(hipMemcpyToSymbol(HIP_SYMBOL(halo_ffn_probe_zero), &z, sizeof z), "probe zero"); }
#endif
    if (const char * v = getenv("HALO_FFN_A4_DENSE")) {
        const std::string value(v);
        const bool probe = HALO_FFN_PROBE && atoi(v) >= DLS_PROBE_LO && atoi(v) <= DLS_PROBE_HI;
        if (!probe && value != "0" && value != "1" && value != "2" && value != "3")
            throw std::runtime_error("ffn_batch: HALO_FFN_A4_DENSE must be 0 (fenced, depth 2), "
                                     "1 (loads may cross, depth 2), 2 (loads may cross, depth 3) "
                                     "or 3 (fenced, depth 3)");
        p->a4_dense = atoi(v);
    }
    for (int l = 0; l < NLAYER; ++l) p->post_norm_s[l] = host_layers[l].post_norm_s;

    // Shared activation workspace, sized for the widest fragment (FP16) so every mode reuses it.
    // A projection walks whole tile groups, so a batch whose tile count the width does not divide
    // reads up to one tile group of fragments past the batch; those columns alias earlier k-slices
    // or the slack, belong to invalid tokens, and their outputs are discarded by the row guard.
    const int Npad = p->npad_max;
    Arena work;
    try {
        work.zeroed((uint8_t *&) p->Bin,  (size_t) (D / 16) * Npad * 32 + FRAG_SLACK);
        work.zeroed((uint8_t *&) p->Bhid, (size_t) (FF / 16) * Npad * 32 + FRAG_SLACK);
        work.zeroed(p->csin,  (size_t) (Npad + ROW_SLACK) * NB_IN);
        work.zeroed(p->cshid, (size_t) (Npad + ROW_SLACK) * NB_HID);
        work.zeroed(p->hidden, (size_t) Npad * FF);
        p->bytes += work.bytes;
    } catch (const std::exception & e) {
        work.give_back();
        delete p;
        throw std::runtime_error(std::string("ffn_batch: ") + e.what() + " for the "
                                 + gb(ffn_batch_workspace_bytes(max_rows)) + " activation workspace at "
                                 + std::to_string(max_rows) + " rows. " + device_memory_phrase() + ".");
    }

    // A weight image that does not fit takes the whole state with it: there is nothing useful to
    // hand back, and a half-created state would be a worse thing to own than an exception.
    try {
        ffn_batch_prepare(p, modes);
    } catch (...) {
        destroy_ffn_batch(p);
        throw;
    }
    return p;
}

bool run_ffn_batch(FfnBatch * p, int layer, int mode, int rows,
                   const float * input, float * output, hipStream_t stream) {
    if (!p || layer < 0 || layer >= NLAYER || rows < 1 || rows > p->max_rows) return false;
    if (!ffn_batch_supports(p, mode)) return false;
    const int Npad = (rows + 15) / 16 * 16;
    switch (mode) {
        case FFN_BATCH_IU8_A8:    run_mode<FFN_BATCH_IU8_A8>(p, layer, rows, input, output, stream); return true;
        case FFN_BATCH_IU4_A4:    run_mode<FFN_BATCH_IU4_A4>(p, layer, rows, input, output, stream); return true;
        case FFN_BATCH_SCALED_A8: run_mode<FFN_BATCH_SCALED_A8>(p, layer, rows, input, output, stream); return true;

        // Mode 6: wide loads and the adaptive shared-tile schedule at every row count. Kelana
        // measured that as the best IU8 whole-FFN schedule at 32, 128 and 256 rows; the wide load
        // alone loses under control ownership, so the two levers ship together.
        case FFN_BATCH_IU8_A8_OPT:
            run_opt<FFN_BATCH_IU8_A8_OPT, OP_IU8, FFN_IMAGE_LANE2, WM_TWO, LAYOUT_LANE, SCHED_ADAPT>(
                p, layer, rows, input, output, stream);
            return true;

        // Mode 7: wide loads with the control's ownership up to two token tiles, the deployed word
        // order with the adaptive shared-tile schedule above. On this map the two levers do not
        // compose: their combination is the worst measured cell at 32 and 256 rows.
        //
        // The switch sits at the one measured size that prefers wide loads. 48 to 112 rows are
        // interpolation, not measurement: Kelana measured 32, 128 and 256, and named 64 as the
        // size where the gate/up stage's width preference and the down stage's fan-out preference
        // are closest. Both arms are correct at every row count, so re-fitting the threshold later
        // is a one-line change.
        //
        // Above 32 rows the wide-wave target replaces the four-per-SIMD32 one. It changes only the
        // down stage, to W = 4 and two token tiles, and measures 1.045x on the whole FFN at 128
        // rows. Gate/up keeps the control's mapping at full width under either target. The two
        // targets first differ again at 256 rows, which this engine's 128-row batch cap never
        // reaches, so that cell is unmeasured and unreachable.
        case FFN_BATCH_SCALED_A8_OPT:
            if (Npad <= 32)
                run_opt<FFN_BATCH_SCALED_A8_OPT, OP_F16, FFN_IMAGE_LANE2, WM_TWO, LAYOUT_LANE, SCHED_TILE>(
                    p, layer, rows, input, output, stream);
            else
                run_opt<FFN_BATCH_SCALED_A8_OPT, OP_F16, FFN_IMAGE_SLICE2, WM_TWO, LAYOUT_SLICE, SCHED_WIDE>(
                    p, layer, rows, input, output, stream);
            return true;

        // Mode 8: the dense five-trit image where the FFN is weight-bandwidth bound, pair codes
        // above it, where the dense byte's extra operand work costs more than its 19% fewer code
        // bytes buy. An option can restrict either side to one image. Same interpolation caveat as
        // mode 7: 32, 128 and 256 rows were measured, the sizes between them were not.
        //
        // The dense arm keeps the control's ownership. At 32 rows one wave already covers the whole
        // batch at two tiles, so sharing only idles wave slots and it measures 1.20x slower. The
        // pair arm takes the wide-target rule, which is worth 1.21x on the whole FFN at 128 rows.
        //
        // Which image wins below 33 rows is a measured question, not a property of the map, and
        //
        // All three ownerships stay reachable through ffn_batch_set_a4_sched, because they separate
        // two different questions and a later re-fit needs both answers: SCHED_ADAPT shares a weight
        // row tile across a workgroup's waves at all (with one tile per wave a 128-row batch needs
        // two passes at four token tiles, so each wave walked its tile's whole weight stream twice),
        // and SCHED_WIDE additionally aims the sharing at eight waves per SIMD32 rather than four,
        // which is what the thin 160-row-tile down stage wants. HALO_FFN_A4_SCHED pins one.
        case FFN_BATCH_IU4_A4_OPT: {
            const bool has_dense = p->img[FFN_IMAGE_DENSE5].resident();
            const bool has_pair  = p->img[FFN_IMAGE_PAIR].resident();
            // A dense stage always takes tile ownership: at 32 rows one wave already covers the
            // batch at two token tiles, so sharing only splits the token axis and makes every
            // extra wave rebuild the same five-trit operands. The whole arm forced to share
            // measures 33.8 -> 42.7 ms. A pair stage takes whatever ownership is configured,
            // because its operand build is cheap enough that the extra waves are worth it: the
            // pair down stage goes from 160 working waves to 640 across that axis and gains 21 ms
            // where the dense one loses 9. docs/ffn-decode-shape.md.
            int img_gu = A4_IMAGE_DENSE, img_dn = A4_IMAGE_DENSE;
            switch (p->a4_image) {
                case A4_IMAGE_DENSE:   break;
                case A4_IMAGE_PAIR:    img_gu = img_dn = A4_IMAGE_PAIR; break;
                case A4_IMAGE_SPLIT:   img_dn = A4_IMAGE_PAIR; break;
                case A4_IMAGE_SPLIT_R: img_gu = A4_IMAGE_PAIR; break;
                default:
                    if (Npad > A4_GU_PAIR_ROWS) img_gu = A4_IMAGE_PAIR;
                    if (Npad > A4_DN_PAIR_ROWS) img_dn = A4_IMAGE_PAIR;
                    break;
            }
            if (!has_dense) img_gu = img_dn = A4_IMAGE_PAIR;
            if (!has_pair)  img_gu = img_dn = A4_IMAGE_DENSE;
            run_a4_stages<LAYOUT_LANE>(p, layer, rows, input, output, img_gu, img_dn, stream);
            return true;
        }
        default: return false;
    }
}

void destroy_ffn_batch(FfnBatch * p) {
    if (!p) return;
    for (int i = 0; i < FFN_IMAGE_COUNT; ++i)
        for (void * q : { (void *) p->img[i].gate, (void *) p->img[i].up,
                          (void *) p->img[i].dn_a, (void *) p->img[i].dn_b })
            if (q) (void) hipFree(q);
    for (void * q : { (void *) p->sc_gate, (void *) p->sc_up, (void *) p->sc_dn_a, (void *) p->sc_dn_b,
                      p->Bin, p->Bhid, (void *) p->csin, (void *) p->cshid, (void *) p->hidden })
        if (q) (void) hipFree(q);
    delete p;
}

size_t ffn_batch_bytes(const FfnBatch * p) { return p ? p->bytes : 0; }

} // namespace halo

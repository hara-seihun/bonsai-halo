#include "head_batch.h"
#include "device.hpp"
#include "phases.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace halo {
namespace {

void check(hipError_t e) { if (e != hipSuccess) throw std::runtime_error(std::string("head_batch: ") + hipGetErrorString(e)); }

// -1 leaves the process default in force; a panel sets 0 or 1 per launch.
int head_arm_override = -1;
// -1 leaves the shape rule and HALO_HEAD_TT in force; a panel pins a token-group width per launch.
int head_tt_override = -1;

struct HeadArgs {
    const uint8_t * w;      // output.weight in HALO tiles: [VOCAB/32][NB_D] blocks of 896 bytes
    const uint8_t * B;      // wide int8 operand, int4v element (k16 * npad + token)
    const float * scales;   // [token][NB_D]
    const int * sums;       // [token][NB_D]
    float * logits;         // [token][VOCAB]
    // `lo` and `rows` are the half-open row selection: a logit is stored only for a token column in
    // `[lo, rows)`. `base` is the token column workgroup zero starts at, which is `lo` pushed down
    // far enough that the last group still lies inside the padded operand. The columns between
    // `base` and `lo` are read and their products thrown away, so a selection never costs more than
    // one token group of arithmetic beyond what it asked for.
    int rows, npad, base, lo;
};

// The zero-point term of the unsigned trit codes. The deployed quantiser produces it per
// (row, 128-block) as warp_sum_i over the block's four-element groups; integer addition is
// exact and order-free, so recovering it from the written operand gives the same value.
__global__ __launch_bounds__(256) void head_sums(const uint8_t * B, int * sums, int rows, int npad, int base) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    const int n = rows - base;
    if (i >= n * NB_D) return;
    const int b = i / n, token = base + (i - b * n);
    int acc = 0;
    #pragma unroll
    for (int s = 0; s < 8; s++) {
        const int8_t * p = (const int8_t *) (B + ((size_t) ((b * 8 + s) * npad + token)) * 16);
        #pragma unroll
        for (int j = 0; j < 16; j++) acc += p[j];
    }
    sums[(size_t) token * NB_D + b] = acc;
}

// One workgroup owns a 32-row weight tile and TT * 16 token columns. Wave w owns K blocks
// [5w, 5w + 5), the deployed unit split, and its five contributions enter one fmaf chain; wave 0
// then adds waves 1..7 in order. Every output logit therefore keeps the association the
// eight-row kernel gives it, and only the number of filled matrix columns changes.
//
// `OP` names the ternary operand map (`kernels/halo_expand.hpp`): 0 is the peel this kernel shipped
// with, 1 keeps the peel's product and gathers four trits per `v_perm_b32`. Both build the same 32
// operand dwords from the same 26 stored bytes, so the matrix instructions, the accumulator seed,
// the wave reduction and every logit bit are the arm above's.
//
// The arm enters as one call inside the block loop and nothing else moves. An earlier version of
// this switch carried the block's registers in a `HeadOperand<OP>::Regs` struct so the two arms
// could differ in what they load; that cost **89 work instructions** of arm 0's own 504, more than
// the expansion it was selecting, because the struct copy of the lookahead turned the compiler's
// SSA rename into 23 `v_mov_b32`. The control arm here is the deployed body instruction for
// instruction, which is what makes the difference below the operand map and nothing else.
template <int TT, int OP>
__global__ __launch_bounds__(NT) void head_tile(HeadArgs a) {
    constexpr int PW = NB_D / NW;   // 5 K-blocks per wave
    static_assert(PW * NW == NB_D, "the head's K blocks must split evenly over the waves");
    // Token groups are the fast grid axis, so the workgroups that share a weight tile are
    // dispatched together and only the first of them pays for the tile's memory traffic.
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, half = lane >> 4, col = lane & 15;
    const int tile = blockIdx.y, first = a.base + blockIdx.x * TT * 16;
    __shared__ float red[(NW - 1) * 8 * 32];
    __shared__ float scl_all[NW * 32];
    float * scl = scl_all + wave * 32;
    const int wb = wave * PW;
    const uint8_t * run = a.w + ((size_t) tile * NB_D + wb) * TILE_BLOCK_BYTES;

    float y1[TT][8], y2[TT][8];
    #pragma unroll
    for (int t = 0; t < TT; t++)
        #pragma unroll
        for (int r = 0; r < 8; r++) { y1[t][r] = 0.0f; y2[t][r] = 0.0f; }

    // One running operand offset per token group. The fragment a lane wants advances by exactly
    // npad * 16 bytes per K16 slice, and the eight slices of a block carry it to the next block, so
    // the whole K walk is one add per load instead of rebuilding (kblk * 8 + kb) * npad + token
    // each time. The offset stays 32-bit -- the operand is at most (D / 16) * 128 * 16 bytes -- so
    // the load keeps its scalar base and the addresses and their order are the control's.
    unsigned boff[TT];
    #pragma unroll
    for (int t = 0; t < TT; t++)
        boff[t] = ((unsigned) wb * 8 * (unsigned) a.npad + (unsigned) (first + t * 16 + col)) * 16u;
    const unsigned bstep = (unsigned) a.npad * 16u;

    uint4 qa = *(const uint4 *) (run + tile_off_qs_a(lane));
    uint2 qb = *(const uint2 *) (run + tile_off_qs_b(lane));
    unsigned tail = *(const unsigned *) (run + tile_off_tail(lane));
    #pragma unroll 1
    for (int b = 0; b < PW; b++) {
        uint4 nqa = qa; uint2 nqb = qb; unsigned ntail = tail;
        if (b + 1 < PW) {
            const uint8_t * nrun = run + (b + 1) * TILE_BLOCK_BYTES;
            nqa = *(const uint4 *) (nrun + tile_off_qs_a(lane));
            nqb = *(const uint2 *) (nrun + tile_off_qs_b(lane));
            ntail = *(const unsigned *) (nrun + tile_off_tail(lane));
        }
        unsigned tr[32];
        {
            const unsigned dw[6] = { qa.x, qa.y, qa.z, qa.w, qb.x, qb.y };
            if constexpr (OP == 1) hx_expand_perm(dw, tail, tr);
            else                   hx_expand_peel(dw, tail, tr);
        }
        // scale table: [0][m] = even weight row 2m, [1][m] = odd row 2m+1
        scl[(lane & 1) * 16 + (lane >> 1)] = __half2float(__ushort_as_half((unsigned short) (tail >> 16)));
        const float4 * s1p = (const float4 *) (scl + half * 16 + 8 * half);
        const float4 * s2p = (const float4 *) (scl + half * 16 + 8 * (half ^ 1));
        const float4 s1a = s1p[0], s1b = s1p[1], s2a = s2p[0], s2b = s2p[1];
        const float s1[8] = { s1a.x, s1a.y, s1a.z, s1a.w, s1b.x, s1b.y, s1b.z, s1b.w };
        const float s2[8] = { s2a.x, s2a.y, s2a.z, s2a.w, s2b.x, s2b.y, s2b.z, s2b.w };
        const int kblk = wb + b;
        #pragma unroll
        for (int t = 0; t < TT; t++) {
            const int token = first + t * 16 + col;
            const int xsc = a.sums[(size_t) token * NB_D + kblk];
            const float xsb = a.scales[(size_t) token * NB_D + kblk];
            // The A operand carries unsigned trit codes, so every product owes one activation per
            // weight and the epilogue removes the block's activation sum. Seeding the accumulator
            // with it instead costs the moves the zero seed already cost and removes one integer
            // subtraction per output element. Integer accumulation is exact and the block sum is at
            // most 16,256 against products under 32,512, so the drained value is the control's.
            const int z = -xsc;
            v8i C1 = { z, z, z, z, z, z, z, z }, C2 = { z, z, z, z, z, z, z, z };
            #pragma unroll
            for (int kb = 0; kb < 8; kb++) {
                const v4i A = { (int) tr[4 * kb], (int) tr[4 * kb + 1], (int) tr[4 * kb + 2], (int) tr[4 * kb + 3] };
                const v4i As = { swap16(A.x), swap16(A.y), swap16(A.z), swap16(A.w) };
                const int4 bx = *(const int4 *) (a.B + boff[t]);
                boff[t] += bstep;
                const v4i B = { bx.x, bx.y, bx.z, bx.w };
                C1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, A, true, B, C1, false);
                C2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, As, true, B, C2, false);
            }
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                y1[t][r] = fmaf((float) C1[r], s1[r] * xsb, y1[t][r]);
                y2[t][r] = fmaf((float) C2[r], s2[r] * xsb, y2[t][r]);
            }
        }
        qa = nqa; qb = nqb; tail = ntail;
    }

    // A1: half0 -> even rows 2r, half1 -> odd 2r+17; A2: half0 -> even 2r+16, half1 -> odd 2r+1.
#define HEAD_EMIT(Y, ODD, EVEN)                                                                  \
    do {                                                                                         \
        if (wave > 0) { _Pragma("unroll") for (int r = 0; r < 8; r++) red[((wave - 1) * 8 + r) * 32 + lane] = (Y)[r]; } \
        __syncthreads();                                                                         \
        if (wave == 0 && token >= a.lo && token < a.rows) {                                      \
            _Pragma("unroll") for (int r = 0; r < 8; r++) {                                      \
                float v = (Y)[r];                                                                \
                _Pragma("unroll") for (int w = 1; w < NW; w++) v += red[((w - 1) * 8 + r) * 32 + lane]; \
                a.logits[(size_t) token * VOCAB + tile * 32 + (half ? (ODD) : (EVEN))] = v;      \
            }                                                                                    \
        }                                                                                        \
        __syncthreads();                                                                         \
    } while (0)

    #pragma unroll
    for (int t = 0; t < TT; t++) {
        const int token = first + t * 16 + col;
        HEAD_EMIT(y1[t], 2 * r + 17, 2 * r);
        HEAD_EMIT(y2[t], 2 * r + 1, 2 * r + 16);
    }
#undef HEAD_EMIT
}

__global__ __launch_bounds__(NT) void head_argmax_slices(const float * logits, float * vals, int * idxs, int lo) {
    __shared__ float lds[64];
    const int row = lo + blockIdx.x, sl = blockIdx.y;
    argmax_slice(logits + (size_t) row * VOCAB, sl * 1024, min((sl + 1) * 1024, VOCAB),
                 vals + row * AMAX_SLICES, idxs + row * AMAX_SLICES, sl, lds);
}

__global__ __launch_bounds__(NT) void head_argmax_final(const float * vals, const int * idxs, int * out, int lo) {
    __shared__ float lds[64];
    const int row = lo + blockIdx.x;
    argmax_final(vals + row * AMAX_SLICES, idxs + row * AMAX_SLICES, out + row, lds);
}

} // namespace

struct HeadBatch {
    // `tt` is the HALO_HEAD_TT override, zero when the width is chosen per call.
    int capacity = 0, cols = 0, tt = 0;
    uint8_t * B = nullptr;
    float * scales = nullptr;
    int * sums = nullptr;
    float * logits = nullptr;
    float * amax_val = nullptr;
    int * amax_idx = nullptr;
    int * argmax = nullptr;
    size_t bytes = 0;
    std::vector<void *> allocations;
    template <class T> void alloc(T *& p, size_t n) {
        void * q = nullptr; check(hipMalloc(&q, n * sizeof(T)));
        allocations.push_back(q); p = (T *) q; bytes += n * sizeof(T);
        check(hipMemset(q, 0, n * sizeof(T)));
    }
    ~HeadBatch() { for (void * p : allocations) (void) hipFree(p); }
};

// Which operand map the head runs. The arms read the same image and write the same logits, so this
// is a measurement axis rather than a capability: `HALO_HEAD_OP=0` puts the peel back, and a panel
// that wants both arms under one clock flips `head_set_operand_arm` between launches instead of
// running two processes.
int head_operand_arm() {
    static int arm = [] {
        const char * v = getenv("HALO_HEAD_OP");
        if (!v) return 1;
        const std::string value(v);
        if (value != "0" && value != "1") throw std::runtime_error("HALO_HEAD_OP must be 0 or 1");
        return atoi(v);
    }();
    return head_arm_override >= 0 ? head_arm_override : arm;
}
void head_set_operand_arm(int arm) {
    if (arm < -1 || arm > 1) throw std::runtime_error("head operand arm must be -1 (env default), 0 or 1");
    head_arm_override = arm;
}

// Token columns per workgroup, as a panel axis. The width is a pure schedule - every width produces
// the same logits - and it trades weight traffic against accumulators: at 128 rows, TT = 1 reads the
// 278 MB image eight times and TT = 4 twice. `HALO_HEAD_TT` pins it for a process; this pins it for
// one launch so a panel can walk the ladder against the operand arm under one clock.
void head_set_tile_width(int tt) {
    if (tt != -1 && tt != 1 && tt != 2 && tt != 4 && tt != 8)
        throw std::runtime_error("head tile width must be -1 (shape rule), 1, 2, 4 or 8");
    head_tt_override = tt;
}

size_t head_batch_required_bytes(int capacity) {
    // The bound is the pass width, not the sequence-slot count: the head's buffers are indexed by
    // batch row, and MAXSLOTS only ever meant how many independent sequence states exist.
    if (capacity <= RMAX || capacity > PASSMAX) return 0;
    if (const char * v = getenv("HALO_WIDE_HEAD")) {
        const std::string value(v);
        if (value != "0" && value != "1") throw std::runtime_error("HALO_WIDE_HEAD must be 0 or 1");
        if (value == "0") return 0;
    }
    if (const char * v = getenv("HALO_HEAD_TT")) {
        const std::string value(v);
        if (value != "1" && value != "2" && value != "4" && value != "8")
            throw std::runtime_error("HALO_HEAD_TT must be 1, 2, 4 or 8");
    }
    const size_t cols = (capacity + 127) / 128 * 128;
    return (size_t)(D / 16) * cols * 16
        + cols * NB_D * sizeof(float)
        + cols * NB_D * sizeof(int)
        + (size_t)capacity * VOCAB * sizeof(float)
        + (size_t)capacity * AMAX_SLICES * sizeof(float)
        + (size_t)capacity * AMAX_SLICES * sizeof(int)
        + (size_t)capacity * sizeof(int);
}

HeadBatch * create_head_batch(int capacity) {
    if (capacity <= RMAX || capacity > PASSMAX) return nullptr;
    if (const char * v = getenv("HALO_WIDE_HEAD")) {
        const std::string value(v);
        if (value != "0" && value != "1") throw std::runtime_error("HALO_WIDE_HEAD must be 0 or 1");
        if (value == "0") return nullptr;
    }
    auto * h = new HeadBatch;
    try {
        h->capacity = capacity;
        // Token columns are read TT * 16 at a time, so the operand is padded to the widest group.
        h->cols = (capacity + 127) / 128 * 128;
        if (const char * v = getenv("HALO_HEAD_TT")) {
            const std::string value(v);
            if (value != "1" && value != "2" && value != "4" && value != "8")
                throw std::runtime_error("HALO_HEAD_TT must be 1, 2, 4 or 8");
            h->tt = atoi(v);
        }
        h->alloc(h->B, (size_t) (D / 16) * h->cols * 16);
        h->alloc(h->scales, (size_t) h->cols * NB_D);
        h->alloc(h->sums, (size_t) h->cols * NB_D);
        h->alloc(h->logits, (size_t) capacity * VOCAB);
        h->alloc(h->amax_val, (size_t) capacity * AMAX_SLICES);
        h->alloc(h->amax_idx, (size_t) capacity * AMAX_SLICES);
        h->alloc(h->argmax, (size_t) capacity);
        if (h->bytes != head_batch_required_bytes(capacity)) throw std::runtime_error("head memory estimate mismatch");
        return h;
    } catch (...) { delete h; throw; }
}

void destroy_head_batch(HeadBatch * h) { delete h; }
size_t head_batch_bytes(const HeadBatch * h) { return h ? h->bytes : 0; }
int head_batch_capacity(const HeadBatch * h) { return h ? h->capacity : 0; }
const float * head_logits(const HeadBatch * h) { return h ? h->logits : nullptr; }
const int * head_argmax(const HeadBatch * h) { return h ? h->argmax : nullptr; }

void head_bind(HeadBatch * h, FwdParams & p, int offset, int total) {
    if (!h || offset < 0 || p.nrows < 1 || offset + p.nrows > total || total > h->capacity)
        throw std::runtime_error("invalid wide head binding");
    p.sequence_q = (unsigned *) h->B;
    p.sequence_scales = h->scales;
    p.sequence_npad = (total + 15) / 16 * 16;
    p.sequence_offset = offset;
}

void run_head_batch(HeadBatch * h, const FwdParams & p, int total, int first_row, hipStream_t stream) {
    if (!h || total < 1 || total > h->capacity || !p.output)
        throw std::runtime_error("invalid wide head shape");
    if (first_row < 0 || first_row >= total) throw std::runtime_error("invalid wide head row selection");
    const int npad = (total + 15) / 16 * 16;
    const int selected = total - first_row;
    // Token columns per workgroup. Wider groups read the weight image fewer times but hold more
    // accumulators; measured on this device, two 16-column groups win from 32 rows upward and the
    // eight-group instantiation spills. The width follows the SELECTED rows rather than the pass:
    // a one-row selection wants one group, and it wants it whatever the pass around it is doing.
    // HALO_HEAD_TT overrides for comparisons, and every width produces the same logit bits.
    const int tt = head_tt_override > 0 ? head_tt_override : h->tt ? h->tt : (selected <= 16 ? 1 : 2);
    const int group = tt * 16;
    // Workgroup zero starts at the first selected column, pushed down only if the last group would
    // otherwise run past the padded operand. Both bounds then stay inside allocated columns and the
    // store guard drops whatever lands outside the selection.
    const int base = std::min(first_row, std::max(0, npad - group));
    head_sums<<<((total - base) * NB_D + NT - 1) / NT, NT, 0, stream>>>(h->B, h->sums, total, npad, base);
    HeadArgs a{ p.output, h->B, h->scales, h->sums, h->logits, total, npad, base, first_row };
    dim3 grid((total - base + group - 1) / group, VOCAB / 32);
    if (head_operand_arm() == 0) {
        switch (tt) {
            case 1: head_tile<1, 0><<<grid, NT, 0, stream>>>(a); break;
            case 2: head_tile<2, 0><<<grid, NT, 0, stream>>>(a); break;
            case 4: head_tile<4, 0><<<grid, NT, 0, stream>>>(a); break;
            default: head_tile<8, 0><<<grid, NT, 0, stream>>>(a); break;
        }
    } else {
        switch (tt) {
            case 1: head_tile<1, 1><<<grid, NT, 0, stream>>>(a); break;
            case 2: head_tile<2, 1><<<grid, NT, 0, stream>>>(a); break;
            case 4: head_tile<4, 1><<<grid, NT, 0, stream>>>(a); break;
            default: head_tile<8, 1><<<grid, NT, 0, stream>>>(a); break;
        }
    }
    head_argmax_slices<<<dim3(selected, AMAX_SLICES), NT, 0, stream>>>(h->logits, h->amax_val, h->amax_idx, first_row);
    head_argmax_final<<<selected, NT, 0, stream>>>(h->amax_val, h->amax_idx, h->argmax, first_row);
    check(hipGetLastError());
}

} // namespace halo

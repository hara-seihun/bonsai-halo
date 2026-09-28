// Shared host/device declarations: model constants, kernel argument structs, launchers.
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include "../src/halo_format.h"

namespace halo {

// Bonsai 2 27B (qwen35 hybrid) geometry
constexpr int D        = 5120;
constexpr int FF       = 17408;
constexpr int NLAYER   = 64;
constexpr int VOCAB    = 248320;
constexpr int HAD      = 1024;
// gated delta net
constexpr int HV       = 48;      // value heads
constexpr int HK       = 16;      // key heads
constexpr int SS       = 128;     // head dim / state size
constexpr int KDIM     = HK * SS; // 2048
constexpr int VDIM     = HV * SS; // 6144
constexpr int CONV_CH  = 2 * KDIM + VDIM; // 10240
constexpr int QKV_OUT  = CONV_CH;
// full attention
constexpr int NH       = 24;
constexpr int NKV      = 4;
constexpr int HD       = 256;
constexpr int NROT     = 64;
constexpr int Q_OUT    = NH * HD * 2;  // 12288, q and gate interleaved per head
constexpr int KV_OUT   = NKV * HD;     // 1024
constexpr int ATTN_OUT = NH * HD;      // 6144
constexpr int NATTN    = 16;           // attention layers (every 4th)
constexpr int MAXCTX   = 32768;
constexpr int ATTN_CHUNK = 256;
constexpr int ATTN_MAX_CHUNKS = MAXCTX / ATTN_CHUNK;
constexpr int TOK_RING = 64;   // pinned host ring of pending input tokens, indexed by position (v1 path)

// multi-row forward (verification blocks and batched sequences)
constexpr int RMAX     = 8;    // rows per pass
// Rows in one deployed FFN slice. `mvw_rows` feeds a 16-column iu8 WMMA, so a slice of eight rows
// issues exactly the instructions a slice of sixteen does and discards half the result. The FFN
// slice kernel reads no row metadata, no block cache and no state, so it can be wider than the pass
// row limit; `RMAX` still bounds anything that replays, rolls back or attends.
//
// Sixteen is one column group. Above it a unit takes a second group of sixteen columns against the
// same peeled weight fragment, so thirty-two rows cost one weight stream instead of two: the extra
// rows pay for their matrix instructions, their activation fragments and their accumulators, and
// for none of the block's load, radix-3 expansion or half swap. The buffers that scale with this
// are the engine's `gu` (2 * FMAX * FF floats) and quantised activation workspace.
//
// TWO GROUPS IS A PLATEAU, NOT A CEILING, AND THE LADDER STOPS BECAUSE THE PHASE STOPPED CARING.
// `-DHALO_SLICE_WIDE=1` builds the four-group rung (64 rows, `FMAX` below), which halves the
// weight stream and the radix-3 expansion of a 128-row pass and is **bit-identical and a measured
// null**: 0.02229 against 0.02233 ms per row per layer, two rounds each, same residual FNV-64. The
// register file is not what stops it - `tools/kernel_resources.py -D HALO_SLICE_PROBE=64` reports
// 240 VGPR at six waves per SIMD32 against the deployed 238, the same 8228-byte shared block and
// the same 60-workgroup grid - and neither is the weight stream. docs/ffn-slice-rows.md has the
// census and what it rules out; THREE groups is a 29% loss per row and is why this is a flag with
// two values rather than a free parameter.
#ifndef HALO_SLICE_WIDE
#define HALO_SLICE_WIDE 0
#endif
constexpr int FMID     = 32;   // the deployed two-group width
constexpr int FMAX     = HALO_SLICE_WIDE ? 64 : FMID;   // rows in one k_ffn_slice launch
constexpr int MAXSEQ   = 8;    // sequences in one persistent-kernel slice
constexpr int MAXSLOTS = 128;  // independent sequence state slots across wide batches
constexpr int KV_SLOTS = NATTN + 1 + 5; // 16 target attention layers + the MTP head + 5 DFlash2 layers
constexpr int KV_SLOT_ELEMS = NKV * MAXCTX * HD; // half elements per (seq, layer) slot; the drafter's 8 x 128 heads use the same size
constexpr int ACHUNK   = 128;  // attention keys per unit in the multi-row kernel
constexpr int AMAX_CHUNKS = MAXCTX / ACHUNK;
// The key cache's storage coordinate, and it is a lane placement rather than a packing. A score
// unit's dot product is a sum of thirty-two eight-dimension leaves; the deployed shape gives one
// leaf to each lane of a wave and pays a five-level cross-lane reduction per (key, row) to put
// them back together. Storing a key's leaves so that the SAME leaf of thirty-two CONSECUTIVE keys
// is contiguous lets one coalesced sixteen-byte load hand a lane a whole leaf of its own key: the
// reduction disappears and the query becomes wave-uniform. Same bytes, same cache lines per wave,
// same fibre per key - only which lane holds which leaf moves. Applies to the target geometry's K
// cache (HD = 256); V stays row-major because its loop is already dim-per-thread, and the
// drafter's own DF_HD layers keep the row-major form their kernel writes and reads.
constexpr int KTILE = 32;   // keys whose leaves interleave
constexpr int KLEAF = 8;    // dims in a leaf, and the length of one fma chain in the score unit
// Which score body a kernel compiles. The coordinate a cache is in is a process setting; which
// ALGORITHM reads it is a property of one kernel, and the two are independent because `k_pos_off`
// below lets the row-major body address either form.
constexpr int ATTN_KM_ROW  = 0;   // row-major score body alone - correct in either coordinate
constexpr int ATTN_KM_LEAF = 1;   // lane-per-key body alone - requires the leaf coordinate
constexpr int ATTN_KM_DYN  = 2;   // both, behind a runtime branch: the delivery control
__host__ __device__ __forceinline__ size_t k_leaf_off(int pos, int d, int hd) {
    return (size_t) (pos / KTILE) * (KTILE * hd) + (size_t) (d / KLEAF) * (KTILE * KLEAF)
         + (size_t) (pos % KTILE) * KLEAF + (size_t) (d % KLEAF);
}
// Where key `pos`'s dimensions [d, d + KLEAF) sit, in whichever coordinate the cache is in.
//
// THE COORDINATE IS AN ADDRESS, NOT A BODY. A whole eight-dimension leaf is contiguous in both
// forms - row-major keeps a key's 256 dims together, leaf-interleaved keeps a leaf of 32 keys
// together - so any reader that loads sixteen bytes of one key's leaf reads either cache by
// changing this offset and nothing else. Same bytes, same values, same order of use. That is what
// lets the lane-per-key ALGORITHM be a compile-time choice of one kernel while the coordinate
// stays a process setting every reader has to honour: a score unit does not need a second body to
// be correct in the other coordinate, only a different address. Kernels that carried both bodies
// behind a runtime branch paid the union of their register allocations - see
// docs/attn-score-lane.md for what that cost the persistent kernel.
__host__ __device__ __forceinline__ size_t k_pos_off(bool tiled, int pos, int d, int hd) {
    return tiled ? k_leaf_off(pos, d, hd) : (size_t) pos * hd + d;
}
// Tokens a tiled K cache must reserve for `n` positions. The engine rounds its context and every
// snapshot copy to this, so a tile is never partially inside the allocation.
__host__ __device__ __forceinline__ size_t k_tiled_tokens(size_t n) { return (n + KTILE - 1) / KTILE * KTILE; }
constexpr int GDN_STATE_FLOATS = HV * SS * SS;          // per layer per seq
constexpr int GDN_RING_FLOATS  = 3 * CONV_CH;            // per layer per seq
// Storage coordinate for that state. F32 is the deployed map and the exact one. The packed
// coordinates keep the recurrence's operand order and summation tree and change only the rounding
// of the value that crosses memory; they exist because a generation step's state round trip runs at
// the machine's memory bandwidth, where the only lever left is bytes. kernels/gdn_state_codec.hpp
// holds the layout and the argument for a shared exponent over fp16.
constexpr int GDN_STATE_F32 = 0, GDN_STATE_F16 = 1, GDN_STATE_I16 = 2, GDN_STATE_I8 = 3;
// The control that separates storage rounding from kernel selection: fp32 values, stored and read
// exactly as the deployed route stores them, but dispatched to the packed instantiations of the
// persistent kernel, which have their own registers and their own cooperative grid. Anything this
// arm moves is the kernel change, not the coordinate.
constexpr int GDN_STATE_F32_PK = 4;
// The memory coordinate of that int8 state, which is a free parameter and not a precision choice.
// int8 took a 32-stream step's round trip from 9.664 GB to 2.415 and the phase from 48.1 ms to
// 17.74 - 2.7x for 4x fewer bytes and 4x fewer requests - so the packed arm stopped being
// byte-bound and what was left of it was never counted. It is the codec, not the token loop: a
// generation-shape unit loads R rows, walks one token and stores R rows, and that load and store
// are 511 issue slots against the token body's 166.
//
// `GDN_STATE_I8` now stores the four rows one wave owns in one contiguous 512-byte run, so the wave
// reads them with one dwordx4 per lane instead of four dwords 1 kB apart, and writes the row's
// shared exponent with full exec instead of under `lane == 0`. Same int8 values, same rounding,
// same (lane, s) -> column map, same summation tree: docs/gdn-state-coord.md has the panel and the
// equal residual hashes. These two are the controls that measured it, kept so it can be re-measured
// rather than re-argued, and they are selectable in one process with `--gdn-state 3,5,6`:
//   I8C  the coordinate int8 had until 2026-09-21: row-major runs, `lane == 0` scale store.
//   I8S  row-major runs with the full-exec scale store, which separates the store predicate's
//        basic blocks from the request width. It is a 32-stream null and a 1.4% prefill win.
constexpr int GDN_STATE_I8C = 5, GDN_STATE_I8S = 6;
constexpr int GDN_STATE_FMT_MAX = GDN_STATE_I8S;
// The other half of that lever: how many steps may share one write-back. A step's whole effect on a
// head is one rank-1 update, so the packed coordinate's spare region can hold a few of them and the
// state itself needs writing only once per depth. The bound is what fits beside a packed head.
constexpr int GDN_DEFER_MAX = 8;
// An exact coordinate's pending term is wider - see gdn_fmt_defer_classes - and its measured
// ladder is flat from two steps up, so its list stops where the win already is. Four times two
// delta classes is eight times one, which is what keeps the staging arrays the same size for both.
constexpr int GDN_DEFER_EXACT_MAX = 4;

// ---------------------------------------------------------------------------------------------
// How much of a (slot, layer) region a coordinate actually occupies.
//
// `GDN_STATE_FLOATS` is the fp32 state's size, and it used to be the stride the engine allocated
// and every kernel addressed with in *every* coordinate. A packed head is a quarter of that at
// eight bits and a half at sixteen, so an int8 process reserved 3.000 MiB per (slot, layer) to
// hold 1.150 - 144 MiB of a 237 MiB sequence slot, and 18.0 GiB at 128 slots to hold 6.9. That
// reservation, not throughput, is what pinned the batched operating width: `hipMalloc` refused 64
// slots with both A4 images resident, and the 128-stream panel needed a 43 GiB process budget to
// exist at all. docs/state-region-stride.md has the panel.
//
// Everything here is an address. The values, their order, the (lane, s) -> column map, the
// summation tree and the rounding belong to kernels/gdn_state_codec.hpp; this only says where each
// part of a region starts, and a coordinate that changes no value cannot change a bit of output.
constexpr size_t GDN_PACK_SCALES = (size_t) HV * SS;                        // one per (head, row)
// How many roundings of one step's per-row delta a pending term has to carry.
//
// `warp_sum` does not leave one value in a wave, it leaves TWO: after the two quad stages the two
// rotations give lane l the sum in the order set by bit 2 of its index, so a wave ends a reduction
// holding two bit patterns of the same mathematical sum (docs/gdn-lane-layout.md measured the
// signature directly, and they differ in 793 of 2000 random waves). `gdn_token` then applies lane
// l's delta to the columns lane l owns, which makes the per-lane rounding part of the deployed
// numerical map. A pending term that keeps only lane 0's delta therefore replays one class over
// all 128 columns, and at fp32 - where nothing else rounds - that is visible: it diverged 9 of 32
// greedy continuations in eight steps. An exact coordinate keeps both classes and is bit-identical
// to the eager path; a packed one rounds the state to eight or sixteen bits on every commit, far
// past the gap between the classes, and keeps the single-delta term it was built and measured with.
constexpr bool gdn_fmt_is_eight_bit(int fmt) {
    return fmt == GDN_STATE_I8 || fmt == GDN_STATE_I8C || fmt == GDN_STATE_I8S;
}
constexpr bool gdn_fmt_is_exact(int fmt) { return fmt == GDN_STATE_F32 || fmt == GDN_STATE_F32_PK; }
constexpr int gdn_fmt_defer_classes(int fmt) { return gdn_fmt_is_exact(fmt) ? 2 : 1; }
constexpr int gdn_fmt_defer_cap(int fmt) { return gdn_fmt_is_exact(fmt) ? GDN_DEFER_EXACT_MAX : GDN_DEFER_MAX; }
// One head's pending block: g[C], a[C][classes][SS], k[C][SS].
constexpr size_t gdn_fmt_defer_head(int fmt) {
    return (size_t) gdn_fmt_defer_cap(fmt) * (size_t) (1 + (1 + gdn_fmt_defer_classes(fmt)) * SS);
}
constexpr size_t GDN_DEFER_HEAD  = (size_t) GDN_DEFER_MAX * (1 + 2 * SS);   // the packed block, unchanged
// Floats of a region holding packed values: the whole thing at fp32, half at sixteen bits, a
// quarter at eight.
constexpr size_t gdn_fmt_value_floats(int fmt) {
    return gdn_fmt_is_exact(fmt)     ? (size_t) GDN_STATE_FLOATS
         : gdn_fmt_is_eight_bit(fmt) ? (size_t) GDN_STATE_FLOATS / 4
                                     : (size_t) GDN_STATE_FLOATS / 2;
}
// The per-row shared exponents start where the values end, the deferred commit's control word
// after them, and its pending triples after that.
constexpr size_t gdn_fmt_scale_off(int fmt)  { return gdn_fmt_value_floats(fmt); }
constexpr size_t gdn_fmt_defer_ctl(int fmt)  { return gdn_fmt_scale_off(fmt) + GDN_PACK_SCALES; }
constexpr size_t gdn_fmt_defer_off(int fmt)  { return gdn_fmt_defer_ctl(fmt) + 4; }
// Floats a region needs to hold its values, its scales and a full pending list. A packed
// coordinate gets this for nothing because its values are a quarter or a half of the fp32 ones;
// an exact coordinate has to be given the room, which is what a deferring fp32 process asks for
// and what `gdn_defer_reserve_depth` reserves. Rounded to 4 KiB so a region keeps the alignment a
// head's loads were fitted against, and so the stride is a round number in a dump.
constexpr size_t gdn_fmt_defer_room(int fmt) {
    return (gdn_fmt_defer_off(fmt) + (size_t) HV * gdn_fmt_defer_head(fmt) + 1023) / 1024 * 1024;
}
// Floats between one (slot, layer) region and the next, for a process that does not defer.
constexpr size_t gdn_fmt_region_floats(int fmt) {
    return gdn_fmt_is_exact(fmt) ? (size_t) GDN_STATE_FLOATS : gdn_fmt_defer_room(fmt);
}
static_assert(gdn_fmt_region_floats(GDN_STATE_I8) * 4 < 1200 * 1024, "an int8 region is about 1.15 MiB");
static_assert(gdn_fmt_region_floats(GDN_STATE_I16) <= (size_t) GDN_STATE_FLOATS, "a packed region fits an fp32 one");
static_assert(gdn_fmt_region_floats(GDN_STATE_I8) <= gdn_fmt_region_floats(GDN_STATE_I16), "eight bits fits sixteen");
constexpr int BLK_TOKEN_FLOATS = 2 * CONV_CH + 2 * HV;   // per cached token: pre-conv, conv output, alpha, beta
constexpr int NCAP     = 5;    // captured target layers for DFlash2
// The widest batched pass `forward_batch` admits, and the capacity `prepare_batch` accepts. Raise
// it in one place: every weight-bearing kernel re-streams its whole image once per pass, so a wider
// pass amortizes the fixed half of a pass over more rows.
constexpr int PASSMAX  = 256;  // rows in one batched pass
// What a wide prompt pass takes unless a caller asks for another width. It is a policy, not a
// limit: `--pass-rows` and the measurement tools choose their own, and the ceiling above is what
// the buffers and kernel guards admit.
constexpr int PASS_ROWS_DEFAULT = 256;
// Rows of captured drafter features the engine holds. A drafted prompt used to ingest one
// eight-row pass at a time, so RMAX was enough; a wide prompt pass captures every row of the batch
// and hands the drafter RMAX rows at a time out of the same buffer. It tracks the pass width rather
// than a literal, so widening a pass cannot silently drop the drafted path back to eight rows:
// `hcap` is 13.1 MB at 128 rows and `hfinal` 2.6 MB, against the engine's 6.6 GB of weights.
constexpr int CAPMAX   = PASSMAX;
// The five target layers DFlash2 reads. Slot j is the residual stream *entering* layer
// CAP_LAYERS[j] + 1, which is the output of layer CAP_LAYERS[j]: the eight-row prep captures it
// inside the persistent kernel and a wide pass copies it out of the batch residual buffer, so both
// routes must name the same boundary.
constexpr int CAP_LAYERS[NCAP] = { 5, 19, 33, 47, 61 };
// DFlash2 drafter geometry (Qwen3 style, 5 layers)
constexpr int DF_LAYERS = 5, DF_NH = 32, DF_NKV = 8, DF_HD = 128, DF_Q = DF_NH * DF_HD, DF_KV = DF_NKV * DF_HD;
constexpr int DF_CONV_GROUPS = D / 16, DF_KPROJ = 2 * 2 * DF_CONV_GROUPS; // 1280
constexpr int DF_RANK = 256, DF_TOPK = 16, DF_BLOCK = 8, DF_WINDOW = 2048;

constexpr int NB_D  = D / BLOCK;   // 40
constexpr int NB_V  = VDIM / BLOCK; // 48
constexpr int NB_FF = FF / BLOCK;  // 136

constexpr float NORM_EPS = 1e-6f;

struct AttnPartial { float m; float l; float pad[2]; float acc[HD]; };

struct MvSeg { const uint8_t * w; float * out; int ntiles; int add; };
struct MvArgs { MvSeg seg[4]; int nseg; int total_tiles; const int8_t * xq; const float * xs; const int * xsum; };

enum PrepFlags {
    PREP_NORM = 1, PREP_SILU_MUL = 2, PREP_PERMUTE_GDN = 4, PREP_SIGN = 8, PREP_HADAMARD = 16,
    PREP_QUANT = 32, PREP_STORE_F32 = 64, PREP_SIGN_AFTER = 128,
    PREP_GDN_NORM = 256,     // x = raw GDN head outputs o[6144]; apply per-head RMSnorm * norm_w[128] * silu(x2 = z); implies PERMUTE_GDN
    PREP_ATTN_COMBINE = 512, // x unused; combine attention partials for the 4 heads of this chunk and apply sigmoid(gate) from x2 = q_full
    PREP_FOLDED = 1024,      // fused kernel: `kvec` (n floats) already holds norm weight * sign; replaces norm_w/signs loads
};
struct PrepArgs {
    const float * x; const float * x2; const float * norm_w; const float * signs;
    float eps; int n; int flags;
    int8_t * xq; float * xs; int * xsum; float * out_f32; float * out_norm;
    const AttnPartial * partials; const int * pos;
    const float * kvec;
};

// Per-layer device weight pointers, shared by the host loader and the persistent forward kernel.
struct LayerW {
    int recurrent;
    int kv_slot;
    const uint8_t * qkv, * z, * ssm_out;
    const uint8_t * q, * k, * v, * o;
    const uint8_t * gate, * up, * down;
    const float * attn_norm, * post_norm, * q_norm, * k_norm;
    const float * conv_w, * ssm_a, * ssm_dt, * ssm_norm;
    const float * attn_norm_s, * post_norm_s, * ssm_norm_s; // norm weight * sign vector (ssm: per-head weight expanded over 6144)
    const unsigned short * alpha_beta;
};

struct RowInfo { int token; int seq; int pos; int pad; };
// Per active sequence in a pass: rows [row0, row0 + nrows) belong to it, in position order.
// n_replay cached tokens of the previous block (parity ^ 1) are committed into the GDN state first.
struct SeqCtl { int row0; int nrows; int n_replay; int parity; };

// Q8 tile: 32 rows x 128 int8 per block, then 32 fp16 scales. Same tile/row mapping as HALO.
constexpr int Q8_TILE_BLOCK_BYTES = TILE_ROWS * 128 + TILE_ROWS * 2; // 4160

struct FwdParams {
    const LayerW * layers;
    const uint8_t * tok_embd, * output; const float * output_norm;
    const float * signs5120, * signs6144, * signs17408, * output_norm_s;
    float * x, * xn, * tmp; int8_t * xq; float * xs; int * xsum;
    float * big, * zbuf, * ab, * o, * gu, * qfull, * kbuf, * vbuf, * logits;
    AttnPartial * partials;
    float * gdn_state, * conv_ring; // [slot][48][...] and [slot][48][3][CONV_CH]
    __half * kcache, * vcache;
    int k_tiled;                    // 1 = the leaf-interleaved K coordinate (k_leaf_off)
    const int * tok_ring; int * pos; int * tok_out;
    unsigned * bar; unsigned * work; float * amax_val; int * amax_idx;
    unsigned long long * prof; // null or 1024 stamps
    int with_logits;
    int debug_layers; // 0 = all
    int debug_stop;   // return after this many device barriers (0 = run to completion)
    // ---- multi-row state (v3 kernel) ----
    RowInfo rows[RMAX]; int nrows;
    SeqCtl seqs[MAXSEQ]; int nseq;
    float * blk_cache;   // [MAXSEQ][2][48][RMAX][BLK_TOKEN_FLOATS]
    float * qrot;        // [RMAX][ATTN_OUT] rotated, normed q
    float * hcap;        // [RMAX][NCAP][D] captured layer outputs (may be null)
    float * hfinal;      // [RMAX][D] final normed hidden (may be null)
    int * argmax_out;    // [RMAX]
    float * ninv;        // [RMAX] RMS norm scalar of the current attn_norm prep
    int attn_key_min;    // first valid key position (1 for the MTP head, whose position 0 never exists)
    int attn_noncausal;  // every row sees every key up to the last row's position (DFlash block)
    int attn_window;     // 0 = none; otherwise keys with query_pos - key_pos < window
    // Whether the window decides the unit list as well as the mask. 1 is what the drafter ships
    // with: a chunk every key of which the window excludes gets no unit, so it costs no key read,
    // no dot product and no partial. 0 restores the schedule that dealt every chunk from position 0
    // and masked the result to zero - the in-process control arm, and the only reason this is a
    // field rather than a constant. It moves no output bit either way: the chunks it drops carry
    // `m = -inf, l = 0, acc = 0` and contribute `exp(-inf - M) = 0` to the fold. Ignored wherever
    // `attn_window` is 0, which is every target-model pass. HALO_ATTN_DEAL, engine.cpp.
    int attn_window_deal = 1;
    // A score unit is written for TT rows and a row group often carries fewer. Every generation
    // step in this engine is made of one-row groups, and the TT loop clamps its row index with
    // min(i, nrows-1), so those units recompute row 0 up to TT times and discard the copies.
    // 1 dispatches the unit to the narrowest instantiation that covers the group, which runs the
    // surviving row's code with the loop taken once: same q, same key order, same fmac chain, same
    // warp_sum, same block reductions, same partial. 0 is the wide-only path this replaces and is
    // kept as the in-process control. kernels/phases.hpp:attn_chunk_group.
    int attn_narrow = 1;
    // Query heads of one KV group per score unit, for a launch whose groups all carry one row.
    // A key chunk costs the same bytes whoever reads it; this is how many dot products get charged
    // to those bytes. A prompt pass amortises them over eight neighbouring rows, a generation step
    // has one row per sequence and can only amortise over heads. 1 is the deployed unit; the only
    // other instantiation is the whole KV group (GQA = 6). Ignored wherever a group is wider than
    // one row, because eight rows by six heads does not fit the query registers.
    //
    // The default is the whole KV group. It is bit-identical, it cannot fire on any shape with a
    // multi-row group - so prefill and the drafted verify pass never see it - and on the shape it
    // does serve it is worth 2.6x of the phase and 17% of the step at a 960-token prefix.
    // HALO_ATTN_HG=1 is the control.
    int attn_hg = NH / NKV;
    int context = MAXCTX; // allocated KV positions per head, shared by target and drafters
    // Single-token (one row) execution map. 0 keeps the deployed path. The optimised maps are
    // cumulative and live in the same executable, so one process can interleave them:
    //   1  palette operand map for the HALO matvecs (integer identical, bit-exact logits)
    //   2  1 + sixteenth-of-a-head GDN state units (bit-exact)
    //   3  2 + K-split retiling of the 40-block matvecs (float reassociation, not bit-exact)
    //   4  the deployed map, launched on its own occupancy grid instead of the shared minimum
    // Ignored unless nrows == 1; every other row count and the Q8 drafters keep the old path.
    int single_map = 0;
    // Workgroups for the persistent row kernel, 0 = the occupancy grid. Capped at that grid, and a
    // pure scheduling choice: it moves whole units between workgroups and changes no output bit.
    // HALO_GRID_ROWS and Engine::set_grid_rows select it inside one process.
    int grid_rows = 0;
    // Register budget asked of the persistent kernel. -1 takes the measured rule (the packed-state
    // kernels ask, the fp32 ones do not), 0 lets the allocator choose, 1 asks for a 120-register
    // allocation. Both budgets are compiled into the same executable as separate instantiations, so
    // one process can interleave them; HALO_PK_OCC and Engine::set_pk_occ select one. It is a pure
    // allocation choice: same arithmetic, same operand order, same units, so the output bits do not
    // move, and only the workgroup count the device will hold does. See docs/rows-grid-occupancy.md.
    int pk_occ = -1;
    // Direct-commit prefill. 0 is the deployed speculative contract: the pass leaves the committed
    // GDN state and conv ring where its predecessor's accepted prefix left them, and the next pass
    // replays SeqCtl::n_replay cached tokens into them before its own rows.
    // 1 commits this pass instead: ph_gdn_pre stores the conv ring after its last row and ph_gdn
    // stores the GDN state after its last token, for every sequence in the pass. The arithmetic,
    // the operand order and every output are those of the replay path with all rows accepted; only
    // the point at which state is persisted moves, so the next pass must arrive with n_replay == 0.
    // Committing a pass and then replaying it applies its tokens twice. Incoming n_replay is still
    // honoured, so a pass may commit a prior uncommitted block and its own rows together, and rows
    // are still cached in blk_cache, so a later uncommitted pass is replayed in the usual way.
    // Selects the deployed arithmetic map; single_map is ignored while it is set.
    int commit_state = 0;
    // ---- direct wide operand (no capture kernel) ----
    // Set these and the PART 1 / PART 2 quantiser writes its int8 codes straight into the column
    // layout the wide WMMA projection reads, instead of the row-major P.xq/P.xs/P.xsum triple.
    // The arithmetic is the deployed quantiser's: same norm, same Hadamard, same amax over the
    // same 128 lanes, same round-to-nearest-even codes. Only the destination moves.
    //
    //   sequence_q       operand base, 16B-aligned, same buffer the projection takes as B.
    //                    A lane holding elements [base, base + 4) of row `row` writes one word at
    //                    ((base / 16) * sequence_npad + sequence_offset + row) * 4 + (base % 16) / 4,
    //                    so 16 consecutive elements of a row form one WMMA k-group and the groups
    //                    of one token column sit sequence_npad apart.
    //   sequence_scales  amax/127 per (row, 128-block) at
    //                    [(sequence_offset + row) * (n / 128) + base / 128]; n is the projection's
    //                    K (D for part 1, VDIM/ATTN_OUT for part 2), so the row stride is NB_D
    //                    for part 1 and NB_V for part 2.
    //   sequence_npad    token columns in the operand, the batch total rounded up to 16.
    //   sequence_offset  column of this pass's row 0 within the batch.
    //
    // Nothing row-major is written on this route and no xsum is produced: the wide projection is
    // signed x signed with no zero-point term, so the block sum has no consumer. P.xq/P.xs/P.xsum
    // are left untouched by that prep. Leave sequence_q null and both parts behave exactly as they
    // do today, including the staged capture/restore route.
    unsigned * sequence_q = nullptr;
    float * sequence_scales = nullptr;
    int sequence_npad = 0;
    int sequence_offset = 0;
};
constexpr int SINGLE_MAP_MAX = 4;

// A committed batch can carry arbitrary sequence lengths rather than eight-row slices.
struct ResidentSeq { int row0, nrows, n_replay, parity, slot; };
struct GdnResidentParams {
    FwdParams p;
    ResidentSeq seqs[128];
    int nseq=0, nrows=0;
    float *tokens=nullptr; // [row][CONV_CH + 2*HV], no raw-preactivation copy
};
void launch_gdn_resident(const GdnResidentParams &,int layer,int split,hipStream_t);
// How wide a state unit is: a unit owns 128/SPLIT rows of one (sequence, head) and each of its
// eight waves owns 16/SPLIT of them. Every row belongs to exactly one unit and nothing is reduced
// across units or waves, so this is a work decomposition and carries no numerical consequence.
// docs/gdn-unit-width.md holds the ladder and says which end of it the two deployed shapes want.
constexpr bool gdn_split_valid(int s){return s==2||s==4||s==8||s==16;}   // 1 measured a loss, see the doc
// Selects where the recurrent decay gate is evaluated: 1 once per (row, head) in the resident
// convolution kernel, 0 per token inside every state unit. Both arms compute one expression, so
// this is a placement choice, not a numerical one. See docs/resident-gdn.md.
void gdn_resident_set_pregate(int on);   // 0 in the token loop, 1 once per row and head, -1 by shape
int gdn_resident_pregate();
int gdn_resident_gate_for(int nrows, int nseq);   // the gate placement this shape selects
// Whether this build's block token cache holds evaluated decay gates rather than the raw alpha and
// beta projections. Compile-time (HALO_GDN_BLK_GATE): every producer and reader of that cache has
// to agree, and the sliced state phase is one of them. docs/gdn-sliced-gates.md.
bool gdn_blk_pregate();
void gdn_resident_set_split(int s);      // 1, 2, 4, 8 or 16 row groups per head; -1 as configured
int gdn_resident_split();
int gdn_resident_split_for(int nseq);    // the default width for a grid of nseq sequences
// Selects the state unit's lane ownership: 0 the deployed rows-per-lane layout, 1 one lane per
// 32 columns of one row, -1 by shape. Both arms run the same products in the same summation trees
// and produce the same bits; the column arm needs fp32 state and a committed pass, which
// gdn_resident_cols_available() reports. See docs/gdn-lane-layout.md.
void gdn_resident_set_cols(int on);
int gdn_resident_cols();
bool gdn_resident_cols_available();

// Selects the state's storage coordinate for every route that owns GDN state. The format is a
// property of what is already in `FwdParams::gdn_state`, so changing it invalidates every live
// sequence: the setter synchronises the device and callers must reset or re-prefill their slots.
// Process default from HALO_GDN_STATE (f32, f16, i16, i8, or 0..3).
void gdn_state_set_format(int fmt);
int gdn_state_format();
const char * gdn_state_format_name(int fmt);

// Floats between one (slot, layer) state region and the next, for this process. Every allocation,
// memset, slot copy and kernel has to agree on it, so it is decided once: the first call freezes it
// at the widest format the process has reserved, uploads it to the kernels and returns it. The
// engine calls it when it allocates `gdn_state`, which is before any launch can read a region.
size_t gdn_region_floats();
// Widen that stride before it is frozen. A process that switches coordinates - the measurement
// tools sweep `--gdn-state` inside one engine - has to say so up front, because a region's start is
// an address and the slots already hold state at the old one. Reserving is free for a format the
// process never selects; it only costs the memory of the widest one.
void gdn_state_reserve_format(int fmt);
// The coordinate the process starts in, chosen before the engine allocates. The stride is frozen
// from the format the process is in plus whatever it reserved, so a panel whose whole case list is
// packed says so here and gets packed-sized regions; without it the fp32 default reserves three
// times the memory for a coordinate the panel never selects. After the freeze this is the ordinary
// setter and obeys its contract.
void gdn_state_boot_format(int fmt);
// Whether this process allocates the compact regions (HALO_GDN_REGION=packed) or the fp32-sized
// ones every coordinate has had until now. Recorded by the measurement tools, because it decides
// how much memory a slot costs and which acceptance applies.
int gdn_region_compact();

// How many generation steps of a sequence share one state write-back. 1 commits every step. Above
// that the resident route keeps each step's rank-1 update beside the state and rebuilds the sum at
// load. A packed coordinate has room for the pending list in the region it already leaves unused;
// an exact one has to reserve it, which grows a (slot, layer) region by about 13% and is why the
// reservation is asked for rather than assumed - `gdn_defer_effective()` is the depth the process
// is actually running at. Pending updates belong to the sequences that own them, so changing the
// depth has a format switch's contract: the setter synchronises and the caller resets or
// re-prefills. Process default from HALO_GDN_DEFER.
void gdn_defer_set_depth(int depth);
// Says a depth this process may ask for later, before the engine allocates, the way
// `gdn_state_reserve_format` says a coordinate. After the regions exist a depth that does not fit
// in them aborts rather than quietly commits every step: a told depth is a contract.
void gdn_defer_reserve_depth(int depth);
int gdn_defer_depth();
int gdn_defer_effective();
// Which slots may hold pending updates is host state, because the host is what decides whether a
// pass on a route that cannot rebuild them has to commit first. A reset empties a slot's list and a
// slot copy moves it; every other transition is handled inside the launchers.
void gdn_defer_note_reset(int slot);
void gdn_defer_note_copy(int dst, int src);
// Commits a slot's pending write-back for a route that reads a state region directly instead of
// through the rebuild. A no-op when the slot has nothing pending, which is every non-deferring
// process, so a caller may ask unconditionally.
void gdn_defer_flush_pending(float * gdn_state, int slot, hipStream_t st);
// Selects the key cache's storage coordinate, and with it the score unit's lane placement. Like
// the state format it describes what is already in `kcache`, so changing it invalidates every
// live sequence and the caller must re-prefill. Both arms compute the same scores bit for bit;
// 1 is the leaf-interleaved coordinate and 0 is the row-major deployed one. Default 0 from
// HALO_ATTN_COORD: the score arm is measured and exact, but a second body inlined into
// k_attn_wide costs that kernel 45 VGPRs and six wave slots, so it ships selectable rather than
// selected until the arm dispatch moves up to the launcher. docs/attn-score-lane.md has both.
void attn_coord_set(int tiled);
int attn_coord();

void launch_forward_rows(const FwdParams & p, hipStream_t st);
// Stage -1 embeds, 0..63 executes that layer's attention/GDN, 64 computes logits.
void launch_forward_slice(const FwdParams & p, int stage, hipStream_t st);
void launch_ffn_slice(const FwdParams & p, int layer, hipStream_t st);
// Which register budget the FFN slice's matrix bodies run under: 0 is the shipped ask, 1 asks for
// eight waves per SIMD32, which is a fourth workgroup per WGP. -1 restores the fitted rule, which
// is 0 because arm 1 was measured and lost (docs/ffn-slice-grid.md). Only a build with
// -DHALO_SLICE_OCC_CONTROL=1 carries both arms; elsewhere this reports the rule and selects
// nothing, so a measurement tool can walk the axis against any build and say what it got.
void ffn_slice_set_occ(int arm);
int ffn_slice_occ(void);

// Which operand the deployed FFN slice's block loop requests first, for the widths that carry more
// than one column group: 0 requests each group's operands inside the loop that consumes them, 1
// requests the first operand of every group after the first at the top of the block. Both arms are
// in every build and compute the same bits; `HALO_MVW_ORDER` gives the same choice to a process
// that cannot call the setter. docs/ffn-slice-rows.md and docs/ffn-slice-issue-order.md.
void ffn_slice_set_order(int ord);
int ffn_slice_order(void);

// Which unit shape the deployed FFN slice runs. 0 is the predecessor, which left the unit's whole
// drain on wave 0 in 2 * GROUPS sequential passes; 1 spreads it over the eight waves that computed
// it and is bit-identical by construction. 2..4 are the activation/weight FOOTPRINT ABLATIONS that
// measure what each stream's delivery costs; they compute wrong numbers by design, only a
// `-DHALO_MVW_ABL=1` build carries them, and `ffn_slice_arm()` reports what a process actually got.
// `HALO_MVW_ARM` gives the same choice to a process that cannot call the setter.
// docs/ffn-slice-activation.md.
void ffn_slice_set_arm(int arm);
int ffn_slice_arm(void);

// One layer of the sequence path, split so a wider projection can own its matvecs. Both parts run
// the same phase code as the whole-layer kernel, on their own occupancy grid, for layer in
// [0, NLAYER) and 1..RMAX rows. Each is a separate cooperative launch, so P.bar (4 bytes) and
// P.work (4096 ints) must be zeroed before each one. P.single_map is ignored.
//
// SEQ_PART_PREP consumes P.x and fills P.xq/P.xs/P.xsum for the layer's input projection; on a
// recurrent layer it also fills P.ab (stride 2*HV per row) and P.ninv (one scalar per row). It
// returns after that phase's barrier and zeroes none of the projection outputs.
//
// Both parts take the direct wide operand instead when P.sequence_q is set: the quantiser writes
// the WMMA column layout itself and leaves P.xq/P.xs/P.xsum alone (see FwdParams). P.ab and P.ninv
// keep their row-major strides, so a per-batch metadata buffer is bound by offsetting the pointers
// (ab + offset * 2 * HV, ninv + offset).
//
// SEQ_PART_STATE assumes the projection outputs have been restored into the buffers the deployed
// path uses, together with P.ab and P.ninv: recurrent layers read P.big (stride QKV_OUT) and
// P.zbuf (stride VDIM); attention layers read P.qfull (stride Q_OUT, q and gate interleaved per
// head), P.kbuf and P.vbuf (stride KV_OUT). It runs the conv ring, the GDN state and the GDN-norm
// output prep, or rope/KV write, attention and the combine output prep, and returns after the
// output prep's barrier with P.xq/P.xs/P.xsum ready for the ssm_out/o projection. That projection
// accumulates into the residual P.x, which this part does not touch. P.commit_state applies here.
//
// With P.prof set, part 1 writes prof[0..1] (one interval) and part 2 prof[0..3] (three intervals:
// conv ring | state | output prep, or rope/KV | attention | combine).
//
// SEQ_PART_HEAD is the same idea after the last layer: it runs the final norm, Hadamard and
// quantiser of the output head and returns, leaving the wide operand ready for a vocabulary
// projection that owns every row of the batch (kernels/head_batch.h). It takes layer == NLAYER,
// reads P.x and writes nothing else. Like the other parts it honours the direct wide operand.
constexpr int SEQ_PART_PREP = 1, SEQ_PART_STATE = 2, SEQ_PART_HEAD = 3;
void launch_sequence_part(const FwdParams & p, int layer, int part, hipStream_t st);

// ---- MTP head (Qwen3.8 NextN block, Q8 weights) ----
struct MtpW {
    const uint8_t * fc, * q, * k, * v, * o, * gate, * up, * down;   // Q8 tiles
    const float * in_norm, * post_norm, * q_norm, * k_norm, * enorm, * hnorm, * out_norm, * out_norm_s; // f32 (out_norm_s = out_norm * signs)
};
struct MtpParams {
    FwdParams P;            // activation buffers, rows/seqs, KV cache (slot NATTN), lm head, barrier state
    MtpW w;
    const float * h_in;     // [row][D] hidden inputs (stride D)
    float * h_out;          // [RMAX][D] normed block output (next chain input)
};
void launch_mtp(const MtpParams & p, hipStream_t st);

// ---- DFlash2 block drafter (Q8 or Q4 weight tiles; DflashParams::q4 says which) ----
constexpr int DF_TOPCAP = 1024; // candidate list capacity per row for the top-k threshold pass
struct DflashLayerW {
    const uint8_t * q, * k, * v, * o, * gate, * up, * down, * akp, * mkp; // weight tiles; akp/mkp: conv kernel projections [1280][D]
    const float * in_norm, * post_norm, * q_norm, * k_norm;
    const float * abase, * mbase;                                          // conv base kernels [2 (prepare/finish)][2 taps][D]
};
struct DflashW {
    const uint8_t * fc;                        // Q8 [D][NCAP*D]
    const float * hidden_norm, * norm, * norm_s; // norm_s = norm * signs (LM head input)
    const uint8_t * hproj;                     // Q8 [DF_RANK][D]
    const unsigned short * pred_cb, * succ_cb; // bf16 [VOCAB][DF_RANK]
    DflashLayerW layer[DF_LAYERS];
};
struct DflashParams {
    FwdParams P;               // block rows (DF_BLOCK) in P.rows / P.seqs[0]; activation buffers; KV cache; barrier state
    DflashW w;
    RowInfo ctx_rows[RMAX]; int n_ctx; // target rows whose captured features enter the drafter's KV cache
    const float * hcap;        // [n_ctx][NCAP * D]
    float * th;                // [RMAX][D] target features after fc + hidden_norm
    float * xn;                // [DF_BLOCK][D] normed block hidden
    float * dyn;               // [DF_BLOCK][DF_KPROJ] dynamic conv kernels
    float * cbuf;              // [DF_BLOCK][D] conv output
    float * abuf;              // [DF_BLOCK][DF_Q] attention output
    float * tmpo;              // [DF_BLOCK][D] o_proj / down_proj output before the finish conv
    float * hproj_out;         // [DF_BLOCK][DF_RANK]
    float * cand_v; int * cand_i; int * cand_n; float * thr; // top-k scratch: [DF_BLOCK][DF_TOPCAP], counters, thresholds
    float * topv; int * topi;  // [DF_BLOCK][DF_TOPK]
    int * draft_out;           // [DF_BLOCK - 1]
    int anchor;                // block row 0 token
    int do_block;              // 0 = ingest only
    int q4;                    // 1 = the weight pointers in `w` are Q4 tiles (kernels/q4_format.h)
};
void launch_dflash(const DflashParams & p, hipStream_t st);
// Workgroups for the drafter's cooperative kernel, as an in-process arm. 0 takes the build's own
// occupancy grid and a positive value pins that many, capped there. A unit is whole work moved
// between workgroups, so nothing about a drafted run's output can change; docs/drafter-occupancy.md
// walks the curve with it. HALO_DRAFT_GRID sets it for a process that cannot call the setter.
void set_dflash_grid(int g);
void set_layer_table_rows(const LayerW * host_layers);

void set_layer_table(const LayerW * host_layers);
void set_forward_flags(int no_prefetch);
int  forward_grid_size();
void launch_forward(const FwdParams & p, hipStream_t st);

struct GdnArgs {
    const float * conv_out; const float * alpha; const float * beta; const float * z;
    const float * ssm_a; const float * dt_bias; const float * ssm_norm;
    float * state; float * y; float eps;
};

struct AttnPreArgs {
    const float * q_full; const float * k; const float * v; const float * q_norm; const float * k_norm;
    float eps; const int * pos; __half * kcache; __half * vcache; float * q_out;
    int context = MAXCTX;
    int k_tiled = 1;
};


struct AttnArgs {
    const float * q; const float * q_full; const __half * kcache; const __half * vcache; const int * pos;
    AttnPartial * partials; float * y;
    int context = MAXCTX;
    int k_tiled = 1;
};

void launch_matvec(const MvArgs & a, int K, hipStream_t st);
void launch_prep(const PrepArgs & a, hipStream_t st);
void launch_matvec_bf16(const unsigned short * w, const float * x, float * out, int rows, int K, hipStream_t st);
void launch_gdn_conv(const float * x, const float * w, float * state, float * out, hipStream_t st);
void launch_gdn(const GdnArgs & a, hipStream_t st);
void launch_attn_pre(const AttnPreArgs & a, hipStream_t st);
void launch_attn(const AttnArgs & a, hipStream_t st);
void launch_embed_row(const uint8_t * w, const int * tok_ring, const int * pos, float * out, hipStream_t st);
void launch_argmax(const float * logits, int n, int * out, int * out2, hipStream_t st);
void launch_set_int(int * p, int v, hipStream_t st);
void launch_inc_int(int * p, hipStream_t st);

} // namespace halo

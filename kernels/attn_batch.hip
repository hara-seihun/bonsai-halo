// Wide attention launches. See attn_batch.h for what this replaces and why.
//
// Every kernel here is the persistent kernel's own phase body, instantiated for an ordinary launch
// whose grid is the unit count. `ph_attn_pre`, `attn_chunk_unit` and `ph_prep` are shared with
// halo_rows.hip through phases.hpp, so this module cannot drift from the deployed arithmetic: the
// only thing that differs is which workgroup runs which unit, and how many rows one launch covers.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include "halo_kernels.h"
#include "device.hpp"
#include "phases.hpp"
#include "attn_batch.h"
#include "prep_batch.h"

namespace halo {

namespace {

#define ATTN_CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) throw std::runtime_error(std::string("attn_batch: ") + hipGetErrorString(e_)); } while (0)

int env_int(const char * name, int dflt) {
    const char * v = getenv(name);
    if (!v || !*v) return dflt;
    return atoi(v);
}

// One workgroup, one unit, no persistent work queue: c.bar/c.work stay null because neither
// grid_sync nor next_unit runs in these instantiations.
#define ATTN_CTX(name) \
    __shared__ __attribute__((aligned(16))) float lds_[LDS_FLOATS]; \
    __shared__ float red_[8]; \
    __shared__ int unit_; \
    Ctx name { nullptr, nullptr, nullptr, 0, 0, true, lds_, red_, &unit_ }

__global__ void __launch_bounds__(NT) k_attn_pre_wide(FwdParams P, const float * __restrict__ q_norm,
        const float * __restrict__ k_norm, int kv_slot, const RowInfo * __restrict__ rows, int nrows) {
    ATTN_CTX(c);
    ph_attn_pre<true>(c, P, q_norm, k_norm, kv_slot, rows, nrows);
}

// grid.x = NH * maxchunks, grid.y = row groups in this launch.
// HG folds a KV group's query heads into one unit. It is a launch-level choice because it sets the
// unit count, and the host takes it only when every group in the pass carries one row - the
// generation shape, where there are no neighbouring rows to share a key chunk with and the heads
// are the only reuse axis left. HG = 1 is the deployed instantiation.
//
// KM says which score body this kernel compiles. It is a launch-level choice for the same reason
// the register file is: a runtime branch inside the unit makes the allocator hold the union of
// both bodies for every wave, whether or not that wave takes it.
template <int TT, int SCHED, int HG = 1, int KM = ATTN_KM_ROW, int HB = 1, int PROBE = 0>
__global__ void __launch_bounds__(NT) k_attn_wide(FwdParams P, int kv_slot,
        const RowInfo * __restrict__ rows, const SeqCtl * __restrict__ groups,
        int g0, int rowbase, AttnPartial * part, int part_stride) {
    ATTN_CTX(c);
    constexpr int HU = NH / HG;
    const SeqCtl S = groups[g0 + blockIdx.y];
    const int hq = (int) (blockIdx.x % HU) * HG, chunk = blockIdx.x / HU;
    // A group whose last row sits before this chunk has nothing here. Its partials are never read:
    // the combine walks chunk 0 .. pos/ACHUNK for each row.
    if (chunk * ACHUNK > rows[S.row0 + S.nrows - 1].pos) return;
    // The row-major fold is a one-row unit by construction - its query is per lane, so eight rows
    // by six heads is 384 VGPRs. The leaf fold carries the group it was given, because its query is
    // wave-uniform; see attn_chunk_unit_leaf.
    if constexpr (HG > 1 && KM != ATTN_KM_LEAF) attn_chunk_unit<1, NH, NKV, HD, SCHED, HG>(c, P, rows, S, hq, chunk, kv_slot, part, part_stride, rowbase);
    else attn_chunk_group<TT, NH, NKV, HD, SCHED, true, KM, HG, HB, PROBE>(c, P, rows, S, hq, chunk, kv_slot, part, part_stride, rowbase);
}

// The combine is the quantiser of the sequence OUTPUT projection for the 16 attention layers: it
// writes the wide operand `project<part 2>` reads. QL and NIB carry the output stage's activation
// coordinate and reach `prep_chunk_r` through `ph_prep`; <127,false> is the deployed instantiation
// and compiles to the code it always did. docs/sequence-output-a4.md.
template <int QL, bool NIB>
__global__ void __launch_bounds__(NT) k_attn_combine_wide(FwdParams P, const RowInfo * __restrict__ rows,
        const AttnPartial * part, int part_stride, int nrows) {
    ATTN_CTX(c);
    PrepR p{};
    p.x2 = P.qfull; p.partials = part; p.rows = rows; p.kvec = P.signs6144; p.n = ATTN_OUT;
    p.flags = PREP_ATTN_COMBINE | PREP_HADAMARD | PREP_QUANT;
    p.xq = P.xq; p.xs = P.xs; p.xsum = P.xsum;
    p.part_stride = part_stride;
    bind_seq_dest<true>(p, P);
    ph_prep<true, true, QL, NIB>(c, p, nrows);
}

} // namespace

struct AttnBatch {
    int capacity = 0;
    int group_rows = 8;
    int sched = 1;
    int fold = 1;
    int part_window_mb = 0;
    int pass_rows = 0, pass_groups = 0;   // the shape attn_batch_begin was handed, for the window
    bool enabled = true;
    float * qrot = nullptr;            // [capacity][ATTN_OUT]
    RowInfo * rows = nullptr;          // [capacity]
    SeqCtl * groups = nullptr;         // [capacity]
    AttnPartial * part = nullptr;      // [part_rows][NH][stride], grown to the budget
    size_t part_bytes = 0, budget = 0;
    size_t bytes = 0;
};

int attn_group_rows() {
    const int tt = env_int("HALO_ATTN_TT", 8);
    return tt >= 16 ? 16 : 8;
}

// The score unit's schedule: 0 is the deployed body kept as the exactness control, 1 the
// rescheduled one. Both compute the same floats, so this is a timing axis and never a route
// choice. See docs/attn-score-schedule.md.
//
// 2 is not a schedule. It runs the rescheduled body through the FUSED kernel - the one that also
// compiles the lane-per-key body behind a runtime branch on the coordinate - so a panel can price
// what that delivery costs against 1, in one process, on one clock, with everything else equal.
// It is the control for docs/attn-score-lane.md's reversal and nothing selects it by default.
int attn_sched_default() { const int v = env_int("HALO_ATTN_SCHED", 1); return v < 0 ? 0 : (v > 2 ? 2 : v); }

// How many of a KV group's query heads one score unit carries, and how its leaf walk is blocked.
// The arm is a single code because it is one axis for a measurement to walk:
//
//   1  no fold, one head per unit - the deployed arm in either coordinate
//   2  HG = 2, HB = 2: two heads per unit, one leaf walk, HB * RB tree partials
//   3  HG = 2, HB = 1: two heads per unit, a leaf walk each, RB tree partials and the second walk
//                      reading leaves the same wave read a few hundred cycles earlier
//   6  HG = 6, HB = 6: the whole KV group, which only the one-row generation shape has LDS for
//
// Only the leaf coordinate can take it: a folded head costs qr[TT][DPL] in the row-major body and
// one score register here. docs/attn-reuse-fold.md.
int attn_fold_default() {
    const int v = env_int("HALO_ATTN_FOLD", 0);
#ifdef HALO_ATTN_PROBE
    // 8, 9 and 10 are the ablation, available only in a probe build. They compute the wrong
    // attention on purpose: the unit issues its usual instructions against a pinned chunk, so the
    // panel reads what the phase costs when its keys, its values or both are already resident.
    if (v >= 8 && v <= 10) return v;
#endif
    return (v == 0 || v == 2 || v == 3 || v == 6) ? v : 1;
}

// The chunk partials are the half of this phase nobody has priced. A score unit writes 1040 bytes
// per (row, head, chunk) and the combine reads every one of them back, so a 128-row pass at
// position 3072 turns 3.1 MB of attention output into 79 MB of partials PER LAYER - written once,
// read once, and far larger than the last-level cache that would otherwise have carried them.
//
// The driver already walks row groups in score/combine pairs, so the working set between a write
// and its read is `groups per launch` wide and nothing else has to change to shrink it.
//
// 16 MB is the default because it is what the device gives back: the same bytes, the same combine
// expression and the same grid per row read at 154 GB/s when they were written 79.9 MB ago and at
// 365 GB/s when they were written 16 MB ago. It is bit-identical (residual FNV 16232333786579957879
// over 95,354,880 logits at 128 and 256 rows in both arms), it is exactly inert on any pass whose
// partials already fit - short prompts and every decode shape launch the identical geometry - and
// it is worth -57.6% of the combine and +0.75% of whole-document prompt throughput at 3072 tokens,
// growing with context. 0 restores the one-pair-per-pass geometry as the in-process control.
// docs/attn-partial-window.md.
int attn_part_window_default() { const int v = env_int("HALO_ATTN_PART_WINDOW_MB", 16); return v < 0 ? 0 : v; }

// The floor the window is not allowed to cross, in row groups per launch. A pair buys residency and
// pays one launch tail, and a score launch that carries too few groups has no grid left to amortise
// that tail against: at ONE group the score kernel is +45.8% and the combine loses two thirds of its
// own gain because its grid collapses too (48 workgroups). Four groups keeps the score grid at
// 4 x NH x chunks and caps a pass at four pairs however long the context gets, which is what stops
// a fixed megabyte window from splitting further and further as the partials grow with position.
int attn_part_min_groups() { const int v = env_int("HALO_ATTN_PART_MIN_GROUPS", 4); return v < 1 ? 1 : v; }

bool attn_trace_split() { return env_int("HALO_ATTN_TRACE_SPLIT", 0) != 0; }

size_t attn_batch_required_bytes(int capacity) {
    if (capacity <= RMAX || capacity > PASSMAX || env_int("HALO_WIDE_ATTN", 1) == 0) return 0;
    return (size_t) capacity * (ATTN_OUT * sizeof(float) + sizeof(RowInfo) + sizeof(SeqCtl))
         + ((size_t) env_int("HALO_ATTN_PART_MB", 192) << 20);
}

AttnBatch * create_attn_batch(int capacity) {
    if (capacity <= RMAX || capacity > PASSMAX) return nullptr;
    if (env_int("HALO_WIDE_ATTN", 1) == 0) return nullptr;
    AttnBatch * a = new AttnBatch();
    a->capacity = capacity;
    a->group_rows = attn_group_rows();
    a->sched = attn_sched_default();
    a->fold = attn_fold_default();
    a->part_window_mb = attn_part_window_default();
    a->budget = (size_t) env_int("HALO_ATTN_PART_MB", 192) << 20;
    void * p = nullptr;
    ATTN_CHECK(hipMalloc(&p, (size_t) capacity * ATTN_OUT * sizeof(float))); a->qrot = (float *) p;
    ATTN_CHECK(hipMalloc(&p, (size_t) capacity * sizeof(RowInfo))); a->rows = (RowInfo *) p;
    ATTN_CHECK(hipMalloc(&p, (size_t) capacity * sizeof(SeqCtl))); a->groups = (SeqCtl *) p;
    a->bytes = (size_t) capacity * (ATTN_OUT * sizeof(float) + sizeof(RowInfo) + sizeof(SeqCtl));
    return a;
}

void destroy_attn_batch(AttnBatch * a) {
    if (!a) return;
    (void) hipFree(a->qrot); (void) hipFree(a->rows); (void) hipFree(a->groups); (void) hipFree(a->part);
    delete a;
}

size_t attn_batch_bytes(const AttnBatch * a) { return a ? a->bytes : 0; }
int attn_batch_capacity(const AttnBatch * a) { return a ? a->capacity : 0; }
int attn_batch_group_rows(const AttnBatch * a) { return a ? a->group_rows : 8; }
int attn_batch_sched(const AttnBatch * a) { return a ? a->sched : attn_sched_default(); }
void attn_batch_set_sched(AttnBatch * a, int s) { if (a) a->sched = s < 0 ? 0 : (s > 2 ? 2 : s); }
int attn_batch_part_window_mb(const AttnBatch * a) { return a ? a->part_window_mb : 0; }
void attn_batch_set_part_window_mb(AttnBatch * a, int mb) { if (a) a->part_window_mb = mb < 0 ? 0 : mb; }
int attn_batch_fold(const AttnBatch * a) { return a ? a->fold : 1; }
void attn_batch_set_fold(AttnBatch * a, int f) {
    if (!a) return;
#ifdef HALO_ATTN_PROBE
    if (f >= 8 && f <= 10) { a->fold = f; return; }
#endif
    a->fold = (f == 0 || f == 2 || f == 3 || f == 6) ? f : 1;
}

// 0 is the shape rule, and the shape it reads is how many key chunks the launch carries. A fold
// buys reuse and spends four wave slots (112 VGPR against 94), so it pays where a unit's key loads
// are worth something and costs where the unit is mostly its own prologue: on a 4096-token document
// it is +25% of the score phase at position 0, crosses zero near position 640 and reaches -5% at
// 3456. docs/attn-reuse-fold.md has the curve that fixes the threshold.
static int attn_fold_for(const AttnBatch * a, int maxchunks) {
    const int set = a ? a->fold : 0;
    if (set != 0) return set;
    return maxchunks >= 8 ? 2 : 1;
}
bool attn_batch_enabled(const AttnBatch * a) { return a && a->enabled; }

// The per-slice attention is the control for everything this module does, and it is reachable
// without rebuilding or reallocating: the buffers stay, the guard in forward_batch stops asking
// for them. HALO_WIDE_ATTN=0 refuses the module outright and allocates nothing, which is the right
// switch for a memory question and the wrong one for a timing pair.
void attn_batch_set_enabled(AttnBatch * a, bool on) { if (a) a->enabled = on; }

// Both widths are the same arithmetic per row, so a process can walk them as a case axis. A wider
// group merges two neighbours that a slice would have kept apart, which only lets the later rows'
// chunk cover positions the earlier rows mask off as exact zeros.
void attn_batch_set_group_rows(AttnBatch * a, int rows) {
    if (!a) return;
    a->group_rows = rows >= 16 ? 16 : 8;
}
float * attn_batch_qrot(AttnBatch * a) { return a ? a->qrot : nullptr; }

void attn_batch_begin(AttnBatch * a, const RowInfo * rows, int nrows,
                      const SeqCtl * groups, int ngroups, hipStream_t stream) {
    if (!a || nrows < 1 || nrows > a->capacity || ngroups < 1 || ngroups > a->capacity)
        throw std::runtime_error("attn_batch_begin: invalid pass shape");
    ATTN_CHECK(hipMemcpyAsync(a->rows, rows, (size_t) nrows * sizeof(RowInfo), hipMemcpyHostToDevice, stream));
    ATTN_CHECK(hipMemcpyAsync(a->groups, groups, (size_t) ngroups * sizeof(SeqCtl), hipMemcpyHostToDevice, stream));
    // The window needs the partial bytes this pass will actually write, and a group's row count is
    // the one thing the launch sizing cannot infer: a prompt pass hands eight-row groups and a
    // generation step hands one-row groups of the same nominal width. Charging a decode step for
    // eight rows a group would split it eight times over a working set already small enough to be
    // resident, so the window would cost launches and buy nothing.
    a->pass_rows = nrows; a->pass_groups = ngroups;
}

int attn_batch_groups_per_launch(AttnBatch * a, int part_stride) {
    if (!a || part_stride < 1) return 0;
    const size_t per_group = (size_t) a->group_rows * NH * part_stride * sizeof(AttnPartial);
    size_t want = per_group * (size_t) ((a->capacity + a->group_rows - 1) / a->group_rows);
    if (want > a->budget) want = per_group * (a->budget / per_group ? a->budget / per_group : 1);
    if (want > a->part_bytes) {
        void * p = nullptr;
        ATTN_CHECK(hipMalloc(&p, want));
        (void) hipFree(a->part);
        a->bytes += want - a->part_bytes;
        a->part = (AttnPartial *) p;
        a->part_bytes = want;
    }
    int gpl = (int) (a->part_bytes / per_group);
    // The window caps how much of the allocated buffer one launch pair is allowed to fill. It never
    // grows or frees anything, so an arm that sets it shares its allocation with the arm that does
    // not and the pair differs only in how many groups a score launch carries.
    if (a->part_window_mb > 0 && a->pass_groups > 0 && a->pass_rows > 0) {
        // Bytes one group of THIS pass writes, not what the allocation was sized for.
        const double rows_per_group = (double) a->pass_rows / (double) a->pass_groups;
        const double bytes_per_group = rows_per_group * (double) NH * (double) part_stride * (double) sizeof(AttnPartial);
        int wgpl = (int) (((double) a->part_window_mb * 1048576.0) / (bytes_per_group > 1.0 ? bytes_per_group : 1.0));
        const int floor_g = attn_part_min_groups();
        if (wgpl < floor_g) wgpl = floor_g;
        if (wgpl < gpl) gpl = wgpl;
    }
    return gpl < 1 ? 1 : gpl;
}

void launch_attn_pre_wide(AttnBatch * a, const FwdParams & p, const float * q_norm,
                          const float * k_norm, int kv_slot, int nrows, hipStream_t stream) {
    const int units = nrows * (NH + NKV);
    hipLaunchKernelGGL(k_attn_pre_wide, dim3(units), dim3(NT), 0, stream, p, q_norm, k_norm, kv_slot, a->rows, nrows);
}

void launch_attn_wide(AttnBatch * a, const FwdParams & p, int kv_slot, int g0, int ng,
                      int rowbase, int maxchunks, int part_stride, bool one_row, hipStream_t stream) {
    constexpr int GQA = NH / NKV;
    if (one_row && p.attn_hg > 1) {
        const dim3 hgrid(NH / GQA * maxchunks, ng);
        // The generation shape already folds the whole KV group. In the leaf coordinate it can do
        // that without the per-lane query, which is a different register count for the same map.
        if (p.k_tiled && attn_fold_for(a, maxchunks) == 6)
            hipLaunchKernelGGL((k_attn_wide<1, 0, GQA, ATTN_KM_LEAF, GQA>), hgrid, dim3(NT), 0, stream, p, kv_slot, a->rows, a->groups, g0, rowbase, a->part, part_stride);
        else
            hipLaunchKernelGGL((k_attn_wide<1, 0, GQA>), hgrid, dim3(NT), 0, stream, p, kv_slot, a->rows, a->groups, g0, rowbase, a->part, part_stride);
        return;
    }
    const dim3 grid(NH * maxchunks, ng);
    // A folded launch has HG times fewer head units, and that is the whole grid change: the unit
    // keeps its rows, its chunk and its partials, and takes HG heads of one KV group.
#ifdef HALO_ATTN_PROBE
    // The ablation runs the DEPLOYED unfolded arm with a pinned chunk, so the pair it belongs to is
    // fold 1, not fold 2. Its output is wrong; only its time means anything.
#define PROBE_ARM(TTv, Pv) do { \
        if ((Pv) == 1) hipLaunchKernelGGL((k_attn_wide<TTv, 0, 1, ATTN_KM_LEAF, 1, 1>), grid, dim3(NT), 0, stream, p, kv_slot, a->rows, a->groups, g0, rowbase, a->part, part_stride); \
        else if ((Pv) == 2) hipLaunchKernelGGL((k_attn_wide<TTv, 0, 1, ATTN_KM_LEAF, 1, 2>), grid, dim3(NT), 0, stream, p, kv_slot, a->rows, a->groups, g0, rowbase, a->part, part_stride); \
        else hipLaunchKernelGGL((k_attn_wide<TTv, 0, 1, ATTN_KM_LEAF, 1, 3>), grid, dim3(NT), 0, stream, p, kv_slot, a->rows, a->groups, g0, rowbase, a->part, part_stride); } while (0)
#endif
#define FOLD(TTv, HGv, HBv) do { const dim3 fgrid(NH / (HGv) * maxchunks, ng); \
        hipLaunchKernelGGL((k_attn_wide<TTv, 0, HGv, ATTN_KM_LEAF, HBv>), fgrid, dim3(NT), 0, stream, \
                            p, kv_slot, a->rows, a->groups, g0, rowbase, a->part, part_stride); } while (0)
#define WIDE(TTv, SCHEDv, KMv) hipLaunchKernelGGL((k_attn_wide<TTv, SCHEDv, 1, KMv>), grid, dim3(NT), 0, stream, \
                                                  p, kv_slot, a->rows, a->groups, g0, rowbase, a->part, part_stride)
    // THE ARM IS CHOSEN HERE, NOT INSIDE THE UNIT. The cache's coordinate is a process setting the
    // whole engine agrees on; which score body reads it is this launch's business, and making it a
    // template parameter is what gives each kernel its own register count. k_attn_wide<8> is 92
    // VGPR / 16 waves per SIMD32 with one body and 130 / 10 with both, which is the whole reason
    // the lane-per-key arm measured -13.5% on one build and +9.7% on another.
    //
    // The leaf body has no schedule axis - it has no cross-lane reduction for the rescheduled
    // body's medicine to treat - so it takes SCHED 0 and the sched setting is not a route choice
    // in that coordinate either.
    if (a->sched == 2) {   // the fused kernel, kept as the control; see attn_sched_default
        if (a->group_rows >= 16) WIDE(16, 1, ATTN_KM_DYN);
        else                     WIDE(8, 1, ATTN_KM_DYN);
    } else if (p.k_tiled) {
        const int fold = attn_fold_for(a, maxchunks);
        if (a->group_rows >= 16)      WIDE(16, 0, ATTN_KM_LEAF);
        else if (fold == 2)           FOLD(8, 2, 2);
        else if (fold == 3)           FOLD(8, 2, 1);
#ifdef HALO_ATTN_PROBE
        else if (fold >= 8)           PROBE_ARM(8, fold - 7);
#endif
        else                          WIDE(8, 0, ATTN_KM_LEAF);
    } else if (a->group_rows >= 16) {
        if (a->sched == 0)       WIDE(16, 0, ATTN_KM_ROW);
        else                     WIDE(16, 1, ATTN_KM_ROW);
    } else {
        if (a->sched == 0)       WIDE(8, 0, ATTN_KM_ROW);
        else                     WIDE(8, 1, ATTN_KM_ROW);
    }
#undef WIDE
#undef FOLD
#ifdef HALO_ATTN_PROBE
#undef PROBE_ARM
#endif
}

void launch_attn_combine_wide(AttnBatch * a, const FwdParams & p, int rowbase, int nrows,
                              int part_stride, hipStream_t stream) {
    const int units = nrows * (ATTN_OUT / 1024);
    const int q = sequence_quant();
#define COMBINE(QL, NIB) hipLaunchKernelGGL((k_attn_combine_wide<QL, NIB>), dim3(units), dim3(NT), 0, stream, \
                                            p, a->rows + rowbase, a->part, part_stride, nrows)
    if (!sequence_quant_out_seven(q))          COMBINE(127, false);
    else if (sequence_quant_out_nibble(q))     COMBINE(7, true);
    else                                       COMBINE(7, false);
#undef COMBINE
}

} // namespace halo

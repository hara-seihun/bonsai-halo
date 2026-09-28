// Wide attention: one launch per attention layer for every row of a batched pass.
//
// The attention phases ran inside the persistent kernel, once per eight-row slice, so a 128-row
// pass made sixteen cooperative launches per attention layer and each of them scanned the layer's
// whole KV cache for its own eight rows. It was the last phase in a wide pass still shaped by
// RMAX. This module keeps the deployed arithmetic exactly - the same rope, the same chunked
// softmax over ACHUNK keys, the same per-row reduction order and the same combine - and widens the
// row axis of the launch instead. A row group here is what a slice handed the phase: at most
// `attn_group_rows()` consecutive rows of one sequence.
//
// Chunk partials are the memory this costs. The deployed path keeps eight rows' worth with a
// compile-time AMAX_CHUNKS stride; a wide pass needs rows x heads x chunks, which grows with the
// context. The module allocates against a byte budget (HALO_ATTN_PART_MB, default 192) and reports
// how many row groups fit in one launch, so a long context splits into a few launches instead of
// demanding a gigabyte.
#pragma once
#include "halo_kernels.h"

namespace halo {

struct AttnBatch;

// Null when the route is off (HALO_WIDE_ATTN=0) or the capacity cannot use it. The capacity bound
// is PASSMAX: it tracks the widest pass `forward_batch` admits, because a pass this module refuses
// does not fail, it silently falls back to the per-slice attention this module exists to replace.
AttnBatch * create_attn_batch(int capacity);
void destroy_attn_batch(AttnBatch * a);
size_t attn_batch_required_bytes(int capacity);   // fixed buffers plus the whole partials budget
size_t attn_batch_bytes(const AttnBatch * a);
int attn_batch_capacity(const AttnBatch * a);

// Rows per unit group. 8 keeps the groups a slice would have made; 16 halves the number of times a
// key chunk is read, at twice the q registers. HALO_ATTN_TT selects the process default; the
// setter makes it an in-process case axis, which is the only way to order an effect this size on
// this device. Read the module's current width with the getter, not the environment.
int attn_group_rows();
int attn_batch_group_rows(const AttnBatch * a);
void attn_batch_set_group_rows(AttnBatch * a, int rows);

// Score-unit schedule: 0 is the deployed body kept verbatim as the exactness control, 1 the
// rescheduled one (key lookahead, breadth-first wave reductions, batched softmax reductions,
// four-key value block). Both arms compute the same floats, so this is a pure timing axis and
// never a route choice; the partials a pass writes are bit-identical either way.
// HALO_ATTN_SCHED selects the process default, the setter makes it an in-process case axis.
int attn_batch_sched(const AttnBatch * a);
void attn_batch_set_sched(AttnBatch * a, int s);

// How many query heads of one KV group a score unit carries (HALO_ATTN_FOLD, default 1 = none).
// 0 is the shape rule (fold when the launch carries eight or more key chunks). 2 and 3 fold two heads at eight-row groups, differing in whether one leaf walk serves both heads
// or each head walks the leaves itself; 6 folds a whole KV group in the one-row generation shape.
// The fold needs the leaf coordinate - only there is the query wave-uniform - and it is exact: a
// head's query, scores, softmax and partial never meet another head's. docs/attn-reuse-fold.md.
int attn_batch_fold(const AttnBatch * a);
void attn_batch_set_fold(AttnBatch * a, int f);

// How many megabytes of chunk partials one score launch is allowed to leave in flight before the
// driver runs the combine that reads them. 0 is the deployed behaviour: one score launch and one
// combine per pass, whatever that costs in bytes.
//
// It is a CACHE RESIDENCY knob, not an allocation one. The buffer stays the size the budget grew it
// to, so two arms share a process and an allocation; only how many row groups a launch pair carries
// moves. That makes it exact by construction - every (row, head, chunk) partial is computed by the
// same unit from the same keys and read by the same combine expression, and the driver already
// runs this loop whenever the budget is the smaller number. docs/attn-partial-window.md.
int attn_batch_part_window_mb(const AttnBatch * a);
void attn_batch_set_part_window_mb(AttnBatch * a, int mb);

// Route switch for a timing pair: off sends the pass back to the per-slice attention with the
// module's buffers still allocated, so two arms share a process, a clock and a warm device.
bool attn_batch_enabled(const AttnBatch * a);
void attn_batch_set_enabled(AttnBatch * a, bool on);

// Trace the three launches of an attention layer as separate phases (HALO_ATTN_TRACE_SPLIT=1).
// Default off so a trace stays comparable with every panel this lane has already taken: the
// combine reads the chunk partials and the chunk units write them, and only the split says which
// side of that buffer a position-dependent cost is on.
bool attn_trace_split();

// Uploads one pass's row and group metadata. Groups must tile [0, nrows) in row order, and each
// group must hold consecutive rows of one sequence.
void attn_batch_begin(AttnBatch * a, const RowInfo * rows, int nrows,
                      const SeqCtl * groups, int ngroups, hipStream_t stream);

// Row groups whose partials fit one launch, given this pass's chunk stride.
int attn_batch_groups_per_launch(AttnBatch * a, int part_stride);

// q/k norms, rope and the KV cache write for every row of the pass. `p` must be bound for the
// whole batch (offset 0, nrows = total) and carry the module's wide qrot.
void launch_attn_pre_wide(AttnBatch * a, const FwdParams & p, const float * q_norm,
                          const float * k_norm, int kv_slot, int nrows, hipStream_t stream);

// Scores, softmax and the value accumulation for row groups [g0, g0 + ng), writing chunk partials
// for rows [rowbase, ...). maxchunks is the grid's chunk extent; a group with fewer chunks leaves
// the extra units to exit.
//
// `one_row` says every group in this launch carries exactly one row, which is the generation shape.
// Such a launch has no neighbouring rows to share a key chunk with, so with p.attn_hg it folds a
// KV group's query heads into one unit instead. The caller owns that test because it sets the unit
// count and therefore the grid.
void launch_attn_wide(AttnBatch * a, const FwdParams & p, int kv_slot, int g0, int ng,
                      int rowbase, int maxchunks, int part_stride, bool one_row, hipStream_t stream);

// Combine, gate, Hadamard and quantise for rows [rowbase, rowbase + nrows). `p` must be bound at
// rowbase so the quantised output lands in the right rows of the wide operand.
void launch_attn_combine_wide(AttnBatch * a, const FwdParams & p, int rowbase, int nrows,
                              int part_stride, hipStream_t stream);

float * attn_batch_qrot(AttnBatch * a);

} // namespace halo

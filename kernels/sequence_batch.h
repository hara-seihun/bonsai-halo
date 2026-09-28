#pragma once
#include "halo_kernels.h"
#include <cstddef>
namespace halo {
struct SequenceBatch;
// Which order the weight image stores its (row tile, K block) runs in. Same bytes, same values,
// same order of use; only which 512-byte run a pair lands on moves, so both orders produce the
// same output bits. Tile-major is the deployed one and puts a wave's own blocks together;
// block-major puts every concurrently resident wave's block inside one contiguous run, which is
// what the DRAM path is measured against in docs/weight-stream-order.md.
enum SeqImage { SEQ_IMAGE_TILE = 0, SEQ_IMAGE_BLOCK = 1 };
SequenceBatch *create_sequence_batch(const LayerW *layers, int capacity);
void destroy_sequence_batch(SequenceBatch *);
size_t sequence_batch_required_bytes(const LayerW *layers, int capacity);
size_t sequence_batch_bytes(const SequenceBatch *);
bool sequence_direct_layout(const SequenceBatch *);
bool sequence_resident_enabled(const SequenceBatch *);
bool sequence_resident_layer(const SequenceBatch *,int layer);
void sequence_run_resident(SequenceBatch *,FwdParams,int layer,const ResidentSeq *,int nseq,int rows,hipStream_t);
// Bind producer destinations and consumer inputs directly. Returns false for the staged control.
bool sequence_bind(SequenceBatch *, FwdParams &, int layer, int offset, int total_rows, int part);
// Staged control and scaled-FP16 experiment: part 1 captures input quantizer and
// alpha/beta outputs; part 2 captures the output quantizer. Direct int8 binds
// destinations with sequence_bind instead and never launches these copies.
void sequence_capture(SequenceBatch *, const FwdParams &, int layer, int offset, int total_rows, int part, hipStream_t);
void sequence_project(SequenceBatch *, int layer, int total_rows, int part, float *residual, hipStream_t);
// Row-tile ownership for the integer projections: 0 keeps a wave on its own row tile, 1 and 2 give
// one row tile to two or four waves of a workgroup and narrow the accumulator width to match.
// Settable between calls so a measurement can order the shapes inside one process.
int sequence_batch_sched(const SequenceBatch *);
void sequence_batch_set_sched(SequenceBatch *, int sched);
// The (token tiles, waves) shape overrides, effective on the next launch. Zero is the fitted rule.
void sequence_batch_set_shape(SequenceBatch *, int tt, int wv);
int sequence_batch_shape_tt(const SequenceBatch *);
int sequence_batch_shape_w(const SequenceBatch *);
// The stored run order. `set` permutes the image in place through one scratch matrix, which costs
// about 30 ms and belongs between measured arms, never inside one. `HALO_SEQUENCE_IMAGE=tile|block`
// picks the order a process starts in.
int sequence_batch_image(const SequenceBatch *);
void sequence_batch_set_image(SequenceBatch *, int image, hipStream_t);
// The four-bit operand map, 1 for the nibble sign extension and 2 for the nine-valued pair codes
// (selected). The map and the stored code order are one choice, so `set` takes effect through the
// image sync on the next projection. A default build carries only the selected arm and `set`
// refuses the other; `sequence_batch_i4op_control()` says whether both are present, which is what
// `DEFS='-DHALO_SEQ_I4OP_CONTROL=1'` builds. docs/seq-pair-operand.md.
void sequence_batch_set_i4op(SequenceBatch *, int op);
int sequence_batch_i4op(const SequenceBatch *);
bool sequence_batch_i4op_control();
// Which token a (token tile, lane) position of the block loop carries: 0 tile-major (the deployed
// map, token = first + t*16 + col), 1 lane-major (token = first + col*TT + t, which makes a lane's
// TT activation fragments one contiguous run). The wmma reduces over K alone, so this is a
// relabelling of the sixteen independent B columns and every output element keeps its own products
// in their own order. No stored byte moves, so `set` needs no image sync.
// `DEFS='-DHALO_SEQ_TMAP_CONTROL=1'` compiles both. docs/seq-operand-coord.md.
void sequence_batch_set_tmap(SequenceBatch *, int map);
int sequence_batch_tmap(const SequenceBatch *);
bool sequence_batch_tmap_control();
// The cache-pin ablation, at runtime. Bit 0 stops the weight stream's block walk and bit 1 the
// activation fragment's, so the named stream collapses onto block 0 and hits in cache with every
// instruction, request and matrix operation of the deployed loop intact. A pinned arm is
// numerically invalid by construction; it exists to split a projection phase into the time its
// instructions cost and the time its memory round trip costs. `-DHALO_SEQ_PIN_CONTROL=1` compiles
// the runtime form and `sequence_pin_control()` says whether this build carries it.
// docs/seq-block-roundtrip.md.
void sequence_set_pin(int mask);
bool sequence_pin_control();
// Bring each stage's matrices' stored code order into line with the current `sequence_quant()`.
// Each order is what its own operand expansion reads for free and the switch is a bit permutation
// over 1.3 GB for the input stage and 0.5 for the output one, so a measurement that changes
// precision calls this between arms rather than paying for it inside a timed pass;
// `sequence_project` calls it too, so a caller that forgets is correct and slow rather than wrong.
void sequence_batch_sync_quant(SequenceBatch *, hipStream_t);
// Tell the batch which route is about to write the output operand of the layer it will project.
// The four-bit output coordinate lives in the wide writers - `resident_output` for a recurrent
// layer, the wide attention combine for an attention one - and the per-slice half-layer inside the
// persistent kernel has no nibble store, so `sequence_project(part = 2)` refuses rather than read
// an eight-bit operand through the four-bit instruction.
void sequence_set_output_wide(SequenceBatch *, bool wide);
void sequence_restore(SequenceBatch *, const FwdParams &, int layer, int offset, hipStream_t);
}

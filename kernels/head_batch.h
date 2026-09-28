// Wide vocabulary projection: one pass over the output weight image for every row of a batch.
//
// The deployed head runs inside the persistent kernel, once per eight-row slice, so a 128-row
// prompt pass visits the 278 MB output image sixteen times and fills only eight of the sixteen
// columns each matrix instruction offers. This module keeps the deployed arithmetic map - the same
// quantiser, the same 32-row HALO tiles peeled the same way, the same eight-wave split of the 40
// K-blocks and the same reduction order - and widens the token axis instead.
#pragma once
#include "halo_kernels.h"

namespace halo {

struct HeadBatch;

// Returns null when the route is switched off (HALO_WIDE_HEAD=0) or the capacity cannot use it.
HeadBatch * create_head_batch(int capacity);
void destroy_head_batch(HeadBatch * h);
size_t head_batch_required_bytes(int capacity);
size_t head_batch_bytes(const HeadBatch * h);
int head_batch_capacity(const HeadBatch * h);

// Binds one slice's quantiser destination inside the batch-wide WMMA operand, exactly as
// sequence_bind does for a layer projection. The slice then runs SEQ_PART_HEAD.
void head_bind(HeadBatch * h, FwdParams & p, int offset, int total);

// Vocabulary projection and argmax over rows `[first_row, total)` of the batch, from the operand
// the slices wrote. `first_row = 0` is every row. The head is the only phase whose whole cost is
// output width, and prompt ingestion reads one row of it, so a caller that will read one row asks
// for one row. Rows below `first_row` keep whatever `logits` and `argmax` already held; nothing
// outside the selection is written.
void run_head_batch(HeadBatch * h, const FwdParams & p, int total, int first_row, hipStream_t stream);

const float * head_logits(const HeadBatch * h);
const int * head_argmax(const HeadBatch * h);

// Which ternary operand map `head_tile` builds: 0 the peel, 1 the product gather
// (`kernels/halo_expand.hpp`). The arms read the same image and write the same logit bits, so this
// is a measurement axis, not a capability. `HALO_HEAD_OP` sets the process default and
// `head_set_operand_arm` lets one process alternate them in a panel.
int head_operand_arm();
void head_set_operand_arm(int arm);

// Token columns per workgroup for the next launch: -1 the shape rule, else 1, 2, 4 or 8. Every
// width produces the same logits and reads the weight image a different number of times.
void head_set_tile_width(int tt);

} // namespace halo

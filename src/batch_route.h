// Which schedule `forward_batch` gives a pass, as a function of how many rows it carries.
//
// A batched pass has two schedules available. The wide one runs each layer as host-ordered kernels
// over the whole pass: one input prep, one input projection, one recurrent-state or attention core,
// one output projection, one FFN, one head. The sliced one hands eight rows at a time to the
// persistent kernel, which re-reads that layer's qkv / o / ssm_out weight stream for every slice.
//
// The row count decides, and until this knob existed the deciding constant was the literal 32 in
// two expressions. Below it a pass lost four things at once: the wide sequence projections, the
// wide input prep, the resident GDN state, and - for mode 19 - the A4 FFN module, which fell back
// to the engine's own ternary FFN. At 32 rows all four arrived together. Nothing in any wide kernel
// asks for 32 rows: `wide_head` and `wide_attn` already admit at `total > RMAX`, the projections
// take their token-tile count from the row count, and the resident state is built per sequence.
// `RMAX + 1` is therefore the lowest floor the route is defined at, and it is the first row count
// at which the sliced route needs a second pass over the whole weight stream.
//
// CROSSING THE FLOOR IS A NUMERICAL CHANGE, NOT A SCHEDULE CHANGE, and that is why the default
// below is still 32. Two floors that route the same pass the same way produce the same bits; a
// floor that moves a pass onto the wide route gives it the wide route's map, which reduces FP32 in
// a different order from the persistent kernel and (since `c131fe2` and `f91a815`) quantises the
// sequence projections' activations to four bits. `docs/serve-wide-decode.md` measures what the
// band is worth on both shapes, what the map costs in greedy continuations, and what a process
// that wants the speed without the four-bit operand should pin.
//
// `HALO_WIDE_MIN` gives the same choice to a process that cannot call the setter.
#pragma once

namespace halo {

// The floor this engine shipped with, and still runs by default.
constexpr int WIDE_MIN_DEFAULT = 32;

// Rows at or above which an automatic mode takes the wide schedule. Clamped to 1..PASSMAX.
int  batch_wide_min();
void batch_set_wide_min(int rows);

}

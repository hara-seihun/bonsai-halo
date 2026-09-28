# How wide a state unit should be, and why the answer is one sequence deep

The resident recurrence splits a head's 128 state rows between `SPLIT` units, and only the losing
half of that axis had ever been compiled. `gdn_load`, `gdn_token` and `resident_state` are written
as `R = 16 / SPLIT` rows per wave and `RPU = 128 / SPLIT` rows per unit, with
`static_assert(1 << SHIFT == SPLIT, "SPLIT is 4, 8 or 16")` and `create_sequence_batch` refusing any
other value, so [resident GDN](resident-gdn.md) could measure 4, 8 and 16 and nothing wider.

It measured 8 and 16 as losses - 242.8 and 253.4 ms against 235.6 at 4 on a 128-row pass - and named
the reason: **a unit re-reads the whole token's `k` and `q` however few rows it owns**, so narrowing
the unit multiplies its operand stream. [Batching the row reductions](gdn-token-reduce.md) closed
with the same sentence from the other side: "the decomposition that pays is one that adds units
without adding operand reads, and it lives in `resident_state`."

Going the other way is that decomposition read backwards, and it is worth **-6.4% of the phase on a
128-row prompt pass, bit-identical, with one line of arithmetic unchanged**. It is also worth
**+1 to +5% against you** on every shape with more than one sequence, which is the more useful half
of the result.

## The ladder

Mode 19, pinned clock, both arms interleaved in one process, `gdn-resident-core` beside the phases
the width cannot reach (`ffn`, both sequence projections, `sequence-core`, the head and the input
prep). Raw in [`batch-comparison/gdn-unit-width/`](../../../data/bonsai2/batch-comparison/gdn-unit-width/).

| SPLIT | rows/wave | rows/unit | units, 1 seq | `gdn-resident-core` | of untouched | against SPLIT 4 |
|---:|---:|---:|---:|---:|---:|---:|
| 4 *(was the default)* | 4 | 32 | 192 | 15.400 | 0.09916 | — |
| **2** | **8** | **64** | **96** | **14.513** | **0.09332** | **-5.9%** |
| 1 | 16 | 128 | 48 | 16.102 | 0.10419 | +5.1% |

Samples do not overlap: 15.410/15.391 at 4, 14.579/14.446 at 2, 16.154/16.049 at 1. Every arm
returns residual FNV-64 `7446865760224376151`, the hash this lane has published for a 128-row mode
19 pass, so the whole ladder is one numerical map.

**The axis has an interior optimum and both neighbours are worse.** That shape is what two competing
terms look like. The per-token cost that does *not* scale with the rows a wave owns - the gate, the
`k` and `q` fragments every wave of the block reads identically, the loop and its addressing - is
paid once per wave per token, so it falls as `1/R`. What rises as `R` grows is how little of the
machine the grid can fill: one sequence deals `48 * SPLIT` units, and at SPLIT 1 that is 48 blocks
of a 20-WGP machine, 384 waves for 640 wave slots.

## Which shape wants which width

Same panel construction, one process per shape:

| shape | rows per sequence | SPLIT 4 | SPLIT 2 | change |
|---|---:|---:|---:|---:|
| 1 seq x 128 rows | 128 | 15.400 | **14.513** | **-5.9%** |
| 1 seq x 32 rows | 32 | 4.686 | **4.490** | **-4.2%** |
| 4 seq x 32 tokens | 32 | 14.890 | 14.618 | null (0.09403 -> 0.09381) |
| 2 seq x 64 tokens | 64 | 15.364 | 16.094 | +4.8% |
| 32 seq x 8 tokens | 8 | 63.883 | 64.695 | +1.3% |
| 32 seq x 1 token | 1 | 44.926 | 45.791 | +1.9% |

**The axis is the depth of the grid. It is not prefill against decode and it is not tokens per
sequence,** and the 4x32 and 2x64 rows are what settle that: both carry 128 rows in 32-token
sequences, one is a null and the other is the worst cell in the table, and they differ only in how
many units their `nseq` deals. One sequence is the only shape where SPLIT 4's 192 units exceed the
160 blocks this machine holds - 1.2 rounds of work taking two, the 1.67x makespan
[gdn-token-reduce](gdn-token-reduce.md) published - while its wave slots sit idle. Two sequences
deal 384 units, and from there on other waves already cover the per-token fixed cost, so all that
is left of a wider unit is its lower occupancy and a coarser round.

`gdn_resident_split_for(nseq)` is therefore `nseq == 1 ? 2 : 4`, which is the deployed prompt
ingestion path exactly: `Engine::prefill` ingests one sequence per pass. Batched generation and
batched ingestion keep the width they had, bit for bit and instruction for instruction.

## What it costs, from the compiler and from the runtime

`tools/gdn_occ_probe` (extended here with the 1 and 2 instantiations), gfx1151, no spill in any arm:

| instantiation | VGPR | LDS | blocks/WGP | waves/SIMD32 |
|---|---:|---:|---:|---:|
| `state<4,*,commit>` | 64 | 0 | 8 | 16 |
| `state<4,*,defer>` | 64 | 5156 | 8 | 16 |
| `state<2,*,commit>` | 100 | 0 | 7 | 14 |
| `state<2,*,defer>` | 105 | 6180 | 6 | 12 |
| `state<1,HOIST,commit>` | 173 | 0 | 4 | 8 |
| `state<1,HOIST,defer>` | 194 | 8228 | 3 | 6 |

`m[R][4]` is 16 floats per lane at R=4 and 32 at R=8, and the two `GdnTokT<R>` in flight are 10 and
18 more, which is the whole register story. **The lower occupancy never binds at one sequence**: 96
blocks against a 140-block capacity is still one round. It is exactly what costs the wider unit its
1 to 5% everywhere else, where the grid is deep enough to want all 160.

The deferred arm's LDS grows with `RPU` because `sdels` stages `GDN_DEFER_MAX * RPU` floats, which
is why the defer column drops a block per step of the ladder.

## Two bounded negatives from this turn

**SPLIT 1 is not in the tree.** It is a measured loss and four instantiations of a 173-VGPR body
cost this translation unit about eight seconds, so `launch_resident_arm` stops at 2 and
`gdn_split_valid` rejects 1. `tools/gdn_occ_probe` still instantiates it, which is where the
register ladder above stays reproducible.

**Hand-writing the token cursor loses, and the compiler was already doing it.**
[resident GDN](resident-gdn.md)'s census attributes 30 of a token's 355 issue slots to rebuilding
`a.tokens + (size_t)(S.row0 + t) * RESIDENT_TOKEN_FLOATS` and the output row beside it, and
[wide-head](wide-head.md) removed exactly that defect from `head_tile` for 109 of 733 slots. Here it
does not transfer. Replacing both with loop-carried cursors - once as a `const float *` advanced by
`RESIDENT_TOKEN_FLOATS`, once as `wide-head`'s prescribed 32-bit offset against an invariant base -
makes `resident_state<2,HOIST,commit>` **larger**, and the reason is visible in one line of the
census:

| arm | whole kernel | SALU | `global_load_b32` scalar base + lane offset | 64-bit VGPR address pair |
|---|---:|---:|---:|---:|
| recomputed (in the tree) | 3016 | 712 | 92 | 3 |
| 64-bit cursor | 3111 | 702 | 64 | **47** |
| 32-bit offset | 3116 | 723 | 64 | **47** |

The kernel hands its token pointer to `gdn_load`, which does its own lane indexing off it. When that
pointer is recomputed from `t`, the compiler strength-reduces it itself and keeps every load in the
`SADDR + lane offset` form; when it arrives as a loop-carried value, 47 loads fall back to per-lane
64-bit address pairs. **The 30 slots are in the source and not in the object.** Reproduce with
`hipcc ... --cuda-device-only -S tools/gdn_occ_probe.hip` and
`tools/isa_loop_count.py /tmp/x.s resident_stateILi2ELi1ELb0`, which costs ten seconds and no lock.

## Full model, and what this is not worth

`gdn-resident-core` is 15.2 ms of a 172 ms 128-row pass on the shipping build, so -6.4% of it is
0.96 ms, **half a percent of prompt ingestion**. A 1536-token document in 128-row passes, arms
interleaved in one process, reads 698.0/810.2/804.2 tok/s pinned at 4 against 711.1/809.7/809.0
under the rule: paired by round that is +1.9%, -0.1% and +0.6%, which is this box's spread and not a
measurement of a half-percent change. 32-stream aggregate generation is **a null by construction** -
the rule returns 4 there, so the two arms run the same instantiation on the same grid - and it reads
320.7/320.9 against 321.5/322.5 tok/s, +0.3%.

**Re-measured on `82dc7c7`, where [the four-bit sequence operand](seq-a4-default.md) became the
default under this result, the width is worth more than it was**: `gdn-resident-core` 15.133 ->
13.736 ms, 0.11528 -> 0.10564 of the untouched phases, -8.4% normalised with non-overlapping
samples and one residual hash (`8607785813945241221`) across both arms. The phase did not get
slower; everything around it got faster, so the same 1.4 ms is now 10.6% of a pass instead of 9.3%,
and about 1% of prompt ingestion. **A phase's share is a moving target in a lane this busy - take
the ratio, name the build, and expect the value of your own change to drift with someone else's.**

**The phase panel is the evidence. The full-model column is the regression check.** Single-stream
`--bench` never enters this kernel at all: it runs `ph_gdn` inside `k_forward_rows`, whose
`GDN_SPLIT` is untouched.

## Acceptance

- Residual FNV-64 equal between arms at every shape measured: `7446865760224376151` (128 rows),
  `1873717996769712938` (32 rows), `8087160186626784132` (32 streams x 1),
  `375599202597429117` (32x4), `15979693088082977245` (32x8), `14064500533971998802` (2x64),
  `17951926377674091439` (4x32).
- [Mixed commit, replay, ragged-slice and rollback state acceptance](../../../data/bonsai2/batch-comparison/gdn-unit-width/state-check.json)
  passes on the shipping build: seven mixed passes equal, state and ring equal at the synchronising
  commit, rollback boundary clean.
- `tools/direct-commit/Makefile` had drifted from the root `KSRC` and could not link
  `sequence_state_check` at all - eight undefined `halo::attn_batch_*` symbols - so the state
  acceptance of any change to this kernel was unavailable until this turn. Fixed here.

## The next questions, in the order the evidence ranks them

1. **R = 6 is the one point on this axis the indexing cannot express, and it is the predicted
   optimum.** `SPLIT * R = 16` forces R into powers of two; 48 rows per unit means 144 units for one
   sequence, one round of a 160-block machine *and* three quarters of the operand reads. It needs
   `SHIFT`, `blockIdx.x % SPLIT` and `1 << SHIFT == SPLIT` replaced by a divide and a ragged last
   unit of 32 rows whose two spare rows are masked out of their stores. The measured curve says the
   prize is small - the optimum is flat between 4 and 8 - but it is the only untested point, and
   `ca87b207` derived the makespan arithmetic that names it.
2. **Rows per wave and waves per block are independent, and only the first has been walked.**
   `6fd822ef` priced the second: `R = 128/(SPLIT*W)` with `W` hardcoded 8 by `__launch_bounds__(256)`
   and by the literal 8 in `part*RPU + wave + 8*r`, so R=8 is also reachable as (SPLIT 4, W 4) with
   **192 blocks instead of 96** - the same rows in the same lanes with a quarter of the CU imbalance.
   Two places break silently if the thread count moves and not if only SPLIT does: the conv-ring
   commit's `group < CONV_CH/256` with `group*256 + threadIdx.x`, and `gdn_defer_stage(..., 256, ...)`.
3. **The state row layout is orthogonal to both.** `3c5f09be` is building the permutation that puts
   a lane's rows in one contiguous run, which at R=8 turns eight narrow loads into two `dwordx4`.
   It composes with this change and needs nothing from it.
4. **`warp_sum` leaves two distinct bit patterns in a wave, not one.** `e0c9652e` derived it: after
   the two quad permutes and the two row rotations, lane `l` holds a sum whose association depends on
   bit 2 of the lane index, so `d[r]` differs between lane classes and `out_o` always stores the even
   class. Row ownership is blind to it - this change moves whole rows and keeps L=32 - but anything
   that moves a **column** between lanes has to reproduce both classes, and a single rounding is the
   wrong invariant to code against.

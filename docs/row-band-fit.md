# The projections' token-tile ladder, measured at fixed width, under both operand widths

[The decode map](decode-map.md) closed with a section called "The 33 to 127 row band is unfitted"
and two suspects in it. This document measures both and **ships neither**. One suspect had already
been fixed by a later iteration; the other is a real 2.6 to 4.8% under the eight-bit operand and
**a 10 to 17% loss under the four-bit operand the route now defaults to**, which is the finding.

What is in the tree is the instrument: the shape overrides at runtime, and the ladder as case axes
on `tools/batch_profile`, so the next engineer walks this in one process instead of one process per
point. Raw samples and every failed cell are in
[`batch-comparison/row-band-fit/`](../../../data/bonsai2/batch-comparison/row-band-fit/).

## The register table, which is what made this look like an oversight

`kernel_resources.py` on `kernels/sequence_batch.hip`, deployed `project` instantiations
(`RT = 1`, lane operand path), one compile and no lock. `int4v` is the eight-bit activation operand
and `int2v` the four-bit one:

| token tiles | VGPR, eight-bit | waves | VGPR, four-bit `SB=1` | waves | VGPR, four-bit `SB=2` | waves |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 77-78 | 16 | 73-74 | 16 | 73-74 | 16 |
| 2 | 95 | 16 | 90-91 | 16 | 90-91 | 16 |
| 4 | 144 | 10 | **180** | **8** | 133 | 10 |
| 8 | 236-240 | 6 | 213 | 7 | 213 | 7 |

Two token tiles is the last width that costs nothing. `sequence_shape` asks for as many as the
accumulator budget `cap = 4` allows once the device holds `SEQ_WAVES_WANTED = 640` waves, so it
asks for four wherever the token axis reaches it and pays a third of the wave slots for it.
**`cap` is a correctness bound on the instantiation, not the shape you want** - and that reading is
still wrong, for a reason no register table shows.

**No `project` instantiation spills or uses scratch at any width.** That disposes of the decode
map's own hypothesis about its 64-row cell ("that smells like spill") in one compile.

## The ladder at fixed width, eight-bit operand

The map's three cells changed the row count and the token-tile count together, which is why no cost
model fits them: 12.924 ms at one tile, 28.467 at two, 28.382 at four implies 15.5 ms per token tile
between the first two cells and zero between the last two. Hold the width fixed and the ladder is
flat to two tiles and then a cliff.

Mode 19, f32 state, **a8** activations (the default until `c131fe2`), 32 streams, width set by
`--decode-tokens`, one process, `--seq-tt` forcing the shape on both stages. `ctl` is the four
phases a shape cannot reach (`ffn`, `gdn-resident-core`, `sequence-core`, `head-projection`), which
held inside 1% across all eight cells of each panel.

| width | token tiles | IN ms | OUT ms | IN+OUT | ctl ms |
|---|---|---:|---:|---:|---:|
| 32 rows | 1 | 13.380 | **7.128** | 20.509 | 81.98 |
| 32 rows | 2 | **9.894** | 7.175 | 17.069 | 81.87 |
| 32 rows | 4 | 19.970 | 9.664 | 29.634 | 81.86 |
| 32 rows | the rule | 9.877 | 7.177 | 17.054 | 81.83 |
| 64 rows | 1 | 24.247 | 9.823 | 34.070 | 95.54 |
| 64 rows | 2 | **20.342** | **9.471** | **29.812** | 95.96 |
| 64 rows | 4 | 21.442 | 14.584 | 36.026 | 95.74 |
| 64 rows | the rule | 21.806 | 9.517 | 31.322 | 96.61 |
| 128 rows | 1 | 45.464 | 19.066 | 64.530 | 137.33 |
| 128 rows | 2 | **38.129** | **15.927** | **54.056** | 137.93 |
| 128 rows | 4 | 39.242 | 16.127 | 55.369 | 137.64 |
| 128 rows | the rule | 39.359 | 16.114 | 55.473 | 137.19 |

- **The two stages take different shapes and each had its own optimum at 32 rows.** Their tile
  counts differ (`w.n / 16`), so the rule gives the input stage two tiles and the output stage one,
  and at 32 rows that is exactly the best cell of each.
- **Four tiles never won under this operand**: 2.4% behind two tiles at 128 rows, 20.8% at 64,
  73.6% at 32. The rule picks four at 64 rows for the input stage and at 128 for both.
- Clamping the rule at two tiles was **bit-identical** - residual FNV-64 `18005677593235076800` in
  every 128-row cell of both arms and `18336743695239402094` in every 256-row cell - and worth
  -2.74% of the projections at 128 rows, -4.2% at 64, a null at 32 and -0.67% at 256, normalised
  against `ctl`. Full model at 1536 tokens it was **+0.2%, inside a 4% spread**.

## Then the operand halved, and the sign flipped

`c131fe2` made four-bit activations the default on the sequence input projection. Same panel, same
widths, arms interleaved, `quant a4` in every cell:

| width | rule | IN ms | OUT ms | IN+OUT | ctl ms | (IN+OUT)/ctl |
|---|---|---:|---:|---:|---:|---:|
| 64 rows | four tiles (the rule) | **11.798** | 9.539 | **21.337** | 96.96 | **0.2199** |
| 64 rows | clamped to two | 15.240 | 9.626 | 24.866 | 96.88 | 0.2565 (**+16.6%**) |
| 128 rows | four tiles (the rule) | **21.293** | 15.833 | **37.126** | 135.46 | **0.2741** |
| 128 rows | clamped to two | 25.321 | 15.452 | 40.773 | 134.86 | 0.3023 (**+10.3%**) |

The whole reversal is in the input stage: 11.798 against 15.240 at 64 rows, 21.293 against 25.321 at
128. The output stage, which still reads an eight-bit operand, does not move.

**Why it flips.** A token tile costs a wave one fragment fetch per K16 slice and one accumulator
group. The four-bit operand halves the fetch - `int2v` instead of `int4v` - so a token tile costs
half as much to feed, and the amortisation that wave slots were being spent on arrives for free.
At 180 VGPR and eight waves per SIMD32, four four-bit token tiles beat two at sixteen waves. **The
token-tile optimum is a property of the operand width, not of the row count**, and the rule that
looked mis-fitted against the register table is right on the route that ships.

## The wave slot the second barrier buys does not pay either

The four-bit `TT = 4` arm is 180 VGPR at eight waves with the deployed barrier and 133 at ten with
a barrier every second slice, so that arm has a free occupancy step in it. It loses. Same shape,
same widths, `--seq-sched 1,5`, `HALO_SEQ_BAND=0`:

| width | barrier | IN ms | OUT ms | IN+OUT | ratio |
|---|---|---:|---:|---:|---:|
| 64 rows | deployed (8 waves) | **11.903** | 9.594 | **21.497** | **0.2213** |
| 64 rows | every second slice (10 waves) | 14.304 | 9.535 | 23.838 | 0.2448 (**+10.9%**) |
| 128 rows | deployed (8 waves) | **20.910** | 15.605 | **36.515** | **0.2737** |
| 128 rows | every second slice (10 waves) | 23.471 | 14.712 | 38.183 | 0.2852 (**+4.6%**) |

A quarter more wave slots, 5 to 11% slower. The input stage pays 12 to 20% for the barrier and the
output stage does not care. This is the third measured case in this kernel where a wave slot was
not what it wanted, after [cache policy](cache-policy.md) and
[fragment reuse](sequence-fragment-reuse.md).

One cell in that table is the only unexplored thing left on this axis: at 128 rows the **output**
stage reads 14.712 ms with the second barrier against 15.605 without, while the input stage reads
23.471 against 20.910. The two stages want opposite barriers and `sequence_shape` hands them one
`sched`. The output-stage difference is 5.7% of its phase on n=2 with a 7% spread across my four
panels at that width, so it is not resolved - but a per-stage barrier is a real degree of freedom,
and [the FFN's image rule](ffn-decode-shape.md) went per-stage for exactly this reason.

## The FFN half of the band was already fixed

The map's larger number - forcing the dense five-trit image at 64 rows for `ffn` 68.780 -> 57.272 ms
and the step 181.662 -> 172.203 - **is already shipped, and re-deriving it would be a regression.**
The map measured all-dense against the rule of its day, pair codes on both stages above 32 rows.
[The decode-shape fit](ffn-decode-shape.md) then gave the rule one threshold per stage,
`A4_GU_PAIR_ROWS = 32` with `A4_DN_PAIR_ROWS = 64`: pair gate/up with dense down at 64 rows, 51.88
and 52.13 ms against dense+dense 53.14/53.28 and pair+pair 62.58/62.82, +6.9% full-model prompt at
64 rows per pass. Forcing all-dense at 64 rows today is 1.3 ms *worse*.

The stride rule cannot move that crossover either: it bought the pair arm 11.7%, which takes
pair+pair at 64 rows from 62.6 to about 55.3, still 3.4 ms behind the shipping pair+dense.

## What is in the tree

- `sequence_batch_set_shape(s, tt, wv)` sets the shape overrides at runtime. `sequence_shape` runs
  on every `sequence_project` call, so they take effect on the next launch and a ladder walks
  inside one process against one warmed device and one clock. `HALO_SEQUENCE_TT`/`_W` are read in
  `create_sequence_batch` and cost every point its own process on a box that moves 15% between
  them, which is why this axis had never been walked.
- `tools/batch_profile --seq-tt N,... --seq-w N,...` are those overrides as decode-loop case axes,
  recorded per sample as `seq_tt` and `seq_w`.
- **`--decode-tokens K` at 32 streams is a fixed-width probe for any width `32K`.** The projections
  see `npad = 32K` and take the instantiation and grid a prefill pass of that width takes, which is
  how the 33..127-row band gets measured at all without a batched drafter. Use it before
  interpolating a shape rule.

## The next useful question

**The input projection has no fixed cost and the output projection is nearly all fixed cost**, and
nothing has used that. Fitting the eight-bit panels: `IN` is 0.327 ms per row with an intercept
inside noise of zero, while `OUT` is 4.94 ms of weight stream plus 0.07 ms per row - so at 32 rows
the output stage spends 69% of its time on a stream it walks whatever the batch is, with its grid
fixed at `w.n / 16 = 320` workgroups and no way to put more waves on that stream without splitting
K. Its drain is FP32 per block, so a K split is a reassociation and needs its own quality arm; the
[horizon instrument](seq-a4-default.md) now carries the precision axis and can price one.

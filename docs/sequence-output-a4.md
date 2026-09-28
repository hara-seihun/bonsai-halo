# Four-bit activations on the sequence output projection

`sequence-output-projection` is the second of the two `project<>` stages and the largest phase this
lane had left unclaimed: 10.1% of a 256-row prompt pass and 10% of a 32-stream generation step.
[The document that read it](sequence-output-projection.md) priced four-bit activations off a
memory-path model, found that `iu8` and `iu4` ask the operand path for the same 0.5 bytes per lane
per cycle, and closed with *"do not start that build on the arithmetic alone"*.

Then [the input stage was built](sequence-input-a4.md) and measured **1.84x**. Same kernel, same
instruction, one stage over. This iteration runs the same instrument on the stage the negative was
written about, and gives `HALO_SEQ_QUANT` a coordinate per stage so the two can be separated.

## What was missing, and it was only the store

`project<TT,ADD,PLANAR,WV,SHARE,RT,SB,LA,I4>` has been stage-agnostic since `53c8c97`: `part == 1`
and `part == 2` are the same kernel with `ADD` and a different matrix. What gated the output stage
was the *quantiser*, and the sequence path has three of them:

| operand | written by | file |
|---|---|---|
| input projection, wide route | `prep_chunk` | `kernels/prep_batch.hip` |
| output projection, 48 recurrent layers | `resident_output` | `kernels/halo_rows.hip` |
| output projection, 16 attention layers | `k_attn_combine_wide` | `kernels/attn_batch.hip` |
| either, per-slice fallback | `ph_prep<DQ_*>` inside `k_forward_rows` | `kernels/halo_rows.hip` |

`prep_chunk_r` already carried `<DIRECT, QL, NIB>`; only the first of those four callers passed
anything but the default. The change is to pass it: `ph_prep` gains `QL`/`NIB` and hands them
through, and the two wide output writers become templates on them with `<127,false>` as the
deployed instantiation. Nothing else in either kernel moves, and neither does the per-slice
fallback — which is why the route refuses rather than falls back (below).

## The coordinate is per stage

`HALO_SEQ_QUANT` was one process-wide level describing the input stage. It now names both:

| value | input stage | output stage |
|---|---|---|
| `a8` (0) | 127 levels, `iu8` | 127 levels, `iu8` |
| `a4e` (1) | 7 levels, `iu8` | `a8` |
| `a4` (2) | 7 levels, `iu4` | `a8` |
| `a4e-out` (3) | `a8` | 7 levels, `iu8` |
| `a4-out` (4) | `a8` | 7 levels, `iu4` |
| `a4e-both` (5) | 7 levels, `iu8` | 7 levels, `iu8` |
| `a4-both` (6) | 7 levels, `iu4` | 7 levels, `iu4` |

0 to 2 mean exactly what they meant when they were measured, so
[the input stage's numbers](sequence-input-a4.md) still name the same arms. `a8` is the default and
is untouched.

The `-e` arms are the same device: seven levels stored as bytes and multiplied by the `iu8`
instruction, so every product, every `int32` accumulator and every FP32 drain is the one the
four-bit kernel computes. `a4e-out` against `a4-out` is therefore a bit-equality test of the kernel
against its own numerical map, and `a8` against `a4e-out` is the quality question with no kernel in
it.

## Why it refuses instead of falling back

The stored code order of a projection matrix is a process-wide property of the image: the spread
order `expand_i8_spread` reads and the nibble order `expand_i4_nib` reads are the same 2.000 bits
per weight with the codes on different bits, and `sequence_batch_sync_quant` permutes the image in
place when the coordinate changes. Each stage now carries its own order.

So an eight-bit *operand* cannot be quietly projected while the coordinate is four-bit: the weights
under it are in nibble order and the `iu8` expansion would read them as spread codes. A pass whose
output operand came from the per-slice half-layer — which happens when the wide attention route is
not taken, and that route wants more than `RMAX` rows — has no nibble store, so
`sequence_project(part = 2)` throws instead. `SequenceBatch::out_wide`, set per layer in
`src/batch.cpp` by whichever route is about to write, is what it checks.

That makes `a4-out` and `a4-both` wide-route coordinates: they need mode 19 or 20 and a pass wider
than eight rows. The input stage has the softer version of the same restriction (it needs the wide
prep) and it is the same reason.

## What the shape says before the device does

A serial issue model at the 2428 MHz [this box runs at](sequence-projection-operands.md), 32 cycles
for `v_wmma_i32_16x16x16_iu8` and 16 for `iu4`, one for every other slot, against the measured
phase:

| shape | block slots | of which matrix | issue model | measured | over model |
|---|---:|---:|---:|---:|---:|
| output, 128 rows, `TT=4` | 374 | 32 x 32 cycles | 13.83 ms | 21.8 | 158% |
| output, 32 rows, `TT=1` | 209 | 8 x 32 cycles | 4.63 ms | 7.9 | 171% |

Halving the matrix cycles and holding the non-issue residue fixed predicts 21.8 → 16.6 ms at 128
rows and 7.9 → 6.6 at 32, which is about +3% of a prompt pass and +1.7% of a 32-stream step. That
is smaller than the input stage got, and the reason is in the same table: the output stage carries
42% and 71% of non-issue time where the input stage carries 29%.

**The output stage always runs 640 waves.** `sequence_shape` targets `SEQ_WAVES_WANTED = 640` from
the row-tile count, and the output matrix has exactly 320 row tiles, so the rule lands `W=2` with
`TT=4` at 128 rows and `TT=1` at 32 — 320 workgroups of two waves either way. That is eight waves
per SIMD32 on a kernel the compiler holds at ten (`TT=4`, 135 VGPR) and sixteen (`TT=1`, 78). The
input matrix has 1024 row tiles and gets 2048 waves for the same work per row. Whatever the output
stage's non-issue residue is, it has eight waves to hide it behind and the deployed decomposition
has no more to give.

## Measured

**Two builds, because the weight image moved under this one.**
[The block-major run order](weight-stream-order.md) landed between the first panel and publication
and it takes 3.9 ms out of this same phase, so the headline below is the panel on the build that
ships and the earlier one is kept as the second table. Both agree on the phase and disagree on what
it is worth, which is the point: two levers on one phase do not compose by multiplying.

### On the build that ships (`d9cdb70`, block-major image)

`tools/batch_profile` `7638debad86f6ed1`, mode 19, 128-row passes, three rounds with the first
dropped, arms interleaved, `--pin-clock`, normalised by four unreachable phases:

| phase | `a8` | `a4e-out` | `a4-out` | norm `a4e-out` | norm `a4-out` |
|---|---:|---:|---:|---:|---:|
| `ffn` | 83.793 | 82.942 | 79.628 | -0.69% | +0.72% |
| `sequence-input-projection` | 46.726 | 46.425 | 44.459 | -0.32% | +0.84% |
| **`sequence-output-projection`** | **17.769** | 17.637 | **10.073** | -0.42% | **-39.92%** |
| `gdn-resident-core` | 18.282 | 18.193 | 17.337 | -0.16% | +0.50% |
| `sequence-core` | 4.562 | 4.543 | 4.340 | -0.09% | +0.83% |

Full model, 1024-token document, `--only prefill`, three rounds with the first dropped:

| prompt tok/s | `a8` | `a4-out` | `a4-both` |
|---|---:|---:|---:|
| 128-row passes | 823.9 | 858.6 (**+4.2%**) | 966.1 (**+17.3%**) |
| 256-row passes | 833.3 | 872.1 (**+4.7%**) | 978.4 (**+17.4%**) |

The residual FNV-64 is the same on both images in every arm — `7446865760224376151` for `a8`,
`8442032652492096033` for `a4e-out` and `a4-out`, `3240780174642567821` for `a4-both` — so the run
order is bit-identical as its author says, and this coordinate reproduces its own map on either one.

### Installed acceptance

Canonical tree, `tools/batch_profile` `fe3bb8d5bd95be07`, same panel:
`sequence-output-projection` **16.407 to 9.969 ms, -39.41%** normalised, reproducing the checkout's
-39.9%, and the residual FNV-64 is the same value the checkout produced in every arm
(`7446865760224376151` for `a8`, `8442032652492096033` for `a4e-out` and `a4-out`). `a4e-out` is
+0.89%, the null it has to be. The resident service was restored and is answering.

The installed full-model prefill panel ran while several engineers were compiling and its arms
spread 3% within themselves (`a8` 741.5-763.4, `a4-out` 745.2-782.0, `a4-both` 843.9-853.8 tok/s at
256-row passes); it agrees in direction and does not have the resolution of the checkout panel
above, which is the one to quote.

### On the build it was developed against (`bc74f2f`, tile-major image)

`tools/batch_profile` `913fa4e21f801603`, mode 19, one process, arms interleaved and
reshuffled per round, three rounds with the first dropped, heads on, `--pin-clock`. Four phases the
arm cannot reach are the in-panel control and every one of them is inside 0.6%; the `norm` columns
divide by their geometric mean.

| rows 128 | `a8` | `a4e-out` | `a4-out` | norm `a4e-out` | norm `a4-out` |
|---|---:|---:|---:|---:|---:|
| `ffn` | 72.678 | 72.256 | 72.963 | +0.15% | +0.34% |
| `sequence-input-projection` | 43.393 | 42.889 | 43.507 | -0.44% | +0.21% |
| **`sequence-output-projection`** | **21.707** | 21.361 | **9.903** | -0.87% | **-54.40%** |
| `gdn-resident-core` | 16.074 | 15.936 | 15.986 | -0.13% | -0.60% |
| `sequence-core` | 4.272 | 4.254 | 4.285 | +0.32% | +0.25% |
| `head-projection` | 1.580 | 1.563 | 1.581 | -0.34% | +0.02% |
| device span | 164.991 | 163.458 | 153.338 | -0.93% | **-7.06%** |

| rows 32 | `a8` | `a4e-out` | `a4-out` | norm `a4-out` |
|---|---:|---:|---:|---:|
| **`sequence-output-projection`** | **7.613** | 7.595 | **6.727** | **-11.45%** |
| device span | 62.375 | 62.204 | 61.354 | -1.64% |

A 32-stream generation step, `--decode-streams 32 --decode-prompt 64`, same instrument:

| 32 streams x 1 token | `a8` | `a4e-out` | `a4-out` | norm `a4-out` |
|---|---:|---:|---:|---:|
| `gdn-resident-core` | 44.565 | 44.546 | 44.502 | +0.01% |
| `ffn` | 32.382 | 32.433 | 32.292 | -0.13% |
| `sequence-input-projection` | 13.722 | 13.813 | 13.890 | +1.38% |
| **`sequence-output-projection`** | **11.365** | 11.786 | **9.632** | **-15.11%** |
| device span | 110.347 | 110.868 | 108.591 | **-1.59%** |

### Full model on that build

`tools/batch_compare --only prefill`, a 1024-token document, `--pin-clock`, three rounds with the
arms interleaved and reshuffled inside each round. Every round agrees to within 1.5 tok/s:

| prompt tok/s | `a8` | `a4-out` | `a4-both` |
|---|---:|---:|---:|
| 128-row passes | 770.2 | 824.2 (**+7.0%**) | 941.3 (**+22.2%**) |
| 256-row passes | 811.2 | 845.2 (**+4.2%**) | 952.7 (**+17.4%**) |

Aggregate generation at 32 streams is a null on the product path: 297.3 / 297.5 / 298.8 tok/s for
`a8` / `a4-out` / `a4-both` over ten steps, one round. The phase moves by 15% and the step's device
span by 1.6%, and a generation step has enough else in it that 1.7 ms does not surface at this
resolution. **The output stage's four-bit route is a prompt-ingestion lever, not a generation one.**

### It composes with the input stage almost exactly

Same panel, same samples:

| rows 128 | `a8` | `a4` (input) | `a4-both` |
|---|---:|---:|---:|
| `sequence-input-projection` | 43.393 | 23.471 (-45.7%) | 23.606 (-45.5%) |
| `sequence-output-projection` | 21.707 | 21.711 (+0.4%) | 9.905 (-54.3%) |
| device span | 164.991 | 144.759 (**-12.26%**) | 133.083 (**-19.34%**) |

Neither stage costs the other anything: the phase each one does not touch moves by less than half a
percent, and the spans multiply to within a tenth of a point (0.877 x 0.929 = 0.815 against a
measured 0.807).

### The kernel reproduces its own map, and the quantiser is not the gain

`a4e-out` and `a4-out` produce the **same residual FNV-64 in every sample** —
`4653234265538931311` at 32 rows, `8442032652492096033` at 128, `2166855752027739325` in the
32-stream step — so the four-bit kernel computes the seven-level map to the bit. And `a4e-out`,
which is that map on the deployed instruction, is a time null (-0.87% and -0.03%): **the gain is
the instruction, not the quantiser.**

## Quality

**Resolved, and it is the default now.** The 127-prediction panel below could not rank this arm and
says so. [The horizon instrument](seq-a4-default.md) can: 1024 paired teacher-forced predictions,
stream-clustered, ±0.008 nats. Against the same route's eight-bit operand the output stage alone is
**+0.00458 ± 0.00901** nats and both stages together **+0.00397 ± 0.00940**, against +0.00770 ±
0.00798 for the four-bit input stage that was already the default and +0.01866 ± 0.00687 for the A4
FFN this engine already serves. Moving the default from `a4` to `a4-both` is **-0.00373 ± 0.00934**.
[`docs/seq-out-a4-default.md`](seq-out-a4-default.md) owns that acceptance, the -40.05% phase on the
current build, the +5.2% full model, and the route fallback a default needs that an explicit route
did not.

This is an approximate map and it carries its own evidence. `tools/batch_compare --only quality`
over modes 0 and 19 together, so the reference is the **exact deployed engine** rather than the wide
route's own eight-bit arm:

| 128 rows, 127 predictions | teacher-forced dNLL vs mode 0 | greedy vs mode 0 | mean KL vs mode 0 |
|---|---:|---:|---:|
| mode 19 `a8` | +0.04147 +- 0.02276 | 115/128 | 0.0338 |
| mode 19 `a4-out` | **+0.00357 +- 0.02666** | 115/128 | 0.0393 |
| mode 19 `a4-both` | +0.03147 +- 0.02715 | 114/128 | 0.0403 |

The four-bit output projection is **not resolved** by this instrument: it moves teacher-forced NLL
by less than its own standard error, in the direction of the exact engine rather than away from it,
and leaves greedy agreement where the eight-bit wide route puts it. What it does add is 0.0055 nats
of mean KL on top of the 0.0338 the A4 FFN route already costs — a sixth more divergence than the
route already has, from a stage that is 13% of the pass.

Every `mode 0` cell is exactly zero in every arm, which is the control that the coordinate reaches
only the wide route. And `a4e-out` reproduces `a4-out` in **every** number above, to the digit.

The resolution is the honest limit here, and
[the state horizon panel](gdn-state-horizon.md) says why: 127 predictions give +-0.027 nats, per-
prediction spread and greedy agreement saturate long before the mean does, and only the mean ranks
an approximate arm. An effect below 0.03 nats is bounded, not measured.

## Why it is 54% and not the 24% the issue model predicts

The block loop by `tools/isa_loop_count.py`, output arm, the two selected instantiations:

| | 128 rows, `TT=4` | | 32 rows, `TT=1` | |
|---|---:|---:|---:|---:|
| | `iu8` | `iu4` | `iu8` | `iu4` |
| block instructions | 375 | 371 | 209 | 218 |
| matrix | 32 | 32 | 8 | 8 |
| cycle equivalents (32 / 16 per matrix) | 1367 | 851 | 457 | 338 |
| ratio | | **1.607** | | **1.352** |

Hold the 42% of non-issue time fixed and 1.607 on the issue half predicts 21.8 → 16.6 ms, about
-24%. The measurement is -54%, and the difference is the part
[the earlier document](sequence-output-projection.md#the-arithmetic-that-makes-this-matter) got
wrong. Its table is correct: `iu8` wants 16 B per lane in 32 cycles and `iu4` wants 8 in 16, so both
ask for 0.5 bytes per lane per cycle and a swap changes nothing about the *ratio*. What does not
follow is the conclusion, and the rule that replaces it is:

> **Bytes per cycle is the invariant at the matrix-pipe boundary. A kernel that is over its issue
> model is not at that boundary, and below it the absolute bytes are what it waits on.**

That document measured this phase at 158% of its issue model one section above the argument, which
is the proof that it was below the boundary. There, halving the operand halves the wait *and* the
issue, and the two compound. Afterwards the phase sits at 115% of its own four-bit issue model
(8.6 ms modelled against 9.9 measured) instead of 158%: **it has moved from operand-bound to
issue-bound, and the next lever on it is a different one.**

The row axis says the same thing from the other side. At `TT=1` a block issues eight fragment loads
against 32 at `TT=4`, so the operand term is a quarter the size and the weight stream — which this
change does not touch — is a larger share. The gain falls to -11.5% at 32 rows and -15.1% in a
generation step, near the 1.352 cycle ratio's own prediction rather than above it.

### The registers are free here and are not free one stage over

`tools/kernel_resources.py kernels/sequence_batch.hip`, the selected ownership `W=2` shared,
`RT=1`, `LA=1`:

| `TT` | `iu8` | `iu4` |
|---:|---|---|
| 1 | 78 VGPR, 16 waves | 74, 16 |
| 2 | 95, 16 | 91, 16 |
| 4 | 135, 10 | **180, 8** |
| 8 | 240, 6 | 213, 7 |

At four token tiles the four-bit arm costs 45 registers and a step of occupancy. **The output stage
does not pay for it**: 320 row tiles at `W=2` is 640 waves, eight per SIMD32, so its grid never asks
for the tenth wave. The input matrix has 1024 row tiles and wants 2048 waves, so there the same
instantiation runs at 8 where 10 were available — and still measured 1.84x.

That is recoverable and nobody has measured it yet. The slice scheduling barrier is
[fitted](sequence-fragment-reuse.md) on the eight-bit arm, where dropping it at four token tiles is
worth 5% and costs nothing (135 VGPR either way). On the four-bit arm it is worth 47 registers and
the occupancy step: `SB=2` compiles `TT=4, I4` to **133 VGPR and ten waves**. `--seq-sched 5`
selects it today with no code change.


## Reproducing

```sh
# quality against the exact engine, and the kernel against its own map
tools/run-batch-compare --pin-clock --tag seq-output-a4-quality2 --only quality --modes 0,19 \
  --seq-quant a8,a4e-out,a4-out,a4-both --quality-rows 32,128 --quality-shapes prefill --slots 8
tools/seq_quant_quality.py ../../data/bonsai2/batch-comparison/seq-output-a4-quality2

# device phases, both widths, arms interleaved in one process
tools/run-batch-compare --pin-clock --profile-tool --modes 19 --rows 32,128 --heads 1 --traces 1 \
  --seq-quant 0,3,4 --rounds 2 --out .../seq-output-a4/prefill.json
tools/run-batch-compare --pin-clock --profile-tool --modes 19 --decode-streams 32 \
  --decode-prompt 64 --seq-quant 0,4 --rounds 2 --out .../seq-output-a4/decode.json
tools/seq_out_a4_panel.py .../seq-output-a4/prefill.json --arms a8,a4e-out,a4-out

# full model
tools/run-batch-compare --pin-clock --tag seq-output-a4-prefill --only prefill --modes 19 \
  --seq-quant a8,a4-out,a4-both --prefill-rows 128,256 --prefill-tokens 1024 --context 1280 \
  --rounds 3 --slots 8
tools/run-batch-compare --pin-clock --tag seq-output-a4-decode --only decode --modes 19 \
  --seq-quant a8,a4-out,a4-both --streams 32 --slots 32 --context 256 --gen-steps 24 --rounds 3

# the block loop and the register ladder, no GPU
hipcc --offload-arch=gfx1151 -O3 -std=c++17 -Isrc -Ikernels --cuda-device-only -S \
  kernels/sequence_batch.hip -o /tmp/seq.s
tools/isa_loop_count.py /tmp/seq.s projectILi4ELb1ELb0ELi2ELb1ELi1ELi1ELi1ELb1EE
tools/kernel_resources.py kernels/sequence_batch.hip project
```

Raw: `../../data/bonsai2/batch-comparison/seq-output-a4{,-quality,-quality2,-prefill,-decode}/`.

## What it does not do, and what is open

- **It is a wide-route coordinate.** Mode 19 or 20 and a pass wider than `RMAX`; `sequence_project`
  refuses rather than reading an eight-bit operand through the four-bit instruction. Giving the
  per-slice half-layer a nibble store would remove that, at the cost of a template axis on
  `k_forward_rows` — which is the kernel whose register ceiling sets the whole model's cooperative
  grid, so it is not obviously worth it.
- **`a8` remains the default** and every deployed instantiation is unchanged.
- **The barrier fit.** `TT=4, I4` is 180 VGPR and eight waves with the deployed slice-barrier
  setting and 133 and ten with `SB=2`. The output stage cannot use the tenth wave; the input stage
  can and is not getting it. `--seq-sched 5` is the arm, and nobody has run it against `--seq-quant
  a4` yet.
- **What binds it now.** The phase has moved to 115% of its four-bit issue model, so the operand
  path is no longer the first thing to attack there. The remaining 15% and the shape's eight waves
  are what is left, and 320 row tiles is where the eight waves come from.

# The dot4/WMMA crossover, and the grid it was hiding

`ph_matvec_auto` sends a pass of five or more rows to the WMMA matvec body and anything narrower to
the scalar-fed dot4 body. The constant that decides it, `MV_DOT4_MAX`, was written as 4 when the
WMMA body landed in `1ce5238` and has never been set since, in any build, by anyone. It turns out
to have been wrong for one of the two weight representations, and the reason it was wrong is not
the reason anyone would guess.

This matters where it is measured: the drafted single-stream step is what `bonsai-halo.service`
runs, and [five sixths of it](serving-decode.md) is one eight-row pass of the target model.

## Result

| | before | after |
|---|---:|---:|
| drafted verify pass, per step | 53.5 ms | **49.9 ms** |
| drafted generation | 49.1 tok/s | **51.6 tok/s** |
| single-token generation | 33.66 tok/s | 33.67 tok/s |
| `k_forward_rows<8>` | 217 VGPR, 6 waves/SIMD32 | 136 VGPR, 10 waves/SIMD32 |
| deployed cooperative grid | 60 workgroups | 100 |

**Bit-identical, and checked three ways.** All 248,320 full-vocabulary logit floats of a fixed
prompt compare byte for byte equal. The greedy token stream of every drafted run digests to
`16657175983209876683` in all 22 samples. And the drafted counts — 133 drafted, 42 accepted — are
the same in every one: a lossless drafter verifies each token against the target, so corrupt target
arithmetic still produces correct text and only changes how many drafts survive. Identical
acceptance means the target's argmax agreed at all 133 positions.

That identity is structural, not lucky. Both bodies accumulate one exact `int32` per 128-wide block
from the same weights in the same K order, subtract the same activation sum, and fold it into the
same running float with the same `fmaf(acc, wscale * xscale, y)` in the same block order; the
cross-wave reduction then adds wave 0 through wave 7 in that order in both. Nothing is reassociated
and no operand narrows, so this is a schedule choice and not a numerical one.

Raw samples, binaries and hashes:
[`batch-comparison/matvec-crossover/`](../../../data/bonsai2/batch-comparison/matvec-crossover/README.md).

## The instruction count says dot4 should win, and the instruction count is wrong

Per 32-row, 128-K tile block, one wave:

| | dot4 body | WMMA body |
|---|---|---|
| expansion | one peel to `tr[32]`, shared by every row | the same peel |
| matrix | 32 `sudot4` per row — 256 at eight rows | 16 `wmma_iu8`, about 512 cycles |
| activation operand | 32 dwords per row through the **scalar** cache | 8 `int4` VMEM fragment loads |
| columns used | 8 of 8 | **8 of 16** |

On RDNA3.5 the WMMA instruction runs on the VALU, not on a separate matrix core, and
`v_dot4_i32_iu8` retires 4 MACs on each of 32 lanes — 128 int8 MACs per cycle, exactly what
`wmma_i32_16x16x16_iu8` averages over its 32 cycles. There is no arithmetic advantage to the matrix
form at all. Its advantages are operand reuse and sixteen columns, and a deployed pass has eight
rows to put in those columns, so half of that second advantage is thrown away.

So the model predicts the eight-row matvec roughly halves. It does not. **The whole eight-row pass
moved 4.5%**, and the phase it lives in is 79.4% of the pass.

The reason is the one `5a3dbb4e` supplied from the other side of the same dispatch: the WMMA body
is nowhere near its issue model either. Its compiled slot count prices a whole deployed pass at
about 23 ms against 41.9 ms measured, so roughly 1.8x of that phase is weight-load stall, and
deleting issue slots cannot touch stall. Both bodies read the same 5.9 GB and both wait for it.

## What actually moved it: the body's register footprint, not its instructions

Force both bodies onto the same 60-workgroup grid and **the dot4 body loses**, 62.8 ms against 53.0.
It is not faster. It is *smaller*: 136 VGPRs against 217, which takes `k_forward_rows<8>` from six
resident waves per SIMD32 to ten, and the deployed cooperative grid — the minimum over the TT 1, 2,
4 and 8 instantiations — from 60 workgroups to 100.

The eight-row pass wants that grid enormously:

| deployed grid | 60 | 72 | 84 | 100 |
|---|---:|---:|---:|---:|
| drafted verify pass | 62.8 ms | 55.6 | 52.6 | **49.8** |

A 21% range. That is the whole result: an extra four resident waves per SIMD32 hide the weight-load
stall that neither instruction schedule could reach.

It is a property of this pass and not of weight-streaming cooperative kernels in general. The same
sweep on the drafter, whose whole job is to stream 1.23 GB once per step, is flat from 40
workgroups upward: [`drafter-occupancy.md`](drafter-occupancy.md).

## The same grid at one row goes the other way, so it stops being one number

[Single-stream decode](single-stream.md) measured the identical knob at one row and found it worth
±0.4% — 48 workgroups beat 60 by 0.43%, 40 lost 0.81% — and left the default alone because that is
smaller than the drift between two panels. It also named the gap this document closes: *"48 is
untested at eight rows, where the same grid serves a different kernel with different unit counts."*

At eight rows the same knob is worth 21%, and in the opposite direction. One row is bandwidth
shaped, and an extra workgroup adds a grid-sync participant and a shorter unit run without adding
any bandwidth; eight rows are stall shaped, and an extra workgroup is another four waves of latency
hiding.

Letting the raised ceiling reach both costs single-token generation 1.6% (33.66 to 33.12 tok/s,
four paired rounds, non-overlapping). So the grid is now chosen by row count. Rows 1 to 4 keep
`ROWS_GRID_NARROW`, which is exactly the 60 they have been running on — the narrow value belongs to
`docs/single-stream.md` and its in-process paired instrument, not to this panel, and that constant
is the one place to change when it concludes. Rows 5 to 8 take the whole grid.

Single-token generation after the split: 33.57 and 33.76 tok/s against a control's 33.73 and 33.73.
A null, by construction rather than by luck — those passes launch on the same grid they always did.

## The crossover is per representation, and the drafter wants the opposite arm

Moving every eight-row matvec to dot4 makes the DFlash2 drafter **5% slower**: 10.7 ms per block to
11.2. The drafter's weights are Q8 tiles, and that changes the shape of the body completely. A Q8
tile block is 4160 bytes against a ternary tile block's 896 and carries no peel at all, so a wave
demands about 6.8 bytes per cycle where 242 GB/s across 80 SIMD32s can supply roughly 1.3. The
drafter's matvec is five times oversupplied with issue slots and starved of bytes; its schedule is
about hiding load latency, and the WMMA body's wider operand loads do that better.

So `ph_matvec_auto` takes its row limit from the weight representation — 8 for ternary HALO tiles,
4 for Q8 — and the drafted step gets both halves: the target's verify at 49.9 ms and the drafter's
block still at 10.9.

| arm | verify ms | draft ms | drafted tok/s |
|---|---:|---:|---:|
| `MV_DOT4_MAX` 4 / 4 (before) | 53.5 | 10.7 | 49.1 |
| 8 / 8, both representations | 49.9 | 11.2 | 51.2 |
| **8 / 4, by representation** | **49.9** | **10.9** | **51.6** |

Medians over 22 drafted samples; arms were run in both orders after the first panels put the
control first every time and flattered it by one warm-up run.

## What this leaves for the next engineer

- **The eight-row pass is stall-bound, and occupancy is the lever that reaches stall.** 136 VGPRs
  still buys only ten waves. The next register the WMMA or dot4 body gives back is worth more than
  the next instruction either of them deletes. `5a3dbb4e`'s weight-stream lookahead in `mvw_rows`
  attacks the same stall from inside the body and composes with this.
- **100 is a ceiling, not an optimum.** Every measured point up to it improved, so the eight-row
  pass has never been measured on a grid it did not want more of. The grid sweep stops where
  occupancy stops.
- **Nothing here tested 5, 6 or 7 rows**, which take the same TT = 8 instantiation and the same new
  grid. A three-row MTP verify takes the narrow grid and was not measured either.
- **`MV_DOT4_MAX_Q8` was set from one drafter.** It is a statement about Q8 tiles at eight rows on
  this device, measured on DFlash2's block pass only.
- **The drafter's bytes were the lever and they have been taken.**
  [Four-bit drafter weights](drafter-q4.md) removed 0.898 GB of the step and took the `draft` phase
  from 11.35 to 7.65 ms, +10.4% on drafted generation, greedy digest equal in every case. The same
  document measures `MV_DOT4_MAX_Q4` instead of inheriting it, and the answer sharpens the register
  result above rather than repeating it: the Q4 dot4 arm holds `k_dflash` in **145 VGPRs and nine
  waves per SIMD32 against WMMA's 239 and six, and is 9% slower**. Occupancy is worth what the
  phase is short of; a phase short of bytes does not spend it.

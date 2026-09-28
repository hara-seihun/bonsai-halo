# The eight-row kernel's grid is already the best one it can reach (bounded negative)

Raw: [`batch-comparison/rows-lds-occupancy/`](../../../data/bonsai2/batch-comparison/rows-lds-occupancy/README.md).
Landed and installed on canonical `cb9204c`, executable `978acfc617de31360e33c623`; installed
acceptance `--bench --dflash`, 48 tokens, verify **42.241 ms**, digest `14250041415274144751` and
112 drafted / 32 accepted — the exact integers the binary before it produced. Service restored and
answering a chat completion on the new build.

[The grid occupancy result](rows-grid-occupancy.md) won +22.8% prompt and +20.0% generation by
taking the eight-row persistent kernel from 80 workgroups to 100, and it left a sentence that has
been read ever since as a ceiling waiting to be lifted: "13 buys 96 and the six that LDS caps this
kernel at". This iteration lifted that cap, measured every grid from 80 to 160 on one binary in one
process, and **the deployed 100 is the top of the curve.** The direction is closed, with the
register ladder enumerated so nobody has to re-derive which points are even reachable.

Two changes ship out of it, both bit-identical and both measured null in time. They are here
because the *instrument* is worth keeping: without them the grid cannot be moved above 100 at all,
and a future register cut would silently buy nothing and be blamed on registers.

## What set the budget, from four compiles and no GPU

`k_forward_rows` is one kernel whose register allocation is the maximum over every phase it runs,
and `HALO_ROWS_DROP` prices a phase by deleting it:

| build | waves/SIMD32 | VGPR | scratch |
|---|---:|---:|---:|
| packed state, `waves_per_eu(12)` | 12 | 120 | 24 B |
| attention dropped | 12 | 120 | 18 B |
| the GDN recurrence dropped | 12 | 120 | 5 B |
| **both dropped** | **16** | **83** | **0** |

**Attention and the recurrence each independently demand 120 registers; the matvec and prep phases
that are four fifths of the pass need 83 and spill nothing.** At `waves_per_eu(13)` the whole kernel
fits in 96 with 32 B/lane of scratch, and the core still spills nothing — every spilled byte belongs
to the two phases that were never the budget's customer.

## The shared block was the second cap, and it held a kilobyte nothing writes

`ph_matvec` stages its cross-wave drain in `[8][TT][32]`, but the body writes it under `wave > 0`
and wave 0 keeps its own partials in registers, so `32 * TT` floats of the array were never written
by anything. At `TT = 8` that is 1 kB of a 9.5 kB block, and the block is what the occupancy API
divides 65536 by: 9508 bytes is six workgroups per WGP, 7204 is nine.

`rows_lds_floats(TT)` gives this kernel the same treatment `slice_lds_floats` already gives
`k_ffn_slice`: the maximum over the phases a forward pass actually runs, not over every phase in
`phases.hpp` plus the drafter's top-k list. The drain is rebased to `(wave - 1)`, which is the form
`ph_matvec_w` has always used. The kernel's shared block goes **9508 -> 7204 bytes** and the same
partials are summed in the same left-to-right order at a different address.

## The grid ladder, enumerated

`coop_grid` is `min(occupancy API, register granule blocks) * 20`, and the API's own ladder for this
kernel is 96 VGPR -> six blocks, 97..135 -> five, 136 and up -> four. What the compiler will
actually produce for this body, with the shared block at 7204 bytes:

| `waves_per_eu` ask | VGPR | scratch | blocks/WGP | grid |
|---|---:|---:|---:|---:|
| 13 and up | 96 | 32 B | 8 | **160** |
| 11, 12 (deployed) | 120 | 14 B | 5 | **100** |
| 10 | 143 | 0 | 4 | 80 |
| none | 148 | 0 | 4 | 80 |

**There is no reachable point between 120 and 143**, so the "free headroom to 135 registers" the API
ladder seems to offer is not addressable by asking. Four grids exist for this kernel: 80, 100, 160,
and whatever a pinned `--grid-rows` picks below the build's own base.

## The curve, one binary, one process, one pinned clock

The 96-register build has base grid 160, so `--grid-rows` can pin any point from 64 to 160 — the
first time this axis has been measurable above 100. Arms are a palindrome, so every point is
measured early and late against the drift of the process. Milliseconds of one eight-row verify pass,
the shape `bonsai-halo.service` generates on:

| grid | workgroups/WGP | verify ms (early, late) | mean | against 100 |
|---:|---:|---|---:|---:|
| 80 | 4 | 57.34, 54.10 | 55.72 | **+24.0%** |
| 90 | 4.5 | 45.74, 47.87 | 46.80 | +5.6% |
| **100** | **5** | 46.43, 43.40 / 43.44, 45.20 | **44.62** | — |
| 110 | 5.5 | 42.68, 43.78 | 43.23 | -3.1% |
| 120 | 6 | 48.53, 46.14 | 47.34 | +6.1% |
| 140 | 7 | 48.99, 45.43 | 47.21 | +5.8% |
| 160 | 8 | 51.27 (+54.77, 54.33 on a second panel) | 51.27 | **+14.9%** |

**Every grid is bit-identical.** The greedy digest is `1448224317935851035`,
`12231090205376880997`, `9596179497239680963` and `14250041415274144751` on the four prompts
measured, equal at every grid in every panel, which also confirms the `ROWS_GRID_ORDER_MAX = 160`
bound that keeps a `KS = 2` matvec's two parts in order: 160 is exactly at it and holds.

The 80 -> 100 step reproduces [the published grid result](rows-grid-occupancy.md) from the other
side (-19.4% here against its +20.0%), which is what calibrates the instrument.

### 110 looks like a win and is not one

110 is 3.1% below 100 *within the 96-register build*, and that build starts behind: alternating it
against canonical on one lock, `A100 44.04 ms, B110 44.65, B100 45.07`. The register relaxation
costs about 2.3% and the grid gives 3.1% back, so the pair lands inside the noise of the deployed
configuration. **There is no shipping win on this axis**, and `HALO_ROWS_WPE` stays at 12.

## What it is not: the barrier

The obvious mechanism for a grid getting worse is the device-wide barrier, because a verify pass
runs about 480 of them and each waits for every workgroup. `bench/coop_cost --mode 0` prices it with
no model: **862.9 ns at grid 64, 1139.4 at 100, 1347.2 at 120, 1759.6 at 160**, near-linear at about
10.5 ns per workgroup. Over 480 barriers that is 0.55 ms of a 44.6 ms pass at grid 100 and 0.84 ms
at 160 — **0.30 ms of the 6.7 ms the grid change costs, under 5%.** The barrier is not what makes a
bigger grid lose.

What is left is the law [the served verify pass](drafted-serve-step.md) measured from the other
side: a phase's byte rate rises with its grid rounds, and a bigger grid gives every phase *fewer*
rounds. `mv_gate_up`'s 1088 units are 11 rounds at grid 100 and 7 at 160; `mv_down`'s 320 are 3.2
and 2. Interpolating that probe's own curve predicts about 4-5% on the matvec phases, so it accounts
for roughly a third of the 14.9%. The rest is unexplained and is the open question below.

## The second arm: the perm gather in the deployed matvec body (null)

`9681dcc7` landed [`kernels/halo_expand.hpp`](../kernels/halo_expand.hpp) and the perm gather in the
FFN slice for -1.66% of that phase, and named the eight-row verify body as the consumer nobody had
taken. `mv_expand` and `mv_rows_deployed` now call the header instead of carrying their own copy of
the peel: 168 operations per 128-trit block instead of 232, the same 32 operand dwords byte for byte
(`make kernels/head_op_check` proves it against `halo::decode_block` over 3 uniform, 256 byte-sweep
and 4000 random blocks), the same 120 VGPRs and the same 14 B of scratch in `k_forward_rows<8>`.

Alternating binaries under one lock, `-n 48`: **verify 42.628 ms against 42.617**, digests equal.
**A null.** That is the fourth instruction cut to read null on this body — after the two-trit
palette (-12%), the A4 block (-9.4%) and `mv_rows_t`'s -10.5% — and it is the strongest evidence yet
that the eight-row block loop is not issue-bound: a 27.6% cut of its decode arithmetic, with no
register cost and no extra bytes, moves nothing. It ships anyway, because it deletes the duplicate
expansion and puts both ternary consumers on one map, so the next improvement to that map reaches
the verify pass for free.

## For the next engineer

- **Do not spend registers on occupancy in this kernel.** 96 registers is reachable, its grid is
  reachable, and every point above 100 workgroups loses. A phase body that costs registers while
  staying at or below 120 is free; one that crosses 135 falls to grid 80 and loses a fifth.
  [The parked vector-path matvec arm](mv-scalar-waits.md) that measured +8.0% and spills 370
  registers is still not affordable, but its obstacle is the 120-register wall, not a missing grid.
- **The unexplained two thirds of the grid 160 loss is the open question.** Barriers are ruled out
  and the round-count law explains about a third. The remaining suspect is the memory system seeing
  60% more concurrent tile-run streams: 160 workgroups walk 160 runs at a stride of `NB * 896`,
  where 100 walk 100. `bench/wstream` prices exactly that geometry with no model and has never been
  pointed at the HALO tile image, which `a607c971` closed with "the persistent kernel HALO tile
  image is CLOSED with no panel".
- **The shared block has room now.** 7204 bytes is nine workgroups per WGP of LDS headroom against a
  register allocation that allows five. Anything that wants LDS in the forward pass — an LDS B stage
  for the matvec operand, a wider attention score array — can have about 2.3 kB before it costs a
  block, and `rows_lds_floats` is where to declare it.

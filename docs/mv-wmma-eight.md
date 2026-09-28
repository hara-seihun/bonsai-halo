# The matrix instruction loses the eight-row matvec by 15%, and now it has been asked at the same grid

[`docs/matvec-crossover.md`](matvec-crossover.md) is the reason the deployed verify pass runs the
scalar-fed `dot4` body, and it left one sentence that has been quoted ever since: *"Force both
bodies onto the same 60-workgroup grid and the dot4 body loses, 62.8 ms against 53.0"*, with `dot4`
shipping anyway because it holds `k_forward_rows<8>` in 136 VGPR where `ph_matvec_w` holds it in
217 — grid 100 against grid 60, and the eight-row pass wants the grid more than it wants the body.

**So the comparison the constant `MV_DOT4_MAX = 8` rests on has only ever been made on a grid
neither body would choose.** This iteration built a matrix body that fits the deployed grid and made
the comparison at grid 100 on both sides. **The matrix body loses by 15.6%, bit-identically**, and
the arithmetic that says why is a number this lane got wrong twice in one evening.

Held and released: `kernels/mv_wmma8.hpp` (new), `kernels/mv8_expand_check.cpp` (new),
`ph_matvec_pk` in `kernels/phases.hpp`, `tools/mv8-ab.sh`, `bench/mv8-prompts.txt`.
**Nothing in the deployed runtime moved**: `HALO_MV_WMMA8` defaults to 0, and the default build's
`k_forward_rows<8, 0, 0, 0, false, 12>` is the 120 VGPR / 8 B scratch / 7204 B block it was before.
Raw panels, the register census and the build hashes are in
[`batch-comparison/mv-wmma-eight/`](../../../data/bonsai2/batch-comparison/mv-wmma-eight/README.md).

## The body

`kernels/mv_wmma8.hpp` is `ph_matvec_w`'s map at `COLS = 8, GROUPS = 1, Q8 = false` with the live
set of a block cut where it could be cut:

- **The expansion is incremental.** `src/halo_format.h` fixes the trit order so that source dword
  `d` carries operand dwords `tr[4d .. 4d+3]` — exactly the K slice `kb = d` consumes — plus one
  fifth-trit dword `tr[24+d]`. Expanding a dword straight into the matrix instruction that eats it
  holds four operand dwords and six fifths instead of thirty-two. The K slices are taken in the
  order `0,1,2,3,6,4,5,7`, because slice 6 is built from the fifths of dwords 0..3 and retiring it
  early frees four of them; slices meet in one `int32` accumulator, so the order is free.
- **One accumulator pair, one column group.** A 32-row output tile is two matrix fragments (`A` and
  its half swap), so `C1`/`C2` are the tile, not a latency device — `bench/wmma_chain` measures the
  instruction flat in chain depth.
- **Four scale floats live at a time** instead of sixteen, read from this wave's own LDS table at
  the point of use.

`kernels/mv8_expand_check` holds the incremental expansion against `hx_expand_perm` and against
`decode_block`: **20,000 blocks, 160,000 K slices, every operand dword equal and every operand byte
equal to the packed trit**, on the host in a second with no GPU.

The register census, `tools/kernel_resources.py -D HALO_ROWS_PROBE=8 -D HALO_ROWS_PROBE_PK=0`, VGPR
and scratch bytes per lane:

| budget | dot4 (ships) | matrix body | matrix body, spread drain |
|---|---|---|---|
| `OCC 1`, the allocator's | 149, no spill | **193**, no spill | 209, no spill |
| `OCC 10` = 144 VGPR | 144, no spill | 144, 27 B | 144, 52 B |
| `OCC 12` = 120 VGPR (deployed) | 120, 8 B | 120, 60 B | 120, 151 B |

`ph_matvec_w` is 217 VGPR in the same kernel, which is three blocks per WGP and grid 60. **193 is
four blocks and 144 is five**, so the lean body reaches the rung the deployed body runs on, and both
arms of every panel below launched on **grid 100** — read out of `coop_grid` itself with
`HALO_ROWS_GRID=1`, `numRegs 120 (granule 120, 12 waves, 6 blocks/WGP), lds 7204/8228, api 5 ->
grid 100` in both. The shared block grows 7204 → 8228 bytes for the per-wave scale table and does
not cost a block: `bbfca1db` measured a WGP's LDS at 128 kB, and the occupancy API's count is 5
either way.

## What it measures

Two binaries from one tree, one `-D` apart, arms alternating inside one `tools/run-batch-compare
--pin-clock` hold, `--bench --dflash` over three prompts, 24 tokens each. The drafter's own pass is
the in-panel control: it runs `k_dflash`, which this change cannot reach.

| arm | verify ms/step | drafted tok/s | draft ms/step (control) |
|---|---|---|---|
| dot4, 120 VGPR (ships) | **40.297 / 40.436 / 40.564** | 46.82 / 86.62 / 64.32 | 6.296 / 5.738 / 6.072 |
| matrix body, 144 VGPR | 46.427 / 46.695 / 47.194 | 41.38 / 76.21 / 56.24 | 6.295 / 5.788 / 6.142 |
| matrix body, 144 VGPR, spread drain | 48.954 / 49.312 / 49.349 | 39.49 / 72.60 / 54.09 | 6.288 / 5.783 / 6.111 |
| matrix body, 120 VGPR | 50.823 / 50.419 / 51.287 | 38.04 / 70.92 / 52.07 | 6.518 / 5.976 / 6.318 |

**+15.6% on the verify pass and −11.4% on drafted generation**, at grid 100 on both sides, with the
drafter's phase inside 1% in every arm. Both panels ran at 2812 and 2827 MHz p50 (97% of nominal),
the quietest windows this lane has had all evening.

**Bit-identical, and the drafter's integers are the second witness.** Greedy digests
`2279305512340435356`, `12301665195185690968`, `2622622355779879709` and the exact counts
`77/12`, `42/21`, `56/15` drafted/accepted are equal in **every arm of every panel**, including the
two register budgets and both drain forms. A lossless drafter verifies every token against the
target, so equal acceptance over 175 drafts is the target's argmax agreeing at all 175 positions.
The map is the one the lane already argued for: same image, same blocks, same K order, one exact
`int32` per block with the same `-sum(x)` seed, and the same `fmaf(acc, wscale * xscale, y)` in the
same block and wave order.

### The phase table says the loss is ordered by blocks per wave

Per instance, `HALO_PROFILE=1`, one panel (the 120 VGPR arm, so the spill is in it):

| phase | blocks/wave | dot4 | matrix | ratio |
|---|---:|---:|---:|---:|
| `mv_ssm_out` | 3 | 58.6 | 90.1 | **1.538** |
| `mv_o` | 3 | 59.2 | 90.4 | **1.527** |
| `mv_down` | 8/9 | 144.4 | 175.5 | 1.215 |
| `mv_gate_up` | 5 | 235.5 | 276.3 | 1.173 |
| `mv_qkv` | 5 | 112.6 | 125.9 | 1.118 |
| `mv_qkv_z` | 5 | 123.0 | 135.4 | 1.101 |
| `mv_lm_head` | 5 | 1417.2 | 1550.4 | 1.094 |
| `gdn`, `attn`, `prep_norm` (control) | — | 52.2 / 37.8 / 11.0 | 49.5 / 35.6 / 10.4 | 0.95 |

The three-block phases lose half again as much as the five-block ones, which is the per-unit
overhead of this body divided by the weight bytes a unit carries. **The drain is not that overhead,
and the arm that proves it is in the tree**: `HALO_MV8_SPREAD=1` gives each accumulator element to
the wave that computed it — `docs/matvec-drain.md`'s change in this body's coordinate, where it
should be worth twice as much because a 32-row tile is two fragments — and it is **+5.3% worse**,
because 16 more registers of live `y` at 144 VGPR is 52 bytes of scratch a lane against 27. In this
kernel a register is worth more than a barrier.

## Why, and the constant it confirms

**`v_wmma_i32_16x16x16_iu8` costs 34.3 cycles per SIMD32** (`bench/mvblock`, `d9b78400`, after
correcting `multiProcessorCount` — the part is 80 SIMD32, not 40, and the first published figure of
17.14 was half). Per 128-wide block, one 32-row tile:

| | matrix instructions | issue cycles of arithmetic | activation rows served |
|---|---:|---:|---:|
| matrix body | 16 `wmma_iu8` | 549 | 8 of the 16 columns |
| dot4 body | 256 `v_dot4_iu8` | 256 | 8 |

**At eight rows the matrix instruction costs 2.1x the arithmetic issue of the scalar-fed body,
because half of its sixteen columns carry a duplicate activation row.** The dot4 body pays for that
with its radix-3 expansion and its scalar activation batches — 852 block-loop slots against this
body's 255 instructions plus 16 matrix instructions, about 789 cycle-equivalents — so the corrected
model puts the two bodies within 8% of each other, and the measurement puts the matrix body 15.6%
behind. What is left over is what a block census cannot see: the A-operand hazard
(`d9b78400` prices a matrix instruction whose A dword was written by the `v_perm` in front of it at
about three cycles; `HALO_MV8_PIPE=1` builds the operand a slice ahead and is in the tree unmeasured),
the B fragment gather (eight lanes at a 5 to 6 kB row stride per load, where the dot4 body's
activations are one scalar-cache batch), and two more barriers per unit.

**So `MV_DOT4_MAX = 8` is right, and it is right for a different reason than the document that set
it.** The crossover was measured on a grid neither body chooses, with a register difference of 81
folded into it. At equal grid, equal co-residency and a body built for the budget, the answer is the
same and the margin is 15.6%.

### The line this draws, which is worth more than the arm

The matrix body's 549 cycles serve **sixteen** activation rows for exactly what they cost for eight:
`COLS = 16` changes no matrix instruction, only which rows the columns carry. The dot4 body's 256
cycles double to 512 at sixteen rows while its ~600 cycles of expansion and operand work stay.

| rows | matrix body | dot4 body | ratio |
|---:|---:|---:|---:|
| 8 | 789 | 852 (measured: 15.6% better) | ~1.0 |
| 16 | 789 | ~1108 | **1.40 for the matrix body** |

**The eight-row verify pass is the shape that makes the matrix instruction lose.** A verify pass
carrying sixteen draft rows would run its matvecs on the same weight stream, in the same number of
matrix instructions, for twice the tokens — and it is the one change that flips this result. That is
a drafter and `RMAX` question, not a matvec one: `kernels/halo_rows.hip` is instantiated at 1, 2, 4
and 8 rows, the register peak of the attention unit scales with `TT`, and the DFlash2 drafter
currently offers seven tokens and a root. The arithmetic above is what it would be worth.

## Reproducing

```sh
make DEFS='-DHALO_MV_WMMA8=1 -DHALO_ROWS_WPE=10' bonsai-halo    # the arm
make bonsai-halo                                                # the control
make kernels/mv8_expand_check && kernels/mv8_expand_check       # the operand map, no GPU
tools/kernel_resources.py -D HALO_ROWS_PROBE=8 -D HALO_ROWS_PROBE_PK=0 -D HALO_ROWS_PROBE_OCC=10 \
    -D HALO_MV_WMMA8=1 kernels/halo_rows.hip k_forward_rows     # the census, nine seconds, no GPU
tools/run-batch-compare --pin-clock --exec tools/mv8-ab.sh 'ctl wmma8' OUTDIR
```

`tools/mv8-ab.sh` expects `bonsai-halo-ARM` binaries beside the tree and defaults to
`bench/mv8-prompts.txt` and 24 tokens, which is one bounded agent call per two arms.

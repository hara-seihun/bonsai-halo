# The served drafted step, phase by phase, and the retiling its own probe asked for

[`docs/serving-decode.md`](serving-decode.md) was the only map of the step `bonsai-halo.service`
generates on, and it was taken six builds ago. [The drafter's own segments](drafted-step.md) measured
the other half. This is the verify pass — 87% of a drafted step — traced on the current tree, the
scaffolding surface that explains why its seven matvec phases read 113 to 203 GB/s through one body,
and the change that surface asks for, built and rejected.

Raw: [`batch-comparison/drafted-serve-step/`](../../../data/bonsai2/batch-comparison/drafted-serve-step/).

> **Refreshed, and its first open question is now three axes narrower.**
> [The ceiling panel](ternary-coordinate-ceiling.md) re-took this map on `f7b680d` at 97% of nominal
> clock and prices what this document could not: **9.0% of the pass is the partial last round**
> (`ceil(units / grid)` against `units / grid`, 20% on every 320-unit projection), no grid below 100
> is better, and the residue is 3.2 us per *identical* unit between an eleven-round phase and a
> seventy-eight-round one.

## Where a verify pass goes

`--bench --dflash`, ten prompts, 40 tokens each, `HALO_PROFILE=1`, `--pin-clock`, canonical
`66263d0`, binary `c2f1c7023c9b1b66387e7601`. Medians over four traced passes; the byte column is the
HALO tile arithmetic (`N * K * 896 / 4096`), which closes against the 5.878 GB cache file.

| phase | ms | share | launches | weight bytes | GB/s | units per phase |
|---|---:|---:|---:|---:|---:|---:|
| `mv_gate_up` | 16.70 | 37.3% | 64 | 2.497 GB | 149.5 | 1088 |
| `mv_down` | 9.68 | 21.6% | 64 | 1.248 GB | 129.0 | 320 |
| `mv_qkv_z` | 5.93 | 13.2% | 48 | 0.881 GB | 148.6 | 512 |
| `mv_ssm_out` | 2.86 | 6.4% | 48 | 0.330 GB | 115.5 | 320 |
| `gdn` | 2.13 | 4.8% | 48 | | | |
| `mv_qkv` | 1.78 | 4.0% | 16 | 0.238 GB | 134.0 | 416 |
| **`mv_lm_head`** | **1.37** | 3.1% | 1 | 0.278 GB | **203.3** | **7760** |
| `prep_norm` | 1.24 | 2.8% | 129 | | | |
| `mv_o` | 0.97 | 2.2% | 16 | 0.110 GB | 113.6 | 320 |
| `attn`, `gdn_pre`, `prep_*`, `argmax`, `embed` | 2.14 | 4.8% | 213 | | | |
| **total** | **44.76** | | **~580** | **5.582 GB** | **142** | |

**Seven phases run the same `mv_rows_t<PW, 8>` body over the same image format and their byte rates
differ by 1.8x.** The vocabulary head is the fast one and it is the one with 7760 units; the three
5120-row projections are the slow ones and they have 320. Nothing else orders that list: `mv_down`
and `mv_lm_head` differ by 57% with the same row count, the same operand map and the same clock.

## The scaffolding surface: at eight rows a phase's length sets its rate

`bench/coop_cost` runs the engine's own `grid_sync`, `next_unit` and `mv_rows_t` with no model
behind them. It gains `--tt`, the row count the unit body carries, and `--exact-units`, which deals
the count asked for instead of rounding it down to whole grid rounds. GB/s of distinct weight bytes,
grid 100 (the eight-row pass's own), 32 phases per launch, best of four:

| units | rounds | tt 8, pw 3 | tt 8, pw 5 | tt 8, pw 8 | tt 1, pw 3 | tt 1, pw 5 | tt 1, pw 8 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 1 | 117.8 | 141.9 | 137.6 | 167.8 | 187.4 | 185.5 |
| 300 | 3 | 155.6 | 153.9 | 145.1 | 203.0 | 210.6 | 209.0 |
| 400 | 4 | 162.9 | 162.4 | 145.5 | 210.2 | 218.6 | 216.6 |
| 512 | 6 | 173.5 | 170.6 | 148.7 | 217.0 | 220.1 | 214.4 |
| 1088 | 11 | 191.9 | 184.3 | 160.1 | 220.9 | 225.4 | 222.0 |
| 2000 | 20 | 198.4 | 194.1 | 177.3 | 229.2 | 228.7 | 224.5 |
| 7760 | 78 | 206.1 | 204.2 | 198.1 | 232.3 | 230.3 | 225.1 |

**A phase needs about twenty grid rounds to reach its steady byte rate, and the eight-row pass gives
its phases three to eleven.** At four rounds the same code moves 162 GB/s where it moves 204 at
seventy-eight. One row ramps too, and far less: 218.6 against 230.3 at the same two points.

It is the unit *count*, not the bytes and not the grid. At equal bytes per phase (14.3 MB, tt 8):

| shape | GB/s |
|---|---:|
| pw 3 x 667 units | **176.9** |
| pw 5 x 400 units | 159.7 |
| pw 8 x 250 units | 138.6 |
| pw 20 x 100 units | 141.4 |
| pw 40 x 50 units | 115.5 |

**Many small units beat few large ones by 1.53x for the same work**, which is the opposite of what
the one-row shape wants — [the phase-length probe](matvec-phase-length.md) found `pw = 3` costing 30%
against `pw = 5` at one row, and that is why the deployed splits leave 320-unit phases. Sweeping the
grid at a fixed 320-unit phase (40, 50, 60, 80, 100) moves the rate 149 to 165 GB/s with no trend:
the grid is not the variable.

The engine's own phase table sits on this curve. `mv_lm_head` at 7760 units measures 203.3 GB/s
against the probe's 204.2 — the same number to half a percent.

## What the surface asks for, and what it costs

Every matvec phase here already has a retiling knob. `k_forward_rows` splits a 40-block matvec into
`KS40` K parts, a 48-block one into `KSV` and the 136-block one into `KSFF`, and the single-token
path already uses 5, 6 and 17 where the multi-row path uses 1, 2 and 2 — for the reason its own
comment gives, round quantisation. Taking the single-token values at eight rows turns 320-unit phases
into 960 to 2720 and 1088 into 5440. The probe prices those shapes at **182 to 217 GB/s**.

Built (`mv-retile-arm.patch` beside the raw samples), and it is **+31% on the verify pass**.

Two binaries from one tree, one constant apart, arms alternating inside one lock hold, four prompts,
`--pin-clock`:

| phase | KS 1/2/2 (ships) | KS 5/6/17 | ratio |
|---|---:|---:|---:|
| `mv_gate_up` | 16.96 | 24.60 | **1.450** |
| `mv_down` | 9.87 | 12.06 | 1.221 |
| `mv_qkv_z` | 6.09 | 8.89 | **1.460** |
| `mv_ssm_out` | 2.94 | 3.38 | 1.152 |
| `mv_qkv` | 1.84 | 2.62 | 1.423 |
| `mv_o` | 1.00 | 1.12 | 1.120 |
| `prep_norm` (carries the zeroing) | 1.28 | 1.61 | 1.260 |
| **`mv_lm_head` (KS = 1 in both)** | **1.38** | **1.40** | **1.011** |
| `gdn` / `attn` / `gdn_pre` / `prep_silu` | 4.18 | 4.05 | 0.969 |
| verify pass, host | 45.6 | 60.1 | **1.318** |

The head and the four untouched phases hold the panel to about 1%, and every retiled phase loses in
proportion to how much it was split. Drafted generation goes 46.3 / 87.8 / 55.0 / 76.5 tok/s to
31.6 / 68.8 / 43.7 / 60.2.

**The probe is not wrong; it does not model the unit boundary the engine pays.** A `ph_matvec` unit
ends with eight waves publishing their partial sums through LDS, two `__syncthreads`, a 32-lane by
8-row drain and — wherever `KS > 1` — 256 `atomicAdd`s, none of which the probe's register
accumulate has. Priced from the panel, an extra unit costs **2.7 us of workgroup time**: 27-29 ns per
unit per grid for the three phases that also acquire atomics and zeroing at the wider split, 12-15 ns
for the three that already had them. The stream gains 10-20%; the boundary costs more.

**One confound, named and bounded.** The split arm is also **156 VGPR at nine waves per SIMD32
against 141 at ten** (`tools/kernel_resources.py kernels/halo_rows.hip k_forward_rows`, both arms),
and the cooperative grid comes from that occupancy, so it ran about a tenth fewer workgroups.
[The grid curve for this pass](matvec-crossover.md) is 62.8 / 55.6 / 52.6 / 49.8 ms at 60 / 72 / 84 /
100 workgroups, which puts a tenth of the grid at roughly 4% — a seventh of the 31%, in the same
direction. The per-phase ratios say the same thing: the head runs on the same reduced grid and moves
1.1%, so the grid cannot be carrying 45% of `mv_gate_up`. The conclusion survives the confound; the
exact split between the boundary and the wave slot does not, and a register-neutral retiling would
have to be built to separate them.

Going the other way is a null. `KSV = KSFF = 1` — 160-unit phases, no atomics at all on the three
residual-accumulating projections — measures 45.44 against 45.39 ms with `mv_down` at 1.003 and
`mv_ssm_out` at 1.008. **The deployed split sits on a plateau**, and the axis is closed at both ends
for this shape.

### The K split is a numerical map, not a schedule

A grid choice moves which workgroup computes a whole tile and changes no output bit. A **`KS`** choice
moves where the K axis is cut, and a K part's partial sum reaches the output through `atomicAdd`, so
the FP32 association changes with it. It is observable: on the first of four prompts the greedy digest
went `5074777665048850553` to `5919438133006348203` and the step's accepted count 2.38 to 2.07
tokens — the same alternative continuation under *both* the wider and the narrower split, with the
other three prompts identical in all three arms. Anyone moving these constants is changing what the
served model emits, not only how fast it emits it.

## What is left in this pass, ordered

> **The unit-count reading of the table above has a second half, and it ships.** A phase's rate
> depends on how many grid rounds it gets because `grid_sync` releases every workgroup at the same
> instant and their weight streams then collide; staggering the phase start is **-9.6% of this pass**
> and +7.2% drafted generation, bit-identical, with three prefetch arms and an address de-phasing
> arm measured as nulls. [`docs/phase-ramp-prefetch.md`](phase-ramp-prefetch.md) owns it.

1. **`mv_gate_up` is 37% of the step and 26% off the probe's own number for its shape.** The probe
   reads 184.3 GB/s at 1088 units and the engine gets 149.5 through the same body at the same unit
   count, while `mv_lm_head` matches its probe cell exactly. Whatever separates a 64-launch phase
   from a 1-launch one is worth 3.8 ms of a 45 ms pass and is not the unit count, the grid or the
   block count.
2. **The three 5120-row projections are at 113 to 129 GB/s and cannot be retiled.** `mv_down` alone
   is 9.7 ms. Their phases are short by construction — 160 output tiles is all the model has — so
   anything that helps them has to come from inside the unit.
   **And they cannot be *coarsened* either, which this document asked and
   [`docs/mv-tile-fold.md`](mv-tile-fold.md) answered**: a tile fold that halves the unit count with
   no atomics, no boundary work and no numerical change at all costs `mv_ssm_out` **+47.7%**,
   `mv_o` +43.5% and `mv_down` +28.5%, while `mv_lm_head` at 7760 units costs 5.0%. The retiling
   arm above and the fold bracket the deployed unit count from both sides, so **a unit that gets
   bigger has to buy back forty percent inside itself on the 160-tile projections**.
3. **The head is the shape that works.** One launch, 7760 units, 203 GB/s, 3% of the pass. It is the
   existence proof that this body reaches the stream on this machine when the phase is long enough.

## Two things this iteration leaves in the tree

**`bench/coop_cost --tt N --exact-units`.** The probe was one-row-only, so every scaffolding number
this lane has published — barrier cost, unit cost, the `pw` curve — describes the single-token step
and not the eight-row pass that serves generation. The two shapes disagree about `pw`, about unit
count and about how long a phase must be, so the row count belongs in every future measurement here.

**`make DEFS='-DHALO_ROWS_TRIM=1'`.** `kernels/halo_rows.o` had grown past 55 seconds of device
codegen, which is longer than one agent tool call: `make`, `tools/split-build --step 0` and every
`kernel_resources` reading on that file were failing for everyone, and the work was unmeasurable
rather than slow. Row counts in that kernel are capacities rather than contracts — `nrows` is a
runtime argument and every body guards on it — so a measurement build can serve 2- and 4-row passes
from the 8-row kernel, which drops roughly half the instantiations and brings the translation unit
back to **36-42 seconds**. Both arms of the panel above were built that way. It is a build flag, not
a default: a trimmed object must not be linked beside a full one, and the shipped binary carries all
four widths.

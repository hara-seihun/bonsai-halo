# One wave was draining eight rows while seven waited, and that was the gap to the probe

[The served verify pass](drafted-serve-step.md) left one question first in its own list: `mv_gate_up`
reads **149.5 GB/s** where `bench/coop_cost` reads **184.3** through the same body at the same unit
count, `mv_lm_head` matches its probe cell exactly, and "whatever separates a 64-launch phase from a
1-launch one is worth 3.8 ms of a 45 ms pass and is not the unit count, the grid or the block count".

It is the drain, plus about half of the number. The probe's unit ends in registers; the engine's unit
ends with eight waves publishing partial sums through LDS, **one** wave summing all of them and
writing the output row, and two `__syncthreads` around it. Priced directly, that drain costs a
1088-unit phase **4.4%** and a 320-unit K-split phase **8.9%**. Spreading it over the eight waves
that computed it — wave `w` owns row `w` — returns all of the first and half of the second, changes
no output bit, and costs no LDS, because `red[NW][TT][32]` has always allocated wave 0's eighth and
never written it.

Raw samples, panel scripts and the probe arms:
[`batch-comparison/matvec-drain/`](../../../data/bonsai2/batch-comparison/matvec-drain/README.md).

## What the deployed drain was, and what replaced it

```c
if (wave > 0) for (r) red[((wave - 1) * TT + r) * 32 + lane] = y[r];
__syncthreads();
if (wave == 0) for (r) { v = y[r] + red[1..7][r]; out = v; }   // 56 LDS reads, 8 writes, at TT = 8
__syncthreads();
```

Wave 0 kept its own partial in registers and summed the other seven itself. At eight rows that is
fifty-six `ds_read_b32` and eight output writes in one wave while seven waves sit at the second
barrier. The engine pays it **once per unit**, and the deployed verify pass deals about 150,000
units.

Row `r` belongs to wave `r` now. Eight partials fit in the block's **seven** slots because a row's
owner reads its own partial from a register, which frees its slot for that row, and wave 0 writes
there:

```c
if (wave > 0) for (r != wave)  red[((wave - 1) * TT + r) * 32 + lane] = y[r];
else          for (r >= 1)     red[((r    - 1) * TT + r) * 32 + lane] = y[r];   // the owner's freed slot
__syncthreads();
if (wave < TT) { r = wave; v = (r ? red[(r-1)*TT+r] : y[r]);
                 for (w = 1..7) v += (w == r) ? y[r] : red[(w-1)*TT+r]; out = v; }
__syncthreads();
```

Every `(slot, row)` still has exactly one writer, and the sum is still
`y_0[r] + y_1[r] + ... + y_7[r]` left to right — the owner splices its register in at position
`wave` instead of reading a slot. **Bit-identical by construction**, and it takes no LDS the block
did not already have, so the grid is the one the register budget gives. `TT == 1` keeps the
register form, where there is one row to own and the extra publish would be pure cost, which makes
single-token decode untouched code.

**The first form of this change published wave 0 into an eighth slot**, which is what
`red[NW][TT][32]` allowed when the panel below was measured. Canonical main rebased the array to
`(wave - 1)` and sized `rows_lds_floats` at `(NW - 1) * TT * 32` in the same hour, so that form
would have written one slot below the array. The seven-slot version above is what ships; both
panels are kept because they measure the same mechanism on two bases.

## The ladder that found it

`bench/coop_cost` gains four arms, all of them differences between the probe and the engine that
nobody had priced. Every number below is grid 120, `tt 8`, 64 phases, four rounds, `--exact-units`,
arms alternating inside one `tools/run-batch-compare --pin-clock` hold.

| what was added to the probe | 1088 units, pw 5 | 320 units, pw 3 |
|---|---:|---:|
| nothing (what the probe measured before) | 177.4 / 176.4 | 148.2 / 148.9 |
| `--runtime-rows`: `nrows` a runtime argument, not `TT` | 176.1 / 176.2 | 148.3 / 149.6 |
| `--drain`: the engine's LDS publish, reduction and write | 172.2 / 170.9 | 136.1 / 135.4 |
| `--split 2`: the part-major K-split stream geometry | 165.6 / 166.5 | 130.2 / 130.6 |
| the engine's own phase, same shape | **166.8** | **113 - 115** |

`mv_gate_up` is `KS = 1`, so its row is the first three lines: 177 GB/s in the probe, 171 with the
drain and the runtime row count, **166.8 in the engine**. The 26% is a drain, a row guard and about
6% of residue — and the residue is inside the panel-to-panel drift measured below.

**Three things it is not**, each a separate arm and each a null:

- **Where the bytes live.** `Engine::upload_halo` gives every ternary tensor its own `hipMalloc`,
  about 400 allocations for 5.878 GB, and the probe walks one 2 GiB allocation in ascending windows,
  so phase `p+1` starts where phase `p` stopped. `--regions N` gives every phase its own allocation
  and `--scatter` permutes the windows of one: **192.0 / 191.0 / 191.3** at 1088 units and **158.5 /
  157.8 / 156.3** at 320 for sequential, scattered and separate. The head, the one engine phase that
  matches its probe cell, reads one 278 MB allocation in one sweep, and that is not why it matches.
  A weight slab laid out in pass order buys nothing.
- **The tiny phase between two streams.** The engine never runs two matvec phases back to back: a
  prep phase normalises and quantises the row each one reads, 5 to 10 us against the matvec's 60 to
  234. `--gap-units` inserts one. An empty gap phase and a 40-unit prep-sized gap phase cost **their
  own time and 1.3 us more** (0.2193 + 0.0026 against 0.2232 measured).
- **An extra phase body in the kernel.** `k_forward_rows` carries ten phase bodies and the probe
  carried one. `-DCOOP_GAP=1` compiles one more body in, disabled at runtime: +4 VGPR, the same 16
  waves, and **188.4/188.1/189.1 against 187.2/188.3/189.4** over three alternating pairs.

**And a methodology fact the lane should carry.** The same probe arm reads **177.5 GB/s in one panel
and 196.7 in another**, both under `--pin-clock`, while arms *inside* one panel agree to 0.6%. Any
engine-against-probe number taken from two panels carries a 10% error bar, which is larger than most
of what this lane measures. The tables above are all single-panel.

## What it is worth on the model

Two binaries one file apart, both `DEFS='-DHALO_ROWS_TRIM=1'`, alternating inside one `--pin-clock`
hold, `--bench -n 32 --dflash`, two warm-up runs discarded. The base arm of an ordinary A/B/A/B panel
falls monotonically across the panel here, so the arms run **ABBA**, three blocks:

| block | verify ms, wave-0 drain | verify ms, spread | delta |
|---|---:|---:|---:|
| 1 | 45.289 | 44.137 | -1.152 |
| 2 | 44.369 | 42.924 | -1.445 |
| 3 | 43.535 | 40.934 | -2.601 |

**Mean paired delta -1.733 ms, -3.9%, three of three blocks negative.** Greedy digest
`5074777665048850553` and 13 steps / 91 drafted / 18 accepted in all twelve runs, and in the eight
runs of the A/B/A/B panel beside it.

The first form of the change, measured on base `5ae01b4` over five paired rounds with one warm-up:
42.979 / 44.670 / 43.176 / 43.889 / 42.568 against 42.245 / 41.801 / 41.503 / 41.802 / 40.628, a
**median paired delta of -1.940 ms, -4.5%, five of five negative**, with the drafter's own phase the
in-process control at -0.4% and the same digest in all ten runs. Per phase over three paired rounds
with the tracer (medians, us per launch): `mv_o` 63.6 to 59.1, `mv_ssm_out` 61.8 to 59.0, `mv_down`
146.3 to 143.9, `mv_gate_up` 237.1 to 236.3, `mv_qkv_z` 121.4 to 121.7. Three rounds does not
separate 2-5% per phase and the per-step number is the result; the sign of the K-split phases
matches the probe, which is where the spread returns the most.

Single-stream plain decode is untouched code and measures as one: 33.41 against 32.85 tok/s median
over four paired rounds, with rounds 3 and 4 slower in **both** arms.

## What this does not reach

- **`TT == 1`.** One row, one owner; the register form stays.
- **The wide route.** `ph_matvec` lives in the persistent kernel. Mode 19/20 prompt ingestion and the
  wide batched step run `k_ffn_slice`, `sequence_*` and `attn_*`, which have their own drains.
- **`ph_matvec_w`**, the WMMA body for more than `MV_DOT4_MAX` rows, whose reduction is a different
  shape and is held elsewhere.

## What is left here

1. **The atomic half.** At 320 units the spread returns 137.9/138.3 of the 144.1/143.6 that no drain
   costs, where the store form returns all of it. What the `atomicAdd` costs beyond the reduction is
   still in that phase, and `KSV = KSFF = 1` - the coarsening that removes the atomics - is a
   [measured null](drafted-serve-step.md) because it doubles the unit size at the same time. A drain
   that keeps `KS = 2` units and writes once per tile would separate them.
2. **The 320-unit residue.** With the drain, the runtime rows and the split geometry the probe reads
   130.2/130.6 and the engine reads 113 to 115. That is 13% on `mv_ssm_out` and `mv_o`, 3.9 ms of a
   42 ms pass, and it is the last unexplained shape in this pass.
3. **`bench/coop_cost` now models the engine's unit**, not a register accumulate: `--drain 1..4`,
   `--runtime-rows`, `--split KS`, `--regions N`, `--scatter`, `--gap-units N`, `--no-stream`. A
   scaffolding number taken without those arms describes a machine the engine does not run.

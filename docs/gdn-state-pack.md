# The recurrent state's storage coordinate

A 32-stream generation step is not a small prefill pass. [The decode map](decode-map.md) measured
where it goes: `gdn-resident-core` owns 38% of the step, it moves 9.664 GB of gated-delta state, and
it moves it at **224 GB/s against the 242 GB/s `bench/bw` reaches on this device**. That phase is
not waiting on arithmetic, occupancy, latency or scheduling. It is the one phase in this engine that
is already finished, and the decode map says so in its own bounded negative: within any schedule
that keeps fp32 state in memory between layers, nothing you do to the recurrence takes those 48 ms
below about 40.

The premise that negative rests on is the fp32 state. This document drops it, and the result is
**+21.9% aggregate batched generation on the full model** with a measured prefill null.

It also leaves a premise of its own, which [the memory coordinate](gdn-state-coord.md) later
dropped: this document reads the phase as bandwidth-shaped at *every* precision, and after int8
it is not. 4x fewer bytes and 4x fewer requests bought 2.7x the time, and what was left was the
codec's request count rather than its bytes or its issue slots. The coordinate published here is
`i8c` from 2026-09-21; `i8` is the same values in a grouped row order, 9.4% faster at a
generation shape and bit-identical to every number below.

## What the state is, and what reads it

Per sequence and recurrent layer the state is `HV * SS * SS` floats: 48 heads of a 128 x 128 matrix,
3.146 MB, 151 MB over 48 layers, loaded and stored once per layer per step. A row of one head's
state is owned end to end by one wave: its 32 lanes hold columns `lane + 32*s` for `s` in 0..3, and
both consumers of that row are dot products reduced across the wave by `warp_sum` —

    part_j = sum_c S[j][c] k[c]        delta_j = (v_j - part_j) * beta
    S[j][c] += delta_j * k[c]          out_j = sum_c S[j][c] q[c]

`k` and `q` are L2 normalised per head, so each is a unit vector over 128 columns and the readout
error that matters is **absolute**, accumulated over the row, against a signal of the row's own
scale. That is the whole design argument:

- **fp16** keeps eleven bits of *relative* precision on every element, including the ones near zero
  that contribute nothing to either dot product, and spends five bits on an exponent range a
  normalised state row does not use.
- **int16 with one fp32 scale per state row** spends all sixteen bits on mantissa relative to the
  row's own magnitude. For a row whose columns are comparable in size — which is what
  `S[j][:] = sum_t delta_j(t) k(t)` produces, since the variation across columns is the variation of
  a normalised vector's components — the readout error is about 4.5x smaller at the same byte count.

The scale is wave-uniform, so reading it is one scalar load per row, and computing it is one
`warp_max` over a wave that already holds the row. `int8` with the same per-row scale is built and
measured too.

## What is preserved and what is not

**Preserved exactly.** The `(lane, s) -> column` map. Every packed coordinate decodes into the same
register the fp32 state loaded into, so the recurrence runs the same summation tree over the same
operands in the same order, the same `warp_sum` butterfly, the same fused multiply-adds. The only
numerical difference between a packed arm and the fp32 arm is the rounding of the value that crosses
memory, and the `f32pk` control below proves it rather than asserting it.

**Changed.** The column order *in memory*: column `c` is stored at `(c & 31) * 4 + (c >> 5)`, which
puts one lane's four columns in eight contiguous bytes and makes a row one 256-byte wave transaction
instead of four 64-byte ones. Nothing outside `kernels/gdn_state_codec.hpp` sees that order.

**Where the scales live.** The state buffer already reserves `GDN_STATE_FLOATS` floats per
(slot, layer) and a packed head needs at most half of it, so the scales sit in the free upper half of
the same region. No allocation, reset, snapshot, slot copy or rollback changes, and a zeroed region
still decodes to a zero state in every format.

**One invariant weakens, and it is worth stating plainly.** The sliced eight-row route and the
resident route commit state at different frequencies — once per slice against once per pass — so
under a packed coordinate they round a different number of times and are no longer bit-identical to
each other. Both remain correct approximations of the same recurrence; the `HALO_GDN_COMPARE`
diagnostic that checks them against each other is armed only in fp32.

## The deployed persistent kernel keeps its registers

`ph_gdn` lives inside `k_forward_rows`, which is a persistent kernel: it allocates one register
maximum over every phase in its body. Reading the format from a constant inside it costs the
deployed single-token pass **96 VGPRs to 135, sixteen resident waves per SIMD32 down to ten**, on a
route that never asks for a packed state. So the format is a template parameter there, the packed
kernels are their own instantiations with their own occupancy grid, and every fp32 instantiation is
at or below the registers it has today (`k_forward_rows<1, 0, 0, 0>` is 95, one better than before).

A second register trap in the same place is worth carrying: passing the wave's `float m[R][4]` to
the codec **by reference** takes its address and costs exactly one VGPR — 96 to 97, which is the
same cliff. The codec passes one row's four values by value instead.

`resident_state` is a small kernel at 16 waves per SIMD32 with room to spare (56 to 64 VGPRs), so it
takes the format from a `__constant__` and carries all coordinates in one build. That is what lets
`--gdn-state` walk them as a case axis inside one process.

## Measured: a generation step

`tools/batch_profile --decode-streams 32 --decode-prompt 64 --gdn-state ...`, mode 19, arms
interleaved in one process, every case re-prefilling all 32 slots into its own coordinate. Untraced
wall time for one real generation step with logits and argmax on every row:

| state | state bytes per step | step ms | aggregate tok/s | vs fp32 |
|---|---:|---:|---:|---:|
| fp32 | 9.664 GB | 113.2 | 282.7 | |
| fp16 | 4.832 GB | 92.6 | 345.7 | +22.3% |
| int16 + row scale | 4.870 GB | 93.4 | 342.6 | +21.2% |
| int8 + row scale | 2.435 GB | 84.3 | 379.5 | +34.3% |
| fp32 through the packed kernels | 9.664 GB | 114.0 | 280.8 | control |

The traced arms say the time comes out of exactly one phase and nothing else moves. Same process,
same clock, fp32 against int16:

| phase | fp32 | int16 | change |
|---|---:|---:|---:|
| `gdn-resident-core` | 44.437 ms | **24.232 ms** | **-45.5%** |
| `ffn` | 36.258 | 36.509 | +0.7% |
| `sequence-input-projection` | 13.743 | 13.737 | -0.04% |
| `sequence-output-projection` | 11.118 | 10.902 | -1.9% |
| `sequence-core` | 5.478 | 5.458 | -0.4% |
| `head-projection` | 2.535 | 2.522 | -0.5% |
| `sequence-input-prep` | 1.614 | 1.589 | -1.5% |
| device span | 116.332 | 96.065 | -17.4% |

The six phases the codec cannot reach are the in-process control. int16 costs 0.9% more step time
than fp16 for the same traffic — one scalar scale load and one `warp_max` per row — and buys the
precision the quality section measures.

**The `f32pk` control separates the coordinate from the kernel.** It stores fp32 values through the
packed *instantiations*, which have different registers and their own cooperative grid. Its residual
FNV-64 is `8087160186626784132`, the same value plain fp32 produces in three separate processes, and
its step time is within 0.4%. So the kernel split is numerically free and timing-neutral, and
everything a packed arm changes is the storage rounding. Raw:
`batch-comparison/gdn-state/decode-smoke.json`, `decode-ladder.json`, `decode-pkcontrol.json`.

## Measured: the full model

`tools/batch_compare --only prefill,decode --modes 19 --gdn-state 0,2 --streams 32
--prefill-rows 128 --rounds 2`, arms interleaved in one process, 384-token document in 128-row
passes, 16 generation steps across 32 streams:

| workload | fp32 | int16 | change |
|---|---:|---:|---:|
| **generation, 32 streams, aggregate** | 285.8, 286.6 tok/s | **348.4, 349.3 tok/s** | **+21.9%** |
| prefill, 128 rows per pass | 768.4, 761.7 tok/s | 757.2, 771.7 tok/s | **-0.08%** |

Prefill is a null by construction and measures as one: a prefill pass loads and stores the state
once per layer for 128 tokens, so its state traffic is 1/128 of a generation step's per token and
the phase is instruction-issue bound rather than memory bound. Raw: `batch-comparison/gdn-state-tps/`.

**Installed acceptance.** The canonical build
(`bonsai-halo` `94e8f22e761bfeff8cb1c8757a62f0c78a0ba439b68c18a36ad0fdb42678b02c`,
`tools/batch_profile` `feb6415604c78ac168d0ce33ebf27ab5257858a41a9b29edcc0dd5a7bc0cfedf`, revision
`77b92b5`) reproduced both residual hashes exactly — `8087160186626784132` for fp32 and
`14246133829617533007` for int16, the same values three development processes produced — at
109.944 ms against 89.566 ms, **+22.7%**. Single-stream `--bench` on the installed binary is
33.35 tok/s, inside this box's 33.05-33.80 range and untouched by construction: with the default
fp32 coordinate the single-token kernel is the same instantiation it was, one register smaller.
Raw: `batch-comparison/gdn-state/installed-acceptance.json`.

**Single-stream generation is a measured null**, and the arithmetic says why. `--engine --bench
-n 60` under the same wrapper: 33.35 tok/s in fp32, 33.33 with `HALO_GDN_STATE=i16`. One token of
single-stream decode moves 5.9 GB of weights beside 0.302 GB of state, so halving the state removes
0.7 ms of a 30 ms token at this device's read rate - 2.4% at best, under this box's 33.05-33.80
spread for that benchmark. The packed coordinate pays where the state is the traffic, which is a
batched step, not a single stream.

## Quality, and why the obvious metric cannot rank these arms

**Logit difference against the fp32 arm is saturated in this engine and must not be used to compare
coordinates.** Every activation quantiser here scales a 128-block by that block's own `amax`, so any
perturbation large enough to move an `amax` changes all 128 codes by a step, and the output
difference stops depending on how small the perturbation was. The measurement says so directly: at
32 rows in mode 19, fp32-against-int8, fp32-against-int16 and int16-against-int8 all differ by
rms 0.20-0.21 logits, and fp16-against-int16 is 0.209 — three coordinates mutually equidistant, with
a sixteen-fold difference in rounding between two of them.

Teacher-forced negative log likelihood on the document the rows came from has no such floor, is an
absolute number rather than a difference, and is paired per row. Mode 0 is the arm to read: its FFN
is exact, so nothing masks the state. 254 predictions pooled over two documents, 256 and 512 tokens
of recurrent history, `tools/gdn_state_quality.py`:

| state | ΔNLL vs fp32 (nats) | mode 0 mean KL | greedy top-1 agreement |
|---|---:|---:|---:|
| fp16 | +0.00245 ± 0.00217 (n=127) | | |
| int16 + row scale | **+0.00240 ± 0.00159** | 0.000285 | 125/128 |
| int8 + row scale | +0.00354 ± 0.00259 | 0.000503 | 127/128 |

For scale, on the same instrument the **A4 FFN map this repository already ships as its batched
headline costs +0.0615 nats** (mode 0 fp32 2.16604 against mode 19 fp32 2.22753 on the same 127
predictions). The int16 state coordinate costs about a twenty-fifth of that, and its cost is not
resolved at 2σ by a 254-prediction paired sample: the honest statement is **below +0.006 nats**,
under 0.3% of perplexity.

In mode 0, where the FFN is exact, KL is not fully saturated and the ladder is visible in the right
order (int8 is 1.77x int16). In mode 19 both saturate at KL 0.037, which is itself half of the
published 0.0704 the A4 map costs against mode 0.

Raw: `batch-comparison/gdn-state-quality/`, `gdn-state-nll/`, `gdn-state-nll2/`.

**This table is the wrong shape and [a later measurement replaces it](gdn-state-horizon.md).** It is
a prefill-shaped window, and a prefill pass commits the state once per 128 tokens per layer where a
generation step commits once per token. So it applies a few dozen roundings, not a few hundred, and
the accumulation it was meant to detect had nowhere to happen. On the generation shape at 256
commits per stream, int8 measures `-0.0165 +- 0.0082` nats and int16 `-0.0042 +- 0.0077`, with no
drift across the horizon and a control that reproduces its reference to zero. Read the numbers here
as what a prompt pass costs and the horizon document as what a generation run costs.

## What is selected

**fp32 remains the default everywhere**, including the resident service and every batch mode. The
packed coordinates are explicit alternative routes, like the A4 FFN map beside the exact one:

    HALO_GDN_STATE=f32 | f16 | i16 | i8 | f32pk      process-wide, read once
    --gdn-state 0,1,2,3,4                            case axis in both measurement tools

**int8 is the recommended packed coordinate** for batched generation, and this paragraph used to
recommend int16. [The horizon measurement](gdn-state-horizon.md) is the evidence that moved it: the
quality window this section was written against applies a few dozen state roundings, where a
generation step applies one per token per layer. Run on the generation shape for 256 commits per
stream, paired against a control that reproduces its reference exactly, int8 costs
`-0.0165 +- 0.0082` nats and int16 `-0.0042 +- 0.0077`, neither drifts across the horizon, and int8
is **+37.3% aggregate generation against fp32's 300 tok/s**, 12.6 points ahead of int16. The
per-(row, 32-column) scale this document held in reserve for int8 is measured there too and is not
worth building: it buys 1.45x in readout error for 12.5% more state traffic, against an output
measurement that cannot see a 257-fold difference in rounding.

Switching format invalidates every live sequence, so `gdn_state_set_format` synchronises the device
and the caller must reset or re-prefill its slots. Both measurement tools do that per case; the
`--v1` per-op path refuses any coordinate but fp32 rather than reading a state it cannot decode.

# Four bits on both sequence projections is the default

[The four-bit output projection](sequence-output-a4.md) was built, measured at **-40% of its phase**,
proved bit-identical to its own emulation, and then left switched off for one reason its own
document states plainly: *"not resolved by this instrument"*. The instrument was a 127-prediction
teacher-forced window that resolves about ±0.027 nats, and the arm it was asked to rank differs from
its reference by less than that.

That instrument was replaced hours later. [The horizon panel](seq-a4-default.md) carries the
precision axis now — 1024 paired teacher-forced predictions in the generation shape, stream-
clustered, ±0.008 nats — and it is what made four-bit activations the default on the *input* stage.
This document runs it on the output stage. The answer is that the second stage costs nothing the
panel can see, and that the two stages do not add:

| mode 19, against the same route's eight-bit operand | ΔNLL | stream-clustered |
|---|---:|---:|
| input four bits alone (`a4`, today's default) | +0.00770 | ±0.00798 |
| output four bits alone (`a4-out`) | +0.00458 | ±0.00901 |
| **both (`a4-both`)** | **+0.00397** | **±0.00940** |
| the A4 FFN this engine already serves (mode 18 → 19), same panel | +0.01866 | ±0.00687 |

So moving from today's default to `a4-both` is **-0.00373 ± 0.00934 nats**: the point estimate moves
the wide route *towards* the exact engine, 15 of 32 streams get worse, and teacher-forced top-1 goes
0.3975 → 0.4033. The whole wide A4 route is +0.02263 from the eight-bit-FFN reference where today's
default is +0.02636.

It is worth **+5.2% full-model prompt throughput at 128-row passes and +5.1% at 256**, with
batched generation +0.4% and the deployed sliced route bit-identical.

`HALO_SEQ_QUANT=a4` restores the previous default exactly, `a8` the exact operand on both stages.

## Two quantisers on one route do not add

The naive expectation is that two independent activation quantisers cost the sum of their errors,
+0.01228 nats here. The measurement is +0.00397, a third of that, and each stage measured alone is
larger than the pair.

That is not a paradox and it is not noise fitting: an activation quantiser is a *perturbation of the
residual stream*, and the two stages perturb it at different points of the same layer with errors
that are independent of each other. Teacher-forced NLL is not a norm on that perturbation — it is
the log-likelihood of one token under the perturbed distribution, and independent perturbations of a
deep residual stream land on both sides of it. The quarters of the horizon say the same thing in the
time axis:

| arm | steps 7-63 | 71-127 | 135-191 | 199-255 |
|---|---:|---:|---:|---:|
| `a4` | +0.00668 | -0.00555 | +0.01514 | +0.01454 |
| `a4-out` | +0.00540 | -0.01945 | +0.02173 | +0.01064 |
| `a4-both` | +0.02242 | -0.01182 | +0.02504 | -0.01977 |

The sign changes and the magnitude does not grow. **Read the slope before the mean**: an activation
quantiser is memoryless, because the next token's operand is re-derived from the residual stream,
which is the opposite of [the packed recurrent state](gdn-state-horizon.md) whose error the
recurrence never re-derives. The instrument exists for the second case and is being read here for
the first.

What this does *not* say is that a third four-bit stage would also be free. It says that the cost of
this one is bounded by the panel's resolution and that the composition is not additive. The next
approximate stage gets its own panel.

## The kernel computes its own map, twice over

`a4e-both` is the same seven-level numerical map on the eight-bit matrix instruction: same products,
same `int32` accumulators, same FP32 drain, different instruction and different stored code order.
It is the control that separates *the map* from *the kernel that runs it*.

- Horizon, 1024 paired predictions: `max |ΔNLL| = 0.000e+00` and 1024/1024 argmax equal.
- `--prefill-identity`, a 384-token document, every logit hashed: **95,354,880 logits, FNV-64
  `16232333786579957879`**, for `a4-both` and for `a4e-both`, at 128 and at 256 rows per pass.

Four numbers, one value. The four-bit kernel on both stages reproduces the eight-bit instruction's
arithmetic to the last bit, and the map is invariant to pass width.

## What the coordinate cannot reach

Mode 0 — the deployed sliced route, which is what single-stream and drafted generation run — hashes
**95,354,880 logits to `7241142880296265908` under `a4` and under `a4-both`**. Identical. The
sequence projections' coordinate lives in the wide route's own weight images and the sliced route
never reads them, so this is measured rather than argued.

The wide route's own map does move, as it must: mode 19 is `1429845836133227146` at `a4` and
`16232333786579957879` at `a4-both`, both width-invariant across 128- and 256-row passes.

## Measured speed, on the build that ships

### The phase

`tools/batch_profile`, mode 19, 128 rows, logits off, `--pin-clock`, both arms interleaved in one
process, three rounds. `sequence-output-projection` normalised by the four phases the arm cannot
reach (`ffn`, `gdn-resident-core`, `sequence-core`, `sequence-input-prep`), which is how this lane
reads a phase panel on a box whose wall time moved 124-144 ms inside this panel.

| arm | out-proj ms | normalised | in-proj normalised | residual FNV-64 |
|---|---:|---:|---:|---|
| `a4` | 17.062 / 16.773 / 14.814 | 0.17096 | 0.25007 | `8607785813945241221` |
| **`a4-both`** | **9.954 / 8.891 / 9.133** | **0.10249 (-40.05%)** | 0.25052 (+0.18%) | `3240780174642567821` |

The input projection is the in-panel null the arm has to leave, to a fifth of a percent. The -40.05%
reproduces the -39.4% [the arm's own document](sequence-output-a4.md) published on an executable
from the previous day, across the block-major images, the GDN lane layout, the matvec's scalar
waits, the four-bit input default and the peel gather that have landed since. `3240780174642567821`
is the residual that document published for this arm.

### The model

`tools/batch_compare --only prefill`, a 1024-token document, mode 19, tail-only head, `--pin-clock`,
three rounds with the arms interleaved and reshuffled after round 1.

| prompt tok/s | `a4` (default) | `a4-both` | median |
|---|---|---|---:|
| 128-row passes | 945.1 / 963.0 / 961.6 | 939.1 / 1011.7 / 1011.9 | 961.6 → **1011.7 (+5.2%)** |
| 256-row passes | 981.0 / 972.4 / 974.7 | 992.3 / 1024.7 / 1030.5 | 974.7 → **1024.7 (+5.1%)** |

The first `a4-both` round at 128 rows is the one cell below its own arm's median and below every
`a4` cell; rounds 1 and 2 agree to 0.2 tok/s with each other and were taken in opposite arm order.

Aggregate batched generation, 32 streams, context 256, 24 steps, three rounds:
**328.1 / 328.5 / 328.7 → 329.8 / 330.0 / 329.8 tok/s, +0.42%.** The phase is 10-15% of a generation
step and the change is worth about half a percent of it there; this is a prompt-ingestion lever, as
its own document said.

## The part that made it a default rather than a flag

A four-bit operand exists only where a wide kernel wrote it, because *the writer is the quantiser*:
`resident_output` for a recurrent layer, the wide attention combine for an attention one, and the
wide prep for the input stage. The per-slice half-layer inside the persistent kernel has no
seven-level store. The stored code order is a process-wide property of the image
(`sequence_batch_sync_quant` permutes it in place), so a silent fallback to the eight-bit kernel with
the image untouched would read nibble-ordered codes as spread ones — a wrong number, not a slow one.

`sequence_project` therefore **threw**. As an explicit route that is correct. As a default it is a
crash: `mode 17` (wide sequence, no committed state, so no resident writer) and `HALO_WIDE_PREP=0`
are both live measurement controls that reach it.

The rule this now follows, and it is the general one:

> **A coordinate the process was told is a contract; the default is a preference.**

`sequence_quant_explicit()` separates them. `HALO_SEQ_QUANT=...` and `sequence_set_quant()` — which
is what every measurement arm in `batch_compare` and `batch_profile` calls — mark the coordinate as
told, and an unsupported route still fails, loudly, with the reason:

```
batch_compare failed: a four-bit sequence output projection needs the direct layout and a wide
writer (resident GDN for a recurrent layer, wide attention for an attention one, which wants a pass
wider than RMAX rows), and HALO_SEQ_QUANT=a4-both asked for one this pass cannot carry
```

A process that merely took the default drops *that stage* to the exact eight-bit operand, once, with
one line on stderr naming what it dropped and how to demand it instead, and re-syncs the image so
the codes match the kernel that reads them. Measured: mode 17, 128-row passes, 384-token document —
`475.7 tok/s` with the fallback where the same command with `HALO_SEQ_QUANT=a4-both` exits 1.

Two consequences worth knowing:

- **The fallback is sticky for the process,** because it is a property of the routes this process can
  run, not of one pass. It is not a per-pass decision: switching back would permute ~420 MB of
  output-projection codes on every pass that changed its mind.
- **A panel now records the coordinate that ran, not the one it asked for.** `batch_compare` reads
  `sequence_quant()` back *after* each timed case into `seq_quant` and keeps the request in
  `seq_quant_asked`. Before this, a fallback would have been invisible in the JSON. `batch_profile`
  already built its sample record after the run and needed no change.

## A tool repair that came out of it

`Telemetry` samples the GPU's sysfs counters on a worker thread and had no destructor, so an
exception leaving a run destroyed a joinable `std::thread` — `std::terminate`, `SIGABRT`, a core
dump, and **the engine's own error message never printed**. That is how the explicit-contract check
above first appeared: `terminate called without an active exception`, with nothing to read. One
destructor that stops the sampler restores the `batch_compare failed: ...` line for every error in
this tool, not only this one.

## Installed acceptance

Published as `f91a815`; `bonsai-halo`, `tools/batch_compare` and `tools/batch_profile` installed into
`.` at 20:33, `bonsai-halo`
`d988698f0980e6d1737c21bc…`, `tools/batch_compare` `594a7d2d14f7bdf33b0f17d8…`. A peer merged and
rebuilt `bonsai-halo` three minutes later, so the installed engine is now
`0ff2260a7421db0b8339c5c7…` on `ab3908f`, which has `f91a815` as an ancestor and carries this
default; the panels below ran on `tools/batch_compare` `594a7d2d`, which is this change's build.

1024-token document, 128-row passes, both arms interleaved in one process and reshuffled between
rounds, `--pin-clock`:

| route | `a4` | `a4-both` | |
|---|---|---|---:|
| mode 19, the benchmark route | 953.5 / 951.1 | 998.0 / 993.5 | **+4.6%** |
| **mode 20, the route the service ingests prompts on** | 558.5 / 559.7 | 577.1 / 577.3 | **+3.2%** |

And the installed binary reproduces both hashes at the default: mode 0 `7241142880296265908`, mode
19 `16232333786579957879`, each identical at 128 and 256 rows per pass.

## What this does to a consumer that wanted token identity

The wide route now applies two activation quantisers the eight-row route does not, so **crossing 32
rows is a numerical change for every consumer of `forward_batch`**, not only for a panel that asked
for it. `bc30b5f7` hit this while pricing a served *decode* route on token identity: 18 of 32
concurrent greedy completions diverged from the eight-row route at 32 rows where 24 of 24 agreed at
16, and the first suspect was their own scheduler. Half of that was already true from the four-bit
input default; this change is the other half.

Pinning `HALO_SEQ_QUANT=a8` restores the exact operand on both projections, and it is worth knowing
exactly what that does and does not buy. Measured on the installed build, 384-token document, every
logit hashed:

| route, `a8` | 128 rows/pass | 256 rows/pass |
|---|---|---|
| mode 0, the sliced route | `7241142880296265908` | `7241142880296265908` |
| mode 20, wide schedule over the deployed FFN | `15043080294956075508` | `15043080294956075508` |

**The wide route is not bit-identical to the sliced one at any activation coordinate** - the wide
projections reassociate the FP32 drain, which [`wide-sequence.md`](wide-sequence.md) has always
said - so token agreement between mode 4 and mode 20 is a property of the prompts that were
measured. That document now carries this correction at the top, because three of its results read
as route properties and were arm properties.

## Reproducing

```sh
# quality: one arm per process, arms pool across processes on identical tokens
for q in a8 a4 a4-out a4-both a4e-both; do
  tools/run-batch-compare --tag seq-out-a4-horizon/m19-$q --only horizon --modes 19 \
    --slots 32 --context 512 --prefill-tokens 384 \
    --horizon-doc ../../data/bonsai2/batch-comparison/_inputs/horizon-corpus.txt \
    --horizon-streams 32 --horizon-ctx 64 --horizon-tokens 256 --horizon-every 8 \
    --seq-quant $q --rounds 0 --no-logits-dump
done
tools/run-batch-compare --tag seq-out-a4-horizon/m18-a8 --only horizon --modes 18 ... --seq-quant a8
tools/seq_quant_quality.py ../../data/bonsai2/batch-comparison/seq-out-a4-horizon/*

# the kernel against its own map, and what the coordinate cannot reach (one process per arm:
# --prefill-identity runs after the timed rounds, so --rounds 1 is what applies the arm)
tools/run-batch-compare --tag seq-out-a4-default/identity-a4both --only prefill --prefill-identity \
  --modes 0,19 --prefill-rows 128,256 --prefill-tokens 384 --context 512 --seq-quant a4-both \
  --rounds 1 --slots 8

# the phase and the model
tools/run-batch-compare --pin-clock --profile-tool --modes 19 --rows 128 --heads 0 --traces 1 \
  --seq-quant 2,6 --rounds 3 --out .../seq-out-a4-default/phase-128.json
tools/run-batch-compare --pin-clock --tag seq-out-a4-default/prefill --only prefill --modes 19 \
  --seq-quant a4,a4-both --prefill-rows 128,256 --prefill-tokens 1024 --context 1280 --rounds 3 --slots 8
tools/run-batch-compare --pin-clock --tag seq-out-a4-default/decode --only decode --modes 19 \
  --seq-quant a4,a4-both --streams 32 --slots 32 --context 256 --gen-steps 24 --rounds 3

# the route the default has to survive
tools/run-batch-compare --tag x --only prefill --modes 17 --prefill-rows 128 --prefill-tokens 256 --rounds 1
HALO_SEQ_QUANT=a4-both tools/run-batch-compare --tag x --only prefill --modes 17 ...   # exits 1
```

Raw samples, every arm and the failed cells are in
[`batch-comparison/seq-out-a4-default/`](../../../data/bonsai2/batch-comparison/seq-out-a4-default/README.md)
and [`batch-comparison/seq-out-a4-horizon/`](../../../data/bonsai2/batch-comparison/seq-out-a4-horizon/README.md).

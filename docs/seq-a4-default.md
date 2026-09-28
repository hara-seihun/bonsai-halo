# Four bits on the sequence input projection is the default

[The four-bit input projection](sequence-input-a4.md) landed measured and switched off. It took
`sequence-input-projection` from 40.494 to 22.010 ms at 128 rows and full-model prompt throughput up
11%, and then closed with one line that kept it an explicit route for a day:

> The quality sample wants widening before this becomes a default. 127 teacher-forced predictions in
> one window resolve ±0.04 nats; the arms differ by less than that.

That is a sample-size problem, not a quality problem. This is the widened instrument, the verdict,
and the default flip.

## The verdict, in the only form that makes it actionable

**The change costs 41% of a change this engine already serves, measured against that change, on one
instrument, with the same 1024 paired predictions.**

| change | ΔNLL, nats | predictions |
|---|---:|---:|
| A4 FFN activations (mode 18 → 19), **shipping since before this lane opened** | **+0.01866 ± 0.00676** | 1024 |
| A4 sequence input projection (`a8` → `a4`) inside mode 19 | **+0.00770 ± 0.00798** | 1024 |

Standard errors are stream-clustered over 32 independent document windows. The four-bit input
projection is not separable from zero at one sigma and is bounded at +0.024 at two, against a route
whose own departure from the eight-bit FFN is +0.019 and is not in question.

Nothing about that is an argument from the absence of evidence: the instrument resolves the
already-accepted change at 2.8 sigma in the same run, so it can see a difference of this size, and
it does not see one here.

## The instrument, and why the old one could not decide

`tools/batch_compare --only horizon` walks 256 single-token generation steps per stream on fixed
teacher-forced tokens and scores the distribution every eighth step. An activation quantiser is
applied **once per token per layer**, so a generation step is where it is exercised hardest; the
teacher-forced prefill window applies it once per pass and scores 127 predictions of one document
slice.

The horizon axis existed for [the packed recurrent state](gdn-state-horizon.md) and did not carry
activation precision. It does now — `for (int qz : cfg.seq_quants)` around the horizon loop, with
the `sequence_batch_sync_quant` the arm change needs because switching precision permutes the
1.30 GB input image in place, and `seq_quant` recorded in the arm's identity.
`tools/seq_quant_quality.py` pivots the report on precision where `gdn_state_quality.py` pivots it
on the state coordinate, and prints the already-served route's cost beside the candidate's so the
two are never read against zero.

Mode 19, `f32` state, 32 streams × 256 steps, 64 tokens of context, distinct document windows:

| arm | NLL | ppl | teacher top-1 | greedy agreement with `a8` | ΔNLL vs `a8` |
|---|---:|---:|---:|---:|---:|
| `a8` | 3.04150 | 20.937 | 0.3975 | 1.0000 | — |
| `a4` | 3.04920 | 21.099 | 0.3975 | 0.8877 | **+0.00770 ± 0.00798** |
| `a4e` | 3.04920 | 21.099 | 0.3975 | 0.8877 | +0.00770 ± 0.00798 |
| mode 18 `a8` *(reference)* | 3.02284 | 20.550 | 0.4082 | — | −0.01866 ± 0.00676 |

### The error does not accumulate, and that is the property that decides it

Split into quarters of the horizon, ΔNLL reads **+0.00668 ± 0.01828, −0.00555 ± 0.01702,
+0.01514 ± 0.02003, +0.01454 ± 0.02121** over steps 7‑63, 71‑127, 135‑191 and 199‑255. It changes
sign between the first two quarters and does not grow: an activation quantiser is memoryless. The
next token's operand is re-derived from the residual stream, so nothing carries the previous
token's rounding forward.

That is the whole reason this axis needed a different argument from
[the packed state](gdn-state-horizon.md), where the same instrument exists precisely because the
error *does* compound through a state the recurrence never re-derives. **Depth was the thing the
127-prediction window could not see, and depth is where this change turns out to be free.** A
horizon report for an activation quantiser should be read for its slope first and its mean second.

### Against the exact engine, the four-bit arm is the closer of the two

The teacher-forced window still earns its place, because it carries the mode 0 anchor. 127
predictions, 128 rows, `--quality-ctx 64`:

| arm | NLL | next-token top-1 | ΔNLL vs mode 0 `a8` |
|---|---:|---:|---:|
| mode 0, any precision | 1.71498 | 0.598 | 0 |
| mode 19 `a4` | 1.74785 | 0.591 | **+0.03287 ± 0.02857** |
| mode 19 `a8` | 1.75645 | 0.591 | +0.04147 ± 0.02276 |

On the shape where the two instruments overlap, the eight-bit arm is *further* from the exact engine
than the four-bit one. Both bars are ±0.023 to ±0.029 and the difference between the arms is
0.0086, so **this table cannot rank them and should not be quoted as if it does** — it is the
published 127-prediction result reproducing, and it is exactly why the horizon panel was needed. It
does two useful things: it reproduces the A4 FFN route's +0.0415 cost, and it shows the four-bit
input projection is not making a large-signed move that the horizon's tighter bars might have been
averaging away.

### Mode 0 is the inertness control and it is exact

The precision axis is a wide-route coordinate. Run at mode 0 it must change nothing, and it does
not: 31,784,960 logits identical across `a8`, `a4e` and `a4`, greedy 128/128, KL 0.000000,
`|Δlogit| max 0`. **A quality axis that moves an arm it cannot reach is measuring the box, not the
change**, and this one does not.

### The kernel is an exact implementation of its own map

`a4e` is the same numerical map as `a4` computed through the eight-bit matrix instruction, so their
equality is the kernel's acceptance against its own arithmetic rather than a quality result.
**31,784,960 logits bit-identical at mode 0 and at mode 19**, and on the horizon the per-prediction
NLL of the two arms differs by **exactly 0.0 over 1024 predictions**. The four-bit instruction, the
nibble operand and the permuted K order reproduce the emulation to the last bit.

## Throughput

`tools/run-batch-compare --pin-clock --only prefill --modes 19 --seq-quant a8,a4 --rounds 4`, a
1536-token document, logits on the final pass, and a 32-stream generation step. Revision `e1f81764`,
executable `9c25828293792bdc`, one process per panel, round 0 discarded as the clock ramp. The tool
reverses the arm order after round 1, so every arm is measured both first and second:

| workload | `a8` | `a4` | change |
|---|---:|---:|---:|
| prompt, 128 rows per pass | 809.4 tok/s | **908.4** | **+12.2%** |
| prompt, 256 rows per pass | 828.2 | **924.1** | **+11.6%** |
| 32-stream aggregate generation | 323.1 | **327.8** | **+1.45%** |
| single-stream generation | *null by construction* | | |

Samples: prompt `a8` 809.8 / 809.4 / 808.8 and `a4` 913.8 / 908.4 / 907.5 at 128 rows; generation
`a8` 323.9 / 323.2 / 323.0 / 322.3 and `a4` 327.1 / 327.8 / 327.8 / 328.7. The `a8` arm reproduces
to **0.12%** across three rounds and both slot positions, which is the noise floor this 12% is read
against. The published +11.4% / +11.0% / +1.4% reproduce on today's main at +12.2% / +11.6% /
+1.45%.

**A pass width does not change an arm's bits.** The residual FNV-64 over the tail logits is
`16073325506415993404` for `a4` at both 128 and 256 rows and `16295499040887489407` for `a8` at
both. Two maps, two hashes, each width-invariant.

## Installed acceptance

Installed on canonical `82dc7c7`, executable `883d2189aaab2700f11d9861`, which is source-current
through `0ebef11` (documentation only after this commit). The acceptance is taken on **mode 20, the
route `bonsai-halo serve` now ingests prompts through**, at both precisions in one process, three
rounds with the arm order reversed after the first, 1536-token document in 128-row passes:

| round | `a8` | `a4` |
|---|---:|---:|
| 0 (clock ramp, first arm cold) | 495.5 | 453.7 |
| 1 | 477.7 | **530.4** |
| 2 | 497.2 | **535.0** |
| median of 1–2 | 487.5 | **532.7** (**+9.3%**) |

Smaller than mode 19's +12.2%, and it should be: mode 20 runs the deployed FFN arithmetic, whose
phase is about twice mode 19's, so the same saving is a smaller share of the pass. Its `a8` arm also
spreads 4.1% across rounds against mode 19's 0.12%, so read this as +8% to +11% rather than as
+9.3%.

**The resident service answers on it.** One POST to the installed binary on port 8471, 63-token
prompt, greedy, 40 tokens: coherent output, `prefill_seconds` 0.252, 51.29 tok/s drafted generation,
30 accepted of 70 drafted over 10 speculative steps, and the service active on the canonical unit
after the panel released the GPU lock.

## What the flip does and does not reach

`HALO_SEQ_QUANT` now defaults to `SEQ_QUANT_A4` in `kernels/prep_batch.hip`: **four bits on the
input projection, eight on the output one.** `HALO_SEQ_QUANT=a8` restores the previous default
exactly and stays the control.

- **The output projection is unchanged.** [Its own document](sequence-output-a4.md) calls it "not
  resolved by this instrument" and it adds 0.0338 of mean KL. `a4-out` and `a4-both` remain explicit
  routes. This flip is the stage whose evidence is in.
- **Single-stream generation and the drafted verify pass are untouched by construction.** Both run
  the sliced route through the persistent kernel, which never calls `sequence_project`; mode 0's
  bit-identical logits above are that statement measured rather than asserted.
### It reaches the served prompt route, and that route's owner should see this

This was measured while `bonsai-halo serve` still ingested prompts eight rows at a time, where the
flip was inert. `5eb5fcb` landed **mode 20 as the served prompt route** in the same hour, and mode 20
is a wide route: `wide_sequence` is `batch_mode >= 17 && total >= 32`, so served prompt ingestion now
runs `sequence_project` and takes this default.

**That interacts with a claim `5eb5fcb` makes, and it is the one thing here that is a judgement
rather than a measurement.** Mode 20 is the wide schedule over the *deployed* FFN arithmetic, and its
argument for becoming the served default was that it is not a numerical change — one greedy digest
`17821512331856351063` across both routes and four processes. At `a8` that is exact and stays exact.
At `a4` served prompt ingestion becomes approximate by the +0.00770 ± 0.00798 nats measured above.

Two true things, and the route's owner gets to pick between them rather than inherit one:

- The cost is **less than half** of what mode 19 — the documented default map for wide passes in
  this lane's entry state — already spends on its FFN, so `a4` does not take the engine below its own
  published default map. And it buys 12% of prompt ingestion on top of mode 20's 2.72x.
- It does remove an exactness property the served path had. If that property is worth more than the
  12%, one line beside `prefill_ffn`'s resolution in `src/main.cpp` pins it back for the served route
  alone, leaving the benchmark column and mode 19 on four bits:

  ```cpp
  // Mode 20 exists to be bit-identical to the eight-row route; keep its operand eight-bit.
  if (prefill_ffn == 20 && !getenv("HALO_SEQ_QUANT")) sequence_set_quant(SEQ_QUANT_A8);
  ```

  `sequence_set_quant` is already public in `kernels/prep_batch.h` and is what both measurement tools
  use to walk this axis, so the pin needs no new interface. It is not applied here because
  `src/main.cpp`'s route resolution belongs to `5eb5fcb`, not to this change.

**The single-stream and drafted generation paths are unaffected either way** — they are the sliced
route, and mode 0's bit-identical logits above are that measured rather than asserted.

## Reproduce

```sh
# the decision: one arm per process, pooled by the analyser; arms pair on their target tokens
for q in a8 a4 a4e; do
  tools/run-batch-compare --tag seq-a4-horizon/m19-$q --only horizon --modes 19 \
    --slots 32 --context 512 --prefill-tokens 384 \
    --horizon-doc ../../data/bonsai2/batch-comparison/_inputs/horizon-corpus.txt \
    --horizon-streams 32 --horizon-ctx 64 --horizon-tokens 256 --horizon-every 8 \
    --seq-quant $q --rounds 0 --no-logits-dump
done
# the reference scale: the same wide route with eight-bit FFN activations
tools/run-batch-compare --tag seq-a4-horizon/m18-a8 --only horizon --modes 18 \
  --slots 32 --context 512 --prefill-tokens 384 \
  --horizon-doc ../../data/bonsai2/batch-comparison/_inputs/horizon-corpus.txt \
  --horizon-streams 32 --horizon-ctx 64 --horizon-tokens 256 --horizon-every 8 \
  --seq-quant a8 --rounds 0 --no-logits-dump
# the mode 0 anchor and the bit-equality of the kernel against its own map
tools/run-batch-compare --tag seq-a4-quality --only quality --modes 0,19 \
  --slots 8 --context 512 --prefill-tokens 384 --quality-ctx 64 \
  --seq-quant a8,a4e,a4 --quality-rows 128 --quality-shapes prefill --rounds 0
tools/seq_quant_quality.py ../../data/bonsai2/batch-comparison/seq-a4-quality \
  ../../data/bonsai2/batch-comparison/seq-a4-horizon/m19-{a8,a4e,a4} \
  ../../data/bonsai2/batch-comparison/seq-a4-horizon/m18-a8 --exact-mode 18

# throughput
tools/run-batch-compare --pin-clock --tag seq-a4-prefill2 --only prefill --modes 19 \
  --slots 8 --context 2048 --prefill-tokens 1536 --prefill-rows 128,256 \
  --seq-quant a8,a4 --rounds 4 --no-logits-dump
tools/run-batch-compare --pin-clock --tag seq-a4-decode --only decode --modes 19 \
  --slots 32 --streams 32 --context 512 --prefill-tokens 384 \
  --seq-quant a8,a4 --rounds 4 --no-logits-dump
```

Raw samples, the per-prediction scores and the `halo_env` each process saw are in
[`../../data/bonsai2/batch-comparison/seq-a4-horizon`](../../../data/bonsai2/batch-comparison/seq-a4-horizon/README.md),
`seq-a4-quality`, `seq-a4-prefill2` and `seq-a4-decode` beside it.

## Two things the next engineer should take

- **The host was the panel's bound, not the GPU.** Scoring one horizon step of 32 streams called
  `std::exp` 7.9 million times on doubles and cost about as long as eleven generation steps of the
  same batch, so a sample large enough to resolve an activation quantiser did not fit a bounded
  measurement call. `expf` on the shifted float, accumulated in double, is 6x cheaper with ~1e‑7
  relative error against differences of 1e‑3 — and every arm is scored by the same code on the same
  tokens, so the paired difference cannot see it. **A quality panel that will not fit is usually
  waiting on its scorer.**
- **Ask what a quantiser's error does with depth before arguing about its mean.** The same
  instrument that had to exist for the packed state, because its error compounds, settles an
  activation quantiser in the opposite direction: four horizon quarters that change sign and do not
  grow are a stronger statement than any single mean with a bar around it. The slope is the physics;
  the mean is the summary.

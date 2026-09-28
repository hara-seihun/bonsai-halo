# Where a batched generation step spends its time

Every published phase map of this engine is prefill-shaped: one sequence, many tokens. Aggregate
batched generation is the opposite shape, 32 sequences of one token each, and nothing had measured
it. This is that map, and it says something the prefill maps cannot: **at 32 streams the step is
dominated by recurrent state traffic, and that phase is already running at the machine's memory
bandwidth.**

> **32 streams is not the operating point.** [Sequences per generation step](decode-streams.md)
> walks the other axis of this shape and measures 597 tok/s at 64 streams against 477 at 32, with
> identical output tokens. It also re-prices the FP32 cells below on an INT8 state with a deferred
> commit, where `gdn-resident-core` is 15.6 ms rather than 48.5, and it refutes the weight-image
> explanation of the 64-row surprise in the last section of this document.

## The step

Mode 19 (A4 FFN, wide committed sequence path, resident GDN split 4), 32 slots, context 512, a
64-token prompt per stream prefilled outside timing, then one real generation step with logits and
argmax on every row. Source `1f21c1d`, `tools/batch_profile` SHA-256 prefix `460e40c2f34f`.

Untraced steps took 120.871 and 121.426 ms; the traced arm's device span was 125.890 ms, so
instrumentation costs about 4%. All four samples produced the same residual FNV-64. A second traced
sample ran to 138.064 ms and is retained in the raw file; the breakdown below is the first.

| phase | launches | ms | share |
|---|---:|---:|---:|
| `gdn-resident-core` | 48 | 48.141 | 38% |
| `ffn` | 64 | 36.386 | 29% |
| `sequence-input-projection` | 64 | 16.831 | 13% |
| `sequence-output-projection` | 64 | 13.037 | 10% |
| `sequence-core` | 64 | 5.643 | 4% |
| `head-projection` | 1 | 3.035 | 2% |
| `sequence-input-prep` | 64 | 1.663 | 1% |
| `embed`, `head-prep` | 8 | 0.122 | |

The same phases in a 128-row prefill pass are 22.0 ms of GDN core against 97.1 ms of FFN. The two
shapes are not small and large versions of each other: the largest phase changes.

## The recurrent state is the step

Hold the row count fixed at 32 and change only how those rows are distributed. Both panels run the
same 32 tokens through the same 48-layer gated-delta recurrence with the same per-token arithmetic;
one gives all 32 tokens to one sequence, the other gives one token to each of 32 sequences.

| phase | 1 sequence x 32 tokens | 32 sequences x 1 token | ratio |
|---|---:|---:|---:|
| `gdn-resident-core` | 6.393 | 48.141 | **7.53** |
| `ffn` | 32.917 | 36.386 | 1.11 |
| `sequence-output-projection` | 8.617 | 13.037 | 1.51 |
| `sequence-core` | 3.771 | 5.643 | 1.50 |
| `sequence-input-projection` | 16.776 | 16.831 | 1.00 |
| `sequence-input-prep` | 1.666 | 1.663 | 1.00 |
| `head-projection` | 3.010 | 3.035 | 1.01 |
| device span | 74.252 | 125.890 | 1.70 |

Nothing else moves by more than half. The one quantity that scales with the sequence count in that
phase is the state itself. A sequence's recurrent state is `HV * SS * SS` floats per layer,
48 x 128 x 128 x 4 = 3.146 MB, and a decode step visits each of the 48 recurrent layers exactly
once, so it loads and stores 302 MB of state per sequence per step.

| | state bytes moved | `gdn-resident-core` | rate |
|---|---:|---:|---:|
| 1 sequence | 0.302 GB | 6.393 ms | 47 GB/s |
| 32 sequences | 9.664 GB | 48.141 ms | 201 GB/s |
| the increment | 9.362 GB | 41.748 ms | **224 GB/s** |

`bench/bw` measures 242 GB/s on this box for every allocation type. The marginal state traffic is
moving at 93% of that. The convolution ring adds 123 KB per layer per sequence, 377 MB per step,
and is included in neither column; it does not change the conclusion.

## What this rules out

Within the grammar of *any schedule of the existing gated-delta recurrence that keeps FP32 state in
memory between layers*, no change to arithmetic, lane ownership, unit split, state-row split or
synchronization can take 32-stream decode's GDN phase below roughly 40 ms. That is simply the time
to move its bytes at the rate this machine moves bytes.

The state must round-trip through memory once per layer per step: 48 layers x 3.146 MB is 151 MB
per sequence, far past any on-chip storage, and each layer is visited once. There is no reuse to
find and no residency to extend, because [resident GDN](resident-gdn.md) already loads each
sequence's state exactly once per layer.

This does not bound the GDN recurrence in *prefill*, where the same 302 MB is amortized over 128
tokens and the phase runs at 47 GB/s, nowhere near the roof. The 4x gap between the prefill GDN
phase and its arithmetic budget, recorded in [the resident GDN notes](resident-gdn.md), is still
open and is still a scheduling question. The two shapes need different answers.

**That difference has since been measured on one change rather than argued.** Batching the token
loop's row reductions removes 24.5% of the phase's issue slots and is worth -15.1% in a 128-row
pass and **+0.03% here**, same kernel, same build, same arms — the bound above being paid in full.
If you are optimising the recurrence, [read which shape you are optimising](gdn-token-reduce.md)
before you price the work.

## What it leaves open, priced

**Fewer state bytes.** Halving the state width removes about 20 ms of the step, roughly 318
aggregate tok/s. It is a numerical change to a recurrence that accumulates over the whole sequence,
so it needs its own quality evidence and would be an explicit alternative, not a default.

**Stream scaling is now predictable.** Each additional stream adds 151 MB of state traffic per
step, about 0.62 ms at this machine's bandwidth, plus its share of the row-scaled phases. Aggregate
generation cannot be bought by raising the stream count alone.

**More tokens per state residency** was the third lever here, priced by arithmetic at about 410
aggregate tok/s for a two-token step. That estimate is now replaced by measurement, below, and it
was wrong in both directions: a two-token step is worse than 410 and a four-token step is much
better.

## A step is cheap in tokens and expensive in sequences

The cost above is per step, not per token. The same 9.664 GB of state moves whether each sequence
advances one token or four, so the question is what the rest of the step charges for the extra
rows. `--decode-tokens K` walks that axis as a case inside one process, so every K shares one
build, one warmed device and one clock.

Mode 19, 32 streams, 64-token prefix per stream, logits on every row, source `ce4ba22`, one round,
cases interleaved. `K` is rows per sequence in the step, so the step is 32K rows.

| phase | K=1 (32 rows) | K=2 (64 rows) | K=4 (128 rows) | ms per extra row |
|---|---:|---:|---:|---:|
| `gdn-resident-core` | 48.455 | 48.616 | 53.040 | **0.048** |
| `ffn` | 36.043 | 68.891 | 85.350 | 0.514 |
| `sequence-input-projection` | 16.605 | 24.302 | 39.102 | 0.234 |
| `sequence-output-projection` | 12.924 | 28.467 | 28.382 | 0.161 |
| `sequence-core` | 5.410 | 6.569 | 12.992 | 0.079 |
| `head-projection` | 2.951 | 5.650 | 11.405 | 0.088 |
| `sequence-input-prep` | 1.583 | 2.253 | 3.552 | 0.021 |
| device span | 125.129 | 186.182 | 235.939 | |
| step, untraced | **121.074** | **181.352** | **230.375** | |
| aggregate tok/s at full acceptance | **264.3** | **352.9** | **555.6** | |

A four-token step produces 2.10x the tokens per second of a one-token step on the same executable.
That is the largest single factor available in this workload and it needs no numerical change.

The reason is the first row. An extra token costs the recurrent phase 0.048 ms; an extra stream
costs it 1.347 ms. **A token is 28 times cheaper than a sequence in the phase that dominates the
step**, because the state round trip is indexed by (sequence, layer) and a token only rides along.
Everything else in the step is priced per row and does not care which sequence the row came from.

The marginal token is also getting cheaper as the step widens: 1.884 ms per token going from K=1 to
K=2, then 0.766 ms from K=2 to K=4. That is backwards from how a batch usually behaves, and it is
not a state effect. Two row-count rules fitted at 32 and 128 rows both misbehave at 64.

### Where the recurrent phase stops being a memory machine

The recurrent row deserves a closer reading, because it is not flat and it is not linear either.
`gdn-resident-core` costs 48.455, 48.616 and 53.040 ms at K=1, 2 and 4. The second token is free
and the third and fourth are not.

That is a crossover, and it lands exactly where a straight line through the two wide cells says it
should. Fit the K=2 and K=4 cells: 2.212 ms per extra token per sequence, intercept 44.192. Run
that line back to K=1 and it predicts 46.404 ms, which is 2.05 ms below the 48.455 measured. So the
phase is at its memory floor at K=1, the K=2 token hides inside the slack in that floor, and from
K=2 up the phase is an instruction-issue machine costing 2.212 ms per token.

This matches what the resident-state work found in prefill, where the phase tracks an issue-slot
count to 89 to 92% and has no scheduling story left in it. The decode shape adds the crossover
point: one token per sequence is the only width at which that phase is bandwidth-bound. Two
consequences. Fewer state bytes helps only at K=1, because above it the state stream is no longer
the binding constraint. And instruction count in the token loop, which is worth nothing at K=1,
becomes worth 2.2 ms per token at the K=4 operating point this document recommends.

## What a drafter has to clear

`forward_batch` refuses a loaded drafter, because the wide path does not capture drafter features,
so nothing drives these rows today. The step costs above say exactly how good a batched drafter has
to be. To beat one-token generation at 264.3 tok/s, a K-row verify step needs mean accepted tokens
per sequence per step `a` with `32a / step_seconds > 264.3`:

| step | step ms | break-even `a` | as a share of K |
|---|---:|---:|---:|
| K=2 | 181.352 | 1.498 | 74.9% |
| K=2, dense FFN arm | 172.203 | 1.422 | 71.1% |
| K=4 | 230.375 | 1.902 | 47.6% |

K=4 is the operating point worth building for. The single-stream drafters already land 2.3 to 4.5
accepted tokens per step, comfortably past 1.902, while K=2 demands 75% of drafted tokens and is a
bad bet. Anyone integrating a batched drafter should size it for four drafted rows per sequence and
measure acceptance, not step time.

## The 33 to 127 row band is unfitted

> **Both suspects in this section have since been measured and neither is open.** The FFN image
> threshold was fitted per stage in [the decode-shape panel](ffn-decode-shape.md) - pair gate/up
> with dense down at 64 rows, which already holds the 11.5 ms below and beats the all-dense arm this
> section forced by 1.3 ms. The output projection's 64-row cell is not spill and not occupancy:
> [the token-tile ladder](row-band-fit.md) walks the shape at fixed width under both operand widths,
> finds this section's three cells confounded by moving the row count and the tile count together,
> and finds the deployed rule at its optimum on the four-bit operand the route now defaults to.

Both surprises in the table sit at 64 rows, and both come from a shape rule fitted at 32 and 128
rows with nothing in between.

`ffn` costs 1.03 ms per row from 32 to 64 rows and 0.26 ms per row from 64 to 128. The A4 module
picks its weight image with `Npad <= 32` and hands anything wider to the pair code. Forcing the
image at 64 rows, same build, same process:

| image at 64 rows | `ffn` | step, untraced | aggregate tok/s |
|---|---:|---:|---:|
| dense five-trit | **57.272** | **172.203** | **371.7** |
| pair code (what the rule picks) | 68.780 | 181.662 | 352.3 |

**This table has since reversed and the image rule is off the hook.** Re-measured across the whole
band on `b4e5365`, pair codes cost 41.022 ms at 64 rows against dense five-trit's 49.274, and the
rule is within 2.2% of the best arm at every width in 40..128 rows. The band is still non-monotone,
but the term is not the image: `FETCH_SIZE` per FFN launch is identical at 32, 64 and 128 rows.
[Sequences per generation step](decode-streams.md) carries both panels.

The dense image is 11.508 ms cheaper, 16.7% of the phase and 5.5% of the whole step, and the
residual FNV-64 is `11326044657355096693` in all four samples, so the two arms are bit-identical
here as well as at 32 and 128 rows. The in-process controls are the phases this cannot touch:
input projection 24.588 against 24.308, output projection 28.577 against 28.290, recurrent core
49.178 against 48.908. Pair still wins at 128 rows, so the crossover is somewhere in 65 to 127 and
nobody has measured it. Raw: `decode-step/k2-image.json`. The constant lives in
`kernels/ffn_batch.hip`, which this iteration does not hold.

`sequence-output-projection` is the other one, and it is stranger. It costs 12.924 ms at TT=2,
28.467 at TT=4 and 28.382 at TT=8. Going from two token tiles to four costs 15.5 ms; going from
four to eight costs nothing. A 128-row output projection is free relative to a 64-row one. The
weight stream is identical in all three cells and the wave count is fixed at 320 by `w.n = D = 5120`
regardless of rows, so neither traffic nor grid occupancy explains it, which leaves register
pressure inside `project<TT,ADD,PLANAR>`. At 64 rows even TT=2 with two y-groups, which re-reads
the whole weight stream, should land near 25.8 ms and beat the TT=4 cell. That smells like spill
and it belongs to whoever holds `kernels/sequence_batch.hip`.

## Reproduce

```sh
make -j8 tools/batch_profile
# 32 sequences, one token each: the aggregate generation shape
tools/run-batch-compare --profile-tool --modes 19 --decode-streams 32 --decode-prompt 64 \
  --rounds 2 --warmup-ms 800 --rows 32 \
  --out ../../data/bonsai2/batch-comparison/decode-map/m19.json
# the matched prefill shape: the same 32 rows in one sequence
tools/run-batch-compare --profile-tool --modes 19 --rows 32 --heads 1 --traces 1 --rounds 1 \
  --out ../../data/bonsai2/batch-comparison/decode-map/prefill-r32.json
```

```sh
# the step-width axis: 32 streams advancing 1, 2 and 4 tokens, interleaved in one process
tools/run-batch-compare --profile-tool --modes 19 --decode-streams 32 --decode-tokens 1,2,4 \
  --decode-prompt 64 --rounds 1 --warmup-ms 500 --rows 32 \
  --out ../../data/bonsai2/batch-comparison/decode-step/tokens-m19.json
# the FFN weight image at 64 rows
tools/run-batch-compare --profile-tool --modes 19 --decode-streams 32 --decode-tokens 2 \
  --ffn-image 1,2 --decode-prompt 64 --rounds 1 --warmup-ms 400 --rows 32 \
  --out ../../data/bonsai2/batch-comparison/decode-step/k2-image.json
```

`--decode-streams N` allocates N slots, prefills each from a distinct slice of the document, and
profiles one generation step; `--decode-prompt` sets that prefix. Below 32 streams the engine
leaves the wide sequence path entirely, so `--decode-streams 16` measures a different route and is
not a scaling point; that run is retained in `decode-map/m19-s16.json` with this caveat.

`--decode-tokens K` gives every stream K rows in the step and walks K as a case axis, so the widths
share a clock. 32 streams at K=4 is 128 rows, the pass limit, so K above 4 needs fewer streams. A
three-width round with both trace arms fits a 55-second call; the default `--rounds 3` and
`--warmup-ms 2000` do not.

One honesty note on these numbers. A K-row step feeds K copies of one token id per sequence, which
costs exactly what a verify step costs because the arithmetic does not depend on the token values,
but the tokens are not drafted and the aggregate rates are therefore at full acceptance. They are
step costs, not a throughput claim for a system that exists. The break-even table above is the part
to quote.

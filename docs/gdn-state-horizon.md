# What the recurrent state's precision costs, and the shape that could have shown it

[The state coordinate](gdn-state-pack.md) landed int16 and int8 storage for the gated-delta state,
measured them at +21.9% and +34.3% aggregate generation, and then left fp32 as the default
everywhere, because the quality evidence could not rank them. A teacher-forced window of 64 rows
scored int16 at +0.0024 nats and int8 at +0.0035, both inside their own error bars, with a
sixteen-fold difference in rounding between them. That is not a result. It is an instrument at the
end of its range.

This is the measurement that ranks them, and the answer is that nothing does.

## The shape the window could not reach

A packed state is re-rounded every time it crosses memory. In a generation step that is once per
token per layer. In a 128-row prompt pass it is once per 128 tokens per layer. So the number of
roundings a quality workload applies is the number of **commits**, not the number of rows, and a
64-row teacher-forced pass applies at most a handful.

`tools/batch_compare --only horizon` runs the generation shape on fixed tokens. Thirty-two streams
take distinct 321-token windows of a 75,945-token document, each stream is given 64 tokens of
context, and then the engine walks 256 single-token steps per stream. Every arm is fed exactly the
same tokens, so the comparison pairs per (stream, step) against a fixed target and no arm's own
output is privileged. The distribution is scored every eighth step: 1024 paired predictions per arm
over 256 commits.

Teacher forcing is not a convenience here. Scoring an arm's own free-running continuation under the
fp32 engine, which is the experiment the previous iteration proposed, cannot rank coordinates at
all: the fp32 arm's greedy continuation is by construction the reference's own argmax path, so it
takes the minimum achievable score whatever its quality, and every other arm is penalised for
diverging rather than for being wrong.

## The result

Mode 19, one arm per process, 1024 scored predictions each. `f32pk` stores fp32 values through the
packed kernel instantiations, which have different registers and their own cooperative grid.

| arm | NLL | perplexity | top-1 | agrees with fp32 | dNLL vs fp32 |
|---|---:|---:|---:|---:|---:|
| `f32` | 3.04150 | 20.9366 | 0.3975 | 1.0000 | reference |
| `f32pk` | 3.04150 | 20.9366 | 0.3975 | **1.0000** | **+0.00000 +- 0.00000** |
| `i16` | 3.03732 | 20.8493 | 0.4082 | 0.8936 | -0.00418 +- 0.00769 |
| `i8` | 3.02500 | 20.5939 | 0.4082 | 0.8926 | -0.01650 +- 0.00820 |

Errors are stream-clustered, treating each of the 32 document windows as one observation, because a
stream's state error persists across its own steps. The naive per-prediction errors are 0.00881 and
0.00873, so the clustering barely moves them and within-stream persistence is weak.

**The control has no noise at all.** `f32pk` reproduces every one of the 1024 predictions exactly,
in a separate process, through different kernels. So the engine is bit-deterministic across
processes, the pairing is exact, and everything the packed arms move is the storage rounding and
nothing else. An instrument with a zero floor is worth more here than a bigger sample.

Neither packed arm is distinguishable from fp32, and both point the same way. At two sigma, int16
costs at most +0.011 nats and int8 at most +0.000. For scale, the A4 FFN map this repository already
ships as its batched headline costs **+0.0615 nats** on the same kind of instrument.

Nor is there a tail hiding under the mean, which is the risk that matters for a coordinate that
feeds its own output back. Per-stream mean dNLL runs from -0.065 to +0.112 for int16 and -0.132 to
+0.062 for int8, so **int8's worst stream is better than int16's**, 12 of 32 streams are worse than
fp32 under int16 against 10 under int8, and no single prediction in either arm is more than 2 nats
worse than its fp32 pair. The largest single paired difference is 1.589 nats for int16 and 1.586 for
int8, which is the same saturation again at the other end of the distribution.

That per-stream spread is also what sets this instrument's resolution, and it is worth knowing
before anyone spends a run trying to sharpen it. The per-stream standard deviation is about 0.045
nats, so the clustered error over 32 streams is 0.045 / sqrt(32) = 0.008, which is exactly what the
table reports. **A longer horizon does not tighten it and more streams do.** Halving the error bar
needs 128 independent document windows, which means 128 slots and 128 prompts, not 1024 steps.

And there is no horizon trend, which is the thing this workload exists to look for:

| arm | steps 7-63 | 71-127 | 135-191 | 199-255 |
|---|---:|---:|---:|---:|
| `i16` | +0.00810 +- 0.01827 | +0.00292 +- 0.02088 | -0.00499 +- 0.01228 | -0.02274 +- 0.01305 |
| `i8` | -0.01420 +- 0.01705 | -0.02694 +- 0.01656 | -0.01131 +- 0.01512 | -0.01357 +- 0.01695 |

## The theory that predicted a trend, and why it does not reach the output

Rounding injected into a decaying recurrence should accumulate. Write the storage error as
`E_t = g_t E_{t-1} + q_t`, where `g_t` is the same decay the recurrence applies and `q_t` is the
fresh rounding of one commit. The quantiser is relative, so `q_t` scales with the state's own
magnitude, and the state accumulates by the same geometric law. Both sides come out as
`1 / (1 - g^2)` in energy, and the relative readout error at horizon is

    eps / sqrt(1 - g^2)

with `eps` the single-commit relative error. `docs/gdn-map/results/gate-stats.txt` puts the median
head near `g = 0.973` and a slow head near `g = 0.996`, which is an amplification of 4.3x and 11x,
and time constants of about 19 and 125 steps. Both are well inside 256.

The single-commit numbers are not close either. On the real state array, int8 measures a per-row
relative readout error of **1.10e-2** and int16 **4.27e-5**, a factor of 257. Put through the
amplification, int8 should be arriving at the readout with five to twelve percent relative error by
step 200 while int16 sits near 3e-4.

None of that reaches the output. The clearest evidence is not the means, which are noisy, but the
spread:

- the paired per-prediction standard deviation is **0.282 nats for int16 and 0.279 for int8**,
- and 10.6% of argmaxes flip for int16 against 10.7% for int8.

Those are the same numbers for a 257-fold difference in perturbation size. A packed state moves
individual predictions a long way, and it moves them exactly as far whether the perturbation is one
part in a hundred or one part in twenty-five thousand. So the trajectory decorrelates on contact and
then stops caring how hard it was hit, and only the mean ranks arms. That is the same saturation the
previous iteration found in logit differences, one level deeper: it is not an artefact of comparing
logits, it is a property of running a 64-layer recurrent model forward from a perturbed state.

The honest scope: 256 commits, one document, one model, one decode shape. A slow head with
`g = 0.999` has a 500-step time constant and is not resolved here. What is ruled out is the version
of the accumulation story that predicts int8 diverging from int16 inside a few hundred tokens, which
is the version that was holding the default.

## Choosing the coordinate offline, before building one

`tools/batch_compare --only state-dump` writes the fp32 state of a warmed sequence, and
`tools/gdn_state_format.py` searches the whole family against it. One GPU run answers every
candidate, where measuring each on the device costs a kernel, a build and a panel.

It scores per-row relative L2 error, and that choice is forced: both consumers of a state row are
dot products against an L2-normalised vector, so for a row `s` with elementwise error `e` the
readout error has mean square `||e||^2 / 128` against a signal of `||s||^2 / 128`. The ratio
`||e|| / ||s||` is the whole story. Absolute RMS error would rank a large row and a small one on the
same scale, and per-element relative error would weight a near-zero element as heavily as the one
carrying the row.

Five layers, 48 heads each, after 384 teacher-forced tokens:

| coordinate | bits/element | rel L2 | p99 | SNR |
|---|---:|---:|---:|---:|
| `i16` per row | 16.25 | 4.269e-05 | 8.691e-05 | 87.4 dB |
| `fp16` | 16.00 | 2.083e-04 | 2.900e-04 | 73.6 dB |
| `bf16` | 16.00 | 1.656e-03 | 2.311e-03 | 55.6 dB |
| `i8` per (row, 32 col) | 9.00 | 7.552e-03 | 1.206e-02 | 42.4 dB |
| `i8` per row | 8.25 | 1.096e-02 | 2.227e-02 | 39.2 dB |
| `i8` per column | 8.25 | 3.374e-01 | 9.984e-01 | 9.4 dB |
| `i6` per row | 6.25 | 4.430e-02 | 8.244e-02 | 27.1 dB |

Three things fall out, and two of them close questions that were open.

**The dynamic range is in the rows.** One scale per column costs the same 128 scales per head as one
per row and measures 3.374e-1 against 1.096e-2, thirty times worse. The value channel carries the
spread and the key channel does not, so the row block the kernel already uses is the right block and
there is no cheaper one. That also means nobody needs to build the cross-wave reduction a column
block would require.

**A finer block is not worth its bytes.** Rows have a crest factor of 4.07 to 4.59 at the median and
7.7 to 10.4 at p99, but a row's four 32-column groups differ in magnitude by only 1.35 to 1.48. So a
per-(row, 32-column) scale, which this lane had open as the fix if int8 turned out too coarse, buys
1.45x in readout error for 9% more bits and 12.5% more state traffic. Against a horizon measurement
that cannot see a 257x difference, paying traffic for 1.45x is backwards. It is not implemented, and
this is the reason.

**int16 beats fp16 by 4.9x on the real array**, 4.269e-5 against 2.083e-4, which is the
shared-exponent argument in [the state coordinate](gdn-state-pack.md) measured rather than asserted.
bf16 is another eightfold worse again and has no business near this state.

## Throughput, on one build, in one process

Mode 19, two rounds with the arm order reshuffled between them, 384-token document in 128-row passes
and 32 independent streams generating.

| coordinate | prefill, 128 rows/pass | 32-stream aggregate generation |
|---|---:|---:|
| `f32` | 792.7, 740.8 | 299.9, 300.3 |
| `i16` | 670.6, 792.9 | 365.7, 366.0 (**+21.9%**) |
| `i8` | 805.3, 792.4 | **411.6, 412.8 (+37.3%)** |

Generation separates cleanly and repeats to within 0.3% in both rounds, with the arms in a different
order each round, so it is not a warming artefact. int8 is 12.6 points ahead of int16, and int16
reproduces the +21.9% the state-coordinate iteration published. Prefill is a null by construction,
because a pass pays the state round trip once per layer for 128 tokens, and it measures as one: the
six prefill cells span 740.8 to 805.3 with no ordering by arm, a spread larger than any difference
between them.

The narrow shapes are not a null, which the previous iteration's arithmetic said they would be. One
process, three rounds, arm order reversed in the third:

| shape | route | `f32` | `i8` | |
|---|---|---:|---:|---:|
| 1 stream generating | deployed, the single-stream path | 34.1, 34.3, 34.1 | 34.5, 34.4, 34.6 | +1.1% |
| 8 streams generating | sliced deployed | 121.9, 121.5, 121.1 | 129.8, 130.4, 130.2 | **+7.2%** |

Every cell repeats within 0.5% and the sign is the same in all three rounds including the reversed
one. **Measure this pair in one process or do not measure it.** Two `--engine --bench` runs twelve
minutes apart, on binaries whose single-stream path differs only by the state format, read 33.25 and
31.52 tok/s, which says int8 is 5.2% slower. It is not. Nine engineers were compiling on this host;
the cross-process spread is several times the effect.

Single-stream stays small for the reason the state-coordinate document gave: a token there moves
5.9 GB of weights beside 0.302 GB of fp32 state, so taking the state to 0.076 GB is about a
millisecond of a thirty-millisecond token. The eight-stream cell is where it starts to show,
because the state is paid eight times against one pass over the weights.

## What this changes, and what it does not

**int8 replaces int16 as the packed coordinate to use.** It is 12.6 points of aggregate generation
ahead, its quality is bounded no worse by the only instrument that can see either of them, and the
finer block that was held in reserve for it is now measured and not worth building. Anyone building
on top of a packed state should build on int8: `HALO_GDN_STATE=i8`, or `--gdn-state 3` as a case axis
in both measurement tools.

**The process default stays fp32, and the reason is the reference, not the size of the win.**
`r_gdn_fmt` is one `__constant__` for the whole process, not a property of a route, so moving it
moves every route at once. Three things depend on the fp32 state being what an unconfigured process
runs: the `--v1` per-op path refuses to read a state it cannot decode, `HALO_GDN_COMPARE` is armed
only in fp32 because the sliced and resident routes commit at different frequencies and round a
different number of times under a packed coordinate, and every residual FNV-64 this repository
publishes as a regression anchor is an fp32-state value. Losing all three to gain 1.1% on the shape
the resident server actually generates is a bad trade. It stops being a bad trade the day something
here serves eight concurrent streams, where the same flag is worth 7.2%, or thirty-two, where it is
worth 37.3%.

**The state has stopped being the large term in a generation step.** It was 38% of a 32-stream step
in fp32 and about 48 ms; at int8 it is roughly a quarter of that. The next lever in that shape is the
FFN and the two sequence projections, not the recurrence, and the [decode map](decode-map.md)'s
bounded negative about the state round trip is now spent.

## Reproducing it

```sh
make -j8 tools/batch_compare
# The horizon corpus: any long document works, this is the one the table used.
cat README.md PLAN.md orchestration/HANDOFF.md docs/decode-map.md docs/wide-sequence.md \
    docs/throughput-budgets.md > ../../data/bonsai2/batch-comparison/_inputs/horizon-corpus.txt

# One arm per process. Quality arms are safe to split across processes and timing arms are not:
# the engine is bit-deterministic on fixed inputs, which the f32pk control proves.
for s in 0 4 2 3; do
  tools/run-batch-compare --tag gdn-state-horizon/h256-$s --only horizon --modes 19 \
    --slots 32 --context 512 --prefill-tokens 384 \
    --horizon-doc ../../data/bonsai2/batch-comparison/_inputs/horizon-corpus.txt \
    --horizon-streams 32 --horizon-ctx 64 --horizon-tokens 256 --horizon-every 8 \
    --gdn-state $s --rounds 0 --no-logits-dump
done
tools/gdn_state_quality.py ../../data/bonsai2/batch-comparison/gdn-state-horizon/h256-*

# The coordinate family, offline, from one dump.
tools/run-batch-compare --tag gdn-state-format --only state-dump --modes 19 --slots 1 \
  --context 512 --prefill-tokens 384 --state-dump-ctx 384 --rounds 0 --no-logits-dump
tools/gdn_state_format.py ../../data/bonsai2/batch-comparison/gdn-state-format

# Throughput, all three coordinates interleaved in one process.
tools/run-batch-compare --tag gdn-state-frontier --modes 19 --gdn-state 0,2,3 \
  --prefill-rows 128 --streams 32 --rounds 2 --only prefill,decode --no-logits-dump
```

Raw samples, the per-prediction scores and the `halo_env` each process saw are in
[`../../data/bonsai2/batch-comparison/gdn-state-horizon`](../../../data/bonsai2/batch-comparison/gdn-state-horizon),
`gdn-state-format` and `gdn-state-frontier` beside it.

## What the instrument is good for next

The horizon workload is not about the state. It is a paired, absolute, zero-floor way to ask what
any approximate numerical map costs the model's output, and this lane has several of those either
running or parked:

- four-bit activations on the QKV and GDN input projections, whose first half is explicitly a
  quality panel rather than a kernel,
- a deferred state commit across C steps, which reassociates the readout and rounds the packed state
  C times less often, so its C and this document's coordinate are the same experiment axis,
- a lower-precision drafter, where the output cannot change at all and only the acceptance rate can,
  so it needs the opposite instrument.

Run the `f32pk`-style control every time. A quality arm that reproduces its reference exactly, in a
separate process, through different kernels, is what separates a real difference from an engine that
wanders.

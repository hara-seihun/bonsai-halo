# Activation quantization is a marker, not the slow call's budget

The installed Qwen3.6-35B-A3B depth-1024 trace has a striking pattern: a slow Q8 or routed Q4 matvec nearly always follows a slow activation quantizer of the same shape. That does not mean optimizing the quantizer will recover the matvec time. This accounting separates their exposed device durations on the same single queue. It changes the next native experiment: time the paired operations without the profiler, then investigate the persistent matvec regime instead of replacing activation coding on the strength of correlation.

On the seven warm decode tokens, 291 `quantize_q8_1` launches per token consume **0.696 ms/token** in total, or 3.34% of the 20.846 ms/token summed device duration. Erasing every one of these launches at zero replacement cost removes at most their 0.696 ms of *recorded kernel duration*. That is not a bound on a redesigned packed consumer, on changed launch gaps, or on unprofiled wall time.

| Paired family, 40 calls/token | Ordinary / slow calls over seven tokens | All preceding quantizers, ms/token | Slow quantizer excess over ordinary mean, ms/token | Slow matvec excess over ordinary mean, ms/token |
| --- | ---: | ---: | ---: | ---: |
| Q8_0 width 8192 | 230 / 50 | .09175 | .03396 | .17148 |
| Routed Q4_K width 512 | 217 / 63 | .10190 | .04432 | .56961 |
| Both | 447 / 113 | .19365 | .07828 | .74109 |

The two family thresholds, 105 µs for wide Q8 and 80 µs for routed Q4, come from the [paired trace classification](qwen-moe-trace-overlap.md). Excess means each slow event's duration minus the ordinary event mean within its family. It is a descriptive replacement calculation, not an achievable optimization. The matvec excess is 9.47 times the paired quantizer excess. Even the more generous thought experiment of deleting **all** 80 paired quantizer calls per token removes just .194 ms of recorded device work. This doesn't say that a faster quantizer cannot affect the following kernel through caches, graph scheduling or power. It says the preceding quantizer's own measured time is too small to account for the correlated slow matvec episode.

The input is the installed source `b4c67ced9f6bac6b1661b25714946d999374e06f`, the pinned GGUF and the counter-free trace from [the installed phase report](qwen-moe-q8-unprofiled.md). The [receipt](../../data/qwen-moe/q8-unprofiled/activation-bound.json) hashes the trace, installed receipt, model, binaries and analysis script and retains all eight token sums and every paired event. The script sorts by device start timestamp, excludes the two synthetic-depth setup evaluations and labels token 0 cold. It checks 1,565 dispatches and 291 quantizers on every decode token and forty Q8 and forty Q4 pairs. The seven warm tokens are reused observations from that trace, not seven independent unprofiled runs. No GPU reservation, selected runtime or service changed.

Reproduce in seconds:

```sh
python3 tools/qwen-moe/activation_bound.py \
  ../../data/qwen-moe/q8-unprofiled/trace/gpu-host/1886571_kernel_trace.csv \
  --receipt ../../data/qwen-moe/q8-unprofiled/receipt.json \
  --output ../../data/qwen-moe/q8-unprofiled/activation-bound.json
```

The next experiment is an interleaved unprofiled complete-model panel with per-layer activation and matvec markers, preserving output-head rows and occupied depth. If the paired regime persists, measure whether ordinary Q8 consumption or routing/graph scheduling accounts for the rest of the conditional gap. A quantizer-only port is not supported by these device-duration numbers. The native selected arithmetic map stays unchanged.

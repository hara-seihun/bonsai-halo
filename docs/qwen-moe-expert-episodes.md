# Layer-persistent slow periods in the installed Qwen trace

The [paired expert trace](qwen-moe-expert-latency.md) has 64 slow routed Q4 calls across eight decode tokens. I asked whether they were random slow episodes around expert dispatch or whether layer position predicts them. The answer changes the next native experiment: the same layer indices repeatedly slow *unrelated* dispatches before the router and expert gather. An expert-only Q4 operand rewrite would leave those delays in place.

The observation is the retained `rocprofv3 --kernel-trace` of the selected `b4c67ced9f6bac6b1661b25714946d999374e06f` runtime, at occupied depth 1024. The [installed receipt](../../data/qwen-moe/q8-unprofiled/receipt.json) fixes the GGUF, binary, profile workload and GPU-wrapper restoration. [`expert_episodes.py`](../tools/qwen-moe/expert_episodes.py) reads the same eight 1,565-dispatch decode spans as the parent study. The [derived receipt](../../data/qwen-moe/q8-unprofiled/expert-episodes.json) hashes its source and raw CSV and keeps the slow layers by token, complete per-layer counts, and compared medians. This was a CPU analysis of the existing GPU trace. No new GPU reservation, executable or service change.

I excluded the first measured decode token from the warm comparison. The 80 µs Q4 split lies in the gap between the observed 67.477 µs ordinary and 89.757 µs slow calls. Label a layer persistent when it is slow in at least two of warm tokens 1–3, then predict tokens 4–7 without refitting. The selected layer IDs are `1,2,3,6,7,13,19,26`. They predict 29 of the 37 held slow calls, with three false positives among 123 held ordinary calls. That is 78.4% recall and 90.6% precision. Layers 1, 3, 6, 13, 19 and 26 are slow on every one of the seven warm tokens. Layer 33 emerges later and accounts for four of the eight false negatives; this is not a perfectly stationary layer mask.

| Same warm layer-token observation | Ordinary Q4, n=217 | Slow Q4, n=63 |
| --- | ---: | ---: |
| Pre-router RMS norm, same kernel/grid, median µs | 3.000, n=208 | 10.520, n=63 |
| Router top-k, median kernel µs | 4.399 | 19.720 |
| Routed Q4 gate/up, median kernel µs | 50.518 | 114.556 |
| Routed down, median kernel µs | 38.598 | 92.117 |
| Shared-expert Q8, median kernel µs | 13.799 | 20.400 |
| Gap before router, median µs | 2.759 | 8.080 |

The pre-router control is the exact `rms_norm_f32<1024, true, false>` instantiation with grid X=1024 among the three immediately preceding the router. Nine ordinary samples lack this control in that three-dispatch window and are excluded only from that row. The Q4 grid is always `(16384,8,1)` and router/Q4 are both on queue 2. This identical RMS norm slows 3.5-fold and the router itself 4.5-fold at the same layer positions where the Q4 call slows 2.3-fold. In layer 19, however, the router is ordinarily fast while Q4 remains slow on all warm tokens. The slow period can start between dispatches, so one must not equate the classifier with a fixed, per-layer weight property or a clock measurement. The preceding kernel and the gate/up do not even read the same expert bank.

This is a measured negative for the claim that *all* 1.935 ms/token of conditional routed-expert gap is a fixed Q4/Q5 operand cost. It does not subtract any measured wall time, establish a physical DRAM count, or diagnose whether the layer persistence comes from graph allocation, power/clock behavior, cache or profiling. HIP event durations and gaps are not interchangeable with unprofiled full-model phase time. The trace is one eight-token profile, and the predictor's four held tokens are from that same run.

The useful next experiment is a paired, counter-free unprofiled depth-1024 panel with per-layer device markers and one unchanged full-model wall measurement under the GPU wrapper. Keep prompt, head rows, binary and route workload fixed. Repeat the held layer mask across runs; measure clocks around the router and a small pre-router kernel as well as Q4/down. If the same mask persists without the profiler, investigate per-layer allocation or graph scheduling before changing quantized dot instructions. If it vanishes, optimize the ordinary expert path against a newly measured unprofiled phase budget. Neither outcome is an excuse to call this isolated profiler ratio a whole-model gain.

Regenerate the CPU receipt with:

```sh
python3 tools/qwen-moe/expert_episodes.py \
  ../../data/qwen-moe/q8-unprofiled/trace/gpu-host/1886571_kernel_trace.csv \
  --output ../../data/qwen-moe/q8-unprofiled/expert-episodes.json
```

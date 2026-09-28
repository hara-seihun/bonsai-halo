# Ordinary Q8 image-size fit exposes a wide-call penalty in the selected decode trace

**Question.** Can the selected Qwen native Q8 phase's conditional one-read gap be treated as one uniform bandwidth/dispatch cost? This is a shape-conditioned *counter-free profiler* result, not the unprofiled device-marker panel still needed before changing the engine. The target is the installed GGUF and native source `b4c67ced9`; the original [depth-1024 full-head trace and matched unprofiled wall receipt](qwen-moe-q8-unprofiled.md) retain the executable, model, input and clock provenance.

[`q8_wall_probe.py`](../tools/qwen-moe/q8_wall_probe.py) joins each of eight complete 1,565-dispatch decode tokens to **all 210 Q8 calls** and the exact tensor inventory. There are forty 17,825,792-byte Q8 calls, thirty 8,912,896-byte gate calls plus forty 8,912,896-byte 2,048-output calls, forty fused 2,228,224-byte shared gate/up calls, and sixty 1,114,112-byte calls. The per-layer large/small 2,048-output calls alternate; the source order and full inventory, not their durations, assign image sizes. It excludes calls at or within 64 dispatches after a ≥0.5-ms profiler gap, as fixed by the earlier [two-trace gap study](qwen-moe-q8-gap-separation.md). These pauses appear at recorder-shaped ordinals and cannot be priced as native host idle. First four measured tokens fit the model; last four are held.

| Exact Q8 image bytes/call | Inspection ordinary calls, median µs | Held ordinary calls, median µs | Held image bytes / median duration |
| ---: | ---: | ---: | ---: |
| 1,114,112 | 182, 7.64 | 168, 7.7395 | 143.95 GB/s |
| 2,228,224 | 114, 13.439 | 104, 13.6395 | 163.37 GB/s |
| 8,912,896 | 213, 43.559 | 200, 44.3385 | 201.02 GB/s |
| 17,825,792 | 117, 94.397 | 108, 94.057 | 189.52 GB/s |

The inspection 1.114/8.913-MB medians fit the *explicit restricted cost grammar* `time = 2.5087 µs + bytes / 217.12 GB/s`, with unchanged kernel launch placement and no shape penalty. It predicts **84.609 µs** for a 17.826-MB call. The held wide median is **94.057 µs**, **9.448 µs/call** above that extrapolation; every withheld token's wide median exceeds the frozen fit by **8.51–10.95 µs**. Forty such calls yield **0.378 ms/token** as a descriptive extra *within this profiled fit*, around 1.85% of the trace's 20.398-ms summed device duration. The independent **same-executable depth-zero trace** repeats the size effect over seven warm complete tokens: its inspection fit predicts **83.914 µs**, while 100 held ordinary wide calls take **94.237 µs median**, **10.323 µs** excess (9.08–11.76 in each held token). The depth-zero 8.913-MB and 1.114-MB held medians are 44.299 and 7.899 µs. These two recorded profiler runs support a stable within-trace shape penalty, not an attainable full-model speedup or a lower bound on a different program. The 242-GB/s once-read roof is a bandwidth comparator, not a measured DRAM count.

**Decision.** One uniform affine byte-stream-plus-dispatch model fails held wide calls even away from recorder gaps. If a native unprofiled marker panel confirms the difference, investigate the 8,192-output Q8 row/grid/issue behavior and actual physical bytes separately from the short 1-MB launch floor. A blanket Q8 scale dictionary (saving only 1.105% conditional whole-model one-read bytes), another wave-width trial or layer-ID special case has no support from this result. Do not port a kernel on this trace alone: event markers must include Q8/router/Q4/Q5 seams with a complete-head matched wall panel and service restoration. The short-call and wide-call distributions are stable on held tokens, but profiling can perturb *both* relative to ordinary execution.

[Depth-1024 CPU receipt](../../data/qwen-moe/q8-unprofiled/q8-ordinary-cost.json) and [independent depth-zero receipt](../../data/qwen-moe/q8-unprofiled/q8-ordinary-depth0.json) contain every token/ordinal, gap distance, duration and assigned image byte count, as well as SHA-256 for the analyzer, both original traces and GGUF inventory. No GPU command, installed executable, resident service or numerical map changed during this analysis.

```sh
python3 tools/qwen-moe/q8_wall_probe.py \
  ../../data/qwen-moe/q8-unprofiled/trace/gpu-host/1886571_kernel_trace.csv \
  ../../data/qwen-moe/traffic.json \
  --out ../../data/qwen-moe/q8-unprofiled/q8-ordinary-cost.json
python3 tools/qwen-moe/q8_wall_probe.py \
  ../../data/qwen-moe/depth-regime/trace/gpu-host/231106_kernel_trace.csv \
  ../../data/qwen-moe/traffic.json --depth-zero \
  --out ../../data/qwen-moe/q8-unprofiled/q8-ordinary-depth0.json
```

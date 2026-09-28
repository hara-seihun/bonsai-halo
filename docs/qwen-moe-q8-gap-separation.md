# Wide Q8 slow episodes follow recorder-sized pauses in two Qwen traces

The selected Qwen3.6-35B-A3B native runtime's **8,192-output Q8_0** slow-call mask is not an isolated Q8 operand cost. On the retained counter-free depth-1024 profiler trace, all **30/30 held** Q8 calls above the previously fixed 105-µs cutoff occur within 64 dispatches *after* a ≥0.5-ms device-timestamp gap; none of the **108 farther** held calls is slow. A second trace with the **same executable and GGUF** at depth zero reproduces **42/45** slow calls near such gaps over seven warm tokens; of three farther slow calls, one is the first Q8 call before any gap and the other two are at distances 93 and 102. This is an independently recorded depth intervention and a second profiling run, not an unprofiled speed result or proof that a recorder pause causes long GPU kernels.

The 64-dispatch gate, ≥0.5-ms gap and >105-µs Q8 slow cutoff were fixed by the preceding [expert gap](qwen-moe-slow-gap-separation.md), [timestamp audit](qwen-moe-launch-gap.md) and [Q8 episodes](qwen-moe-q8-episodes.md) before this join. The inspection/held split of the depth-1024 trace is warm tokens 1–3 versus 4–7. Each complete token has exactly 1,565 dispatches, forty routers, one pre-router 8,192-output Q8 call per layer, and a full 248,320-row output-head call. The depth-1024 trace excludes two synthetic-depth setup heads and cold token 0; the depth-zero first head has an incomplete preceding layer sequence, so that trace's independent panel uses the seven complete warm token spans. Every occurrence is paired by ordered router boundary and recorded with its exact dispatch ordinal, duration and distance to the nearest preceding gap.

| Recorder distance | Depth-1024 inspection slow near / slow far | Depth-1024 held slow near / slow far | Depth-zero warm slow near / slow far |
| --- | ---: | ---: | ---: |
| ≤8 dispatches | 3 / 17 | 4 / 26 | 6 / 39 |
| ≤32 dispatches | 14 / 6 | 20 / 10 | 26 / 19 |
| **≤64 dispatches** | **19 / 1** (41 near, 79 far) | **30 / 0** (52 near, 108 far) | **42 / 3** (86 near, 194 far) |
| ≤128 dispatches | 19 / 1 | 30 / 0 | 44 / 1 |

At the precommitted 64-dispatch gate, depth-1024 held near calls average **107.052 µs** and far calls **92.405 µs**; depth-zero near/far averages **105.104 / 92.375 µs**. The warm depth-1024 Q8 family previously had a **.9405-ms/token** gap above a *conditional* once-read 242-GB/s comparator. Even an impossible free replacement of every >105-µs event by the ordinary median could remove only **.1614 ms/token** of that event gap, as [Q8 episodes](qwen-moe-q8-episodes.md) established. The current join changes the *interpretation* of those events: apparent persistent slow layer IDs coincide with the recorder's fixed ordinal pauses (30, 62, 126, 254, 510, 766, 1022, 1278) across unrelated operations, rather than being evidence of a Q8-specific slow image. The matched depth-zero run also has those pause sites. It does **not** make the remaining ordinary Q8 conditional byte comparison into observed DRAM traffic or measured unprofiled wall time.

**Decision:** do not port a layer-specific Q8 kernel or spend an estimated .16 ms/token based on this profiled slow mask. The same recorder-locality issue now affects both Q8 and Q4/Q5 interpretations; ordinary Q8/expert physical execution is still the relevant independent engine question. Take a short *unprofiled*, low-overhead per-layer device-marker panel across normal Q8, router, Q4 and Q5 calls, with full output heads, equal occupied context, fixed clock and a matched whole-model wall run through `tools/run-batch-compare`. Distinguish recorder artifacts from actual time before considering Q8 image layouts or direct expert consumers. No installed executable, weight, numerical map, GPU service or serving default changed in this CPU analysis.

[`q8_gap_separation.py`](../tools/qwen-moe/q8_gap_separation.py) writes every matched case and both panel summaries. [The CPU receipt](../../data/qwen-moe/q8-unprofiled/q8-gap-separation.json) binds the two complete original-trace hashes, both GPU measurement receipts, same binary/model, analyzer source and every raw distance/duration. Reproduce without GPU:

```sh
python3 tools/qwen-moe/q8_gap_separation.py \
  ../../data/qwen-moe/q8-unprofiled/trace/gpu-host/1886571_kernel_trace.csv \
  --receipt ../../data/qwen-moe/q8-unprofiled/receipt.json \
  --depth0-trace ../../data/qwen-moe/depth-regime/trace/gpu-host/231106_kernel_trace.csv \
  --depth0-receipt ../../data/qwen-moe/depth-regime/receipt.json \
  --out ../../data/qwen-moe/q8-unprofiled/q8-gap-separation.json
```

# Qwen expert slow-layer regime survives removal of 1024-token occupied context

The [prior layer-paired device trace](qwen-moe-slow-regime.md) found that slow routed Q4_K gate/up calls co-occur with earlier slow router top-k and later slow Q5_K down calls. A matched *same-executable* depth intervention now asks whether that mask requires the 1024-token synthetic prefix. It does not: among four withheld warm decode tokens, **26 of 28** layer/token Q4 calls above the prior 80 µs threshold at depth 0 are also above it at depth 1024; 121/160 are ordinary in both. The other 11 slow-at-depth-1024 calls and two slow-at-depth-0 calls show that context and execution still matter. This is a device-event observation, not proof that a weight tensor or fixed layer alone causes the slowdown.

| Four held tokens, 160 paired layer calls | Q4 ≤80 µs at depth 1024 | Q4 >80 µs at depth 1024 |
| --- | ---: | ---: |
| Q4 ≤80 µs at depth 0 | 121 | 11 |
| Q4 >80 µs at depth 0 | 2 | **26** |

The *same* `b4c67ced9f6bac6b1661b25714946d999374e06f` on-demand HIP binary, GGUF and 512/256 batch, eight CPU threads, flash attention and complete vocabulary head were used for both eight-step traces. Depth 1024 reuses the retained counter-free trace, with two synthetic-depth setup heads excluded. Depth 0 was newly traced with `rocprofv3 --kernel-trace` and no counters. Seven warm tokens in each trace have exactly 1,565 dispatches per head, one measured queue per token and forty each of router, Q4 gate/up and down kernels. The CSV rows are sorted by device start timestamp before matching the native dispatch order. Token 0 is excluded because depth-0 first-use setup overlaps its span. Thresholds (Q4 80 µs, router 10 µs, Q5 down 65 µs) were inherited unchanged from the prior inspection panel; tokens 4–7 are the paired assessment panel.

At depth 0, **24/28** slow-Q4 events have a router over 10 µs *before* gate/up and **28/28** have a Q5 down over 65 µs *after* it. Depth 1024 has 33/37 on each. The depth-0 router averages 17.58 µs on slow Q4 layers and 4.68 µs on ordinary layers; the depth-1024 figures are 18.09 and 4.64 µs. Removing the attention/KV work of an occupied prefix therefore does not erase the shared router/expert layer mask. An isolated Q4 arithmetic port is a poor causal explanation for the earlier-router slowdown at either depth.

One depth-0 event at token 7/layer 25 lasts **29,079.928 µs** while its depth-1024 partner lasts 49.278 µs; its preceding router lasts only 4.36 µs. This is a separate isolated traced anomaly, not the usual ~115 µs slow mode. It raises the untrimmed depth-0 Q4 sum to 9,741 µs/token against 2,611 µs/token at depth 1024. Retaining the event in the raw result, but excluding only this >200 µs anomaly from the *typical-duration comparison*, yields 2,471 vs 2,599 µs/token and paired duration correlation **0.793** across the other 159 held calls. Without that separation the correlation is −0.039; reporting only either number would misdescribe the trace. The anomaly is not a measured causal model stall or repeatable kernel slowdown.

A separate unprofiled ABBA wall panel at 32 generated tokens and complete output heads measured **34.37, 50.92, 50.87, 40.07 tokens/s** at depths **0, 1024, 1024, 0**. Its clock log reports median 2,663 MHz and competing host work, and depth 0's large within-arm drift prevents using the cross-depth wall ratios as a speed claim. This experiment changed no weights, selected runtime, numerical map or resident serving default. It does *not* reproduce the exact output logits across depths (different prefix and token states), and a profiler event sum is not an unprofiled model phase measurement.

**Result and next experiment.** The slow Q4/router/Q5 cluster is substantially layer-persistent even when the synthetic occupied context is removed; chasing only KV length or replacing only the Q4 operand cannot explain it. Place low-overhead native device markers around router, gate/up, down and the next Q8 on the **current selected runtime** in a matched whole-model panel, with fixed clock, complete requested heads, source/binary hashes and wall timing; cross the slow/ordinary layer mask against kernel geometry and physical memory observations. A repeated same-depth traced control can separately diagnose the single 29-ms profiler event. Do not convert the 26/28 association or the profiled event sums into an attainable TPS gain.

[Raw depth-0 CSV, old depth-1024 CSV identity, full 280 warm layer pairs, four wall records, clock log, binary/source/model hashes and restored-service receipt](../../data/qwen-moe/depth-regime/receipt.json) remain in the data owner. Recompute the joined CPU result in seconds:

```sh
python3 tools/qwen-moe/depth_regime.py \
  ../../data/qwen-moe/depth-regime/trace/gpu-host/231106_kernel_trace.csv \
  ../../data/qwen-moe/q8-unprofiled/trace/gpu-host/1886571_kernel_trace.csv \
  ../../data/qwen-moe/depth-regime/wall.jsonl \
  --output ../../data/qwen-moe/depth-regime/result.json
```

# The routed expert slowdown begins before gate/up and crosses a layer boundary

The pinned Qwen3.6-35B-A3B counter-free HIP event trace contains forty Q4_K gate/up calls per complete decode token. Slow gate/up calls are **not isolated to that consumer**: in four withheld warm tokens, 33 of 37 slow gate/up calls also have a slow *earlier router top-k*, and 33 of the same 37 have a slow *later Q5_K down projection*. None of the 111 ordinary gate/up calls with a Q5 down has a slow Q5. These are independent kernel bodies and distinct weights. A gate/up-only operand port cannot by itself explain why the earlier router already slowed.

The structural alignment adds a useful boundary: a slow gate/up at layer `l` is more associated with the **next layer's** wide Q8 projection than the same layer's wide Q8. That suggests measuring the layer seam and dispatch regime before another packed consumer change; it does not prove one kernel causes the next to slow. These layer masks persist across tokens, so tensor-specific properties and queue/clock/cache conditions are not separated by this observational trace.

| Warm split, 40 layer calls/token | Gate/up >80 µs | Of those, paired other kernel slow | Other slow among ordinary gate/up |
| --- | ---: | ---: | ---: |
| Inspection tokens 1–3, router top-k >10 µs | 26/120 | 23/26 | 4/94 |
| Withheld tokens 4–7, router top-k >10 µs | 37/160 | **33/37** | **2/123** |
| Inspection tokens 1–3, Q5_K down >65 µs | 26/120 | 24/26 | 0/85 Q5 layers |
| Withheld tokens 4–7, Q5_K down >65 µs | 37/160 | **33/37** | **0/111 Q5 layers** |
| Inspection tokens 1–3, next-layer Q8 >105 µs | 26/120 | 14/26 | 5/91 valid next layers |
| Withheld tokens 4–7, next-layer Q8 >105 µs | 37/160 | **22/37** | **8/119 valid next layers** |
| Withheld tokens 4–7, same-layer Q8 >105 µs | 37/160 | 17/37 | 13/123 |

All thresholds were fixed before scoring the withheld tokens. Q6_K serves three down layers and is not classified under the Q5 threshold. By assigning each preceding router and following down to the actual Q4 dispatch in the same layer, and each next-layer Q8 to its own subsequent layer, the result does not mistake generic adjacency for paired work. The ten complete-head boundaries isolate two synthetic setup evaluations and eight decode tokens; token 0 is cold and excluded from the splits. Every decode token has 1,565 dispatches on one measured queue.

On the withheld 160 layer calls, mean gate/up duration rises **50.609 → 114.028 µs** on its slow subset. The *earlier* router rises **4.641 → 18.095 µs**, the later down **38.812 → 86.718 µs**, the gate/up input coder **1.416 → 6.362 µs**, and the down input coder **1.359 → 7.888 µs** on those same selected layers. Merely deleting all gate/up input-coder duration cannot remove the router's preceding slowdown, the down coder or down matvec. The observed slow/ordinary duration ratios are also not one universal multiplicative factor across kernel bodies (gate/up 2.25×, router 3.90×, down 2.23×); this rejects a *pure common multiplier of event durations*, not clock variation mixed with fixed overhead and cache effects. The earlier [activation accounting](qwen-moe-activation-bound.md) already prices the coder's own excess, while [the overlap trace](qwen-moe-trace-overlap.md) establishes that these events do not overlap on this queue. This analysis specifically pairs the router, the second expert consumer and the **following layer** under a frozen inspection/withheld split.

**Next native experiment:** on the selected on-demand runtime, place low-overhead unprofiled device markers around the pre-router normalization/top-k, input quantizer, Q4 gate/up, Q5 down and the following layer's first Q8. Alternate unchanged complete-model panels at occupied depth 1024 and the generated 32-stream shape, recording full requested heads, wall TPS, clocks, source/binary hashes and service restoration through `tools/run-batch-compare`. Check whether the within-layer and layer-seam associations survive without trace collection; compare slow versus ordinary at matched kernel geometry before porting Q4/Q8 arithmetic. A counter-free trace does **not** measure physical DRAM bytes, clock cause or an attainable whole-model gain.

[The complete CPU receipt](../../data/qwen-moe/q8-unprofiled/slow-regime.json) records all 320 token/layer pairs, inspection and withheld contingency matrices, mean durations and hashes of source script, trace, installed receipt, binaries and model. Reproduce in seconds without reserving the GPU:

```sh
python3 tools/qwen-moe/slow_regime.py \
  ../../data/qwen-moe/q8-unprofiled/trace/gpu-host/1886571_kernel_trace.csv \
  --receipt ../../data/qwen-moe/q8-unprofiled/receipt.json \
  --output ../../data/qwen-moe/q8-unprofiled/slow-regime.json
```

No runtime, model image, installed executable, GPU reservation or resident service changed.

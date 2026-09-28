# Wide Q8 costs on ordinary and slow Qwen layers

The installed target's 8,192-output Q8 projections have about .94 ms/token of conditional room above one image read at 242 GB/s on the seven warm depth-1024 trace tokens. The layer-persistent slow periods account for only .16 ms/token of that difference under a deliberately generous replacement: make every slow Q8 call cost the median ordinary call without paying for a repair. Most of this Q8 comparison survives. The ordinary width-8192 consumer, not just the slow-layer episode, is worth measuring without a profiler before changing its kernel.

I re-sliced the [selected runtime's counter-free trace](qwen-moe-q8-unprofiled.md). Each of its eight measured decode spans has exactly forty top-k router calls and one width-8192 Q8 call before each router. I matched those pairs by layer, then paired the forty routed Q4 calls by their execution order. Requiring each Q4 to lie between its router and the next router would be wrong: graph dispatch sometimes moves a routed call across that boundary. Token 0 is cold and excluded from the seven-token warm comparison. Source, original trace, pinned GGUF and installed executable hashes, all 320 layer observations, and fitted/held masks are in the [CPU receipt](../../data/qwen-moe/q8-unprofiled/q8-episodes.json). [`q8_episodes.py`](../tools/qwen-moe/q8_episodes.py) checks the original trace against its installed receipt.

| Warm width-8192 Q8 calls | Count | Mean device µs |
| --- | ---: | ---: |
| Ordinary, at most 105 µs | 230 | 92.887 |
| Slow, above 105 µs | 50 | 116.893 |
| All | 280 | 97.174 |

The seven warm tokens spend 3.886947 ms/token in these forty kernels. Their 713.032 MB image takes 2.946413 ms/token at the separate 242 GB/s one-read reference. Replacing every slow call by the ordinary median, 94.297 µs, would subtract just .161402 ms/token of device events, about 17% of that conditional .9405 ms difference. This replacement is not a hardware speedup and does not show that the remaining difference is DRAM traffic. It establishes that fixing only the conspicuous slow Q8 episodes cannot close the width group's comparator gap in this trace.

Fit a layer mask on warm tokens 1–3 by calling a layer persistent if at least two of those three Q8 calls exceed 105 µs. Layers 2, 7, 13, 20, 27 and 33 qualify. On untouched tokens 4–7 the mask predicts 23 of 30 slow calls, with one false positive out of 130 ordinary calls. The mask is useful for targeted per-layer markers, not an explanation for its cause. Thresholds of 100 and 110 µs yield nearly the same held recall, 26/30 and 23/28 respectively, and two and one false positives. Across the seven warm tokens, slow Q8 and slow Q4 overlap only 28 times: 22 slow Q8 calls have ordinary Q4, while 35 slow Q4 calls have ordinary Q8. The expert study's eight persistent Q4 layers and these six wide-Q8 layers are different sets. A single per-layer slow/fast bit is therefore not an adequate replacement for separate kernel markers.

These are device event durations from a profiler, not unprofiled phase or wall time. The 242 GB/s comparator assumes each image is read once and that its independent benchmark bandwidth applies here; the broken GL2C collection supplies no physical bytes. Neither the .161 ms event counterfactual nor the .94 ms comparator is a full-model speedup. Next take a short paired unprofiled wall and per-layer marker panel, preserving output-head rows and occupied depth. Include both persistent masks and ordinary layers, measure clocks and Q8/Q4 durations, then choose whether to port the ordinary Q8 packed consumer or repair a shared dispatch condition. No runtime, weights, numerical map, installed artifact or resident service changed in this CPU iteration.

Reproduce the receipt with:

```sh
python3 tools/qwen-moe/q8_episodes.py \
  ../../data/qwen-moe/q8-unprofiled/trace/gpu-host/1886571_kernel_trace.csv \
  --receipt ../../data/qwen-moe/q8-unprofiled/receipt.json \
  --output ../../data/qwen-moe/q8-unprofiled/q8-episodes.json
```

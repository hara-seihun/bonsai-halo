# Finite expert-image cache on actual Qwen routes

The [layer-0 route traffic study](qwen-moe-route-traffic.md) found almost no adjacent-token overlap, but that alone does not settle a cache. An expert can return after several tokens. This CPU study asks how much a cache of complete packed expert images can save on the 113-token train and disjoint 126-token held prompts captured from the installed Qwen3.6-35B-A3B GGUF. It changes neither the model nor its arithmetic.

One expert's Q4 gate, Q4 up and Q5 down images occupy 1,900,544 bytes in the pinned layer-0 bank. Each token demands eight distinct images, in router rank order. The experiment charges a miss for one logical full-image read, retains at most `C` complete images and does not let other weights or state displace them. Its cold LRU and cold Belady arms start empty. Belady evicts the image whose next use lies farthest ahead, so it minimizes misses for this finite sequence under unit-size, non-bypass, full-image replacement. The pinned arm preloads the `C` most frequent experts on train and holds them through held; its compulsory initial loading is **not** charged. Held-hindsight pinned selection and future-aware Belady are noncausal controls, not deployable policies. The grammar excludes fractional-image retention, grouped token tiles, overlapping weight loads with compute and any altered expert arithmetic.

| Held, 1,008 assignments | Capacity | Resident bytes | Cold LRU misses | Train-pinned misses | Held-hindsight pinned misses | Cold Belady misses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No retention | 0 | 0 | 1,008 | 1,008 | 1,008 | 1,008 |
| Eight images | 8 | 15.204 MB | 993 | 896 | 871 | 802 |
| Sixteen images | 16 | 30.409 MB | 944 | 823 | 777 | 666 |
| Thirty-two images | 32 | 60.817 MB | 806 | 696 | 622 | 497 |
| Sixty-four images | 64 | 121.635 MB | 589 | 469 | 383 | 332 |

At sixteen images, holding the train-selected hot experts saves 185 images, or 2.790 MB per held token. Future-aware replacement saves 342 images, or 5.159 MB per token, despite the tiny adjacent-token intersection. So the negative adjacent-overlap observation did *not* imply zero reuse at longer intervals. But the unrealistically generous cache already holds 30.4 MB, almost all of this GPU's measured 32 MB last-level cache before any Q8 projection, shared expert, KV/state or vocabulary traffic. Its optimal layer-0 saving is 0.1964% of the 2,626.188 MB complete-model one-read-per-selected-image weight stream per token; train-pinned is 0.1063%. Even granting the same route behavior, uninterrupted residency and saving in **all forty layers** would give conditional weight-byte reductions of 7.86% and 4.25%, respectively. We did not capture those 39 routes, and these are not physical DRAM or latency reductions. A cold cache must also fetch its initial images; the pinned figures favor pinning by omitting that cost.

The capacity result is also sensitive to policy. LRU at sixteen saves only 64 images on held, while train-pinned saves 185 and Belady saves 342. A cache claiming to exploit this skew must retain cold-in-rank but frequent-in-prompt experts across the intervening token requests. Native `mul_mat_id` already groups assignments across a prompt. The earlier tile-eight, 32-token held count is 500 logical reads; its grouping contract allows reordering eight assignments per expert tile, while this study consumes token requests in order. These counts are **not** competing implementations at equal scheduling constraints, and neither is an incremental gain over the native grouped runtime.

The exact result is a finite-family miss optimum, not a hardware lower bound or whole-model speedup. It argues against building a dedicated complete-image layer-0 decode cache first: the ideal byte prize at a plausible cache footprint is small and already assumes impossible isolation from the other 2 GB/token of weights. The next native question is physical routed Q4/Q5 bytes and unprofiled phase time at one, eight and 32 rows alongside Q8, using the existing grouped consumer as control. If bytes remain high, test smaller expert tiles or a shared packed consumer rather than reserving most of cache for sixteen full images. Prompt routes are not a sample of generated decode routes; capture those before extrapolating this policy to serving.

`tools/qwen-moe/route_cache.py` validates the captured ID hashes against the capture receipt, reads the GGUF tensor inventory, and records source, loader, input, token/text and header hashes with each result in [`data/qwen-moe/route-cache/receipt.json`](../../data/qwen-moe/route-cache/receipt.json). Reproduce without a GPU:

```sh
python3 tools/qwen-moe/route_cache.py ../../data/qwen-moe/route-capture \
  ../../data/qwen-moe/traffic.json \
  --output ../../data/qwen-moe/route-cache/receipt.json
```

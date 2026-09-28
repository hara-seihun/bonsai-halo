# A ternary MoE bank does not turn serial route reuse into a large cache gain

The current Qwen3.6-35B-A3B image is **not ternary**. This is a conditional cost bound for replacing *all three* 2,048×512 routed matrices per expert with a distinct, directly consumed, complete image at a specified **paid** rate, while leaving its nonexpert bank and complete vocabulary head unchanged. It does not construct that image, preserve its language quality, or price a decoder. The small Qwen3-0.6B pilot's 1.65007-BPW paid ternary point has 4.498337 held NLL versus 3.6392 BF16 on its earlier test; moving that number to Qwen3.6 is a **scenario**, not a quality transfer. In particular a hypothetical 1.65-BPW image here must include its scales, metadata, alignment and every expert direction inside that rate.

## The complete-map cache theorem

In serial single-stream generation, one step visits forty layers and eight distinct expert IDs per layer. Let `D_t` be its demanded physical expert-byte addresses, each consumed **once** in the step, and `C_t` the 32-MiB cache contents when that step begins. Distinct `(layer,expert)` images occupy disjoint addresses; all other weight and activation traffic bypasses the cache free. Grant byte-granular residency, perfect foreknowledge, bypass and zero replacement cost. Without changing the arithmetic/labels, the avoided one-read bytes in step `t` are at most `|D_t ∩ C_t| ≤ |C_t| ≤ 33,554,432`: a byte loaded later cannot produce a same-step *reuse* when it has only one demand. In `T` cold steps, at most `(T−1)×33,554,432` bytes are avoidable. This proof works for arbitrary paid image sizes and arbitrary cross-token reuse, **provided physical images are disjoint, the step remains serial and each byte is demanded only once per token**. Shared physical codebooks, sub-one-read native scheduling, independently batched streams and within-token duplicate accesses are outside the theorem. The byte ceiling says nothing about DRAM transaction counts or execution time.

For an image at 1.65 paid bits per routed weight, an expert has 3,145,728 weights and needs at least `ceil(3,145,728×1.65/8) = 648,807` bytes under the stated uniform-rate scenario. Eight images at all forty layers occupy **207,618,240 bytes**, 6.19× the nominal cache. All 320 images could fit only at **≤0.266667 BPW including all metadata** under this same disjoint, equal-rate grammar. This is not a model-independent entropy lower bound: sufficiently structured learned images can be cheaper, and shared labels violate disjointness. It identifies the threshold a changed representation would have to cross to make forty-layer full residency plausible.

| Hypothetical complete routed-image BPW | Bytes / expert | Full-image slots in 32 MiB | Complete one-read bytes/token (nonexpert unchanged) | Static bank replacement saving vs selected Q4/Q5 image | Maximum *additional* cold 48-token byte-cache saving / new complete stream |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.585 (near ternary information floor, not a paid model) | 623,248 | 53 | 2,214,110,848 | 15.691% | 1.484% |
| 1.650 (hypothetical paid point) | 648,807 | 51 | 2,222,289,728 | 15.380% | 1.478% |
| 1.727 (raw small-pilot rate transferred as a scenario) | 679,085 | 49 | 2,231,978,688 | 15.011% | 1.472% |

The static savings are *conditional one-read weight bytes*, not speed or a quality-matched alternative. The nonexpert baseline is 2,014,671,488 B/token, including one embedding lookup and the complete output head. The original selected total is 2,626,187,904 B/token. A compressed routed bank materially lowers this logical weight stream, but **cross-token caching contributes at most another ~1.5%** of the *new* complete stream, before cache competition with the 2.015-GB nonexpert work or any packed-reader cost. Thus route locality alone is not a good justification for constructing a ternary bank: quality and direct packed consumption would have to carry the case.

## Two actual generated-route controls

The independently captured train and held continuations each have 48 generated tokens, forty layers and eight IDs/layer/token. A callback cuts graph fusion, so these are not the uninstrumented selected-map routes, though their captured and plain generated token IDs agree. Under uniform equal-sized images, an offline next-use policy with free bypass provides a **feasible, clairvoyant full-image cache schedule**: farthest-next-use replacement is hit-count optimal by exchange for equal-sized images. LRU is zero on these serial forty-layer streams. With 1.65-BPW images, the offline schedule gets **2,345 train / 2,376 held hits**, against 2,397 possible if all 51 slots hit on each of the 47 later tokens. The held retained bytes are 1,541,565,432, or **1.4452%** of its conditional 48-token complete one-read stream; the unrestricted-byte proof caps it at 1,577,058,304 B / **1.4784%**. The remaining 35,492,872-byte gap includes indivisible image rounding and the finite route schedule. At 1.585/1.727 BPW the held oracle respectively gives 2,466/2,285 hits and 1.4462%/1.4484% of its new stream. These are ideal logical traffic schedules, not GPU-cache hits or native throughput.

[`ternary_cache_bound.py`](../tools/qwen-moe/ternary_cache_bound.py) recomputes the three scenarios, checks all eighty route files, binds the pinned inventory, acquisition and observer, and saves per-token hits plus hashes in the [CPU receipt](../../data/qwen-moe/generated-routes/ternary-cache-bound.json) (SHA-256 `a2121df9c020488bf3690b4bea4948d48c10fa06e3de5b83355bb3e593ba1a7f`). Run:

```sh
python3 tools/qwen-moe/ternary_cache_bound.py \
  ../../data/qwen-moe/generated-routes \
  ../../data/qwen-moe/traffic.json \
  --output ../../data/qwen-moe/generated-routes/ternary-cache-bound.json
```

This bound changes the next question: do **not** build a route-history full-image cache as the selling point for Qwen ternary. Learn a *complete paid image* on broad actual quantized routed producers, test disjoint whole-model held language loss against a matched Q4 and the current selected model, then price a direct packed consumer including metadata and native traffic. Independently, unprofiled ordinary Q8/expert timing remains the runtime priority. No image, GPU, executable, selected numerical map or service changed here.

# Consecutive Qwen prompt tokens do not preserve the full routed set

**Question.** Can the preceding token's top-eight IDs be reused at a layer instead of computing the current 256-way router? This is not expert-image caching: the desired observation is the *current set of eight IDs*. The selected native route also needs their current normalized FP32 scores, which this decision-only experiment deliberately grants free when bounding the possible saving.

`tools/qwen-moe/router_temporal_certificate.py` decodes each of the forty installed 256 × 2,048 F32 router matrices. At every layer it reads the two disjoint actual 64-token post-attention producer and selected-ID captures. FP64 products of stored F32 weights and F32 producer values recover the **set** of all 5,120 native captured routes. The [forty layer/source/weight/input/ID-hashed receipt](../../data/qwen-moe/router-temporal-certificate/receipt.json) retains every adjacent comparison (63 per layer and text). These are callback-cut *prompt* producers, not generated tokens or the unchanged no-callback native graph.

| Consecutive layer/token pairs | Train | Held |
| --- | ---: | ---: |
| Pairs | 2,520 | 2,520 |
| Identical entire eight-ID set | **6** | **1** |
| Mean retained IDs of eight | 3.036 | 3.343 |
| Seven-ID overlap | 44 | 56 |
| Zero-ID overlap | 232 | 161 |
| Certified unchanged using previous logits and pairwise norms | 0 | 0 |
| Certified using eight *current* selected logits and previous outsider bounds | 0 | 0 |

For stored real logits `s_i(x) = w_i·x`, let `d = x_t − x_(t−1)` and let `S` be the preceding token's eight IDs. The check `s_i(x_(t−1)) − s_j(x_(t−1)) > ||w_i − w_j||₂ ||d||₂` for every `i∈S,j∉S` certifies `S` remains the set, by Cauchy–Schwarz on the *difference of two logits*. It grants the complete preceding 256 logits and precomputed pairwise row norms. A second, differently priced sufficient check grants the exact current logits for all eight previously selected rows and bounds each outsider by `s_j(x_(t−1)) + ||w_j||₂||d||₂`. Neither certifies an actual pair, **including all seven pairs whose eight IDs happen to remain unchanged**. Strict comparison handles ties; the proof concerns real sums of stored numbers, not the native FP32 reduction and normalization bits.

A free hindsight oracle that knows whether `S` remains valid could avoid all router weights on only **1/2,520 held layer-token decisions**, at most `83,886,080 / 2,520 = 33,288` router image bytes per token averaged across forty layers, or **0.0012675%** of the conditional 2,626,187,904-byte complete-model one-read stream. Actual certificate saving is zero, and even the oracle still owes current normalized scores; no native GPU time or whole-model speed follows. A partial overlap is not a valid full-route reuse, though it could guide candidate-first execution with a separate exact correction.

**Disposition.** Do not add preceding-token full-set memoization or these norm guards to the selected router. The informative next question is whether a cheap learned candidate set plus exact correction can retain *current* scores at lower native cost, particularly on generated-token producers; a prior route alone is not a full-map coordinate. Larger independent engine value lies in unprofiled ordinary Q8/expert physical-time diagnosis with complete heads. This observed-domain negative does not rule out learned joint routing, different certificates, generated text or changed paid expert images. No GPU, runtime, model, service or serving defaults changed.

CPU reproduction (one process finishes within the lane's foreground timeout):

```sh
OPENBLAS_NUM_THREADS=1 python3 tools/qwen-moe/router_temporal_certificate.py --first 0 --count 40
OPENBLAS_NUM_THREADS=1 python3 tools/qwen-moe/router_temporal_certificate.py --summarize
```

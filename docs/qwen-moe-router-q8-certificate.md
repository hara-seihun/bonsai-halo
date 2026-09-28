# Q8-first Qwen router: interval certification almost always falls back

The frozen installed Qwen3.6-35B-A3B router is a 256 × 2048 F32 matrix at every one of forty layers. On actual disjoint 64-token train/held post-attention producers, a proposed **Q8_0-shaped router-weight image plus group-32 Q8 activation** predicts the same eight-ID *set* on 2,443/2,560 train and 2,449/2,560 held layer/token decisions. This high empirical agreement is **not** enough to skip the F32 router while retaining its exact route and score contract. A sufficient interval certificate from the known input and paid groupwise weight metadata certifies only **2/2,560 train and 1/2,560 held** decisions. The remaining 2,559 held decisions must run the original router, including 2,448 whose Q8 choice happened to be correct.

## Complete observation and construction

The installed F32 router tensor bytes and actual producer/route captures are pinned by the [ordinary input-precision report](qwen-moe-router-input-precision.md). This experiment quantizes **both** weight and producer, unlike that report's activation-only control. For each group of 32, code `round(value / float16(max(abs(group))/127))` into signed Q8, with the rounded FP16 scale. Zero groups receive code zero. Let `Wq,xq` denote the expanded code/scale images; the predicted ideal-real logits are `p = Wq xq`, the F32-weight ideal-real target is `l = Wx`. For every router row and input group,

`|l_i-p_i| ≤ Σ_g ( ||(W_i-Wq_i)_g||₂ ||x_g||₂ + ||(Wq_i)_g||₂ ||(x-xq)_g||₂ ) = B_i`.

Both row norms can be precomputed from the fixed installed weight and its proposed Q8 image; the original runtime input `x` is already available. No original `Wx` logit is used to form an interval. If the lowest lower endpoint among the eight predicted winners exceeds the highest upper endpoint among the other 248, their ID set is exactly the full-router top eight in **ideal real arithmetic**. The predicate is strict, so tied-ranking behavior is irrelevant. On certification, compute the original F32 logits for those eight rows before the selected-score softmax and ordered expert reduction. On failure, run all 256 original F32 rows. This is a structural exact-route sketch, **not** an FP32 native implementation or proof that the selected CUDA graph/fused top-k map can switch branches bit-identically.

The source also tests a cheaper global one-norm-pair-per-row certificate and two generous controls: Q8 weights with original activation, and Q8 activation with original weights. The latter controls are *not* candidates saving both weight and activation work; they isolate what makes the joint certificate fail.

| Producer split, 2,560 layer/token rows | Q8 top-eight sets correct | Certified, global row norms | Certified, group-32 row norms |
| --- | ---: | ---: | ---: |
| Train | 2,443 | 0 | 2 |
| Held | 2,449 | 1 | 1 |

At held, weight-only predicts 2,458 sets correctly and certifies **26** with group bounds; activation-only predicts 2,499 and certifies **25**. The joint errors plus independent interval slack eliminate almost every certificate. Every certified case was directly checked against the installed full F32-weight dot on its saved producer. Even a free hindsight witness for the **predicted winner set** could approve at most the 2,449 correctly predicted held sets; the large additional gap to one approved case comes from the inexpensive interval bound. That witness would require original logits and is not a program. 

## Paid bytes and work

A per-layer group-32 Q8 router stores `256×64×(32+2) = 557,056` weight bytes. Two FP64 precomputed norm arrays cost `256×64×2×8 = 262,144` bytes, or **819,200 bytes/layer** against the installed **2,097,152 F32 bytes/layer**. Global row norms would add only 4,096 bytes but certify no train and one held row. The group certificate requires the input's 64 group norms and error norms and two 64-term matrix-vector bounds for *all 256* router rows, on top of the 256 Q8 dots and input quantization. The exact eight F32 scores cost at least `8×2048×4 = 65,536` original-weight bytes on a certified row; a fallback reads the original 2,097,152-byte matrix. Since **2,559/2,560 held decisions fall back**, the proposed group image/metadata adds about 819 kB per layer/token of logical reads before cache effects and extra arithmetic rather than removing the original router. Across forty layers this is ~32.8 MB/token, **1.25% of the 2.626-GB conditional complete-model one-read weight stream**. The F32 sidecar also remains stored, so this is no complete-image rate saving. FP64 norms are charged for the ideal-real certificate; native F32 bounds would need outward rounding and an FP32 arithmetic-error enclosure, both additional work. None of these bytes is measured DRAM traffic, and there was no GPU panel or TPS change.

**Decision:** reject this Q8/Q8 plus independent group-2-norm interval/fallback construction for the exact installed router. The 95.66% held route agreement is a possible *approximate* model proposal only after paid complete-model language quality, not evidence for an exact default. A distinct exact question would need a cheap **correlated, pairwise top-eight margin witness** whose bound is much tighter than independently enclosing 256 rows, and must price its metadata against simply running the 2-MB F32 router once. Do not repeat this scalar norm family at a finer group size without pricing the larger metadata and online reductions.

## Reproduction and custody

From the Bonsai root, CPU only (each shard fits the 55-second attended command limit):

```sh
OPENBLAS_NUM_THREADS=4 python3 tools/qwen-moe/router_q8_certificate.py --first 0 --last 20 --output ../../data/qwen-moe/router-q8-certificate/part-0-19.json
OPENBLAS_NUM_THREADS=4 python3 tools/qwen-moe/router_q8_certificate.py --first 20 --last 40 --output ../../data/qwen-moe/router-q8-certificate/part-20-39.json
```

[The two-shard data manifest](../../data/qwen-moe/router-q8-certificate/README.md) records immutable hashes, model/capture/source identity and all four counts. The raw shards retain each layer's split input/route hashes, interval counts, stable-set counts and fallback token indices. CPU FP64 dot/certificate semantics deliberately differ from selected native FP32 reduction and graph scheduling; neither native bit identity nor model-quality retention follows from this study.

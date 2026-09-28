# Complete forty-layer omission/cancellation oracle

Can two or more unchanged routed expert outputs cancel when neither one alone is safe to omit? The earlier exhaustive 256-subset study saw only layer 0. The subsequent all-layer capture makes the same question decidable on **both 64-token real-text prefixes at all forty Qwen3.6-35B-A3B layers**. It includes actual installed-GGUF post-attention producers, selected IDs, normalized scores and eight 2,048-wide routed down outputs for every layer/token. The callback cuts the graph; this is a local-output study, not unobserved plain-native FP32 or language quality.

For a token/layer, let `v_e = float64(score_e) * float64(down_e)`, `y = sum_e v_e`, and `G_ij = v_i · v_j`. The residual of omitting subset `m` has squared norm `mᵀGm`. Enumerating all 256 masks gives the **minimum local relative Euclidean error at each omitted count**, including cancellation. The adaptive oracle then takes the *largest* count under a per-token threshold. It sees every down output and pays nothing to choose; this is deliberately stronger than any early-stop implementation using unchanged contributions. The comparison score-prefix arm also cheats by testing the true error after each prefix. Scores and vectors are converted from captured FP32 to FP64 offline, not associated as in the native routed sum.

| Split, 2,560 layer-token observations | Limit | Oracle omitted assignments / 20,480 | Score-prefix omitted assignments | Oracle local RMS after adaptive skips | Conditional complete one-read byte ceiling |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 1% | 0 | 0 | 0 | 0 |
| Held | 1% | **0** | 0 | 0 | **0** |
| Train | 5% | 138 | 104 | .01777 | .15837% |
| Held | 5% | **149** | 118 | .01648 | **.16940%** |
| Train | 10% | 1,112 | 894 | .06279 | 1.26934% |
| Held | 10% | 1,150 | 894 | .06574 | 1.30991% |
| Train | 20% | 4,645 | 3,991 | .16152 | 5.28604% |
| Held | 20% | 4,702 | 4,046 | .15162 | 5.34793% |

At 5%, the held oracle skips nothing on **2,445/2,560** observations, one on 97, two on eight, three on six, four on three and six on one. Thus the earlier best-one study's 115 feasible decisions become just 149 skipped assignments even after *all* subset cancellations are granted for free. Across **all 5,120** train+held layer-token observations, the best two-vector omission never has smaller norm than the best one-vector omission. In particular no multi-output cancellation rescues a 1% omission. This is an observed finite-domain bound, not a theorem about all inputs or reweighted/new directions. Layer 0 alone accounts for 38 of the held 149 skipped assignments at 5%; extrapolating its rate to the remaining layers is misleading. At four retained outputs irrespective of threshold, the held hindsight oracle's local relative RMS is **.29103**, versus **.31936** for the highest-score four; neither is a low-error local replacement.

Bytes are priced from the actual per-layer packed gate/up/down expert bank in `traffic.json` (one selected expert image is 1,900,544–2,039,808 bytes depending on layer). Sum the skipped assignments' own layer bytes over 64 tokens and divide by **2,626,187,904 conditional whole-model one-read bytes per token**. The 5% held ceiling is about 4.45 MB/token, even before choosing the subset, dispatch/weight-grouping overhead or any change in physical DRAM traffic. A per-layer 5% local bound is *not* a complete-model NLL or acceptable quality criterion; aggregate model deviations can compound. This does not constrain trained packed replacement directions, a new shared consumer coordinate, or a downstream observation weaker than the routed vector. It does settle the proposed free unchanged-output cancellation shortcut on these disjoint observed prefixes: do not build an expert-skipping native path on its presumed gain. Use these actual producers to train changed codes and freeze a paid forty-layer image, then test disjoint complete-model language loss and native work.

`tools/qwen-moe/all_layer_subset.py` recomputes both splits from the [actual all-layer capture](qwen-moe-all-producers.md). The [saved receipt](../../data/qwen-moe/all-layer-subsets/receipt.json) includes every layer's frontier, per-layer expert image bytes, source SHA-256, pinned traffic-inventory SHA-256 and capture receipt SHA-256. It checks the score/down file hashes against that earlier capture's independently retained inventory. Run CPU-only:

```sh
python3 tools/qwen-moe/all_layer_subset.py ../../data/qwen-moe/all-producers \
  --out ../../data/qwen-moe/all-layer-subsets/receipt.json
```

No GPU, serving service, installed image or runtime was changed.

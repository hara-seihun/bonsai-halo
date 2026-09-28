# Free-exact-norm routed stopping across forty layers

The earlier [layer-0 score-order certificate](../../kelana/research/moe/score-order/README.md) offered an optimistic cheap stopping rule for the eight routed Qwen3.6-35B-A3B outputs: grant the **exact norm of every uncomputed output for free**, but not its direction. This study applies that *same* local real-arithmetic observation contract to [both disjoint 64-token captures across all forty layers](qwen-moe-all-producers.md). It answers whether layer 0 was representative before considering a native early-stop implementation. It is a distinct, much narrower implementability question than the [all-layer exhaustive hindsight omission oracle](qwen-moe-all-layer-subsets.md), which reads every output and decides from the true error.

For sorted contributions `v_e = float64(score_e) * float64(down_e)`, a computed prefix `p_k = sum_{e<k} v_e` and omitted scalar norms `B_k = sum_{e>=k} ||v_e||`, the triangle inequality gives `||y-p_k||/||y|| <= B_k/(||p_k||-B_k)` when the denominator is positive. The first `k` with `B_k <= tolerance*||p_k||/(1+tolerance)` is certified to meet the local Euclidean threshold. One arm uses the available score order; another gets the omitted norms **and** a hindsight descending-norm order without paying for either. FP64 reconstructs the captured FP32 vectors and scores; this identity is not a bit-identical native FP32 reduction.

| Split (2,560 layer-tokens) | Local tolerance | Score-order skipped of 20,480 assignments | Free norm-order skipped | Score-order conditional complete one-read bytes saved | Free norm-order ceiling |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 1% | 0 | 0 | 0% | 0% |
| Held | 1% | **0** | **0** | **0%** | **0%** |
| Train | 5% | 77 | 98 | .08848% | .11247% |
| Held | 5% | **88** | **117** | **.10025%** | **.13305%** |
| Train | 10% | 558 | 700 | .63809% | .79966% |
| Held | 10% | 562 | 731 | .64005% | .83297% |

At held 5%, score order can stop on only **76/2,560** layer-tokens (one omitted on 67, two on six and three on three). The free norm-order control stops on 102. The exhaustive oracle, which has every direction and may choose any omitted subset, skips 149 assignments and reaches .16940% conditional bytes; even this much stronger rule has no 1% opportunities. The old layer-0-only score certificate skips 23 of 64 assignments in this capture at 5%, or .3594 per layer-token, but applying it to all layers would predict about 1.05% complete one-read savings; the actual weighted forty-layer ceiling is **.10025%**. Only 19 of 40 held layers certify *any* score-order stop. Layer 0 accounts for 23 of the total 88 skipped assignments.

The conditional denominator is 2,626,187,904 one-read bytes/token including nonexpert weights, with each skipped expert charged its own layer's actual packed gate/up/down image size (1,900,544–2,039,808 bytes). There is no credit for a partial expert, duplicate grouped reads or native physical DRAM transactions, and no charge for computing the impossible free output norms, deciding, graph dispatch or partial-vector materialization. In particular a 5% **local** bound is not a permissible complete-model loss budget. The existing native grouped dispatch does not already implement this stop. This finite observed negative closes a triangle/norm-only expert-skip port on these short prompt producers as a material inference improvement; it does **not** bound learned directions, a sharper certificate exploiting their geometry, unseen generated tokens or a downstream observation weaker than the routed sum. A next representation experiment should freeze changed expert codes on broader actual producers and compare a *paid complete image and held language loss*; ordinary unprofiled Q8/expert latency remains independent.

`tools/qwen-moe/all_layer_norm_certificate.py` checks each input array against the [original capture's hashes](../../data/qwen-moe/all-producers/receipt.json), validates every certified stop against the reconstructed full FP64 sum and prices actual per-layer images from `traffic.json`. [The source-, inventory- and capture-hashed receipt](../../data/qwen-moe/all-layer-norm-certificate/receipt.json) retains every layer and split. Reproduce CPU-only:

```sh
python3 tools/qwen-moe/all_layer_norm_certificate.py ../../data/qwen-moe/all-producers --out ../../data/qwen-moe/all-layer-norm-certificate/receipt.json
```

No GPU, selected runtime, service, image or serving default was changed.

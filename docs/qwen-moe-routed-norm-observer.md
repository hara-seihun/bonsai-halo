# A free radial coordinate does not rescue unchanged routed-output skipping

The [forty-layer exhaustive routed-sum omission panel](qwen-moe-all-layer-subsets.md) uses Euclidean error on the complete score-weighted expert sum. Could a consumer that discards its magnitude admit materially more skipping? This CPU experiment deliberately grants a *weaker* observation: only the direction of the routed sum, with a free positive scalar gain chosen in hindsight. It is **not** Qwen's actual next RMSNorm input: the available callback did not save the residual or shared-expert output. Neither a residual-plus-shared next-layer observation nor composed language quality follows from this experiment.

For each of the 5,120 actual layer/token observations in two disjoint 64-token captures, set `v_i = float64(captured_score_i) * float64(captured_down_i)`, `y = Σ_i v_i`, and `G_ij = v_i·v_j`. For every subset `m` of the eight installed routed outputs, let `z = y - Σ_{i∈m} v_i`. The tested normalized-vector distance is `sqrt(2 - 2 y·z/(||y|| ||z||))`. The zero vector is ineligible. Computing `y·z` and `||z||²` from the 8×8 Gram checks all 256 subsets, including cancellations; the oracle chooses the most omitted assignments within the local threshold after seeing all outputs. The score-prefix control also gets the true error for free. This ideal projective quotient removes the radial degree of freedom of the **isolated routed sum** only, and is not native FP32 RMSNorm (whose epsilon and rounding also matter).

| Split / local direction limit | Free oracle skips / 20,480 | Score-prefix skips | Conditional complete one-read byte ceiling | Euclidean oracle skips |
| --- | ---: | ---: | ---: | ---: |
| Train 1% | 0 | 0 | 0 | 0 |
| Held 1% | **0** | 0 | **0** | 0 |
| Train 5% | 138 | 104 | .15837% | 138 |
| Held 5% | **155** | 121 | **.17651%** | 149 |
| Train 10% | 1,119 | 900 | 1.27775% | 1,112 |
| Held 10% | 1,157 | 903 | 1.31832% | 1,150 |
| Train 20% | 4,635 | 3,992 | 5.27639% | 4,645 |
| Held 20% | 4,693 | 4,042 | 5.33916% | 4,702 |

The held best-one omission still has median normalized-vector distance **.12030**, 95th percentile **.21440**. Removing the entire radial constraint gains just **six** held skipped assignments and **.00711 percentage points** of conditional complete bytes at 5%; it gains none at 1%. At 20% the maximal omitted *count* can even decline relative to Euclidean normalization by `||y||`: these are different objectives with different thresholds, not a monotone relaxation of that particular relative-error metric. At 5%, this construction rejects the proposed *isolated-sum projective shortcut* on observed prompts, not actual next-layer RMSNorm skipping. If that is the desired observer, capture the pre-FFN residual plus shared branch (and actual post-add tensor) rather than infer its behavior from these routed vectors.

Conditional bytes sum the installed per-layer gate/up/down image bytes of skipped assignments divided by 64 × 2,626,187,904 whole-model one-read bytes. Hindsight selection, positive rescaling and projective comparison are free; no image change, native packed reader, DRAM count, GPU run or whole-model TPS is asserted. The experiment instead directs representation work to **changed** learned expert outputs, paid broad-producer images and held complete-model quality, rather than a radial correction on the unchanged eight outputs.

`tools/qwen-moe/routed-norm-observer.py` verifies every score/down array hash against the [capture inventory](qwen-moe-all-producers.md), checks the installed traffic inventory and saves the [full CPU receipt](../../data/qwen-moe/all-producers/routed-norm-observer.json) with source/capture/traffic SHA-256 and both split frontiers. Reproduce:

```sh
python3 tools/qwen-moe/routed-norm-observer.py ../../data/qwen-moe/all-producers \
  --out ../../data/qwen-moe/all-producers/routed-norm-observer.json
```

The GPU reservation, installed Qwen executable, model bytes, numerical map and Bonsai service were untouched.

# Actual Qwen router top-eight resists suffix-norm early exit

**Question.** Can a cheap, exact *decision* about the eight routed experts avoid reading the entire installed FP32 router matrix? The observation is the unordered top-eight set of the real-valued 256-logit map, not the normalized scores or the later weighted expert sum. This is an independent question from skipping unchanged expert output: the selected engine already runs a fused router/top-k, and an early decision does not itself preserve its eight score values.

`tools/qwen-moe/router_prefix_cert.py` reads every installed `blk.L.ffn_gate_inp.weight` F32 matrix (256 × 2,048, 2,097,152 bytes), and the actual 64-token train/64-token held post-attention normalized inputs and native eight IDs at all forty layers. The full FP64 matrix product recovers the *set* of all eight native IDs for all **5,120 layer/token decisions**. The code and [complete per-layer source/model/capture-hashed receipt](../../data/qwen-moe/router-prefix-cert/receipt.json) retain sixteen 128-coordinate checkpoints and both panels. The capture callback cuts the graph, and this is not a no-cut native timing experiment.

For prefix length `k`, evaluate prefix dot `p_i`. For the remaining input `x_s` and stored row `w_{i,s}`, Cauchy–Schwarz encloses the real score in `[p_i − ||x_s||₂||w_{i,s}||₂, p_i + ||x_s||₂||w_{i,s}||₂]`. Eight lower endpoints exceeding *every* outsider upper endpoint suffice for an exact top-eight set, independently of the uncomputed suffix. A stronger pairwise certificate uses `p_i − p_j > ||x_s||₂ ||w_{i,s} − w_{j,s}||₂` for every candidate `i` and outsider `j`; shared suffix directions cancel. Both are precomputed suffix-weight-norm grammars with a live input suffix norm, not a general lower bound on all router algorithms. We also give each certificate a **free hindsight choice of the true eight IDs** so that prefix candidate selection is not why a certificate fails. Strict inequalities avoid unspecified ties. FP64 is an analysis of the *real stored-weight map*, not a proof of bit-identical native FP32 logits or scores.

| Prefix | Train decisions certified, interval / pairwise | Held decisions certified, interval / pairwise | Held prefix candidate sets already correct |
| ---: | ---: | ---: | ---: |
| 1,024 | 0 / 0 | 0 / 0 | 0 / 2,560 |
| 1,536 | 0 / 0 | 0 / 0 | 1 / 2,560 |
| 1,792 | 0 / 0 | 0 / 0 | 23 / 2,560 |
| 1,920 | 0 / 0 | 0 / 0 | 191 / 2,560 |
| 2,048 | 2,560 / 2,560 | 2,560 / 2,560 | 2,560 / 2,560 |

**All** early checkpoints, including 1,920, certify zero decisions in both panels under both grammars, **even with true-route hindsight**. The full map's minimum eighth–ninth real-score margin is 5.95e-6 on train and 4.70e-6 on held; narrow decisions make a worst-case suffix enclosure costly. This is not evidence that a partial dot cannot predict the route: 191 held decisions already have the correct prefix set at 1,920, but none has the certificate needed for an exact decision. Even a free last-128-coordinate stop would avoid at most 40 × 256 × 128 × 4 = **5,242,880 bytes/token**, or **0.1996%** of the conditional 2,626,187,904-byte whole-model one-read image; the observed stop avoids zero. The full forty-layer router image is 83,886,080 bytes, **3.194%** of that comparator. Reading a prefix, computing suffix norms, loading precomputed distance metadata, checking up to 8 × 248 rival inequalities and preserving normalized score bits would add costs, not subtract them.

**Result and next question.** Do not port prefix-Cauchy top-k to this Qwen engine: even a free oracle/metadata version cannot stop early on these actual prompt producers. This does not reject a learned route predictor with an exact inexpensive correction, a different tighter data-dependent suffix enclosure, generated-token distributions or a native fused-router scheduling change. A genuinely useful route consumer must both preserve scores and beat the current fused router in a complete-model unprofiled panel. The independent larger engine opportunity remains the ordinary Q8 and routed-call host/device critical path under full-head unprofiled markers; do not claim this CPU structural result as TPS.

Reproduce on CPU in bounded chunks (no GPU/service change):

```sh
OPENBLAS_NUM_THREADS=1 python3 tools/qwen-moe/router_prefix_cert.py --first 0 --count 20
OPENBLAS_NUM_THREADS=1 python3 tools/qwen-moe/router_prefix_cert.py --first 20 --count 20
OPENBLAS_NUM_THREADS=1 python3 tools/qwen-moe/router_prefix_cert.py --summarize
```

# Lowest router score is not a transferable omission certificate

The selected Qwen3.6-35B-A3B runtime evaluates eight routed experts and one shared expert per token. A tempting lower-cost observer would use the already-available normalized router score to omit the eighth expert before reading its gate/up/down images. This test separates that cheap decision from the earlier free hindsight subset oracle: the threshold is chosen **only on the disjoint train text**, then applied unchanged to held text.

## Finite contract and construction

Use the actual quantized-runtime callback's 64 train and 64 held prompt tokens at **all 40 layers**: 2,560 layer-token decisions per split. For each layer/token, form in FP64 the weighted routed sum `y = sum_i score_i down_i`. The cost-free local omission error for the lowest-score expert is `||score_8 down_8||_2 / ||y||_2`; no renormalization, retraining or shared-expert correction is allowed. For an error tolerance `t`, set a strict score threshold equal to the **minimum training score of any unsafe (`error > t`) rank-eight omission**. A second arm fits that threshold separately for each layer. If a layer has no unsafe training example, its threshold is unbounded. These are maximal lower-score threshold policies with **zero training local-error violations** in the stated family. Evaluate unchanged on held text. This is not a universal risk guarantee, and the subsequent transformer and complete-model language loss are not assessed.

| Local tolerance | Policy | Train skips / violations | Held skips / violations | Held maximum error | Held conditional whole-model one-read bytes saved |
| --- | --- | ---: | ---: | ---: | ---: |
| 1% | global | 0 / 0 | 1 / **1** | 8.12% | 0.00113% |
| 1% | per-layer | 0 / 0 | 16 / **16** | 43.91% | 0.01826% |
| 5% | global | 0 / 0 | 1 / **1** | 8.12% | 0.00113% |
| 5% | per-layer | 18 / 0 | 43 / **24** | 43.91% | 0.04887% |
| 10% | per-layer | 166 / 0 | 176 / **40** | 43.91% | 0.20051% |
| 20% | per-layer | 796 / 0 | 768 / **16** | 43.91% | 0.87514% |

At 5%, a hindsight oracle that reads the omitted output to decide could skip 88 held rank-eight assignments / 0.10000% conditional complete one-read bytes. It cannot avoid the cost of producing that output by using its norm afterward. The existing exhaustive eight-output oracle obtains 149 held assignments / 0.16940% at 5% by changing which expert is omitted and by allowing multi-expert cancellation; neither oracle is a score-only executable decision. At 1%, even the free rank-eight oracle skips zero on held; every score-only held skip violates the local criterion.

Bytes use each layer's installed expert image size from `traffic.json` and divide by the 2,626,187,904-byte conditional full-model one-read comparator. They grant free decision, no gather/dispatch overhead and unchanged surviving outputs; these fractions are **not measured DRAM savings or TPS**. Score and down files, receipt, traffic inventory and this executable are SHA-256-bound in the [CPU result](../../data/qwen-moe/route-score-concentration/receipt.json). Run `python3 tools/qwen-moe/route-score-concentration.py ../../data/qwen-moe/all-producers --out ../../data/qwen-moe/route-score-concentration/receipt.json`. No GPU, selected model or service changed.

## Decision

A zero-violation threshold on 64-token train text fails on different text, even when made layer-specific, and its saved bytes are tiny before overhead. Router score alone neither certifies a small omitted *vector* nor predicts held local safety reliably in this threshold family. A useful next representation question is an **input-dependent learned replacement direction** trained on broader real routed producers and accepted by complete forty-layer paid-image held language loss. For exact engine work, diagnose ordinary Q8/routed phase time or the first divergent split-GDN operands; do not add a score-only expert skip to serving.

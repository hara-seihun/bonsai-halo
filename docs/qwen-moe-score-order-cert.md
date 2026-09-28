# Actual forty-layer routed-sum prefix certificate

The Qwen3.6-35B-A3B router supplies eight expert IDs and normalized scores in rank order. Could the weighted down-output sum stop early with a **certified** local Euclidean error, instead of guessing from the smallest score? On both disjoint 64-token, forty-layer actual GGUF producer captures, a deliberately privileged certificate that knows every *uncomputed* expert's exact weighted-output norm skips **88 of 20,480 held expert assignments at 5% local relative error** in router order. That is at most **0.100253%** of the conditional complete one-read selected-weight bytes, before paying to obtain those norms or changing native dispatch. At 1%, no prefix omits anything, even after free norm-based reordering or free inspection of the true error. This closes norm-only early stopping as a high-fidelity frozen-bank runtime target on these observed prompt routes. It does not close changed expert representations or exact packed computation.

## Observation and certificate

Let `v_i = a_i down_i(h_i)` be the captured score-weighted FP32 output of expert `i`, recombined here in FP64. For a prefix `p_k = sum_{i<k} v_i`, grant the *true* scalar norm of every missing `v_i` without computing it; put `B_k = sum_{i>=k} ||v_i||`. Then `||y-p_k|| <= B_k` and `||y|| >= ||p_k||-B_k`, so for a positive denominator

`||y-p_k|| / ||y|| <= B_k / (||p_k|| - B_k)`.

The executable chooses the longest certified omitted suffix, not merely the first stopping opportunity. A real bound on uncomputed output norms can only loosen this particular triangle certificate. As a stronger, also nonexecutable control, it chooses the longest prefix whose *actual* full-vector error fits the tolerance. The router order is already available; the separate **norm oracle order** sorts contributions by their true weighted-output norms and grants that ordering for free. Reordering expert contributions changes native FP32 summation order; these FP64 observations do not license an exact runtime change. Both arms keep original contributions and a prefix-only omission grammar; arbitrary-subset or changed-output programs are outside it.

| Split; local tolerance | Order | Certified skipped assignments / 20,480 | Actual-safe prefix skipped | Certified cases / 2,560 | Certified complete one-read byte fraction |
| --- | --- | ---: | ---: | ---: | ---: |
| train; 1% | router / norm oracle | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0% |
| held; 1% | router / norm oracle | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0% |
| train; 5% | router | 77 | 104 | 69 | 0.088477% |
| train; 5% | norm oracle | 98 | 138 | 87 | 0.112472% |
| held; 5% | router | **88** | 118 | 76 | **0.100253%** |
| held; 5% | norm oracle | **117** | 149 | 102 | **0.133045%** |
| train; 10% | router / norm oracle | 558 / 700 | 894 / 1,112 | 510 / 646 | 0.638092 / 0.799655% |
| held; 10% | router / norm oracle | 562 / 731 | 894 / 1,150 | 510 / 664 | 0.640047 / 0.832969% |

At 5%, most certified cases omit one expert; router order certifies 88 omissions across 76 held cases. The [prior cheap score-only threshold](qwen-moe-route-score-concentration.md) guessed 43 held layer/token omissions with 24 unsafe errors; this certificate is safe **only because it receives omitted output norms for free**. The [all-subset hindsight oracle](qwen-moe-all-layer-span.md) may choose any of eight outputs and cancellation and reaches 149 held skips at 5%, compared with 118 true-safe router prefixes. Those are different grammars, not contradictory measurements. The earlier [layer-0 norm certificate](../kelana/research/moe/score-order/README.md) used a longer 113/126-token capture and an extrapolated forty-layer byte number; this report actually measures every layer on the same two 64-token captured splits.

Bytes use each layer's pinned expert bank size divided by 256 and a **2,626,187,904-byte complete conditional one-read stream per token**. This grants zero online norm/partial-sum/launch cost and supposes removing an assignment removes one full expert image read; native `mul_mat_id` grouping, cache and physical traffic are not measured. The target is the isolated weighted routed sum, not the residual-plus-shared output or downstream RMSNorm. A five-percent local vector tolerance is not a language-quality budget. No selected model, runtime, GPU lock or resident service changed.

## Custody and next question

`tools/qwen-moe/score_order_cert.py` validates the 160 captured score/output array hashes against the capture receipt, checks finite scores, recombines in FP64 and checks every certified output against the actual FP64 relative error. Its source, capture receipt, inventory and all input hashes, skip histograms and both oracle arms are saved in [the CPU receipt](../../data/qwen-moe/score-order-cert/receipt.json). Reproduce without a GPU:

```sh
python3 tools/qwen-moe/score_order_cert.py ../../data/qwen-moe/all-producers \
  --out ../../data/qwen-moe/score-order-cert/receipt.json
```

At strict local quality there is not enough unchanged-output suffix to justify constructing per-token bound machinery. For representation work, acquire broader actual *generated* and prompt producers, change the paid expert image jointly, and judge held complete-model loss and direct packed-consumer cost; no scalar norm or score-only policy solves the missing output direction. Independent engine work should focus ordinary Q8/expert physical latency with whole-head matched timing.

# A 64-example shared router-input subspace fails on actual routed producers

**Question.** Can one low-dimensional activation coordinate serve all 256 router rows in each Qwen3.6 layer, replacing the F32 router with a compact factor while preserving observed routes? Unlike exact weight-only rank, this tests the actual producer domain. This is an *approximate* carrier and must not be confused with native FP32 identity.

`tools/qwen-moe/router_producer_subspace.py` fits a centered PCA basis separately in each of forty layers using the selected GGUF's 64 train post-attention normalized activations. A held 64-token disjoint text capture supplies inputs, eight IDs, normalized scores and complete selected down outputs. For each input, the tested mathematical map is

\[c=(x-\mu)B^T,\quad l'=WB^Tc+W\mu,\quad B\in\mathbb R^{r\times2048},\quad r\in\{8,16,32,63\}.\]

The original `W` is the installed F32 256×2048 router. This differs from [Kelana's existing layer-0 uncentered input-subspace study](../../kelana/research/moe/input-subspace/README.md), which carries the input through decoded gate/up, SwiGLU and down, not the routers; the present test spans all forty router matrices and measures top-eight selection. Both full and factored logits are evaluated in FP64; the full logits' top-eight *sets* reproduce all 2,560 captured held sets. Scores and fixed-down-output weighted sums are compared **only when the route set survives**, so a changed expert's output is never invented. Input PCA minimizes train input reconstruction, not held route error or language loss; rank 63 spans the centered 64 training inputs (within numerical precision). The original router remains the control, not a native replacement.

| Rank | Held unchanged top-eight sets / 2,560 | Held tokens with ≥1 changed layer / 64 | Median layer held input relative RMS | Median layer held router-logit relative RMS | F32 factor bytes, 40 layers | Multiplies/token, 40 routers |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 5 | 64 | .9121 | .1139 | 3,317,760 | 737,280 |
| 16 | 16 | 64 | .8968 | .1093 | 6,266,880 | 1,474,560 |
| 32 | 21 | 64 | .8824 | .1066 | 12,165,120 | 2,949,120 |
| 63 | **33** | **64** | **.8660** | **.1034** | **23,592,960** | **5,806,080** |
| Installed full router | 2,560 | 0 | 0 | 0 | 83,886,080 | 20,971,520 |

At rank 63, 98.71% of held route sets change although the 64 training inputs have a complete centered-coordinate representation. The small fraction of stable cases is selected on agreement, so its local sum error is not a general quality estimate (their rank-63 RMS is .235). A router candidate needs output coordinates for changed experts, full-model held loss, quantized paid image, a directly consumed basis, and actual native timing. Even the hypothetical 72% F32 router-byte reduction is bounded to the router bank, not the complete 2.626-GB conditional one-read stream; multiplying and storing the factor are charged above, but packing and GPU behavior are not. The 83.89-MB original router occupies 3.194% of that complete stream; deleting *all* its bytes for free cannot beat that conditional byte fraction. This is not an impossibility theorem for supervised low-rank router learning, broader producer diversity, nonlinear labels or a joint activation/expert consumer.

**Decision.** Do not port this short-capture PCA router factor. Its cheap coordinate destroys the selected expert map before the scored sum can be evaluated, consistent with the short-capture output-lookup failure. The better next representation question is a changed router/expert image trained on substantially broader real producer routes and selected on disjoint complete-model language loss at paid rate, not an even narrower PCA on these 64 samples. Independent unprofiled Q8/expert native timing remains a higher immediate engine priority.

[Receipt with all 40 layers, 320 capture-array hashes, router-tensor hashes, source/model/traffic hashes and four shard hashes](../../data/qwen-moe/router-producer-subspace/receipt.json) and `part-{0,10,20,30}.json` are retained outside Git. Reproduce one bounded shard with `OPENBLAS_NUM_THREADS=2 python3 tools/qwen-moe/router_producer_subspace.py --first 0 --count 10 --output ../../data/qwen-moe/router-producer-subspace/part-0.json`, then the three subsequent tens and `--aggregate --output ../../data/qwen-moe/router-producer-subspace/receipt.json`. CPU only; no GPU, runtime, model image, service or serving default changed.

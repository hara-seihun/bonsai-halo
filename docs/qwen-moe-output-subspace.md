# A short actual-route common output basis does not transfer

[Kelana's earlier layer-0 route-covariance experiment](../../kelana/research/moe/route-covariance/README.md) already found .708799 held error at rank 512 even after fitting 113 layer-0 train tokens; a sampled full-bank prior barely helped. The new forty-layer native capture gives two disjoint 64-token real-text prefixes with the actual installed Qwen3.6-35B-A3B post-attention producers, eight IDs, normalized scores and eight down outputs at **each** layer. This study asks whether one **fixed common linear output coordinate per layer**, trained on the first text, can carry the *weighted routed sum* of the second across the actual complete stack. Its contribution is a full-stack transfer/cost census, not a novel layer-0 fitting recipe. This is a deliberately favorable screening test for shared-output factoring: projecting an already-computed output gives the factors their coefficients for free, before any weight recoding, packed-consumer cost or FP16 rounding.

For each layer, form the 512 train vectors `v[t,e] = float64(score[t,e]) * float64(down[t,e])` of width 2048. Use the leading left singular vectors of this `[512,2048]` matrix as a common orthonormal output basis `U_r`. Project each of eight held vectors into that same basis and sum, then compare with the FP64 sum of all eight held vectors. Error is `sqrt(sum_t ||sum_e v[t,e] - U_r U_r^T sum_e v[t,e]||² / sum_t ||sum_e v[t,e]||²)`. The held-slot SVD is a separate **hindsight** reference, not a deployable training method. Another arm fits the basis to the 64 *train weighted sums directly*, testing whether training against the final observation instead of individual outputs fixes transfer.

| Rank | Median train local sum error, slot basis | Median held, train slot basis | Pooled held over all 40 layers | Median held, hindsight held-slot basis | Median held, train-sum basis |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | .53461 | **.95487** | .84821 | .54298 | .95679 |
| 128 | .38623 | **.92723** | .81932 | .39640 | — |
| 256 | .21613 | **.88124** | .77496 | .21707 | — |
| 384 | .10480 | **.83928** | .73837 | .10632 | — |
| 512 | 0 (train span) | **.79948** | .70492 | 0 (held span) | — |

Even the whole 512-dimensional train-output span misses 79.95% median local held-sum RMS; the rank-256 train/held gap is .21613→.88124. Rank-64 directly fitted *weighted sums* fares no better on held (.95679). This is not merely an unseen-ID effect: 14,648/20,480 held assignments use an expert ID seen in that layer's train text; at rank 256 the median held **individual-slot** projection error is .87620 for seen IDs versus .92013 for unseen IDs. Sixty-four train tokens do not cover each expert's producer-conditioned output directions. Conversely, a basis fitted with hindsight to 512 held slot outputs attains .21707 median at rank 256. The gap changes the next experiment from another rank sweep on short captures to broad producer-diverse fitting followed by a frozen complete paid image and disjoint whole-model language loss.

## Rate and online-work boundary

For the *dense FP16 factor grammar* `D_e ≈ U B_e`, keeping one `U[2048,r]` per layer and all `B[256,512,r]` for 40 layers, the paid raw image is `40*(2048+256*512)*r*2` bytes. One selected eight-expert/token read is `40*(2048+8*512)*r*2` bytes, assuming no cache reuse. The existing mixed Q5_K/Q6_K down banks occupy **7,488,929,792** bytes (37×184,549,376 plus 3×220,200,960), or **234,029,056** conditional bytes for eight of 256 experts per layer. This is an explicit hypothetical FP16 factor image *size*, not a constructed image or entropy estimate; headers, alignment, encoding, scale metadata and repacking can only add bytes. The ideal scalar dense MAC count becomes `r*(8*512+2048)` against `8*512*2048` direct; the actual packed native consumer is not a dense-FP16 MAC machine.

| Rank | FP16 factor-bank plus basis bytes | Conditional selected-down bytes/token | Dense scalar MACs/token/layer | Median held local sum RMS |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 681,574,400 | 31,457,280 | 393,216 | .95487 |
| 256 | 2,726,297,600 | 125,829,120 | 1,572,864 | .88124 |
| 384 | 4,089,446,400 | 188,743,680 | 2,359,296 | .83928 |
| 512 | 5,452,595,200 | 251,658,240 | 3,145,728 | .79948 |

At r=256, an *ideal* factor reader would save 108,199,936 conditional down bytes/token, **4.1200% of the complete 2,626,187,904-byte one-read model comparator**, before any runtime costs, but the measured held local sum error is .88124. At r=512 it reads **17,629,184 more** conditional bytes/token than the existing packed down image, before converting FP16 to an operand. Under this grammar rank ≤476 is needed merely to save one-read down bytes. Replacing the actual Q5/Q6 decoder, eight factor multiplications, score application, a new dense projection, scheduling and synchronization remain unpriced native work. The outputs' FP64 summation/projection is not the installed FP32 addition order; no bit-identity, language quality or TPS follows.

This is a **measured negative for train-fitted short-capture fixed output bases**, not a rank lower bound on all reachable outputs or a claim that MoE cannot be compressed. The held-hindsight slot basis is optimal for held *slot* reconstruction, not necessarily the weighted sum; 64 held weighted sums themselves can trivially be spanned at rank 64 with hindsight. Captured callback execution alters graph topology. The short train/held texts are different prefixes, not a broad fit/validation corpus. Learned input-dependent coordinates, route-conditioned bases, quantized factors and nonlinear consumers are outside this grammar. The dense sub-bit and complete ternary failures warn that even a much better local fit still requires a paid forty-layer model and disjoint whole-model loss before native adoption.

## Reproduction and custody

`tools/qwen-moe/output_subspace.py` uses FP64 linear algebra and checks dimensions, finite scores/outputs and orthogonality. Run `OPENBLAS_NUM_THREADS=1 python3 tools/qwen-moe/output_subspace.py ../../data/qwen-moe/all-producers --first 0 --count 10 --output ../../data/qwen-moe/output-subspace/layers-0-9.json` and repeat for first=10,20,30. Each chunk hashes its source script, all scores, down outputs and selected IDs; the [summary and four chunk receipts](../../data/qwen-moe/output-subspace/summary.json) tie the result to the [native capture receipt](../../data/qwen-moe/all-producers/receipt.json), which owns model, runtime binary, producer and input identities. CPU only; no GPU reservation, selected runtime, GGUF or Bonsai resident service changed.

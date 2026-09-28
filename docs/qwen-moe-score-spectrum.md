# Actual Qwen routed sums resist a narrow score-only tangent

The [exact affine-rank witness](../../kelana/research/moe/score-observation-rank/README.md) proves seven score dimensions are needed to reproduce the real weighted sum locally when the eight down vectors are fixed. This experiment asks the approximate version: **how much of the local weighted-sum response would even an output-aware, per-token optimal rank-`r` score carrier discard?** On all 5,120 captured layer/token cases from disjoint 64-token prompt splits, a free rank-three tangent map still loses **0.5627 train / 0.5497 held pooled RMS** of the score-dependent response. Even rank six loses **0.2246 / 0.2226**. The oracle is tailored to the exact eight down outputs of *each* token, with free encoding and decoding; an executable shared carrier cannot claim these errors as its attainable quality.

| Rank `r` of local score carrier | Train pooled lost tangent RMS | Held pooled lost tangent RMS | Held derivative error relative to actual routed sum at tangent norm .01 |
|---:|---:|---:|---:|
| 1 | .8063 | .8005 | .01549 |
| 2 | .6762 | .6618 | .01280 |
| 3 | .5627 | .5497 | .01063 |
| 4 | .4548 | .4452 | .00861 |
| 5 | .3450 | .3398 | .00657 |
| 6 | .2246 | .2226 | .00431 |

The last column scales the **derivative** by .01 in Euclidean score coordinates; it is not a finite score perturbation, model-quality score or one-percent tolerance certificate. On held cases rank-three loss relative to all seven tangent directions has median .6145 and p95 .6888; rank-six median .2559 and p95 .3135. Held Gram condition-number median is 2.067 and p95 4.067, though rare observed routes reach 37.66. The source receipt gives minima, maxima, all ranks, both splits, and input hashes.

## Contract and bound

For one captured `(layer, token)`, fix the eight FP32 down vectors `Y_i` as exact real constants and vary positive normalized real scores `a_i` in a sufficiently small open neighborhood of the measured scores renormalized by their positive sum. Let `H` be the explicitly checked orthonormal 8×7 Helmert matrix spanning the zero-sum score tangent. The complete isolated routed-sum derivative is `D = Yᵀ H`, a 2048×7 real map. Its seven positive Gram eigenvalues `λ₁≥...≥λ₇` are computed in FP64 from the saved FP32 vectors. For an isotropic *unit* tangent displacement, the best rank-`r` linear derivative approximant has expected squared error `Σ_{j>r} λ_j / 7`, by Eckart–Young; the full derivative has `Σ_j λ_j / 7`. Each reported pooled response ratio is `sqrt(Σ_cases Σ_{j>r} λ_j / Σ_cases Σ_j λ_j)`. The final table column is `.01 sqrt(Σ_cases Σ_{j>r} λ_j / (7 Σ_cases ||Σ_i a_i Y_i||²))`. A differentiable nonlinear score-only encoder/consumer with `r` real intermediate coordinates has a derivative of rank at most `r`, so the same **local first-order** lower bound holds for its Jacobian at each fixed-output observation. The full nonlinear finite-neighborhood error need not equal this derivative prediction.

The proof is exact-real linear algebra; the positive numerical eigenvalues and their magnitudes are FP64 measurements, not interval-certified singular values. The prior modular-rank witness supplies the exact positive-rank certificate on the same captured vectors. The per-token optimal basis is allowed to depend on the *full* output vectors, a computationally expensive oracle, and the isotropic tangent measure is a declared hypothetical perturbation rather than the router's empirical score distribution. The actual following residual-plus-shared branch, RMSNorm epsilon, FP32 fold order, route changes, cross-token carriers and joint producer/score encodings are **outside** this observation contract. No image, native consumer, full-model loss, physical traffic or TPS changed.

## Implication and custody

Dropping three or more of seven normalized-score degrees through a score-only carrier does not leave a nearly invisible local output response on these real routes, even with a free per-token optimal basis. This narrows the next representation search toward **joint producer/output/score labels that survive the next consumer**, rather than another score-only narrow projection. If approximating scores is still attractive, obtain broader generated producers, specify an actual score distribution and paid carrier, then evaluate held complete-model language loss. The independent high-cost engine target is ordinary Q8/expert execution; the traced router top-k budget is small.

Run from Bonsai's root with NumPy: `OPENBLAS_NUM_THREADS=1 python3 tools/qwen-moe/score_spectrum.py --output ../../data/qwen-moe/score-spectrum/receipt.json`. The [CPU receipt](../../data/qwen-moe/score-spectrum/receipt.json) binds source SHA-256, native capture receipt/model identity and every one of 160 down/score array hashes; all 5,120 cases are included. This used no GPU and did not alter the selected runtime or resident service.

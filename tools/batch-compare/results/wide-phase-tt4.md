# Full-model phase profile

Source: `../../data/bonsai2/batch-comparison/wide-phase-tt4.json`.

Prefix 128 tokens, 3 rounds. Every sample is retained; phase values are per-group medians. No cross-mode timing ratio is inferred from counter-instrumented runs.

## Tracing perturbation

| mode | rows | logits | native wall ms | traced wall ms | traced device span ms | residual hashes |
|---|---:|---|---|---|---|---:|
| 18 | 128 | True | 316.85, 315.21, 315.86 | 330.62, 327.56, 327.47 | 327.05, 324.02, 323.26 | 1 |

A single residual hash per group means the 64-bit residual checksums agree across these samples. Trace wall time includes collection; device spans exclude readback. Outliers remain in the list and in the median calculation.

## Mode 18, 128 rows, logits on every row

| phase | median ms |
|---|---:|
| embed | 0.261 |
| sequence-input-prep | 26.478 |
| sequence-input-projection | 39.517 |
| sequence-core | 69.511 |
| sequence-output-projection | 15.620 |
| ffn | 139.551 |
| head | 26.575 |
| between launches | 6.229 |
| embed: embedding gather | 0.089 |
| embed: embedding transform | 0.034 |
| gdn: input prep + alpha/beta | 8.962 |
| gdn: conv + GDN prep | 5.634 |
| gdn: GDN state + replay | 24.720 |
| gdn: output prep | 2.392 |
| attention: input prep | 1.137 |
| attention: rotary + KV prep | 1.418 |
| attention: QK/softmax/AV | 9.252 |
| attention: output prep | 0.995 |
| head: input prep | 0.109 |
| head: vocabulary projection | 24.960 |
| head: argmax slices | 1.400 |

The sequence total contains the GDN and attention subphases. Do not add them twice. Barrier stamps include grid waits; event totals also include unstamped prologue and epilogue work.

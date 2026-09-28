# Full-model phase profile

Source: `../../data/bonsai2/batch-comparison/direct-layout-phase-staged.json`.

Prefix 128 tokens, 3 rounds. Every sample is retained; phase values are per-group medians. No cross-mode timing ratio is inferred from counter-instrumented runs.

Sequence layout `staged`, operand `int8`, tile width `auto`.

## Tracing perturbation

| mode | rows | logits | native wall ms | traced wall ms | traced device span ms | residual hashes |
|---|---:|---|---|---|---|---:|
| 19 | 128 | True | 265.75, 265.99, 265.83 | 280.22, 278.14, 279.26 | 276.76, 275.40, 274.69 | 1 |

A single residual hash per group means the 64-bit residual checksums agree across these samples. Trace wall time includes collection; device spans exclude readback. Outliers remain in the list and in the median calculation.

## Mode 19, 128 rows, logits on every row

| phase | median ms |
|---|---:|
| embed | 0.253 |
| sequence-input-prep | 25.456 |
| sequence-input-projection | 37.478 |
| sequence-core | 67.126 |
| sequence-output-projection | 16.546 |
| ffn | 97.327 |
| head | 25.985 |
| between launches | 6.046 |
| embed: embedding gather | 0.085 |
| embed: embedding transform | 0.034 |
| gdn: input prep + alpha/beta | 8.525 |
| gdn: conv + GDN prep | 5.414 |
| gdn: GDN state + replay | 23.669 |
| gdn: output prep | 2.358 |
| attention: input prep | 1.058 |
| attention: rotary + KV prep | 1.297 |
| attention: QK/softmax/AV | 8.829 |
| attention: output prep | 0.972 |
| head: input prep | 0.100 |
| head: vocabulary projection | 24.384 |
| head: argmax slices | 1.359 |

The sequence total contains the GDN and attention subphases. Do not add them twice. Barrier stamps include grid waits; event totals also include unstamped prologue and epilogue work.

# Full-model phase profile

Source: `../../data/bonsai2/batch-comparison/direct-layout-plane.json`.

Prefix 128 tokens, 3 rounds. Every sample is retained; phase values are per-group medians. No cross-mode timing ratio is inferred from counter-instrumented runs.

Sequence layout `direct`, operand `int8`, tile width `auto`.

## Tracing perturbation

| mode | rows | logits | native wall ms | traced wall ms | traced device span ms | residual hashes |
|---|---:|---|---|---|---|---:|
| 19 | 128 | True | 252.39, 252.40, 253.15 | 260.29, 261.34, 260.99 | 257.70, 259.25, 258.76 | 1 |

A single residual hash per group means the 64-bit residual checksums agree across these samples. Trace wall time includes collection; device spans exclude readback. Outliers remain in the list and in the median calculation.

## Mode 19, 128 rows, logits on every row

| phase | median ms |
|---|---:|
| embed | 0.247 |
| sequence-input-prep | 19.724 |
| sequence-input-projection | 37.808 |
| sequence-core | 56.375 |
| sequence-output-projection | 16.054 |
| ffn | 96.694 |
| head | 25.826 |
| between launches | 6.095 |
| embed: embedding gather | 0.085 |
| embed: embedding transform | 0.027 |
| gdn: input prep + alpha/beta | 9.048 |
| gdn: conv + GDN prep | 6.851 |
| gdn: GDN state + replay | 23.608 |
| gdn: output prep | 3.860 |
| attention: input prep | 1.275 |
| attention: rotary + KV prep | 1.677 |
| attention: QK/softmax/AV | 8.820 |
| attention: output prep | 1.357 |
| head: input prep | 0.106 |
| head: vocabulary projection | 24.211 |
| head: argmax slices | 1.337 |

The sequence total contains the GDN and attention subphases. Do not add them twice. Barrier stamps include grid waits; event totals also include unstamped prologue and epilogue work.

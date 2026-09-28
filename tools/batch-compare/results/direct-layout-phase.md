# Full-model phase profile

Source: `../../data/bonsai2/batch-comparison/direct-layout-phase.json`.

Prefix 128 tokens, 3 rounds. Every sample is retained; phase values are per-group medians. No cross-mode timing ratio is inferred from counter-instrumented runs.

Sequence layout `direct`, operand `int8`, tile width `auto`.

## Tracing perturbation

| mode | rows | logits | native wall ms | traced wall ms | traced device span ms | residual hashes |
|---|---:|---|---|---|---|---:|
| 19 | 128 | True | 255.61, 256.38, 255.88 | 263.65, 265.65, 263.89 | 261.46, 262.26, 261.48 | 1 |

A single residual hash per group means the 64-bit residual checksums agree across these samples. Trace wall time includes collection; device spans exclude readback. Outliers remain in the list and in the median calculation.

## Mode 19, 128 rows, logits on every row

| phase | median ms |
|---|---:|
| embed | 0.251 |
| sequence-input-prep | 19.791 |
| sequence-input-projection | 41.168 |
| sequence-core | 55.926 |
| sequence-output-projection | 16.105 |
| ffn | 96.531 |
| head | 26.029 |
| between launches | 6.025 |
| embed: embedding gather | 0.085 |
| embed: embedding transform | 0.034 |
| gdn: input prep + alpha/beta | 9.115 |
| gdn: conv + GDN prep | 7.073 |
| gdn: GDN state + replay | 23.612 |
| gdn: output prep | 3.378 |
| attention: input prep | 1.276 |
| attention: rotary + KV prep | 1.723 |
| attention: QK/softmax/AV | 8.837 |
| attention: output prep | 1.278 |
| head: input prep | 0.105 |
| head: vocabulary projection | 24.431 |
| head: argmax slices | 1.316 |

The sequence total contains the GDN and attention subphases. Do not add them twice. Barrier stamps include grid waits; event totals also include unstamped prologue and epilogue work.

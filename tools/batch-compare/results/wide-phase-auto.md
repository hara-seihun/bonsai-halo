# Full-model phase profile

Source: `../../data/bonsai2/batch-comparison/wide-phase-auto.json`.

Prefix 128 tokens, 3 rounds. Every sample is retained; phase values are per-group medians. No cross-mode timing ratio is inferred from counter-instrumented runs.

## Tracing perturbation

| mode | rows | logits | native wall ms | traced wall ms | traced device span ms | residual hashes |
|---|---:|---|---|---|---|---:|
| 10 | 128 | True | 460.15, 457.61, 456.94 | 465.92, 463.15, 461.55 | 463.80, 460.91, 456.86 | 1 |
| 18 | 128 | True | 314.91, 310.69, 303.97 | 323.00, 317.75, 319.47 | 319.63, 311.07, 313.13 | 1 |
| 19 | 128 | True | 265.86, 264.71, 263.31 | 278.13, 277.40, 276.02 | 274.67, 274.20, 272.57 | 1 |

A single residual hash per group means the 64-bit residual checksums agree across these samples. Trace wall time includes collection; device spans exclude readback. Outliers remain in the list and in the median calculation.

## Mode 10, 128 rows, logits on every row

| phase | median ms |
|---|---:|
| embed | 0.258 |
| sequence | 292.811 |
| ffn | 138.834 |
| head | 26.087 |
| between launches | 3.107 |
| embed: embedding gather | 0.089 |
| embed: embedding transform | 0.034 |
| gdn: input prep + alpha/beta | 16.642 |
| gdn: qkv + gate projection | 88.719 |
| gdn: conv + GDN prep | 7.645 |
| gdn: GDN state + replay | 60.668 |
| gdn: output prep | 3.456 |
| gdn: output projection | 50.716 |
| attention: input prep | 1.576 |
| attention: qkv projection | 24.718 |
| attention: rotary + KV prep | 1.344 |
| attention: QK/softmax/AV | 9.623 |
| attention: output prep | 1.094 |
| attention: output projection | 15.909 |
| head: input prep | 0.104 |
| head: vocabulary projection | 24.465 |
| head: argmax slices | 1.372 |

The sequence total contains the GDN and attention subphases. Do not add them twice. Barrier stamps include grid waits; event totals also include unstamped prologue and epilogue work.

## Mode 18, 128 rows, logits on every row

| phase | median ms |
|---|---:|
| embed | 0.258 |
| sequence-input-prep | 24.961 |
| sequence-input-projection | 37.904 |
| sequence-core | 65.571 |
| sequence-output-projection | 16.165 |
| ffn | 136.101 |
| head | 25.956 |
| between launches | 6.285 |
| embed: embedding gather | 0.086 |
| embed: embedding transform | 0.034 |
| gdn: input prep + alpha/beta | 8.649 |
| gdn: conv + GDN prep | 5.406 |
| gdn: GDN state + replay | 23.475 |
| gdn: output prep | 2.329 |
| attention: input prep | 1.097 |
| attention: rotary + KV prep | 1.325 |
| attention: QK/softmax/AV | 8.828 |
| attention: output prep | 0.946 |
| head: input prep | 0.108 |
| head: vocabulary projection | 24.314 |
| head: argmax slices | 1.374 |

The sequence total contains the GDN and attention subphases. Do not add them twice. Barrier stamps include grid waits; event totals also include unstamped prologue and epilogue work.

## Mode 19, 128 rows, logits on every row

| phase | median ms |
|---|---:|
| embed | 0.254 |
| sequence-input-prep | 25.311 |
| sequence-input-projection | 37.753 |
| sequence-core | 66.456 |
| sequence-output-projection | 16.289 |
| ffn | 96.558 |
| head | 25.424 |
| between launches | 6.130 |
| embed: embedding gather | 0.087 |
| embed: embedding transform | 0.033 |
| gdn: input prep + alpha/beta | 8.641 |
| gdn: conv + GDN prep | 5.418 |
| gdn: GDN state + replay | 23.625 |
| gdn: output prep | 2.315 |
| attention: input prep | 1.088 |
| attention: rotary + KV prep | 1.340 |
| attention: QK/softmax/AV | 8.807 |
| attention: output prep | 0.945 |
| head: input prep | 0.105 |
| head: vocabulary projection | 23.809 |
| head: argmax slices | 1.350 |

The sequence total contains the GDN and attention subphases. Do not add them twice. Barrier stamps include grid waits; event totals also include unstamped prologue and epilogue work.

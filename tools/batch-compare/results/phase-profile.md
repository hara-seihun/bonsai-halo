# Full-model phase profile

Source: `../../data/bonsai2/batch-comparison/phase-profile/run.json`.

Prefix 128 tokens, 3 rounds. Every sample is retained; phase values are per-group medians. No cross-mode timing ratio is inferred from counter-instrumented runs.

## Tracing perturbation

| mode | rows | logits | native wall ms | traced wall ms | traced device span ms | residual hashes |
|---|---:|---|---|---|---|---:|
| 10 | 32 | False | 129.24, 127.85, 129.96 | 130.81, 130.85, 130.07 | 130.14, 129.57, 128.95 | 1 |
| 10 | 32 | True | 135.79, 134.87, 134.66 | 136.41, 137.64, 136.71 | 135.92, 136.47, 135.83 | 1 |
| 10 | 128 | False | 425.78, 427.60, 434.70 | 437.62, 431.11, 431.78 | 435.98, 427.63, 428.56 | 1 |
| 10 | 128 | True | 461.26, 450.55, 462.43 | 463.58, 459.17, 458.94 | 461.48, 455.61, 455.87 | 1 |
| 11 | 32 | False | 109.31, 108.99, 109.06 | 110.79, 109.26, 110.40 | 110.19, 108.37, 109.97 | 1 |
| 11 | 32 | True | 115.78, 114.54, 115.33 | 117.04, 116.98, 116.96 | 116.46, 116.32, 116.15 | 1 |
| 11 | 128 | False | 385.40, 385.70, 1472.50 | 390.87, 389.56, 389.62 | 389.04, 387.85, 388.06 | 1 |
| 11 | 128 | True | 410.92, 410.70, 410.22 | 415.67, 415.14, 414.03 | 413.86, 413.13, 412.25 | 1 |

A single residual hash per group means the 64-bit residual checksums agree across these samples. Trace wall time includes collection; device spans exclude readback. Outliers remain in the list and in the median calculation.

## Mode 10, 32 rows, logits on every row

| phase | median ms |
|---|---:|
| embed | 0.069 |
| sequence | 75.591 |
| ffn | 53.153 |
| head | 6.249 |
| between launches | 0.878 |
| embed: embedding gather | 0.023 |
| embed: embedding transform | 0.009 |
| gdn: input prep + alpha/beta | 4.226 |
| gdn: qkv + gate projection | 22.849 |
| gdn: conv + GDN prep | 2.111 |
| gdn: GDN state + replay | 15.593 |
| gdn: output prep | 0.819 |
| gdn: output projection | 13.150 |
| attention: input prep | 0.394 |
| attention: qkv projection | 6.385 |
| attention: rotary + KV prep | 0.349 |
| attention: QK/softmax/AV | 2.540 |
| attention: output prep | 0.313 |
| attention: output projection | 4.125 |
| head: input prep | 0.026 |
| head: vocabulary projection | 5.834 |
| head: argmax slices | 0.343 |

The sequence total contains the GDN and attention subphases. Do not add them twice. Barrier stamps include grid waits; event totals also include unstamped prologue and epilogue work.

## Mode 10, 128 rows, logits on every row

| phase | median ms |
|---|---:|
| embed | 0.261 |
| sequence | 287.739 |
| ffn | 139.727 |
| head | 25.455 |
| between launches | 3.015 |
| embed: embedding gather | 0.086 |
| embed: embedding transform | 0.033 |
| gdn: input prep + alpha/beta | 16.457 |
| gdn: qkv + gate projection | 87.340 |
| gdn: conv + GDN prep | 7.845 |
| gdn: GDN state + replay | 59.494 |
| gdn: output prep | 3.277 |
| gdn: output projection | 50.219 |
| attention: input prep | 1.536 |
| attention: qkv projection | 24.069 |
| attention: rotary + KV prep | 1.349 |
| attention: QK/softmax/AV | 9.231 |
| attention: output prep | 1.043 |
| attention: output projection | 15.473 |
| head: input prep | 0.105 |
| head: vocabulary projection | 23.830 |
| head: argmax slices | 1.367 |

The sequence total contains the GDN and attention subphases. Do not add them twice. Barrier stamps include grid waits; event totals also include unstamped prologue and epilogue work.

## Mode 11, 32 rows, logits on every row

| phase | median ms |
|---|---:|
| embed | 0.067 |
| sequence | 75.809 |
| ffn | 33.307 |
| head | 6.264 |
| between launches | 0.869 |
| embed: embedding gather | 0.022 |
| embed: embedding transform | 0.009 |
| gdn: input prep + alpha/beta | 4.253 |
| gdn: qkv + gate projection | 23.001 |
| gdn: conv + GDN prep | 2.125 |
| gdn: GDN state + replay | 15.622 |
| gdn: output prep | 0.825 |
| gdn: output projection | 13.159 |
| attention: input prep | 0.410 |
| attention: qkv projection | 6.459 |
| attention: rotary + KV prep | 0.352 |
| attention: QK/softmax/AV | 2.545 |
| attention: output prep | 0.327 |
| attention: output projection | 4.136 |
| head: input prep | 0.027 |
| head: vocabulary projection | 5.865 |
| head: argmax slices | 0.339 |

The sequence total contains the GDN and attention subphases. Do not add them twice. Barrier stamps include grid waits; event totals also include unstamped prologue and epilogue work.

## Mode 11, 128 rows, logits on every row

| phase | median ms |
|---|---:|
| embed | 0.251 |
| sequence | 288.400 |
| ffn | 96.178 |
| head | 25.196 |
| between launches | 2.992 |
| embed: embedding gather | 0.085 |
| embed: embedding transform | 0.033 |
| gdn: input prep + alpha/beta | 16.344 |
| gdn: qkv + gate projection | 87.534 |
| gdn: conv + GDN prep | 7.881 |
| gdn: GDN state + replay | 59.522 |
| gdn: output prep | 3.312 |
| gdn: output projection | 50.355 |
| attention: input prep | 1.564 |
| attention: qkv projection | 24.275 |
| attention: rotary + KV prep | 1.333 |
| attention: QK/softmax/AV | 9.189 |
| attention: output prep | 1.029 |
| attention: output projection | 15.470 |
| head: input prep | 0.107 |
| head: vocabulary projection | 23.555 |
| head: argmax slices | 1.370 |

The sequence total contains the GDN and attention subphases. Do not add them twice. Barrier stamps include grid waits; event totals also include unstamped prologue and epilogue work.

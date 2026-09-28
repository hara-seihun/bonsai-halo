# Wide vocabulary head: full-model comparison

Same executable on both sides (`tools/batch_compare` SHA-256 prefix `516fb65d8713`), source
`b5a8c55` plus the wide-head change. The control arm sets `HALO_WIDE_HEAD=0`, which also allocates
none of the module's memory. Document: 384 tokens of `PLAN.md`, logits on the final pass only.
Generation: independent streams from `tools/batch-compare/prompts.txt`, eight greedy steps, context
512. Three rounds each, medians with every sample listed.

## Prompt processing (tokens/s)

| mode | rows/pass | sliced head | wide head | gain |
|---|---:|---:|---:|---:|
| 19 wide-commit-a4 | 128 | 600.3 (600.8, 599.3, 600.3) | 614.7 (614.7, 613.4, 615.5) | 2.4% |
| 19 wide-commit-a4 | 32 | 441.1 (441.1, 440.5, 441.2) | 444.0 (444.2, 443.8, 444.0) | 0.7% |
| 18 wide-commit | 128 | 485.5 (485.5, 486.5, 485.4) | 497.7 (495.1, 497.7, 501.8) | 2.5% |
| 18 wide-commit | 32 | 344.9 (344.9, 344.6, 345.1) | 346.8 (346.8, 346.1, 347.2) | 0.6% |

## Aggregate generation (tokens/s)

| mode | streams | sliced head | wide head | gain |
|---|---:|---:|---:|---:|
| 19 wide-commit-a4 | 32 | 253.4 (253.1, 253.5, 253.4) | 260.0 (260.1, 260.0, 259.5) | 2.6% |
| 19 wide-commit-a4 | 8 | 147.4 (147.4, 147.9, 147.4) | 147.5 (148.0, 147.5, 146.3) | none |
| 18 wide-commit | 32 | 210.4 (210.3, 210.6, 210.4) | 215.0 (215.0, 215.0, 215.0) | 2.2% |
| 18 wide-commit | 8 | 147.5 (147.5, 147.5, 147.5) | 147.5 (147.5, 147.5, 147.4) | none |

Eight streams is the designed null: the wide head needs more than `RMAX` rows, so that shape keeps
the sliced head and must not move.

## Isolated head cost

`tools/batch_profile --heads 0,1` runs the same pass with and without the output head; the
difference is the whole head. Medians of three samples, mode 19, 128-token prefix. `tools/batch_profile`
SHA-256 prefixes `91fd6175cc5d66bc` (128-row panels) and `25e3b915d99e5bbe` (32-row panels).

| route | 128-row pass | 32-row pass |
|---|---:|---:|
| sliced | 26.554 ms | 6.370 ms |
| wide, `HALO_HEAD_TT=1` | 13.070 | 2.999 |
| wide, `HALO_HEAD_TT=2` (selected) | 11.204 | 2.690 |
| wide, `HALO_HEAD_TT=4` | 13.160 | 5.665 |
| wide, `HALO_HEAD_TT=8` | 20.323 | |

Raw: `wide-head-phase/{control,tt1,tt2,tt4,wide,r32-*}.json` and the rocprof kernel trace in
`wide-head-trace/`, both under
[`../../data/bonsai2/batch-comparison`](../../../../../data/bonsai2/batch-comparison/README.md).

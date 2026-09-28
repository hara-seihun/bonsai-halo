# Wide sequence input prep: full-model comparison

Same executable on both sides (`tools/batch_compare` SHA-256 prefix `0b6c070442e6`), source
`f069a08` plus the head trace fix and the wide-prep change. The control arm sets
`HALO_WIDE_PREP=0`. Document: 384 tokens of `PLAN.md`, logits on the final pass only.
Generation: independent streams from `tools/batch-compare/prompts.txt`, eight greedy steps,
context 512. Medians with every sample listed.

## Prompt processing (tokens/s)

| mode | rows/pass | per-slice prep | wide prep | gain |
|---|---:|---:|---:|---:|
| 19 wide-commit-a4 | 128 | 604.7 (604.7, 597.9, 615.5) | 661.5 (656.8, 661.5, 671.1) | 9.4% |
| 19 wide-commit-a4 | 32 | 441.4 (442.2, 441.4, 441.1) | 467.5 (466.5, 467.5, 469.3) | 5.9% |

## Aggregate generation (tokens/s)

| mode | streams | per-slice prep | wide prep | gain |
|---|---:|---:|---:|---:|
| 19 wide-commit-a4 | 32 | 260.6 (260.3, 260.6, 261.0) | 268.7 (268.7, 268.6) | 3.1% |

The selected decode arm has two samples, not three: the measurement call's timeout ended the
panel during its third round. The two recorded samples differ by 0.1 tok/s and both exceed every
control sample.

Eight streams and single-stream decode do not reach this route at all. The wide sequence path
needs 32 rows, so those shapes keep the deployed per-slice prep and are unaffected by
construction, not by measurement.

## Isolated prep cost

One traced 128-row A4 pass with logits, `tools/batch_profile --modes 19 --rows 128 --heads 1`:

| route | launches | `sequence-input-prep` | device span |
|---|---:|---:|---:|
| per-slice | 1024 | 19.872 ms | 221.88 ms |
| wide | 64 | 3.574 ms | 204.69 ms |

Both report residual FNV-64 `7446865760224376151`.

Per-launch, the wide route costs 66.0 microseconds on a recurrent layer and 25.4 on an attention
layer; the difference is the alpha/beta projection, about 1.95 ms of the pass. The per-slice route
cost 21.08 and 14.37 microseconds per launch, of which 11.95 and 5.06 were inside the kernel.

Raw: `wide-prep-a4-{prefill,decode}-{on,off}/` and
`ffn-frontier/phase-m19{,-wideprep}.json` under
[`../../data/bonsai2/batch-comparison`](../../../../../data/bonsai2/batch-comparison/README.md).

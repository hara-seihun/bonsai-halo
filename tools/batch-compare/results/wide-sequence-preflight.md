# Batch mode comparison: wide-sequence-preflight

Source: `../../data/bonsai2/batch-comparison/wide-sequence-preflight/run.json`, run 2026-09-20T23:13:02 to 2026-09-20T23:13:11, resident server paused: True, full model layers: 64, batch capacity 128.
Model ../../data/bonsai2/PTQ1_0.gguf, context 512, slots 2, max rows per pass 128, document PLAN.md, prompts tools/batch-compare/prompts.txt.
0 rounds, mode order reshuffled per round (seed 1234), clock ramp skipped, no timed workload, single process, state reset before every timed region.
Device memory: 7.44 GB for the model, 18.21 GB after `prepare_batch(128, 0x80)`, of which 8.84 GB is the batched FFN module. Weight images for the modes measured budget at 8.82 GB, against 16.58 GB to hold every mode at once. Each image is a further representation of all 64 layers, held alongside the deployed weights for as long as the mode that reads it is available.
Resident weight images: SCALES 0.27 GB, SLICE2 4.28 GB, LANE2 4.28 GB.
Modes measured: 0 deployed (`--ffn deployed`), 10 auto-optimized-scaled-a8 (`--ffn auto-map-scaled-a8`), 16 commit-scaled (`--ffn commit-scaled`), 17 wide-sequence (`--ffn wide-sequence`), 18 wide-commit (`--ffn wide-commit`).
Sequence routes: mode 16 commits without widening; 17 widens without committing; 18 and 19 do both. Widening applies at 32 or more total rows; committing applies at every row count. FFN automatic dispatch is separate from this choice.
Wide-sequence operand: `int8`; token tile width: `auto`. `scaled-f16` rounds dequantised A8 inputs to FP16 and changes accumulation order; it is a different numerical map, not a layout-only option.
Matched reference pairs from run.json --reference-pairs: `16:10`, `18:17`.
A pair's scope selects which reported shape the pair is applied to. It does not certify that every pass inside that measurement took the matched route. Teacher-forced context, prompt prefill and the tail pass of a document can carry fewer than 32 rows, and there an automatic mode runs the deployed or sliced path however wide the reported rows are. `9:5` needs no scope for a different reason: both modes switch on the same row counts, so they agree pass for pass including the narrow ones. Read every other pair as a statement about the shape it names.

No timed rounds in this run.

### quality, prefill shape, 32 rows

32 teacher-forced rows, full 248320-entry vocabulary, identical token inputs per mode.
Rows are document tokens following the same 64-token context, so a smaller row count is a prefix of a larger one: the row counts are nested, not independent samples.

| mode | top1 agreement | rows changed | mean KL(0->m) | max KL | max centred logit diff | mean top5 overlap | worst row rank of mode0 top1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 deployed | reference | 0 | - | - | - | - | - |
| 10 auto-optimized-scaled-a8 | 32/32 | 0 | 0.0008 | 0.0051 | 0.329 | 4.9/5 | 0 |
| 16 commit-scaled | 32/32 | 0 | 0.0008 | 0.0051 | 0.329 | 4.9/5 | 0 |
| 17 wide-sequence | 32/32 | 0 | 0.0007 | 0.0028 | 0.304 | 4.9/5 | 0 |
| 18 wide-commit | 32/32 | 0 | 0.0007 | 0.0028 | 0.304 | 4.9/5 | 0 |

Against the matched reference for each mode, same tokens, same rows:

| pair | top1 agreement | logit elements differing | rows differing | RMS logit diff | max abs diff | non-finite mode/ref |
|---|---:|---:|---:|---:|---:|---:|
| 16 vs 10 | 32/32 | 0/7946240 | 0/32 | 0.000e+00 | 0.000e+00 | 0/0 |
| 18 vs 17 | 32/32 | 0/7946240 | 0/32 | 0.000e+00 | 0.000e+00 | 0/0 |

- mode 16 reproduced mode 10 exactly here: all 7946240 logit values are finite and have identical bit patterns.
- mode 18 reproduced mode 17 exactly here: all 7946240 logit values are finite and have identical bit patterns.

These verdicts are about the shape named above, not about every pass that produced it; see the scope note at the top of this report.

### quality, prefill shape, 40 rows

40 teacher-forced rows, full 248320-entry vocabulary, identical token inputs per mode.
Rows are document tokens following the same 64-token context, so a smaller row count is a prefix of a larger one: the row counts are nested, not independent samples.

| mode | top1 agreement | rows changed | mean KL(0->m) | max KL | max centred logit diff | mean top5 overlap | worst row rank of mode0 top1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 deployed | reference | 0 | - | - | - | - | - |
| 10 auto-optimized-scaled-a8 | 40/40 | 0 | 0.0007 | 0.0051 | 0.329 | 4.8/5 | 0 |
| 16 commit-scaled | 40/40 | 0 | 0.0007 | 0.0051 | 0.329 | 4.8/5 | 0 |
| 17 wide-sequence | 40/40 | 0 | 0.0007 | 0.0028 | 0.304 | 4.9/5 | 0 |
| 18 wide-commit | 40/40 | 0 | 0.0007 | 0.0028 | 0.304 | 4.9/5 | 0 |

Against the matched reference for each mode, same tokens, same rows:

| pair | top1 agreement | logit elements differing | rows differing | RMS logit diff | max abs diff | non-finite mode/ref |
|---|---:|---:|---:|---:|---:|---:|
| 16 vs 10 | 40/40 | 0/9932800 | 0/40 | 0.000e+00 | 0.000e+00 | 0/0 |
| 18 vs 17 | 40/40 | 0/9932800 | 0/40 | 0.000e+00 | 0.000e+00 | 0/0 |

- mode 16 reproduced mode 10 exactly here: all 9932800 logit values are finite and have identical bit patterns.
- mode 18 reproduced mode 17 exactly here: all 9932800 logit values are finite and have identical bit patterns.

These verdicts are about the shape named above, not about every pass that produced it; see the scope note at the top of this report.

### quality, prefill shape, 128 rows

128 teacher-forced rows, full 248320-entry vocabulary, identical token inputs per mode.
Rows are document tokens following the same 64-token context, so a smaller row count is a prefix of a larger one: the row counts are nested, not independent samples.

| mode | top1 agreement | rows changed | mean KL(0->m) | max KL | max centred logit diff | mean top5 overlap | worst row rank of mode0 top1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 deployed | reference | 0 | - | - | - | - | - |
| 10 auto-optimized-scaled-a8 | 128/128 | 0 | 0.0004 | 0.0051 | 1.234 | 4.9/5 | 0 |
| 16 commit-scaled | 128/128 | 0 | 0.0004 | 0.0051 | 1.234 | 4.9/5 | 0 |
| 17 wide-sequence | 128/128 | 0 | 0.0004 | 0.0028 | 1.458 | 4.9/5 | 0 |
| 18 wide-commit | 128/128 | 0 | 0.0004 | 0.0028 | 1.458 | 4.9/5 | 0 |

Against the matched reference for each mode, same tokens, same rows:

| pair | top1 agreement | logit elements differing | rows differing | RMS logit diff | max abs diff | non-finite mode/ref |
|---|---:|---:|---:|---:|---:|---:|
| 16 vs 10 | 128/128 | 0/31784960 | 0/128 | 0.000e+00 | 0.000e+00 | 0/0 |
| 18 vs 17 | 128/128 | 0/31784960 | 0/128 | 0.000e+00 | 0.000e+00 | 0/0 |

- mode 16 reproduced mode 10 exactly here: all 31784960 logit values are finite and have identical bit patterns.
- mode 18 reproduced mode 17 exactly here: all 31784960 logit values are finite and have identical bit patterns.

These verdicts are about the shape named above, not about every pass that produced it; see the scope note at the top of this report.

## Reading this

Every number above comes from the full model through Engine::forward_batch. Prefill and decode are separate workloads; decode tokens are real generated tokens at a fixed step count, which means a stream that emitted EOS kept going.
Sample size is small by design. The quality table compares a handful of teacher-forced rows, so it can show a mode is wrong; it cannot show a mode is good enough to deploy. Raise --rounds, --gen-steps and --quality-rows before drawing conclusions about a close result.
The pair tables answer a narrower question than the mode 0 tables: whether a mode reproduces the arithmetic it was written to reproduce. A mode that matches its pair exactly can still differ from the deployed path, and a faster approximate mode is still an approximation; the mode 0 tables above are where that cost stays visible.

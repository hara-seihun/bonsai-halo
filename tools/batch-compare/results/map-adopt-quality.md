# Batch mode comparison: map-adopt-quality-09a1848

Source: `../../data/bonsai2/batch-comparison/map-adopt-quality-09a1848/run.json`, run 2026-09-20T20:44:13 to 2026-09-20T20:44:44, resident server paused: True, full model layers: 64, batch capacity 128.
Model ../../data/bonsai2/PTQ1_0.gguf, context 512, slots 1, max rows per pass 128, document PLAN.md, prompts tools/batch-compare/prompts.txt.
0 rounds, mode order reshuffled per round (seed 1234), clock ramp skipped, no timed workload, single process, state reset before every timed region.
Device memory: 7.17 GB for the model, 23.77 GB after `prepare_batch(128, 0x1ce)`, of which 16.59 GB is the batched FFN module. Weight images for the modes measured budget at 16.58 GB, against 16.58 GB to hold every mode at once. Each image is a further representation of all 64 layers, held alongside the deployed weights for as long as the mode that reads it is available.
Resident weight images: SCALES 0.27 GB, SLICE2 4.28 GB, LANE2 4.28 GB, PAIR 4.28 GB, DENSE5 3.48 GB.
Modes measured: 1 integer-a8 (`--ffn a8`), 2 integer-a4 (`--ffn a4`), 3 scaled-a8 (`--ffn scaled-a8`), 6 optimized-iu8-a8 (`--ffn map-a8`), 7 optimized-scaled-a8 (`--ffn map-scaled-a8`), 8 optimized-a4 (`--ffn map-a4`).
Matched reference pairs from run.json --reference-pairs: `6:1`, `7:3`, `8:2`.
A pair's scope selects which reported shape the pair is applied to. It does not certify that every pass inside that measurement took the matched route. Teacher-forced context, prompt prefill and the tail pass of a document can carry fewer than 32 rows, and there an automatic mode runs the deployed or sliced path however wide the reported rows are. `9:5` needs no scope for a different reason: both modes switch on the same row counts, so they agree pass for pass including the narrow ones. Read every other pair as a statement about the shape it names.

No timed rounds in this run.

### quality, prefill shape, 32 rows

Mode 0 did not run in this group, so there is no deployed-path comparison here.

Against the matched reference for each mode, same tokens, same rows:

| pair | top1 agreement | logit elements differing | rows differing | RMS logit diff | max abs diff | non-finite mode/ref |
|---|---:|---:|---:|---:|---:|---:|
| 6 vs 1 | 32/32 | 0/7946240 | 0/32 | 0.000e+00 | 0.000e+00 | 0/0 |
| 7 vs 3 | 32/32 | 0/7946240 | 0/32 | 0.000e+00 | 0.000e+00 | 0/0 |
| 8 vs 2 | 32/32 | 0/7946240 | 0/32 | 0.000e+00 | 0.000e+00 | 0/0 |

- mode 6 reproduced mode 1 exactly here: all 7946240 logit values are finite and have identical bit patterns.
- mode 7 reproduced mode 3 exactly here: all 7946240 logit values are finite and have identical bit patterns.
- mode 8 reproduced mode 2 exactly here: all 7946240 logit values are finite and have identical bit patterns.

These verdicts are about the shape named above, not about every pass that produced it; see the scope note at the top of this report.

### quality, prefill shape, 40 rows

Mode 0 did not run in this group, so there is no deployed-path comparison here.

Against the matched reference for each mode, same tokens, same rows:

| pair | top1 agreement | logit elements differing | rows differing | RMS logit diff | max abs diff | non-finite mode/ref |
|---|---:|---:|---:|---:|---:|---:|
| 6 vs 1 | 40/40 | 0/9932800 | 0/40 | 0.000e+00 | 0.000e+00 | 0/0 |
| 7 vs 3 | 40/40 | 0/9932800 | 0/40 | 0.000e+00 | 0.000e+00 | 0/0 |
| 8 vs 2 | 40/40 | 0/9932800 | 0/40 | 0.000e+00 | 0.000e+00 | 0/0 |

- mode 6 reproduced mode 1 exactly here: all 9932800 logit values are finite and have identical bit patterns.
- mode 7 reproduced mode 3 exactly here: all 9932800 logit values are finite and have identical bit patterns.
- mode 8 reproduced mode 2 exactly here: all 9932800 logit values are finite and have identical bit patterns.

These verdicts are about the shape named above, not about every pass that produced it; see the scope note at the top of this report.

### quality, prefill shape, 88 rows

Mode 0 did not run in this group, so there is no deployed-path comparison here.

Against the matched reference for each mode, same tokens, same rows:

| pair | top1 agreement | logit elements differing | rows differing | RMS logit diff | max abs diff | non-finite mode/ref |
|---|---:|---:|---:|---:|---:|---:|
| 6 vs 1 | 88/88 | 0/21852160 | 0/88 | 0.000e+00 | 0.000e+00 | 0/0 |
| 7 vs 3 | 88/88 | 0/21852160 | 0/88 | 0.000e+00 | 0.000e+00 | 0/0 |
| 8 vs 2 | 88/88 | 0/21852160 | 0/88 | 0.000e+00 | 0.000e+00 | 0/0 |

- mode 6 reproduced mode 1 exactly here: all 21852160 logit values are finite and have identical bit patterns.
- mode 7 reproduced mode 3 exactly here: all 21852160 logit values are finite and have identical bit patterns.
- mode 8 reproduced mode 2 exactly here: all 21852160 logit values are finite and have identical bit patterns.

These verdicts are about the shape named above, not about every pass that produced it; see the scope note at the top of this report.

### quality, prefill shape, 128 rows

Mode 0 did not run in this group, so there is no deployed-path comparison here.

Against the matched reference for each mode, same tokens, same rows:

| pair | top1 agreement | logit elements differing | rows differing | RMS logit diff | max abs diff | non-finite mode/ref |
|---|---:|---:|---:|---:|---:|---:|
| 6 vs 1 | 128/128 | 0/31784960 | 0/128 | 0.000e+00 | 0.000e+00 | 0/0 |
| 7 vs 3 | 128/128 | 0/31784960 | 0/128 | 0.000e+00 | 0.000e+00 | 0/0 |
| 8 vs 2 | 128/128 | 0/31784960 | 0/128 | 0.000e+00 | 0.000e+00 | 0/0 |

- mode 6 reproduced mode 1 exactly here: all 31784960 logit values are finite and have identical bit patterns.
- mode 7 reproduced mode 3 exactly here: all 31784960 logit values are finite and have identical bit patterns.
- mode 8 reproduced mode 2 exactly here: all 31784960 logit values are finite and have identical bit patterns.

These verdicts are about the shape named above, not about every pass that produced it; see the scope note at the top of this report.

## Reading this

Every number above comes from the full model through Engine::forward_batch. Prefill and decode are separate workloads; decode tokens are real generated tokens at a fixed step count, which means a stream that emitted EOS kept going.
Sample size is small by design. The quality table compares a handful of teacher-forced rows, so it can show a mode is wrong; it cannot show a mode is good enough to deploy. Raise --rounds, --gen-steps and --quality-rows before drawing conclusions about a close result.
The pair tables answer a narrower question than the mode 0 tables: whether a mode reproduces the arithmetic it was written to reproduce. A mode that matches its pair exactly can still differ from the deployed path, and a faster approximate mode is still an approximation; the mode 0 tables above are where that cost stays visible.


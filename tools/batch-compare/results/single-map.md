# Batch mode comparison: single-map-be08457

Source: `../../data/bonsai2/batch-comparison/single-map-be08457/run.json`, run 2026-09-20T22:00:08 to 2026-09-20T22:01:22, resident server paused: True, full model layers: 64, batch capacity 128.
Model ../../data/bonsai2/PTQ1_0.gguf, context 512, slots 1, max rows per pass 8, document PLAN.md, prompts tools/batch-compare/prompts.txt.
5 rounds, mode order reshuffled per round (seed 1234), warmup 10.1 s, single process, state reset before every timed region.
Device memory: 7.17 GB for the model, 7.19 GB after `prepare_batch(128, 0x40000000)`, of which 0.01 GB is the batched FFN module. Weight images for the modes measured budget at 0.00 GB, against 16.58 GB to hold every mode at once. Each image is a further representation of all 64 layers, held alongside the deployed weights for as long as the mode that reads it is available.
Modes measured: 0 deployed (`--ffn deployed`), 15 single-grid (`--ffn single-grid`), 12 single-map (`--ffn single-map`), 13 single-state (`--ffn single-state`), 14 single-retile (`--ffn single-retile`).
Matched reference pairs from run.json --reference-pairs: `12:0`, `13:0`, `15:0`.
A pair's scope selects which reported shape the pair is applied to. It does not certify that every pass inside that measurement took the matched route. Teacher-forced context, prompt prefill and the tail pass of a document can carry fewer than 32 rows, and there an automatic mode runs the deployed or sliced path however wide the reported rows are. `9:5` needs no scope for a different reason: both modes switch on the same row counts, so they agree pass for pass including the narrow ones. Read every other pair as a statement about the shape it names.

### decode: 1 streams x 64 steps

Fixed 64 greedy steps per stream, EOS emitted and fed back (streams are never stopped early), setup and prompt prefill excluded from the timing.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 34.0 | 34.0, 33.8, 34.0, 34.0, 33.9 | 0.6% | 1.000x | 1.882, 1.893, 1.881, 1.881, 1.886 | 2897 | 105 |
| 12 single-map | 33.4 | 33.4, 33.5, 33.4, 33.4, 33.3 | 0.4% | 0.983x | 1.916, 1.913, 1.918, 1.914, 1.920 | 2897 | 103 |
| 13 single-state | 33.1 | 33.1, 33.1, 33.2, 33.1, 33.1 | 0.4% | 0.973x | 1.936, 1.931, 1.929, 1.935, 1.934 | 2895 | 105 |
| 14 single-retile | 34.5 | 34.6, 34.5, 34.5, 34.5, 34.6 | 0.2% | 1.016x | 1.852, 1.855, 1.853, 1.855, 1.851 | 2898 | 104 |
| 15 single-grid | 33.6 | 33.6, 33.6, 33.6, 33.6, 33.6 | 0.1% | 0.987x | 1.907, 1.906, 1.906, 1.905, 1.905 | 2897 | 104 |

Against the matched reference for each mode, at this shape:

| pair | mode tok/s | reference tok/s | mode / reference |
|---|---:|---:|---:|
| 12 single-map vs 0 deployed | 33.4 | 34.0 | 0.983x |
| 13 single-state vs 0 deployed | 33.1 | 34.0 | 0.973x |
| 15 single-grid vs 0 deployed | 33.6 | 34.0 | 0.987x |


5 rounds per mode; worst spread 0.6%.

Host sysfs during these regions: gfx clock mean 2890-2900 MHz across modes, edge temperature max 56 C, board power mean 101-106 W.

### multistep state check

1 sequences carried 32 greedy steps from their own prompts, identical inputs per mode. Slot 0 length after the run: 67.

Step 0's token is the argmax of the prompt prefill, so a divergence at step 0 is a prefill-pass difference, not a decode-state one. Mode 4 retains the original FFN computation, so differences there call for inspection of state, replay, parity and floating evaluation order before blaming a new FFN map. The approximate modes carry a numeric difference into every step, so their divergence at any step, early or late, can be numerics alone.

| mode | streams matching mode 0 | tokens matching | first divergent step | first divergent stream |
|---|---:|---:|---:|---:|
| 0 deployed | reference | reference | - | - |
| 12 single-map | 1/1 | 32/32 | - | - |
| 13 single-state | 1/1 | 32/32 | - | - |
| 14 single-retile | 1/1 | 32/32 | - | - |
| 15 single-grid | 1/1 | 32/32 | - | - |

Greedy decode amplifies a small logit difference into a different token, so a divergence here is not by itself a quality verdict; read it with the logit tables below.

Against the matched reference for each mode, same 1 prompts and 32 steps. Each step is a 1-row pass, which is what decides how an automatic mode dispatches.

| pair | streams matching | tokens matching | first divergent step |
|---|---:|---:|---:|
| 12 vs 0 | 1/1 | 32/32 | - |
| 13 vs 0 | 1/1 | 32/32 | - |
| 15 vs 0 | 1/1 | 32/32 | - |

### quality, decode shape, 1 rows

1 teacher-forced rows, full 248320-entry vocabulary, identical token inputs per mode.
Rows are streams 0..n-1 of the prompts file, so a smaller row count is a prefix of a larger one: the row counts are nested, not independent samples.

| mode | top1 agreement | rows changed | mean KL(0->m) | max KL | max centred logit diff | mean top5 overlap | worst row rank of mode0 top1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 deployed | reference | 0 | - | - | - | - | - |
| 12 single-map | 1/1 | 0 | 0.0000 | 0.0000 | 0.000 | 5.0/5 | 0 |
| 13 single-state | 1/1 | 0 | 0.0000 | 0.0000 | 0.000 | 5.0/5 | 0 |
| 14 single-retile | 1/1 | 0 | 0.0000 | 0.0000 | 0.097 | 5.0/5 | 0 |
| 15 single-grid | 1/1 | 0 | 0.0000 | 0.0000 | 0.000 | 5.0/5 | 0 |

Against the matched reference for each mode, same tokens, same rows:

| pair | top1 agreement | logit elements differing | rows differing | RMS logit diff | max abs diff | non-finite mode/ref |
|---|---:|---:|---:|---:|---:|---:|
| 12 vs 0 | 1/1 | 0/248320 | 0/1 | 0.000e+00 | 0.000e+00 | 0/0 |
| 13 vs 0 | 1/1 | 0/248320 | 0/1 | 0.000e+00 | 0.000e+00 | 0/0 |
| 15 vs 0 | 1/1 | 0/248320 | 0/1 | 0.000e+00 | 0.000e+00 | 0/0 |

- mode 12 reproduced mode 0 exactly here: all 248320 logit values are finite and have identical bit patterns.
- mode 13 reproduced mode 0 exactly here: all 248320 logit values are finite and have identical bit patterns.
- mode 15 reproduced mode 0 exactly here: all 248320 logit values are finite and have identical bit patterns.

These verdicts are about the shape named above, not about every pass that produced it; see the scope note at the top of this report.

### quality, prefill shape, 1 rows

1 teacher-forced rows, full 248320-entry vocabulary, identical token inputs per mode.
Rows are document tokens following the same 64-token context, so a smaller row count is a prefix of a larger one: the row counts are nested, not independent samples.

| mode | top1 agreement | rows changed | mean KL(0->m) | max KL | max centred logit diff | mean top5 overlap | worst row rank of mode0 top1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 deployed | reference | 0 | - | - | - | - | - |
| 12 single-map | 1/1 | 0 | 0.0000 | 0.0000 | 0.000 | 5.0/5 | 0 |
| 13 single-state | 1/1 | 0 | 0.0000 | 0.0000 | 0.000 | 5.0/5 | 0 |
| 14 single-retile | 1/1 | 0 | 0.0003 | 0.0003 | 0.167 | 5.0/5 | 0 |
| 15 single-grid | 1/1 | 0 | 0.0000 | 0.0000 | 0.000 | 5.0/5 | 0 |

Against the matched reference for each mode, same tokens, same rows:

| pair | top1 agreement | logit elements differing | rows differing | RMS logit diff | max abs diff | non-finite mode/ref |
|---|---:|---:|---:|---:|---:|---:|
| 12 vs 0 | 1/1 | 0/248320 | 0/1 | 0.000e+00 | 0.000e+00 | 0/0 |
| 13 vs 0 | 1/1 | 0/248320 | 0/1 | 0.000e+00 | 0.000e+00 | 0/0 |
| 15 vs 0 | 1/1 | 0/248320 | 0/1 | 0.000e+00 | 0.000e+00 | 0/0 |

- mode 12 reproduced mode 0 exactly here: all 248320 logit values are finite and have identical bit patterns.
- mode 13 reproduced mode 0 exactly here: all 248320 logit values are finite and have identical bit patterns.
- mode 15 reproduced mode 0 exactly here: all 248320 logit values are finite and have identical bit patterns.

These verdicts are about the shape named above, not about every pass that produced it; see the scope note at the top of this report.

## Reading this

Every number above comes from the full model through Engine::forward_batch. Prefill and decode are separate workloads; decode tokens are real generated tokens at a fixed step count, which means a stream that emitted EOS kept going.
Sample size is small by design. The quality table compares a handful of teacher-forced rows, so it can show a mode is wrong; it cannot show a mode is good enough to deploy. Raise --rounds, --gen-steps and --quality-rows before drawing conclusions about a close result.
The pair tables answer a narrower question than the mode 0 tables: whether a mode reproduces the arithmetic it was written to reproduce. A mode that matches its pair exactly can still differ from the deployed path, and a faster approximate mode is still an approximation; the mode 0 tables above are where that cost stays visible.

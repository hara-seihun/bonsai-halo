# Batch mode comparison: map-a8-tps-006c2d5

Source: `../../data/bonsai2/batch-comparison/map-a8-tps-006c2d5/run.json`, run 2026-09-20T20:50:48 to 2026-09-20T20:54:03, resident server paused: True, full model layers: 64, batch capacity 128.
Model ../../data/bonsai2/PTQ1_0.gguf, context 512, slots 32, max rows per pass 128, document PLAN.md, prompts tools/batch-compare/prompts.txt.
3 rounds, mode order reshuffled per round (seed 1234), warmup 8.4 s, single process, state reset before every timed region.
Device memory: 15.43 GB for the model, 24.27 GB after `prepare_batch(128, 0xc2)`, of which 8.84 GB is the batched FFN module. Weight images for the modes measured budget at 8.82 GB, against 16.58 GB to hold every mode at once. Each image is a further representation of all 64 layers, held alongside the deployed weights for as long as the mode that reads it is available.
Resident weight images: SCALES 0.27 GB, SLICE2 4.28 GB, LANE2 4.28 GB.
Modes measured: 0 deployed (`--ffn deployed`), 5 auto-a8 (`--ffn auto`), 9 auto-optimized-a8 (`--ffn auto-map-a8`), 10 auto-optimized-scaled-a8 (`--ffn auto-map-scaled-a8`).
Matched reference pairs from run.json --reference-pairs: `9:5`.
A pair's scope selects which reported shape the pair is applied to. It does not certify that every pass inside that measurement took the matched route. Teacher-forced context, prompt prefill and the tail pass of a document can carry fewer than 32 rows, and there an automatic mode runs the deployed or sliced path however wide the reported rows are. `9:5` needs no scope for a different reason: both modes switch on the same row counts, so they agree pass for pass including the narrow ones. Read every other pair as a statement about the shape it names.

### decode: 32 streams x 8 steps

Fixed 8 greedy steps per stream, EOS emitted and fed back (streams are never stopped early), setup and prompt prefill excluded from the timing.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 133.1 | 133.1, 133.1, 133.3 | 0.1% | 1.000x | 1.924, 1.924, 1.921 | 2898 | 119 |
| 5 auto-a8 | 160.1 | 160.2, 160.1, 160.0 | 0.1% | 1.203x | 1.598, 1.599, 1.600 | 2899 | 116 |
| 9 auto-optimized-a8 | 165.8 | 166.0, 165.8, 165.8 | 0.1% | 1.246x | 1.542, 1.544, 1.544 | 2899 | 116 |
| 10 auto-optimized-scaled-a8 | 176.6 | 176.6, 176.6, 176.5 | 0.1% | 1.327x | 1.449, 1.450, 1.451 | 2894 | 118 |

Against the matched reference for each mode, at this shape:

| pair | mode tok/s | reference tok/s | mode / reference |
|---|---:|---:|---:|
| 9 auto-optimized-a8 vs 5 auto-a8 | 165.8 | 160.1 | 1.036x |


3 rounds per mode; worst spread 0.1%.

Host sysfs during these regions: gfx clock mean 2883-2900 MHz across modes, edge temperature max 69 C, board power mean 115-119 W.

### decode: 8 streams x 8 steps

Fixed 8 greedy steps per stream, EOS emitted and fed back (streams are never stopped early), setup and prompt prefill excluded from the timing.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 133.0 | 133.5, 133.0, 133.0 | 0.4% | 1.000x | 0.479, 0.481, 0.481 | 2839 | 117 |
| 5 auto-a8 | 142.2 | 142.2, 142.7, 142.2 | 0.4% | 1.069x | 0.450, 0.449, 0.450 | 2893 | 120 |
| 9 auto-optimized-a8 | 142.2 | 142.2, 142.5, 142.2 | 0.2% | 1.069x | 0.450, 0.449, 0.450 | 2896 | 120 |
| 10 auto-optimized-scaled-a8 | 142.3 | 142.3, 142.6, 142.3 | 0.2% | 1.070x | 0.450, 0.449, 0.450 | 2891 | 118 |

Against the matched reference for each mode, at this shape:

| pair | mode tok/s | reference tok/s | mode / reference |
|---|---:|---:|---:|
| 9 auto-optimized-a8 vs 5 auto-a8 | 142.2 | 142.2 | 1.000x |


3 rounds per mode; worst spread 0.4%.

Host sysfs during these regions: gfx clock mean 2724-2899 MHz across modes, edge temperature max 69 C, board power mean 112-122 W.

### prefill: 128 rows/pass, 256 tokens

Logits (lm_head) computed on: last-pass-only, which here is 128 logit rows of 256 tokens, not one token: the final pass carries 128 rows and the lm_head runs on all of them. Every mode pays that same cost, but it is a real part of these numbers and it differs between the 128-row and other row settings. Token count is the document's real tokens.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 155.3 | 155.3, 155.3, 155.3 | 0.0% | 1.000x | 1.649, 1.649, 1.649 | 2892 | 120 |
| 5 auto-a8 | 291.5 | 288.7, 291.9, 291.5 | 1.1% | 1.877x | 0.887, 0.877, 0.878 | 2887 | 122 |
| 9 auto-optimized-a8 | 287.3 | 287.9, 287.3, 287.1 | 0.3% | 1.851x | 0.889, 0.891, 0.892 | 2888 | 122 |
| 10 auto-optimized-scaled-a8 | 291.6 | 291.9, 291.6, 289.6 | 0.8% | 1.878x | 0.877, 0.878, 0.884 | 2883 | 122 |

Against the matched reference for each mode, at this shape:

| pair | mode tok/s | reference tok/s | mode / reference |
|---|---:|---:|---:|
| 9 auto-optimized-a8 vs 5 auto-a8 | 287.3 | 291.5 | 0.986x |


3 rounds per mode; worst spread 1.1%.

Host sysfs during these regions: gfx clock mean 2872-2897 MHz across modes, edge temperature max 71 C, board power mean 118-124 W.

### prefill: 32 rows/pass, 256 tokens

Logits (lm_head) computed on: last-pass-only, which here is 32 logit rows of 256 tokens, not one token: the final pass carries 32 rows and the lm_head runs on all of them. Every mode pays that same cost, but it is a real part of these numbers and it differs between the 32-row and other row settings. Token count is the document's real tokens.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 157.0 | 157.0, 157.0, 157.1 | 0.1% | 1.000x | 1.630, 1.631, 1.629 | 2896 | 117 |
| 5 auto-a8 | 216.9 | 216.5, 216.9, 217.2 | 0.3% | 1.382x | 1.182, 1.180, 1.179 | 2899 | 116 |
| 9 auto-optimized-a8 | 229.7 | 229.6, 229.7, 230.1 | 0.2% | 1.463x | 1.115, 1.114, 1.112 | 2893 | 119 |
| 10 auto-optimized-scaled-a8 | 247.1 | 246.3, 247.1, 247.1 | 0.3% | 1.574x | 1.039, 1.036, 1.036 | 2895 | 119 |

Against the matched reference for each mode, at this shape:

| pair | mode tok/s | reference tok/s | mode / reference |
|---|---:|---:|---:|
| 9 auto-optimized-a8 vs 5 auto-a8 | 229.7 | 216.9 | 1.059x |


3 rounds per mode; worst spread 0.3%.

Host sysfs during these regions: gfx clock mean 2887-2899 MHz across modes, edge temperature max 68 C, board power mean 113-120 W.

### multistep state check

32 sequences carried 4 greedy steps from their own prompts, identical inputs per mode. Slot 0 length after the run: 39.

Step 0's token is the argmax of the prompt prefill, so a divergence at step 0 is a prefill-pass difference, not a decode-state one. Mode 4 retains the original FFN computation, so differences there call for inspection of state, replay, parity and floating evaluation order before blaming a new FFN map. The approximate modes carry a numeric difference into every step, so their divergence at any step, early or late, can be numerics alone.

| mode | streams matching mode 0 | tokens matching | first divergent step | first divergent stream |
|---|---:|---:|---:|---:|
| 0 deployed | reference | reference | - | - |
| 5 auto-a8 | 31/32 | 125/128 | 1 | 19 |
| 9 auto-optimized-a8 | 31/32 | 125/128 | 1 | 19 |
| 10 auto-optimized-scaled-a8 | 32/32 | 128/128 | - | - |

- mode 5 stream 19 first differs at step 1: mode 0 364, mode 5 264
  - mode 0 text: 'Testing for **sub'
  - mode 5 text: 'Testing a random number'
- mode 9 stream 19 first differs at step 1: mode 0 364, mode 9 264
  - mode 0 text: 'Testing for **sub'
  - mode 9 text: 'Testing a random number'

Greedy decode amplifies a small logit difference into a different token, so a divergence here is not by itself a quality verdict; read it with the logit tables below.

Against the matched reference for each mode, same 32 prompts and 4 steps. Each step is a 32-row pass, which is what decides how an automatic mode dispatches.

| pair | streams matching | tokens matching | first divergent step |
|---|---:|---:|---:|
| 9 vs 5 | 32/32 | 128/128 | - |

### quality, decode shape, 32 rows

32 teacher-forced rows, full 248320-entry vocabulary, identical token inputs per mode.
Rows are streams 0..n-1 of the prompts file, so a smaller row count is a prefix of a larger one: the row counts are nested, not independent samples.

| mode | top1 agreement | rows changed | mean KL(0->m) | max KL | max centred logit diff | mean top5 overlap | worst row rank of mode0 top1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 deployed | reference | 0 | - | - | - | - | - |
| 5 auto-a8 | 32/32 | 0 | 0.0001 | 0.0004 | 0.126 | 5.0/5 | 0 |
| 9 auto-optimized-a8 | 32/32 | 0 | 0.0001 | 0.0004 | 0.126 | 5.0/5 | 0 |
| 10 auto-optimized-scaled-a8 | 32/32 | 0 | 0.0002 | 0.0004 | 0.141 | 5.0/5 | 0 |

Against the matched reference for each mode, same tokens, same rows:

| pair | top1 agreement | logit elements differing | rows differing | RMS logit diff | max abs diff | non-finite mode/ref |
|---|---:|---:|---:|---:|---:|---:|
| 9 vs 5 | 32/32 | 0/7946240 | 0/32 | 0.000e+00 | 0.000e+00 | 0/0 |

- mode 9 reproduced mode 5 exactly here: all 7946240 logit values are finite and have identical bit patterns.

These verdicts are about the shape named above, not about every pass that produced it; see the scope note at the top of this report.

### quality, prefill shape, 32 rows

32 teacher-forced rows, full 248320-entry vocabulary, identical token inputs per mode.
Rows are document tokens following the same 64-token context, so a smaller row count is a prefix of a larger one: the row counts are nested, not independent samples.

| mode | top1 agreement | rows changed | mean KL(0->m) | max KL | max centred logit diff | mean top5 overlap | worst row rank of mode0 top1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 deployed | reference | 0 | - | - | - | - | - |
| 5 auto-a8 | 31/32 | 1 | 0.0006 | 0.0027 | 0.339 | 4.9/5 | 1 |
| 9 auto-optimized-a8 | 31/32 | 1 | 0.0006 | 0.0027 | 0.339 | 4.9/5 | 1 |
| 10 auto-optimized-scaled-a8 | 32/32 | 0 | 0.0008 | 0.0051 | 0.329 | 4.9/5 | 0 |

- mode 5 row 12: mode0 picks 14 (p=0.106), mode picks 318; mode0's token falls to rank 1 with p=0.104
- mode 9 row 12: mode0 picks 14 (p=0.106), mode picks 318; mode0's token falls to rank 1 with p=0.104

Against the matched reference for each mode, same tokens, same rows:

| pair | top1 agreement | logit elements differing | rows differing | RMS logit diff | max abs diff | non-finite mode/ref |
|---|---:|---:|---:|---:|---:|---:|
| 9 vs 5 | 32/32 | 0/7946240 | 0/32 | 0.000e+00 | 0.000e+00 | 0/0 |

- mode 9 reproduced mode 5 exactly here: all 7946240 logit values are finite and have identical bit patterns.

These verdicts are about the shape named above, not about every pass that produced it; see the scope note at the top of this report.

## Reading this

Every number above comes from the full model through Engine::forward_batch. Prefill and decode are separate workloads; decode tokens are real generated tokens at a fixed step count, which means a stream that emitted EOS kept going.
Sample size is small by design. The quality table compares a handful of teacher-forced rows, so it can show a mode is wrong; it cannot show a mode is good enough to deploy. Raise --rounds, --gen-steps and --quality-rows before drawing conclusions about a close result.
The pair tables answer a narrower question than the mode 0 tables: whether a mode reproduces the arithmetic it was written to reproduce. A mode that matches its pair exactly can still differ from the deployed path, and a faster approximate mode is still an approximation; the mode 0 tables above are where that cost stays visible.


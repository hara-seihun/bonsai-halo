# Batch mode comparison: wide-sequence-tps

Source: `../../data/bonsai2/batch-comparison/wide-sequence-tps/run.json`, run 2026-09-20T23:13:46 to 2026-09-20T23:17:36, resident server paused: True, full model layers: 64, batch capacity 128.
Model ../../data/bonsai2/PTQ1_0.gguf, context 512, slots 32, max rows per pass 128, document PLAN.md, prompts tools/batch-compare/prompts.txt.
3 rounds, mode order reshuffled per round (seed 1234), warmup 10.5 s, single process, state reset before every timed region.
Device memory: 15.43 GB for the model, 26.19 GB after `prepare_batch(128, 0x80)`, of which 8.84 GB is the batched FFN module. Weight images for the modes measured budget at 8.82 GB, against 16.58 GB to hold every mode at once. Each image is a further representation of all 64 layers, held alongside the deployed weights for as long as the mode that reads it is available.
Resident weight images: SCALES 0.27 GB, SLICE2 4.28 GB, LANE2 4.28 GB.
Modes measured: 0 deployed (`--ffn deployed`), 10 auto-optimized-scaled-a8 (`--ffn auto-map-scaled-a8`), 16 commit-scaled (`--ffn commit-scaled`), 17 wide-sequence (`--ffn wide-sequence`), 18 wide-commit (`--ffn wide-commit`).
Sequence routes: mode 16 commits without widening; 17 widens without committing; 18 and 19 do both. Widening applies at 32 or more total rows; committing applies at every row count. FFN automatic dispatch is separate from this choice.
Wide-sequence operand: `int8`; token tile width: `auto`. `scaled-f16` rounds dequantised A8 inputs to FP16 and changes accumulation order; it is a different numerical map, not a layout-only option.
Matched reference pairs from run.json --reference-pairs: `16:10`, `18:17`.
A pair's scope selects which reported shape the pair is applied to. It does not certify that every pass inside that measurement took the matched route. Teacher-forced context, prompt prefill and the tail pass of a document can carry fewer than 32 rows, and there an automatic mode runs the deployed or sliced path however wide the reported rows are. `9:5` needs no scope for a different reason: both modes switch on the same row counts, so they agree pass for pass including the narrow ones. Read every other pair as a statement about the shape it names.

### decode: 32 streams x 8 steps

Fixed 8 greedy steps per stream, EOS emitted and fed back (streams are never stopped early), setup and prompt prefill excluded from the timing.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 133.2 | 133.1, 133.2, 133.2 | 0.1% | 1.000x | 1.923, 1.922, 1.923 | 2899 | 119 |
| 10 auto-optimized-scaled-a8 | 176.1 | 176.1, 176.1, 176.5 | 0.3% | 1.323x | 1.453, 1.454, 1.450 | 2899 | 117 |
| 16 commit-scaled | 182.5 | 182.5, 182.6, 182.4 | 0.1% | 1.370x | 1.403, 1.402, 1.404 | 2894 | 117 |
| 17 wide-sequence | 198.9 | 198.8, 198.9, 199.0 | 0.1% | 1.494x | 1.288, 1.287, 1.286 | 2894 | 117 |
| 18 wide-commit | 200.6 | 200.6, 200.6, 200.6 | 0.0% | 1.506x | 1.276, 1.276, 1.276 | 2899 | 118 |

Against the matched reference for each mode, at this shape:

| pair | mode tok/s | reference tok/s | mode / reference |
|---|---:|---:|---:|
| 16 commit-scaled vs 10 auto-optimized-scaled-a8 | 182.5 | 176.1 | 1.036x |
| 18 wide-commit vs 17 wide-sequence | 200.6 | 198.9 | 1.008x |


3 rounds per mode; worst spread 0.3%.

Host sysfs during these regions: gfx clock mean 2882-2900 MHz across modes, edge temperature max 70 C, board power mean 116-120 W.

### decode: 8 streams x 8 steps

Fixed 8 greedy steps per stream, EOS emitted and fed back (streams are never stopped early), setup and prompt prefill excluded from the timing.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 132.9 | 133.5, 132.9, 132.8 | 0.6% | 1.000x | 0.479, 0.482, 0.482 | 2896 | 120 |
| 10 auto-optimized-scaled-a8 | 142.0 | 142.5, 142.0, 141.9 | 0.4% | 1.069x | 0.449, 0.451, 0.451 | 2885 | 121 |
| 16 commit-scaled | 147.2 | 147.2, 148.2, 147.2 | 0.7% | 1.108x | 0.435, 0.432, 0.435 | 2874 | 119 |
| 17 wide-sequence | 142.1 | 142.2, 142.1, 142.0 | 0.1% | 1.070x | 0.450, 0.450, 0.451 | 2876 | 118 |
| 18 wide-commit | 147.4 | 148.0, 147.4, 147.3 | 0.4% | 1.109x | 0.433, 0.434, 0.434 | 2883 | 118 |

Against the matched reference for each mode, at this shape:

| pair | mode tok/s | reference tok/s | mode / reference |
|---|---:|---:|---:|
| 16 commit-scaled vs 10 auto-optimized-scaled-a8 | 147.2 | 142.0 | 1.036x |
| 18 wide-commit vs 17 wide-sequence | 147.4 | 142.1 | 1.037x |


3 rounds per mode; worst spread 0.7%.

Host sysfs during these regions: gfx clock mean 2851-2899 MHz across modes, edge temperature max 70 C, board power mean 117-127 W.

### prefill: 128 rows/pass, 256 tokens

Logits (lm_head) computed on: last-pass-only, which here is 128 logit rows of 256 tokens, not one token: the final pass carries 128 rows and the lm_head runs on all of them. Every mode pays that same cost, but it is a real part of these numbers and it differs between the 128-row and other row settings. Token count is the document's real tokens.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 155.1 | 155.4, 155.1, 147.0 | 5.5% | 1.000x | 1.647, 1.650, 1.742 | 2886 | 119 |
| 10 auto-optimized-scaled-a8 | 291.1 | 289.0, 291.1, 291.1 | 0.7% | 1.877x | 0.886, 0.880, 0.879 | 2878 | 124 |
| 16 commit-scaled | 308.5 | 310.9, 223.1, 308.5 | 28.5% | 1.989x | 0.823, 1.147, 0.830 | 2826 | 119 |
| 17 wide-sequence | 417.0 | 422.4, 414.4, 417.0 | 1.9% | 2.689x | 0.606, 0.618, 0.614 | 2877 | 122 |
| 18 wide-commit | 440.2 | 442.6, 440.2, 435.4 | 1.6% | 2.838x | 0.578, 0.582, 0.588 | 2888 | 122 |

Against the matched reference for each mode, at this shape:

| pair | mode tok/s | reference tok/s | mode / reference |
|---|---:|---:|---:|
| 16 commit-scaled vs 10 auto-optimized-scaled-a8 | 308.5 | 291.1 | 1.060x |
| 18 wide-commit vs 17 wide-sequence | 440.2 | 417.0 | 1.056x |


Run-to-run spread reaches 28.5% over 3 rounds. Differences smaller than that are not resolved by this sample.

Host sysfs during these regions: gfx clock mean 2709-2897 MHz across modes, edge temperature max 73 C, board power mean 113-128 W.

### prefill: 32 rows/pass, 256 tokens

Logits (lm_head) computed on: last-pass-only, which here is 32 logit rows of 256 tokens, not one token: the final pass carries 32 rows and the lm_head runs on all of them. Every mode pays that same cost, but it is a real part of these numbers and it differs between the 32-row and other row settings. Token count is the document's real tokens.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 156.8 | 156.6, 156.8, 156.9 | 0.2% | 1.000x | 1.634, 1.633, 1.632 | 2894 | 119 |
| 10 auto-optimized-scaled-a8 | 247.7 | 246.9, 247.7, 248.0 | 0.4% | 1.580x | 1.037, 1.033, 1.032 | 2885 | 119 |
| 16 commit-scaled | 261.0 | 261.4, 261.0, 260.5 | 0.3% | 1.665x | 0.979, 0.981, 0.983 | 2891 | 120 |
| 17 wide-sequence | 305.1 | 303.6, 305.1, 307.0 | 1.1% | 1.946x | 0.843, 0.839, 0.834 | 2889 | 119 |
| 18 wide-commit | 317.0 | 317.0, 317.6, 316.5 | 0.4% | 2.022x | 0.807, 0.806, 0.809 | 2898 | 119 |

Against the matched reference for each mode, at this shape:

| pair | mode tok/s | reference tok/s | mode / reference |
|---|---:|---:|---:|
| 16 commit-scaled vs 10 auto-optimized-scaled-a8 | 261.0 | 247.7 | 1.054x |
| 18 wide-commit vs 17 wide-sequence | 317.0 | 305.1 | 1.039x |


3 rounds per mode; worst spread 1.1%.

Host sysfs during these regions: gfx clock mean 2877-2899 MHz across modes, edge temperature max 68 C, board power mean 118-121 W.

### stalled samples

1 timed sample(s) ran materially longer than the other rounds of the same mode and shape. They are kept in every table above and in the raw JSON. A stall is a real thing that happened to this machine, and dropping it would make the benchmark look steadier than the machine is. The median is the headline statistic precisely so one stall cannot move it, and the spread figure is where it shows.

| round | mode | shape | wall s | median s | ratio | worst pass/step s | GPU busy % | host CPU busy |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 16 commit-scaled | prefill, 128 rows/pass, 256 tokens | 1.147 | 0.830 | 1.38x | 1.144 | 88 | 2.17 |

- round 1 mode 16 commit-scaled, prefill 128 rows/pass, 256 tokens: 1.147 s against a 0.830 s median, 0.317 s excess. pass 1 of 2 took 1.144 s, 100% of the region
  - host during the region: CPU busy 2.17 thread-seconds per wall second, io full PSI 0.00 s, io some PSI 0.00 s, cpu some PSI 0.01 s, 19 involuntary context switches, MemAvailable 26.5 to 26.5 GB
  - none of the host counters account for the excess: not faulting, not preempted, not stalled on memory or I/O

### multistep state check

32 sequences carried 4 greedy steps from their own prompts, identical inputs per mode. Slot 0 length after the run: 39.

Step 0's token is the argmax of the prompt prefill, so a divergence at step 0 is a prefill-pass difference, not a decode-state one. Mode 4 retains the original FFN computation, so differences there call for inspection of state, replay, parity and floating evaluation order before blaming a new FFN map. The approximate modes carry a numeric difference into every step, so their divergence at any step, early or late, can be numerics alone.

| mode | streams matching mode 0 | tokens matching | first divergent step | first divergent stream |
|---|---:|---:|---:|---:|
| 0 deployed | reference | reference | - | - |
| 10 auto-optimized-scaled-a8 | 32/32 | 128/128 | - | - |
| 16 commit-scaled | 32/32 | 128/128 | - | - |
| 17 wide-sequence | 31/32 | 127/128 | 3 | 17 |
| 18 wide-commit | 31/32 | 127/128 | 3 | 17 |

- mode 17 stream 17 first differs at step 3: mode 0 41080, mode 17 4203
  - mode 0 text: 'Speculative decoding fundamentally'
  - mode 17 text: 'Speculative decoding changes'
- mode 18 stream 17 first differs at step 3: mode 0 41080, mode 18 4203
  - mode 0 text: 'Speculative decoding fundamentally'
  - mode 18 text: 'Speculative decoding changes'

Greedy decode amplifies a small logit difference into a different token, so a divergence here is not by itself a quality verdict; read it with the logit tables below.

Against the matched reference for each mode, same 32 prompts and 4 steps. Each step is a 32-row pass, which is what decides how an automatic mode dispatches.

| pair | streams matching | tokens matching | first divergent step |
|---|---:|---:|---:|
| 16 vs 10 | 32/32 | 128/128 | - |
| 18 vs 17 | 32/32 | 128/128 | - |

### quality, decode shape, 32 rows

32 teacher-forced rows, full 248320-entry vocabulary, identical token inputs per mode.
Rows are streams 0..n-1 of the prompts file, so a smaller row count is a prefix of a larger one: the row counts are nested, not independent samples.

| mode | top1 agreement | rows changed | mean KL(0->m) | max KL | max centred logit diff | mean top5 overlap | worst row rank of mode0 top1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 deployed | reference | 0 | - | - | - | - | - |
| 10 auto-optimized-scaled-a8 | 32/32 | 0 | 0.0002 | 0.0004 | 0.141 | 5.0/5 | 0 |
| 16 commit-scaled | 32/32 | 0 | 0.0002 | 0.0004 | 0.141 | 5.0/5 | 0 |
| 17 wide-sequence | 32/32 | 0 | 0.0001 | 0.0007 | 0.141 | 5.0/5 | 0 |
| 18 wide-commit | 32/32 | 0 | 0.0001 | 0.0007 | 0.141 | 5.0/5 | 0 |

Against the matched reference for each mode, same tokens, same rows:

| pair | top1 agreement | logit elements differing | rows differing | RMS logit diff | max abs diff | non-finite mode/ref |
|---|---:|---:|---:|---:|---:|---:|
| 16 vs 10 | 32/32 | 0/7946240 | 0/32 | 0.000e+00 | 0.000e+00 | 0/0 |
| 18 vs 17 | 32/32 | 0/7946240 | 0/32 | 0.000e+00 | 0.000e+00 | 0/0 |

- mode 16 reproduced mode 10 exactly here: all 7946240 logit values are finite and have identical bit patterns.
- mode 18 reproduced mode 17 exactly here: all 7946240 logit values are finite and have identical bit patterns.

These verdicts are about the shape named above, not about every pass that produced it; see the scope note at the top of this report.

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

## Reading this

Every number above comes from the full model through Engine::forward_batch. Prefill and decode are separate workloads; decode tokens are real generated tokens at a fixed step count, which means a stream that emitted EOS kept going.
Sample size is small by design. The quality table compares a handful of teacher-forced rows, so it can show a mode is wrong; it cannot show a mode is good enough to deploy. Raise --rounds, --gen-steps and --quality-rows before drawing conclusions about a close result.
The pair tables answer a narrower question than the mode 0 tables: whether a mode reproduces the arithmetic it was written to reproduce. A mode that matches its pair exactly can still differ from the deployed path, and a faster approximate mode is still an approximation; the mode 0 tables above are where that cost stays visible.

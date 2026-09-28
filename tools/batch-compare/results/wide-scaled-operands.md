# Batch mode comparison: wide-scaled-operands

Source: `../../data/bonsai2/batch-comparison/wide-scaled-operands/run.json`, run 2026-09-20T23:26:46 to 2026-09-20T23:29:11, resident server paused: True, full model layers: 64, batch capacity 128.
Model ../../data/bonsai2/PTQ1_0.gguf, context 512, slots 32, max rows per pass 128, document PLAN.md, prompts tools/batch-compare/prompts.txt.
3 rounds, mode order reshuffled per round (seed 1234), warmup 6.3 s, single process, state reset before every timed region.
Device memory: 15.43 GB for the model, 26.19 GB after `prepare_batch(128, 0x80)`, of which 8.84 GB is the batched FFN module. Weight images for the modes measured budget at 8.82 GB, against 16.58 GB to hold every mode at once. Each image is a further representation of all 64 layers, held alongside the deployed weights for as long as the mode that reads it is available.
Resident weight images: SCALES 0.27 GB, SLICE2 4.28 GB, LANE2 4.28 GB.
Modes measured: 0 deployed (`--ffn deployed`), 10 auto-optimized-scaled-a8 (`--ffn auto-map-scaled-a8`), 18 wide-commit (`--ffn wide-commit`).
Sequence routes: mode 16 commits without widening; 17 widens without committing; 18 and 19 do both. Widening applies at 32 or more total rows; committing applies at every row count. FFN automatic dispatch is separate from this choice.
Wide-sequence operand: `scaled-f16`; token tile width: `auto`. `scaled-f16` rounds dequantised A8 inputs to FP16 and changes accumulation order; it is a different numerical map, not a layout-only option.

### decode: 32 streams x 8 steps

Fixed 8 greedy steps per stream, EOS emitted and fed back (streams are never stopped early), setup and prompt prefill excluded from the timing.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 133.2 | 133.2, 133.1, 133.2 | 0.0% | 1.000x | 1.923, 1.923, 1.922 | 2898 | 119 |
| 10 auto-optimized-scaled-a8 | 176.6 | 177.3, 176.5, 176.6 | 0.4% | 1.327x | 1.444, 1.450, 1.449 | 2884 | 119 |
| 18 wide-commit | 184.0 | 184.4, 184.0, 183.9 | 0.3% | 1.382x | 1.388, 1.391, 1.392 | 2898 | 116 |

3 rounds per mode; worst spread 0.4%.

Host sysfs during these regions: gfx clock mean 2856-2900 MHz across modes, edge temperature max 69 C, board power mean 115-120 W.

### decode: 8 streams x 8 steps

Fixed 8 greedy steps per stream, EOS emitted and fed back (streams are never stopped early), setup and prompt prefill excluded from the timing.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 132.9 | 132.9, 132.9, 133.5 | 0.5% | 1.000x | 0.482, 0.481, 0.479 | 2894 | 120 |
| 10 auto-optimized-scaled-a8 | 142.1 | 141.2, 142.1, 142.4 | 0.9% | 1.069x | 0.453, 0.450, 0.449 | 2803 | 125 |
| 18 wide-commit | 148.1 | 148.3, 147.3, 148.1 | 0.6% | 1.114x | 0.432, 0.434, 0.432 | 2872 | 118 |

3 rounds per mode; worst spread 0.9%.

Host sysfs during these regions: gfx clock mean 2642-2898 MHz across modes, edge temperature max 69 C, board power mean 118-138 W.

### prefill: 128 rows/pass, 256 tokens

Logits (lm_head) computed on: last-pass-only, which here is 128 logit rows of 256 tokens, not one token: the final pass carries 128 rows and the lm_head runs on all of them. Every mode pays that same cost, but it is a real part of these numbers and it differs between the 128-row and other row settings. Token count is the document's real tokens.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 155.1 | 155.4, 154.6, 155.1 | 0.5% | 1.000x | 1.647, 1.655, 1.650 | 2894 | 119 |
| 10 auto-optimized-scaled-a8 | 288.4 | 247.8, 289.3, 288.4 | 14.4% | 1.859x | 1.033, 0.885, 0.888 | 2810 | 125 |
| 18 wide-commit | 432.8 | 435.8, 432.8, 432.2 | 0.8% | 2.790x | 0.587, 0.591, 0.592 | 2874 | 122 |

Run-to-run spread reaches 14.4% over 3 rounds. Differences smaller than that are not resolved by this sample.

Host sysfs during these regions: gfx clock mean 2661-2895 MHz across modes, edge temperature max 71 C, board power mean 119-129 W.

### prefill: 32 rows/pass, 256 tokens

Logits (lm_head) computed on: last-pass-only, which here is 32 logit rows of 256 tokens, not one token: the final pass carries 32 rows and the lm_head runs on all of them. Every mode pays that same cost, but it is a real part of these numbers and it differs between the 32-row and other row settings. Token count is the document's real tokens.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 156.7 | 156.7, 156.4, 157.9 | 0.9% | 1.000x | 1.633, 1.636, 1.621 | 2889 | 119 |
| 10 auto-optimized-scaled-a8 | 246.8 | 246.8, 246.6, 246.8 | 0.1% | 1.574x | 1.037, 1.038, 1.037 | 2895 | 120 |
| 18 wide-commit | 306.2 | 308.4, 306.2, 305.7 | 0.9% | 1.954x | 0.830, 0.836, 0.837 | 2891 | 119 |

3 rounds per mode; worst spread 0.9%.

Host sysfs during these regions: gfx clock mean 2872-2899 MHz across modes, edge temperature max 68 C, board power mean 117-122 W.

### multistep state check

32 sequences carried 4 greedy steps from their own prompts, identical inputs per mode. Slot 0 length after the run: 39.

Step 0's token is the argmax of the prompt prefill, so a divergence at step 0 is a prefill-pass difference, not a decode-state one. Mode 4 retains the original FFN computation, so differences there call for inspection of state, replay, parity and floating evaluation order before blaming a new FFN map. The approximate modes carry a numeric difference into every step, so their divergence at any step, early or late, can be numerics alone.

| mode | streams matching mode 0 | tokens matching | first divergent step | first divergent stream |
|---|---:|---:|---:|---:|
| 0 deployed | reference | reference | - | - |
| 10 auto-optimized-scaled-a8 | 32/32 | 128/128 | - | - |
| 18 wide-commit | 31/32 | 127/128 | 3 | 17 |

- mode 18 stream 17 first differs at step 3: mode 0 41080, mode 18 4203
  - mode 0 text: 'Speculative decoding fundamentally'
  - mode 18 text: 'Speculative decoding changes'

Greedy decode amplifies a small logit difference into a different token, so a divergence here is not by itself a quality verdict; read it with the logit tables below.

### quality, decode shape, 32 rows

32 teacher-forced rows, full 248320-entry vocabulary, identical token inputs per mode.
Rows are streams 0..n-1 of the prompts file, so a smaller row count is a prefix of a larger one: the row counts are nested, not independent samples.

| mode | top1 agreement | rows changed | mean KL(0->m) | max KL | max centred logit diff | mean top5 overlap | worst row rank of mode0 top1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 deployed | reference | 0 | - | - | - | - | - |
| 10 auto-optimized-scaled-a8 | 32/32 | 0 | 0.0002 | 0.0004 | 0.141 | 5.0/5 | 0 |
| 18 wide-commit | 32/32 | 0 | 0.0002 | 0.0016 | 0.176 | 5.0/5 | 0 |

### quality, prefill shape, 32 rows

32 teacher-forced rows, full 248320-entry vocabulary, identical token inputs per mode.
Rows are document tokens following the same 64-token context, so a smaller row count is a prefix of a larger one: the row counts are nested, not independent samples.

| mode | top1 agreement | rows changed | mean KL(0->m) | max KL | max centred logit diff | mean top5 overlap | worst row rank of mode0 top1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 deployed | reference | 0 | - | - | - | - | - |
| 10 auto-optimized-scaled-a8 | 32/32 | 0 | 0.0008 | 0.0051 | 0.329 | 4.9/5 | 0 |
| 18 wide-commit | 32/32 | 0 | 0.0007 | 0.0035 | 0.318 | 5.0/5 | 0 |

## Reading this

Every number above comes from the full model through Engine::forward_batch. Prefill and decode are separate workloads; decode tokens are real generated tokens at a fixed step count, which means a stream that emitted EOS kept going.
Sample size is small by design. The quality table compares a handful of teacher-forced rows, so it can show a mode is wrong; it cannot show a mode is good enough to deploy. Raise --rounds, --gen-steps and --quality-rows before drawing conclusions about a close result.

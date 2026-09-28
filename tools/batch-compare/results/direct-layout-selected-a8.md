# Batch mode comparison: direct-layout-selected-a8

Source: `../../data/bonsai2/batch-comparison/direct-layout-selected-a8/run.json`, run 2026-09-21T00:00:31 to 2026-09-21T00:01:58, resident server paused: True, full model layers: 64, batch capacity 128.
Model ../../data/bonsai2/PTQ1_0.gguf, context 512, slots 32, max rows per pass 128, document PLAN.md, prompts tools/batch-compare/prompts.txt.
3 rounds, mode order reshuffled per round (seed 1234), warmup 4.2 s, single process, state reset before every timed region.
Device memory: 15.43 GB for the model, 26.19 GB after `prepare_batch(128, 0x80)`, of which 8.84 GB is the batched FFN module. Weight images for the modes measured budget at 8.82 GB, against 16.58 GB to hold every mode at once. Each image is a further representation of all 64 layers, held alongside the deployed weights for as long as the mode that reads it is available.
Resident weight images: SCALES 0.27 GB, SLICE2 4.28 GB, LANE2 4.28 GB.
Modes measured: 0 deployed (`--ffn deployed`), 18 wide-commit (`--ffn wide-commit`).
Sequence routes: mode 16 commits without widening; 17 widens without committing; 18 and 19 do both. Widening applies at 32 or more total rows; committing applies at every row count. FFN automatic dispatch is separate from this choice.
Wide-sequence layout: `direct`; operand: `int8`; token tile width: `auto`. `scaled-f16` rounds dequantised A8 inputs to FP16 and changes accumulation order; it is a different numerical map, not a layout-only option.

### decode: 32 streams x 8 steps

Fixed 8 greedy steps per stream, EOS emitted and fed back (streams are never stopped early), setup and prompt prefill excluded from the timing.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 132.8 | 132.8, 132.8, 133.2 | 0.3% | 1.000x | 1.927, 1.927, 1.922 | 2899 | 118 |
| 18 wide-commit | 204.8 | 204.8, 204.8, 203.8 | 0.5% | 1.541x | 1.250, 1.250, 1.256 | 2892 | 117 |

3 rounds per mode; worst spread 0.5%.

Host sysfs during these regions: gfx clock mean 2880-2899 MHz across modes, edge temperature max 66 C, board power mean 115-119 W.

### prefill: 128 rows/pass, 256 tokens

Logits (lm_head) computed on: last-pass-only, which here is 128 logit rows of 256 tokens, not one token: the final pass carries 128 rows and the lm_head runs on all of them. Every mode pays that same cost, but it is a real part of these numbers and it differs between the 128-row and other row settings. Token count is the document's real tokens.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 155.1 | 155.1, 150.2, 155.3 | 3.3% | 1.000x | 1.651, 1.705, 1.649 | 2890 | 119 |
| 18 wide-commit | 457.3 | 450.0, 463.5, 457.3 | 3.0% | 2.949x | 0.569, 0.552, 0.560 | 2864 | 123 |

3 rounds per mode; worst spread 3.3%.

Host sysfs during these regions: gfx clock mean 2846-2893 MHz across modes, edge temperature max 70 C, board power mean 116-125 W.

### prefill: 32 rows/pass, 256 tokens

Logits (lm_head) computed on: last-pass-only, which here is 32 logit rows of 256 tokens, not one token: the final pass carries 32 rows and the lm_head runs on all of them. Every mode pays that same cost, but it is a real part of these numbers and it differs between the 32-row and other row settings. Token count is the document's real tokens.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 156.5 | 156.5, 156.6, 156.4 | 0.1% | 1.000x | 1.636, 1.635, 1.637 | 2895 | 115 |
| 18 wide-commit | 325.8 | 325.8, 325.2, 326.4 | 0.4% | 2.082x | 0.786, 0.787, 0.784 | 2888 | 116 |

3 rounds per mode; worst spread 0.4%.

Host sysfs during these regions: gfx clock mean 2876-2896 MHz across modes, edge temperature max 65 C, board power mean 110-120 W.

### multistep state check

32 sequences carried 4 greedy steps from their own prompts, identical inputs per mode. Slot 0 length after the run: 39.

Step 0's token is the argmax of the prompt prefill, so a divergence at step 0 is a prefill-pass difference, not a decode-state one. Mode 4 retains the original FFN computation, so differences there call for inspection of state, replay, parity and floating evaluation order before blaming a new FFN map. The approximate modes carry a numeric difference into every step, so their divergence at any step, early or late, can be numerics alone.

| mode | streams matching mode 0 | tokens matching | first divergent step | first divergent stream |
|---|---:|---:|---:|---:|
| 0 deployed | reference | reference | - | - |
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
| 18 wide-commit | 32/32 | 0 | 0.0001 | 0.0007 | 0.141 | 5.0/5 | 0 |

### quality, prefill shape, 32 rows

32 teacher-forced rows, full 248320-entry vocabulary, identical token inputs per mode.
Rows are document tokens following the same 64-token context, so a smaller row count is a prefix of a larger one: the row counts are nested, not independent samples.

| mode | top1 agreement | rows changed | mean KL(0->m) | max KL | max centred logit diff | mean top5 overlap | worst row rank of mode0 top1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 deployed | reference | 0 | - | - | - | - | - |
| 18 wide-commit | 32/32 | 0 | 0.0007 | 0.0028 | 0.304 | 4.9/5 | 0 |

## Reading this

Every number above comes from the full model through Engine::forward_batch. Prefill and decode are separate workloads; decode tokens are real generated tokens at a fixed step count, which means a stream that emitted EOS kept going.
Sample size is small by design. The quality table compares a handful of teacher-forced rows, so it can show a mode is wrong; it cannot show a mode is good enough to deploy. Raise --rounds, --gen-steps and --quality-rows before drawing conclusions about a close result.

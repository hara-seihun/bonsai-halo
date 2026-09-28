# Batch mode comparison: auto-a8-ramped

Source: `../../data/bonsai2/batch-comparison/auto-a8-ramped/run.json`, run 2026-09-20T10:47:28 to 2026-09-20T10:49:16, resident server paused: True, full model layers: 64, batch capacity 128.
Model ../../data/bonsai2/PTQ1_0.gguf, context 512, slots 32, max rows per pass 128, document PLAN.md, prompts tools/batch-compare/prompts.txt.
3 rounds, mode order reshuffled per round (seed 1234), warmup 4.3 s, single process, state reset before every timed region.

### decode: 32 streams x 8 steps

Fixed 8 greedy steps per stream, EOS emitted and fed back (streams are never stopped early), setup and prompt prefill excluded from the timing.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 132.8 | 133.1, 132.8, 132.8 | 0.2% | 1.000x | 1.924, 1.928, 1.928 | 2899 | 120 |
| 5 auto-a8 | 159.4 | 159.6, 159.4, 159.1 | 0.3% | 1.200x | 1.604, 1.606, 1.609 | 2893 | 117 |

3 rounds per mode; worst spread 0.3%.

Host sysfs during these regions: gfx clock mean 2885-2899 MHz across modes, edge temperature max 71 C, board power mean 114-120 W.

### decode: 8 streams x 8 steps

Fixed 8 greedy steps per stream, EOS emitted and fed back (streams are never stopped early), setup and prompt prefill excluded from the timing.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 133.1 | 133.0, 133.1, 133.6 | 0.5% | 1.000x | 0.481, 0.481, 0.479 | 2897 | 120 |
| 5 auto-a8 | 142.5 | 142.5, 142.7, 141.4 | 1.0% | 1.071x | 0.449, 0.448, 0.453 | 2871 | 121 |

3 rounds per mode; worst spread 1.0%.

Host sysfs during these regions: gfx clock mean 2825-2898 MHz across modes, edge temperature max 71 C, board power mean 118-124 W.

### prefill: 128 rows/pass, 256 tokens

Logits (lm_head) computed on: last-pass-only, which here is 128 logit rows of 256 tokens, not one token: the final pass carries 128 rows and the lm_head runs on all of them. Every mode pays that same cost, but it is a real part of these numbers and it differs between the 128-row and other row settings. Token count is the document's real tokens.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 154.9 | 155.0, 154.9, 154.9 | 0.0% | 1.000x | 1.652, 1.653, 1.653 | 2894 | 118 |
| 5 auto-a8 | 291.2 | 289.2, 291.5, 291.2 | 0.8% | 1.880x | 0.885, 0.878, 0.879 | 2887 | 122 |

3 rounds per mode; worst spread 0.8%.

Host sysfs during these regions: gfx clock mean 2878-2895 MHz across modes, edge temperature max 71 C, board power mean 116-123 W.

### prefill: 32 rows/pass, 256 tokens

Logits (lm_head) computed on: last-pass-only, which here is 32 logit rows of 256 tokens, not one token: the final pass carries 32 rows and the lm_head runs on all of them. Every mode pays that same cost, but it is a real part of these numbers and it differs between the 32-row and other row settings. Token count is the document's real tokens.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 156.8 | 157.0, 156.7, 156.8 | 0.2% | 1.000x | 1.631, 1.634, 1.633 | 2894 | 117 |
| 5 auto-a8 | 216.0 | 216.4, 216.0, 215.6 | 0.4% | 1.378x | 1.183, 1.185, 1.188 | 2891 | 118 |

3 rounds per mode; worst spread 0.4%.

Host sysfs during these regions: gfx clock mean 2885-2898 MHz across modes, edge temperature max 67 C, board power mean 113-120 W.

### multistep state check

32 sequences carried 4 greedy steps from their own prompts, identical inputs per mode. Slot 0 length after the run: 39.

Step 0's token is the argmax of the prompt prefill, so a divergence at step 0 is a prefill-pass difference, not a decode-state one. Mode 4 retains the original FFN computation, so differences there call for inspection of state, replay, parity and floating evaluation order before blaming a new FFN map. The approximate modes carry a numeric difference into every step, so their divergence at any step, early or late, can be numerics alone.

| mode | streams matching mode 0 | tokens matching | first divergent step | first divergent stream |
|---|---:|---:|---:|---:|
| 0 deployed | reference | reference | - | - |
| 5 auto-a8 | 31/32 | 125/128 | 1 | 19 |

- mode 5 stream 19 first differs at step 1: mode 0 364, mode 5 264
  - mode 0 text: 'Testing for **sub'
  - mode 5 text: 'Testing a random number'

Greedy decode amplifies a small logit difference into a different token, so a divergence here is not by itself a quality verdict; read it with the logit tables below.

### quality, decode shape, 8 rows

8 teacher-forced rows, full 248320-entry vocabulary, identical token inputs per mode.
Rows are streams 0..n-1 of the prompts file, so a smaller row count is a prefix of a larger one: the row counts are nested, not independent samples.

| mode | top1 agreement | rows changed | mean KL(0->m) | max KL | max centred logit diff | mean top5 overlap | worst row rank of mode0 top1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 deployed | reference | 0 | - | - | - | - | - |
| 5 auto-a8 | 8/8 | 0 | 0.0000 | 0.0001 | 0.115 | 5.0/5 | 0 |

### quality, decode shape, 32 rows

32 teacher-forced rows, full 248320-entry vocabulary, identical token inputs per mode.
Rows are streams 0..n-1 of the prompts file, so a smaller row count is a prefix of a larger one: the row counts are nested, not independent samples.

| mode | top1 agreement | rows changed | mean KL(0->m) | max KL | max centred logit diff | mean top5 overlap | worst row rank of mode0 top1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 deployed | reference | 0 | - | - | - | - | - |
| 5 auto-a8 | 32/32 | 0 | 0.0001 | 0.0004 | 0.126 | 5.0/5 | 0 |

### quality, prefill shape, 8 rows

8 teacher-forced rows, full 248320-entry vocabulary, identical token inputs per mode.
Rows are document tokens following the same 64-token context, so a smaller row count is a prefix of a larger one: the row counts are nested, not independent samples.

| mode | top1 agreement | rows changed | mean KL(0->m) | max KL | max centred logit diff | mean top5 overlap | worst row rank of mode0 top1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 deployed | reference | 0 | - | - | - | - | - |
| 5 auto-a8 | 8/8 | 0 | 0.0003 | 0.0005 | 0.234 | 5.0/5 | 0 |

### quality, prefill shape, 32 rows

32 teacher-forced rows, full 248320-entry vocabulary, identical token inputs per mode.
Rows are document tokens following the same 64-token context, so a smaller row count is a prefix of a larger one: the row counts are nested, not independent samples.

| mode | top1 agreement | rows changed | mean KL(0->m) | max KL | max centred logit diff | mean top5 overlap | worst row rank of mode0 top1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 deployed | reference | 0 | - | - | - | - | - |
| 5 auto-a8 | 31/32 | 1 | 0.0006 | 0.0027 | 0.339 | 4.9/5 | 1 |

- mode 5 row 12: mode0 picks 14 (p=0.106), mode picks 318; mode0's token falls to rank 1 with p=0.104

## Reading this

Every number above comes from the full model through Engine::forward_batch. Prefill and decode are separate workloads; decode tokens are real generated tokens at a fixed step count, which means a stream that emitted EOS kept going.
Sample size is small by design. The quality table compares a handful of teacher-forced rows, so it can show a mode is wrong; it cannot show a mode is good enough to deploy. Raise --rounds, --gen-steps and --quality-rows before drawing conclusions about a close result.


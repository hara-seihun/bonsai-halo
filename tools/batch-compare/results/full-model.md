# Batch mode comparison: full-model-780ff71

Source: `../../data/bonsai2/batch-comparison/full-model-780ff71/run.json`, run 2026-09-20T10:34:38 to 2026-09-20T10:38:41, resident server paused: True, full model layers: 64, batch capacity 128.
Model ../../data/bonsai2/PTQ1_0.gguf, context 512, slots 32, max rows per pass 128, document PLAN.md, prompts tools/batch-compare/prompts.txt.
3 rounds, mode order reshuffled per round (seed 1234), warmup 3.1 s, single process, state reset before every timed region.

### decode: 32 streams x 8 steps

Fixed 8 greedy steps per stream, EOS emitted and fed back (streams are never stopped early), setup and prompt prefill excluded from the timing.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 132.6 | 132.6, 133.0, 132.6 | 0.3% | 1.000x | 1.931, 1.925, 1.930 | 2897 | 119 |
| 1 integer-a8 | 159.7 | 160.0, 159.7, 159.6 | 0.3% | 1.204x | 1.600, 1.603, 1.604 | 2898 | 114 |
| 2 integer-a4 | 164.8 | 164.8, 164.7, 165.1 | 0.2% | 1.242x | 1.554, 1.554, 1.551 | 2899 | 109 |
| 3 scaled-a8 | 157.9 | 157.9, 157.9, 157.8 | 0.1% | 1.190x | 1.621, 1.621, 1.622 | 2898 | 114 |
| 4 sliced-deployed | 148.4 | 148.4, 148.4, 148.4 | 0.0% | 1.119x | 1.725, 1.726, 1.725 | 2896 | 121 |

3 rounds per mode; worst spread 0.3%.

Host sysfs during these regions: gfx clock mean 2892-2900 MHz across modes, edge temperature max 70 C, board power mean 104-124 W.

### decode: 8 streams x 8 steps

Fixed 8 greedy steps per stream, EOS emitted and fed back (streams are never stopped early), setup and prompt prefill excluded from the timing.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 133.0 | 133.0, 132.9, 133.3 | 0.3% | 1.000x | 0.481, 0.481, 0.480 | 2898 | 120 |
| 1 integer-a8 | 82.6 | 82.5, 82.6, 82.6 | 0.2% | 0.621x | 0.776, 0.774, 0.775 | 2893 | 114 |
| 2 integer-a4 | 84.9 | 84.9, 85.0, 84.9 | 0.2% | 0.638x | 0.754, 0.753, 0.754 | 2891 | 111 |
| 3 scaled-a8 | 83.7 | 83.8, 83.7, 83.4 | 0.5% | 0.629x | 0.764, 0.765, 0.767 | 2889 | 115 |
| 4 sliced-deployed | 142.0 | 142.0, 142.0, 141.9 | 0.1% | 1.067x | 0.451, 0.451, 0.451 | 2875 | 122 |

3 rounds per mode; worst spread 0.5%.

Host sysfs during these regions: gfx clock mean 2872-2899 MHz across modes, edge temperature max 71 C, board power mean 109-129 W.

### prefill: 128 rows/pass, 256 tokens

Logits (lm_head) computed on: last-pass-only, which here is 128 logit rows of 256 tokens, not one token: the final pass carries 128 rows and the lm_head runs on all of them. Every mode pays that same cost, but it is a real part of these numbers and it differs between the 128-row and other row settings. Token count is the document's real tokens.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 154.9 | 154.9, 154.9, 154.8 | 0.1% | 1.000x | 1.653, 1.652, 1.654 | 2891 | 120 |
| 1 integer-a8 | 289.5 | 290.0, 289.4, 289.5 | 0.2% | 1.869x | 0.883, 0.885, 0.884 | 2888 | 124 |
| 2 integer-a4 | 311.7 | 312.5, 311.7, 311.1 | 0.5% | 2.012x | 0.819, 0.821, 0.823 | 2889 | 120 |
| 3 scaled-a8 | 291.0 | 291.0, 291.2, 286.8 | 1.5% | 1.879x | 0.880, 0.879, 0.893 | 2870 | 124 |
| 4 sliced-deployed | 191.0 | 191.0, 192.0, 190.3 | 0.9% | 1.233x | 1.340, 1.333, 1.345 | 2860 | 126 |

3 rounds per mode; worst spread 1.5%.

Host sysfs during these regions: gfx clock mean 2849-2897 MHz across modes, edge temperature max 74 C, board power mean 117-130 W.

### prefill: 32 rows/pass, 256 tokens

Logits (lm_head) computed on: last-pass-only, which here is 32 logit rows of 256 tokens, not one token: the final pass carries 32 rows and the lm_head runs on all of them. Every mode pays that same cost, but it is a real part of these numbers and it differs between the 32-row and other row settings. Token count is the document's real tokens.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 156.7 | 156.7, 156.8, 152.4 | 2.8% | 1.000x | 1.633, 1.633, 1.679 | 2887 | 116 |
| 1 integer-a8 | 216.2 | 216.9, 216.2, 216.0 | 0.4% | 1.379x | 1.180, 1.184, 1.185 | 2889 | 119 |
| 2 integer-a4 | 225.5 | 225.4, 225.6, 225.5 | 0.1% | 1.439x | 1.136, 1.135, 1.135 | 2893 | 118 |
| 3 scaled-a8 | 217.8 | 217.8, 217.4, 217.9 | 0.2% | 1.389x | 1.176, 1.177, 1.175 | 2893 | 119 |
| 4 sliced-deployed | 189.8 | 190.2, 188.1, 189.8 | 1.1% | 1.211x | 1.346, 1.361, 1.349 | 2864 | 120 |

3 rounds per mode; worst spread 2.8%.

Host sysfs during these regions: gfx clock mean 2851-2897 MHz across modes, edge temperature max 71 C, board power mean 111-124 W.

### multistep state check

32 sequences carried 4 greedy steps from their own prompts, identical inputs per mode. Slot 0 length after the run: 39.

Step 0's token is the argmax of the prompt prefill, so a divergence at step 0 is a prefill-pass difference, not a decode-state one. Mode 4 retains the original FFN computation, so differences there call for inspection of state, replay, parity and floating evaluation order before blaming a new FFN map. The approximate modes carry a numeric difference into every step, so their divergence at any step, early or late, can be numerics alone.

| mode | streams matching mode 0 | tokens matching | first divergent step | first divergent stream |
|---|---:|---:|---:|---:|
| 0 deployed | reference | reference | - | - |
| 1 integer-a8 | 32/32 | 128/128 | - | - |
| 2 integer-a4 | 26/32 | 114/128 | 0 | 6 |
| 3 scaled-a8 | 32/32 | 128/128 | - | - |
| 4 sliced-deployed | 32/32 | 128/128 | - | - |

- mode 2 stream 6 first differs at step 0: mode 0 13962, mode 2 1206
  - mode 0 text: '### How Ternary'
  - mode 2 text: 'To understand how tern'

Greedy decode amplifies a small logit difference into a different token, so a divergence here is not by itself a quality verdict; read it with the logit tables below.

At least one mode diverges at step 0, which points at the prompt prefill passes rather than at the decode loop.

### quality, decode shape, 8 rows

8 teacher-forced rows, full 248320-entry vocabulary, identical token inputs per mode.
Rows are streams 0..n-1 of the prompts file, so a smaller row count is a prefix of a larger one: the row counts are nested, not independent samples.

| mode | top1 agreement | rows changed | mean KL(0->m) | max KL | max centred logit diff | mean top5 overlap | worst row rank of mode0 top1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 deployed | reference | 0 | - | - | - | - | - |
| 1 integer-a8 | 8/8 | 0 | 0.0002 | 0.0006 | 0.193 | 5.0/5 | 0 |
| 2 integer-a4 | 7/8 | 1 | 0.0212 | 0.0848 | 1.700 | 4.2/5 | 1 |
| 3 scaled-a8 | 8/8 | 0 | 0.0002 | 0.0004 | 0.171 | 5.0/5 | 0 |
| 4 sliced-deployed | 8/8 | 0 | 0.0000 | 0.0000 | 0.000 | 5.0/5 | 0 |

- mode 2 row 6: mode0 picks 13962 (p=0.177), mode picks 1206; mode0's token falls to rank 1 with p=0.154

### quality, decode shape, 32 rows

32 teacher-forced rows, full 248320-entry vocabulary, identical token inputs per mode.
Rows are streams 0..n-1 of the prompts file, so a smaller row count is a prefix of a larger one: the row counts are nested, not independent samples.

| mode | top1 agreement | rows changed | mean KL(0->m) | max KL | max centred logit diff | mean top5 overlap | worst row rank of mode0 top1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 deployed | reference | 0 | - | - | - | - | - |
| 1 integer-a8 | 32/32 | 0 | 0.0003 | 0.0010 | 0.204 | 5.0/5 | 0 |
| 2 integer-a4 | 31/32 | 1 | 0.0236 | 0.0848 | 1.788 | 4.6/5 | 1 |
| 3 scaled-a8 | 32/32 | 0 | 0.0003 | 0.0010 | 0.171 | 5.0/5 | 0 |
| 4 sliced-deployed | 32/32 | 0 | 0.0000 | 0.0000 | 0.000 | 5.0/5 | 0 |

- mode 2 row 6: mode0 picks 13962 (p=0.177), mode picks 1206; mode0's token falls to rank 1 with p=0.154

### quality, prefill shape, 8 rows

8 teacher-forced rows, full 248320-entry vocabulary, identical token inputs per mode.
Rows are document tokens following the same 64-token context, so a smaller row count is a prefix of a larger one: the row counts are nested, not independent samples.

| mode | top1 agreement | rows changed | mean KL(0->m) | max KL | max centred logit diff | mean top5 overlap | worst row rank of mode0 top1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 deployed | reference | 0 | - | - | - | - | - |
| 1 integer-a8 | 8/8 | 0 | 0.0005 | 0.0013 | 0.281 | 5.0/5 | 0 |
| 2 integer-a4 | 8/8 | 0 | 0.0702 | 0.1375 | 3.560 | 4.5/5 | 0 |
| 3 scaled-a8 | 8/8 | 0 | 0.0006 | 0.0012 | 0.329 | 5.0/5 | 0 |
| 4 sliced-deployed | 8/8 | 0 | 0.0000 | 0.0000 | 0.000 | 5.0/5 | 0 |

### quality, prefill shape, 32 rows

32 teacher-forced rows, full 248320-entry vocabulary, identical token inputs per mode.
Rows are document tokens following the same 64-token context, so a smaller row count is a prefix of a larger one: the row counts are nested, not independent samples.

| mode | top1 agreement | rows changed | mean KL(0->m) | max KL | max centred logit diff | mean top5 overlap | worst row rank of mode0 top1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 deployed | reference | 0 | - | - | - | - | - |
| 1 integer-a8 | 31/32 | 1 | 0.0006 | 0.0027 | 0.339 | 4.9/5 | 1 |
| 2 integer-a4 | 28/32 | 4 | 0.0651 | 0.4581 | 3.560 | 4.3/5 | 3 |
| 3 scaled-a8 | 32/32 | 0 | 0.0008 | 0.0051 | 0.329 | 4.9/5 | 0 |
| 4 sliced-deployed | 32/32 | 0 | 0.0000 | 0.0000 | 0.000 | 5.0/5 | 0 |

- mode 1 row 12: mode0 picks 14 (p=0.106), mode picks 318; mode0's token falls to rank 1 with p=0.104
- mode 2 row 12: mode0 picks 14 (p=0.106), mode picks 59781; mode0's token falls to rank 2 with p=0.107
- mode 2 row 13: mode0 picks 585 (p=0.175), mode picks 318; mode0's token falls to rank 2 with p=0.068
- mode 2 row 15: mode0 picks 16 (p=0.287), mode picks 220; mode0's token falls to rank 3 with p=0.055
- mode 2 row 17: mode0 picks 11 (p=0.220), mode picks 8; mode0's token falls to rank 1 with p=0.181

## Reading this

Every number above comes from the full model through Engine::forward_batch. Prefill and decode are separate workloads; decode tokens are real generated tokens at a fixed step count, which means a stream that emitted EOS kept going.
Sample size is small by design. The quality table compares a handful of teacher-forced rows, so it can show a mode is wrong; it cannot show a mode is good enough to deploy. Raise --rounds, --gen-steps and --quality-rows before drawing conclusions about a close result.


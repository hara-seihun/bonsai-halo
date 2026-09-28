# Batch mode comparison: resident-tps-selected-m19

Source: `../../data/bonsai2/batch-comparison/resident-tps-selected-m19/run.json`, run 2026-09-21T01:01:22 to 2026-09-21T01:02:58, resident server paused: True, full model layers: 64, batch capacity 128.
Model ../../data/bonsai2/PTQ1_0.gguf, context 512, slots 32, max rows per pass 128, document PLAN.md, prompts tools/batch-compare/prompts.txt.
3 rounds, mode order reshuffled per round (seed 1234), warmup 4.2 s, single process, state reset before every timed region.
Device memory: 15.43 GB for the model, 25.40 GB after `prepare_batch(128, 0x100)`, of which 8.04 GB is the batched FFN module. Weight images for the modes measured budget at 8.02 GB, against 16.58 GB to hold every mode at once. Each image is a further representation of all 64 layers, held alongside the deployed weights for as long as the mode that reads it is available.
Resident weight images: SCALES 0.27 GB, PAIR 4.28 GB, DENSE5 3.48 GB.
Modes measured: 0 deployed (`--ffn deployed`), 19 wide-commit-a4 (`--ffn wide-commit-a4`).
Sequence routes: mode 16 commits without widening; 17 widens without committing; 18 and 19 do both. Widening applies at 32 or more total rows; committing applies at every row count. FFN automatic dispatch is separate from this choice.
Wide-sequence layout: `direct`; operand: `int8`; token tile width: `auto`. `scaled-f16` rounds dequantised A8 inputs to FP16 and changes accumulation order; it is a different numerical map, not a layout-only option.
Resident GDN state: `True`; state-row split: `4`. When enabled, committed wide GDN passes keep each sequence's state in registers across its tokens. Attention and uncommitted passes retain the sliced route.

### decode: 32 streams x 8 steps

Fixed 8 greedy steps per stream, EOS emitted and fed back (streams are never stopped early), setup and prompt prefill excluded from the timing.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 133.0 | 133.0, 133.0, 133.0 | 0.0% | 1.000x | 1.925, 1.925, 1.925 | 2899 | 119 |
| 19 wide-commit-a4 | 252.9 | 252.9, 251.9, 253.1 | 0.5% | 1.901x | 1.012, 1.016, 1.012 | 2867 | 119 |

3 rounds per mode; worst spread 0.5%.

Host sysfs during these regions: gfx clock mean 2810-2900 MHz across modes, edge temperature max 71 C, board power mean 117-123 W.

### decode: 8 streams x 8 steps

Fixed 8 greedy steps per stream, EOS emitted and fed back (streams are never stopped early), setup and prompt prefill excluded from the timing.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 133.0 | 132.8, 133.0, 133.9 | 0.9% | 1.000x | 0.482, 0.481, 0.478 | 2897 | 119 |
| 19 wide-commit-a4 | 147.4 | 147.3, 147.4, 148.0 | 0.5% | 1.108x | 0.435, 0.434, 0.432 | 2891 | 122 |

3 rounds per mode; worst spread 0.9%.

Host sysfs during these regions: gfx clock mean 2882-2898 MHz across modes, edge temperature max 71 C, board power mean 119-127 W.

### prefill: 128 rows/pass, 256 tokens

Logits (lm_head) computed on: last-pass-only, which here is 128 logit rows of 256 tokens, not one token: the final pass carries 128 rows and the lm_head runs on all of them. Every mode pays that same cost, but it is a real part of these numbers and it differs between the 128-row and other row settings. Token count is the document's real tokens.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 154.8 | 154.8, 154.8, 155.6 | 0.5% | 1.000x | 1.653, 1.653, 1.645 | 2893 | 119 |
| 19 wide-commit-a4 | 589.7 | 586.3, 590.8, 589.7 | 0.8% | 3.808x | 0.437, 0.433, 0.434 | 2882 | 121 |

3 rounds per mode; worst spread 0.8%.

Host sysfs during these regions: gfx clock mean 2862-2897 MHz across modes, edge temperature max 72 C, board power mean 119-121 W.

### prefill: 32 rows/pass, 256 tokens

Logits (lm_head) computed on: last-pass-only, which here is 32 logit rows of 256 tokens, not one token: the final pass carries 32 rows and the lm_head runs on all of them. Every mode pays that same cost, but it is a real part of these numbers and it differs between the 32-row and other row settings. Token count is the document's real tokens.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 156.7 | 156.6, 157.9, 156.7 | 0.8% | 1.000x | 1.635, 1.621, 1.634 | 2884 | 119 |
| 19 wide-commit-a4 | 440.6 | 440.3, 441.2, 440.6 | 0.2% | 2.812x | 0.581, 0.580, 0.581 | 2892 | 118 |

3 rounds per mode; worst spread 0.8%.

Host sysfs during these regions: gfx clock mean 2865-2899 MHz across modes, edge temperature max 68 C, board power mean 113-120 W.

### multistep state check

32 sequences carried 4 greedy steps from their own prompts, identical inputs per mode. Slot 0 length after the run: 39.

Step 0's token is the argmax of the prompt prefill, so a divergence at step 0 is a prefill-pass difference, not a decode-state one. Mode 4 retains the original FFN computation, so differences there call for inspection of state, replay, parity and floating evaluation order before blaming a new FFN map. The approximate modes carry a numeric difference into every step, so their divergence at any step, early or late, can be numerics alone.

| mode | streams matching mode 0 | tokens matching | first divergent step | first divergent stream |
|---|---:|---:|---:|---:|
| 0 deployed | reference | reference | - | - |
| 19 wide-commit-a4 | 28/32 | 118/128 | 0 | 6 |

- mode 19 stream 6 first differs at step 0: mode 0 13962, mode 19 2
  - mode 0 text: '### How Ternary'
  - mode 19 text: '# Ternary Weight'

Greedy decode amplifies a small logit difference into a different token, so a divergence here is not by itself a quality verdict; read it with the logit tables below.

At least one mode diverges at step 0, which points at the prompt prefill passes rather than at the decode loop.

### quality, decode shape, 32 rows

32 teacher-forced rows, full 248320-entry vocabulary, identical token inputs per mode.
Rows are streams 0..n-1 of the prompts file, so a smaller row count is a prefix of a larger one: the row counts are nested, not independent samples.

| mode | top1 agreement | rows changed | mean KL(0->m) | max KL | max centred logit diff | mean top5 overlap | worst row rank of mode0 top1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 deployed | reference | 0 | - | - | - | - | - |
| 19 wide-commit-a4 | 30/32 | 2 | 0.0126 | 0.0445 | 1.468 | 4.7/5 | 1 |

- mode 19 row 6: mode0 picks 13962 (p=0.177), mode picks 32; mode0's token falls to rank 1 with p=0.155
- mode 19 row 12: mode0 picks 71093 (p=0.355), mode picks 8160; mode0's token falls to rank 1 with p=0.285

### quality, prefill shape, 32 rows

32 teacher-forced rows, full 248320-entry vocabulary, identical token inputs per mode.
Rows are document tokens following the same 64-token context, so a smaller row count is a prefix of a larger one: the row counts are nested, not independent samples.

| mode | top1 agreement | rows changed | mean KL(0->m) | max KL | max centred logit diff | mean top5 overlap | worst row rank of mode0 top1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 deployed | reference | 0 | - | - | - | - | - |
| 19 wide-commit-a4 | 25/32 | 7 | 0.0704 | 0.3404 | 2.973 | 4.5/5 | 2 |

- mode 19 row 7: mode0 picks 220 (p=0.428), mode picks 4361; mode0's token falls to rank 1 with p=0.315
- mode 19 row 12: mode0 picks 14 (p=0.106), mode picks 318; mode0's token falls to rank 1 with p=0.107
- mode 19 row 13: mode0 picks 585 (p=0.175), mode picks 318; mode0's token falls to rank 1 with p=0.096
- mode 19 row 16: mode0 picks 19 (p=0.254), mode picks 17; mode0's token falls to rank 2 with p=0.165
- mode 19 row 17: mode0 picks 11 (p=0.220), mode picks 8; mode0's token falls to rank 1 with p=0.142
- mode 19 row 18: mode0 picks 11 (p=0.247), mode picks 15; mode0's token falls to rank 2 with p=0.128
- mode 19 row 23: mode0 picks 351 (p=0.275), mode picks 24; mode0's token falls to rank 1 with p=0.161

## Reading this

Every number above comes from the full model through Engine::forward_batch. Prefill and decode are separate workloads; decode tokens are real generated tokens at a fixed step count, which means a stream that emitted EOS kept going.
Sample size is small by design. The quality table compares a handful of teacher-forced rows, so it can show a mode is wrong; it cannot show a mode is good enough to deploy. Raise --rounds, --gen-steps and --quality-rows before drawing conclusions about a close result.

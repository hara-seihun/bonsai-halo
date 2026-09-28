# Batch mode comparison: direct-layout-a4

Source: `../../data/bonsai2/batch-comparison/direct-layout-a4/run.json`, run 2026-09-20T23:55:33 to 2026-09-20T23:56:59, resident server paused: True, full model layers: 64, batch capacity 128.
Model ../../data/bonsai2/PTQ1_0.gguf, context 512, slots 32, max rows per pass 128, document PLAN.md, prompts tools/batch-compare/prompts.txt.
3 rounds, mode order reshuffled per round (seed 1234), warmup 4.2 s, single process, state reset before every timed region.
Device memory: 15.43 GB for the model, 25.39 GB after `prepare_batch(128, 0x100)`, of which 8.04 GB is the batched FFN module. Weight images for the modes measured budget at 8.02 GB, against 16.58 GB to hold every mode at once. Each image is a further representation of all 64 layers, held alongside the deployed weights for as long as the mode that reads it is available.
Resident weight images: SCALES 0.27 GB, PAIR 4.28 GB, DENSE5 3.48 GB.
Modes measured: 0 deployed (`--ffn deployed`), 19 wide-commit-a4 (`--ffn wide-commit-a4`).
Sequence routes: mode 16 commits without widening; 17 widens without committing; 18 and 19 do both. Widening applies at 32 or more total rows; committing applies at every row count. FFN automatic dispatch is separate from this choice.
Wide-sequence layout: `direct`; operand: `int8`; token tile width: `auto`. `scaled-f16` rounds dequantised A8 inputs to FP16 and changes accumulation order; it is a different numerical map, not a layout-only option.

### decode: 32 streams x 8 steps

Fixed 8 greedy steps per stream, EOS emitted and fed back (streams are never stopped early), setup and prompt prefill excluded from the timing.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 133.2 | 133.2, 132.9, 133.2 | 0.2% | 1.000x | 1.922, 1.926, 1.922 | 2899 | 120 |
| 19 wide-commit-a4 | 245.0 | 244.9, 245.1, 245.0 | 0.1% | 1.839x | 1.046, 1.044, 1.045 | 2893 | 117 |

3 rounds per mode; worst spread 0.2%.

Host sysfs during these regions: gfx clock mean 2889-2900 MHz across modes, edge temperature max 70 C, board power mean 115-120 W.

### prefill: 128 rows/pass, 256 tokens

Logits (lm_head) computed on: last-pass-only, which here is 128 logit rows of 256 tokens, not one token: the final pass carries 128 rows and the lm_head runs on all of them. Every mode pays that same cost, but it is a real part of these numbers and it differs between the 128-row and other row settings. Token count is the document's real tokens.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 154.5 | 155.0, 154.2, 154.5 | 0.5% | 1.000x | 1.652, 1.660, 1.657 | 2885 | 120 |
| 19 wide-commit-a4 | 534.2 | 533.5, 535.3, 534.2 | 0.3% | 3.457x | 0.480, 0.478, 0.479 | 2892 | 120 |

3 rounds per mode; worst spread 0.5%.

Host sysfs during these regions: gfx clock mean 2867-2900 MHz across modes, edge temperature max 70 C, board power mean 119-122 W.

### prefill: 32 rows/pass, 256 tokens

Logits (lm_head) computed on: last-pass-only, which here is 32 logit rows of 256 tokens, not one token: the final pass carries 32 rows and the lm_head runs on all of them. Every mode pays that same cost, but it is a real part of these numbers and it differs between the 32-row and other row settings. Token count is the document's real tokens.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 156.6 | 156.6, 156.6, 155.6 | 0.6% | 1.000x | 1.635, 1.635, 1.645 | 2890 | 120 |
| 19 wide-commit-a4 | 408.5 | 408.3, 410.5, 408.5 | 0.5% | 2.609x | 0.627, 0.624, 0.627 | 2894 | 118 |

3 rounds per mode; worst spread 0.6%.

Host sysfs during these regions: gfx clock mean 2884-2900 MHz across modes, edge temperature max 68 C, board power mean 114-121 W.

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

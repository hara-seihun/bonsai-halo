# Batch mode comparison: wide-sequence-a4

Source: `../../data/bonsai2/batch-comparison/wide-sequence-a4/run.json`, run 2026-09-20T23:20:31 to 2026-09-20T23:22:51, resident server paused: True, full model layers: 64, batch capacity 128.
Model ../../data/bonsai2/PTQ1_0.gguf, context 512, slots 32, max rows per pass 128, document PLAN.md, prompts tools/batch-compare/prompts.txt.
3 rounds, mode order reshuffled per round (seed 1234), warmup 6.3 s, single process, state reset before every timed region.
Device memory: 15.43 GB for the model, 25.39 GB after `prepare_batch(128, 0x100)`, of which 8.04 GB is the batched FFN module. Weight images for the modes measured budget at 8.02 GB, against 16.58 GB to hold every mode at once. Each image is a further representation of all 64 layers, held alongside the deployed weights for as long as the mode that reads it is available.
Resident weight images: SCALES 0.27 GB, PAIR 4.28 GB, DENSE5 3.48 GB.
Modes measured: 0 deployed (`--ffn deployed`), 11 auto-optimized-a4 (`--ffn auto-map-a4`), 19 wide-commit-a4 (`--ffn wide-commit-a4`).
Sequence routes: mode 16 commits without widening; 17 widens without committing; 18 and 19 do both. Widening applies at 32 or more total rows; committing applies at every row count. FFN automatic dispatch is separate from this choice.
Wide-sequence operand: `int8`; token tile width: `auto`. `scaled-f16` rounds dequantised A8 inputs to FP16 and changes accumulation order; it is a different numerical map, not a layout-only option.

### decode: 32 streams x 8 steps

Fixed 8 greedy steps per stream, EOS emitted and fed back (streams are never stopped early), setup and prompt prefill excluded from the timing.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 133.6 | 133.6, 134.4, 133.4 | 0.8% | 1.000x | 1.917, 1.904, 1.919 | 2899 | 120 |
| 11 auto-optimized-a4 | 205.7 | 205.7, 206.2, 205.5 | 0.4% | 1.540x | 1.245, 1.241, 1.246 | 2892 | 118 |
| 19 wide-commit-a4 | 239.1 | 238.6, 239.5, 239.1 | 0.4% | 1.790x | 1.073, 1.069, 1.071 | 2890 | 117 |

3 rounds per mode; worst spread 0.8%.

Host sysfs during these regions: gfx clock mean 2882-2899 MHz across modes, edge temperature max 68 C, board power mean 116-120 W.

### decode: 8 streams x 8 steps

Fixed 8 greedy steps per stream, EOS emitted and fed back (streams are never stopped early), setup and prompt prefill excluded from the timing.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 133.6 | 133.3, 133.6, 133.7 | 0.3% | 1.000x | 0.480, 0.479, 0.479 | 2895 | 119 |
| 11 auto-optimized-a4 | 142.0 | 142.0, 141.9, 142.1 | 0.1% | 1.063x | 0.451, 0.451, 0.450 | 2889 | 121 |
| 19 wide-commit-a4 | 148.3 | 148.3, 148.6, 147.1 | 1.0% | 1.110x | 0.432, 0.431, 0.435 | 2890 | 120 |

3 rounds per mode; worst spread 1.0%.

Host sysfs during these regions: gfx clock mean 2884-2899 MHz across modes, edge temperature max 68 C, board power mean 118-123 W.

### prefill: 128 rows/pass, 256 tokens

Logits (lm_head) computed on: last-pass-only, which here is 128 logit rows of 256 tokens, not one token: the final pass carries 128 rows and the lm_head runs on all of them. Every mode pays that same cost, but it is a real part of these numbers and it differs between the 128-row and other row settings. Token count is the document's real tokens.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 156.4 | 156.3, 156.4, 157.3 | 0.7% | 1.000x | 1.638, 1.637, 1.628 | 2894 | 120 |
| 11 auto-optimized-a4 | 324.4 | 324.0, 324.4, 324.6 | 0.2% | 2.074x | 0.790, 0.789, 0.789 | 2879 | 122 |
| 19 wide-commit-a4 | 518.6 | 518.6, 519.4, 515.3 | 0.8% | 3.315x | 0.494, 0.493, 0.497 | 2873 | 120 |

3 rounds per mode; worst spread 0.8%.

Host sysfs during these regions: gfx clock mean 2843-2896 MHz across modes, edge temperature max 66 C, board power mean 119-127 W.

### prefill: 32 rows/pass, 256 tokens

Logits (lm_head) computed on: last-pass-only, which here is 32 logit rows of 256 tokens, not one token: the final pass carries 32 rows and the lm_head runs on all of them. Every mode pays that same cost, but it is a real part of these numbers and it differs between the 32-row and other row settings. Token count is the document's real tokens.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 157.0 | 156.9, 157.5, 157.0 | 0.4% | 1.000x | 1.632, 1.625, 1.631 | 2891 | 119 |
| 11 auto-optimized-a4 | 293.6 | 291.8, 293.6, 293.6 | 0.6% | 1.870x | 0.877, 0.872, 0.872 | 2886 | 121 |
| 19 wide-commit-a4 | 395.4 | 394.4, 399.6, 395.4 | 1.3% | 2.519x | 0.649, 0.641, 0.647 | 2887 | 119 |

3 rounds per mode; worst spread 1.3%.

Host sysfs during these regions: gfx clock mean 2863-2899 MHz across modes, edge temperature max 66 C, board power mean 118-123 W.

### multistep state check

32 sequences carried 4 greedy steps from their own prompts, identical inputs per mode. Slot 0 length after the run: 39.

Step 0's token is the argmax of the prompt prefill, so a divergence at step 0 is a prefill-pass difference, not a decode-state one. Mode 4 retains the original FFN computation, so differences there call for inspection of state, replay, parity and floating evaluation order before blaming a new FFN map. The approximate modes carry a numeric difference into every step, so their divergence at any step, early or late, can be numerics alone.

| mode | streams matching mode 0 | tokens matching | first divergent step | first divergent stream |
|---|---:|---:|---:|---:|
| 0 deployed | reference | reference | - | - |
| 11 auto-optimized-a4 | 28/32 | 115/128 | 0 | 6 |
| 19 wide-commit-a4 | 28/32 | 118/128 | 0 | 6 |

- mode 11 stream 6 first differs at step 0: mode 0 13962, mode 11 1206
  - mode 0 text: '### How Ternary'
  - mode 11 text: 'To understand how tern'
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
| 11 auto-optimized-a4 | 31/32 | 1 | 0.0151 | 0.0774 | 1.682 | 4.6/5 | 1 |
| 19 wide-commit-a4 | 30/32 | 2 | 0.0126 | 0.0445 | 1.468 | 4.7/5 | 1 |

- mode 11 row 6: mode0 picks 13962 (p=0.177), mode picks 32; mode0's token falls to rank 1 with p=0.151
- mode 19 row 6: mode0 picks 13962 (p=0.177), mode picks 32; mode0's token falls to rank 1 with p=0.155
- mode 19 row 12: mode0 picks 71093 (p=0.355), mode picks 8160; mode0's token falls to rank 1 with p=0.285

### quality, prefill shape, 32 rows

32 teacher-forced rows, full 248320-entry vocabulary, identical token inputs per mode.
Rows are document tokens following the same 64-token context, so a smaller row count is a prefix of a larger one: the row counts are nested, not independent samples.

| mode | top1 agreement | rows changed | mean KL(0->m) | max KL | max centred logit diff | mean top5 overlap | worst row rank of mode0 top1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 deployed | reference | 0 | - | - | - | - | - |
| 11 auto-optimized-a4 | 28/32 | 4 | 0.0651 | 0.4581 | 3.560 | 4.3/5 | 3 |
| 19 wide-commit-a4 | 25/32 | 7 | 0.0704 | 0.3404 | 2.973 | 4.5/5 | 2 |

- mode 11 row 12: mode0 picks 14 (p=0.106), mode picks 59781; mode0's token falls to rank 2 with p=0.107
- mode 11 row 13: mode0 picks 585 (p=0.175), mode picks 318; mode0's token falls to rank 2 with p=0.068
- mode 11 row 15: mode0 picks 16 (p=0.287), mode picks 220; mode0's token falls to rank 3 with p=0.055
- mode 11 row 17: mode0 picks 11 (p=0.220), mode picks 8; mode0's token falls to rank 1 with p=0.181
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

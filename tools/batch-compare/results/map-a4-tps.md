# Batch mode comparison: map-a4-tps-006c2d5

Source: `../../data/bonsai2/batch-comparison/map-a4-tps-006c2d5/run.json`, run 2026-09-20T20:54:14 to 2026-09-20T20:56:57, resident server paused: True, full model layers: 64, batch capacity 128.
Model ../../data/bonsai2/PTQ1_0.gguf, context 512, slots 32, max rows per pass 128, document PLAN.md, prompts tools/batch-compare/prompts.txt.
3 rounds, mode order reshuffled per round (seed 1234), warmup 6.4 s, single process, state reset before every timed region.
Device memory: 15.43 GB for the model, 27.74 GB after `prepare_batch(128, 0x102)`, of which 12.31 GB is the batched FFN module. Weight images for the modes measured budget at 12.30 GB, against 16.58 GB to hold every mode at once. Each image is a further representation of all 64 layers, held alongside the deployed weights for as long as the mode that reads it is available.
Resident weight images: SCALES 0.27 GB, SLICE2 4.28 GB, PAIR 4.28 GB, DENSE5 3.48 GB.
Modes measured: 0 deployed (`--ffn deployed`), 5 auto-a8 (`--ffn auto`), 11 auto-optimized-a4 (`--ffn auto-map-a4`).

### decode: 32 streams x 8 steps

Fixed 8 greedy steps per stream, EOS emitted and fed back (streams are never stopped early), setup and prompt prefill excluded from the timing.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 133.5 | 133.6, 133.5, 18.5 | 86.2% | 1.000x | 1.917, 1.918, 13.834 | 2381 | 100 |
| 5 auto-a8 | 159.9 | 160.0, 159.9, 159.9 | 0.1% | 1.198x | 1.600, 1.601, 1.601 | 2895 | 116 |
| 11 auto-optimized-a4 | 206.1 | 205.8, 206.6, 206.1 | 0.4% | 1.544x | 1.244, 1.239, 1.242 | 2898 | 119 |

Run-to-run spread reaches 86.2% over 3 rounds. Differences smaller than that are not resolved by this sample.

Host sysfs during these regions: gfx clock mean 1345-2900 MHz across modes, edge temperature max 73 C, board power mean 62-120 W.

### decode: 8 streams x 8 steps

Fixed 8 greedy steps per stream, EOS emitted and fed back (streams are never stopped early), setup and prompt prefill excluded from the timing.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 133.1 | 134.1, 133.1, 133.0 | 0.9% | 1.000x | 0.477, 0.481, 0.481 | 2899 | 120 |
| 5 auto-a8 | 142.4 | 142.4, 142.7, 142.2 | 0.3% | 1.070x | 0.450, 0.449, 0.450 | 2883 | 122 |
| 11 auto-optimized-a4 | 142.2 | 142.1, 142.4, 142.2 | 0.2% | 1.068x | 0.450, 0.449, 0.450 | 2898 | 120 |

3 rounds per mode; worst spread 0.9%.

Host sysfs during these regions: gfx clock mean 2880-2899 MHz across modes, edge temperature max 72 C, board power mean 117-130 W.

### prefill: 128 rows/pass, 256 tokens

Logits (lm_head) computed on: last-pass-only, which here is 128 logit rows of 256 tokens, not one token: the final pass carries 128 rows and the lm_head runs on all of them. Every mode pays that same cost, but it is a real part of these numbers and it differs between the 128-row and other row settings. Token count is the document's real tokens.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 155.1 | 155.1, 155.3, 155.0 | 0.2% | 1.000x | 1.651, 1.649, 1.651 | 2895 | 120 |
| 5 auto-a8 | 290.5 | 289.2, 290.5, 291.3 | 0.7% | 1.873x | 0.885, 0.881, 0.879 | 2875 | 122 |
| 11 auto-optimized-a4 | 324.5 | 324.3, 324.5, 325.9 | 0.5% | 2.092x | 0.789, 0.789, 0.785 | 2895 | 120 |

3 rounds per mode; worst spread 0.7%.

Host sysfs during these regions: gfx clock mean 2872-2898 MHz across modes, edge temperature max 74 C, board power mean 119-125 W.

### prefill: 32 rows/pass, 256 tokens

Logits (lm_head) computed on: last-pass-only, which here is 32 logit rows of 256 tokens, not one token: the final pass carries 32 rows and the lm_head runs on all of them. Every mode pays that same cost, but it is a real part of these numbers and it differs between the 32-row and other row settings. Token count is the document's real tokens.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 156.7 | 156.8, 156.7, 156.7 | 0.0% | 1.000x | 1.633, 1.633, 1.633 | 2884 | 119 |
| 5 auto-a8 | 216.4 | 216.4, 216.9, 216.2 | 0.3% | 1.380x | 1.183, 1.180, 1.184 | 2893 | 119 |
| 11 auto-optimized-a4 | 293.4 | 293.0, 293.8, 293.4 | 0.3% | 1.872x | 0.874, 0.871, 0.872 | 2893 | 119 |

3 rounds per mode; worst spread 0.3%.

Host sysfs during these regions: gfx clock mean 2881-2898 MHz across modes, edge temperature max 70 C, board power mean 117-120 W.

### multistep state check

32 sequences carried 4 greedy steps from their own prompts, identical inputs per mode. Slot 0 length after the run: 39.

Step 0's token is the argmax of the prompt prefill, so a divergence at step 0 is a prefill-pass difference, not a decode-state one. Mode 4 retains the original FFN computation, so differences there call for inspection of state, replay, parity and floating evaluation order before blaming a new FFN map. The approximate modes carry a numeric difference into every step, so their divergence at any step, early or late, can be numerics alone.

| mode | streams matching mode 0 | tokens matching | first divergent step | first divergent stream |
|---|---:|---:|---:|---:|
| 0 deployed | reference | reference | - | - |
| 5 auto-a8 | 31/32 | 125/128 | 1 | 19 |
| 11 auto-optimized-a4 | 28/32 | 115/128 | 0 | 6 |

- mode 5 stream 19 first differs at step 1: mode 0 364, mode 5 264
  - mode 0 text: 'Testing for **sub'
  - mode 5 text: 'Testing a random number'
- mode 11 stream 6 first differs at step 0: mode 0 13962, mode 11 1206
  - mode 0 text: '### How Ternary'
  - mode 11 text: 'To understand how tern'

Greedy decode amplifies a small logit difference into a different token, so a divergence here is not by itself a quality verdict; read it with the logit tables below.

At least one mode diverges at step 0, which points at the prompt prefill passes rather than at the decode loop.

### quality, decode shape, 32 rows

32 teacher-forced rows, full 248320-entry vocabulary, identical token inputs per mode.
Rows are streams 0..n-1 of the prompts file, so a smaller row count is a prefix of a larger one: the row counts are nested, not independent samples.

| mode | top1 agreement | rows changed | mean KL(0->m) | max KL | max centred logit diff | mean top5 overlap | worst row rank of mode0 top1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 deployed | reference | 0 | - | - | - | - | - |
| 5 auto-a8 | 32/32 | 0 | 0.0001 | 0.0004 | 0.126 | 5.0/5 | 0 |
| 11 auto-optimized-a4 | 31/32 | 1 | 0.0151 | 0.0774 | 1.682 | 4.6/5 | 1 |

- mode 11 row 6: mode0 picks 13962 (p=0.177), mode picks 32; mode0's token falls to rank 1 with p=0.151

### quality, prefill shape, 32 rows

32 teacher-forced rows, full 248320-entry vocabulary, identical token inputs per mode.
Rows are document tokens following the same 64-token context, so a smaller row count is a prefix of a larger one: the row counts are nested, not independent samples.

| mode | top1 agreement | rows changed | mean KL(0->m) | max KL | max centred logit diff | mean top5 overlap | worst row rank of mode0 top1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 deployed | reference | 0 | - | - | - | - | - |
| 5 auto-a8 | 31/32 | 1 | 0.0006 | 0.0027 | 0.339 | 4.9/5 | 1 |
| 11 auto-optimized-a4 | 28/32 | 4 | 0.0651 | 0.4581 | 3.560 | 4.3/5 | 3 |

- mode 5 row 12: mode0 picks 14 (p=0.106), mode picks 318; mode0's token falls to rank 1 with p=0.104
- mode 11 row 12: mode0 picks 14 (p=0.106), mode picks 59781; mode0's token falls to rank 2 with p=0.107
- mode 11 row 13: mode0 picks 585 (p=0.175), mode picks 318; mode0's token falls to rank 2 with p=0.068
- mode 11 row 15: mode0 picks 16 (p=0.287), mode picks 220; mode0's token falls to rank 3 with p=0.055
- mode 11 row 17: mode0 picks 11 (p=0.220), mode picks 8; mode0's token falls to rank 1 with p=0.181

## Reading this

Every number above comes from the full model through Engine::forward_batch. Prefill and decode are separate workloads; decode tokens are real generated tokens at a fixed step count, which means a stream that emitted EOS kept going.
Sample size is small by design. The quality table compares a handful of teacher-forced rows, so it can show a mode is wrong; it cannot show a mode is good enough to deploy. Raise --rounds, --gen-steps and --quality-rows before drawing conclusions about a close result.


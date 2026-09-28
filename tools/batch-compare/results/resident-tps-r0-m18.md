# Batch mode comparison: resident-tps-r0-m18

Source: `../../data/bonsai2/batch-comparison/resident-tps-r0-m18/run.json`, run 2026-09-21T00:55:11 to 2026-09-21T00:56:50, resident server paused: True, full model layers: 64, batch capacity 128.
Model ../../data/bonsai2/PTQ1_0.gguf, context 512, slots 32, max rows per pass 128, document PLAN.md, prompts tools/batch-compare/prompts.txt.
3 rounds, mode order reshuffled per round (seed 1234), warmup 4.2 s, single process, state reset before every timed region.
Device memory: 15.43 GB for the model, 26.19 GB after `prepare_batch(128, 0x80)`, of which 8.84 GB is the batched FFN module. Weight images for the modes measured budget at 8.82 GB, against 16.58 GB to hold every mode at once. Each image is a further representation of all 64 layers, held alongside the deployed weights for as long as the mode that reads it is available.
Resident weight images: SCALES 0.27 GB, SLICE2 4.28 GB, LANE2 4.28 GB.
Modes measured: 0 deployed (`--ffn deployed`), 18 wide-commit (`--ffn wide-commit`).
Sequence routes: mode 16 commits without widening; 17 widens without committing; 18 and 19 do both. Widening applies at 32 or more total rows; committing applies at every row count. FFN automatic dispatch is separate from this choice.
Wide-sequence layout: `direct`; operand: `int8`; token tile width: `auto`. `scaled-f16` rounds dequantised A8 inputs to FP16 and changes accumulation order; it is a different numerical map, not a layout-only option.
Resident GDN state: `False`; state-row split: `4`. When enabled, committed wide GDN passes keep each sequence's state in registers across its tokens. Attention and uncommitted passes retain the sliced route.

### decode: 32 streams x 8 steps

Fixed 8 greedy steps per stream, EOS emitted and fed back (streams are never stopped early), setup and prompt prefill excluded from the timing.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 133.2 | 133.3, 133.0, 133.2 | 0.2% | 1.000x | 1.921, 1.924, 1.922 | 2892 | 117 |
| 18 wide-commit | 205.3 | 205.0, 205.3, 205.3 | 0.1% | 1.541x | 1.249, 1.247, 1.247 | 2895 | 116 |

3 rounds per mode; worst spread 0.2%.

Host sysfs during these regions: gfx clock mean 2879-2900 MHz across modes, edge temperature max 65 C, board power mean 114-119 W.

### decode: 8 streams x 8 steps

Fixed 8 greedy steps per stream, EOS emitted and fed back (streams are never stopped early), setup and prompt prefill excluded from the timing.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 132.9 | 132.9, 133.5, 132.8 | 0.5% | 1.000x | 0.482, 0.479, 0.482 | 2897 | 118 |
| 18 wide-commit | 147.3 | 148.1, 147.3, 147.3 | 0.6% | 1.108x | 0.432, 0.434, 0.434 | 2880 | 122 |

3 rounds per mode; worst spread 0.6%.

Host sysfs during these regions: gfx clock mean 2871-2899 MHz across modes, edge temperature max 67 C, board power mean 117-127 W.

### prefill: 128 rows/pass, 256 tokens

Logits (lm_head) computed on: last-pass-only, which here is 128 logit rows of 256 tokens, not one token: the final pass carries 128 rows and the lm_head runs on all of them. Every mode pays that same cost, but it is a real part of these numbers and it differs between the 128-row and other row settings. Token count is the document's real tokens.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 155.1 | 155.0, 155.1, 155.4 | 0.3% | 1.000x | 1.652, 1.651, 1.647 | 2897 | 118 |
| 18 wide-commit | 453.2 | 453.2, 465.1, 331.8 | 29.4% | 2.922x | 0.565, 0.550, 0.772 | 2845 | 121 |

Run-to-run spread reaches 29.4% over 3 rounds. Differences smaller than that are not resolved by this sample.

Host sysfs during these regions: gfx clock mean 2797-2898 MHz across modes, edge temperature max 68 C, board power mean 114-126 W.

### prefill: 32 rows/pass, 256 tokens

Logits (lm_head) computed on: last-pass-only, which here is 32 logit rows of 256 tokens, not one token: the final pass carries 32 rows and the lm_head runs on all of them. Every mode pays that same cost, but it is a real part of these numbers and it differs between the 32-row and other row settings. Token count is the document's real tokens.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 157.0 | 156.7, 157.0, 157.0 | 0.2% | 1.000x | 1.634, 1.630, 1.631 | 2895 | 116 |
| 18 wide-commit | 326.9 | 326.4, 327.0, 326.9 | 0.2% | 2.083x | 0.784, 0.783, 0.783 | 2890 | 117 |

3 rounds per mode; worst spread 0.2%.

Host sysfs during these regions: gfx clock mean 2884-2896 MHz across modes, edge temperature max 63 C, board power mean 111-121 W.

### stalled samples

1 timed sample(s) ran materially longer than the other rounds of the same mode and shape. They are kept in every table above and in the raw JSON. A stall is a real thing that happened to this machine, and dropping it would make the benchmark look steadier than the machine is. The median is the headline statistic precisely so one stall cannot move it, and the spread figure is where it shows.

| round | mode | shape | wall s | median s | ratio | worst pass/step s | GPU busy % | host CPU busy |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 2 | 18 wide-commit | prefill, 128 rows/pass, 256 tokens | 0.772 | 0.565 | 1.37x | 0.766 | 90 | 2.04 |

- round 2 mode 18 wide-commit, prefill 128 rows/pass, 256 tokens: 0.772 s against a 0.565 s median, 0.207 s excess. pass 1 of 2 took 0.766 s, 99% of the region
  - host during the region: CPU busy 2.04 thread-seconds per wall second, io full PSI 0.00 s, io some PSI 0.00 s, cpu some PSI 0.00 s, 32 involuntary context switches, MemAvailable 27.0 to 27.0 GB
  - none of the host counters account for the excess: not faulting, not preempted, not stalled on memory or I/O

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

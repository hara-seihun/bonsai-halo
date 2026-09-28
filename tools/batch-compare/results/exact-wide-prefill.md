# Batch mode comparison: exact-wide-prefill-r2

Source: `../../data/bonsai2/batch-comparison/exact-wide-prefill-r2/run.json`, run 2026-09-21T13:49:17 to None, resident server paused: True, full model layers: 64, batch capacity 128.
Model ../../data/bonsai2/PTQ1_0.gguf, context 512, slots 32, max rows per pass 128, document PLAN.md, prompts tools/batch-compare/prompts.txt.
5 rounds, mode order reshuffled per round (seed 1234), warmup 6.0 s, single process, state reset before every timed region.
Device memory: 15.43 GB for the model, 25.53 GB after `prepare_batch(128, 0x100)`, of which 8.04 GB is the batched FFN module. Weight images for the modes measured budget at 8.02 GB, against 16.58 GB to hold every mode at once. Each image is a further representation of all 64 layers, held alongside the deployed weights for as long as the mode that reads it is available.
Resident weight images: SCALES 0.27 GB, PAIR 4.28 GB, DENSE5 3.48 GB.
Modes measured: 0 deployed (`--ffn deployed`), 20 wide-deployed (`--ffn wide-deployed`), 19 wide-commit-a4 (`--ffn wide-commit-a4`).
Sequence routes: mode 16 commits without widening; 17 widens without committing; 18 and 19 do both. Widening applies at 32 or more total rows; committing applies at every row count. FFN automatic dispatch is separate from this choice.
Wide-sequence layout: `direct`; operand: `int8`; token tile width: `auto`. `scaled-f16` rounds dequantised A8 inputs to FP16 and changes accumulation order; it is a different numerical map, not a layout-only option.
Resident GDN state: `True`; state-row split: `4`. When enabled, committed wide GDN passes keep each sequence's state in registers across its tokens. Attention and uncommitted passes retain the sliced route.

### prefill: 128 rows/pass, 384 tokens

Logits (lm_head) computed on: last-pass-only, which here is 128 logit rows of 384 tokens, not one token: the final pass carries 128 rows and the lm_head runs on all of them. Every mode pays that same cost, but it is a real part of these numbers and it differs between the 128-row and other row settings. Token count is the document's real tokens.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 156.4 | 156.6, 156.1, 157.4, 154.7 | 1.7% | 1.000x | 2.452, 2.460, 2.440, 2.482 | 2786 | 119 |
| 19 wide-commit-a4 | 726.0 | 725.3, 726.8, 726.7, 715.4 | 1.6% | 4.643x | 0.529, 0.528, 0.528, 0.537 | 2721 | 120 |
| 20 wide-deployed | 265.4 | 265.6, 265.2, 265.9, 264.2 | 0.6% | 1.697x | 1.446, 1.448, 1.444, 1.454 | 2693 | 120 |

4 rounds per mode; worst spread 1.7%.

Host sysfs during these regions: gfx clock mean 2638-2809 MHz across modes, edge temperature max 77 C, board power mean 119-123 W.

No quality pass in this run.

## Reading this

Every number above comes from the full model through Engine::forward_batch. Prefill and decode are separate workloads; decode tokens are real generated tokens at a fixed step count, which means a stream that emitted EOS kept going.
Sample size is small by design. The quality table compares a handful of teacher-forced rows, so it can show a mode is wrong; it cannot show a mode is good enough to deploy. Raise --rounds, --gen-steps and --quality-rows before drawing conclusions about a close result.


# Batch mode comparison: resident-a8-repeat-r0

Source: `../../data/bonsai2/batch-comparison/resident-a8-repeat-r0/run.json`, run 2026-09-21T01:03:41 to 2026-09-21T01:03:46, resident server paused: True, full model layers: 64, batch capacity 128.
Model ../../data/bonsai2/PTQ1_0.gguf, context 512, slots 1, max rows per pass 128, document PLAN.md, prompts tools/batch-compare/prompts.txt.
5 rounds, mode order reshuffled per round (seed 1234), warmup 2.0 s, single process, state reset before every timed region.
Device memory: 7.17 GB for the model, 17.94 GB after `prepare_batch(128, 0x80)`, of which 8.84 GB is the batched FFN module. Weight images for the modes measured budget at 8.82 GB, against 16.58 GB to hold every mode at once. Each image is a further representation of all 64 layers, held alongside the deployed weights for as long as the mode that reads it is available.
Resident weight images: SCALES 0.27 GB, SLICE2 4.28 GB, LANE2 4.28 GB.
Modes measured: 18 wide-commit (`--ffn wide-commit`).
Sequence routes: mode 16 commits without widening; 17 widens without committing; 18 and 19 do both. Widening applies at 32 or more total rows; committing applies at every row count. FFN automatic dispatch is separate from this choice.
Wide-sequence layout: `direct`; operand: `int8`; token tile width: `auto`. `scaled-f16` rounds dequantised A8 inputs to FP16 and changes accumulation order; it is a different numerical map, not a layout-only option.
Resident GDN state: `False`; state-row split: `4`. When enabled, committed wide GDN passes keep each sequence's state in registers across its tokens. Attention and uncommitted passes retain the sliced route.

### prefill: 128 rows/pass, 256 tokens

Logits (lm_head) computed on: last-pass-only, which here is 128 logit rows of 256 tokens, not one token: the final pass carries 128 rows and the lm_head runs on all of them. Every mode pays that same cost, but it is a real part of these numbers and it differs between the 128-row and other row settings. Token count is the document's real tokens.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 18 wide-commit | 451.4 | 458.6, 443.5, 451.4, 448.5, 452.0 | 3.4% | n/a | 0.558, 0.577, 0.567, 0.571, 0.566 | 2697 | 124 |

5 rounds per mode; worst spread 3.4%.

Host sysfs during these regions: gfx clock mean 2671-2742 MHz across modes, edge temperature max 63 C, board power mean 105-135 W.

No quality pass in this run.

## Reading this

Every number above comes from the full model through Engine::forward_batch. Prefill and decode are separate workloads; decode tokens are real generated tokens at a fixed step count, which means a stream that emitted EOS kept going.
Sample size is small by design. The quality table compares a handful of teacher-forced rows, so it can show a mode is wrong; it cannot show a mode is good enough to deploy. Raise --rounds, --gen-steps and --quality-rows before drawing conclusions about a close result.

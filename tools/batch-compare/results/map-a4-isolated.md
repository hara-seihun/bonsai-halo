# Batch mode comparison: map-a4-isolated-943379a

Source: `../../data/bonsai2/batch-comparison/map-a4-isolated-943379a/run.json`, run 2026-09-20T21:19:21 to 2026-09-20T21:21:40, resident server paused: True, full model layers: 64, batch capacity 128.
Model ../../data/bonsai2/PTQ1_0.gguf, context 512, slots 32, max rows per pass 32, document PLAN.md, prompts tools/batch-compare/prompts.txt.
5 rounds, mode order reshuffled per round (seed 1234), warmup 6.4 s, single process, state reset before every timed region.
Device memory: 15.43 GB for the model, 27.74 GB after `prepare_batch(128, 0x102)`, of which 12.31 GB is the batched FFN module. Weight images for the modes measured budget at 12.30 GB, against 16.58 GB to hold every mode at once. Each image is a further representation of all 64 layers, held alongside the deployed weights for as long as the mode that reads it is available.
Resident weight images: SCALES 0.27 GB, SLICE2 4.28 GB, PAIR 4.28 GB, DENSE5 3.48 GB.
Modes measured: 0 deployed (`--ffn deployed`), 5 auto-a8 (`--ffn auto`), 11 auto-optimized-a4 (`--ffn auto-map-a4`).

### decode: 32 streams x 16 steps

Fixed 16 greedy steps per stream, EOS emitted and fed back (streams are never stopped early), setup and prompt prefill excluded from the timing.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 133.6 | 133.4, 133.6, 133.9, 133.6, 134.0 | 0.4% | 1.000x | 3.837, 3.831, 3.823, 3.833, 3.821 | 2899 | 119 |
| 5 auto-a8 | 160.4 | 160.3, 160.6, 160.6, 160.4, 160.4 | 0.2% | 1.200x | 3.194, 3.187, 3.188, 3.192, 3.191 | 2899 | 113 |
| 11 auto-optimized-a4 | 206.2 | 206.2, 206.4, 206.2, 206.1, 206.8 | 0.3% | 1.543x | 2.483, 2.481, 2.483, 2.484, 2.476 | 2897 | 118 |

5 rounds per mode; worst spread 0.4%.

Host sysfs during these regions: gfx clock mean 2892-2900 MHz across modes, edge temperature max 71 C, board power mean 112-120 W.

No quality pass in this run.

## Reading this

Every number above comes from the full model through Engine::forward_batch. Prefill and decode are separate workloads; decode tokens are real generated tokens at a fixed step count, which means a stream that emitted EOS kept going.
Sample size is small by design. The quality table compares a handful of teacher-forced rows, so it can show a mode is wrong; it cannot show a mode is good enough to deploy. Raise --rounds, --gen-steps and --quality-rows before drawing conclusions about a close result.


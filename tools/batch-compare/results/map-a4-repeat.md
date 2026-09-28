# Batch mode comparison: map-a4-repeat-006c2d5

Source: `../../data/bonsai2/batch-comparison/map-a4-repeat-006c2d5/run.json`, run 2026-09-20T21:13:36 to 2026-09-20T21:16:05, resident server paused: True, full model layers: 64, batch capacity 128.
Model ../../data/bonsai2/PTQ1_0.gguf, context 512, slots 32, max rows per pass 32, document PLAN.md, prompts tools/batch-compare/prompts.txt.
5 rounds, mode order reshuffled per round (seed 1234), warmup 6.5 s, single process, state reset before every timed region.
Device memory: 15.43 GB for the model, 27.74 GB after `prepare_batch(128, 0x102)`, of which 12.31 GB is the batched FFN module. Weight images for the modes measured budget at 12.30 GB, against 16.58 GB to hold every mode at once. Each image is a further representation of all 64 layers, held alongside the deployed weights for as long as the mode that reads it is available.
Resident weight images: SCALES 0.27 GB, SLICE2 4.28 GB, PAIR 4.28 GB, DENSE5 3.48 GB.
Modes measured: 0 deployed (`--ffn deployed`), 5 auto-a8 (`--ffn auto`), 11 auto-optimized-a4 (`--ffn auto-map-a4`).

### decode: 32 streams x 16 steps

Fixed 16 greedy steps per stream, EOS emitted and fed back (streams are never stopped early), setup and prompt prefill excluded from the timing.

| mode | tok/s median | per-round tok/s | spread | vs mode 0 | wall s | gfx MHz | W |
|---|---:|---|---:|---:|---|---:|---:|
| 0 deployed | 133.0 | 133.2, 133.0, 132.9, 133.0, 39.6 | 70.4% | 1.000x | 3.844, 3.849, 3.852, 3.849, 12.942 | 2693 | 111 |
| 5 auto-a8 | 160.0 | 156.4, 160.0, 160.5, 160.1, 160.0 | 2.6% | 1.203x | 3.274, 3.200, 3.191, 3.198, 3.199 | 2890 | 115 |
| 11 auto-optimized-a4 | 204.9 | 202.8, 205.5, 204.9, 205.3, 181.0 | 12.0% | 1.540x | 2.525, 2.491, 2.499, 2.493, 2.829 | 2863 | 120 |

Run-to-run spread reaches 70.4% over 5 rounds. Differences smaller than that are not resolved by this sample.

Host sysfs during these regions: gfx clock mean 1870-2899 MHz across modes, edge temperature max 69 C, board power mean 77-126 W.

### stalled samples

1 timed sample(s) ran materially longer than the other rounds of the same mode and shape. They are kept in every table above and in the raw JSON. A stall is a real thing that happened to this machine, and dropping it would make the benchmark look steadier than the machine is. The median is the headline statistic precisely so one stall cannot move it, and the spread figure is where it shows.

| round | mode | shape | wall s | median s | ratio | worst pass/step s | GPU busy % | host CPU busy |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 4 | 0 deployed | decode, 32 streams x 16 steps | 12.942 | 3.849 | 3.36x | 2.778 | 44 | - |

- round 4 mode 0 deployed, decode 32 streams x 16 steps: 12.942 s against a 3.849 s median, 9.092 s excess. step 9 of 16 took 2.778 s, 21% of the region
  - no host diagnostics in this record, so the stall cannot be attributed

Records without host diagnostics predate them. Re-running with the current driver captures rusage, PSI and MemAvailable per measurement, which is what distinguishes memory reclaim, paging, CPU contention and blocking I/O from each other.

No quality pass in this run.

## Reading this

Every number above comes from the full model through Engine::forward_batch. Prefill and decode are separate workloads; decode tokens are real generated tokens at a fixed step count, which means a stream that emitted EOS kept going.
Sample size is small by design. The quality table compares a handful of teacher-forced rows, so it can show a mode is wrong; it cannot show a mode is good enough to deploy. Raise --rounds, --gen-steps and --quality-rows before drawing conclusions about a close result.


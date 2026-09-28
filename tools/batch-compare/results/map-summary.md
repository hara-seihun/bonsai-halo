# Full-model results from the packed-map FFN work

The full 64-layer model, context allocation 512, one Radeon 8060S. These are aggregate generation rates across 32 independent streams, not single-stream rates.

| Mode | Generation tokens/s | Against original in its panel | Against previous automatic A8 |
|---|---:|---:|---:|
| Original | 133.1 / 133.6 | reference | |
| Previous automatic A8 | 160.1 / 160.4 | 1.20x | reference |
| Optimized integer A8 | 165.8 | 1.246x | 1.036x |
| Optimized scaled A8 | 176.6 | 1.327x | 1.103x |
| Optimized A4 | 206.2 | 1.543x | 1.285x |

A8 and A4 use separate panels because holding every additional weight image beside 32 sequence slots exceeds the host's GPU allocation budget. Both panels include original and automatic-A8 controls. [A8 results](map-a8-tps.md) have three randomized rounds and eight generation steps per stream. The final [isolated A4 results](map-a4-isolated.md) have five randomized rounds and sixteen steps. Each run ramps the GPU before timing; clocks in the 32-stream regions are about 2.9 GHz. One-time loading and repacking are excluded, all input-dependent inference is included.

## Prompt processing

256 document tokens, 128 rows per pass, full vocabulary computed on every row of the final pass:

| Mode | Tokens/s |
|---|---:|
| Original | 155.3 / 155.1 |
| Previous automatic A8 | 291.5 / 290.5 |
| Optimized integer A8 | 287.3 |
| Optimized scaled A8 | 291.6 |
| Optimized A4 | 324.5 |

The new integer A8 map helps 32-stream generation but loses 1.4% to the previous path at this prefill shape. Scaled A8 ties the previous prefill rate. A4 is 2.09x the original and 1.12x previous automatic A8. At 32 rows per pass, A4 reaches 293.4 versus 156.7 original; that configuration computes fewer output-head rows, so it is not directly comparable to the 128-row column.

At eight generation streams, all automatic modes retain the same sliced original path, about 142 tokens/s versus 133 original. No new single-token improvement is claimed here.

## Numerical changes

[Matched-family full-model logits](map-adopt-quality.md) are bit-identical and finite for each new path against its numerical control at 32, 40, 88 and 128 rows. That establishes that these storage and scheduling changes did not add a new numerical perturbation on the tested inputs. It does not make A4 equivalent to A8.

Against the original engine on the 32-row fixed-input checks:

| Mode | Decode top-1 agreement | Prefill top-1 agreement | Mean prefill KL | Four-step continuation token agreement |
|---|---:|---:|---:|---:|
| Previous and optimized integer A8 | 32/32 | 31/32 | 0.0006 | 125/128 |
| Optimized scaled A8 | 32/32 | 32/32 | 0.0008 | 128/128 |
| Optimized A4 | 31/32 | 28/32 | 0.0651 | 115/128 |

A4 is the throughput winner with a larger numerical change. These short checks do not constitute task-quality acceptance. The default engine and serving behavior remain unchanged.

## Isolation failure and repair

The first A4 run and its first repeat contain large baseline stalls. They remain in [the first report](map-a4-tps.md) and [the repeat report](map-a4-repeat.md). A new Pi session restarted `bonsai-halo.service` while each benchmark was running:

- First run 20:54:14 to 20:56:57, service start 20:56:02.
- Repeat 21:13:36 to 21:16:05, service start 21:15:44.

Stopping the transient service on entry was insufficient. `tools/run-batch-compare` now runtime-masks it while holding the research GPU lock, removes only the mask it created, and restores the entry state on exit. A native test confirmed that `systemd-run` cannot recreate the unit while masked. The driver records that mask and host fault, scheduling and pressure counters. The five-round isolated confirmation has 0.4% baseline spread and 0.3% A4 spread, with no recurrence of the stalls. It owns the generation headline above. The first run's prefill samples precede the service restart; its numerical outputs are retained alongside timing.

## Reproduction

```sh
make -j8 tools/batch_compare
# A8 panel
tools/run-batch-compare --tag maps-a8 --modes 0,5,9,10 --streams 8,32 \
  --prefill-rows 32,128 --prefill-tokens 256 --rounds 3 --gen-steps 8 \
  --quality-rows 32 --multistep-streams 32 --multistep-steps 4 --reference-pairs 9:5
# A4 generation confirmation
tools/run-batch-compare --tag maps-a4 --modes 0,5,11 --only decode --streams 32 \
  --rounds 5 --gen-steps 16 --slots 32 --context 512 --telemetry-raw
```

Raw runs live under `../../data/bonsai2/batch-comparison/`. Each derived report names its source directory, commit, timestamp and memory use. The A8 panel holds 24.27 GB total device allocations and the A4 panel 27.74 GB. A production run selecting only one numerical mode needs fewer images than either comparison panel.

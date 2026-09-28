# A4 row-tile ownership: one build, one process

Three ownerships of mode 8's pair-code weight row tile, measured against each other inside a
single process with the case order reshuffled every round. `ffn_batch_set_a4_sched()` switches
them between calls, so both the executable and the clock are shared by every arm.

- `TILE` (0) one row tile per wave, the schedule mode 8 shipped with
- `ADAPT` (1) one row tile shared by a workgroup's waves, targeting four waves per SIMD32
- `WIDE` (2) the same sharing targeting eight, **selected**

This matters because the arms are close enough that pairing separate processes cannot order them.
Two back-to-back processes on this device moved mode 0 by 8% on one morning's runs, which is larger
than the TILE-to-ADAPT step. Every number below comes from one process.

## Whole pass

`tools/batch_profile`, mode 19 (A4), 128-token prefix then one pass with logits on every row,
three rounds, median of three:

| rows | TILE | ADAPT | WIDE | ADAPT vs TILE | WIDE vs TILE |
|---:|---:|---:|---:|---:|---:|
| 128 | 203.563 ms | 199.633 ms | **193.884 ms** | +1.97% | **+4.99%** |
| 32 | 72.647 ms | 72.804 ms | 71.628 ms | −0.22% | +1.42% |

Samples at 128 rows: TILE 203.6 / 203.0 / 204.5, ADAPT 201.8 / 199.6 / 199.0, WIDE 193.9 / 194.9 /
192.5. The three groups do not overlap, so the ordering is resolved.

The 32-row row is a null, and reads as one. At 32 rows `run_ffn_batch` takes mode 8's dense
five-trit arm, which this selector does not reach, so all three arms execute the same kernel on the
same bytes. Their 1.4% spread is therefore this panel's noise floor at that shape, not an effect,
and it bounds what three rounds can resolve here.

## Full model

`tools/batch_compare --modes 19 --ffn-sched 0,2`, which interleaves the ownership with the mode
inside one process. 256-token document in 128-row passes with logits on the last pass; generation
is aggregate output across 32 streams. Three rounds, median:

| workload | TILE | WIDE | change |
|---|---:|---:|---:|
| prefill, 128 rows/pass | 672.1 tok/s | **720.4 tok/s** | **+7.19%** |
| generation, 32 streams | 268.7 tok/s | 268.9 tok/s | +0.07% |

Prefill samples: TILE 643.5 / 672.1 / 672.8, WIDE 703.6 / 720.4 / 723.6.
Generation samples: TILE 269.2 / 268.7 / 268.0, WIDE 266.7 / 268.9 / 269.2.

Generation does not move, and should not: a 32-stream step is 32 rows, which takes the dense arm.
This is a prompt-processing change. It is worth saying out loud because the two workloads are
separate outcomes and only one of them is touched.

## Numerical result

The three ownerships change which workgroup owns a weight row tile. They build the same A operand
values from the same codes, issue the same WMMA instructions against the same accumulators in the
same order, apply the same block scales and store the same elements from the same wave.

[Full-vocabulary comparison](ffn-sched-identity.json): 71,516,160 finite logits at 32, 40, 88 and
128 prefill rows, TILE against WIDE, **zero differing bits** and zero non-finite pairs. Both arms
are the same executable, separated by `HALO_FFN_A4_SCHED`, which `run.json` records.

Every timing panel also hashes the 128×5120 FP32 residual: one FNV-64 per row count across all
three ownerships, in both the three-way and the two-way panels.

## Raw data

In [the comparison directory](../../../../data/bonsai2/batch-comparison/README.md):

| Run | Contents |
|---|---|
| `wide-prep-launch/a4-sched-threeway.json` | the three-way, three rounds, 32 and 128 rows |
| `wide-prep-launch/a4-share-inproc.json` | the earlier TILE/ADAPT two-way that corrected a cross-process estimate of that step from 2.9% to 1.76% of a pass |
| `ffn-sched-tps/` | full-model prefill and 32-stream generation, ownership interleaved in-process |
| `ffn-sched-ident-tile/`, `ffn-sched-ident-wide/` | the paired logit dumps |

## Reproducing

```sh
make -j8 bonsai-halo tools/batch_compare tools/batch_profile
tools/run-batch-compare --profile-tool --modes 19 --rows 32,128 --heads 1 --traces 0 \
  --ffn-sched 0,1,2 --rounds 3 --out .../a4-sched-threeway.json
tools/run-batch-compare --tag ffn-sched-tps --modes 19 --ffn-sched 0,2 \
  --only prefill,decode --prefill-rows 128 --streams 32 --prefill-tokens 256 --rounds 3 --slots 32
```

`HALO_FFN_A4_SCHED=0|1|2` pins one ownership for a whole process, for a run that cannot use the
setter. `--ffn-sched` accepts `-1` to leave whatever the build selects.

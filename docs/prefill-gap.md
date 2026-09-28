# Where the prompt-processing time goes

The remaining gap is mostly outside the FFNs. I measured the real 64-layer model, rather than estimating phase time from its operation count.

The [phase tables](../tools/batch-compare/results/phase-profile.md) contain three randomized rounds with tracing on and off. Each case starts from the same 128-token prefix and processes 32 or 128 new tokens. The output head either runs on every new row or is omitted. HIP events bracket launches; the existing device barrier timestamps divide the persistent sequence kernels. No kernel arithmetic changed for this measurement.

## The measured split

Median milliseconds for 128 new tokens with logits on every row:

| Work | Scaled A8, mode 10 | FFN A4, mode 11 |
|---|---:|---:|
| All FFNs, including their producers | 139.7 | 96.2 |
| GDN projections | 137.6 | 137.9 |
| Attention projections | 39.5 | 39.7 |
| GDN state update and replay | 59.5 | 59.5 |
| Attention QK, softmax and AV | 9.2 | 9.2 |
| Other sequence prep, barriers and kernel overhead | about 42 | about 42 |
| Output head, including normalization and argmax (now 11.2, see [wide head](wide-head.md)) | 25.5 | 25.2 |
| Embedding and gaps between launches | about 3.3 | about 3.2 |
| Device span | 455.9 | 413.1 |

The uninstrumented A4 passes took 410.9, 410.7 and 410.2 ms. Traced device spans are within about 1% of that. Their residual checksums agree. Scaled A8 varies more, 450.6 to 462.4 ms, with the traced span inside that range. One 1472 ms uninstrumented, no-head sample remains in the source data; the other two in its group were about 386 ms. Nothing was discarded.

The published 324.5 prompt tokens/s measures a 256-token document with the head on the final 128 rows only. This profile charges the head on all 128 measured rows. A no-head pass plus a head pass takes about 796 ms here, or 322 tokens/s for 256 tokens, close to that benchmark.

FFNs are now only 23% of the A4 device span. Even deleting them would leave approximately 317 ms for these 128 rows. Further FFN work is useful, but it cannot close the whole-model gap by itself.

## Eight rows occupy a 16-column matrix instruction

The non-FFN projections still run in eight-row slices. In `mvw_rows`, `col & 7` supplies activation columns 0 through 7 twice to a 16-column IU8 WMMA. The store accepts only `col < nrows`. Half the matrix result is redundant and discarded.

This is an exact count of wasted arithmetic in this schedule, not a measured 2x opportunity for the whole model. Weight expansion, scales, loads, reductions and synchronization also take time. The two WMMA calls per K16 slice cover two different 16-weight-row matrices; those two calls are not themselves duplicates.

Measured useful throughput at 128 rows:

| Region | Useful conventional TOPS | Register-resident WMMA probe TOPS |
|---|---:|---:|
| Scaled-A8 FFNs, including producers | 31.4 | 55.3 FP16 |
| A4 FFNs, including producers | 45.5 | 109.3 IU4 |
| Non-FFN projections | 10.4 | 55.5 IU8 |

The non-FFN projection count includes both recurrent and attention projections. Its 10.4 useful TOPS represents about 20.8 TOPS of issued matrix arithmetic after accounting for duplicated columns. Thus filling the matrix tiles explains one factor of two, not the remaining factor of about 2.7 versus the register-only probe. The probe is an achieved reference rate, not a proved hardware ceiling.

Those projections take about 178 ms, or 43% of the A4 pass. They are the first target for the same wider token ownership and direct operand construction already used by the FFNs.

## The recurrent engine pays for speculative rollback during prefill

`ph_gdn` consumes `S.n_replay + S.nrows` tokens. It first replays the accepted prefix into the committed state, stores that state, then computes the new rows without committing them. The next pass does the work again. `ph_gdn_pre` also maintains the convolution replay records.

This supports rejected speculative tokens. Ordinary prompt ingestion accepts every token and does not need that rollback contract.

A second three-round profile starts with no prefix. Its first eight-row slice has no replay and spends 53.4 to 53.7 microseconds per GDN layer in the state phase. Later eight-row slices replay eight accepted rows and spend about 76.7 microseconds. These are different positions, not an isolated replay-disabled kernel comparison. They locate real extra work and suggest its size; they do not establish the speedup of a replacement.

The whole state-and-replay phase costs about 60 ms. It is not just the few milliseconds obtained by dividing its MAC count by a vector FMA rate. The code carries serial token dependencies, warp reductions, gates and state traffic, on a grid shared with the matrix kernels. A direct-commit, non-speculative prefill path is the next state experiment. It needs its own tested state contract rather than simply setting replay counts to zero, which would lose accepted state.

## Grid and memory observations

Native dispatch tracing records `k_forward_rows<8,0>` at 15,360 threads, 256 per workgroup: **60 workgroups**. It allocates 224 VGPRs per lane. The compiler also reports 20 bytes of private space, zero VGPR spills and 89 SGPR spills; the emitted function uses VGPR lanes for scalar spill traffic and contains no scratch load/store instructions. Private-space allocation is not evidence of hot-loop scratch-memory traffic. This is the register-heavy kernel that also constrains the shared cooperative grid for the original smaller-row variants. Raising occupancy alone is not automatically faster: unit counts and work per unit must suit the grid.

The projections logically revisit their weights once per eight-row slice. At 128 rows, that is sixteen visits. The current A4 FFN map also revisits weights across its four-token-tile groups; neither path implements the ideal "one weight read per 128 tokens" memory model. The output head alone has sixteen visits to its roughly 278 MB weight image.

Counter work exposed a measurement trap. ROCm's `FETCH_SIZE` returns half the known byte count on this device: a contiguous 256 MiB read reports 128 MiB, and a 1 GiB read reports 512 MiB, across repeated calls. The calibration probe and raw counters are retained. External-to-L2 request counts also do not by themselves identify physical DRAM traffic after downstream caches. Therefore this report does not turn that field into an unqualified DRAM-bandwidth verdict. The phase timing and duplicated-column findings do not depend on the counter.

The first full-model counter runs completed inference and wrote their trace files, then crashed while destroying an HSA queue. The [ROCR repair](../../../machine/rocm.md#patched-hsa-runtime-cooperative-queue-teardown) backports the upstream cooperative-queue lifetime fix. After relinking against the repaired runtime, the full-model counter run exits cleanly and reproduces the native residual checksum. `traffic-fixed/` owns the completed trace; failed-run logs remain. The timing table above ran without rocprof.

## What I would change next

1. Implemented in [wide sequence projections](wide-sequence.md): fill 16 useful columns and share weight operands across token tiles. The original A8 quantizer is retained; FP32 projection reduction order changes. The new phase measurements reduce about 180 ms of non-FFN projection time to 54 ms at 128 rows.
2. The [direct-commit GDN path](../tools/direct-commit/README.md) now removes replay, and native mixed-mode state acceptance passes. Carrying more than eight tokens through each state load remains open.
3. Implemented in [the wide vocabulary head](wide-head.md): the head now runs once over all the pass's rows instead of once per eight-row slice, at 11.2 ms rather than 26.6 for 128 rows, with the same logit bits. Skipping vocabulary rows nobody reads remains unexplored and would be a different, non-exact change.
4. Tune FFNs further against matched controls. They still run below their register-only rate, but their smaller share now limits their whole-model payoff.

These are measured priorities, not a promise that the optimistic 1,500-token/s mixed-format budget is reachable. The budget omitted work and assumed reuse that the current execution does not provide.

## Single-token result from the same round

The [five-round native comparison](../tools/batch-compare/results/single-map.md) tested 64 generated tokens per round with one stream:

| Path | Median tokens/s |
|---|---:|
| Original | 34.0 |
| Original arithmetic, own occupancy grid | 33.6 |
| Palette operand map | 33.4 |
| Palette plus finer GDN units | 33.1 |
| Palette, finer state units and finer K splits | 34.5 |

The first three alternatives reproduce the original logits exactly on both recorded one-row inputs. The K-split variant changes floating-point evaluation and gains 1.6%. All variants matched the 32-token continuation in this run. There is no new exact single-stream winner, and the default remains unchanged.

## Reproduce

```sh
make -j6 tools/batch_profile tools/profile_read_check tools/batch_compare
tools/run-batch-compare --profile-tool --modes 10,11 --rows 32,128 --rounds 3 \
  --out ../../data/bonsai2/batch-comparison/phase-profile/run.json
python3 tools/batch_profile.py ../../data/bonsai2/batch-comparison/phase-profile/run.json \
  --out tools/batch-compare/results/phase-profile.md
```

The raw owner is `../../data/bonsai2/batch-comparison/phase-profile/`. `source.txt` pins the first timing binary and source hashes. `no-prefix.json` contains the replay comparison. `calibration/` contains the known-byte counter check. `traffic-fixed.json` and `traffic-fixed/` contain the clean counter run on the repaired runtime; [the calibrated summary](../tools/batch-compare/results/phase-counters.json) retains both raw values and the calibration factor. `resources.log` and `halo_rows.s` hold compiler resources and emitted ISA. Each profiling pass records every event and phase timestamp. The runner holds the same GPU reservation and service mask as the full-model benchmark.

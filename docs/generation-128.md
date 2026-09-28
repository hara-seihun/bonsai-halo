# Generation at 64 and 128 streams

the project owner requested a real 128-stream measurement after the [combined throughput panel](full-tps-20260921.md). All 64 layers ran on the Radeon 8060S, with 64 generated tokens per stream and three measured rounds.

| Streams | Aggregate tokens/s, median | Tokens/s per stream | Aggregate rounds |
|---|---:|---:|---|
| 64 | 463.14 | 7.24 | 463.14, 462.85, 465.76 |
| 128 | 517.73 | 4.04 | 523.30, 517.02, 517.73 |

Doubling streams bought 11.8% more total throughput. It lowered each stream's rate by 44.1%. There is no batched drafter in these measurements.

The numerical settings were mode 19, INT8 recurrent-state storage, A4 FFNs and A4 sequence-input activations. These are approximate target-model routes. Only the pair-code FFN image was resident, saving the dense image's 3.48 GB; the default image selector uses pair codes at 128 rows anyway. At 64 rows, wide-only storage also forces pair codes for the down projection where the both-image selector would use dense codes, so this 64-stream result is the memory-saving configuration rather than an unrestricted optimum.

Context capacity was **256 tokens per stream**, with short distinct starting prompts, not 256 tokens of prefilled context. Generation followed the benchmark's fixed-step policy, feeding EOS back rather than stopping a stream. Prompt processing and loading are excluded from generation time. The driver used 128 sequence slots for both widths. Its input file keeps the existing 40 mixed-task prompts and adds 88 distinct arithmetic, Python, probability and queue-design prompts.

The earlier 32-stream panel measured 383.32 aggregate tokens/s. That run used both FFN images, 1024-token capacity and ordinary engine allocations, so the comparison to it is not an isolated batch-size experiment. The 64 and 128 rows above share one process, one engine and the same allocation policy.

## Making the run fit

The measured process held **43,814,933,576 bytes**, 43.81 GB or 40.81 GiB, of tracked GPU-addressable allocations. INT8 state reduces traffic but the current engine still reserves FP32-sized state regions. That is why 128 streams remain memory-hungry.

Two maintained changes made the experiment executable:

- `HALO_SNAPSHOT_KV_GB=0` now omits the four unused prefix-snapshot state areas as well as disabling snapshot saves. This saves 880,410,624 bytes without changing live sequence storage or kernel arithmetic. Default snapshot behavior is unchanged.
- `HALO_MANAGED_ALLOC=1` deliberately uses `hipMallocManaged` for allocations owned by `Engine::dmalloc`, rather than exhausting ordinary HIP allocation space first. The engine already used managed allocation after an ordinary allocation failure; the explicit policy preserves ordinary allocation capacity for its FFN and projection modules. The tracked-device budget counts both allocation kinds. Kernel code and numerical layouts are unchanged.

The first admitted attempt allocated the base engine but the FFN preparation refused its 4.55 GB image because `hipMemGetInfo` reported only 2.65 GB free of 32.47 GB. This is the host's ordinary GPU allocation policy, described in the [machine GPU-memory handbook](../../../machine/gpu-memory.md), not its total physical RAM. The managed-allocation run completed without changing the driver or rebooting.

The runner gained explicit process and host-reserve budgets. This panel used `--memory-gib 43 --host-reserve-gib 7`, giving a 43 GiB process limit, a 41 GiB tracked-device ceiling and a 7 GiB host reserve. Default budgets remain 28, 26 and 8 GiB respectively. Admission still rejects insufficient available or physical memory. It now allows up to five seconds for GPU allocations to be reclaimed after the resident service exits; a direct observation saw available memory grow from 40,224,912 to 52,370,004 KiB during the first second after stop. CPU-heavy lane work yielded during this large-memory measurement and resumed afterward. The runner restored the resident service, confirmed active after the run.

## Provenance and reproduction

Raw logs, prompts, hashes, failed admission/allocation attempts and the successful `managed/run.json` live in [`tps128-20260921-2306`](../../../data/bonsai2/batch-comparison/tps128-20260921-2306/). `summary.json` contains the medians.

The measured executable is source `1c7d7f2`, based on `83f0478` from the preceding panel with only the snapshot allocation and explicit managed-allocation changes above. Runner source is `dbf23c4`. The engine does not include the deferred-state commit or other kernel changes published during this measurement.

```sh
HALO_SNAPSHOT_KV_GB=0 HALO_MANAGED_ALLOC=1 \
  tools/run-batch-compare --memory-gib 43 --host-reserve-gib 7 --pin-clock \
  --tag tps128-20260921-2306/managed --modes 19 --gdn-state 3 --seq-quant a4 \
  --a4-images wide --streams 64,128 --slots 128 --context 256 --gen-steps 64 \
  --rounds 3 --only decode --prefill-rows 128 --head-rows tail --warmup-steps 8 \
  --prompts ../../data/bonsai2/batch-comparison/tps128-20260921-2306/prompts.txt
```

`--pin-clock` requests the high performance level, not a fixed shader frequency. Each run retains clock, power and temperature telemetry. No serving precision default changed.

## The curve has moved (September 21, 19:00)

[Sequences per generation step](decode-streams.md) re-measured 64 streams on `b4e5365` at **597
tok/s**, above the 517.73 this panel measured at 128, so the conclusion that width buys 11.8% is a
fact about `1c7d7f2` rather than about the engine. 32 streams is 477 on the same executable, so the
32-to-64 step is now +25%. 128 streams has not been re-taken: admission needs 50 GiB of host
`MemAvailable` for the 43 GiB budget above, and the box has been sitting at 38 GiB with the
resident engine up. That document also prices why the budget is that large and what would remove it.

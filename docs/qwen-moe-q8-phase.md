# Q8 decode by output width

The six-token reference decode trace can be joined to the pinned GGUF tensor inventory without guessing that every Q8 call reads one matrix. It identifies where a faster Q8 consumer could move the *whole* step, and closes the tempting explanation that the shared-expert gate/up calls are all separate reads.

`tools/qwen-moe/q8_phase.py` checks the Q8 call count against every nonembedding Q8 tensor. The 512-output group has 100 tensors but 60 calls/token: 40 fused shared-expert gate/up calls each read two tensors, and 20 attention K/V calls read one. The other groups have one call per tensor. The grid is 32 times the output width; all measured Q8 calls use workgroup 32. This match accounts for all 250 Q8 tensors and 210 calls/token. The 540 MB embedding matrix is a row lookup, not a full decode matvec.

| Output width | Calls/token | Q8 image MB/token | Device ms/token | Image-byte rate GB/s | One-read time at 242 GB/s, ms | Difference, ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 60, including 40 fused | 111.411 | 0.783 | 142.3 | 0.460 | 0.322 |
| 2048 | 80 | 401.080 | 2.184 | 183.7 | 1.657 | 0.527 |
| 4096 | 30 | 267.387 | 1.416 | 188.9 | 1.105 | 0.311 |
| 8192 | 40 | 713.032 | 3.805 | 187.4 | 2.946 | 0.858 |
| **All Q8** | **210** | **1492.910** | **8.187** | **182.3** | **6.169** | **2.018** |

At width 512, fused shared gate/up takes 0.592 ms/token and K/V takes 0.190. At width 2048 the kernel's `true,false` variant takes 0.481 ms/token over ten calls and the ordinary variant takes 1.703 over 70 calls. The 8192-output Q/K/QKV projections contribute 42.5% of the Q8 difference against the one-read comparator; the 512-output calls have the worst image-byte rate but only 16.0% of the difference. Prioritize the wide Q8 projection phase if the next hardware counters confirm it spends those bytes on DRAM. Do not start with 512-output shared-expert fusion: it already exists and has less total room.

**Conditional cost result.** If each Q8 matrix is read once from DRAM, no other device work overlaps it, the measured 242 GB/s reference is attainable in every Q8 shape, and every other phase keeps its traced time, replacing *all* Q8 calls by ideal streaming saves at most 2.018 ms of the 19.707 ms/token summed device time. That changes a 50.74 device-time-equivalent tokens/s step to 56.53, an 11.4% rate gain. Perfecting only the 8192 group saves at most 0.858 ms, a 4.6% rate gain on this trace. These are conditional targets, not ISA bounds or predictions of installed full-model throughput. Cache reuse could make the one-read floor too high; repeated reads or unmeasured instruction limits could make it unattainable. The trace has no occupied prefix or warmup, and profiler host overhead breaks wall throughput. Actual DRAM bytes and paired unprofiled phase timing are needed to distinguish traffic from utilization. Neither a single Q8 kernel ratio nor the 242 GB/s comparator may be reported as a full-model speedup.

The retained [six-token trace receipt](../../data/qwen-moe/benchmarks/profile-20260923/receipt.json) pins reference source `1a07bfa5f`, binary and libraries, no-warmup input and model hash. Its CSV SHA-256 is `be66a66aa4dfec333569a373f3d76ccf9688c01fc99c08fd7055291278353528`. The inventory SHA-256 is `dd71329a767b41042e1c130e6da40dd088828854e64be26483e1527b694841b5` and its GGUF header hash is `f789f7e7179d3fe7f216af7c9b4a2bae45e16cf88fce7f5283159fe33a13ffda`. The [joined machine receipt](../../data/qwen-moe/q8-phase.json) retains call variants and per-width tensor counts. Reproduce it with:

```sh
python3 tools/qwen-moe/q8_phase.py \
  ../../data/qwen-moe/benchmarks/profile-20260923/gpu-host/1267049_kernel_trace.csv \
  ../../data/qwen-moe/traffic.json --tokens 6
```

No GPU was reserved for this analysis. No runtime arithmetic, installed artifact, service or numerical map changed. The better next experiment is an actual-byte and timed-phase panel for the 8192-output Q8 group and one routed expert group on the selected installed runtime, with the same depth and prompt across control/candidate; only after that should another Q8 width/kernel port be chosen.

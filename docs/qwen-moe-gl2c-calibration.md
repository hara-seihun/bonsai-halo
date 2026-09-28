# The gfx1151 read counter counts half a known stream

The installed Qwen counter panel reported roughly half of each packed matrix on every positive matvec dispatch. I tested that number without a model or quantization decoder. A kernel read every 32-bit word of a deterministic, nonuniform GPU-generated input exactly once and stored one sum per lane. Inputs from 16 to 256 MiB straddle the 32 MiB last-level cache. `rocprofv3` reported almost exactly **one half of the bytes the kernel loads** at every size. The unaggregated `GL2C_EA_RDREQ_128B` and the derived `_sum` were numerically identical on all four dispatches.

| Input read by kernel | Sum of four read-size counters | Counter / input |
| ---: | ---: | ---: |
| 16 MiB | 8,389,568 B | .5000572 |
| 64 MiB | 33,555,904 B | .5000219 |
| 128 MiB | 67,108,992 B | .5000010 |
| 256 MiB | 134,217,856 B | .5000005 |

A separate four-launch 128 MiB control and a uniform-input size ladder both gave the same half ratio. Hashing the input rules out the uniform fill as the explanation. The output sums were copied to the host, so the reads were observable. The kernel partitions all `input[i]` over 65,536 threads without overlapping or omitting an index. The reported counter does not independently report instruction-side or physical DRAM bytes. This panel does not independently measure DRAM bytes, so it cannot decide how much of the half ratio comes from metric coverage rather than memory-system behavior. The precise hardware reason for the factor of two is still open; equal raw and `_sum` results rule out simply choosing the metric with the other suffix as a correction.

This resolves one interpretation of the [Qwen matvec counter panel](qwen-moe-q8-native.md). Its positive calls reading 0.499–0.504 of their packed image by this metric are **consistent with one complete image access**, not evidence of 50% sparse access or an 8–10% reread overhead. It does not prove how many Qwen bytes reached DRAM: native reuse, caches and decoder access patterns differ from the streaming control. The [targeted profiler selection panel](qwen-moe-gl2c-attribution.md) repairs the native panel's 395-dispatch zero run for every quantized matvec by narrowing the kernel regex, without changing the installed binary or the four counters. Broad selection still loses those values with graphs disabled and with only one metric. Do not multiply the original two-token aggregate by four and call it physical traffic.

The complete Q8/head call capture with quantized-only selection is in the linked report. The independent next native question is paired, counter-free Q8-8192 phase timing and a *separately calibrated physical DRAM metric* if traffic attribution matters. Compare ordinary occupancy and scheduling before changing arithmetic. The current Q8 port has no measured bandwidth defect to fix on the strength of these counters.

## Reproduction and custody

The GPU was reserved and the resident service paused/restored by `tools/run-batch-compare` for each of three short panels. The final hashed-input panel exited with `bonsai-halo.service` active. The [receipt](../../data/qwen-moe/q8-native/calibration/receipt.json) hashes `bench/qwen_gl2c.hip`, its executable, every raw CSV and wrapper log; it retains all per-dispatch counter values and the earlier uniform-input controls. The main comparison uses the hashed-input CSV in `calibration/hashed/gpu-host/`. ROCm is 7.2.3 on gfx1151. The probe is not a whole-model latency measurement and changes no inference map.

```sh
hipcc -O3 --offload-arch=gfx1151 bench/qwen_gl2c.hip -o ../../data/qwen-moe/q8-native/qwen-gl2c-probe
tools/run-batch-compare --runtime-max 30s --memory-gib 8 --host-reserve-gib 4 --exec \
  rocprofv3 --pmc GL2C_EA_RDREQ_128B GL2C_EA_RDREQ_128B_sum \
  GL2C_EA_RDREQ_32B_sum GL2C_EA_RDREQ_64B_sum GL2C_EA_RDREQ_96B_sum \
  --kernel-include-regex '.*stream_once.*' --output-format csv \
  --output-directory ../../data/qwen-moe/q8-native/calibration/hashed -- \
  ../../data/qwen-moe/q8-native/qwen-gl2c-probe
```

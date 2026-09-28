# Qwen native matvec memory-counter panel

The first interpretation of this counter panel was wrong. The two-token,
no-warmup `rocprofv3` CSV has 862 counted matvec dispatches. Its first 36 have
nonzero read counters, the next **395 consecutive dispatches have zero in every
read-request size counter**, and the final 431 are nonzero. The zero run includes
executed Q8 projections, routed experts and the first Q6 vocabulary head. It
cannot mean those kernels read no external memory. Even in the nonzero stretches,
each quantized family reports almost exactly *half* its one-call packed image.
The original fourfold scale happened to compensate for a roughly half-token
measurement hole and a second factor of two in the positive readings. It is not
a measured physical traffic multiplier. No speedup, memory-traffic conclusion
or numerical-map change follows from this profiler run.

| Kernel family | Positive calls / calls | Positive-call counter / image per call | Old fourfold estimate / image |
| --- | ---: | ---: | ---: |
| Q8 output width 512 | 65 / 120 | 0.4993 | 1.0819 |
| Q8 width 2048 | 86 / 160 | 0.5007 | 1.0766 |
| Q8 width 4096 | 33 / 60 | 0.5004 | 1.1008 |
| Q8 width 8192 | 44 / 80 | 0.5003 | 1.1006 |
| Routed Q4 gate/up | 43 / 80 | 0.5007 | 1.0765 |
| Routed Q5 down | 40 / 74 | 0.5006 | 1.0823 |
| Routed Q6 down | 3 / 6 | 0.5005 | 1.0010 |
| Q6 vocabulary head | 1 / 2 | 0.5035 | 1.0069 |

The table groups calls by quantized kernel type and launch grid. Q8 width 512
has 60 calls/token but 100 tensors because shared gate/up is fused. The Q6
head is separate from routed Q6 down experts. All groups match the expected
call count from the six-token trace, so the zeros are present in the CSV rather
than omitted dispatch records. The Q6 head reports 210.030 MB on the one
positive dispatch against its 417.178 MB image; the other head dispatch reports
zero. The positive-call ratio is 0.499–0.504 in *all eight* independently sized
families. Multiplying positive calls by two would roughly recover one image per
call, but this is an inference from image size, not an independent DRAM meter.

The installed ROCm 7.2.3 gfx1151 metric defines `GL2C_EA_RDREQ_128B_sum` as
`reduce(GL2C_EA_RDREQ_128B,sum)` and defines `FETCH_SIZE` as the weighted sum
of the four size counters. Neither definition prescribes a factor of four. The
[known-stream calibration](qwen-moe-gl2c-calibration.md) reports the same half ratio on hashed 16–256 MiB inputs. Raw `GL2C_EA_RDREQ_128B` equals its `_sum` on each of those dispatches; switching suffixes does not repair the factor. The [narrow-kernel profiler intervention](qwen-moe-gl2c-attribution.md) recovers all 267 missing quantized calls in this hole by selecting only `mul_mat_vec_q` kernels, with the same binary and all four metrics; disabling graphs or dropping to one metric retains the hole under the broad filter. The precise profiler-internal cause remains open. The
old 8–10% excess is explained by 53.7–55% positive dispatches in those groups,
against 50% for the head, not by demonstrated rereads. The counter-timed
Q8-8192 sum of 3.669 ms/token is also not a paired unprofiled phase timing.

The targeted panel resolves missing counter attribution for this quantized-kernel selection; inspect each dispatch in future panels rather than scaling a broad-filter trace. The repaired counter values remain about half the packed image and are not independently calibrated DRAM bytes. They do not rule in or rule out rereads, occupancy or scheduling improvements. A counter-free phase panel remains a separate experiment.

[`q8_native.py`](../tools/qwen-moe/q8_native.py) joins each dispatch's four
read-request size counters with the pinned GGUF tensor inventory. Reproduce the
summary from the [raw CSV](../../data/qwen-moe/q8-native/counters/gpu-host/1690756_counter_collection.csv):

```sh
python3 tools/qwen-moe/q8_native.py \
  ../../data/qwen-moe/q8-native/counters/gpu-host/1690756_counter_collection.csv \
  ../../data/qwen-moe/traffic.json --tokens 2
```

The [receipt](../../data/qwen-moe/q8-native/receipt.json) records the
exact wrapper command, installed source revision `b4c67ced9`, model and
binary hashes. The original [summary](../../data/qwen-moe/q8-native/summary.json)
remains intact; [the corrected audit](../../data/qwen-moe/q8-native/counter-audit.json)
adds the consecutive zero runs, positive-call counts, image-normalized positive
readings and source/CSV/inventory hashes. This CPU-only audit took no GPU lock.
The original panel restored the resident service. No serving, runtime, weight
or numerical map changed.

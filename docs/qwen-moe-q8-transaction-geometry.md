# Qwen Q8 MMVQ: the two loads request the same sectors

The ordinary nonembedding Q8_0 bank is 1,492,910,080 bytes across 250 matrices. In the selected one-row Q8 MMVQ, each wave computes one output row, four lanes consume one 34-byte block (one FP16 scale plus 32 codes), and eight blocks are visited per K iteration. The installed ISA has one 64-bit weight-code load and one 16-bit weight-scale load per lane, then two integer dots and the existing FP32 fold. What fraction of the wide-Q8 cost can be explained by *spatial* transaction amplification, and could a pure layout change remove it?

## Exact sector-request construction

The installed per-wave 272-byte region contains eight `(scale[2], codes[32])` blocks. For lane `t=4b+l`, `b=0..7`, `l=0..3`, code load interval is `[34b+2+8l, 34b+10+8l)`, and scale load interval is `[34b,34b+2)` (four lanes request the same scale). This is the selected source's `qi=8`, `vdr=2`, `blocks_per_iter=8`, `get_int_b2` and `block_q8_0` coordinate, with the 64-bit/16-bit load sizes checked in [the installed ISA census](qwen-moe-q8-isa-resource.md). A sector touched by any active lane is counted once **within that instruction and wave**. The separate code and scale instructions are then added, not magically merged. We also count their union within a wave, and the unique sectors covering each entire tensor. The model does not assign a DRAM miss to each requested sector.

An exact same-byte transform puts the eight scales at `[0,16)` and their 256 codes at `[16,272)` in each 272-byte wave group. A consumer indexes scale `2b` and code `16+32b+8l`, retaining the same eight FP16 scale bit patterns, same 32 signed codes per block and the same block/dot/reduction order. There are no tags, padding, changed weight values or output-map rounding changes in this proposed coordinate. Its CPU sector model is compared at the **same** image size, tensor starts and row strides. This is a structural transformation, not an installed reader.

| Sector width | Installed code + scale requests | Transposed requests | Reduction | Union per wave, both layouts | Unique full-tensor coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 B | 2,985,820,160 B | 1,668,546,560 B | **44.12%** | 1,580,728,320 B | 1,492,910,080 B |
| 64 B | 3,337,093,120 B | 2,019,819,520 B | **39.47%** | 1,756,364,800 B | 1,492,926,080 B |
| 128 B | 4,039,639,040 B | 2,722,365,440 B | **32.61%** | 2,107,637,760 B | 1,492,942,080 B |

The reduction is exactly 1,317,273,600 modeled request bytes at every granularity: the scales become one or two neighboring sectors per wave rather than eight scattered scale sectors, while the codes become contiguous. At 32 B, installed code requests already cover **all scale-containing sectors**: the installed separate-request sum is 2.0000× the image but its per-wave merged coverage is only 1.05882× and its whole-tensor unique coverage is exactly 1.0000×. In particular, the 2× sum is **not** evidence of 2× physical DRAM traffic. The 128-B unique excess is only 32,000 bytes, the 250 tensor-edge sectors.

The [source and inventory-hashed receipt](../../data/qwen-moe/q8-transaction-geometry/receipt.json) contains each input-K group's counts at all three sector widths, all 250 installed tensor offsets and byte sizes via `traffic.json` header hash, and the exact scalar enumeration. Reproduce without loading the 22-GB image or using the GPU:

```sh
python3 tools/qwen-moe/q8-transaction-geometry.py \
  --output ../../data/qwen-moe/q8-transaction-geometry/receipt.json
```

Source inputs: installed `traffic.json` SHA-256 `dd71329a767b41042e1c130e6da40dd088828854e64be26483e1527b694841b5`; native `mmvq.cu` SHA-256 `976e35e481eb9db864b1384de5c744ab9a37fb89fa62bd9205919f4b03cabd6c`, `vecdotq.cuh` SHA-256 `491a5169d2d86c4f54136ab829c9450cfef89b7d3c79ae5f60004e7ebf3b376f`. The receipt SHA-256 is `d5cfd3fa70e19c899e718242018377b79d86b5161266050cb0649469642c7b14`.

## Cost boundary and next experiment

The reduction is **instruction-side requested-sector capacity** under a stipulated sector coalescer, not measured physical traffic or a speedup. Cache hits, replay, transaction merging, counters' unknown calibration, routing and occupancy all stand between this count and latency. In the opposite extreme where cached neighboring requests are free, the layout saves no DRAM bytes at all; it also cannot explain an ordinary Q8 wide-call deficit solely by compulsory traffic. A native transposed reader would pay a changed address expression and potentially different waits/occupancy, plus an offline one-time image rewrite; only a full-head installed binary panel can decide whether reducing load requests has a net benefit. The useful next measurement is a matched native same-byte AoSoA reader at widths 512/2048/4096 with compiled ISA request count and register census, exact FP32 heads/state on selected acceptance and whole-model plain/prompt/batched timing. If the hardware merges/caches the original scale requests cheaply, this model predicts a null despite the 44% instruction-side count. No runtime, model image, GPU service or serving default changed here.

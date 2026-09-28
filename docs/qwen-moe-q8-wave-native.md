# Phase-optimal same-byte Q8 wave layout loses in a native reader

**Question.** The [fixed-wave sector proof](qwen-moe-q8-multisector-opt.md) shows that alternating code-first and scale-first 272-byte groups reduce stipulated separate code/scale request sectors across the 1.493-GB installed ordinary Q8 bank. Does that address permutation alone buy elapsed time when compiled for gfx1151? The earlier [global-bank split reader](qwen-moe-q8-layout-native.md) lost, but its two distant streams and separate allocations could be responsible. This panel compares **two equally sized, single-allocation images** instead.

`bench/qwen_q8_wave.hip` runs a deterministic model-less Q8_0 reader. One wave owns a K8192 row (32 groups × 8 Q8 blocks); each lane reads one signed code from each block and the designated lane broadcasts its original FP16 scale. It performs the same ordered per-lane FMA and butterfly fold in both compiled arms. The control places eight 34-byte installed-style Q8 blocks consecutively. The candidate permutes each 272-byte group into 256 code bytes and 16 scale bytes, codes first at even group phases and scales first at odd phases. Because 272 ≡ 16 (mod 32), each group starts at the appropriate phase when the allocation base is aligned. It adds **zero bytes**; initialization writes the same code and scale bits. Unlike the global split, the candidate has only one weight stream. This is a standalone reader, **not** llama.cpp's selected MMVQ with its Q8_1 activation layout, graph dispatch or actual model weights.

All three panels compare every output float bit (zero differences) and alternate arms seven times within the same wrapper invocation. Median device-event times:

| K8192 rows / image bytes per arm | Interleaved ms | Phase-aware ms | Candidate / control | Wrapper clock p50 |
| ---: | ---: | ---: | ---: | ---: |
| 8,192 / 71,303,168 | .723488 | .730647 | 1.00990 | 631 MHz |
| 171,520 / 1,492,910,080 | 14.994012 | 15.232854 | **1.01593** | 689 MHz |
| 171,520 / 1,492,910,080, requested fixed 2400 MHz | 16.011330 | 16.314030 | **1.01891** | 1063 MHz |

The clock request did not deliver a 2400-MHz median, and host load was nonquiet. Alternating arms hold the contemporaneous comparison; this panel cannot establish a controlled-clock hardware optimum. At the complete Q8-bank size, both readings stream about 1.5 GB in 15–16 ms, and the byte-permutation arm is consistently slower. The modeled separate-instruction sector-count reduction therefore **does not convert into a benefit for this same-allocation reader**. This complements the global-split native loss: neither abolishing the 272-byte group boundary nor retaining it with phase parity has yet improved a compiled reader. It does *not* prove that the selected MMVQ cannot benefit from a different layout, nor that physical DRAM traffic tracks requests. In particular these event times are not full-model plain, prompt, or batched TPS, and no runtime or numerical default was changed.

Next native Q8 work should instrument the actual selected graph-on MMVQ and its host/device critical path with complete-head/state controls, rather than port this unselected layout on sector arithmetic alone. Representation work still needs actual routed producers and complete paid quality. The [raw wrapper logs, executable, assembly attempt, hashes and machine-readable receipt](../../data/qwen-moe/q8-wave-native/receipt.json) are in data custody; the wrapper restored `bonsai-halo.service` after every panel.

Reproduce from Bonsai root (one bounded panel per wrapper command):

```sh
hipcc -O3 --offload-arch=gfx1151 -o ../../data/qwen-moe/q8-wave-probe bench/qwen_q8_wave.hip
tools/run-batch-compare --runtime-max 38s --exec ../../data/qwen-moe/q8-wave-probe 171520
python3 tools/qwen-moe/q8-wave-native.py
```

# Phase-aware Q8 wave layout reaches the 32-byte request lower bound

The [selected ordinary Q8 MMVQ geometry](qwen-moe-q8-transaction-geometry.md) reads eight 34-byte blocks per wave: 256 signed code bytes and sixteen FP16 scale bytes. Its proposed fixed scale-first transpose cuts separate code/scale 32-byte-sector requests 44.12%, but leaves one redundant requested code sector in every other 272-byte group. A phase-aware **same-byte** construction removes it without padding, extra weight storage or changing a dot or FP32 fold.

## Construction and proof

Each wave group is 272 bytes. Its start alternates between offsets 0 and 16 modulo 32 in the installed 32-byte-aligned Q8 tensor inventory (including row stride). At phase 0, put all 256 codes first and sixteen scales last; codes then occupy exactly eight 32-byte sectors, scales one. At phase 16, put sixteen scales first and 256 codes last; the codes begin at the next sector boundary and occupy exactly eight sectors, and scales occupy one. The individual scale of block `b` and the four eight-code-byte slices for its four lanes are selected at offsets `scale_base+2b` and `code_base+32b+8l`. They are exactly the original FP16 bits and signed codes; neither block order nor the integer-dot/FP32-fold order changes. A 272-byte byte-permutation round trip for both layouts is checked in the receipt generator.

For **the separate code-load and scale-load instruction sector-union grammar**, at least 256 distinct code bytes must be covered by code requests and sixteen distinct scale bytes by scale requests. Therefore any unpadded layout of these bytes requires at least `ceil(256/32)+ceil(16/32)=9` requested sectors *per wave*, regardless of how codes and scales are interleaved. The construction reaches nine for both observed phases and hence is optimal **in this grammar**. This is neither an ISA-independent time bound nor a proof of DRAM transactions: caches can supply both instructions from one sector. The geometry model counts a sector only once within each instruction/wave; across instructions and waves it adds requests. Its unchanged per-wave union is nine sectors, and the unchanged complete-image unique coverage is 1,492,910,080 bytes.

| Modeled sector | Installed separate requests | Uniform scale-first | Phase-aware | Improvement vs installed | Improvement vs uniform |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 B | 2,985,820,160 B | 1,668,546,560 B | **1,580,728,320 B** | **47.06%** | **5.26%** |
| 64 B | 3,337,093,120 B | 2,019,819,520 B | 1,932,001,280 B | 42.11% | 4.35% |
| 128 B | 4,039,639,040 B | 2,722,365,440 B | 2,634,547,200 B | 34.78% | 3.23% |

[The strengthened full-sector proof](qwen-moe-q8-multisector-opt.md) shows that the same phase-parity layout also attains the 64/128-B floors **within the fixed unpadded 272-byte group and separate-instruction request grammar**. Their naive byte-count bounds are unattainable on unfavorable phases because too few complete aligned sectors lie inside the group. This does not bound layouts that repack across group boundaries. All 5,488,640 wave groups and 250 ordinary nonembedding Q8_0 tensors are included. The 32-B phase distribution is exactly 2,744,320 groups at each phase; code bytes cover 1,405,091,840 modeled request bytes and scales 175,636,480. At 32 B the lower bound equals the phase-aware total. At 64/128 B the simple per-instruction lower bounds do **not** coincide with the measured candidate, so no optimality is claimed there.

Reproduce without the GPU or reading the 22-GB model:

```sh
python3 tools/qwen-moe/q8-sector-opt.py \
  --output ../../data/qwen-moe/q8-sector-opt/receipt.json
```

[Receipt](../../data/qwen-moe/q8-sector-opt/receipt.json) SHA-256 `fd44e7076351d00eb6d8172d73647d895caaef10be605cb987433c53a93cc63a` includes source, preceding geometry source, inventory and GGUF header hashes, phase counts and per-instruction sector totals. The model count assumes the tensor device base preserves the stated sector alignment; a native reader must check its actual allocated addresses. The phase bit can be computed from group/row position, but its address selection, branch or predication, code/scale load shape, register lifetime and cache behavior remain **unpriced**. A layout change must retain the exact installed full-head/state map and win matched plain generation, prompt and batch panels before adoption. The recent [GL2C counter calibration](qwen-moe-q8-dram.md) cannot turn these stipulated requests into physical DRAM bytes. No runtime/model image, GPU, TPS, or resident service changed.

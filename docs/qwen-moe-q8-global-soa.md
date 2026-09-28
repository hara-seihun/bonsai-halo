# Qwen Q8 global split banks remove the wave-local phase penalty

[The wave-local 272-byte proof](qwen-moe-q8-multisector-opt.md) closes *reordering within each eight-block wave group*: its existing phase-aware permutation reaches the stipulated 32/64/128-byte separate-load request floor under that boundary. Removing the unnecessary boundary gives a cheaper exact **whole-bank coordinate** in the same cost model. Put every selected ordinary Q8 block's 32 signed code bytes in one contiguous code bank and every two-byte FP16 scale in a second bank, in tensor/row/block order. The device banks start aligned to 128 bytes. No block is expanded, padded or changed; lookup uses `code_base[t] + 32*block` and `scale_base[t] + 2*block`. The same eight blocks and the same per-block dot, scale and FP32 fold remain available to the selected one-row MMVQ consumer.

## Exact construction and cost

All 250 installed ordinary nonembedding Q8 tensors have both their code and scale sub-bank lengths divisible by 128, so concatenating them without **any logical padding** preserves 128-byte alignment of every tensor's two sub-bank starts. The complete image remains **1,492,910,080 bytes**: 1,405,091,840 code and 87,818,240 scale bytes. A five-wave byte-permutation round trip checks the indexing, including consecutive group boundaries; the pinned inventory check covers all group and tensor lengths. Actual native device allocation must guarantee the stipulated base alignment; the GGUF file's original tensor offsets are *not* the proposed device addresses (they start at 96 modulo 128 in this inventory).

A wave's 256 code bytes now start sector-aligned at 32, 64 and 128 bytes, occupying exactly 8, 4 or 2 code sectors. Its sixteen adjacent scale bytes fit one separate sector, including when four/eight neighboring waves share that physical sector. Thus this single no-padding coordinate reaches the absolute `ceil(256/W)+ceil(16/W)` **per-wave separate-instruction capacity floor** at all three widths; the wave-local bound's 64/128 phase penalty is not a fundamental code/scale information cost.

| Request width | Installed separate requests | Best 272-B wave-local | Global split banks | Global vs wave-local |
| --- | ---: | ---: | ---: | ---: |
| 32 B | 2,985,820,160 B | 1,580,728,320 B | **1,580,728,320 B** | 0% |
| 64 B | 3,337,093,120 B | 1,932,001,280 B | **1,756,364,800 B** | −9.09% |
| 128 B | 4,039,639,040 B | 2,634,547,200 B | **2,107,637,760 B** | −20.00% |

This proof fixes the observation grammar to code and scale load instructions on one selected wave, counts the union of requested `W`-byte sectors separately for each instruction, and sums across waves. It permits arbitrary whole-bank byte permutation but requires exactly the original code/scale values at the reader; it proves neither minimum device time nor physical DRAM bytes. A global bank creates **two far-apart memory streams**, changes per-tensor address arithmetic and runtime repacking, and could lose cache locality or add waits despite fewer instruction-sector requests. The original and split image have the same compulsory unique bytes and conditional complete-model one-read weight budget. Scale sectors shared across neighboring waves may hit cache, making this request count a poor physical-traffic estimate. Native ISA/register inspection, allocation alignment and matched complete-head/state plain/prompt/batched panels are necessary before selection. At 32-byte sectors it offers no modeled gain over phase-aware wave-local packing, but has a simpler phase-free address rule; at wider sectors it supplies a strictly better structural candidate than another within-wave permutation search.

Reproduce on CPU without loading weights or reserving the GPU:

```sh
python3 tools/qwen-moe/q8-global-soa.py \
  --output ../../data/qwen-moe/q8-global-soa/receipt.json
```

[Receipt](../../data/qwen-moe/q8-global-soa/receipt.json) SHA-256 `e8b1cc3d4f2419969e148aee557d0e00b6a9823ab104e0e053a5631e739f0b22` retains source/inventory/header hashes, all 250 tensor alignment checks and request counts. No GPU, model image, runtime, service or measured TPS changed.

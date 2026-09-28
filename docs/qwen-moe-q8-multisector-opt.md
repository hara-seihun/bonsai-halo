# One phase bit attains the Q8 separate-request floor at 32, 64 and 128 bytes

The [phase-aware Q8 wave construction](qwen-moe-q8-sector-opt.md) previously proved optimal only at 32-byte sectors. Its reported 64/128-byte totals are also **lower bounds**, not merely achievable counts. The *same* physical byte permutation for each 272-byte wave group attains all three floors simultaneously: code-first when its start is 0 mod 32, scale-first when it is 16 mod 32. No width-specific image or padding is required.

## Bound and witness

The domain is one fixed, unpadded 272-byte group of eight Q8_0 blocks, holding 256 distinct signed-code bytes and sixteen distinct FP16-scale bytes. Its start has phase `p` mod sector width `S`; here `S` is 32, 64 or 128 bytes and observed `p` is divisible by 16. The observation contract asks separate code and scale instructions to request every original byte and counts the union of sectors *within each instruction/wave*, adding the two counts. Bytes may be permuted arbitrarily **within their existing group**. There is no restriction to a contiguous code or scale region in the lower bound.

A code instruction needs at least `256/S` sectors. Equality needs **all** bytes of each of those sectors; since the group contains exactly 256 distinct codes, each chosen sector must lie fully inside `[p,p+272)`. The number of full sectors in that interval is

```
F(p,S) = floor((p+272)/S) - ceil(p/S).
```

Thus the code lower bound is `256/S + [F(p,S) < 256/S]` sectors. A separate scale instruction needs at least one sector, so add one. This is an actual per-group minimum, not just a loose bound: the code-first/scale-last or scale-first/code-last construction attains it at every observed phase. The short [CPU certificate](../tools/qwen-moe/q8-multisector-opt.py) checks every matrix's actual phase and actual eight-block reader slices against this independent full-sector bound; it asserts the *same* parity decision across all three widths. The prior construction checks that every scale/code byte reconstructs unchanged, preserving block and dot/fold order.

| Width | Favorable phases with complete aligned code region | Minimum at favorable / other phases | Whole-bank minimum separate requests | Installed | Reduction |
| --- | --- | --- | ---: | ---: | ---: |
| 32 B | 0, 16 | 9 / 9 sectors | **1,580,728,320 B** | 2,985,820,160 B | 47.06% |
| 64 B | 0, 48 | 5 / 6 sectors | **1,932,001,280 B** | 3,337,093,120 B | 42.11% |
| 128 B | 0, 112 | 3 / 4 sectors | **2,634,547,200 B** | 4,039,639,040 B | 34.78% |

Across 250 installed ordinary nonembedding Q8 matrices (1,492,910,080 bytes), each of the four 64-byte phases occurs **1,372,160** times and each of the eight 128-byte phases **686,080** times, for 5,488,640 wave groups. The basic `ceil(256/S)+ceil(16/S)` bound had left 175,636,480 B at 64 and 526,909,440 B at 128 unexplained. The fixed 272-byte footprint and its starting phase make those bytes **unavoidable in this separate-instruction sector model**: at the unfavorable phases the group contains too few complete sectors to hold all 256 code bytes in the naive number of requests. Repacking across group boundaries, padding, a combined instruction, or cache reuse is outside this grammar and may change the bound.

Reproduce in seconds without GPU or reading weight payloads:

```sh
python3 tools/qwen-moe/q8-multisector-opt.py \
  --output ../../data/qwen-moe/q8-multisector-opt/receipt.json
```

The [receipt](../../data/qwen-moe/q8-multisector-opt/receipt.json) includes source/inventory/header hashes, every phase count and separately checked code/scale totals. It assumes the selected tensor allocation preserves the GGUF data-base phase; the native device base is not yet checked. Requested sectors are **not** physical DRAM reads; the [counter calibration](qwen-moe-q8-dram.md) cannot resolve that distinction. There is no native reader, measured Q8 kernel gain, full-model TPS or model-quality change. The next experiment is still a native same-byte reader with compiled address/register census and matched full-head/state plain, prompt and batched panels. A larger sector-only theoretical saving cannot be obtained by another within-group permutation under this cost grammar; test whether the proven layout is useful on hardware, or change the instruction/group-boundary grammar.

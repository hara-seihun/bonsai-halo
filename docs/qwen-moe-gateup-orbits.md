# Qwen routed gate/up packet reuse is only zero-block reuse on sampled real routes

A single post-attention activation is shared by the selected eight experts' gate and up projections. We asked whether the installed Q4_K image exposes an additional **shared integer input dot**, either between output rows of one projection or across gate and up. Across three actual selected routes (tokens 1, 32 and 63) in each of two disjoint 64-token texts, at **all forty layers**, no two eight-byte Q4_K code packets at the same input coordinate have equal bytes **when both weight blocks have nonzero Q4_K scale or minimum metadata**. This includes cross-expert and cross-gate/up matches. It closes a direct equality-keyed packed-dot cache for these observed routes; grouped dispatch already shares activation preparation.

The comparison uses every one of the 512 output rows for each selected expert in each gate and up bank, all eight input Q4_K blocks per row and all sixteen eight-byte code packets per block. Each packet contains two nibble-coded input subsequences. Matching its eight code bytes at the same block and packet position is necessary to reuse both integer-dot pieces; different Q4_K scale/minimum metadata would still require separate affine FP32 folds. All selected Q4_K banks are read from the pinned GGUF. A free dictionary, lookup, summation, code-byte skipping and output fanout are granted in the byte ceiling below. No native bit-identical FP32 consumer or inference speed follows from code equality alone.

| Two 3-token route panels × 40 layers | Train | Held |
| --- | ---: | ---: |
| Gate/up packet uses | 125,829,120 | 125,829,120 |
| Raw identical-code uses after first same-K packet | 676,341 | 786,898 |
| Of these, newly shared between gate and up | 91,537 | 93,073 |
| Packet uses in blocks with nonzero scale or minimum | 124,554,752 | 124,499,968 |
| Paired entirely zero gate/up output rows | 4,978 | 5,192 |
| Zero Q4_K blocks outside those paired rows | 0 | 0 |
| **Duplicate same-K packets among active blocks** | **0** | **0** |

The tempting raw matches are not active common subexpressions: every zero-scale/minimum block in these routes belongs to an **entire paired zero gate/up output row**, and all raw repeats disappear when those blocks are excluded. The raw byte value need not be zero when the *decoded weight block* is zero, so filtering only zero-valued code bytes would falsely retain 786,887 held duplicates. The already documented [dormant Qwen experts](qwen-moe-all-layer-dormancy.md) own that opportunity; adding raw code-packet sharing to their byte savings would double count it. The held free raw-byte saving would be at most `786898 × 8 / 120 = 52,459.87` bytes/token, **0.001998%** of the conditional 2,626,187,904-byte complete one-read stream, even before indexing, load granularity, scaling or dot costs. Excluding the zero-weight blocks leaves **zero incremental code-byte savings** for this exact eight-byte same-coordinate reuse grammar on the measured routes.

This is a finite installed-weight/actual-route result, not a proof about all generated routes or every expert combination. It excludes shorter fragments, proportional nonidentical codes, a changed learned expert image, different producer encodings and output-observer approximations. A code match only lets an integer dot potentially share; native FP32 accumulation identity would additionally need a checked fold order. The prior [three-layer static survey](qwen-moe-code-sharing.md) saw low-code coincidences in layer 0; the scale/minimum join explains why retaining those coincidences as new useful compute is misleading. Representation work should instead learn changed paid directions on broad actual producers and measure held complete-model language quality. The independent exact-engine question remains unprofiled router/Q8/expert device time with complete requested heads, not another gate/up packet dictionary.

## Custody

[The receipt](../../data/qwen-moe/gateup-orbits/receipt.json) binds all eighty per-layer tensor hashes and six selected route IDs per layer to the GGUF inventory and model pin, records per-route raw, active and cross-bank counts, and hashes the analysis source. Reproduce the CPU census from Bonsai with `python3 tools/qwen-moe/gateup_orbits.py --first N --count M` in bounded layer ranges followed by `python3 tools/qwen-moe/gateup_orbits.py --summarize`. Neither the installed Qwen runtime, model, GPU reservation nor resident service changed. This is a conditional byte/structural negative, not a whole-model TPS result.

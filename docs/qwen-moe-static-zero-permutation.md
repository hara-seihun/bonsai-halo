# Expert-specific column clustering and routed down zeros

The native layer-0 Q8_1 post-SwiGLU down operand has **no** all-zero 32-coordinate blocks in 1,008 held routed expert slots, despite 39.6% zero floats. This study asks a different question: could one **static column permutation per expert**, fitted on the separate 904 train slots, cluster exact producer zeros into whole operand blocks? It is an optimistic structural test of a new representation, not a bit-exact rearrangement of the installed Q5_K/Q8_1 arithmetic.

A deterministic layout sorts an expert's 512 columns by increasing train nonzero count, then their train activity signature, then column number. Experts absent from train retain the identity layout. It uses only zero masks; no held input, score, weight or down output enters the layout. A free per-slot hindsight upper bound packs all original zeros first, giving `floor(zero_floats/32)` blocks. Each slot has sixteen blocks.

| Layer-0 slots | Installed order | Train-fitted expert order | Free per-slot hindsight order |
| --- | ---: | ---: | ---: |
| Train, 904 slots: zero 32-blocks | 0 | 5,091 (35.20% of blocks) | 5,091 |
| Held, 1,008 slots: zero 32-blocks | 0 | **5,304 (32.89%)** | 5,916 (36.68%) |
| Held slots with at least one zero block | 0 | 900 | 976 |
| Conditional whole-model one-read bytes removable, if all forty layers match | 0 | **2.889%** | 3.222% |

Seventy-six held slots select experts never present in train, so their installed-order fallback explains part of the 612-block gap to the free oracle. The train layout exhausts its per-slot oracle on *train*, but not on independent text: that is precisely why train zero coverage alone cannot select a serving image.

The byte column grants **free** detection, a selectively readable down image and zero cost to permute activations or weights. It applies the observed zero-block share to eight selected down images per layer (`5,767,168` bytes), forty layers, and the documented `2,626,187,904`-byte conditional complete one-read stream. The down image is only 8.78% of that stream. The oracle's 3.222% is a strict upper bound on the modeled byte prize for *any* permutation that only groups these held original zeros, even one chosen afresh for each slot. Neither number predicts DRAM traffic or speed. Sparse metadata, routed gathers, divergent dot loops, packed block-scale reads and repacking spend part of the prize.

An exact-map restriction matters: permuting intact existing groups of 32 leaves every group nonzero, so yields **zero** new skippable Q8_1 blocks. Reordering individual columns changes which floats share a Q8 scale and original-sum correction and changes which packed Q5_K scale applies to each weight. It cannot silently replace the installed FP32 logits. A new sparse image and consumer would need to pay image bits, activation remap, quality on held complete-model text and native prompt/decode time. Even if quality survives, the generous byte ceiling makes steady Q8 and gate/up work higher-priority than a down-sparsity-only port. Conversely the 32.89% held local zero-block share is a real structural reason to test **joint** packed layouts rather than extrapolate the installed-order zero count as an impossibility claim.

The [source](../tools/qwen-moe/static_zero_permutation.py) reads actual selected IDs and hidden producers from the [route capture](qwen-moe-routes.md). The [receipt](../../data/qwen-moe/activation-zero-blocks/static-permutation.json) retains source and input hashes, complete per-token block counts, all 256 train/held expert frequencies and both bounds; SHA-256 `bc93c82f10a692c64c3b900ab307fadb82efe331f97e384d8c388d94ed5b0a72`. Reproduce without a GPU reservation:

```sh
python3 tools/qwen-moe/static_zero_permutation.py \
  --output ../../data/qwen-moe/activation-zero-blocks/static-permutation.json
```

The useful next experiment, if investing in a joint sparse format, is to hold weight rate fixed, remap *actual* quantized expert inputs and Q5_K weight coordinates together, then compare complete-model held loss against the installed image before pricing a native selective-read consumer. This result alone does not earn a runtime arm. No GPU, model image, installed executable or resident service changed.

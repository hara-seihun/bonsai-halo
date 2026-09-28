# Exact zero-block opportunities in routed Qwen activations

The post-SwiGLU routed hidden vectors look sparse: 204,177 of 516,096 held layer-0 floats are exactly zero. That does **not** give the selected Q5_K down consumer a sparse 32-coordinate operand. None of its 16,128 held Q8_1 blocks is zero. The gate/up input has no exact-zero floats on either capture.

The native `quantize_q8_1` kernel takes a maximum over each 32-float group, sets `d = amax / 127`, rounds each `x/d` to an int8 code, and stores the original group's sum in `ds.y`. For a finite nonzero group in the measured range, a maximal-magnitude entry has a nonzero code, close to magnitude 127. Thus a complete Q8_1 block cannot vanish merely because many of its entries are zero. A group of 32 original zeros can vanish, including its affine correction; none occurs in the captures. This is a structural restriction of the installed quantizer and the exact operand-block skipping grammar, not a lower bound on a different code or on direct packed computation.

| Actual layer-0 input | Split | Floats exactly zero | Original-zero 4-tuples | Q8-zero 4-tuples | Original-zero 16-tuples | Original-zero 32-tuples |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Shared gate/up, width 2048 | train, 113 tokens | 0/231,424 | 0/57,856 | 1/57,856 | 0/14,464 | 0/7,232 |
| Shared gate/up, width 2048 | held, 126 tokens | 0/258,048 | 0/64,512 | 0/64,512 | 0/16,128 | 0/8,064 |
| Routed down, eight 512-wide hiddens per token | train | 176,260/462,848 | 4,417/115,712 | 5,339/115,712 | 0/28,928 | 0/14,464 |
| Routed down, eight 512-wide hiddens per token | held | 204,177/516,096 | 5,861/129,024 | 6,894/129,024 | 6/32,256 | 0/16,128 |

The 4-tuple count bounds a more aggressive exact-value dot-skipping idea. Only 4.54% of held down-input tuples are identically zero before quantization. Q8-zero tuples rise to 5.34%, but the Q4_K/Q5_K affine term uses the **original group's sum** stored in Q8_1. A zero code tuple is not by itself permission to omit the affine contribution or to change the accumulation order. The source has a `dp4a` per packed tuple, not a free sparse-index lookup. Even granting free tests and a weight layout that avoids the corresponding fraction of Q5_K down image bytes, repeating the held layer-0 rate at all forty layers saves at most `40 * 5,767,168 * 5,861/129,024 = 10.48 MB` of the conditional 2,626.19 MB/token whole-model one-read stream, **0.399%**. Actual packed block scales and nibble placement do not offer this selective byte read for free. The native kernel always reads its dense image; the measured byte saving is zero. For an arithmetic-only skip, branch, mask, divergence and numerical fold costs still need pricing.

The conclusion is specific. A zero-*block* fast path for the selected Q8_1 gate/up or down kernels has zero opportunity on these real routed producer values. Four-wide sparsity may be useful inside a jointly designed new code and consumer, but its upper bound in the unchanged image is too small to justify a sparsity-only Q5_K port. Target image-rate reductions or ordinary packed operand work instead. This result says nothing about complete-model speed or a changed numerical map; no GPU, weights, executable or service changed.

`tools/qwen-moe/activation_zero_blocks.py` reads the [captured real producer vectors](qwen-moe-routes.md), counts original and simulated Q8_1 zeros, and saves source, input, installed-runtime and acquisition hashes to [the receipt](../../data/qwen-moe/activation-zero-blocks/receipt.json). Both installed `quantize.cu` and `vecdotq.cuh` have the same SHA-256 as the pinned upstream checkout that the script hashes. Reproduce without reserving the GPU:

```sh
python3 tools/qwen-moe/activation_zero_blocks.py \
  --output ../../data/qwen-moe/activation-zero-blocks/receipt.json
```

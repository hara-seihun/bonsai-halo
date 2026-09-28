# Rounded-zero down inputs do not create a Qwen expert-byte shortcut

The actual selected-GGUF post-SwiGLU producer arrays from two disjoint 64-token prompt prefixes cover eight routed experts at every one of forty layers. `tools/qwen-moe/rounded_down_zero.py` applies the installed MMQ down-input **code** rule per 32 floats: `d_inv = 127/max(abs(x))`, then `roundf(x*d_inv)` (CPU float32, signed half-away-from-zero). This differs from dividing by `max(abs(x))/127`; the reciprocal rule is the one used by the selected Q5_K/Q6_K expert MMQ. The callback captures cut the graph, so these are observed callback producers, not an exact no-cut GPU quantizer replay.

| Observed split | Q8_1 zero code bytes / 10,485,760 | All-zero four-code groups / 2,621,440 | New groups beyond original FP32 zeros | All-zero 8 / 16 / 32-code groups | Free four-coordinate byte comparator / whole one-read stream |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 496,786 | 3,033 | 588 | 130 / 1 / **0** | 266,972 B/token / **0.010166%** |
| Held | 496,587 | 3,410 | 598 | 150 / 1 / **0** | 300,216 B/token / **0.011432%** |

On held, **3,296/3,410** four-code zero groups belong to layer 0, rather than representing a repeatable forty-layer sparsity pattern. The previous original-FP32-zero census counted 2,812 held four-tuples and a 247,456 B/token ideal comparator. Allowing Q8 quantization to make *another 598* four-tuples zero raises this optimistic comparator only 52,760 B/token. A nonzero finite 32-float input block necessarily codes its maximal element to a nonzero int8 value; no complete Q8_1 block is zero in either observed split.

The comparator grants free testing, independent four-coordinate addressing and uniform one-128th of the expert's entire Q5_K/Q6_K down image per zero four-code group, averaged over 64 tokens and eight routes. Its denominator is the installed complete-model conditional one-read weight stream of 2,626,187,904 B/token. **It is not saved native traffic or a speedup.** In 37 Q5_K layers the dot's code-dependent term vanishes for a four-code zero group, but the affine minimum term uses the **original 32-float sum** in the FP16 DS4 operand and must still be computed. On held, 3,402/3,410 candidate four-groups are Q5_K: zero codes alone do not allow deletion of their full weight contribution. The three Q6_K layers have no affine sum, but just eight held zero four-groups, worth at most **840 B/token** even under this free independent-slice layout. The packed Q5_K/Q6_K images actually share code and scale bytes over larger K groups, and the selected dense reader does not omit any: measured native byte or time saving is **zero**. Skipping an integer zero dot term in a newly lowered kernel would need a separately priced consumer and bitwise FP32 acceptance.

This is a bounded negative for **unchanged** MMQ Q8_1 codes and a hypothetical independently readable four-coordinate image on these prompt producers. It does not constrain generated-token producers, changed learned directions, cross-layer joint maps, or a newly paid expert image with complete-model held loss. The stronger next engine question is unprofiled Q8/expert physical transactions and wall phase timing with full heads; representation work needs genuinely changed packed expert directions trained on broad actual routes, with paid forty-layer image and held language quality rather than another zero mask.

[The per-layer and input-hashed CPU receipt](../../data/qwen-moe/all-down-zero/rounded-zero-receipt.json) binds the selected runtime metadata, model pin, native quantizer/dot/MMQ sources, traffic inventory and prior [actual-producer capture](../../data/qwen-moe/all-down-zero/README.md). Regenerate from this repository with:

```sh
python3 tools/qwen-moe/rounded_down_zero.py \
  --output ../../data/qwen-moe/all-down-zero/rounded-zero-receipt.json
```

No GPU, installed Qwen executable, Bonsai service or serving numerical default changed.

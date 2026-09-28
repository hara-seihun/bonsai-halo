# Actual forty-layer routed down operands close the native zero-block shortcut

The existing layer-0 post-SwiGLU capture had about 40% exact-zero entries but no all-zero Q8_1 32-coordinate down operands. To test whether later layers revive a direct packed skip, `tools/qwen-moe/capture_all_producers.cpp` now also records the actual `[token, selected expert, 512]` post-SwiGLU tensor in **every one of the forty layers**. On each of two disjoint 64-token real-text prefixes, the other four tensors (producer, IDs, scores and down outputs) and token IDs reproduce the prior capture **byte-for-byte at every layer**. The new layer-0 hiddens reproduce the corresponding first 64 tokens of the older full-text capture. These are selected GGUF/native callback producers, not reconstructed original BF16 values. The callback is a graph cut, not a native speed panel or complete-model quality evaluation.

| Split | Exact-zero FP32 hidden elements / 10,485,760 | Exact-zero 4-tuples / 2,621,440 | 8-tuples | 16-tuples | 32-float Q8_1 blocks / 327,680 | Free four-tuple bytes/token / whole one-read fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 104,689 | 2,445 | 94 | 1 | **0** | 215,160 / **0.008193%** |
| Held | 109,419 | 2,812 | 107 | 1 | **0** | 247,456 / **0.009423%** |

Of the **2,812** held zero four-tuples, **2,779 occur at layer 0**, 32 at layer 1, and one at layer 2; no other layer has one. Among train observations, 2,413 occur at layer 0 and 32 at layer 1. Exact-zero FP32 entries across all layers are only 1.04% on held, rather than layer 0's ~39% on its longer earlier capture. The complete installed routed-down bank costs 234,029,056 bytes per token when eight experts are selected in each layer (37 Q5_K and three Q6_K), out of 2,626,187,904 conditional complete-model one-read weight bytes per token.

**Exact-block result.** The selected quantizer encodes one block of 32 finite FP32 hidden values as Q8_1, with `d = max(abs(x))/127` and rounded codes, alongside the original block sum for affine correction. A block with any nonzero maximal-magnitude element necessarily has a nonzero code at that position; an all-original-zero block has zero codes and zero sum. Thus on these captured hiddens no complete native Q8_1 block has a zero dot **and** zero correction that an exact whole-block test could skip. This holds irrespective of routing order and allows the test and skip to cost zero; there are no successful tests. It is a finite observed-domain negative, not an impossibility proof for changed producer encodings or generated-token distributions.

**More generous finer-grain cost bound.** Give a new consumer free tests, a freely changed column layout, separate four-coordinate weight reads and no dispatch or accumulator cost. Assign each original-zero four-tuple one 128th of its selected expert's whole Q5_K/Q6_K down image; average each layer's observed count over 64 tokens and eight slots. This grants 247,456 skipped weight bytes/token on held, **0.009423%** of the complete one-read weight comparator, and 215,160 / 0.008193% on train. It is a conditional idealized *uniform independent four-tuple image grammar*, not a hard upper bound on arbitrary changed representations. In the installed image, Q5_K/Q6_K scales and packed bytes are shared across larger blocks and the actual kernel reads the dense image, so the achieved saved bytes and measured speedup are **zero**. A Q8-zero-code four-tuple is not automatically skippable: Q8_1 also stores the original group's sum for the affine term, and changing the folded FP32 order changes the exact map. No free new image was built, and no approximate representation quality was evaluated.

The earlier layer-0-only analysis extrapolated its held 4-tuple fraction to all forty layers and quoted **0.399%** conditional bytes under the same generous grammar. The actual full-stack weighted bound is only **0.009423%**, over forty times smaller. This closes zero-block special casing of the selected routed-down Q8_1 operand for these prompt producers and makes a native sparsity-only Q5_K port unattractive. The next independent question is *changed learned codes and producer activations at paid full-model quality*, or native timing of ordinary Q8/expert traffic under unprofiled full heads, not another unchanged-zero reader.

[The saved arrays and hashed receipt](../../data/qwen-moe/all-down-zero/README.md) include both standalone callback binaries/logs, every source tensor hash, the selected installed-runtime metadata and acquired-model receipt hashes, and the exact per-layer counts. Regenerate the CPU census with:

```sh
python3 tools/qwen-moe/all_layer_down_zero.py \
  --capture ../../data/qwen-moe/all-down-zero \
  --output ../../data/qwen-moe/all-down-zero/receipt.json
```

The captures ran in two foreground `tools/run-batch-compare` GPU reservations; each restored the active `bonsai-halo.service`. No model, selected Qwen executable, installed service or default changed.

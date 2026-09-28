# The rejected 31+1 GDN writer first changes the second recurrent layer's convolution row

[The saved convolution-plane comparison](qwen-moe-gdn-conv-planes.md) now localizes this row's changed values to both **old-history** positions: 7,986/8,192 words differ in each, while the newly projected QKV position matches all 8,192 bits. Every subsequent recurrent layer has the opposite pattern (old history identical, new input changed). The next diagnostic is the pre-31-row R1 source and the graph's R1 gather/concat lifetime, not another S-row snapshot or a first guess at layer-0 residual divergence.

The retained failed in-place swapped-batch experiment had 95/96 exact full-vocabulary heads, but changed all 248,320 FP32 values in the first swapped head and the final serialized state. [The source-row proof](qwen-moe-gdn-singleton.md) rules out a main-state source/destination collision: the 31-row call owns rows 1–31 and the subsequent singleton owns row 0. Where does the saved state first change?

[`gdn_state_diff.py`](../tools/qwen-moe/gdn_state_diff.py) parses the selected native revision's `llama_state_get_data` hybrid serializer, instead of treating a 2.114-GB context as an undifferentiated byte string. The format puts 32 per-stream full-attention KV sections first, then all thirty recurrent **R** (convolution) layer tensors, then all thirty recurrent **S** (matrix) layer tensors. Each recurrent row is individually addressable in the serialized image. The parser checks every field's bounds, both arm layouts and the final file length; it compares all metadata, KV and recurrent payload bytes. The pinned model has one full-attention layer every four layers. It uses mmap and does not copy the context into a second full-sized process buffer.

| Comparison after the identical swap workload | Metadata and attention KV | R layer 0 / S layer 0 | R layers 1–38 / S layers 1–38 (recurrent only) | Affected logical rows |
| --- | --- | --- | --- | --- |
| Rejected broad writer against gathered control | identical | identical | 29 / 29 differ | **only row 1** in every differing tensor |
| Installed guarded writer against gathered control | identical | identical | all identical | none |

The **first differing byte** is offset **9,812,533**, precisely the first byte of logical row 1 in R layer 1. That row is 98,304 bytes, of which 59,393 differ; its first two FP32 values are `0.18455142, 0.19792794` in gathered execution versus `-0.06864269, -0.01644424` in the rejected writer. The first S-layer-1 difference is also row 1; 1,752,638 bytes of its 2,097,152-byte row differ. All other 31 serialized logical recurrent rows are bit-identical in every layer, and all attention KV payloads are bit-identical. The R/S layer indices here are model layer numbers, not ordinal indices in the 30-layer recurrent list.

This is a useful localization, **not a claim that layer-1 GDN arithmetic is defective**. R layer 0 and S layer 0 agree after the step, yet the row entering R layer 1 differs. An earlier hidden activation in the layer-0 → layer-1 boundary, a width-dependent projection/graph operand, or an alias in the rejected graph could cause that difference. The complete serializer observes state after both split calls, not a timestamped forward trace; it cannot prove which call first modified row 1 or identify its first divergent temporary. Its row selectivity and identical layer-0 state rule out repairing this failure by copying an allegedly conflicting old main-state row. The installed guarded fallback remains the selected map.

**Next native diagnostic:** replay the saved common pre-swap state with the same no-callback graph. In the 31-row call, place a side-band write to a disjoint buffer at the layer-0 residual entering layer 1 and at layer-1 convolution input, rather than registering a graph-cutting callback. Compare row 1 of these operands and R1 immediately after that call, before the singleton; if identical, repeat around the singleton. Only after locating the first operand should the rejected broad dispatch be reconsidered. Preserve all requested heads and serialized rows in any acceptance; do not loosen to a top-1 or tolerance check.

## Reproduce and custody

```sh
python3 tools/qwen-moe/gdn_state_diff.py \
  ../../data/qwen-moe/gdn-fused/swap-control.state \
  ../../data/qwen-moe/gdn-fused/swap-default.state \
  --output ../../data/qwen-moe/gdn-fused/state-diff-rejected.json
python3 tools/qwen-moe/gdn_state_diff.py \
  ../../data/qwen-moe/gdn-fused/swap-guard-control.state \
  ../../data/qwen-moe/gdn-fused/swap-guard-default.state \
  --output ../../data/qwen-moe/gdn-fused/state-diff-guarded.json
```

The four source state SHA-256s are `ff6eacf57327d4708ad9a2c7ab558e90165b7f8499ffa009ebea5662cb1c1215` (both gathered controls and guarded default) and `292d89b9f24098668d2a793d44ac25795e4c67b0cd938469a8a7a6109b6aa0a9` (rejected default). The two JSON receipt hashes are `f6794929758383240b5ef8025bc42671c17cec05958ea9305a3a0a035b5d5da0` and `66c1cd3a1d946eab6ed013c307b15c7fe40eb1b7dbcae1c3ab505abd304d618c`, respectively. The serializer's field order was checked against source revision `7861dc746ed49c6bec1aa2c4f4b8f25b1bf59674` in `../../data/qwen-moe/runtime-source.git`. The existing [GDN full-head/state receipt](qwen-moe-gdn-fused.md) owns the model, executable and original wrapper logs. This new CPU-only comparison neither launches GPU work nor changes any native artifact, selected image, service or numerical contract.

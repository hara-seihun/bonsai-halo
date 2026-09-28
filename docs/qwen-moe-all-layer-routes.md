# All-layer Qwen expert-image locality on actual prompts

The layer-0 route capture made a 32 MB expert cache look modest, but it could not tell us whether other layers concentrate much more. I captured the eight selected IDs at **all forty layers** for the same disjoint 113-token train and 126-token held prompts, using the selected Qwen GGUF. Both layer-0 arrays match the earlier independent five-node capture byte for byte. On held text, each layer uses 95–194 distinct experts, median 120.5. The first three use 188, 176 and 194; the final layer uses 106. Train uses 90–186, median 120. A cache policy tuned to layer 0 does not describe the whole model.

A full packed expert image is 1,900,544 bytes in most layers; the receipt uses each layer's actual bank size, including the different final-layer quantization. For each contiguous token tile, count one image read per distinct selected `(layer, expert)`, assuming assignments can be grouped by expert within that tile. All eight IDs in a token are distinct. This is a **logical full-image-read grammar**, not native DRAM traffic. It grants perfect one-read grouping, charges no gather/scatter or activation work, and does not retain images across tiles.

| Held 126-token prompt | Logical routed image bytes | Ratio to eight reads/token |
| --- | ---: | ---: |
| One token per tile | 77.051 GB | 1.000 |
| Eight | 39.791 GB | .516 |
| Sixteen | 29.940 GB | .389 |
| Thirty-two | 21.508 GB | .279 |
| Sixty-four | 14.774 GB | .192 |
| Whole prompt | **9.466 GB** | **.123** |

Train has 69.101 GB without grouping, 20.408 GB at tile 32 and 9.329 GB for the whole prompt. For held, widening the logical tile from 32 to the full prompt removes another 12.042 GB of *logical* reads, but the existing selected runtime already batches up to 256 prompt rows and its `MUL_MAT_ID` MMQ path takes the full token axis. This is not a proposal to add grouped dispatch, nor evidence that it currently reads any expert image exactly once from DRAM. The concrete next question is whether the current packed MMQ reader re-reads its selected images and whether its Q4/Q5 kernel or unprofiled dispatch time leaves room after grouping.

Within this grammar, 9.466 GB is the exact compulsory full-image total for held: each of the 40 layers must read every expert selected by at least one token at least once, and grouping all 126 token assignments attains it. This is a family-specific byte floor, not an ISA-independent lower bound. It is 48.4% of the model's 19.569 GB routed bank. Keeping *all* those images resident would take gigabytes rather than this GPU's 32 MB last-level cache. Holding one layer's full selected set likewise takes at least 186 MB on held. A proposed full-image cache cannot eliminate the compulsory reads; it can only address duplicate reads or other scheduling shapes. Small decoded tiles, new packed labels or shared computation lie outside this grammar.

This is not a measured speedup. The callback requests router top-k views and cuts the execution graph, which can change arithmetic. The layer-0 identity check ties both captures at that boundary, not all forty uninstrumented routes. These are prompt-batch, teacher-forced routes rather than generated single-stream decode routes. Token gathering, expert output reduction, activation preparation, shared expert, GDN/attention, nonexpert Q8 projections and vocabulary head are excluded. No selected weights, executable or serving policy changed.

The [raw arrays, wrapper logs and source/model/binary-hashed receipt](../../data/qwen-moe/all-layer-routes/README.md) retain every layer's IDs and byte count. Recompute the result without a GPU:

```sh
python3 tools/qwen-moe/route_locality.py ../../data/qwen-moe/all-layer-routes \
  ../../data/qwen-moe/traffic.json \
  --output ../../data/qwen-moe/all-layer-routes/receipt.json
```

The observer source is `tools/qwen-moe/capture_all_routes.cpp`. It uses the installed library, loads the pinned GGUF and receives one top-k tensor per layer. Both GPU captures ran under `tools/run-batch-compare --runtime-max 45s --memory-gib 30 --host-reserve-gib 4 --exec`; after them the resident service was active and the GPU lock was free. The next native panel should pair unprofiled Q8 and Q4/Q5 phase markers at identical full-model prompt and decode shapes. Route counts alone cannot decide whether bytes, kernel utilization or launch scheduling is the bottleneck.

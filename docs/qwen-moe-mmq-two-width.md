# Two native widths recover most of the routed prompt tile bound

The installed Qwen MMQ prompt route uses J=64 for 65–256 rows. The [per-expert bound](qwen-moe-mmq-hybrid-bound.md) found a much smaller full-tile count but needs up to eight different bodies per layer. I priced the simpler split: at most two native J bodies per layer, with a group assigned wholly to one body and no group allowed more logical packed-image passes than installed J=64. The unchanged image and grouped activation order are the control. This is a static construction, not a GPU result.

For an expert with `n` assignments, a native J body contributes `ceil(n/J) * R * J` issued output positions at Qwen's gate/up row count `R=1024`, because both native I widths divide R. Its image-pass count is `ceil(n/J)`. The assignment is independent across experts once a pair of J bodies is fixed. Enumerate all eight singleton and 28 unordered pairs from J=16,32,...,128, selecting for each group the least positions among bodies that do not exceed its installed pass count. Ties select fewer passes, then fewer row/column blocks. Replaying each of the forty captured layer routes, rather than only their aggregate histogram, counts the number of occupied body launches. This proves the optimum **within this two-body, per-group no-extra-pass grammar**, not an ISA-independent lower bound. The native J=16 and J=64 bodies use different I widths and thread counts; partitioning and another launch have real costs.

| Held 126-token prompt, all forty layers | J bodies | Gate/up full-tile positions | Image passes | Gate/up row/column blocks | Gate/up launches |
| --- | --- | ---: | ---: | ---: | ---: |
| Installed | 64 | 328,138,752 | 5,007 | 40,056 | 40 |
| Less aggressive split | 32,64 | 172,490,752 | 5,007 | 78,056 | 79 |
| **Two-body optimum, no extra passes** | **16,64** | **112,902,144** | **5,007** | **75,088** | **80** |
| Eight-body per-group minimum | 16–128 | 97,271,808 | 4,958 | 77,664 | 188 |
| Two bodies if extra image passes are free | 16,32 | 97,271,808 | 5,397 | 86,352 | 80 |

The J=16/64 split removes 65.59% of installed issued positions, reaching 93.23% of the *available position saving* between installed and the eight-body minimum. It does so without increasing the logical image-pass count. The remaining 15,630,336 positions arise where a group needs a J=32/48/80/96/112 body to avoid either J=64's padding or extra J=16 image passes. Letting J=16/32 read an extra 390 images reaches the full position floor, but is a different traffic trade. Gate/up plus down doubles both position and row/column-block counts; two native launches for each body also double the absolute launch counts. These positions are not dot instructions, and image passes are not measured DRAM bytes.

The independent train prompt agrees on the choice: at 113 rows J=16/64 takes 106,250,240 positions versus installed 321,585,152, at the same 4,907 image passes; the eight-body floor is 91,815,936. Held 64-token tiles similarly take 147,849,216 versus 507,052,032, with 7,737 passes but 159 instead of 80 launches. At 32-token tiles, J=16/32 reaches 188,809,216 against installed J=32's 368,967,680, with unchanged 11,260 passes and 279 instead of 160 launches. The 126-row result is the pertinent prompt-sized candidate for the installed width policy.

[The CPU receipt](../../data/qwen-moe/all-layer-routes/mmq-two-width.json) records every train/held width, the full nondominated two-body frontier, capture/occupancy/model/native hashes and the source hash. Its SHA-256 is `781a6a468cd7de4a1118394c557656056d7479cfd2a11e93414a0ab1dfe4db9e`. Reproduce without a GPU:

```sh
python3 tools/qwen-moe/mmq_two_width.py --output ../../data/qwen-moe/all-layer-routes/mmq-two-width.json
```

Next implement a native J=16/J=64 grouped prompt dispatcher for whole 65–256-row tiles, reusing the installed arithmetic body and masked stores. The experiment must charge expert partition/compaction, extra launches, occupancy from different row geometry and actual device time. Compare complete-model prompt TPS, batched independent-stream generation and logits against the installed J=64 map. Single-token MMVQ cannot benefit. No model image, installed executable, service or GPU state changed in this iteration.

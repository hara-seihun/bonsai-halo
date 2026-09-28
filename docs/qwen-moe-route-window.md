# Finite expert-image cache versus a wider Qwen prompt window

The forty-layer real prompt routes make it possible to settle a narrower question than physical DRAM traffic: can a full-image cache the size of this GPU's 32 MiB last-level cache compensate for processing 32 tokens at a time instead of grouping the entire 126-token prompt? In the full-image grammar, no. A sixteen-image cache reaches its exact offline optimum at 17.839 GB of held routed reads, still 8.373 GB above the whole-prompt 9.466 GB one-read floor.

Each layer has its own packed expert images. For a contiguous token tile, assignments can be grouped and reordered freely by expert, and a selected expert image is read once. Tiles execute in token order; up to C complete images may persist across tiles. The cache starts cold at each layer, allows bypass, and has no other occupants. This grants the cache considerably more control than native hardware. It excludes partial-image caching, new packed labels and changes to the selected numerical map. It also grants gathering and scattering for free. Layer images cost 1,900,544 to 2,039,808 bytes, so sixteen occupy up to 32,636,928 bytes, almost the full 32 MiB cache. Thirty-two images require up to 65,273,856 bytes.

For each tile with selected-image set S and previously seen set P, at most `min(C, |S ∩ P|)` reads can hit, regardless of the offline policy. Summing `|S| - min(C, |S ∩ P|)` is a lower bound, because it gives the cache an independently ideal state before *every* tile. The executable schedule retains the C available images with earliest next tile use, reorders the current tile's expert accesses and bypasses unretained loads. Its count is achievable in this grammar. The lower and constructive counts coincide at C=16 for held and train 32-token tiles, so this is an **exact optimum for that finite workload and grammar**, not merely a heuristic estimate.

| Held 126-token prompt | No cache | C=16 lower = schedule | C=32 lower / schedule | C=64 lower / schedule |
| --- | ---: | ---: | ---: | ---: |
| Eight-token tiles | 39.791 GB | 21.575 / 22.583 GB | 10.711 / 14.984 GB | 9.466 / 10.374 GB |
| Sixteen-token tiles | 29.940 GB | 21.379 GB | 13.807 / 14.639 GB | 9.635 / 10.298 GB |
| Thirty-two-token tiles | 21.508 GB | **17.839 GB** | 14.200 / 14.202 GB | 10.059 / 10.256 GB |
| Sixty-four-token tiles | 14.774 GB | 13.551 GB | 12.328 GB | 10.123 GB |
| Whole 126-token prompt | 9.466 GB | 9.466 GB | 9.466 GB | 9.466 GB |

Train has 113 tokens. At width 32 its no-cache count is 20.408 GB and the C=16 optimum is 16.739 GB; its whole-prompt floor is 9.329 GB. Held C=16 saves 3.669 GB or 17.1% of width-32 routed image reads, but closes only 30.5% of the 12.042 GB distance to whole-prompt grouping. Increasing the tile to 64 with no cache saves 6.734 GB against width 32. Even a C=32 cache, which exceeds the nominal last-level capacity, cannot close that gap at width 32. The sixteen-image optimum is also attained at width 16 on both splits, though it is much worse than grouping 32 or more rows.

This rules out a sixteen-full-image retention policy as a substitute for a wider logical prompt tile on these captured routes. It does **not** identify a native speedup: the selected `MUL_MAT_ID` path already groups the prompt token axis, the cache is shared with other data, and none of these counts measures physical DRAM bytes or elapsed time. Callback-cut prompt routes can change the target arithmetic; generated decode routes are not captured here. A worthwhile next engine measurement is the counter-free unprofiled Q8 and Q4/Q5 phase time on equal full-model prompt and decode workloads. If the packed MMQ consumer re-reads expert images despite grouping, target that implementation; adding a full-image cache to a schedule that already achieves one read cannot improve this byte grammar.

The [raw CPU receipt](../../data/qwen-moe/all-layer-routes/window-bound.json) hashes the source, model identity, input inventory, token and forty layer-ID arrays for each split, and retains per-layer lower and constructive counts. Recompute without using the GPU:

```sh
python3 tools/qwen-moe/route_window_bound.py ../../data/qwen-moe/all-layer-routes \
  ../../data/qwen-moe/traffic.json \
  --output ../../data/qwen-moe/all-layer-routes/window-bound.json
```

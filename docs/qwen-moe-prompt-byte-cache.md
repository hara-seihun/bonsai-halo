# Exact arbitrary-byte cache optimum on actual Qwen prompt routes

At 32-token tiles, even a clairvoyant 32 MiB cache that retains **arbitrary individual bytes** of the unchanged packed expert images can lower held forty-layer routed logical reads only from the full-image offline optimum **17.839 GB to 17.481 GB**. The additional 357,433,344 bytes are **1.662% of uncached tile reads** and **0.1080% of the 126-token conditional complete-model one-read stream**. The remaining gap to grouping each layer's entire prompt once is 8.015 GB. This tightens the earlier [full-image result](qwen-moe-route-window.md) without requiring images to be indivisible; it does not count native physical transactions or establish a full-model speedup.

## Domain, optimum and cost

The disjoint 113/126-token train/held prompt captures contain eight distinct IDs per token at all forty layers. A contiguous tile of W tokens requests each selected `(layer, expert)` packed image once, permitting the selected native-style expert grouping and free reordering *within* a tile. Tiles remain causally ordered; different layer banks have disjoint addresses. Each layer starts with a cold, fully associative 33,554,432-byte cache; all other occupants, gathering and replacement work are free. Cached bytes may be chosen individually, misses may bypass the cache, and an offline controller knows every future tile. A prefetched byte would count as a read. No weight or numerical map changes.

For a fixed layer, every byte of one expert image has the same tile-demand sequence and the image byte sizes are equal. At the end of a tile, the cache can contain only bytes retained from its previous state or demanded by that tile. Retain from that union in increasing order of the next tile demanding each byte, filling the capacity with full images and at most one *partial* image. This is an **exact offline optimum** in this grammar: if a retained byte `b` has a later next use than an available but omitted byte `a`, exchange `b` for `a`. The exchange replaces at most the first miss of `a` by at most the first miss of `b`; after either first demand the byte can be retained again. Repeating exchanges yields the earliest-next-use cache. Within each tile serve existing hits first, bypass misses not kept and retain the chosen bytes. Capacity is never exceeded. The script counts exact integer bytes, not a fractional-byte relaxation. Tie choices among bytes with the same next use do not affect the byte count.

| Held prompt tile | No retention | Sixteen-full-image offline optimum | Arbitrary-byte optimum | Extra over full-image |
| --- | ---: | ---: | ---: | ---: |
| 8 tokens | 39.791 GB | 22.583 GB | 21.444 GB | 1.139 GB |
| 16 tokens | 29.940 GB | 21.379 GB | 20.555 GB | .824 GB |
| **32 tokens** | **21.508 GB** | **17.839 GB** | **17.481 GB** | **.357 GB** |
| 64 tokens | 14.774 GB | 13.551 GB | 13.432 GB | .119 GB |

The 32-token train control is 20.408 / 16.739 / **16.381 GB**, respectively. The held whole-prompt grouped one-read floor is 9.466 GB; a 32-token arbitrary-byte cache recovers 4.027 GB of the 12.042-GB uncached-window gap, or 33.44%, compared with 30.47% for full images. The extra arbitrary-byte flexibility closes only 2.97% of that gap. At the actual wider 64-token shape the extra falls to 119 MB. A cache-line rather than byte cache is a restriction of this free oracle and cannot do better.

This bound concerns **cross-tile retention only** and excludes within-tile repeated weight reads, changed representations, expert batching beyond the tile, other GPU cache traffic and native Q8/expert stalls. The selected runtime already groups expert work; do not build an expert-subimage cache to recover the widening gap on these prompt routes. Measure actual ordinary routed Q4/Q5 and nonexpert Q8 physical traffic/latency in the unprofiled full-model path before choosing a native consumer change. The all-layer routes were captured through a callback that cuts graph fusion, so they are not a fused generated-stream claim. The separate [serial-decode capacity result](qwen-moe-cache-capacity.md) covers a different shape.

[Source-, model-, inventory-, previous-receipt- and forty-layer-capture-hashed CPU receipt](../../data/qwen-moe/all-layer-routes/prompt-byte-cache.json) records every tile hit and retained-byte count. Receipt SHA-256: `9ecec2230cdde01c1782d93a476c16a9dd1df1bffff997c44a546f3c3c8330dd`.

```sh
python3 tools/qwen-moe/prompt_byte_cache.py \
  ../../data/qwen-moe/all-layer-routes \
  ../../data/qwen-moe/traffic.json \
  --output ../../data/qwen-moe/all-layer-routes/prompt-byte-cache.json
```

CPU only; no GPU reservation, native executable, selected model image or resident service changed.

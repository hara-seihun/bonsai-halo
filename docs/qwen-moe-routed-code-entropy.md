# Exact Q4_K code-byte compression on real routed Qwen experts

The selected Qwen3.6-35B-A3B UD-Q4_K_M image has eight routed experts per layer. On **token 0** of each of the disjoint 64-token train and held real-text captures, `tools/qwen-moe/routed_code_entropy.py` reads all forty actual top-eight routes and the installed gate/up Q4_K blocks. It extracts their 128 code bytes per 144-byte block, leaving 16 metadata bytes unchanged, and compresses the *two gate/up code arrays per selected expert* as one independently addressable image. This tests static exact code storage, **not** a new expert dispatch, exact native FP32 arithmetic, complete-model quality change, or GPU time.

| Observed token-0 route | Q4 code B (320 expert images) | Ideal per-image IID-byte B, free probabilities | Paid zlib-1 or raw B | Exact saving / full one-read model bytes |
| --- | ---: | ---: | ---: | ---: |
| Train | 335,544,320 | 322,060,207 | 324,476,093 | 11,068,227 / 0.421456% |
| Held | 335,544,320 | 322,223,702 | 324,657,267 | 10,887,053 / 0.414557% |

The paid arm stores each independently addressable zlib-1 stream plus an eight-byte address/length record where compression wins, otherwise its raw image; all 320 images win in both routes. Each compressed stream is decompressed and compared byte-for-byte to its source. The total model comparator is **2,626,187,904 B/token**, one uncached read of the selected eight experts and nonexpert weights. Raw extraction visits 1,048,576 code bytes per selected expert (gate and up together). Expert-down bytes, router, metadata, nonexperts and the head are unchanged. Distinct routes are separate panels, not additive savings. The conditional one-read ratio is not DRAM traffic or TPS; decompressing streams, buffering, maintaining random access and reading metadata cost work and memory. There is no native reader or selected image change.

A free per-image IID arithmetic code with its exact empirical byte probabilities cannot save more than **13,320,618 B / 0.507223%** of that held one-read stream (finite-symbol overhead and probability tables omitted). Relative to the actual independently addressable zlib streams, at most **2,433,565 B / 0.092665%** more lies in this *IID-byte family* before paying a second codec. This is not a bound on context-aware compression, changed learned expert weights, packed direct consumers or whole-map relabelings. Zlib itself can exploit contexts, so its performance need not be bounded by that IID calculation; here it remains above the IID ideal.

The metadata-defined dormant blocks account for **4,782,080 held code bytes** of the 335,544,320 B. Removing these with a free index and zlib-compressing the remaining contiguous active bytes still saves **8,352,037 B** against the active raw code stream, before any index or reader cost. Those savings are not additive to the paid image: the exercise separates trivial dormant-row capacity from active-code structure. A prior [metadata census](qwen-moe-q4-scale-packets.md) already found the raw metadata repeats solely on dormant blocks. The active-code opportunity does not make another frozen packet dictionary attractive: the paid static byte gain remains small on the whole stream.

The [source/inventory/capture/model and per-tensor hashed CPU receipt](../../data/qwen-moe/routed-code-entropy/receipt.json) records every selected code and compressed-stream hash, entropy, dormant count and paid length in forty layer shards. It uses the installed image rather than original BF16 weights, only the first captured token in each split, and no GPU. Reproduce in bounded CPU shards:

```sh
python3 tools/qwen-moe/routed_code_entropy.py --first 0 --count 15
python3 tools/qwen-moe/routed_code_entropy.py --first 15 --count 15
python3 tools/qwen-moe/routed_code_entropy.py --first 30 --count 10
python3 tools/qwen-moe/routed_code_entropy.py --summarize
```

**Decision:** do not port a zlib-inflate boundary for <0.42% conditional one-read model bytes on these two routes. A compressed-label consumer is worth testing only if it operates directly on changed paid codes and demonstrates complete-model quality and native cost. Independently, the unprofiled Q8/routed expert layer critical path remains the larger engine question; profile that with full heads rather than turn this byte ratio into a claimed speedup.

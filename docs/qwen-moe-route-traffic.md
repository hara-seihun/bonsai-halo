# Routed expert reuse on real layer-0 prompts

The two real-text captures in `data/qwen-moe/route-capture/` answer a question the synthetic route models could not: does expert reuse occur locally enough to matter in a decode step, or only after grouping a prompt's assignments? It is the latter. These captures have 113 train and 126 disjoint held tokens from the installed mixed GGUF reference. The capture owner retained the layer-0 top-eight IDs, normalized scores, input activations and expert outputs. This study reads only the IDs. It does not fit a new model or alter the runtime.

Each selected layer-0 expert occupies 1,900,544 encoded bytes across Q4 gate, Q4 up and Q5 down in the pinned GGUF inventory. For a window of `B` successive tokens and an assumed expert tile capable of consuming `T` assignments per complete image read, the counted reads are

\[
 L(B,T)=\sum_{e=0}^{255}\lceil n_e(B)/T\rceil,\qquad
 n_e(B)=\#\{(token,rank):expert(token,rank)=e\}.
\]

This is the minimum number of full expert-image reads **within this specified tile grammar** when an image is not retained between tiles. It still performs all `8B` expert products and all weighted sums. It is not a bound on another ISA program, MMQ's physical bytes, cache transactions or runtime. `T=1` is the per-assignment logical control, not a claim that native grouped dispatch reads eight complete images when the same expert repeats.

| Split | Window tokens | Tile assignments | Logical reads | Reuse over per-assignment control | Layer-0 image bytes | Bytes avoided against that control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1 | any | 904 total | 1.00x | 1,718 MB | 0 |
| train | 8 | 8 | 756 total | 1.20x | 1,437 MB | 281 MB |
| train | 16 | 8 | 582 total | 1.55x | 1,106 MB | 612 MB |
| train | 32 | 8 | 421 total | 2.15x | 800 MB | 918 MB |
| train | whole 113 | 8 | 205 | 4.41x | 390 MB | 1,329 MB |
| held | 1 | any | 1,008 total | 1.00x | 1,916 MB | 0 |
| held | 8 | 8 | 843 total | 1.20x | 1,602 MB | 314 MB |
| held | 16 | 8 | 682 total | 1.48x | 1,296 MB | 620 MB |
| held | 32 | 8 | 500 total | 2.02x | 950 MB | 965 MB |
| held | whole 126 | 8 | 224 | 4.50x | 426 MB | 1,490 MB |

The 32-token rows include the short final windows, rather than silently dropping them. At tile four the train/held counts rise to 451/530, and at tile one to 904/1,008. The full-size-window ceiling at tile 32 is 421/499, scarcely better than tile eight at 32 tokens. At whole-prompt width, tile 32 needs 169/188 reads, compared with 205/224 at tile eight. Increasing tile width without also increasing the assignment window has almost no capacity to help this layer-0 trace.

The prompt has skew, but not useful adjacent-token locality. The 113-token train route set visits 169 experts and the 126-token held route set 188; independent uniform eight-of-256 routes would visit 248.92 and 251.31 in expectation. Yet consecutive tokens share only 37 expert IDs across 112 train pairs and 30 across 125 held pairs, or 0.330/0.240 per pair. Independent uniform routes share 0.25 in expectation. Zero overlap occurs for 85/112 and 103/125 pairs. A sequential token-to-token expert reuse plan cannot bank on the long-window skew. In single-token decode all eight IDs are distinct and this particular intra-layer sharing count is exactly zero. Keeping an expert image across full model steps would also have to survive the intervening layers and their nonexpert streams; this calculation does not grant that cache residency.

[The finite-cache follow-up](qwen-moe-route-cache.md) checks the apparent contradiction: a future-aware sixteen-image cache can avoid 342 of 1,008 held requests despite low adjacent overlap, but consumes 30.409 MB and saves only 0.1964% of the modeled complete weight stream per token at layer 0. Train-frequency pinning avoids 185. These are generous logical cache bounds, not native physical bytes.

The constructive implication is narrow. Expert-major gathering on sufficiently wide prompt or independent-stream batches has real reuse available at layer 0, while a new sequential decoder cache is a poor first move. Native `mul_mat_id` already groups by expert, so these counts do **not** credit it as a new improvement. The next native measurement should pair actual routing with physical DRAM bytes and timed Q4/Q5 expert phases at 1, 8 and 32 rows, then vary the native MMQ expert tile/grouping while retaining the same arithmetic and output-head workload. If native traffic already approaches this conditional image-read count, larger tiles cannot deliver its full apparent saving. The other 39 layers require their own routes before scaling any layer-0 number to a complete model.

`tools/qwen-moe/route_traffic.py` validates all route IDs, matches the capture owner's ID hashes, joins the pinned GGUF layer-0 inventory, and records every window/tile count, source hash and text/token identity in `../../data/qwen-moe/route-traffic.json`. Reproduce without a GPU:

```sh
python3 tools/qwen-moe/route_traffic.py ../../data/qwen-moe/route-capture \
  ../../data/qwen-moe/traffic.json \
  --output ../../data/qwen-moe/route-traffic.json
```

This is an actual-route occupancy and conditional encoded-byte result. There is no native timing, whole-model speedup, numerical-map change or service change. The complete-image sub-bit failures are why a route-level byte opportunity alone does not license a new approximate expert representation.

# Causal expert anticipation on actual Qwen prompt routes

The layer-0 adjacent-route result ([route traffic](qwen-moe-route-traffic.md)) is not representative of all forty layers. This CPU-only study reads the pinned 113-token train and disjoint 126-token held **batched prompt** routes, eight expert IDs per row, and asks which next-row images are known from the previous row of the same layer. This is a conditional image-read and predictor study, not a native cache or speed measurement. All expert outputs and the selected arithmetic remain required.

The causal rules are (i) retain the previous eight expert IDs, (ii) retain those eight plus the eight most frequent other IDs fitted on train, and (iii) predict 8/16/32 IDs by pooling train-only next-row co-occurrence counts for the eight previous IDs. Controls are a train-frequency-pinned set and a **noncausal held-frequency-pinned** set of the same capacity. The latter is not an oracle for adaptive policies. Every held token except the first per layer is scored: 5,000 layer-row transitions, 40,000 expert assignments. No held next-row routes enter the causal prediction. The train-to-held fit is per layer, not a common expert bank across layers.

| Candidate full-image slots *per layer* | Previous-row + train-hot hits | Train-static hits | Pooled-transition hits | Held-hindsight pinned hits | Entire eight-ID row inside previous+hot |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 15,411 (38.53%) | 7,261 (18.15%) | 9,778 (24.45%) | 14,004 (35.01%) | 6 / 5,000 |
| 16 | 19,311 (48.28%) | 11,235 (28.09%) | 14,486 (36.22%) | 20,233 (50.58%) | 25 / 5,000 |
| 32 | 24,163 (60.41%) | 16,777 (41.94%) | 19,962 (49.91%) | 27,752 (69.38%) | 168 / 5,000 |

The eight-slot previous-row arm is an exact count, not a learned predictor: held adjacent overlap is 15,411 / 40,000, versus train 13,574 / 35,840 (38.53% versus 37.87%). **Layer 0 alone has 0.24 shared IDs per transition; layers 5–31 mostly have 3–4.** Previous routes are substantially more informative than a fixed hot-expert list on the captured deeper layers. This reverses the layer-0-only hypothesis that adjacent-token reuse is uniformly absent. The pooled transition table transfers less well than simply retaining the last eight; route conditioning is real but fitting it from 112 transitions/layer is not the solution.

These hits do **not** imply an 8.97% engine gain. A hypothetical lossless full-image cache that keeps the previous eight images of **every** layer from one token until its next token would avoid 29,428,129,792 logical routed bytes in this 5,000-layer-row panel, 38.50% of its 76,439,552,000 routed request bytes. That is about 235.4 MB per 125-token-transition model step, or 8.96% of the declared 2,626.188 MB complete one-read-per-token weight stream. It requires 320 simultaneously resident images, **611,516,416 bytes (611.5 MB)**, against the device's measured 32 MB last-level cache; the intervening 39 layers and their nonexpert weights compete for that cache. The GGUF already resides in device-addressable memory; merely retaining pointers to the same images does not avoid the next DRAM transfer. Conversely, prefetching a *new* set of C images every row cannot reduce logical reads relative to waiting for the eight known IDs: it reads `C + (8 - hits)` images when the prefetched set is disjoint from already-resident images, versus exactly eight on-demand. At C=8 this is `16 - hits >= 8`, and at C=16 it is at least 16. Prefetch could overlap latency; it is not a byte-saving construction without real cross-token residency.

This bound is deliberately narrow. The captures are grouped whole-prompt callback runs, not generated decode routes; the callback can cut fusion; prompt-native `mul_mat_id` already groups routed assignments across rows, so none of these hits may be credited again as a prompt optimization. Cache set mapping, lines, actual GL2C/DRAM bytes, state and occupancy were not measured. The conditional promise is enough to change the next question: **capture genuine generated single-token routes across all layers without a graph-cut observer, then measure actual expert DRAM transactions with the resident service restored**. If the deeper persistence survives serial decode, test selective image retention or overlap only against native grouped dispatch and the Q8/nonexpert path. Do not build a forty-layer full-image cache from these prompt rows.

The complete per-layer counts, each of eighty capture hashes, source hash, inventory/model hashes and the raw panel live in [`data/qwen-moe/all-layer-routes/prefetch.json`](../../data/qwen-moe/all-layer-routes/prefetch.json). Reproduce on CPU, without taking the GPU lock:

```sh
python3 tools/qwen-moe/route_prefetch.py ../../data/qwen-moe/all-layer-routes \
  ../../data/qwen-moe/traffic.json \
  --output ../../data/qwen-moe/all-layer-routes/prefetch.json
```

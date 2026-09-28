# Builder-free expert MMQ dispatch on actual Qwen routes

The [compact indirect queue](qwen-moe-mmq-indirect.md) replaces 1.475 million full-grid entries with 75,088 useful gate/up tile claims on the held 126-row prompt. It must build descriptor lists after routing and claim each tile. There is another exact schedule: launch one CTA per expert in each J=16/64 body, read the existing `expert_bounds`, return if the group belongs to the other body, and let the CTA loop over that expert's column and output-row tiles. Gate and up use the native sorted input and weight image. Down uses the same group choice, its own post-SwiGLU input, and two output-row halves per gate/up row tile. No device queue, builder, descriptors, host readback or per-tile atomic claim is needed.

This is a construction and a static cost comparison, not a native speedup. Its CPU replay emits the proposed per-expert loop's tile coordinates and compares the complete ordered lists byte for byte with the independently implemented indirect queue for all eighty captured layer/routes. For each expert, `n = expert_bounds[e+1] - expert_bounds[e]`; choose the same no-extra-image-pass J as the two-width bound, then visit `(column,row)` in `[0,ceil(n/J)) × [0,1024/I)`. Those Cartesian products partition exactly the queue's tile list. Each group belongs to one J body, so no output tile is computed twice. Keeping each native tile's K loop, store mask and sorted assignment order retains the arithmetic *within* that tile. The host replay cannot prove that a new native launcher preserves FP32 output or graph scheduling.

| Forty-layer whole prompt, gate/up coordinate | Train 113 rows | Held 126 rows |
| --- | ---: | ---: |
| Useful tiles, identical to compact queue | 74,304 | 75,088 |
| Per-expert CTA launches, both J bodies | 20,480 | 20,480 |
| Active CTAs | 4,888 | 4,958 |
| Empty or other-body returns | 15,592 | 15,522 |
| Longest CTA, sequential native tile bodies | 16 | 16 |
| Builder CTAs, descriptor bytes, task claims | 0 | 0 |
| Compact queue claims for gate/up and down | 222,912 | 225,264 |
| Two native full grids, gate/up entries | 1,474,560 | 1,474,560 |

The gate/up count is per one projection. Gate and up have separate calls; down doubles the row work. Thus across gate, up and down this candidate launches 61,440 CTAs, with 46,566 early returns on held, and executes 300,352 tile bodies. The compact queue executes those same bodies with 300,352 task claims across all three calls, although it can reuse one descriptor list. A direct CTA still reads group bounds and decides its J, while the queue also pays its builder, descriptors and atomic claims. Compared with the full split grids, the proposed direct schedule reduces scheduled gate/up CTAs by 98.61% without construction work.

The price is serial depth. An active J=16 CTA executes sixteen gate/up row tiles for one column; a J=64 CTA executes eight per column and can need two columns. Down requires at most 32 sequential row tiles per CTA. These are lower bounds on a *one-CTA-per-expert, sequential-native-tile-body* realization's critical chain, not a lower bound on all MMQ implementations. The compact queue can run different row tiles of one expert concurrently. It also has 75,088 gate/up CTAs/claims versus 20,480 direct CTAs, so comparing only launch counts would conceal the loss of parallelism. The J=64 body has just 579 active expert CTAs across held forty layers, around 14 per layer. It may underfill the GPU despite the eliminated builder. Native launch overhead, CTA residency, packed-weight caching, group-bound reads and the larger per-CTA loop must decide the winner.

The proposed native experiment needs two dispatch variants sharing the same J=16/64 arithmetic body: one indirect queue and one expert-owned loop. Hold prompt, head rows, batch/ubatch, weights and sorted IDs fixed; compare finite full-model logits, prompt TPS and multi-sequence generation under the measurement wrapper. A direct loop should not replace the selected J=64 prompt route on this static receipt. It is a cheaper scheduling *construction* with an explicit occupancy risk, not a measured faster execution.

[CPU receipt](../../data/qwen-moe/all-layer-routes/mmq-persistent.json), SHA-256 `bae91cfaf9daf50f6db51a94cbb22fe7617965d7d441399ea243a94d487c4c19`, stores each layer's ordered tile digest, active CTA count, longest chain, input hashes, pinned model and native-source identity. Reproduce it with `python3 tools/qwen-moe/mmq_persistent_bound.py --output ../../data/qwen-moe/all-layer-routes/mmq-persistent.json`. No GPU, executable, model image or resident service changed.

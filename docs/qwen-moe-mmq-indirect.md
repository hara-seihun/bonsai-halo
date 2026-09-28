# A bounded GPU-indirect queue for Qwen's two MMQ widths

The [two-width bound](qwen-moe-mmq-two-width.md) saves 65.59% of full gate/up tile positions on the held 126-token prompt. The [full-grid check](qwen-moe-mmq-split-plan.md) showed that simply launching both native bodies schedules 1,474,560 blocks across forty layers, 94.91% of which return. This iteration constructs the missing compact dispatcher against the actual forty-layer routes. It does not measure a GPU kernel or claim that the mixed widths win whole-model time.

A descriptor needs just 17 bits: eight for expert, four for column tile, four for the 1024-row gate/up tile and one for J=16 or 64. A 32-bit word leaves room for simpler loads. Decode it into the original sorted `ids_dst` and `expert_bounds`; neither token sorting nor activation quantization repeats. Every nonempty expert is assigned one native body by the same no-extra-image-pass rule as the two-width study. The body uses the same packed weights, K loop, tile accumulation, masked stores and FP32 output fold as its installed counterpart. For a down projection with 2048 output rows, read each gate/up descriptor twice, with down row tile `2*r + parity`. This avoids a second list. It does **not** reuse down activations, which follow SwiGLU and need their own quantization.

The CPU construction encodes every descriptor, decodes it, checks its expert's prefix segment and column bounds, and checks uniqueness. Input files match all eighty captured route hashes in the occupancy receipt. A one-CTA, 256-lane device builder can read the 257 existing group bounds, assign each lane an expert, scan 256 per-expert descriptor counts and emit the same lists. Under this width policy an expert has at most 16 gate/up descriptors, whether it uses J=16 with one column tile and 16 row tiles, or J=64 with at most two column tiles and eight row tiles. Allocate 4096 descriptors per layer as a fixed worst-case queue; its device count is the prefix end, so no host readback or variable-size launch is required. Separate persistent J=16 and J=64 body launches can claim tasks with a device counter and stop at the device count. A fixed cooperative grid and queue mapping remain native engineering, not a measured consequence of the CPU construction.

| Whole prompt across forty layers | Train 113 rows | Held 126 rows |
| --- | ---: | ---: |
| Expert groups | 4,888 | 4,958 |
| J=16 / J=64 gate/up tasks | 70,096 / 4,208 | 70,064 / 5,024 |
| Compact gate/up tasks | 74,304 | 75,088 |
| Down tasks reusing descriptors | 148,608 | 150,176 |
| Actual gate/up descriptor bytes | 297,216 | 300,352 |
| Largest layer queue / fixed layer capacity | 2,936 / 4,096 | 3,064 / 4,096 |
| Two full native grids, gate/up entries | 1,474,560 | 1,474,560 |

This replaces the 1,399,472 held returning gate/up grid entries with a one-CTA scan/emit pass per layer and queue-claim work. There are 75,088 gate/up claims and 150,176 down claims if one claim owns one output row/column tile. That is 1.875 times the selected J=64's 120,168 active gate/up-plus-down blocks. The compact route also adds a builder launch per layer and an extra body launch per projection. Queue atomics, persistent-grid occupancy, the J=16 body's 128-thread versus J=64's 256-thread geometry, launch gaps and packed-image traffic are unpriced. The 300 KB figure is total descriptor payload across forty layers, not a per-token saving or measured traffic. The largest held J=16 queue has enough tasks for a 100-block persistent grid, but the J=64 arm has only about 126 per layer on average; a fixed grid must not assume both bodies saturate identically.

This gives the native experiment a concrete fork. Reusing full grids pays 1.475 million scheduled entries; a compact device queue pays 225,264 tile claims across gate/up and down, 40 small builder passes and their extra launches. A scheduler could claim several adjacent descriptors at once, but must preserve the same task-to-expert mapping and output fold. Measure both against installed J=64 at matched complete-model prompt and independent-stream shapes, with finite logit bits and greedy IDs compared. The static position saving alone cannot pick the winner.

The [source- and capture-hashed receipt](../../data/qwen-moe/all-layer-routes/mmq-indirect.json), SHA-256 `c84abca0e20f857f73127d9a9617832b9ccc4a897b19c2adfe9ff2bc8db9409f`, keeps each layer's descriptor digest and size, model/native source hashes and both disjoint splits. Recreate it without the GPU:

```sh
python3 tools/qwen-moe/mmq_indirect.py --output ../../data/qwen-moe/all-layer-routes/mmq-indirect.json
```

No selected executable, model image, GPU reservation or resident service changed.

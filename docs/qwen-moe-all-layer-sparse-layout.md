# Forty-layer routed down-zero layout bound

The attractive layer-0 expert-fixed zero mask does **not** extend into a useful whole-model down-weight shortcut. On the two disjoint actual 64-token Qwen3.6 producer captures, a train-fitted independent column permutation for each `(layer, expert)` packs 2,806 skippable 32-input blocks among 20,480 held routed assignments. A free per-assignment hindsight permutation packs 3,064; the unchanged installed order packs **zero**. If selective blocks and every image permutation were free, charging the actual Q5_K/Q6_K down-expert image size per layer gives **1,975,424 bytes/token, 0.0752202%** of the 2,626,187,904-byte conditional complete one-read stream for the static layout. The hindsight ceiling is **2,157,056 bytes/token, 0.0821364%**. These are not observed DRAM traffic or speedups; changing a 256-input quant superblock's columns changes its scale allocation and FP32 reduction map. No runtime or quality result follows.

Layer 0 contributes **2,713/2,806** static slot blocks; layers 1, 2, 3 and 37 contribute 48, 10, 31 and 4 respectively. The other 35 layers yield none under the train-fitted layout. The exact original-float zero masks are constant across every observed train selection of each of 3,987 visited `(layer,expert)` pairs. Among 3,776 held pairs, just one varies: layer 34 expert 48 has coordinate 201 zero on one of six held rows and nonzero on the other five; its ten train rows have no zeros. Thus the layer-0 mask stability is nearly universal **as a finite observed invariance**, but the number of zero coordinates collapses after layer 0. Stable does not mean sparse.

| Observation | Original-order blocks | Train static blocks | Free hindsight blocks | Denominator |
| --- | ---: | ---: | ---: | ---: |
| Each held slot separately, J=1 | 0 | 2,806 | 3,064 | 20,480 slots × 16 blocks |
| Contiguous expert-major tiles, J=16 | 0 | 772 | 927 | 4,112 tiles × 16 blocks |
| Contiguous expert-major tiles, J=64 | 0 | 766 | 921 | 3,776 tiles × 16 blocks |

At J=64, 766/921 possible skippable blocks are reached by the static layout; **766/766 are on experts observed in training**. The 155 missing oracle blocks occur only in held-only experts (5,832 held slot assignments throughout the stack); the one changing zero coordinate creates no extra block. These tile counts are not substituted into the slot-weighted whole-model byte comparison: a grouped MMQ may reuse a weight tile across rows, and its DRAM traffic depends on its schedule and cache. The J=1 bound deliberately grants full-image reads per routed assignment under the established conditional one-read model, then removes the relevant fraction of each *actual layer's* down-expert bytes; its ratio is an optimistic upper opportunity for this grammar, not a performance bound over other algorithms.

For a tile with hidden zero sets `Z_i`, any static coordinate permutation can form at most `floor(|intersection_i Z_i|/32)` disjoint fully zero K blocks. A hindsight tile-specific permutation attains it by placing the intersection first. The script counts those 32-coordinate blocks for original, train-only frequency-sorted and hindsight layouts. The score-weighted down sum would retain all eight branches and their output observation in a hypothetically exact selective execution; there is no zero-output omission. Original FP32 zeros imply an all-zero native Q8_1 input block at that coordinate, but repartitioning the original Q5_K/Q6_K weights and input quantizer changes their scales and rounding. In particular the construction does **not** preserve selected finite logit bits or offer a paid image. It is a structural ceiling for free changed layout and zero-selective block reads on these captured producers.

[Source](../tools/qwen-moe/all_layer_sparse_layout.py) and [source, capture, traffic and per-layer hashed receipt](../../data/qwen-moe/all-down-zero/sparse-layout-receipt.json) retain all 80 producer/ID file hashes, forty per-layer results and all tile widths. The acquisition and capture provenance are in [the owning all-producer receipt](../../data/qwen-moe/all-down-zero/receipt.json). Reproduce on CPU:

```sh
python3 tools/qwen-moe/all_layer_sparse_layout.py \
  --output ../../data/qwen-moe/all-down-zero/sparse-layout-receipt.json
```

This closes broad zero-block column packing on the frozen Qwen down producers as a high-impact MoE engine proposal. The independent native question is unprofiled physical Q8/expert time with complete heads; the changed-representation question is a **paid** expert image fitted on broader real routes and selected by disjoint whole-model loss, not another sparse block count. Neither GPU, selected runtime nor resident service changed.

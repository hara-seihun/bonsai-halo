# Qwen routed Q4 metadata cannot share an active same-K packet

The selected Qwen3.6 GGUF has forty pairs of routed Q4_K gate/up banks. Each 256-input block occupies 144 bytes: four FP16 header bytes (`d,dmin`), twelve packed per-subblock scale/minimum bytes, and 128 code bytes. An exact active-block metadata dictionary looks attractive because gate and up consume the same input, but the **complete forty-layer installed image has no repeated active sixteen-byte metadata packet at the same input block K** across any expert, output row, or gate/up bank. The twelve-byte scale/minimum tail alone also has no active repeat. This eliminates shared *whole metadata* decode/lookup at this coordinate without relying on a short route sample.

The CPU census reads all 83,886,080 Q4_K gate/up blocks from the pinned image. For each of 320 `(layer,K)` groups it compares 262,144 bitwise packets (both banks, 256 experts, 512 rows); different K positions cannot share the same activation-coordinate work. A header with both FP16 magnitudes bit-zero is called dormant, since the decoded weights are zero regardless of its code bytes. This gives 945,586 dormant and 82,940,494 active blocks. Across all groups there are 945,216 raw repeated full sixteen-byte packets and exactly 945,216 raw repeated twelve-byte tails; **zero repeats survive the active mask**. The raw counts are not incremental to the separately counted [joint dormant-row opportunity](qwen-moe-all-layer-dormancy.md). The active count also includes any all-zero integer-code blocks with nonzero affine metadata, which cannot be dismissed without checking the input sum.

The four-byte header *alone* repeats 26,560,689 times among active blocks. Granting free lookup, dictionary, indexing and byte-selective reading, deletion of one duplicate four-byte header per group saves at most **3,320,086 bytes per eight-expert token**, or **0.12642%** of the conditional 2,626,187,904-byte complete one-read stream under uniform eight-of-256 expert selection. This is a free byte comparison, not actual DRAM traffic, and may double-count neither full-metadata nor dormancy. A separately priced, fixed-width random-addressable header dictionary per `(layer,K)` stores each unique four-byte word, a packed `ceil(log2 distinct)`-bit index for every block and an eight-byte directory entry. Its bank image **grows 335,544,320 → 414,004,404 bytes**. A hybrid choosing raw headers independently by layer would save only 116,076 bank bytes (layer 0), or 3,627 bytes per token / 0.000138% of the complete stream before a mode selector and dependent lookup. These are *calculated grammar costs*, not a saved sidecar or an arbitrary-compression lower bound. Full metadata has zero active redundancy in this sharing grammar; any apparent full-packet win comes from already-dormant coordinates.

This bounded negative says not to build a gate/up metadata interner for the frozen Q4 image. It does not reject jointly learned changed codes, compressed cross-K labels with a directly priced consumer, lossy representations at paid complete-model quality, or scheduling/physical-time improvements to the existing grouped native kernel. The productive independent engine question remains ordinary **unprofiled router/Q8/expert** timing with complete heads; representation work needs a learned image on broader actual routed producer activations and disjoint whole-model loss, not another metadata lookup over frozen bytes. There is no native bit-identity or TPS claim.

## Receipt

[`tools/qwen-moe/q4_scale_packets.py`](../tools/qwen-moe/q4_scale_packets.py) checks the installed inventory's header hash, hashes both complete bank payloads at every layer, stores all 320 per-K active/raw counts, and joins forty source/inventory/tensor-hashed layer receipts to [the aggregate receipt](../../data/qwen-moe/q4-scale-packets/receipt.json). Acquisition pins the complete model hash. Reproduce without GPU:

```sh
python3 tools/qwen-moe/q4_scale_packets.py --first 0 --count 10
python3 tools/qwen-moe/q4_scale_packets.py --first 10 --count 10
python3 tools/qwen-moe/q4_scale_packets.py --first 20 --count 10
python3 tools/qwen-moe/q4_scale_packets.py --first 30 --count 10
python3 tools/qwen-moe/q4_scale_packets.py --summarize
```

No selected image, runtime, service or numerical map changed.

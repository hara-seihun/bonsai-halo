# Exact same-K Q8 fragment-dot reuse is negligible

The complete-bank [32-code census](qwen-moe-q8-block-reuse.md) found no nonzero repeated Q8_0 dots across output rows. It left a narrower question: can rows share *parts* of their integer dots even when the full vectors differ? `tools/qwen-moe/q8_fragment_reuse.py` checks all 250 installed nonembedding, nonexpert Q8_0 tensors (43,909,120 blocks; 1,492,910,080 image bytes). It partitions each 32-code block into aligned four- and eight-code fragments, and compares code vectors **within one tensor, one input-block K coordinate and one fragment position** across output rows. Each compared position therefore multiplies the same four/eight input codes at a given token. The full-bank source, inventory, header and each tensor payload are identified in the [receipt](../../data/qwen-moe/q8-fragment-reuse/receipt.json); the verified complete GGUF SHA-256 is `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61` in the acquisition record.

| Aligned fragment | Total positions | Repeated zero fragments | Repeated nonzero fragments | Free incremental code bytes beyond repeated zero blocks | Fixed-index bytes for every fragment |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 codes | 351,272,960 | 207,224 | **2,098** | **8,392** | 526,254,080 |
| 8 codes | 175,636,480 | 103,612 | **37** | **296** | 263,127,040 |

The zero-fragment repetitions are exactly 25,903 × 8 and 25,903 × 4: all are accounted for by the [previously counted repeated full-zero blocks](qwen-moe-q8-block-reuse.md). In this aligned-fragment dictionary family, granting free lookups, storage layout, detection and dot scheduling, the **incremental** reusable *nonzero* code payload is at most 8,392 bytes for four-code fragments, 0.0003195% of the complete 2,626,187,904-byte one-read token weight stream. At eight codes it is 296 bytes. Just 2,098/351,272,960 (0.000597%) and 37/175,636,480 (0.000211%) integer fragment dots could be shared in addition to the full-zero repeats, even before paying for a lookup. Four-code exact reuse is not a useful Q8 inference mechanism on this image. A fixed-width per-(K, fragment) row-index image would pay 526 MB at width four, dwarfing the free 0.837 MB total code redundancy (which includes the already-counted zeros); this index is one explicit paid grammar, not a compression lower bound.

On unrestricted inputs, equal fragment dot maps imply equal signed-code vectors: test the four/eight standard basis inputs. This is an exhaustive census for identical **aligned** fragment vectors, not a bound on nonaligned windows, learned labels, approximate codes, cross-K/cross-layer sharing or a whole-map transformation. Even reusing the same integer dot does not establish bit-identical Q8_0 FP32 scale application and accumulation if the kernel changes its reduction order. The byte fractions are logical one-read ceilings, not measured DRAM or a whole-model speedup. The source scan does not touch the GPU or installed runtime. The better next independent question is the ordinary Q8 consumer's operand staging, occupancy and calibrated physical traffic, rather than another exact code-dictionary variant; the phase measured 8.333 ms/token on the installed unprofiled decode panel.

CPU reproduction (one tensor at a time, resumable and splittable by range):

```sh
python3 tools/qwen-moe/q8_fragment_reuse.py --first 0 --count 250
python3 tools/qwen-moe/q8_fragment_reuse.py --summarize
```

[Aggregate plus 250 per-tensor hashed receipts](../../data/qwen-moe/q8-fragment-reuse/receipt.json) have aggregate SHA-256 `b676fd2fb606532e25726507c371a2a89d7fbec931c48f83413f6f28e60cade7`. The local image header is checked against the inventory; the acquisition receipt, not this scan, verifies the complete image hash. No GPU reservation, model image change, service stop or executable installation was involved.

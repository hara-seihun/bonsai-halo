# Adjacent-K Q8 scale pages do not beat same-K dictionaries

The selected Qwen3.6-35B-A3B GGUF's 250 nonembedding, nonexpert Q8_0 tensors have 43,909,120 FP16 scale words. The [existing paid image](qwen-moe-q8-scale-sharing.md) stores them in 58,793,258 bytes using independent same-K dictionaries and fixed-width output-row indices. This study tests the other natural axis: neighboring input-block scales in **one output row** often have related magnitudes. Can their exact bit patterns share an anchor cheaply enough to justify a different Q8 reader?

[`q8_scale_adjacent.py`](../tools/qwen-moe/q8_scale_adjacent.py) constructs a separately random-addressable page for every output row and consecutive group of up to sixteen K blocks. Each page stores its first original FP16 word (two bytes), one byte giving the maximum bit width of the remaining scale words XORed with that anchor, then those XOR values bit-packed at that width. A 32-bit absolute offset per page plus one terminal offset makes any scale accessible without decoding preceding pages. The reader needs one directory fetch, a width/anchor fetch, bit extraction and XOR; original Q8 code bytes are untouched. All 2,744,320 saved pages were decoded against their original 43,909,120 scale words as they were written. This is an exact *scale-bit* construction, not a native FP32 or throughput claim.

| Complete scale image | Bytes | Difference from same-K dictionary |
| --- | ---: | ---: |
| Original contiguous FP16 scale stream | 87,818,240 | +29,024,982 |
| Existing paid same-K dictionary | **58,793,258** | control |
| Adjacent-K sixteen-block pages, 70,189,882-byte payload + 10,978,280-byte offset directory | **81,168,162** | **+22,374,904** |
| Per-tensor best of the two complete images, with the chosen reader | **58,721,962** | **−71,296** |

Only **34/250** tensors favor the adjacent page. That hybrid saves **0.002715%** of the conditional 2,626,187,904-byte whole-model one-read stream, before paying for a second reader or selection. Even granting *free* adjacent-page offset directories, the best per-tensor combination is 57,980,526 bytes: its extra saving over the dictionary is only 812,732 bytes (**0.030948%** of the complete one-read stream). This is an optimistic image-rate bound for the specified sixteen-block, one-anchor, one-width-per-page grammar, not an ISA-independent impossibility bound. The explicit offset directory is particularly expensive because it has one entry per output-row page; a compressed offset system may reduce it, but cannot turn this grammar into a material whole-model traffic reduction against the dictionary. No scale approximation was introduced, and its saved image does not reduce integer dot work.

The full adjacent image **does** save 6,650,078 bytes relative to untouched FP16 scales. The negative is relative to the existing, cheaper, exact same-K construction, not relative to raw. This is why an appealing local correlation is not enough: per-column reuse across output rows captures far more of the Q8 scale redundancy than this row-local anchor captures after its random-access costs. The next native question is physical Q8 traffic and stalls on ordinary unprofiled calls, not a second indirect scale decoder for a 0.0027% conditional whole-stream hybrid rate point. If real transactions show scale bytes dominating, try a consumer that jointly prices dictionary lookup and code-dot latency; no such speedup follows from either image.

## Contract and custody

The observation is every original two-byte scale word in the pinned installed Q8_0 tensors. The image's XOR and bit packing are exact integer maps; the original 32 int8 code bytes per block remain unchanged. For each tensor the producer records the installed GGUF payload SHA-256, the source/inventory hashes, page dimensions, exact payload and directory sizes, width histogram and saved image SHA-256. The [complete receipt](../../data/qwen-moe/q8-scale-adjacent/receipt.json) retains all 250 receipt/image hashes and the installed header and model identity; the pinned acquisition receipt owns the full GGUF digest. Raw complete `.scales` images and per-tensor JSONs live beside it. All pages were checked against the source words before publication. No GPU, runtime, resident service or model image was changed.

Regenerate in bounded CPU-only batches from this repository:

```sh
python3 tools/qwen-moe/q8_scale_adjacent.py --first 0 --count 50 --image
python3 tools/qwen-moe/q8_scale_adjacent.py --first 50 --count 50 --image
python3 tools/qwen-moe/q8_scale_adjacent.py --first 100 --count 50 --image
python3 tools/qwen-moe/q8_scale_adjacent.py --first 150 --count 50 --image
python3 tools/qwen-moe/q8_scale_adjacent.py --first 200 --count 50 --image
python3 tools/qwen-moe/q8_scale_adjacent.py --summarize
```

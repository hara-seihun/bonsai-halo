# An exact, paid coordinate for Qwen's nonexpert Q8 scales

The selected one-row Q8 consumer pays 8.333 ms/token in the unprofiled depth-1024 device panel. Its 250 nonembedding, nonexpert Q8_0 tensors contain 43,909,120 34-byte blocks, of which each has one 16-bit scale and 32 signed codes. The existing [same-K code census](qwen-moe-q8-block-reuse.md) found no repeated nonzero int8 dot vectors. This experiment instead asks whether the **scale words**, without reusing any dot, admit a cheaper exact image at a shared input-block coordinate.

Across the complete installed image, there are 4,861,045 distinct `(tensor, K-block, FP16-bit-pattern)` instances and 39,048,075 repeat instances. This is not 39 million reusable dots: the 32 codes and their row-dependent int32 products remain distinct. A free scale interner could eliminate at most 78,096,150 bytes, 2.974% of the conditional complete one-read weight stream. It requires a lookup for every block, so this is not a native traffic saving.

## Constructed paid image

[`q8_scale_sharing.py`](../tools/qwen-moe/q8_scale_sharing.py) stores for each tensor and K-block a sorted dictionary of the **original two-byte scale bit patterns**, then a bit-packed fixed-width dictionary index for every output row. Each K-block has an eight-byte `(dictionary length, absolute offset)` directory entry. The code bytes are retained unchanged; the scale sidecar is written to 250 `.scales` files and decoded **for every scale word** against the installed GGUF. Thus the complete assembled Q8 image has the original code stream plus the paid sidecars; no weight approximation or metadata-free dictionary is being counted. The directory takes 133,120 bytes.

| Quantity | Bytes |
| --- | ---: |
| Original 43,909,120 Q8 scale words | 87,818,240 |
| Paid dictionary + indices + all directory entries | 58,793,258 |
| **Lossless image saving** | **29,024,982** |
| Conditional complete one-read stream saving | **1.10521%** |

The untouched Q8 code stream is 1,405,091,840 bytes. Only four of 250 per-tensor sidecars are larger than their original scale streams; per-tensor selection could keep those original at a negligible additional mode cost, but the table charges *all* sidecars. The 16,640 input-block directory entries and every scale image are present, not estimated. A more powerful ideal zero-overhead entropy coder with a free per-K probability model would reduce the original scale side alone by 45,278,817 bytes; that number omits model/index/decoder cost and is not a compressed image. The paid construction occupies a real point between that ideal and the no-index duplicate count.

Within the declared grammar (unchanged 32 signed code bytes per block, same-K scale dictionary, fixed-width row indices, complete original scale bit reconstruction), the saved 29.0 MB is an exact image-rate result. It does **not** eliminate one of the two integer dot products, the per-row FP32 fold, the Q8 kernel launches, or the repeated Q8 code traffic. A new packed consumer must pay the dictionary fetch and row-index extraction, which can lose more time than these bytes save. Reconstructing the FP16 scale bits proves the decoded scale coordinate; it is not a GPU FP32 logit identity or a complete-model speed result. The selected model, installed executable and Bonsai service did not change; the GPU was not used.

The next engine experiment should measure ordinary Q8 physical transactions and issue/wait time on the *unprofiled* one-row path before considering an indirect scale reader. If scale traffic proves material, compare this actual paid sidecar with the unchanged dense Q8 baseline in a whole-model decode/prompt/batch panel under the GPU wrapper, preserving the original int32 dot and FP32 fold order. The conditional 1.105% whole-stream byte difference is not a reason by itself to port it.

## Custody

`../../data/qwen-moe/q8-scale-sharing/receipt.json` records the source/inventory/header/model identities, all 250 tensor payload hashes, per-tensor receipt hashes, all sidecar hashes, paid bytes and counts. The pinned acquisition receipt owns the complete GGUF SHA-256. Reproduce CPU-only in bounded tensor ranges from the Bonsai repository:

```sh
python3 tools/qwen-moe/q8_scale_sharing.py --pack --first 0 --count 50
python3 tools/qwen-moe/q8_scale_sharing.py --pack --first 50 --count 50
python3 tools/qwen-moe/q8_scale_sharing.py --pack --first 100 --count 50
python3 tools/qwen-moe/q8_scale_sharing.py --pack --first 150 --count 50
python3 tools/qwen-moe/q8_scale_sharing.py --pack --first 200 --count 50
python3 tools/qwen-moe/q8_scale_sharing.py --summarize
```

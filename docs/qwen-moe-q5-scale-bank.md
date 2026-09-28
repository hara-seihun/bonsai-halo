# Exact Q5_K routed-down scale bank: paid rate and hot-path boundary

The pinned Qwen3.6-35B-A3B GGUF contains 37 Q5_K routed-down tensors (layers 0–33 and 35–37; three other layers are Q6_K). Each Q5_K 256-weight block stores 172 code/min-scale bytes and two FP16 superblock words `d,dmin` in four additional bytes. This study asks whether sharing those two scale words **across all 256 experts and 2,048 output rows in each layer** pays enough physical rate to warrant an indirect down consumer. It is not another search for repeated quantized dot vectors or a change to the selected GGUF.

## Exact image and observation

For each layer, assign an independent 16-bit dictionary to the `d` and `dmin` bit patterns. Replace each four-byte pair by one packed fixed-width `(d_index,dmin_index)` word in original block order. All 172 other bytes in each block are unchanged. The saved sidecar has a 32-byte header, two complete FP16 dictionaries, and one bitstream, with no entropy estimate or free directory. Because each Q5_K tensor has exactly 1,048,576 blocks, its row/block ordinal provides direct bit-offset addressing; a lookup requires a packed bit extract **and two dependent dictionary reads** before the unchanged integer dot/fold. All zero, subnormal and NaN bit patterns are preserved as opaque 16-bit words. The offline checker decodes every saved index and compares both resulting words bitwise with the source. This proves equality of the reconstructed selected weight image, and therefore the same complete model if the original decoder receives that reconstruction; it does **not** prove a native indirect reader retains the FP32 operation schedule.

| Installed Q5_K down scope | Bytes |
| --- | ---: |
| All 37 routed-down banks, including codes | 6,828,326,912 |
| Original `d,dmin` words | 155,189,248 |
| Paid 37 sidecars | 108,568,374 |
| Exact bank-image saving | **46,620,874** |
| Saving at eight of 256 experts/token | **1,456,902** |
| Fraction of complete modeled 2,626,187,904-byte one-read token stream | **0.0554759%** |

The first fourteen layers mostly require 10 + 12 index bits per block; later layers often require 11 + 12. Dictionaries and headers are charged, so the paid bank saving is less than the ideal two-byte/word duplicate count. The reported selected saving scales eight/256 of whole-bank bytes; it is a **conditional logical one-read comparison**, not measured DRAM traffic or a kernel/whole-model speed result. At 242 GB/s, even deleting these bytes with zero lookup cost buys only about 0.0060 ms/token of that conditional bandwidth budget. This is not a universal bound: other weight bytes, changed code layouts, cross-layer representations, more reuse or different bottlenecks are outside the claim. In this exact sidecar program, index extraction and two dependent scale fetches are a substantial new hot-path obligation against the small byte opportunity. Do not port this coordinate for single-token decode solely to save stream bytes.

## Reproduction and custody

`tools/qwen-moe/q5_scale_bank.py` reads the inventory's hashed GGUF header and each complete tensor payload, writes 37 independently addressed scale sidecars, checks every one of **77,594,624** reconstructed FP16 words, and emits per-tensor source/inventory/payload/image-hashed receipts. The [aggregate receipt](../../data/qwen-moe/q5-scale-bank/receipt.json) also pins the acquisition hash of the complete GGUF. With the existing model and `traffic.json`:

```sh
python3 tools/qwen-moe/q5_scale_bank.py --first 0 --count 13
python3 tools/qwen-moe/q5_scale_bank.py --first 13 --count 12
python3 tools/qwen-moe/q5_scale_bank.py --first 25 --count 12
python3 tools/qwen-moe/q5_scale_bank.py --summarize
```

CPU only. No GPU reservation, installed runtime, selected image, numerical default or resident service changed. The result redirects this exact-scale avenue away from a reader port. The next useful native Qwen question remains ordinary Q8 and expert operand physical traffic/latency; a representation proposal should change the expert's **codes or consumer** enough to exceed the ~0.055% conditional saving, then pay complete-image rate and held language quality before runtime work.

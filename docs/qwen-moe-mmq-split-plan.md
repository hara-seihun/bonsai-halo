# The naive two-width Qwen MMQ grid spends 95% of its entries returning

The [two-width route bound](qwen-moe-mmq-two-width.md) makes J=16/64 attractive at 126 prompt rows: 112.90 million gate/up full-tile positions instead of 328.14 million, without another logical image pass. But the native MMQ launcher does not launch one block per occupied expert tile. On gfx1151 its non-stream-K grid is `(ceil(R/I), ceil(T/J), 256)`, where `T` is the whole prompt width, not the largest group assigned to that body. The kernel checks `expert_bounds` and returns for empty experts and excess columns. Reusing that launcher twice, with a filter for each width, is a much larger scheduling change than the tile count suggests.

On the disjoint held 126-row capture across forty layers:

| Gate/up route | Launches | Launched grid entries | Entries reaching an assigned tile | Entries returning | Active fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Selected J=64 | 40 | 163,840 | 40,056 | 123,784 | 24.45% |
| J=16/64 using two full expert grids | 80 | 1,474,560 | 75,088 | 1,399,472 | 5.09% |

Of the split's returning entries, 760,608 belong to experts absent from the tile, 144,176 belong to an expert assigned to the other body, and 494,688 are columns past the assigned group's size. The selected J=64 grid has 84,512 absent-expert entries and 39,272 column-tail entries. Train at 113 rows gives 163,840 / 39,256 selected entries and 1,474,560 / 74,304 split entries. These are **scheduled grid entries**, not dots, memory bytes, device cycles or measured throughput. An early return is cheap relative to a packed matmul, but 1.4 million returns are not free, and the small body has twice as many output-row tiles.

There is a direct exact construction for a native experiment. `mm_ids_helper` already sorts every token/slot by expert and writes `ids_dst` and `expert_bounds`. Keep those arrays and the one activation quantization. After bounds are complete, make one compact `(expert, column_tile)` task list for each width, choosing a width from the group's size. A task indexes the existing sorted `ids_dst` and Q8 activation columns; its expert index addresses the original packed weights and output stride. For the held prompt, there are exactly 5,007 tasks, unchanged from selected J=64. The two bodies execute 75,088 gate/up row/column blocks, without a second sort or activation conversion. Each expert's assignments remain in the native order and belong to exactly one body, so the current I/J tile's K order, masking and FP32 reduction stay intact. Gate/up and down can use the same partition, but down's distinct post-SwiGLU activation must still be prepared; it cannot reuse gate/up's Q8 operand.

The task list is **not** free. A device pass must inspect 256 group bounds per layer, assign widths and create tasks after `mm_ids_helper` finishes. A fixed-size persistent work queue avoids a CPU readback for variable task counts but pays a task claim and synchronization per tile; two bodies and their added launch gaps remain. Alternatively, adding a width filter to two existing full grids avoids list construction but pays the 1,399,472 returns counted here. Neither has native timings yet. The native `mul_mat_q` currently seeds `ids_dst_shared` before checking group bounds; an implementation must move the check before that setup or use compact indices, not assume the returns have no instruction cost. The next experiment should measure both dispatch forms on complete prompts with equal output-head work, exact installed logits and independent-stream generation; a favorable tile count alone does not select one.

The CPU [receipt](../../data/qwen-moe/all-layer-routes/mmq-split-plan.json), SHA-256 `4af85476a3fb431c7a7622e89f4379ae9ffdc5d0d5b8e7405d5c9886fb26a172`, hashes the source, occupancy, model, capture receipt and the native source recorded by occupancy. Recreate it without the GPU:

```sh
python3 tools/qwen-moe/mmq_split_plan.py --output ../../data/qwen-moe/all-layer-routes/mmq-split-plan.json
```

The count follows `mmid.cu:mm_ids_helper`, `mmq.cu:ggml_cuda_op_mul_mat_q` and `mmq.cuh:launch_mul_mat_q`/`mul_mat_q` in the pinned native source. It assumes the non-stream-K gfx1151 configuration and one full expert grid per native body. No model image, installed runtime, service or GPU state changed.

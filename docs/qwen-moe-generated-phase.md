# Qwen generated 32-stream phase: recurrent state outranks Q8

On the selected on-demand `3f552c2` runtime, two consecutive actual generated
32-stream steps (each with **32 complete vocabulary heads**) spend about **53 ms
of 139 ms summed device time** in the 30 GDN kernels plus their 30 large state
row gathers. This is more than the routed expert kernels (~40 ms) or all 250
nonexpert Q8 projections (~24 ms). The 30 state gathers alone take **19.0–19.2
ms**; they are a concrete independent target that the serial one-row Q8 phase
study could not expose. Prioritize an exact multi-sequence recurrent-state
consumer before another Q8 one-row preload or expert-J tile width.

## Paired real generated workload

The existing `generated_stream_j16` probe seeds 32 independent sequences with
eight shifted natural-text tokens, then feeds back each stream's actual greedy
decision twice. Both arms use the same installed executable and pinned
22,134,528,992-byte GGUF, context capacity 8192, occupied length 8–10,
batch 512, ubatch 256, 8 CPU threads, flash attention and full offload. The
control sets `GGML_CUDA_MMQ_EXPERT_J=0` (global J32); the selected arm uses J16
at 32 routed rows. Each step requests and copies every head row. Kernel trace
has no PMCs; the first 256-row seed ends at the first full Q6_K head, and
both 2,175-dispatch generated steps end at their own full head. The 96
full-vocabulary rows in both arms have identical SHA-256
`359aebfd4af6ee361f620920c4069aa688f5a6fe0832f4f40995df6d3ed68fb1`.

| Summed kernel device ms per generated step | J32 step 1 / 2 | J16 step 1 / 2 |
| --- | ---: | ---: |
| GDN recurrent core, 30 calls | 33.143 / 33.421 | 33.621 / 33.932 |
| GDN-associated indexed state gather, 30 large calls | 18.794 / 19.107 | 19.148 / 19.219 |
| Other indexed float gather, 30 small calls | .551 / .535 | .566 / .571 |
| Routed Q4/Q5/Q6 expert projections, 120 calls | 40.848 / 43.070 | 37.409 / 39.843 |
| Nonexpert Q8, 250 calls | 23.596 / 23.784 | 23.831 / 24.384 |
| BLAS GEMMs, 100 calls | 9.609 / 9.867 | 9.399 / 9.604 |
| Q6 full vocabulary head, one call | 2.382 / 2.413 | 2.416 / 2.516 |
| Other kernels | 9.797 / 9.892 | 10.276 / 10.648 |
| **Sum** | **138.721 / 142.089** | **136.665 / 140.716** |

The J16 expert calls save 3.33 ms on the two-step average; other device work
varies enough to leave 1.71 ms net in this *trace*. Unprofiled whole-model
rate is owned by the [four paired generated-stream panels](qwen-moe-generated-stream-j16.md),
not by this ratio. Here the traced step wall times are 156.5/177.2 ms for J32
and 150.5/178.1 for J16: no new clean wall-rate claim follows. The GPU wrapper
reports 2353 MHz median and 2.8 competing host cores; device durations exclude
host graph/build gaps. No DRAM traffic counter was taken, and no image/kernel
was changed.

## Why the gather is actionable

The installed source `3f552c2:src/models/qwen35moe.cpp` passes the recurrent
state through `build_rs` before the GDN consumer. Its `llama-graph.cpp`
implementation calls `get_state_rows(ctx0, states, state_copy_main)`; the
installed `getrows.cu` vector kernel reads a selected row and writes the same
`int4` to a temporary. The trace has one `k_get_rows_float_vec<float>` per
GDN layer with grid **(8192, 512, 1)** threads and workgroup (256,1,1), plus
one small (8192,24,1) gather. The large grid means 32 selected rows, each
524,288 FP32 state elements (`128*128*32` from `n_embd_s()`), or **67,108,864
written bytes per layer / 2,013,265,920 per 30-layer step** before the
consumer reads them. This is logical copy payload, not measured DRAM bytes.
Its ~19 ms copy time alone is 13.7% of the selected trace's summed device
step; deleting the kernel for free is an upper-bound thought experiment, not
a speedup claim.

There is already a direct-row GDN variant in the adjacent `qwen35.cpp` model,
but its source explicitly notes a multi-sequence hazard: relocation of extra
cache rows can overwrite a main row before the deferred consumer reads it.
The `qwen35moe.cpp` installed path still uses eager `build_rs`. **Next native
construction:** keep the selected GDN FP32 arithmetic, read each main row by
its original state index in the consumer, and enforce the original
main-read-before-extra-relocation order for all 32 sequences (or arrange
nonoverlapping source/destination storage); compare full logits and state
transitions before accepting a wall panel. Do not simply switch to the
single-sequence direct-row mode. If the 19-ms gather vanishes but the GDN
core grows by the same amount, its state-access locality was being purchased
by that copy; measure the combined phase, not the missing launch alone.

[Raw per-arm CSV, all 96 full heads, stdout/stderr and wrapper log, plus
source/probe/library/input-hashed analysis receipt](../../data/qwen-moe/generated-phase/receipt.json)
are retained. Regenerate the receipt from the two CSVs with
`python3 tools/qwen-moe/generated_phase.py ../../data/qwen-moe/generated-phase`.
Both `rocprofv3 --kernel-trace -f csv` runs occurred inside **one**
`tools/run-batch-compare --runtime-max 44s --memory-gib 36 --host-reserve-gib 6`
hold; the wrapper restored `bonsai-halo.service` and released the GPU lock.
The selected serving/runtime image is unchanged.

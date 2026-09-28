# A shared MoE input does not make a large missing Qwen prep phase

Qwen's routed eight and its ninth shared expert consume the **same post-attention `cur` tensor** in `qwen35moe::graph::build_layer_ffn`. The selected one-row HIP execution nevertheless quantizes this F32 input twice: once for the fused routed Q4_K gate/up MMVQ and again for the shared Q8_0 gate/up MMVQ. Can one Q8_1 coordinate serve both consumers, and how much of decode could that remove?

**Yes for the one-row input coordinate; at most a small local prize.** `ggml_cuda_mul_mat_vec_q` allocates its own `block_q8_1` buffer and calls `quantize_row_q8_1_cuda` on `src1` in both cases. This quantizer explicitly ignores the weight type (`GGML_UNUSED(type_src0)`), and the two gate/up input shapes are one F32 row of 2,048 with the same strides and padding. In the selected GGML graph `build_moe_ffn(cur, ...)` receives `cur` by value; the separate `build_ffn(cur, ...)` for the shared branch receives that same input, not a routed output. Their encoded 64 × 36-byte blocks can therefore be constructed once and passed to both one-row readers, preserving each consumer's *input codes* exactly. This is an exact coordinate construction at the quantizer boundary, not a claim that a changed graph/lifetime or complete logits have been installed and accepted. Prompt MMQ uses scatter and potentially different Q8_1 scale layouts; the proof does not cover that path.

The retained depth-1024 full-head decode trace has two setup heads and eight 1,565-dispatch decode spans. [`shared_prep_bound.py`](../tools/qwen-moe/shared_prep_bound.py) sorts by **dispatch ID** (the CSV file rows are not consistently ordered), then locates all forty fused top-k → routed Q4 gate/up → routed down → shared Q8 gate/up sequences in each span. It checks a distinct 2,048-input quantizer immediately before each gate/up MMVQ. The seven warm spans yield:

| Measured device work per decode token | Mean | Per-token duplicate range |
| --- | ---: | ---: |
| Routed gate/up input quantization, 40 launches | 0.101899 ms | — |
| **Duplicate shared gate/up input quantization, 40 launches** | **0.089414 ms** | **0.083794–0.092476 ms** |
| All traced device kernels, same complete-head spans | 20.846359 ms | — |

Even a **free** perfect reuse deleting the forty shared quantizer launches would remove just **0.4289% of the summed traced device duration**, while leaving the 40 Q8 shared gate/up matrix kernels, all eight routed products, shared gate/down, router, state and vocabulary head intact. The duplicated Q8_1 payload is 92,160 logical bytes/token across forty layers, against the 2,626,187,904-byte conditional complete one-read weight stream (0.00351%). This does not bound possible *fusions that also remove matrix work or synchronization*, only this isolated same-input preparation. Nor is a profiled-device fraction a whole-model wall-speedup bound: the graph may overlap or reallocate scheduling costs, and a native reader may need longer buffer lifetime or an extra conversion. No runtime/default, model image, GPU service or full-model map changed.

The [source/trace and all 320 paired dispatches receipt](../../data/qwen-moe/shared-prep-bound/receipt.json) has SHA-256 `d27c40cb8beb9aaadc9ec65d51e1cd4218df9608ad494fc0a6500b7205c71022`; it records the trace hash, script hash, native source hashes at both trace `b4c67ced9` and selected `7861dc7` revisions, and every per-token duration. Reproduce without GPU:

```sh
python3 tools/qwen-moe/shared_prep_bound.py --output ../../data/qwen-moe/shared-prep-bound/receipt.json
```

**Next:** do not port a second Q8_1 buffer-lifetime mechanism for a 0.09-ms isolated target. Ordinary Q8 matrix execution costs ~8.4 ms/token in the low-overhead event study, and routed expert matvecs ~4.5 ms/token; prioritize actual unprofiled physical traffic/reader behavior on those consumers, or a changed paid expert representation selected by broad routed producers and complete-model language quality. A fusion worth revisiting must eliminate more than the duplicate quantizer while preserving the complete native map.

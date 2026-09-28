# Early-issued Q8 next-K loads: native whole-model null

The [compiled construction](qwen-moe-q8-asm-lookahead.md) really issues two next-block loads before the current integer dot, whereas the ordinary C++ preload is sunk after the current fold. This panel applies its retained source patch to the **selected** native Qwen source `3f552c2ef`, yielding experimental `06a917a690` (only `ggml/src/ggml-cuda/mmvq.cu` changes). The candidate was built, packaged and retained in `../../data/qwen-moe/runtime/06a917a6904762f67f20ac9ff31c4547ff40dc36/`, but **not selected**.

The complete matched 465-token prompt and 32 plain greedy target steps reproduce the installed reference: all **7,946,240 FP32 vocabulary-logit bits** and all 32 IDs agree (`qwen-accept`, sixteen prompt repeats). The first two exploratory runs used three and seventeen repeats (101 and 493 prompt tokens), so their attempted comparison with the 465-token fixture was invalid; the correct sixteen-repeat run is retained beside them. Its raw log shows 465 prompt tokens. This establishes the tested path's finite output on that acceptance panel, not a universal proof of the inline assembly's equivalence.

Four whole-model, full-head `llama-bench` decode processes per panel used the **same** 22.13-GB image, occupied depth 1024, batch/ubatch 512/256, flash attention, full offload, eight host threads, 64 generated tokens, one repetition per process, no warmup and baseline/candidate/candidate/baseline order. Output is complete target-head decode, not a Q8 kernel microbenchmark:

| Clock policy | Baseline TPS (two processes) | Candidate TPS (two processes) | Candidate / control mean |
| --- | --- | --- | ---: |
| Automatic | 50.739, 51.894 | 50.718, 50.376 | 0.9850 |
| Fixed 2400 MHz | 49.538, 49.256 | 49.550, 49.450 | 1.0021 |

Both wrapper panels reported competing host work (1.8 and 2.3 extra busy cores); the automatic panel also reported package-power limiting for 7% of its window. These measurements **do not establish a useful whole-model gain**: the fixed-clock repeat is near null and the automatic panel favors the baseline. The compiled load issue order alone was not enough to justify a native port. No claim about physical DRAM transactions follows from these wall timings; the added terminal branch, waits and register demand remain possible costs, not experimentally isolated causes. The selected on-demand runtime, image and Bonsai serving default are unchanged. Because this arm did not win its targeted single-stream decode shape, no prompt or generated multi-stream adoption panel was run.

[Raw candidate full-head logits, IDs, all eight JSONL samples, complete build/wrapper logs and source/library/output hashes](../../data/qwen-moe/q8-next-k/native-panel/receipt.json) have receipt SHA-256 `06763a54e4eb9ee131c2a928ca7928dd65a432da41dc2e748dcfdb4116cd55bf`. Both benchmark wrapper calls and acceptance restored `bonsai-halo.service`; the GPU lock was released.

**Next question:** distinguish request latency from stream bandwidth on ordinary Q8 one-row calls with an unprofiled paired device-time/physical-transaction panel. The no-gain result is specific to this cyclic branch/next-K grammar and machine; a branch-free prologue/steady/epilogue would be a different program, but it should be attempted only after finding a real load-latency interval worth covering. This rejects neither alternative packed whole-map consumers nor MoE route-aware representations.

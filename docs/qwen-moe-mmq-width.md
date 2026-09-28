# Routed Qwen MMQ tile width on the GPU

The native grouped-expert prompt path was paying for empty columns. On real held 126-token routes, [the source-accounted panel](qwen-moe-mmq-work.md) found that J=64 instead of the selected J=128 executes 50.5% as many full-tile output positions, with 0.99% more logical expert-image passes. That count did not price the device. We forced the already compiled J=64 body in the pinned HIP runtime and measured the complete model.

The accepted revision `f60e4fbfabe735b15b4c74bfb1c9392f07329d2b` selects J=64 for routed Q4_K/Q5_K/Q6_K MMQ with 65–256 prompt rows. `GGML_CUDA_MMQ_EXPERT_J=0` restores the original global-width dispatcher on the same binary. Other quant families and the non-routed path keep their selection. Single-token decode uses MMVQ and is untouched. A forced J=16 at 9–32 rows remains a diagnostic, not a default.

| Complete-model shape | Original J, tokens/s | Smaller J, tokens/s | Result |
| --- | ---: | ---: | --- |
| 512 synthetic prompt tokens, batch 512, ubatch 256, source build | 887.76 | 1141.99 | +28.6% |
| Same shape, packaged binary | 872.92 | 1138.18 | +30.4% |
| 128 prompt tokens, source build | 558.27 and 564.92 | 743.05 | Noisier clock-mismatched lead |
| 32 prompt tokens, source build, J=32 vs J=16 | 408.10 | 420.65 | Too small and noisy to select J=16 |
| Depth-1024, 32-token plain decode, source build | 51.55 | 52.05 | MMVQ control |
| Same decode shape, installed candidate | not repeated in this panel | 51.58 | No observed decode regression |

Each 512-token arm contains three raw samples. The paired 512-token panels have near-equal median shader clocks, 2723/2737 MHz and 2689/2704 MHz. Host load and clocks change between panels; a later original-width panel ran at only 2421 MHz and is retained rather than averaged into the paired comparison. The [raw wrapper records and source/model/binary-hashed receipt](../../data/qwen-moe/mmq-width/receipt.json) contain every sample and failed diagnostic. These are whole-model prompt numbers, not a kernel ratio. They include routing, activation preparation, all layers and the requested vocabulary projection. They do not establish aggregate independent-stream generation throughput. The static tile-position ratio alone did not predict the measured rate.

Numerical acceptance uses the unchanged 22.13 GB image, 465 real prompt tokens, 32 greedy continuation steps, and the selected runtime's full-vocabulary reference. The packaged candidate and, after selection, `runtime/current/bin/qwen-accept` each reproduce **all 7,946,240 FP32 logit bits and 32 token IDs**. Both GPU-wrapper runs restored `bonsai-halo.service` to active. The [installed verification log](../../data/qwen-moe/mmq-width/acceptance/installed-selected.log) and captured logits/tokens remain beside the reference. Prompt throughput is a separate synthetic-token `llama-bench` workload.

One early original-width 465-token diagnostic produced a different entire greedy trajectory, then two original-width repeats and two J=64 runs matched the selected reference exactly on their four-step prefix. Its divergent floats and tokens are retained under `acceptance/base465.*`. It did not recur in installed 32-step acceptance. Do not infer from that transient a semantic change caused by J. The next useful engine check is an interleaved multi-sequence generation panel and expert-kernel device durations at J=64/J=128, including actual group sizes and physical image traffic. A wider J=64 policy or per-expert adaptive mixed widths still requires its own complete-model evidence.

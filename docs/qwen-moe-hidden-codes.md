# Real routed hidden codes do not save meaningful decode traffic alone

The selected Qwen3.6 layer-0 capture has the post-SwiGLU 512-vector for each of eight routed experts on 113 train and 126 held text tokens. I asked whether packing **that vector alone**, before the down projection, could make the routed computation cheaper. This is a useful boundary to close: the native Q5_K down matvec already takes quantized activation input, and a smaller intermediate must either save enough traffic to pay for conversion or feed a different packed dot directly.

I accessed the full 256-expert layer-0 routed Q5_K down bank in the pinned GGUF and decoded each expert actually selected in either capture using the selected runtime's `libggml-base`. For each selected `(token, expert)`, I evaluated the captured hidden vector and its group-32 signed code through the *same* dequantized matrix, then combined all eight down outputs with the captured router scores. This includes every observed routed expert, not the first-sixteen BF16 sample. The local reference is the FP64 score-weighted sum of FP32 products with unmodified captured hidden vectors. Both code families use one FP16-rounded scale per 32 coordinates, nearest-even integers, and the same clipping choices 1, .75 and .5 times the group's absolute maximum. Train chooses the clip; held text is separate.

| Code and clip | Bytes per expert hidden | Train routed-sum RMS vs FP32 down | Held routed-sum RMS vs FP32 down | Held RMS vs native captured routed sum |
| --- | ---: | ---: | ---: | ---: |
| signed Q8, max (train choice) | 544 | .00644 | .00686 | .01365 |
| signed Q4, max (train choice) | 288 | .08392 | .08011 | .08141 |
| signed Q4, .75 max | 288 | .21669 | .22045 | .22087 |
| signed Q4, .5 max | 288 | .42967 | .43855 | .43870 |

The native captured sum and the offline unmodified FP32 Q5_K sum themselves differ by .01345/.01513 train/held RMS. The runtime's Q8_1 quantization, dot/reduction order and FP32 execution are not reproduced by this offline dot. Thus the Q8 row is a matched *offline* control, not a bit-exact replay of the existing kernel. The large losses on tighter clipping reflect these actual post-SwiGLU vectors; the synthetic shared-input clipping result does not transfer to a down-hidden clip rule.

The traffic result is harsher than the distortion result. A matched Q8-to-Q4 group-32 code saves 256 bytes per selected expert, 2,048 bytes for eight experts per layer. If the same opportunity existed in all forty layers, the saving would be 81,920 bytes per token, **0.00312%** of the 2,626,187,904-byte modeled complete weight stream. Even within the down phase, its eight Q5_K expert images require 5,767,168 bytes per layer, so this hidden-buffer saving is 0.0355% of down weight bytes. Those are upper bounds on the *modeled traffic reduction*, assuming every hidden code would otherwise reach DRAM once. They say nothing about occupancy, instruction throughput or cache. Actual hidden intermediates may remain cached, and a Q4 consumer needs a changed dot or expansion back to Q8. The quantizer itself must find sixteen maxima and round 512 codes per expert, and a direct Q4 path must pay the different instruction schedule and scales. The native Q8_1 format is not this equal-scale Q8 control.

This closes standalone post-SwiGLU buffer packing as a decode-bandwidth attack for the selected image. The useful next question is a **joint weight/activation code** that reads fewer than the 5.77 MB of down weights per layer and consumes its hidden label directly, or an output-aware expert-bank quantizer evaluated under the complete model. Reducing two kilobytes while keeping Q5_K weights untouched cannot move the 2.626 GB modeled stream. The dense sub-bit pilot is a warning here too: an isolated routed-sum error, even if it improves, does not certify language quality. No runtime, complete-model logits, service or GPU measurement changed.

`tools/qwen-moe/hidden_codes.py` owns the CPU reduction. Its [receipt](../../data/qwen-moe/hidden-codes/receipt.json) hashes the script, all four input arrays per split, Q5_K bank, GGUF inventory, model acquisition and decoder library. Run from the Bonsai repository:

```sh
OPENBLAS_NUM_THREADS=4 python3 tools/qwen-moe/hidden_codes.py \
  --output ../../data/qwen-moe/hidden-codes/receipt.json
```

The calculation's observation contract is the local FP64 score sum of dequantized Q5_K FP32 products. No bit-identity or whole-model inference claim follows from that contract.

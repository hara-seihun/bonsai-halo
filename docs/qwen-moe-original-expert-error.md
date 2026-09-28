# Real routed original-vs-installed layer-0 expert response

On the pinned Qwen3.6-35B-A3B layer-0 actual post-attention producers, the installed Q4_K gate/up and Q5_K down images differ materially from the official BF16 expert map. This is a **matched local representation diagnosis**, not a replacement image. In 64-token disjoint prompt captures, official experts 0–15 occur in 25 train and 26 held selected slots (24 and 25 distinct tokens); the other selected experts are not included because their original tensors were not fetched. All 16 original experts are from the official pinned BF16 slice. The installed GGUF image, router scores, IDs and 2,048-wide producer activations are unchanged.

The offline map computes `h_e = SiLU(x G_e^T) * (x U_e^T)` and `d_e = h_e D_e^T`. For each matched slot it compares (i) all official BF16, (ii) installed Q4_K gate/up with official down, (iii) official gate/up with installed Q5_K down, and (iv) all installed. The score-weighted differences are accumulated into the same output coordinates. Products/SwiGLU use FP32 BLAS and weighted accumulation uses FP64; these are *not* the selected engine's Q8_1 inputs, reduction order or FP32 logits. Unmatched experts are held fixed, so their deltas cancel. Each arm costs only the existing model images as a comparison, not a newly paid candidate image. The captured native complete routed sum is a scale for the partial error vector, not the original BF16 complete-map denominator.

| Split (selected slots) | All installed / official matched-sum RMS | Gate/up only | Down only | All installed delta / native complete routed-sum norm |
| --- | ---: | ---: | ---: | ---: |
| Train (25) | .05695 | .04978 | .02774 | .00666 |
| Held (26) | **.04532** | **.03577** | **.02811** | **.00838** |

On held matched slots, median per-slot output errors are .05006 all installed, .04250 gate/up-only and .02908 down-only; aggregate slot-relative RMS is .05624/.04862/.02826. The gate/up–down interaction vector is .00112 of the official matched-sum norm. Median first-16 decoded **weight** relative RMS is .07407 gate, .07402 up and .04266 down. These are genuinely different tensors and bit rates, not an equal-byte format comparison. The local native-vs-offline fully-installed down-output median across observed experts is .02951 on held, so the offline maps must not be interpreted as native logit or language-loss changes. No accepted runtime/default/service changed, and the GPU was not used.

The result changes the next representation question: the Q5 down-only synthetic scalar-gain negative missed both real producer distribution and the Q4 gate/up contribution. A single down-only recode cannot be selected from a local reduction in error when the gate/up arm is larger here. Acquire broad official routed expert tensors and producers, compare **equal paid complete images**, train gate/up and down jointly against composed outputs, then require held complete-model language loss and direct-consumer cost before changing the native GGUF path. The small-model 1.65–1.73-BPW ternary image has already shown local fit can coexist with poor complete language loss. This 51-slot subset cannot rank all forty layers or estimate the model's NLL.

[CPU receipt](../../data/qwen-moe/original-expert-error/receipt.json) records source, pinned model, traffic header, original 16-expert payloads, first-sixteen GGUF tensor payloads, decoder-library and capture-array hashes, per-expert counts and weight/hidden/output metrics. Reproduce from a Bonsai checkout with the original expert slice and actual producer capture in place:

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python3 tools/qwen-moe/original_expert_error.py
```

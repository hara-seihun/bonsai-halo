# Real layer-0 routes and expert omission

The selected Qwen3.6-35B-A3B GGUF does not concentrate its layer-0 routed output on one or two experts. I captured the actual post-attention normalized input, selected IDs, normalized router weights, post-SwiGLU expert hidden vectors and unweighted down outputs while the installed runtime processed two distinct prose/code prompts. The first prompt has 113 tokens, the second 126. No weights or serving code changed.

| Retained experts, by descending router score | Train routed-output RMS | Held routed-output RMS | Held RMS after renormalizing retained scores | Conditional expert bytes/token | Conditional whole-model weight bytes/token |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | .702 | .679 | 1.769 | 76.44 MB | 2.091 GB |
| 2 | .534 | .473 | .985 | 152.88 MB | 2.168 GB |
| 4 | .312 | .264 | .429 | 305.76 MB | 2.320 GB |
| 6 | .156 | .138 | .185 | 458.64 MB | 2.473 GB |
| 7 | .104 | .084 | .100 | 535.08 MB | 2.550 GB |
| 8 | 0 | 0 | 0 | 611.52 MB | 2.626 GB |

For each token, the reference for this *local* metric is the FP64 sum of eight captured GGUF FP32 down outputs times their captured normalized FP32 scores. The ratio is the square root of summed squared omissions over summed squared reference output, across the named text. Each top-k response keeps the original scores. Renormalization divides by the retained score sum and is worse at every k on both texts. The held mean score mass of the highest-scored expert is .244; four experts hold .688. The highest-score expert contributes only .382 of the squared score mass on average. Even choosing the best one of the eight *with knowledge of the full routed output* still has .586 held RMS when it must return that expert's original weighted output. This is an oracle within that one-output family, not a lower bound on learned codes or on arbitrary replacement maps.

The byte columns charge one uncached read of the pinned image's selected expert weights and the unchanged 2.015 GB nonexpert stream. They assume proportional expert traffic, no launch or gather cost, and unchanged head/state work. Dropping four experts could save at most 306 MB of this modeled 2.626 GB weight stream, only 11.6% of its bytes, while losing .264 of routed-output relative RMS on this held text. It has no plausible exact path to the current output: the omitted captured vectors are generally nonzero. A learned route-conditioned replacement, or a different observation downstream of the routed sum, remains open. Do not ship top-k omission on this local metric. There is no complete-model NLL, logit quality or native speed measurement for omission.

The captured route distribution also limits transfer from the first sixteen official BF16 experts used in earlier CPU studies. Only 4.65% and 5.85% of the selected slots on these texts fall within IDs 0–15. There are 169 and 188 distinct experts and 103 and 114 distinct eight-expert sets. These counts are from just one layer and two short texts, not a global routing survey. The next useful construction should work with all 256 expert codes, using the captured producer vectors and scores, rather than fitting a catalogue from those first sixteen tensors alone. For a fitted representation, use separate train and held text and score the complete model before adoption; the dense sub-bit pilot's isolated-response gains failed at full-model quality.

## Capture and custody

The standalone observer uses llama.cpp's public `cb_eval` scheduler callback. It requests only five named layer-0 nodes from the installed selected runtime and copies the actual tensor values, including a strided top-k view. No native kernel or arithmetic is changed. `tools/qwen-moe/capture_routes.cpp` and `tools/qwen-moe/analyze_routes.py` own the capture and analysis. Build against the selected runtime libraries and the pinned native headers:

```sh
g++ -O2 -std=c++17 tools/qwen-moe/capture_routes.cpp \
  -I../bonsai-hip/include -I../bonsai-hip/ggml/include \
  -L../../data/qwen-moe/runtime/current/bin \
  -Wl,-rpath,../../data/qwen-moe/runtime/current/bin -lllama -lggml -lggml-base \
  -o ../../data/qwen-moe/qwen-capture-routes
for split in train held; do
  tools/run-batch-compare --runtime-max 43s --exec \
    ../../data/qwen-moe/qwen-capture-routes \
    ../../data/qwen-moe/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
    ../../data/qwen-moe/route-capture/$split \
    ../../data/qwen-moe/route-capture/$split.txt \
    > ../../data/qwen-moe/route-capture/$split.log 2>&1
done
python3 tools/qwen-moe/analyze_routes.py ../../data/qwen-moe/route-capture \
  --out ../../data/qwen-moe/route-capture/receipt.json
```

The [data owner](../../data/qwen-moe/route-capture/README.md) identifies model, selected native executable, source/binary hashes, text and token hashes, raw contiguous arrays, callback shapes, wrapper logs and the JSON calculation. Both commands exited successfully, the wrapper restored `bonsai-halo.service`, and the service was active after the panel. This is an observer's CPU reduction over a GPU-produced capture, not a whole-model speed comparison. FP64 recombination is not asserted bit-identical to llama.cpp's GGML reduction order.

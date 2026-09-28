# Qwen Q8 and expert calls without the trace recorder's slow episodes

**Question.** The counter-free profiler's wide Q8 and Q4 expert slow-layer masks track long periodic recorder gaps, yet the ordinary Q8 width-8,192 group still has a ~10-µs excess over a small-image affine fit. Are the >105-µs wide-Q8 and >80-µs Q4 episodes intrinsic to a stream without the trace recorder, including the matched occupied-depth-1024 shape? This experiment measures *kernel event durations*, not graph-enabled whole-model speed or physical DRAM transactions.

`tools/qwen-moe/q8_event_probe.cpp` interposes the selected binary's dynamically linked `hipLaunchKernel`. It places two HIP timing events in the **same stream**, one before and one after each Q8/Q4/Q5/Q6 `mul_mat_vec_q` call, and reads them only after the process's inference has finished. There is no per-call synchronization or engine/kernel source edit. It requires `GGML_CUDA_DISABLE_GRAPHS=1`: with graphs enabled, the hook sees graph construction and only eight graph replays, not each repeated matvec. The p32 no-graph panel reports **zero graph replays**, 18,495 ordinary launches, 2,621 selected quantized launches, eleven full-head markers (two setup, one prompt, eight decode). A second p1024 panel has 31,073 ordinary/2,623 selected launches, thirteen full-head markers (four setup, one prompt, eight decode) and zero graph replays. The parser verifies the seven quantized-shape counts inside each decode step: 40 wide Q8, 30 medium Q8, 80 medium Q8, 60 small Q8, 40 Q4 gate/up, 37 Q5 down and three Q6 down. It excludes the cold first generated step from aggregate statistics and retains every event.

On the **seven warm p32** no-graph steps:

| Call | Warm calls | Median of per-token call medians | Median summed device duration/token | Previous profiler slow cutoff exceeded |
| --- | ---: | ---: | ---: | ---: |
| Q8 grid 8,192 | 280 | **95.917 µs** | **3.754 ms** | **0/280 >105 µs** |
| Q8 grid 4,096 | 210 | 47.379 µs | 1.424 ms | — |
| Q8 grid 2,048 | 560 | 28.559 µs | 2.308 ms | — |
| Q8 grid 512 | 420 | 16.580 µs | .917 ms | — |
| Q4 gate/up | 280 | **53.238 µs** | **2.137 ms** | **0/280 >80 µs** |
| Q5 down | 259 | 40.759 µs | 1.510 ms | — |
| Q6 routed down | 21 | 43.399 µs | .130 ms | — |

At **occupied depth 1024**, the independent same-binary event panel has **1/280** wide-Q8 calls >105 µs (114.076 µs in its final token) and **0/280** Q4 calls >80 µs. Its warm per-token-median wide-Q8 duration is **96.776 µs**, summed wide-Q8 **3.793 ms**; Q4 is **53.698 µs**, summed **2.160 ms**. Thus the matched-context panel also lacks the *recurrent layer-episode population*, without claiming that every call is strictly below the historical cutoff.

The warm p32 Q8 event sum is about **8.40 ms/token**, so removing the profiler's episodes does *not* make ordinary Q8 work disappear; the wide group alone remains ~3.75 ms. This is a useful separation: a layer-ID-specific slow-kernel policy targets an episode absent in the no-graph unprofiled stream, whereas width-8,192 ordinary consumption remains a substantial engine question. The previous [recorder-gap joins](qwen-moe-q8-gap-separation.md) found all 30/30 held depth-1024 profiled slow wide calls near its periodic gaps (and 37/37 Q4 calls); their slow masks should not be converted into a native gain. The event durations here include event-record scheduling effects and can differ under graph replay; neither panel measures DRAM bytes.

**Numerical and perturbation controls.** On the same selected binary, graph-on, no-graph/no-event and no-graph/event executions each wrote four complete 248,320-FP32 heads and four greedy IDs on the same 22-token short prompt. All **993,280 logit words** and IDs have identical file hashes (`1eef184c29c07353ec1029e89e70e9854110ec710b581b49c0546b529afcb0fa` for the floats). The graph-on/no-graph agreement is for this panel, not a universal graph arithmetic theorem. At batch 512/ubatch 256, a separate same-work 32-prompt/8-generation benchmark gave **52.43 tokens/s uninstrumented no-graph** and **48.62 tokens/s with events**; event insertion costs ~7.27% wall rate on this panel, so do not use instrumented TPS as an improvement/regression of the selected runtime. Clocks/host load and service transitions are in the three wrapper logs. No model, selected binary, graph default or service was changed; the resident service is active after every wrapper run.

The [source/binary/model-acquisition, complete-head, raw event and wrapper-log hashed receipt](../../data/qwen-moe/q8-unprofiled/q8-event-receipt.json) retains both depth-32 and depth-1024 eight-token distributions. Reproduce with an ordinary Bonsai writer checkout and existing runtime:

```sh
g++ -D__HIP_PLATFORM_AMD__ -std=c++17 -O2 -fPIC -shared tools/qwen-moe/q8_event_probe.cpp \
  -o ../../data/qwen-moe/q8-unprofiled/q8-event-probe.so \
  -L/run/current-system/sw/lib -lamdhip64 -ldl -pthread
tools/run-batch-compare --runtime-max 41s --memory-gib 32 --host-reserve-gib 4 --exec \
  env GGML_CUDA_DISABLE_GRAPHS=1 \
  LD_PRELOAD=../../data/qwen-moe/q8-unprofiled/q8-event-probe.so \
  Q8_EVENT_OUTPUT=../../data/qwen-moe/q8-unprofiled/q8-event-panel.csv \
  ../../data/qwen-moe/runtime/current/bin/llama-bench \
  -m ../../data/qwen-moe/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  -ngl 99 -fa on -b 512 -ub 256 -t 8 -p 32 -n 8 -r 1
python3 tools/qwen-moe/q8_event_panel.py ../../data/qwen-moe/q8-unprofiled/q8-event-panel.csv \
  --data ../../data/qwen-moe/q8-unprofiled \
  --output ../../data/qwen-moe/q8-unprofiled/q8-event-receipt.json
```

**Next:** diagnose the *ordinary* wide Q8 calls with an unprofiled physical-transaction or load-stall measurement, comparing matched graph-on and no-graph complete heads and wall/clock panels. A width/grid or packed operand construction needs that bottleneck and a full-model win before selection. Do not port a layer slow-mask, count these event durations as default graph timing, or treat the recorder gaps as host idle budget.

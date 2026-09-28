# J=16 on short Qwen routed prompts

## Installed 24–32-row selection, September 23

Native revision `3f552c2` now selects the existing J=16 routed MMQ body for **24–32 rows**, retains J=32 for 9–23, J=64 for 65–256, and leaves single-token MMVQ unchanged. `GGML_CUDA_MMQ_EXPERT_J=0` still restores the original global-width control. This is the same packed model, Q8_1 preparation, expert grouping, J-body arithmetic, output-head work and forty-layer graph; only the selected tile width changes.

One GPU reservation ran four packaged-binary processes in control/candidate/candidate/control order. Each process loaded the complete model once and evaluated natural text prefixes of 16, 24 and 32 tokens five times with KV clearing and a requested full-vocabulary head row. Warm medians (repeats 2–4) on the same installed candidate binary are:

| Natural prompt | Control J32, ms | Selected J16, ms | Complete-model prompt-rate gain |
| ---: | ---: | ---: | ---: |
| 16 rows (unchanged dispatch) | 54.351 | 53.937 | +0.77% run-to-run noise |
| 24 rows | 70.547 | 67.504 | **+4.51%** |
| 32 rows | 77.890 | 74.585 | **+4.43%** |

Every one of the sixty requested 248,320-float logit rows matches its same-width control **bit for bit**, including in-process repeats and separate processes. One 24-row control sample reached 97.506 ms but the median is insensitive. The panel's wrapper reported median shader clock 2584 MHz and 1.9 busy host cores. These are warm complete-model prompt requests, not single expert kernel ratios or cold first-use claims. The first-ever requests still pay the previously diagnosed submit-side startup penalty; this change does not cure it.

For an initial generation-side regression check, a separate wrapper ran eight 32-row steps with **32 independent sequence IDs**, distinct natural tokens per stream and one requested full-vocabulary output row per step. The two same-binary policy arms reproduce all 1,986,560 FP32 output values bit for bit. Warm step medians are 138.763 ms at J32 and 135.953 ms at J16 (+2.07% aggregate token-rate direction), but the wrapper recorded another 1.2 busy host cores, so this is a no-observed-regression check on teacher-forced independent streams, not a clean autoregressive throughput gain. One-row plain decode cannot enter this changed dispatch.

The packaged candidate and subsequently selected `runtime/current` both reproduce the original accepted runtime's **7,946,240 FP32 logit values** and 32 greedy IDs on its *matched* 465-token prompt (16 repeated passages). An initial 15-passage run had 437 tokens and differed from the 465-token reference; retaining that failed input match prevents it being misclassified as a numerical defect. Installed acceptance's wrapper restored `bonsai-halo.service`. Its cold prompt time ran under competing host work and package-power limitation, so it is not used as a speed ratio.

[Source, model, packaged-binary and probe hashes, all raw full-vocabulary rows, per-process timings, independent-stream rows, the failed-input control, selection and installed acceptance](../../data/qwen-moe/short-j16/README.md) are in data custody. `tools/qwen-moe/short_j16_panel.py` and `short_j16_accept.py` check the policy arms and receipt. The native source bundle is retained by the on-demand runtime package. [Actual generated 32-stream continuations with all vocabulary rows](qwen-moe-generated-stream-j16.md) now reproduce 1,271,398,400 logit bits in each of four same-binary J32/J16 processes and show a favorable +0.62–7.69% whole-model generation-rate direction across four paired process orders, with competing host work recorded. **Next:** measure actual repeated served requests including first-use separately, and unprofiled generated multistream expert/Q8 phase costs before widening J=16 beyond 24–32; the heterogeneous J=16/64 dispatcher at longer prompt widths remains independent.

The installed Qwen engine selects J=32 for routed MMQ at 9–32 prompt rows. Its existing `GGML_CUDA_MMQ_EXPERT_J=16` diagnostic instead launches the compiled J=16 body, with the same packed image and output fold. The selected J=64 policy at 65–256 rows and single-row MMVQ are untouched. This panel asks whether J=16 is worth selecting for short complete-model prompts, rather than extrapolating from the earlier 126-row route count.

The installed binary `f60e4fb` ran `llama-bench` with the pinned 22.13 GB GGUF, batch 512, ubatch 256, eight CPU threads, flash attention, full GPU offload and three prompt repetitions per process. Each arm has its own `tools/run-batch-compare` reservation. The first repetition of each process costs 12–19 ms more than the next two; the table gives the median of the two warm samples and keeps all three in the receipt. Throughput includes the complete forty-layer model and the requested output head.

| Prompt rows | Selected warm time, ms | Forced J=16 warm time, ms | Whole-model warm rate change |
| ---: | ---: | ---: | ---: |
| 16 | 53.935 | 53.782 | +0.28% |
| 24 | 69.026 | 66.192 | +4.28% |
| 32, pair 1 | 74.135 | 70.987 | +4.43% |
| 32, pair 2 | 74.317 | 70.995 | +4.68% |

At 32 rows the second forced panel ran at a lower median shader clock, 2615 MHz against 2772 MHz for its control, and still finished earlier. The +4.4–4.7% result is a complete synthetic prompt gain, not an isolated MMQ kernel ratio. At sixteen rows the shorter body is a null. The earlier one-pair 32-row timing had too little resolution to select it; these paired warm samples resolve that question on this workload.

I added an optional short text to the reusable `qwen-accept` driver. Its ordinary prompt remains unchanged. The short prompt has 22 real tokens, so this tests the J=16 body on actual selected routes rather than only synthetic llama-bench tokens. Two independent processes in each arm produced the same 496,640 full-vocabulary FP32 logit floats, all **15,892,480 bits**, and the same two greedy IDs. The short-prompt `prompt_seconds` were 167.878 and 197.677 ms at selected J=32, and 196.065 and 203.675 ms at forced J=16, with comparable reported clocks. That interval includes the real prompt path's first evaluation and varies enough that the synthetic 24-row advantage cannot be promoted to a served-short-prompt gain from it. It also warns against selecting J=16 solely by the synthetic rate.

[The receipt, every raw sample, clocks, hashes and complete acceptance outputs](../../data/qwen-moe/mmq-width16/receipt.json) are in data custody. Regenerate the CPU summary with `python3 tools/qwen-moe/mmq_width16.py`; it checks binary, model, source and the two full-logit digests. The short acceptance executable was compiled from `tools/qwen-moe/accept.cpp` against the installed runtime libraries and is retained beside the receipt. All GPU calls used the wrapper; the resident Bonsai user service is active. No native source, image, installed binary or serving policy changed.

### Repeated natural-token prompt result

The first real-prompt comparison mixed graph setup with prompt work. `short_width_probe.cpp` now loads the unchanged installed model once per process, takes natural text prefixes of exactly 16, 24 and 32 tokens, clears KV state, then evaluates each prefix five or six times. It synchronizes after every complete forty-layer forward pass and writes the full 248,320-float requested logit row, not just a top choice. The first two evaluations of each width are kept but omitted from the steady median because first-use setup appears in both. `short_width_panel.py` hashes every logit row and records every sample. Each arm's repeats and all six arms agree bit-for-bit at each width, including the 32-token head observation. These are natural text inputs, not synthetic benchmark token IDs.

| Paired arm | 16 tokens, J32/J16 ms | 24 tokens, J32/J16 ms | 32 tokens, J32/J16 ms |
| --- | ---: | ---: | ---: |
| Performance-level pinned, A | 53.514 / 53.083 | 69.416 / 66.196 | 76.014 / 72.866 |
| Performance-level pinned, B | 54.891 / 53.125 | 72.510 / 66.245 | 81.649 / 73.279 |
| Fixed 2400 MHz, C | 58.163 / 57.595 | 77.078 / 72.503 | 85.366 / 80.263 |

At 24 and 32 tokens, J16 wins every paired steady panel. The A pair gives +4.86% and +4.32% in complete prompt rate at nearly equal 2894/2898 MHz shader clocks. Fixed-clock C gives +6.31% and +6.36%. The B control had 6.8 busy host cores and only 2752 MHz versus J16's 3.2 cores and 2898 MHz, so its larger +9.46/+11.42% is not a clean estimate. At sixteen tokens A/C gain just .81/.99%, not a useful selection. The logs retain the first-use costs: they range from 134 to 304 ms at 16 tokens, much larger than the 3–5 ms steady saving at longer widths. This is why the original 22-token first-evaluation panel could reverse the warm result. Reusing a context with explicit KV clearing represents a warm serving/model path, not a fresh first request or an independently occupied batch. It does not isolate the expert kernel from the rest of the model.

[All six wrapper logs, clock/host reports, 102 complete-vocabulary rows, probe binary and source-hashed receipt](../../data/qwen-moe/mmq-short-real/receipt.json) are in data custody; regenerate the latter with `python3 tools/qwen-moe/short_width_panel.py`. The pinned model, selected executable, packed image, output arithmetic and resident service did not change. `tools/run-batch-compare` restored the service after each arm.

This establishes a warm natural-prompt gain rather than an installed short-prompt policy. The compact J16/64 dispatcher remains a separate question at 65–256 rows.

### First-request and graph-disable control

A separate first-width probe loads the same installed model once per process and times the **first ever** natural 24- or 32-token complete prompt, followed by four KV-cleared repeats at that width. Unlike the previous probe, width 16 does not run first. Each row includes the requested full 248,320-logit vocabulary head. For each width all five rows within each arm and both J arms have identical finite logit bits, including the graph-disable diagnostic; the 24- and 32-token output digests differ as expected.

| First width and graph | J32 first / warm median, ms | J16 first / warm median, ms | Warm rate gain | First-use penalty range (first minus warm), ms |
| --- | ---: | ---: | ---: | ---: |
| 24, selected graphs | 207.891 / 70.126 | 162.704 / 67.177 | +4.39% | 95.5–137.8 |
| 32, selected graphs | 170.490 / 77.474 | 260.163 / 74.498 | +3.99% | 93.0–185.7 |
| 24, graphs disabled | 160.107 / 69.907 | 195.032 / 67.361 | +3.78% | 90.2–127.7 |

All six separate-process arms used `tools/run-batch-compare` with a 42-second payload bound, matched context/batch/flash/offload and exact installed model/binary. Wrapper shader-clock medians span 2696–2787 MHz; one 32-token J16 panel reports competing host work. The J16 warm advantage remains 3.8–4.4% across these pairs, consistent with the earlier clean panels. **Cold time reverses arm order**: J16 wins by 45 ms at 24 with graphs, loses by 90 ms at 32 with graphs and loses by 35 ms at 24 without graphs. These are independent processes, so differences do not estimate a causal cold-width effect. Disabling graph capture does not eliminate the large first-evaluation cost: it remains at least 90 ms above warm in both diagnostic arms. It does remove the second-evaluation 13–19 ms excess visible in the graph-enabled 24/32 arms. Thus graph capture alone cannot explain the first-use penalty or justify treating a warm kernel width gain as a cold-request gain. The first-use cause is still unresolved; probe HIP lazy setup, graph construction and host dispatch with separately timed regions in one real serving request before selecting a cold dispatch policy.

[Source-, probe-binary-, installed-library-, model- and output-hashed receipt](../../data/qwen-moe/mmq-short-real/first-width-receipt.json) retains all thirty full-vocabulary rows, every timing, clock/host warning and six wrapper logs. Reproduce the CPU summary with `python3 tools/qwen-moe/first_width_panel.py`; the executable source is `tools/qwen-moe/first_width_probe.cpp`. The GPU lock is free and `bonsai-halo.service` was restored after every arm. No image, installed executable, serving default or arithmetic map changed. The first-use region result below settles the API-boundary question; separately measure an actual repeated served request before packaging a 24–32-row J16 selection. Installed acceptance remains required for any runtime change.

### First use is paid inside `llama_decode`, not in the subsequent wait

A new probe partitions the same first-ever natural 24-token request and four KV-cleared repeats into monotonic host wall time spent clearing memory, inside `llama_decode`, inside `llama_synchronize`, and retrieving/scanning the full output logits. It leaves the selected 22.13 GB GGUF, forty-layer target, output head, context 8192, batch 512 / ubatch 256, flash attention and eight CPU threads unchanged. Six **separate** processes/reservations include graph-enabled and graph-disabled J32 replicates and one each at J16. The complete 248,320-float row is bit-identical across all thirty evaluations, including graph-disable and width arms.

| Arm (first minus own warm median, ms) | `llama_decode` | subsequent `llama_synchronize` | complete requested prompt |
| --- | ---: | ---: | ---: |
| Graphs, J32 A / B | +148.5 / +144.9 | −30.5 / −34.2 | +118.1 / +110.8 |
| No graphs, J32 A / B | +263.0 / +167.8 | −27.2 / −29.2 | +235.9 / +138.7 |
| Graphs, J16 | +147.6 | −32.5 | +115.0 |
| No graphs, J16 | +180.7 | −31.9 | +148.8 |

Warm J32 `decode` is 0.24 ms with graphs and 3.66 ms without; its warm synchronization is 69.8 and 66.2 ms respectively. The first J32 graph-on `decode` is 145–149 ms; first graph-off `decode` varies from 171 to 267 ms. In **all six** panels, first-request excess sits before `llama_decode` returns; the later wait is actually shorter. Logit retrieval is about 0.1 ms. Graph disable eliminates the graph-on second-evaluation excess, but does not remove the much larger first-call submit-side work. A smaller J expert body saves warm prompt time without addressing this cold cost. The first-call `decode` interval can contain host graph construction, HIP lazy setup and blocking device work within that API; this partition **does not prove** that the GPU is idle or attribute the time to compilation. It rejects the simpler explanation that the cold penalty resides in a late device wait after an otherwise warm-rate submission. The next diagnostic belongs **inside** `llama_decode`: time GGML graph construction, HIP allocation/initialization and kernel compilation/launch, ideally with device events, then test actual repeated served requests before choosing a cold J policy.

[The source, installed-library, binary, model and thirty full-logit hashes, every region sample, clocks and wrapper restoration logs](../../data/qwen-moe/mmq-cold-phase/receipt.json) are in custody (SHA-256 `fc4a0b8c353e45d896d05b43ae33d27bcc0b35571536849353835c6d89bcc0b6`). Compile `tools/qwen-moe/cold_phase_probe.cpp` against the installed `libllama`, then regenerate the summary with `python3 tools/qwen-moe/cold_phase_panel.py`. All six GPU wrappers restored `bonsai-halo.service`; selected runtime, model, J policy and numerical map remain unchanged.

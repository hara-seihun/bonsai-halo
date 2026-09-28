# Qwen3.6 35B-A3B on the Radeon 8060S

The reference runtime is the HIP llama.cpp build at `../bonsai-hip/build-hip/bin`. Its `qwen35moe` graph handles the Qwen3.6 model and its RDNA4 backend selects quantized MMQ for the routed experts. This is a baseline for new MoE work, not a replacement for the resident Bonsai Halo service.

## Run a bounded panel

`tools/qwen-moe/benchmark.py` records the model's full SHA-256 once, then checks its file identity before each panel. Do not start `prepare` until the GGUF download has finished. The expected model is the 22,134,528,992-byte `unsloth/Qwen3.6-35B-A3B-GGUF` Q4_K_M file at revision `a483e9e6cbd595906af30beda3187c2663a1118c`, SHA-256 `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`.

```sh
cd .
python3 tools/qwen-moe/benchmark.py prepare
python3 tools/qwen-moe/benchmark.py run --phase prompt --tokens 128 --depth 0 --repetitions 1
python3 tools/qwen-moe/benchmark.py run --phase decode --tokens 16 --depth 0 --repetitions 2
python3 tools/qwen-moe/benchmark.py run --phase decode --tokens 16 --depth 1024 --repetitions 2
python3 tools/qwen-moe/benchmark.py sample --tokens 24
```

Each `run` starts llama-bench through `tools/run-batch-compare --exec`. That wrapper takes the exclusive measurement lease, pauses and runtime-masks Bonsai Halo, admits only after checking free memory, starts a 27 GiB / no-swap systemd scope with a 25 GiB tracked device-allocation ceiling, and restores the service on exit. Four GiB remain outside the scope as the minimum host reserve. The job deadline defaults to 42 seconds. A deadline failure remains a failed panel, not a rate. Reduce token count or split panels rather than removing the guard. The wrapper's 130-second lock acquisition can outlive an attended 55-second call if the shared board is busy; check lane state first and do not force the lock.

Prompt ingestion and decode run separately. `--depth` preloads synthetic tokens into KV before the timed region; it is the occupied context, not the measured prompt length. Both phases use synthetic llama-bench tokens, so their `avg_ts` describes execution throughput, not task quality. The `sample` action runs a deterministic greedy real prompt with llama-cli and retains its output; it is not used to infer throughput. Use the same quantization, batch, ubatch, context, threads, flash-attention mode, source and runtime libraries when comparing rates. An experiment that changes any of these is a new coordinate, not an A/B result.

Receipts live in `../../data/qwen-moe/benchmarks/`. `model.json` holds the verified model identity. Each panel writes `receipt.json`, the raw `stdout.jsonl` or `output.txt`, and `stderr.log`. The receipt includes the exact argv, exit code, individual llama-bench averages and spreads, source revision and dirty paths, a tracked-diff digest, executable and locally built shared-library SHA-256 hashes, wrapper revision/hash and memory limits. Failed and timed-out panels are retained. An executable can be a 15 KiB linker stub, so the shared-library hashes matter more than that stub's hash.

The 60 GiB host cannot hold another unbounded model job beside Bonsai Halo. Never invoke the HIP binary directly for a full-model run, or assume the cgroup alone bounds AMD driver GTT allocation. `tools/run-batch-compare` owns both the GPU lease and the service restoration. It does not preempt the separate Fish resident admitted by `gpu-run`; inspect the common GPU board before a panel. This Qwen path does not install a resident server.

## Interpreting the MoE route

The upstream `ggml_cuda_op_mul_mat_id` path already groups tokens by selected expert, and shares an activation quantization between the routed gate and up products. A new expert-major schedule alone is not an invention. The useful questions are whether the selected eight experts' weight streams can share a packed representation without changing the numerical map, and whether a grouped path beats the existing kernel at the actual routing distribution. Measure selection counts per expert and per layer rather than using a uniform-router assumption. Report prompt and decode rates at declared occupied contexts, memory footprint, clock and host contention, and the output comparison beside each optimization.

## First complete-model result, September 23

The pinned GGUF passed the full SHA-256 check before these panels. With HIP llama.cpp `1a07bfa5f`, all 40 layers GPU-offloaded, Q4_K_M mixed image, `--flash-attn on`, eight CPU threads, batch 512, ubatch 256, one stream:

| Synthetic workload | Raw tokens/s | Occupied context |
| --- | --- | ---: |
| prompt, 128 tokens | 526.38, 610.19 | 0 |
| decode, 24 tokens | 49.56, 51.15 | 0 |
| decode, 16 tokens | 49.55, 52.17 | 1024 |
| decode, 24 tokens | 49.43, 51.24 | 4096 |

The measurements are in `benchmarks/20260923T183933Z-prompt-d0-c0fd3812`, `20260923T183941Z-decode-d0-c4b3b6fa`, `20260923T183956Z-decode-d1024-6ba92d0c` and `20260923T184226Z-decode-d4096-f0ed85ad`. Each path is beneath `../../data/qwen-moe/`. The sample in `20260923T184207Z-sample-d0-072d0e2a/output.txt` answers the compass question in one sentence with reasoning disabled. That sample reports 144.9 prompt and 33.4 generation tokens/s in llama-cli; its real prompt, chat template, sampling and token counts differ from the synthetic benchmark, so it is not the headline rate.

This board was not quiet. The wrapper recorded about 4.9 additional host cores busy and a median GPU clock of 2680 MHz in one decode panel. A later depth-0 24-token panel returned 40.01 and 47.93 tokens/s, while a nearby 32-token panel gave 50.32, 51.19 and 52.17. Treat the approximately 50 tokens/s decode result as a working baseline, not a clock-normalized speed record. The conditional 2.626 GB weight stream calculated in `docs/qwen-moe.md` would take 10.85 ms at 242 GB/s; this baseline takes roughly 20 ms/token. That gap includes state/KV, issue, routing, launches and machine contention. The routed expert weights make up only 23.3% of the modeled stream, so an expert-only traffic change cannot be credited with closing it.

The first scheduler comparison held model and prompt size at 256 tokens. At ubatch 256, two panels gave 847.8, 887.2, 504.0 and 720.4, 873.9, 902.8 tokens/s. At ubatch 128, two gave 437.7, 601.0, 625.5 and 614.3, 605.0, 624.4. Ubatch 256 is the better observed route at this shape, but the scatter from concurrent host load is larger than a small kernel improvement. Ubatch 512 on a 256-token prompt gave 589.2, 823.5, 847.9, no convincing gain over 256. These are scheduling controls, not a new MoE representation, and no server default changed.

## What the decode step does

A six-token decode under `rocprofv3 --kernel-trace` wrote `benchmarks/profile-20260923/receipt.json`, `run.jsonl`, the raw CSV and `summary.json`. Its `rocprofv3` timing is not a throughput measurement: it reports 10.97 tokens/s because tracing almost ten thousand dispatches adds host overhead. The CSV's summed device execution is 118.24 ms over six tokens, or 19.71 ms/token, near the unprofiled 20 ms/token. To replay the phase breakdown, run the exact `argv` in its receipt, then:

```sh
python3 tools/qwen-moe/profile_trace.py ../../data/qwen-moe/benchmarks/profile-20260923/gpu-host/1267049_kernel_trace.csv --tokens 6
```

| Kernel family | Calls in six tokens | Summed device ms | Share |
| --- | ---: | ---: | ---: |
| Q8_0 matvec, all variants | 1260 | 49.12 | 41.5% |
| Q4_K matvec | 240 | 14.64 | 12.4% |
| Q6_K matvec | 24 | 11.46 | 9.7% |
| Q5_K matvec | 222 | 10.34 | 8.7% |
| float matvec | 840 | 6.85 | 5.8% |
| Q8 activation preparation | 1746 | 3.61 | 3.0% |
| fused top-8 router | 240 | 1.64 | 1.4% |

The other kernels total 17.37 ms, including norms and elementwise work. The 24 Q6_K matvec calls comprise six output-head calls at grid 7946240×1 and eighteen routed-down calls at grid 65536×8. The head alone takes 10.736 ms; the expert calls take 0.721 ms. They are not four output-head tiles per token. The top-8 router is cheap compared with the projection streams; it is not a plausible sole cause of the remaining decode gap. Q8_0 projections account for the largest device term, while Q4_K and Q5_K include expert work. Profile before changing an expert format or launch schedule. These sums are kernel active times, not a measure of effective DRAM bandwidth or of numerical equivalence after a candidate change.

## Why the active-parameter comparison is suspicious

the project owner challenged the similarity to the 27B dense engine on September 23.
The like-for-like plain rates are 52.29 tokens/s for the installed Qwen runtime
at depth 1024 versus Bonsai's published 33.70–34.5 plain tokens/s at shorter
contexts. Qwen is about 1.5 times faster, not nine times faster. These are not
matched-context panels, and speculative rates must not enter this comparison.

The weight-byte comparison is 2.626 GB/token for Qwen versus 5.647 GB/token
for Bonsai's densely packed ternary representation, a factor of 2.15. Qwen's
small active expert set does not make the entire active computation uniformly
four-bit. Its nonexpert stream is 2.015 GB, including 1.493 GB of Q8 weights,
417 MB for the Q6 vocabulary head and 105 MB of float weights. The selected
routed experts contribute 612 MB. Bonsai also reads/writes roughly 302 MB of
recurrent state per step in its documented budget.

There is still an efficiency deficit to explain. Dividing modeled image bytes
by full step time gives Qwen about 137 GB/s, versus roughly 199 GB/s for the
Bonsai weight-plus-state budget. At the latter rate Qwen's weights alone take
13.2 ms, before its own state/KV and other work, versus the measured 19.1 ms.
This is a diagnostic budget, not an achieved or universal throughput ceiling.

| Qwen family | Active encoded GB/token | Image bytes / traced kernel time |
| --- | ---: | ---: |
| Q8 nonexpert matvecs | 1.493 | 182 GB/s |
| Q6 vocabulary head | 0.417 | 233 GB/s |
| Q4 routed gate/up | 0.377 | 155 GB/s |
| Q5 routed down | 0.213 | 124 GB/s |

These ratios are not DRAM counter measurements. Fused arithmetic, cache hits
and repeated reads can change their interpretation. The head is close to the
242 GB/s streaming reference and is not the first bandwidth target. Q8 has the
largest absolute cost; small expert shapes have worse image-byte rates.

The trace has no prompt, occupied depth or warmup. Its 240 router, 180 GDN,
60 attention and six vocabulary-head calls confirm six target steps. Its
roughly 1,600 dispatches per token deserve scrutiny, but the trace's 10.97
tokens/s wall rate is profiler-distorted. It does not establish large gaps
between launches during normal graph replay. Nor does a top-k boundary delimit
a whole token, so subtracting pre/post-top-k intervals as startup is wrong.

The [Q8 width/call join](qwen-moe-q8-phase.md) accounts for all 250 Q8 nonembedding tensors with 210 matvec calls/token, including 40 fused shared gate/up calls. On the retained trace, 8192-output projections contribute 0.858 ms/token of the 2.018 ms/token Q8 difference against an ideal one-read 242 GB/s comparator. Even idealizing every Q8 call under that conditional model only changes summed device time 19.707 to 17.689 ms/token; the wide shape is the first counter target, not the already fused shared gate/up.

[The installed-runtime GL2C counter audit](qwen-moe-q8-native.md) overturns
that panel's initial physical-byte estimate. The raw CSV has 395 consecutive
executed matvec dispatches reporting zero read requests; every quantized family
on the other dispatches reports about half its one-call packed image. Scaling
the aggregate by four concealed both effects and generated the apparent 8–10%
excess. The counter panel cannot bound Q8 or routed expert rereads. Inspect raw
per-instance counters on a short complete panel, then compare unprofiled phases
at the same depth before choosing an occupancy or scheduling arm. `profile_trace.py`
reports launch geometry alongside quant families to keep the head/expert
attribution reproducible.

Bonsai Halo was active after each wrapper exit. Qwen has an accepted on-demand
runtime, but no Qwen resident service has been deployed.

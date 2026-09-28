# Selected J=16 on generated Qwen MoE streams

The installed Qwen `3f552c2` policy selects J=16 for 24–32 routed MMQ rows. Its earlier independent-stream acceptance fed teacher-forced tokens and requested only one vocabulary row per step. This panel tests the missing complete-map workload: **32 independent autoregressive streams**, every full-vocabulary head, each stream feeding back its own greedy decision. It uses the unchanged pinned 22.13 GB image and same installed library for both arms; `GGML_CUDA_MMQ_EXPERT_J=0` restores the original global J=32 at generated width 32. No image, runtime or serving default changes.

The probe seeds each sequence with eight natural-text tokens, shifted along the same 40+-token text, requests one complete head per sequence, then generates four tokens per sequence. `llama_decode` plus `llama_synchronize` is timed per whole 32-row step; step zero's 256-token prompt is excluded from generation rate. Context 8192, 32 sequence slots, batch 512 / ubatch 256, eight CPU threads, flash attention, full GPU offload. Unlike the prior control, **all 32 vocabulary rows** are read and written on every step. The seed's 32 first greedy decisions contain 18 distinct token IDs, and the fourth generated decisions contain 21. The output work is equal across arms.

| Paired process order | Control J32, four steps ms / aggregate tok/s | Selected J16, four steps ms / aggregate tok/s | J16 rate change |
| --- | ---: | ---: | ---: |
| Default clock, control → J16 | 604.697 / 211.68 | 600.992 / 212.98 | +0.62% |
| Default clock, J16 → control | 614.317 / 208.36 | 590.387 / 216.81 | +4.05% |
| Fixed 2400 MHz, J16 → control | 657.863 / 194.57 | 629.055 / 203.48 | +4.58% |
| Fixed 2400 MHz, control → J16 | 651.296 / 196.53 | 604.773 / 211.65 | +7.69% |

Each entry includes 128 generated decisions and the whole forty-layer model. The last fixed-clock wrapper reports 2399 MHz median, but another 2.6 host cores busy; the preceding fixed-clock panel also reports another 2.6 cores. The two default-clock panels report 2290 and 2382 MHz medians and 1.1/1.9 other busy host cores. Thus four paired directions are favorable, but the 0.62–7.69% spread is a workload/host-sensitive observation, not a clean estimate of the isolated expert kernel's contribution. In particular the selected policy's earlier 4.43% warm **prompt** gain cannot simply be multiplied into generation. First-use 256-row prompt setup is deliberately separated; the panel does not measure HTTP serving, 8/64 streams, long context, or physical weight bytes.

All four runs emit the same SHA-256 `9b3755fdcb597754048c792960478e6659eb95f6317e9fc7431c043a173ea886` over **160 complete 248,320-float head rows**, 1,271,398,400 identical logit bits in each run, and identical 32-stream greedy continuations. This is an exact selected-map acceptance on generated feedback, not merely a matching token sample or an isolated timing ratio. The first and reversed panels' receipts have SHA-256 `70a47c25ad2382948ad257b854942fd3d279bd727ec158b9cb936ce11029b009` and `63e348136a05ceefeaaa65215cc275a4ce849b5ca306b2aace5ca0802ca6408e`; the two fixed-clock receipts have `276ac441bfa640a0ef5c881046d73483149d3d601a110ca3271c3b5d2e71e255` and `c528e95bffae7b75ce9c618823a16651d379cb34367f77008bca70f9c746358f`.

[Raw logits, per-step timings, token IDs, per-arm standard output/error, source/probe/installed-library hashes and receipts](../../data/qwen-moe/generated-stream-j16/) are retained. The final wrapper log is `fixed-reverse-wrapper.log` (SHA-256 `b150f492d495579d5285abe077bc19be4c30b74d76aa83a0595db768b164f97f`). The model SHA-256 is `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`, checked against the pinned acquisition. The installed HIP library's receipt SHA-256 is `85eeaf44e57d03cffd2a82d68eca46c5b791e0b96210533e17e72a2b9aea749d`. All wrappers restored `bonsai-halo.service`; the GPU lock is free.

Reproduce one paired panel from the Bonsai root after compiling `tools/qwen-moe/generated_stream_j16.cpp` against the installed `libllama` and the matching native `llama.h` / `ggml.h` headers; the probe binary and its hash are retained in the data directory:

```sh
tools/run-batch-compare --runtime-max 45s --memory-gib 36 --host-reserve-gib 6 --exec \
  python3 tools/qwen-moe/generated_stream_j16.py \
  ../../data/qwen-moe/generated-stream-j16/probe \
  ../../data/qwen-moe/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  ../../data/qwen-moe/generated-stream-j16/repeat --steps 5 --arms control,candidate
```

**Next:** measure unprofiled expert versus Q8 phase cost for these actually generated 32-stream rows and their route groups, with every head row retained. If expert MMQ is a small share of this step, width tuning has reached its whole-model margin; focus on Q8 and the vocabulary head rather than multiplying static padded-slot savings. For serving, separately measure repeated actual multi-client requests with first-use reported rather than hidden.

# Bonsai Halo: a from-scratch inference engine for Ternary Bonsai 27B on Strix Halo

One model, one machine. Bonsai 2 27B (ternary, Qwen3.8 hybrid GDN/attention, 64 layers) on the Ryzen AI MAX+ 395 with the Radeon 8060S (gfx1151, 40 CUs, 256-bit LPDDR5X-8000). Nothing here is general. Every shape is a compile-time constant and every byte of weight traffic is accounted for.

Cross-project research decisions and comparator boundaries live in
[Kelana's catalogue](https://github.com/hara-seihun/kelana/blob/main/research/catalogue/README.md). This document owns
the engine design and its measured implementation history.

## The physics, measured on this box

| Quantity | Value | Source |
|---|---:|---|
| DRAM peak | 256 GB/s | 8 x 32-bit LPDDR5X at 8000 MT/s (dmidecode) |
| GPU sustained read, any allocation type | **242 GB/s** (94.5%) | `bench/bw` on 2 GiB buffers; hipMalloc, managed, host-pinned and fine-grained all within 3% |
| Trit entropy of the model | 1.5849 bits = log2 3 | histogram of all 25.6 B trits; the Hadamard rotation whitens them, so entropy coding gains nothing |
| Weight bytes per token, floor | 5.47 GB (1.71 bpw incl. fp16 scale per 128) | 25.6 B params x log2(3) + scales |
| Weight bytes per token, PTQ1_0 packing (5 trits/byte) | 5.60 GB (1.75 bpw) | 24 B qs + 2 B qh + 2 B scale per 128 |
| GDN recurrent state | 48 layers x 128 x 6144 = 151 MB in f32, 75 MB in bf16 | read + write every token |
| KV cache | 64 KB per context token in f16 | 16 layers x 4 heads x 256 x 2 |
| llama.cpp HIP baseline, this machine | PTQ1_0 **21.0 tok/s**, PQ2_0 **23.2 tok/s** | `llama-bench -ngl 99 -fa 1 -n 64` |

Single-stream ceilings at 242 GB/s, short context, no MTP:

| Packing | GB/token | tok/s |
|---|---:|---:|
| PTQ1_0 (what we ship first) + f32 state | 5.90 | 41.0 |
| PTQ1_0 + bf16 state | 5.75 | 42.1 |
| entropy floor + bf16 state | 5.62 | 43.1 |

llama.cpp is at 51% of the PTQ1_0 roof. Its PTQ1_0 path is slower than PQ2_0 even though it reads 18% fewer bytes: the HIP kernel is ALU-bound on trit decode. That is the gap this engine closes. The target for v1 is **>= 38 tok/s** single stream (93% of roof); the MTP factor of 2 comes later on top of it.

The initial 2,500-token/s batch estimate added GPU, NPU and CPU throughput without an executable joint schedule and used an incorrect GPU precision/rate attribution. It is not a ceiling. The current [throughput budgets](docs/throughput-budgets.md) count the model's 25.60 billion dense MACs per token, use the corrected physical SIMD geometry, and price memory and state separately. With declared ideal GPU issue periods, the matrix-only budget is about 1,160 tokens/s for IU8 or 2,320 for IU4 at 2.9 GHz. Neither is a theorem about alternative whole-map computations. The single-stream roofs above likewise assume this streamed weight representation and one target token per pass.

## Where the bytes go per token (PTQ1_0, per layer)

| Tensor | Rows x K | Bytes |
|---|---|---:|
| GDN layer (x48) | | |
| attn_qkv | 10240 x 5120 | 11.47 MB |
| attn_gate (z) | 6144 x 5120 | 6.88 MB |
| ssm_out | 5120 x 6144 | 6.88 MB |
| ssm_alpha, ssm_beta (bf16) | 2 x 48 x 5120 | 0.98 MB |
| ffn_gate, ffn_up | 2 x 17408 x 5120 | 39.0 MB |
| ffn_down | 5120 x 17408 | 19.5 MB |
| state r+w (f32) | 48 x 128 x 128 x 4 x 2 | 6.29 MB |
| Attention layer (x16) | | |
| attn_q (q + gate) | 12288 x 5120 | 13.76 MB |
| attn_k, attn_v | 2 x 1024 x 5120 | 2.29 MB |
| attn_output | 5120 x 6144 | 6.88 MB |
| ffn (same as above) | | 58.5 MB |
| KV read | 64 KB x ctx | |
| lm_head | 248320 x 5120 | 278 MB |

Total 5.90 GB. Everything that is not a weight stream is under 3% and is scheduled so it overlaps or is negligible.

## Design

### Weight format: HALO tiles

The model is repacked once at load (multi-threaded, cached to `~/data/bonsai2/halo-cache/`) from the GGUF PTQ1_0 (or PQ2_0, which decodes to the same trits bit-for-bit) into a layout built for one access pattern: **one lane owns one output row, the wave streams 32 rows together, the input block is broadcast from LDS**.

For a tensor of N rows and K columns (nb = K/128 blocks per row), tile T holds rows 32T..32T+31. For each (T, b):

```
896 bytes = [ 32 rows x 24 B qs ][ 32 rows x 2 B qh ][ 32 rows x fp16 scale ]
```

A wave reading block b of its 32 rows touches one contiguous 896 B run. A tile is nb x 896 B contiguous, so the whole tensor is a sequential stream. 1.75 bpw exactly, no padding.

Trit order inside a block is ours, chosen so decode needs no byte shuffles. A byte packs 5 trits base-3 (TQ1_0 trick: `q*3 >> 8` peels the leading trit). Bytes are processed as 16-bit pairs with packed u16 math (`v_pk_mul_lo_u16`, `v_pk_lshrrev_b16`, `v_and`), which peels one trit from two bytes per 3 instructions. Peeling step n of byte pair (m, m+1) yields a dword `[t_n(m), 0, t_n(m+1), 0]`; OR-ing step n with step n' shifted by 8 yields `[t_n(m), t_n'(m), t_n(m+1), t_n'(m+1)]`, a full 4-trit dword for `v_dot4_i32_iu8`. The element order of the input vector is defined to match, so no permutation instruction ever runs. Per 128-trit block per lane: ~170 packed-peel instructions, 30 ORs, 32 dot4, 3 fma. At 242 GB/s a CU receives one 896 B wave-block every ~430 cycles; the decode costs ~240 issue cycles on one of the CU's two SIMD32s. ALU utilization ~28%, so the kernel is memory-bound with margin.

Element map for block-local index e in [0,128): for qs byte pair p = 0..11 (bytes 2p, 2p+1), peel step n = 0..4:
- n in {0,1}: dword A_p = [t_0(2p), t_1(2p), t_0(2p+1), t_1(2p+1)] covers e = 8p + {0,1,2,3}
- n in {2,3}: dword B_p covers e = 8p + {4,5,6,7}
- n = 4: dword C_r for byte quad r = 0..5 covers e = 96 + 4r + {0..3} as [t_4(4r), t_4(4r+2), t_4(4r+1), t_4(4r+3)]
- qh bytes h = 0,1 (4 trits each, same peel): dword D_h covers e = 120 + 4h + {0..3}

### Activations

Integer dot products with per-128-block int8 activations and fp32 scales. For each block: `y += (dot_u8s8(t, xq) - sum(xq)) * w_scale * x_scale` where the subtraction converts the stored unsigned trit {0,1,2} to {-1,0,+1}. `sum(xq)` per block is produced by the prep kernel. Everything outside matvecs is fp32.

### Kernel inventory (batch 1)

| Kernel | Work | Notes |
|---|---|---|
| `prep` | RMSnorm (optional) -> optional elementwise (silu(g)*u, residual) -> sign flip -> block-1024 fast Walsh-Hadamard -> int8 quant + block sums | One workgroup per 1024-block; each recomputes the norm's sum of squares redundantly (5-17K floats from L2, trivial). Replaces llama.cpp's norm + mul + 1024x1024 matmul + quantize chain. |
| `matvec` | Segment table of (weights, N, out, epilogue). WG = 4 or 8 waves, split-K across waves, LDS reduce, lane 0..31 write. Epilogues: store, add-to-residual. | Template on K. One launch per fused group: `qkv+z`, `alpha+beta` (bf16 path), `gate+up`, `down`, `q+k+v`, `o`, `lm_head`. |
| `gdn` | conv1d (state ring of 3) + silu, L2-norm q/k per head, gates, the 128x128 recurrence per head, gated RMSnorm x silu(z), grouped permutation for ssm_out's Hadamard | One WG per head (48). State f32 now, bf16 switchable. |
| `attn_pre` | q/k RMSnorm per head, RoPE on dims 0..63 (neox pairs, base 1e7), append K/V to cache | |
| `attn` | Flash-decoding: WG per (q head, 256-position chunk) then combine; x sigmoid(gate) | GQA 6:1, head dim 256 |
| `embed` | Decode one PTQ1_0 row of `token_embd`, Hadamard, sign flip | |
| `sample` | argmax / top-k top-p temperature over 248320 logits | GPU argmax first; sampling on host from a 1 MB copy |

About 9 launches per GDN layer, 8 per attention layer, ~600 per token. Captured once into a HIP graph and replayed; token, position and context length live in device buffers the host updates before each replay. If graph node overhead still shows in the profile, phase 2 is a persistent megakernel with a device-side phase barrier.

### Semantics pinned from the reference (bonsai-hip `src/models/qwen35.cpp`, `delta-net-base.cpp`, ggml CPU ops)

- Hadamard: for weight W' with input dim n, y = W' H (s * x) blockwise over 1024, H = Sylvester Walsh-Hadamard / sqrt(1024). Sign vectors per width 5120, 6144, 17408. token_embd stores rotated rows: h = s * (H z). ssm_out's input is permuted first: tiled [128, 16, 3] -> grouped [128, 3, 16] (feature (hd, k, r) -> position hd + 128 (r + 3k)).
- GDN per token, head h in 0..47, q/k head h mod 16, S_v = 128, M[j][i] (j value, i key): g = exp(a[h] * softplus(alpha[h] + dt[h])), beta = sigmoid(beta_raw); M *= g; delta[j] = (v[j] - sum_i M[j][i] k[i]) * beta; M[j][i] += delta[j] k[i]; o[j] = sum_i M[j][i] q[i] / sqrt(128). softplus threshold 20.
- Before that: conv over the last 4 pre-activation qkv vectors (channels 10240: q 2048 | k 2048 | v 6144), silu, then q and k L2-normalised per 128-head with scale 1/max(||x||, 1e-6).
- After: per-head RMSnorm(o) * ssm_norm[128] * silu(z), eps 1e-6.
- Attention layer: attn_q output is per head [256 q | 256 gate] interleaved; RMSnorm q,k per head; RoPE neox pairs (i, i+32) over dims 0..63 with theta_i = pos * 1e7^(-i/32); softmax scale 1/16; output * sigmoid(gate); attn_output.
- FFN: down(silu(gate(x)) * up(x)).
- Norm eps 1e-6 everywhere; norm weights f32.

### Correctness plan

`tools/reference.py` dumps per-layer activations from llama.cpp (via `llama-eval-callback` on CPU) for a fixed prompt. The engine has a `--dump` mode writing the same tensors. `tools/compare.py` reports max abs / rel error per tensor. Acceptance: logits within int8-activation noise (cosine > 0.999, argmax agreement over a 64-token greedy continuation).

### Roadmap

1. **v1, single stream** (this pass): repack + load, all kernels, greedy decode, HIP graph, `--bench`. Target >= 38 tok/s at short context.
2. bf16 GDN state and fp16/4-bit KV; measured perplexity delta before switching defaults.
3. MTP drafter (the `dspark`/DFlash head from Bonsai-demo) verified in one pass: the state is loaded once per pass, so accepted tokens cost only the extra compute.
4. Batched decode: WMMA int8 path for the same HALO tiles, chunked GDN, continuous batching. The tile layout already gives 32 rows per wave, which is the WMMA M dimension.
5. Prefill: chunked GDN (64-token chunks as in llama.cpp) and WMMA matmuls.

### What we deliberately reuse

The tokenizer. Qwen's byte-level BPE with its pre-tokenizer regex is a solved problem with no performance content. `libllama` from the bonsai-hip build is loaded vocab-only for `tokenize`/`detokenize`. Nothing else from llama.cpp runs at inference time.

## Results

Both roadmap items 1 and the persistent kernel of item 2 are built. Measured on this machine, single stream, greedy, 20-60 generated tokens after a 5-token raw prompt:

| Path | tok/s | ms/token | Notes |
|---|---:|---:|---|
| llama.cpp HIP PTQ1_0 | 21.0 | 47.6 | baseline |
| v1: per-op kernels, HIP graph per token | 31.5 | 31.7 | ~600 launches/token; matvec kernels alone hit 190-228 GB/s |
| v2: persistent cooperative kernel | **34.0** | **29.4** | one launch per token, 80 workgroups, 584 device barriers |
| roof at 242 GB/s | 41.0 | 24.4 | |

v2 phase breakdown per token (device timestamps at each barrier):

| Phase | us/token | share | GB/s achieved |
|---|---:|---:|---:|
| matvec gate+up (64x) | 11,100 | 37.8% | 225 |
| matvec down (64x) | 6,430 | 21.9% | 194 |
| matvec qkv+z (48x) | 3,860 | 13.1% | 228 |
| matvec ssm_out (48x) | 1,650 | 5.6% | 200 |
| GDN recurrence (48x) | 1,270 | 4.3% | 238 on the 6.3 MB state per layer |
| lm_head (1x) | 1,177 | 4.0% | 236 |
| matvec q,k,v (16x) | 1,138 | 3.9% | 230 |
| prep (257x) | 1,780 | 6.1% | latency-bound, 3-8 us each |
| matvec o (16x) | 578 | 2.0% | 190 |
| attention (16x) | 400 | 1.4% | |

Matvec phases average 227 GB/s over 5.90 GB. The remaining gap to the roof is 2 ms of small phases (prep, GDN, attention) that run while the weight stream is idle, and ~0.7 ms of per-phase ramp and tail inside the matvecs.

### What the persistent kernel taught

- **LDS, not VGPRs, set occupancy on gfx1151.** LDS is 64 KB per WGP (two CUs). A 19.5 KB workgroup gets 3 per WGP; 10 KB gets 6. Staging the 17 KB K=17408 input in LDS was the wrong trade; splitting K across two workgroups (atomicAdd into the residual) keeps the staged slice at 9 KB.
- **Register pressure in a megakernel comes from cross-phase hoisting.** Each phase alone needs 50-120 VGPRs; inlined together the compiler keeps 190+ live. `noinline` phases made it worse (callees defaulted to 230 VGPRs) and 30% slower. What worked: no dynamic indexing into private arrays (scratch), `#pragma unroll 1` on the attention loops, and a compact embed decode. `amdgpu_waves_per_eu` caps the count but spills into the hot loops.
- **The barrier's polling load must be relaxed.** An acquire load per poll iteration emits `buffer_gl0_inv`/`gl1_inv` on every spin of every idle workgroup, invalidating the CU's caches under the one workgroup doing the phase's work. One acquire fence after the spin is enough. Uniform loads of data written by other workgroups go through the scalar cache, so the fence also needs `s_dcache_inv`.
- **Loads complete in order per wave (vmcnt).** A prefetch issued before a phase's own loads delays them; a release fence (`__threadfence`) before the barrier arrival waits for outstanding prefetch loads too. Idle workgroups now arrive without a fence.
- **A single wave is slow in a busy kernel.** The register-resident 1024-point Hadamard on one wave took 10 us inside the kernel and 2 us standalone; spreading each chunk over the workgroup's 8 waves (in-register strides 1-2, lane shuffles 4-64, one LDS pass for 128-512) brought prep to 2 us.
- **The dynamic-unit tail is W/units of the phase.** With 80 workgroups and 160 tiles a phase loses ~12% to imbalance; K-split units (320) halve it. Static first units (`u = blockIdx.x`) remove one contended atomic per phase.
- **Prefetching the next matvec's first block across the barrier moves bytes but does not save time**: the prefetch burst delays the prep phase's own loads by the same amount. Kept because it is not negative.

## Rows, drafters and batch (September 18, second pass)

The single-token kernel became a row kernel (`halo_rows.hip`): one launch processes up to 8 rows, which are a speculative block of one sequence or single tokens of up to 8 sequences (per-slot GDN state, conv ring, KV cache). GDN rollback is by replay: each pass caches its rows' conv outputs and gates and leaves the committed state alone; the next pass first commits the accepted prefix from that cache. Rejection therefore costs nothing.

Two drafters, both lossless (byte-identical greedy output):

- **MTP**: Qwen3.8-27B's NextN head (`mtp.*` tensors from shard 18 of the Qwen checkpoint), Q8 tiles, its own KV slot, chained drafting. 2.3-2.5 tokens/step at 2 drafts; 57-60 tok/s.
- **DFlash2** (`incoai/Qwen3.8-27B-DFlash2`): 5 Qwen3 layers with grouped dynamic convolutions, fed with the target's residuals after layers 5/19/33/47/61 (captured during the verify pass), block of 8 with a non-causal 2048 window over its own KV built from those features, top-16 per position plus the low-rank predecessor/successor selector. Its numpy reference (`/tmp` during development) agreed to cos 0.97-0.99 per layer at Q8/int8. 3.6-4.5 tokens/step; 58-73 tok/s.

Multi-row cost is the limiter. A pass costs 29 ms at 1 row, 45 at 4, 60-64 at 8 (verify 52 ms inside the DFlash2 loop). The 8-row matvec moved from scalar-fed `dot4` (140 GB/s) to int8 WMMA (175 GB/s): the 32-row tile is two 16-row matrices, the second made by one `permlanex16` swap; the hardware reads C row 2r from the lower lane half and 2r+1 from the upper, so the swapped matrix yields rows 2r+16 and 2r+1 (verified against a CPU reference to 1.5e-6). Activations come from L2 (eight distinct 16-byte addresses per fragment load coalesce); staging them in LDS cost occupancy for the whole kernel and was slower. What remains is the trit decode (~200 VALU per block per lane) and the WMMA issue; a byte-LUT decode with `v_perm` was estimated at ~120 and not built.

Measured: batch 8 = 126-134 tok/s aggregate (8 sequences x 60-64 ms/step); 8-row prefill 155 tok/s.

Lessons: K-split matvecs accumulate with atomics and need a zeroed output (the drafters' non-residual outputs were silently garbage until `zero_floats`); the occupancy API reports one workgroup per WGP too many when the VGPR file would be exactly full (253 registers, 6 waves), which deadlocks a cooperative barrier, so `coop_grid` subtracts it; wave reductions through DPP (`row_ror`, `quad_perm`, `permlanex16`) are 2.7x faster than `__shfl_xor`'s `ds_bpermute` and made the 8-token GDN loop viable.

### Next

1. Trit decode through an LDS byte table and `v_perm_b32` (cuts the 8-row matvec's VALU work by ~40%).
2. bf16 recurrent state (halves the 302 MB/token of GDN traffic; needs a perplexity check).
3. Sampling with the drafters (rejection sampling against the target distribution; both drafters are greedy today).
4. Continuous batching with a server front end.


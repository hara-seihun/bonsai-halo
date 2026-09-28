# SGLang Bonsai comparison

Source inspected: SGLang `28be39f72e7d8d71184e62208187990076eb63fe`.
The initial change was `b6ca9f22e1f1a5f4ff31469b870d136fb4952c59`; the current
head is `b1254e29cb48d261411f611737e10f4484da3af5`, published as
[PR #40863](https://github.com/sgl-project/sglang/pull/40863).

## What SGLang runs

The official [Bonsai AWQ recipe](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-AWQ-4bit)
loads an AWQ GEMM checkpoint with four-bit weights, group size 128 and zero points.
Its config uses `Qwen3_5ForConditionalGeneration`, hidden width 5120 and MLP width 17408.
This is the first Bonsai checkpoint, not Halo's Bonsai 2 checkpoint.

SGLang has no Bonsai-specific native ternary route in the inspected revision.
Compatible CUDA AWQ models select Marlin. HIP does not select Marlin; its
`AWQLinearKernel.apply` fully dequantizes each matrix, then invokes `torch.matmul`
on every call. `awq_gemm_triton` already exists in the same source tree, but no
runtime caller uses it.

The source paths are:

- `python/sglang/srt/layers/quantization/awq/awq.py`
- `python/sglang/srt/hardware_backend/gpu/quantization/awq_kernels.py`
- `python/sglang/kernels/ops/quantization/awq_triton.py`

## Actual weights and storage

[`tools/fetch_awq_fixture.py`](../tools/fetch_awq_fixture.py) downloads complete tensor
payloads by HTTP range from the pinned safetensors shard. It retains the config,
index, tensor header, individual payload hashes and range receipts. The
[fixture report](../../data/sglang-bonsai/fixture/README.md) identifies revision
`7f49f5d23a09087131cd50627366967f013b9f9e`.

Layer-0 `linear_attn.in_proj_qkv` and `mlp.gate_proj` contain only raw AWQ codes
7, 8 and 9, and uniformly use zero point 8. The runtime subtracts that zero point,
so these are scaled -1, 0 and +1 weights. Their paid storage is 4.15625 bits per
weight, including BF16 scales and packed zeros.

Halo's row-block format uses 28 bytes for 128 weights, including its FP16 scale,
or 1.75 bits per weight. That is 57.9% fewer bytes at this matrix boundary.
It is a format comparison across two checkpoint generations, not a same-model
RAM or throughput comparison.

## Upstream change and measurements

The follow-up narrows packed AWQ GEMM to gfx1151 HIP batches of 1 through 16
rows. Larger batches and other HIP architectures retain dequantization plus matmul.
CUDA/Marlin is unchanged.
It also replaces FP16/BF16 accumulation and split-K partials with FP32, writes the
final reduction into the activation dtype, and admits BF16 activations on HIP.
There is no new activation quantization and no change to checkpoint packing.

On this Radeon 8060S, nine alternating paired GPU-graph rounds gave:

| Actual checkpoint tensor | Rows | Activation dtype | Dequantize + matmul | Packed path | Median paired ratio |
|---|---:|---|---:|---:|---:|
| Attention, K=5120 N=10240 | 1 | BF16 | 1222.43 us | 169.60 us | 7.22x |
| Gate, K=5120 N=17408 | 16 | BF16 | 2116.53 us | 321.31 us | 6.65x |
| Gate, K=5120 N=17408 | 1 | FP16 | 2089.41 us | 279.21 us | 7.48x |

Weights are real checkpoint tensors. Activations are generated with seed 917,
not captured model inputs. Each graph includes 40 complete calls. The reference
includes dequantization and matmul; the candidate includes packed GEMM and the
split-K reduction. Host work shared the package during these panels, and the
clock and power samples are retained with the raw timings.

Relative squared error against FP32 matmul of the same decoded operands was
2.64e-6 and 2.75e-6 for the two BF16 panels, and 4.27e-8 for FP16. The reference
has essentially the same output-rounding error. Reduction order changes, so
this is not a bit-identical execution claim.

Eighteen focused kernel checks passed in 4.34 seconds. They cover both activation
dtypes, group sizes 32/64/128/full-K, split-K 1/8, output tails and cancellation
that overflows FP16 intermediate sums. Changed files pass SGLang's isort/ruff
settings and registered-test lint checks.

The registered HIP runtime dispatch test passes on this host. Sixteen additional
layer-level cases load both pinned Bonsai matrix tensors and check FP16/BF16,
strided inputs, bias, empty/1/16/17-row dispatch, finite outputs and error against
FP32 decoded-weight matmul. Eager Quark AITER imports were moved into their
own operations so AWQ imports without AITER. The in-tree ROCm AOT extension
now builds for gfx1151 against this host's Torch and supplies SGLang's native
KV-cache operations. With a pinned official Qwen2.5-0.5B AWQ checkpoint, six
FP16 greedy prompts matched the original SGLang dequantize-plus-matmul route on
all 72 generated token IDs and top-two alternatives. The largest chosen-token
log-probability difference was 0.0137. A six-prompt BF16 run completed, but
upstream rejects BF16 AWQ and cannot provide a BF16 full-model comparison. An
instrumented prefill confirmed 96 AWQ linear modules selected packed GEMM.
The PR is ready for review on gfx1151. No HTTP server benchmark, NVIDIA result
or whole-model speedup is reported. CDNA performance is unmeasured, so CDNA
keeps the previous route instead of taking the new default.

## Relation to Halo

The packed-consumption principle transfers. Halo's persistent engine and integer
activation map do not drop into SGLang's floating-point activation path. This PR
extends SGLang's own Apache-licensed code; it does not copy Halo or Kelana source.
It retains AWQ's four-bit storage. A lossless ternary repacker and floating-point
consumer would be a separate contribution with its own model and kernel results.

The [local receipt and replay directory](../../data/sglang-bonsai/README.md)
owns the acquired weights, benchmark sources, environment recipe, source snapshots,
raw timing JSONs, clocks, test output and submitted PR body. Halo serving code and
defaults were not changed.

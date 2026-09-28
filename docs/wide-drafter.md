# A drafter no longer turns wide prompt ingestion off

[Wide prompt ingestion](wide-prefill.md) gave the engine's own chat, `--bench` and `serve` paths a
prompt route worth 1.70x, and then excluded the only configuration this machine actually serves
from. `forward_batch` opened with

    if (mtp.loaded || df.loaded) throw std::runtime_error("wide batch inference does not capture drafter features");

and `prefill_width()` returned to eight rows whenever either drafter was loaded. `bonsai-halo.service`
starts with `--dflash`, so the resident server ingested prompts at about 145 tok/s while the route
that does 250 sat behind a capability gap rather than behind a measurement.

This closes it for DFlash2. A drafted 2663-token prompt ingests in **10.51 s instead of 18.39 s**,
**144.8 to 253.3 tok/s, 1.75x**, with the same generated tokens, the same drafts and the same
acceptance.

## What the drafter actually reads

The gap looked like a kernel problem and is not one. `ph_prep` captures a layer's features with

    if (a.cap) *(float4 *) (a.cap + ((size_t) row * NCAP + a.cap_idx) * D + base) = v;

and `v` at that point is `*(const float4 *) (a.x + rb + base)` — the **raw residual stream entering
the layer**, before the RMS norm, before the Hadamard, before the quantiser. Five of them, entering
layers 6, 20, 34, 48 and 62. Nothing about the capture depends on the prep phase it happens to sit
in; it is in there because in the persistent kernel that is the only phase that touches `x` at a
layer boundary.

On the wide schedule the host owns the layer boundary and the whole batch shares one residual
buffer, so the same capture is a strided copy of `batch_x`:

```c++
if (capture_wide) for (int j = 0; j < NCAP; ++j) if (layer == CAP_LAYERS[j] + 1)
    hipMemcpy2DAsync(hcap + (size_t) j * D, NCAP * D * sizeof(float),
                     batch_x, D * sizeof(float), D * sizeof(float), total,
                     hipMemcpyDeviceToDevice, stream);
```

13.1 MB of copy per 128-row pass, against a pass that streams several gigabytes of weights. The
boundary list moved to `CAP_LAYERS` in `halo_kernels.h` so the two routes cannot drift apart.

The rest is placement. `hcap` and `hfinal` were sized for `RMAX` rows, because a drafted prompt was
always ingested one eight-row pass at a time; they now hold `CAPMAX` rows, 13.1 MB and 2.6 MB.
`CAPMAX` is defined as `PASSMAX`, the widest pass `forward_batch` admits, rather than as its own
literal: `prefill_width()` falls back to eight rows when a drafter is loaded and the batch capacity
exceeds the capture buffer, so a literal would have let a future wider pass silently drop the
served path back to the eight-row route. Widening a pass now widens the capture with it.
`DflashParams::ctx_rows` is still `RMAX` long, so `Engine::prefill` hands the drafter a wide pass in
eight-row chunks with a `cap_row0` offset into the same buffer — 16 ingest launches per 128-row
pass, exactly the 16 the sixteen eight-row passes used to make. A pass too short for the wide
schedule falls to the eight-row slices, and there each slice gets its own rows of `hcap`; a
multi-slice `mode == 0` batch is refused rather than left to write rows 0..7 sixteen times.

`Engine::generate_dflash` had its own copy of the prompt loop, which is the other reason the CLI
could not reach this route. It calls `Engine::prefill` now, so the pass width, the drafter chunking
and the wide route are decided in one place for the CLI and the server alike.

## The drafter costs about one percent of prompt ingestion

Worth stating, because the obvious worry is that 16 drafter launches per wide pass eat the gain.
They do not. An ingest launch reads `w.fc` (Q8 `[D][NCAP*D]`, 131 MB) and the five layers' K and V
projections (2 x 1024 x 5120 each, 52 MB) — about 190 MB, or 0.8 ms at this box's 242 GB/s. It does
not touch the drafter's o_proj, gate, up or down, which only the block path reads. 333 ingests for a
2663-token prompt is roughly 0.27 s.

The measurement agrees: against [the undrafted panel](wide-prefill.md)'s 146.6 and 249.6 tok/s on
the same prompt and the same routes, the drafted arms here read 144.8 and 253.3.

## Measured

`--prefill-sweep` walks prompt-ingestion routes **inside one process**, resetting the sequence
between arms, so the arms share a loaded model, a warmed device and a clock. Each arm prints an
FNV-1a digest of its greedy token stream, which is the thing a lossless drafter must not change.
Measured on `d3bf8e9`, this change against canonical `cc5d259`, executable
`a0eac31a3739474b75b54efee6be2b408c4cc57f12d9788dae542cdbdfc65bc2`, under
`tools/run-batch-compare --engine`; published as `a530171` and `78f2073` after a rebase onto
`9ad103e` that touched no line measured here.

2663-token raw prompt (the first 7000 bytes of `PLAN.md`), `--raw --bench -n 16 --context 4096`,
`--dflash dflash2.safetensors`:

| route | prompt | prompt tok/s | generation tok/s | drafted / accepted | token digest |
|---|---:|---:|---:|---:|---|
| default, eight-row passes | 18.387 s | 144.8 | 22.59 | 70 / 7 (10.0%) | 7795239721289099842 |
| `wide-deployed` | **10.514 s** | **253.3** | 22.59 | 70 / 7 (10.0%) | 7795239721289099842 |
| `wide-commit-a4` | **4.118 s** | **646.6** | 20.49 | 77 / 5 (6.5%) | 5995860338611435317 |

The A4 row is from a second process that also carried mode 20, where mode 20 read 240.6 rather than
253.3; compare the A4 ratio inside that process, not across the two.

The wide arm runs second in that table, and this box's clock ramps over seconds, so the question is
whether the first arm is being measured on a cold clock. The short-prompt panel below answers it:
its rounds alternate, and route 0 reads 152.5 on a cold device and 155.4 after seven seconds of
work, **1.9%**. Across two processes the long-prompt ratio ranged 1.59x to 1.75x, which is
between-process drift of the kind this repository has measured at 8%, not an ordering artefact.

318-token prompt, `-n 24`, all three routes and two extra rounds of the first two, in one process:

| route | prompt tok/s | generation tok/s | drafted / accepted | token digest |
|---|---:|---:|---:|---|
| default | 152.5, 155.4, 154.7 | 43.7, 43.5, 43.5 | 63 / 15 (23.8%) | 12466745589763901277 |
| `wide-deployed` | 258.0, 254.3, 256.2 | 44.0, 44.0, 43.8 | 63 / 15 (23.8%) | 12466745589763901277 |
| `wide-commit-a4` | 660.4 | 43.8 | 63 / 15 (23.8%) | 12466745589763901277 |

**Generation is a measured null in every arm**, which is what the change should do: the route ends
at the last prompt token and the drafted decode loop runs exactly the code it ran before.

### The acceptance rate is the sensitive instrument here

Byte-identical output is necessary but weak evidence for a capture change, because DFlash2 is
lossless: the target model verifies every drafted token, so garbage features would still produce
correct text, only slower. What garbage features would destroy is the *drafts*. Ingesting a wide
pass through the new copy produced `70 drafted, 7 accepted` against the eight-row route's
`70 drafted, 7 accepted` on the long prompt, and `63 / 15` against `63 / 15` on the short one —
identical draft counts, identical accepted counts, identical tokens per step, and the same values
the **canonical pre-change binary** produced on the same prompt (`63 drafted, 15 accepted, 2.67
tokens/step`, generated text byte-identical to the new build's default arm).

### A4 does not preserve the drafted output on a long prompt

`wide-commit-a4` reproduced the digest on the 318-token prompt and **did not** on the 2663-token
one: different tokens, and acceptance fell from 10.0% to 6.5%. That is the documented approximate
map showing up where the sample is long enough to see it, not a capture defect — mode 20 reproduced
its digest in the same process. It is the reason `wide-deployed` is the route worth proposing as a
serving default and `wide-commit-a4` is not.

## Installed and accepted

Published as `a530171`, `78f2073` and `fbbdc04`, rebased onto `c685c66` and built into the
canonical executable `67a6a99bc7c55708da3a1c207ab24e9c2ac4c6199ae01e8f177e5eb2e4a37bba`. On that
binary, the same panel under `tools/run-batch-compare --engine` (the `PASSMAX` rename that followed
substitutes the same value into the same two comparisons and rebuilt to
`a117c3835dd2ca97e39a52c848a2b9f6cefb5a3bd58ce80ce01c5a284ceac808`):

| route | prompt | prompt tok/s | generation tok/s | drafted / accepted | token digest |
|---|---:|---:|---:|---:|---|
| default, eight-row passes | 17.979 s | 148.1 | 22.69 | 70 / 7 | 7795239721289099842 |
| `wide-deployed` | **11.278 s** | **236.1** | 22.59 | 70 / 7 | 7795239721289099842 |

Same two digests as the pre-install panel, same draft and acceptance counts, generation a null.
Single-stream undrafted `--bench -n 100` read 32.98 tok/s (30.32 ms/token) with three other
engineers' panels and builds on the box; that path never calls `forward_batch` and the only thing
this change does to it is 14.8 MB more device memory. The resident server came back on the new
binary.

## What this does not do

- **The MTP head still takes the eight-row route.** It reads `hfinal`, the final normed hidden, and
  on the wide schedule the per-slice head prep writes every slice to rows 0..7 of one `RMAX`-sized
  buffer. Giving each slice `hfinal + offset * D` is a two-line change; nothing measured it, and
  `prefill_width()` still refuses `mtp.loaded` rather than guess.
- **It does not change any default.** `--prefill-ffn` is still opt-in, and with no route selected
  the drafted path runs the code it ran before: same eight-row passes, same one ingest per pass.
- **It does not widen the verify pass.** The drafted decode step is still `DF_BLOCK = 8` rows
  through the persistent kernel. That is the larger remaining prize and it is a different change;
  see [the decode map](decode-map.md) for what a wider step is worth.

## Reproduce

```sh
tools/run-batch-compare --engine -p "$(head -c 7000 PLAN.md)" --raw --bench -n 16 --context 4096 \
    --dflash ~/data/bonsai2/drafters/dflash2.safetensors \
    --prefill-sweep off,wide-deployed --sweep-rounds 1
```

Raw stderr of every panel, the prompts and the executable hash are in
[`batch-comparison/wide-drafter/`](../../../data/bonsai2/batch-comparison/wide-drafter/).

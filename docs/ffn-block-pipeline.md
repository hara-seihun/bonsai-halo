# The A4 block waits on its two scalar words, not on its weight stream

Three engineers have now streamed something into this kernel's block loop a block ahead of the
block that spends it. The weight bytes go through [the dense cursor](ffn-dense-loads.md), the
activation fragments through [the LDS stage](ffn-b-operand.md), and
[the in-order counter](ffn-decode-schedule.md) explains why depth alone bought neither of them
anything. What nobody had read is where the compiled block actually stops.

It stops on **two scalar words and `TT` floats**. The gate/up block issues its weight-scale pair at
slot 20 of 625 and waits for it at 45; it issues one token scale at slot 486 and waits at 491, and
another at 546 waited at 549. Three memory round trips a block with **3, 5 and 24 slots of cover**,
in a kernel whose two large operands each have a whole block of it.

`ScalarStream` in [`kernels/ffn_dense_stream.hpp`](../kernels/ffn_dense_stream.hpp) gives those two
the same treatment: block `b`'s words are requested at the top of block `b-1`, held in `TT + 2`
registers, and rotated into place after the drain has spent the current ones. Nothing about a value
moves - same words, same `half_bits_to_float`, same `__shfl` broadcast positions, same `fmaf` chain,
same order - so this is a schedule change with a numerical control, and the control is flat:
**residual FNV-64 `6678137683989106976` in every traced sample of both arms and
`7839851009357901348` in every untraced one.**

## What the counter does before and after

`tools/vmcnt_cover.py` on the same listing, gate/up dense staged at two token tiles, and the whole
loop rather than one basic block:

| arm | slots | requests | waits inside the block, with their cover |
|---|---:|---:|---|
| scalars in the block that spends them | 623 | 14 | `vmcnt(6)` at 45 (**24 slots**), `vmcnt(0)` at 491 (**5**), `vmcnt(0)` at 549 (**3**) |
| scalars a block ahead | 629 | 14 | `vmcnt(6)` at 77 (52), `vmcnt(6)` at 606 (**581**), `vmcnt(2)` at 617 (571), `vmcnt(0)` at 620 (572) |

The two drains that cost 3 and 5 slots of cover become 571 and 572, and the block body itself -
the 529-instruction basic block that holds all 32 matrix instructions - **issues no request that any
wait in two iterations drains**. The down stage moves the same way: 13 and 170 slots of cover become
16, 33, 259 and 267. Six more issue slots a block pay for it.

## What it is worth

Mode 19, 32 streams, one token each, `--decode-prompt 64`, `--pin-clock`, arms alternating in one
process, paired by round.

| shape | arm 18, scalars in the block | arm 19, a block ahead | |
|---|---:|---:|---:|
| `ffn` phase, traced, six rounds | 31.520 ms | **30.765 ms** | **-2.40%**, 6 of 6 |
| the same normalised by the phases the arm cannot reach | 0.4682 | **0.4577** | -2.31% |
| traced device span | 98.859 | 97.991 | -0.78% |
| **step wall time, untraced, INT8 state, eight rounds** | **69.772 ms** | **69.115 ms** | **-0.99%, 8 of 8** |
| the same as aggregate generation | 458.6 tok/s | **463.0 tok/s** | **+0.95%** |

**A 128-row prompt pass is a null and the reason is in the same census.** At four token tiles the
gate/up arm is pair codes with 64 matrix instructions a block, so the same two scalar loads sit
inside a block four times longer and their cover is already an order of magnitude larger; measured,
`ffn` is 82.565 against 82.883 ms with the span moving -0.82%, which is inside the round-to-round
spread of that shape. This change is a decode-shape change and the document says so.

Registers do not pay for it. The deployed staged gate/up arm is 216 VGPR at seven waves per SIMD32
in both arms with one fewer spilled dword, and **the deployed staged down arm gains a wave slot**
(175 -> 163 VGPR, eight waves per SIMD32 to nine). Across all 272 instantiations of `k_proj_opt`,
47 gain a wave and three lose one; all three are `BSTAGE_OFF` shapes at 120 VGPR, which is exactly
a 24-register granule boundary on this device, and the only one of them a panel can reach is the
`--ffn-dense 15` control arm. Read that as the instrument's cost, not the product's.

## The defect this turn found, which is worth more than the change

**`docs/ffn-b-operand.md`'s selected activation stage had never run.** `create_ffn_batch` read its
setting with

```c
p->b_stage = env_int("HALO_FFN_BSTAGE");     // returns 0 when the variable is unset
```

so the `BSTAGE_WAVE` default in `FfnBatch` was overwritten with `BSTAGE_OFF` in every process that
did not name the variable, and the arm that document measured at -4.06% of the phase and +1.10%
aggregate generation was switched off in the same commit that selected it. The one-argument
`env_int` is right for an *override* whose leave-alone value is zero - `HALO_FFN_TT_GU`,
`HALO_FFN_W_DN`, `HALO_FFN_DN_MATS` all read that way correctly - and wrong for a *selected arm*.
There is now an `env_int(name, default)` beside it, which is the form `attn_batch.hip` already had.

Re-measured on this build, same shape, same instrument, arms alternating in one process:
**70.496 -> 69.732 ms, -1.46%, seven of eight rounds, 453.9 -> 458.9 tok/s**, residual FNV-64
`7839851009357901348` in every sample.

Together the repair and the scalar stream are **70.496 -> 69.115 ms on a 32-stream generation step,
453.9 -> 463.0 tok/s aggregate, +2.0%, with every logit bit unchanged.**

## Installed acceptance

Canonical `bonsai-halo` `e428aba6a3bcf95f97af24b8`, `tools/batch_profile` `e2834f1abc195ab8ff25365a`,
`tools/batch_compare` `14985424438cb8c586f879a3`, built from `deedced`. Four arms and four rounds in
one process on the installed binary, `--ffn-dense 15,18,17,19` so both knobs are walked, mode 19,
32 streams, INT8 state, `--pin-clock`
([`installed-dec32.json`](../../../data/bonsai2/batch-comparison/ffn-block-pipeline/installed-dec32.json)):

| arm | step | aggregate |
|---|---:|---:|
| stage off, scalars in the block - **what canonical ran before this turn** | 71.361 ms | 448.4 tok/s |
| stage off, scalars a block ahead | 70.941 | 451.1 |
| stage on, scalars in the block | 70.443 | 454.3 |
| **stage on, scalars a block ahead - the installed default** | **69.393** | **461.1** |

**-2.76% of the step, four of four rounds, 448.4 -> 461.1 tok/s aggregate, and the residual FNV-64 is
`7839851009357901348` in all sixteen samples of all four arms.**

**The two halves are not independent and the panel says which way.** With the activation stage off,
the scalar stream is worth -0.6% and in an earlier five-round panel it was an exact null; with the
stage on it is worth -1.71%. That is the mechanism stated from the other side: a block with 26
outstanding requests and a wait every twenty slots does not notice three more drains, and a block
whose two large operands are already a block ahead is stalled by exactly what is left.

## Using it

`HALO_FFN_SPIPE=0` restores the deployed schedule and `1` is the default;
`ffn_batch_set_spipe` is the same switch in-process, and `tools/batch_profile --ffn-dense 18,19`
walks both arms in one process on one clock. Every sample records `ffn_spipe` and `ffn_b_stage`.

```sh
tools/run-batch-compare --pin-clock --profile-tool --modes 19 --decode-streams 32 \
    --decode-prompt 64 --gdn-state 3 --ffn-dense 18,19 --traces 0 --rounds 8 \
    --warmup-ms 400 --rows 32 --out DATA/dec32-step.json
tools/run-batch-compare --pin-clock --profile-tool --modes 19 --decode-streams 32 \
    --decode-prompt 64 --ffn-dense 18,19 --traces 1 --rounds 6 --rows 32 --out DATA/dec32-arms.json
hipcc --offload-arch=gfx1151 -O3 -std=c++17 -Isrc -Ikernels --cuda-device-only -S -o /tmp/ffn.s \
    kernels/ffn_batch.hip && tools/vmcnt_cover.py /tmp/ffn.s k_proj_optILi1ELi2ELi1ELb0ELi2E
```

Raw samples, the listings and the per-instantiation register table are in
[`batch-comparison/ffn-block-pipeline/`](../../../data/bonsai2/batch-comparison/ffn-block-pipeline/README.md).

## What is left in this block, in order of what the ISA says it costs

1. **The activation stage's own fetch is the last uncovered request: `vmcnt(6)` at slot 77 for a
   load issued at 25, 52 slots of cover.** The source issues that fetch at the top of block `b` and
   commits it to LDS at the bottom, which would be 500 slots of cover; the compiler hoists the
   `ds_store` to slot 74 to free the four `b128` destinations, and the wait follows it up. Either
   the store is pinned low (a scheduling barrier before the commit, at 16 live VGPRs through the
   body) or the stage carries its own second buffer. This is the one request in the block that a
   wave still stalls on, and it is worth about what the three above were worth together.
2. **The peel, now that the block's memory is in order.** 279 of 602 slots are the radix-3
   expansion and [the shared gather header](../kernels/halo_expand.hpp) has never been tried on
   `dense_slice`. With `v_wmma_i32_16x16x16_iu4` measured at 11.0 issue slots per SIMD32
   (`bench/wmma_cost`), the block is 352 slots of matrix inside 629 counted, so a peel cut converts
   against 44% of the block and not against all of it.
3. **`ScalarStream` is not specific to this kernel.** Every block loop in this engine that reads a
   per-block scale in the block that spends it has the same three-slot drain, and
   `tools/vmcnt_cover.py` finds them in one compile with no GPU.

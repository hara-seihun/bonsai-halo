# The A4 FFN is not priced in bytes, and its activation loads cost 11.6% of the phase by their count

[The byte law](ffn-decode-bytes.md) closed this phase with a correlation: at 32 rows the dense
five-trit image reads 3.476 GB in 32.4 ms and the pair image 4.278 GB in 40.4 ms, both about
107 GB/s, so "the arm with half the issue is 21% slower, by its bytes". Everything built on this
phase since has been chosen against that sentence. **It is not a cause.** An ablation ladder inside
`k_proj_opt` says what the phase actually waits on, and the answer is a different quantity in each
direction:

| arm, 32-stream step, mode 19, two rounds | `ffn` ms | `ffn`/control | against deployed |
|---|---:|---:|---:|
| deployed | 32.58 / 32.59 | 0.4933 / 0.4932 | - |
| **activation address pinned to block 0** | 32.36 / 32.62 | 0.4904 / 0.4926 | **null** |
| **weight stream pinned to block 0** | 24.84 / 24.61 | 0.3566 / 0.3546 | **-27.9%** |
| both pinned | 24.44 / 24.35 | - | same as the weight arm |
| **the block's sixteen B loads CSE'd into one** | 28.77 / 28.86 | 0.4360 / 0.4360 | **-11.6%** |
| the weight-scale broadcast's sixteen `ds_bpermute` removed | 33.42 / 33.33 | 0.5052 / 0.5047 | **+2.4%** |
| both of the last two | 29.45 / 29.34 | 0.4439 / 0.4443 | -10.0% |

*Pinned* means the block index or the cursor step is masked with an **opaque device zero**, so the
same 26 requests issue in the same order with the same instructions and only the address stops
walking; a literal zero would let the compiler hoist the stream out of the loop and the arm would
measure a kernel nobody runs. All four probe instantiations census at 602-603 slots, 26
`global_load`, 32 `v_wmma`, 29 `s_waitcnt`, and **the deployed `dls = 1` instantiation is
byte-identical between the probe build and the shipped one**. The arms are numerically invalid by
construction and exist only to price the phase; `-DHALO_FFN_PROBE=1` compiles them and the default
build has no probe instantiation at all.

## What the ladder establishes

- **The bytes are not the price.** Making the *entire* 3.476 GB weight stream an L1 hit is worth
  7.9 ms of 32.6. A phase byte-bound at 107 GB/s against the 202-203 GB/s `bench/wstream` gives this
  geometry would have collapsed to its issue model when the bytes went away. What remains is the
  block loop **at** its issue model: 602 slots + 32 `v_wmma_i32_16x16x16_iu4` at 16 cycles is 1082
  cycles per wave-block modelled, 1127 measured in the pinned arm, against 1490 deployed.
- **Request count is the currency and locality is not.** Pinning the activation address - perfect
  locality, identical request count - is an exact null. Removing fifteen of those sixteen requests
  is -11.6%, with the same bytes on the weight stream, the same radix-3 peel and the same matrix
  work. [The packed state's law](gdn-state-coord.md) generalises: a phase can sit at half the byte
  roof and half its issue budget and be bound by **how many memory instructions a wave issues** -
  and here it is bound by their count even when every one of them hits.
- **LDS-pipe operations are free in this kernel.** Removing the weight-scale broadcast's sixteen
  `ds_bpermute` per block made it 2.4% *slower*. [The byte law](ffn-decode-bytes.md) handed that
  item to the lane as free money ("a scalar load plus one `v_cndmask` replaces the whole LDS round
  trip"); in `k_proj_opt` it is not. It also means a global load replaced by a `ds_read` is nearly
  pure profit here, which is what the shipped change is.

## What ships: a wave's activation fragments come from LDS

All `WV` waves of a workgroup own different weight row tiles and the **same** token group, so every
wave loads the same 2 kB of B fragments: sixteen `global_load_b64` per wave per 128-block at two
token tiles, 64 per workgroup for the same bytes. `kernels/ffn_b_stage.hpp` stages them instead.
A wave reads the block's 2 kB with **four** `global_load_b128` - the natural coalesced shape, 512
distinct bytes per instruction - writes them to its own LDS buffer, and the block body reads
fragments with `ds_load_2addr_b64`. The transpose is the point: a lane's staging load holds two
*adjacent tokens* and the WMMA wants *its own token* for eight slices, so LDS is the permutation
network, not a cache.

Requests per wave-block go **26 to 14**; the block loop's `ds` count goes 16 to 28 and its
instruction total barely moves. **Bit-identical by construction and measured that way:** the same
8-byte fragments reach the same lanes in the same order, through LDS instead of through L1.

| 32-stream generation step, mode 19, three rounds, one process | `ffn`/control | device span | aggregate |
|---|---:|---:|---:|
| activation stage off (`--ffn-dense 15`) | 0.4955 / 0.4961 / 0.4972 | 98.95 ms | 319.5 tok/s |
| **wave stage (default, `--ffn-dense 17`)** | **0.4793 / 0.4740 / 0.4751** | **97.97 ms** | **323.0 tok/s** |

**-4.06% of the phase, -1.0% of the step, +1.10% aggregate generation**, ranges non-overlapping,
residual FNV-64 `6678137683989106976` in every sample of every arm. Untraced on the same shape, six
paired rounds in one process: 98.44 -> 97.31 ms, **six of six**, -1.15%. A 32-row prefill pass is
-3.2% normalised (`ffn`/control 1.3097 -> 1.2678, 30.49 -> 28.38 ms raw). **A 128-row pass is a null
by construction and measured null** (1.2875 against 1.2893): above 32 rows the row rule hands
gate/up pair codes at four token tiles, which is not an instantiation the stage is built for.

### The arm that lost, and it is the more interesting one

**One copy per workgroup costs a barrier and the barrier costs more than the requests are worth.**
The shared stage is strictly better on paper - one staging request per *workgroup*-block instead of
four per wave, 11 requests against 14, exactly the ablation's -11.6% shape, 4 kB of LDS against 8 -
and it measures **+2.4%**, a loss, in the same builds where the wave-private stage wins 4%. One
`s_barrier` per block over four waves is worth about 9% of this phase. Both arms are in the tree
(`--ffn-dense 16` is the workgroup stage) because the pair is the evidence.

Two register facts that cost a build each and should not be rediscovered:

- **Unconstrained, staging costs about thirty registers and the allocator spends them**: 206 VGPR at
  seven waves per SIMD32 becomes 236 at six, and a wave slot is worth more than the requests. The
  staged instantiations carry their own `amdgpu_waves_per_eu(7)` through a template-dependent
  attribute, so they allocate 216 with spills and **every shipped body keeps exactly the attribute
  it had**. `HALO_FFN_BSTAGE_WPE` re-asks that question in one compile.
- **The spills are outside the block loop.** 9 dwords for the workgroup arm and 23 for the wave arm
  sound fatal and are not: the innermost loop contains zero `scratch_` instructions in both. Read
  where a spill lives before paying a wave slot to avoid it.

## Using it

`ffn_batch_set_b_stage` / `HALO_FFN_BSTAGE` select `0` global loads, `1` the workgroup stage, `2`
the wave stage (the default). `tools/batch_profile --ffn-dense 15,16,17` walks the three arms in one
process, and `8..13` are the ablation ladder in a `-DHALO_FFN_PROBE=1` build. The stage is built
only where it can pay and is otherwise not instantiated: ternary codes, one token group per
workgroup (`SHARE == false`), two token tiles or fewer.

```sh
tools/run-batch-compare --pin-clock --profile-tool --out DATA/stage-dec32.json \
    --modes 19 --decode-streams 32 --heads 1 --traces 1 --rounds 3 --ffn-dense 15,17
tools/run-batch-compare --pin-clock --profile-tool --out DATA/ablate.json \
    --modes 19 --decode-streams 32 --heads 1 --traces 1 --rounds 2 --ffn-dense 1,8,9,11   # probe build
```

Raw samples, the censuses and the probe arms are in
[`batch-comparison/ffn-b-operand/`](../../../data/bonsai2/batch-comparison/ffn-b-operand/README.md).

## Both of these questions are answered, and one of them by a defect in this change

[`docs/ffn-pair-stream.md`](ffn-pair-stream.md) took the ISA reading and ran the panel.

- **The stage below shipped inert**, which two engineers found within twenty minutes of each other:
  `create_ffn_batch` overwrote `b_stage` with an unset `env_int("HALO_FFN_BSTAGE")`, so the default
  this document selected was zero in every process that did not pass `--ffn-dense 16/17`, including
  the installed engine. [`ffn-block-pipeline.md`](ffn-block-pipeline.md) owns the repair and shipped
  it. The independent confirmation, on a traced phase panel rather than a step wall: `ffn`/control
  **-4.29% median paired over four rounds, four of four**, which is this document's own -4.06%
  arriving in the product for the first time.
- **The weight stall is not collectable and the issue floor is not the binding term.** A block
  cursor on the pair arm that removes *every* in-block `vmcnt` wait - verified in the compiled loop,
  at no register cost - is +3.5% of the phase, exactly its instruction cost. Four weight images
  spanning 23% of byte count put this phase on one line at **107 GB/s** regardless of operand work,
  stage, cursor or image, so the 24.5 ms issue floor sits under a 32.5 ms byte term and a slot cut
  at this width cannot pay. Try the gather on the 128-row shape instead, where the same phase is
  four token groups deep.

## The next useful question (as it stood)

**The two terms this ladder separates do not overlap, and only one of them has an owner.** The
weight-stall term is 7.9 ms of 32.6 and every cursor depth and cursor position
[has now been measured against it](ffn-decode-schedule.md) without moving it, because a wave's
`vmcnt` counter is in order and the block's own activation loads sit between the cursor's request
and its use. **The stage changes that geometry**: the block's fragments are `lgkmcnt` now, not
`vmcnt`, so the only vector loads between the cursor's request and the block's work are the four
staging loads. The order `[staging load][weight request][block work][commit]` is one the in-order
counter can honour, and whether the shipped stage already collects part of that 7.9 ms - or whether
moving the commit's wait to `vmcnt(6)` collects the rest - is one ISA reading and one panel.

> **Answered, and the answer was a third thing.**
> [The block pipeline](ffn-block-pipeline.md) read the compiled block: the stage already collects
> its half of the 7.9 ms, the commit's wait is `vmcnt(6)` and not `vmcnt(0)`, and what the block
> still stalled on was neither operand but its two SCALAR words - the weight scale waited 24 slots
> after its request and each token scale 3 to 5 slots after its own. Streaming those a block ahead
> is another -2.40% of this phase and -0.99% of the step. **It also found that this document's
> selected default has never run**: `create_ffn_batch` read `HALO_FFN_BSTAGE` through the
> one-argument `env_int`, which returns 0 when the variable is unset, so `BSTAGE_WAVE` was
> overwritten with `BSTAGE_OFF` in every process that did not name it. Repaired there; re-measured
> at -1.46% of the step, 453.9 -> 458.9 tok/s.

**And the issue floor is now a measured 24.5 ms of a 32.6 ms phase, of which the radix-3 peel is
279 of 602 slots.** [The peel gather](mv-peel-gather.md) cut the *other* ternary expansion in this
repository by 28% for -1.7% of its phase, in a kernel bound by that chain. This kernel runs at 100%
of its issue model once its memory is free, which is the condition under which a slot cut should
translate, and nobody has tried the gather on `dense_slice`'s five-trit peel.

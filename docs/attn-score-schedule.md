# The score unit's reduction is a tree, not a lane layout

[`docs/long-context-attention.md`](long-context-attention.md) priced the attention score unit and
left one target: 64 slots of `v_fmac_f32` per key against 47 of cross-lane DPP reduction and 63 of
`s_delay_alu`, on a phase that is 84% of attention and 28% of an 8k prefill. It proposed a
lane-owns-a-key rewrite worth about 1.8x and filed it as a **reassociation** needing its own quality
panel, beside the flash-attention and larger-`ACHUNK` candidates it had already shown cannot
reproduce the deployed bits.

The reassociation is not needed. This document is what the key loop gives up without it, the census
that set every width in it, and the one register trap that ate the first version.

## The tree is a fixed object, and three of us got its order wrong

`warp_sum` is five steps: `quad_perm(1,0,3,2)`, `quad_perm(2,3,0,1)`, `row_ror:4`, `row_ror:8`,
`permlanex16`. The first two are lane XOR 1 and XOR 2, so after them every lane of a quad holds

    Qk = (t(4k) + t(4k+1)) + (t(4k+2) + t(4k+3)),   leaf t(l) = the fmaf chain over dims 8l..8l+7

**`row_ror:N` makes lane n read lane ((n mod 16) - N) mod 16, not (n + N).** The check that settles
this without reading a rotation convention at all is the canonical wave prefix sum, `row_shr:1, 2,
4, 8`: it is an inclusive scan only if lane n receives from lane n-N, which fixes the direction for
the whole shift/rotate family. So lane 0 - the only lane whose value the score loop stores - reads
lane 12 at `row_ror:4` and lane 8 at `row_ror:8`, and finishes holding

    ((Q0 + Q3) + (Q2 + Q1)) + ((Q4 + Q7) + (Q6 + Q5))

The tree is balanced and its leaves and quads are contiguous, **but the quads of each 16-lane row
pair 0-3 and 2-1, not 0-1 and 2-3.** Three engineers on this lane independently derived the
index-ordered version within an hour, from the mnemonic, and confirmed each other. `leaf_of_pos`
in `kernels/attn_leaf.hpp` is the one-line relabeling that fixes it, and it is an involution.

What survives is the part that matters: **the reduction's shape belongs to the reduction, not to
the lane layout that executes it.** A single lane can reproduce these bits with a carry stack, so
the lane-owns-a-key rewrite is exact and the quality panel it was filed behind is not owed - but
only against the permuted leaf order.

**This change does not depend on any of that**, and the distinction is the useful part. There are
two kinds of exactness argument about cross-lane hardware:

- **Replay** the same cross-lane ops, in a different order, over independent values. Exact by
  construction: you never assert what the op does. `warp_sum_n` is this.
- **Reconstruct** what the ops compute, using a model of them. Needs the device, because the model
  is the risk. Lane-owns-a-key is this, and the model was wrong.

What the deployed shape actually costs is not the tree. It is walking eight of them one after
another: five dependent DPP steps per row, eight rows, and the compiler filling the gaps.
`s_delay_alu` was 29% of the block.

## Four schedule changes, no reassociation

`attn_chunk_unit_fast` in [`kernels/phases.hpp`](../kernels/phases.hpp), helpers in
[`kernels/attn_score.hpp`](../kernels/attn_score.hpp). Each is exact by construction:

1. **Breadth-first wave reductions.** `warp_sum_n` does level one for all eight rows, then level
   two: same ops per value, same order per value, seven independent steps behind every dependent
   one.
2. **Batched block reductions.** The softmax called `block_max_256` and `block_sum_256` once per
   row, two `__syncthreads` each. A TT=8 unit spent **32 barriers** reducing 8 x 128 floats.
   `block_max_n`/`block_sum_n` do all eight with a `[NW][TT]` scratch at the tail of `c.lds`, four
   barriers, and the per-row tree untouched: `warp_sum`, wave `w`'s result into `red[w]`, the eight
   wave results combined in wave order.
3. **Key lookahead.** The block for `p + NW` is issued as soon as the current block is expanded
   into registers, so its round trip is covered by this key's dot products instead of a `vmcnt(0)`
   at the top of the next iteration. `#pragma unroll 1` on that loop means the compiler could not
   do this itself.
4. **Four-key value block (arm 2 only).** One `ds_read_b128` per row per four keys, and the four
   value loads issue together.

## Census, before any GPU

`hipcc --offload-arch=gfx1151 --cuda-device-only -S` on `kernels/attn_batch.hip`, counted with
`tools/isa_loop_count.py`. Per wave per unit at `TT=8`, `ACHUNK=128`, `HD=256`:

| block | executed | arm 0 deployed | arm 1 rescheduled | arm 2 + value block |
|---|---:|---:|---:|---:|
| key loop | 16x | 219 | **122** | **122** |
| value loop | 128x / 32x | 25 | 25 | **80 per four keys** |
| softmax + prologue | 1x | 1329 | 1236 | 1271 |
| slots per unit | | 8033 | **6388** | **5783** |
| `__syncthreads` per unit | | 35 | **7** | **7** |
| VGPR / waves per SIMD32 | | 92 / 16 | **92 / 16** | **93 / 16** |

`s_delay_alu` in the key loop: **63 to 2**. The arithmetic is untouched at 64 `v_fmac_f32` per key
in every arm.

## The register trap, and the two things it teaches

The first version of arm 2 was **120 VGPR and 12 waves per SIMD32** against the control's 92 and 16.
The value loop, not the reductions, spent them: the scheduler hoists all eight rows' `float4` score
reads to cover LDS latency, and eight `float4`s is 32 live registers. One
`__builtin_amdgcn_sched_group_barrier(0x100, 1, 0)` / `(0x002, VW, 0)` pair per row pins one read
to its four FMAs and brings the kernel back to 93.

That matters here more than it would elsewhere, because this is the kernel where wave slots are the
measured medicine: `HALO_ATTN_TT=16` halves the key traffic, costs 92 -> 180 VGPR and 16 -> 8 waves,
and **loses 18%**. A schedule change that quietly bought registers would have been measured as a
loss and filed as one.

The census also corrected an assumption worth more than the change: **the deployed value loop is
already better than its source reads.** `sc[i][p]` and `sc[i+1][p]` are 128 dwords apart, so the
compiler emits `ds_load_2addr_stride64_b32` - four instructions for eight rows, not eight. A
four-key block therefore removes 5 slots per key, not 20. Count emitted LDS instructions, not source
ones.

Two widths were swept and are not free parameters any more:

- `ATTN_RED_W`, rows reduced together in the key loop: **no effect on registers at all** (92 at 1,
  2, 4 and 8; 180 at TT=16 either way). Full breadth is free, so it is the default.
- The value block width: 1 costs nothing, 4 costs one register with the group barrier and 28
  without, and a scalar-read variant of 4 stays at 116 because the scheduler hoists the same values
  either way. That variant is not in the tree.

## What the device said

**The score phase is 11.7% faster and nothing else moves.** Two panels, `--prefill-scan 1536` at
mode 19 in 128-row passes with `HALO_ATTN_TRACE_SPLIT=1`, arms walked in one process as
`--attn-sched`. Eight phases that the change cannot reach are the in-panel control.

Raw panels, all six, under
[`data/bonsai2/batch-comparison/`](../../../data/bonsai2/batch-comparison/): `attn-sched`,
`attn-sched-rev`, `attn-sched-identity`, `attn-sched-decode`, `attn-sched-installed`,
`attn-sched-probe`.

| panel | arm order | arm 1 score / untouched, vs arm 0 | arm 2 |
|---|---|---:|---:|
| [`attn-sched/`](../../../data/bonsai2/batch-comparison/attn-sched/) | 0,1,2 | **0.880** | 0.873 |
| [`attn-sched-rev/`](../../../data/bonsai2/batch-comparison/attn-sched-rev/) round 0 | 2,0,1 | **0.878** | 0.911 |
| [`attn-sched-rev/`](../../../data/bonsai2/batch-comparison/attn-sched-rev/) round 1 | 2,0,1 | **0.883** | 0.895 |

Paired per pass, 12 positions from 0 to 1408: arm 1 is **0.886 +- 0.011** over 24 pairs in the
reversed panel and 0.884 +- 0.017 over 12 in the forward one. The ratio is flat across the position
walk (0.865 to 0.919), so it is a constant factor on the phase and not a fixed cost.

**The raw wall time says +19.9% and all of it but 4% is the clock.** The forward panel reads 630.1,
654.7 and 755.3 prompt tok/s for arms 0, 1 and 2 in that order - and its untouched phases fall
2209 ms to 1853 ms across the same walk, 16%. The reversed panel puts arm 2 first and it reads
696.5 against arm 0's 753.4. This is the third time this exact kernel has produced a large
clock-shaped result that an untouched-phase control had to remove; treat the control as mandatory.

**Bit-identical, against two hashes the lane published before this code existed.** Residual FNV-64
`7446865760224376151` at 128 rows and `18439807992172728808` at 256, in every arm of every round -
the hashes [`ffn-dense-loads`](ffn-dense-loads.md) and the sequence output lookahead published for
these shapes. Eight untouched phases and the attention combine reproduce as well.

### The four-key value block is a measured null, and that is the most useful number here

Arm 2 removed **28% of the unit's issue slots** against arm 0 - 8033 to 5783 per wave - at 93 VGPR
and the same 16 waves per SIMD32, and it measured the same as arm 1, which removes 20%. Against the
census the conversion is stark:

| arm | issue slots per unit | predicted | measured |
|---|---:|---:|---:|
| 1, key loop and barriers | -20% | -20% | **-11.7%** |
| 2, plus the value loop | -28% | -28% | **-11.7%** |

So slots removed from the **key loop** convert at about 0.6, and slots removed from the **value
loop** convert at **zero**. The value loop is not issue-bound. It is not the arithmetic either: 8
`v_fmac` per key per wave. What is left is its one `global_load_u16`, which moves **64 bytes per
wave instruction** - the narrowest load in this engine - 128 times per wave per unit.

Arm 2 is not in the tree. This section is its evidence.

## Decode does not move, and its bits match too

`--decode-streams 32` on the canonical build, arms walked in one process
([`attn-sched-decode/`](../../../data/bonsai2/batch-comparison/attn-sched-decode/)):

| | arm 0 | arm 1 |
|---|---:|---:|
| `sequence-core` | 3.289 ms | **3.042** |
| `gdn-resident-core` | 45.054 | 45.185 |
| `ffn` | 32.801 | 32.819 |
| step wall, untraced | 108.98 ms | 107.92 |

Every phase the change cannot reach is inside 0.3%, attention is -7.5%, and the step is a null -
0.25 ms of 110. Residual FNV-64 `8087160186626784132` in both arms, which is the hash
[`ffn-dense-loads`](ffn-dense-loads.md) published for this step. At `--decode-prompt 64` a decode
step's attention is 3% of it; the shape where this would matter is a decode step against a long
prefilled context, which is measured separately.

## Installed and accepted

Canonical `ccdb7fe`, executable `2dab4a9a53872fdd13d5167d...`, panel
[`attn-sched-installed/`](../../../data/bonsai2/batch-comparison/attn-sched-installed/).
`--prefill-identity` over a whole
document at two pass widths: **95,354,880 full-vocabulary logits, zero nonfinite, FNV
`17618767336181543518` at 128 rows and at 256** - the hash
[the position axis](long-context-attention.md) published before this code existed. The resident
service came back on it and answers on 8471.

## What this is worth on a whole prompt

The score phase is 160.7 ms of a 2370 ms device span for a 1536-token prefill at mode 19, so 11.7%
of it is **0.8% of that prompt**. Put on [the position axis](long-context-attention.md), where
attention is 28% of an 8k prefill and 43% of a 16k one and the score units are 84% of attention,
the same factor is **-2.8% at 8k and -4.2% at 16k**, and nothing at 384 tokens. That is the honest
shape of this result: it is a constant factor on the only phase in this engine that grows with
prompt length, and the panels this lane publishes by default are taken where that phase does not
exist.

## Reading this if you are taking the next step

**The value loop is where this unit now spends itself, and it is not issue-bound.** Per wave per
unit the split is about 1952 slots in the key loop, 3200 in the value loop and 1236 once. Removing
20% of the value loop's slots bought exactly nothing (the section above), so the next attack there
has to be about the *shape* of its memory traffic rather than its instruction count: 128
`global_load_u16` per wave per unit, 64 bytes each, one per key. Every thread owns one dimension
and V is stored `[key][dim]`, so a thread's successive keys are 512 bytes apart and nothing wider
than a `u16` is available to it. A `[dim][key]` V cache would give each thread eight consecutive
keys in one `global_load_b128` and keep the accumulation order - so it is exact - at the cost of a
scattered write on every decode step, which is the trade to price first.

The lane-owns-a-key shape is still worth about 1.8x on the key loop, and the first section removes
its blocker: it is exact, not approximate, against the permuted leaf order. Its open cost is the
same address question from the other side - with K in `[key][dim]` order a lane that owns a key
reads 16 bytes at a 512-byte stride, so one `global_load_b128` touches 32 cache lines instead of 4.
Two engineers are on it as of this writing, both swizzling the K cache to make that read coalesce.
Whoever lands it should note that after this change the key loop is 1952 of about 6400 slots, so an
1.8x there is worth 13% of the unit, not 45%.

One number for whoever is fighting the persistent kernel's register ceiling: `ph_attn` deliberately
keeps the deployed body (`SCHED = 0`). The rescheduled unit is 92 VGPR against the deployed 92 as a
standalone kernel at TT=8, and 180 against 180 at TT=16, so it is very likely free inside
`k_forward_rows` too - but that kernel's VGPR count sets `rows_grid_size` for every wide pass and
three engineers were measuring it the hour this landed. Flipping that default is one line and one
census.

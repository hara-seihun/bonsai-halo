# One key chunk, two heads: what reuse is worth on the score unit

> **The reuse rectangle is register-bound at about 16, and both corners are now priced.**
> `TT=16` was rejected at -18% in the row-major body because `qr[16][8]` is 128 VGPR, and the
> obvious follow-up was to build it in the leaf coordinate where the query is wave-uniform. It does
> not help: `k_attn_wide<16,0,1,LEAF>` is **182 VGPR and eight waves**, against the row-major body's
> 180 and eight. `hs[HG*TT]`, `acc[HG*TT]` and `mx/lsum[HG*TT]` cost three to four registers per
> unit of the rectangle `HG x TT` in *either* coordinate, so `TT=16, HG=1` and this document's
> `TT=8, HG=2` are the same area and the fold already occupies the better-conditioned corner.
> `LDS_FLOATS` is not the binding constraint either - `k_attn_wide` inherits the persistent kernel's
> 9504 bytes and eight waves a workgroup is 76 KB of a 128 KB WGP - so raising its LDS budget to
> reach `TT=16 x HG=2` buys nothing the register file will not take back.
> See [the chunk-partial window](attn-partial-window.md), which is where that turn went instead.

[The score unit's lane coordinate](attn-score-lane.md) removed 2.4x of the score loop's census and
bought **-6.1%**. [The long-context measurement](long-context-attention.md) has the same phase at
**400-450 GB/s of request bandwidth** in every position cell it took, against 67-200 MB of unique
K/V bytes per pass. Neither number is an issue bound, and both point at the same lever: how many dot
products a key chunk's bytes are charged to. That is the rectangle `TT x HG` — rows times query
heads per score unit — and the lane had two points on it that disagree about its price:

| arm | reuse | registers | measured |
|---|---:|---|---|
| `TT=16`, row-major ([long-context](long-context-attention.md)) | 16 | 92 -> 180 VGPR, 16 -> 8 waves | **+18%** |
| `HG=6`, `TT=1`, row-major ([wide attention](wide-attention.md)) | 6 | 70 VGPR, 16 waves | **2.6x** |

Reuse wins when it is free and loses when it costs wave slots. This document prices the third
point: a fold that is *nearly* free, which only the leaf coordinate can build.

## Why a head is cheap here and expensive there

In the row-major body each lane holds eight dimensions of the query for every row — `qr[TT][DPL]`,
64 VGPRs at eight rows — so a second head is 64 more registers. That is why `HG > 1` was restricted
to one-row groups, where the whole array is eight.

In the leaf body the query is **wave-uniform**. A lane owns a key and sums its whole dot product, so
every lane of the wave multiplies the same query value and it arrives on the scalar unit. A folded
head costs one score register per row plus its share of the summation tree. The value loop folds for
free in either coordinate for a different reason: `V` is indexed by the **KV** head, so the six query
heads of a group read the same values, and `acc[HG][TT]` takes one `vc` load where two used to.

`attn_chunk_unit_leaf` takes `HG` heads and `HB` of them per leaf walk. `HB = HG` reads a leaf once
and holds `HB * RB` tree partials; `HB = 1` walks the leaves once per head, holding `RB`, and
re-reads sixteen bytes the same wave read a few hundred cycles earlier.

ISA census, gfx1151, the 32-key tile block of `k_attn_wide<8, 0, HG, LEAF, HB>`
(`tools/isa_loop_count.py`, no GPU):

| arm | block instructions | `v_fmac` | `global_load` | VGPR / waves |
|---|---:|---:|---:|---|
| `HG=1` — the arm it replaces | 1449 | 1024 | 16 | 94 / **16** |
| `HG=2, HB=2` — one walk, two heads | 2853 | 2048 | **16** | 112 / **12** |
| `HG=2, HB=1` — two walks | 2823 | 2048 | 32 | 145 / 9 |

**Twice the arithmetic behind the same sixteen loads**, for four wave slots. Per unit of work the
fold is also marginally cheaper to issue: 1.393 slots per `v_fmac` against 1.415.

Two registers of the unfolded body had to go first. `mx[HG*TT]` and `lsum[HG*TT]` were held from the
block reductions until the partial write, which is `2 * HG * TT` registers for two numbers one lane
stores: at `HG=2` that alone was a wave granule (182 -> 112). The fold writes `m` and `l` as soon as
the reduction has them. The unfolded arm keeps the arrays, so its codegen is still the one
[the dispatch table](attn-body-dispatch.md) was read on — 94 VGPR, sixteen waves, unchanged.

## Exact by construction, and measured so

Nothing crosses the head axis: separate query, separate scores in their own LDS slice, separate
softmax, separate partial, and the combine reads partials per `(row, head, chunk)` exactly as
before. The fold moves which workgroup owns which `(row, head)` and nothing else.

Residual FNV-64 in **every sample of every panel below**, folded and unfolded, leaf and row-major:
`5674578608483866172` at position 1024 and `8952439863933010475` at 3072. The deployed drafted path
returns greedy digest `15165079467408989842` in both coordinates.

## What it is worth

Mode 19, 128-row passes, `--attn-coord 1`, both arms alternating in one process on one pinned clock,
`HALO_ATTN_TRACE_SPLIT=1` so the score kernel is its own phase. Control is the seven phases the
change cannot reach. Raw in
[`attn-reuse-fold/`](../../../data/bonsai2/batch-comparison/attn-reuse-fold/README.md).

| position 3072, ms | `HG=1` | `HG=2` |
|---|---:|---:|
| `sequence-core-score` | 36.959 / 36.428 | **34.895 / 34.807** |
| `sequence-core-combine` | 8.257 / 8.204 | 8.348 / 8.360 |
| control (seven phases) | 118.903 / 118.256 | 117.478 / 117.300 |
| device span | 165.990 / 164.742 | **162.608 / 162.308** |

**-4.06% on the score kernel normalised, -1.76% of the device span**, both rounds the same
direction. At position 1024 the same panel is a null: 18.371 -> 18.431 ms, **-1.0% normalised**.

### The position curve, which is the result

`--prefill-scan 4096` ingests a document in 128-row passes and charges each pass to the position its
first row sat at, so one walk per arm gives the whole curve on one clock
([`scan-4096-leaf.json`](../../../data/bonsai2/batch-comparison/attn-reuse-fold/scan-4096-leaf.json),
round 1; round 0 pays the first walk's page faults in both arms).

| position | score `HG=1` | score `HG=2` | normalised |
|---:|---:|---:|---:|
| 0 | 1.168 | 1.473 | +25.7% |
| 512 | 6.971 | 6.996 | -0.4% |
| 1024 | 12.966 | 12.658 | -2.8% |
| 2048 | 24.969 | 23.874 | -3.8% |
| 2944 | 35.593 | 33.780 | -4.9% |
| 3584 | 42.993 | 40.504 | -5.3% |
| whole document | 775.0 | 745.5 | **-3.8%** |

Whole-document prompt throughput, same walk: 849.7 -> **855.4 tok/s, +0.67%**.

The fold costs where a unit is mostly its own prologue and pays where its key loads are worth
something, so the default is a shape rule — `attn_fold_for`, fold at eight or more key chunks, which
is the first crossing with every sample beyond it negative. `HALO_ATTN_FOLD` and `--attn-fold` pin
an arm: 1 never, 2 the fold, 3 the two-walk arm, 6 the whole KV group at one-row groups.

### The two-walk arm loses, and that is a mechanism not a detail

`HG=2, HB=1` is **+22%** at position 1024 (22.506 against 18.371 ms). It reads each leaf twice from
the same wave that just read it — an L0 hit by any reasonable model — and pays 32 loads and three
wave slots for it. **Re-reading a cached line is not a substitute for holding it in a register while
a second consumer uses it.** Anyone designing reuse on this device should read that against
[the FFN's ablation ladder](../orchestration/HANDOFF.md), which found the opposite currency on
`k_proj_opt`: there pinning an address (perfect locality, same request count) was an exact null and
removing requests was worth -11.6%. Here the request count falls by half and buys 0 to 5%. **Request
count is not a universal currency either. What a wave waits on is.**

## The ablation: the whole memory term of this phase is 15%, and two thirds of it is the values

The fold's own result asks the next question — if halving the key-load instructions is worth 4%,
what is *all* of the memory worth? A `-DHALO_ATTN_PROBE` build answers it directly. `PROBE` bits pin
the key chunk's offset and the value chunk's with an opaque device zero (`blockIdx.z`, which is 1 in
every launch that reaches this body and which the compiler cannot fold), so every unit issues **the
same instructions, the same request count and the same addresses within a chunk** against a working
set the whole device already has resident. The arms compute the wrong attention and only their time
means anything. Every probe instantiation reads **94 VGPR and sixteen waves**, exactly the deployed
arm, so the pair is not confounded by occupancy.

Position 3072, mode 19, 128-row passes, one process, normalised on the seven untouched phases
([`ablation-3072.json`](../../../data/bonsai2/batch-comparison/attn-reuse-fold/ablation-3072.json)):

| arm | `sequence-core-score` | normalised | vs deployed |
|---|---:|---:|---:|
| deployed | 39.248 ms | 0.31115 | — |
| **keys pinned** | 34.928 | 0.29596 | **-4.88%** |
| **values pinned** | 32.193 | 0.27289 | **-12.29%** |
| both pinned | 31.441 | 0.26484 | **-14.88%** |

Three things fall out of four numbers.

1. **Every memory-side change to this kernel is capped at 15%.** Making the entire K and V working
   set resident — which no real change can do — leaves 85% of the phase standing. Cache blocking,
   fp8 caches, a smarter chunk order and this fold are all playing for the same sixth of the phase.
2. **The values are worth 2.5x the keys**, at identical bytes. The score loop everyone has optimised
   — the lane coordinate, the reduction schedule, this fold — is the *cheap* half of the memory. The
   value loop is `#pragma unroll 1` with one `global_load` per key and eight `fmaf`, so it holds one
   outstanding load per wave and cannot cover a miss with anything but other waves.
3. **The fold captured most of what it could.** -4.06% measured against a -4.88% key-side ceiling is
   83% of the available key-side time, on a phase where the key side is worth five percent.

## What this says about the phase, which is worth more than the 4%

Three independent changes now bound the score unit from three sides:

1. **It is not issue-bound.** 2.4x off the census bought -6.1% ([attn-score-lane](attn-score-lane.md)).
2. **It is not request-bound below about a thousand positions.** Halving its key-load instructions at
   constant work and constant bytes is a null at 1024 and -4% at 3072. The fold's gain grows with
   position, which is what a phase looks like when its working set is leaving a cache rather than
   when its wave is short of instructions: a layer's K+V is 4.2 MB at position 1024 and 12.6 MB at
   3072, against a 32 MB last-level cache shared with the weight stream.
3. **It is not short of wave slots.** Twelve waves per SIMD32 doing twice the work per load beat
   sixteen doing half, everywhere past position 640.

The arithmetic left unexplained is still large: at position 3072 the score kernel issues about
1.5 G wave-instructions per pass, which is 7.8 ms at one slot per cycle on 80 SIMD32 at 2.4 GHz, and
takes 34.8. The ablation above removes its memory and leaves 31.4. **So the phase is not its
instructions, not its requests, not its bytes and not its occupancy**, and what is left is the shape
of the unit itself.

The next place to look is named by what the unit does between the key loop and the value loop. The
softmax runs `block_max_256` and `block_sum_256` **per row**: sixteen 256-thread LDS reductions,
each a dependent tree behind `__syncthreads`, for eight rows of one chunk, with the whole workgroup
idle inside each one. [The token loop's row reductions](gdn-token-reduce.md) were exactly this shape
in the recurrence and batching them into one block was **-16% of that phase**. Nobody has tried it
here, the rows are independent, and batching them moves no value: same `fmaxf` tree over the same
256 lanes, same `__expf`, same block sum, one barrier instead of sixteen.

## The coordinate is the gate, and it is half open

The fold only exists in the leaf coordinate, and `attn_coord()` still defaults to 0 for one written
reason: nobody had measured the persistent kernel reading an interleaved cache through `k_pos_off`.
This turn measured it on the deployed drafted path — `--bench --dflash q4`, 1697-token prompt in
256-row passes, 48 greedy tokens, one pinned-clock run per arm:

| | prompt tok/s | drafted tok/s | verify ms/step | draft ms/step | digest |
|---|---:|---:|---:|---:|---|
| `HALO_ATTN_COORD=0` (default) | 510.2 | 31.38 | 47.991 | 8.652 | `15165079467408989842` |
| `HALO_ATTN_COORD=1` (leaf) | 505.8 | 31.34 | 48.073 | 8.641 | `15165079467408989842` |

Same digest, same 189 drafted and 21 accepted, and **-0.9% / -0.1% / +0.2%** across a cross-process
pair whose clock is uncontrolled. So the interleaved cache does not cost the persistent kernel the
granule its addressing arithmetic suggests it might: a row-major wave reading a leaf-ordered cache
touches 32 lines per load instead of 4, and the loop reads every one of those lines sixteen times,
so L0 absorbs it.

That is one pair, not a default. What would settle it is an **in-process** instrument for the
persistent-kernel route, and `tools/batch_profile` does not have one: its decode path rejects mode 0
outright (`profile modes: 4,5,9,10,11,16,17,18,19,20`), so the eight-row route this engine serves
solo requests on cannot be A/B'd inside one process at all. Building that is worth more than this
result, because the coordinate is **-6.7%** of the score phase on the wide route at 3072
(36.175 against 39.241 ms normalised, this turn's own panel) and the fold rides on top of it:
**-10.4% together** against today's default coordinate.

## Installed

Canonical `df44665`, `bonsai-halo` `6f2a48ca90d2b2ac98098375`. The installed acceptance is the
deployed route, which this change cannot reach and which therefore has to be unmoved: greedy digest
`15165079467408989842`, 189 drafted and 21 accepted, 524.7 prompt tok/s and 31.72 drafted tok/s on
the 1697-token prompt, and the resident service answers `/health` on it.

## What this does not touch

`ACHUNK`, the chunk-partial layout, the combine, the softmax, the value loop's arithmetic, the K
cache coordinate itself, `k_leaf_off`, `leaf_of_pos`, the relabeled tree, `attn_chunk_unit_ref`,
`attn_chunk_unit_fast`, `ph_attn` and the row-major body are all exactly as they were. No serving
default moves: `attn_coord()` is still 0, and every kernel the default route launches computes the
bits it computed before.

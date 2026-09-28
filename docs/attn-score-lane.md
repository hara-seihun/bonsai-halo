# One lane owns a key: the score unit without its cross-lane reduction

[The long-context measurement](long-context-attention.md) left the attention score units as the
largest remaining target — 64 slots of arithmetic per key against 47 of cross-lane DPP reduction and
63 of `s_delay_alu`, on a phase that is 84% of attention and 28% of an 8k prefill — and filed the
obvious fix under **not available exactly**:

> A shape where each lane owns a *key* and sums all 256 dimensions in its own register needs no
> cross-lane reduction at all ... It is not bit-identical — a 256-element dot product summed in one
> register is a different tree from eight per-lane terms and a five-level butterfly.

It is bit-identical. The tree is reproducible in one lane, and the whole obstacle was lane
placement, which is a property of the key cache's coordinate rather than of the arithmetic.

## What lane 0 actually computes, and the trap in the middle of it

`warp_sum` combines one eight-dimension `fmaf` chain per lane with `quad_perm(xor 1)`,
`quad_perm(xor 2)`, `row_ror:4`, `row_ror:8` and `permlanex16`, and the kernel stores only lane 0.
Four engineers on this lane, independently and within one hour, worked that sequence out as the
balanced tree **in index order**. All four were wrong in the same place.

`row_ror(N)` makes lane *n* read lane `((n % 16) - N) mod 16`. The direction is the one that makes
`row_shr:1` an inclusive scan — "right" is toward higher lane indices — so lane 0 at `ror:4` reads
lane **12**, not lane 4. Following it through, with `Qk` the sum of the four leaves of quad `k`:

| step | lane 0 holds |
|---|---|
| `xor 1`, `xor 2` | `Q0 = (s0+s1) + (s2+s3)` |
| `row_ror:4` | `Q0 + Q3` — lane 8 meanwhile holds `Q2 + Q1` |
| `row_ror:8` | `(Q0+Q3) + (Q2+Q1)` |
| `permlanex16` | `+ (Q4+Q7) + (Q6+Q5)` from lanes 16-31 |

So the tree is balanced, its leaves are contiguous and its quads are contiguous — but the four quads
of a sixteen-lane row pair **0-3 and 2-1**. `leaf_of_pos` in
[`kernels/attn_leaf.hpp`](../kernels/attn_leaf.hpp) is that relabeling, identity inside a quad and
`{0,3,2,1}` across them. It is an involution, which makes the naive order a free second arm rather
than a second kernel.

**Measured on the device**, `kernels/attn_tree_check` against the engine's own `warp_sum`, 40,000
random `(q, k)` pairs with the real operand distribution:

| in-lane tree | reproduces `warp_sum`'s lane 0 | largest difference |
|---|---:|---:|
| with the `{0,3,2,1}` relabeling | **40000 / 40000** | `0.0` |
| plain index order | 23452 / 40000 | `4.77e-07` |

A 41% mismatch rate is not a rounding curiosity: the score goes through `__expf`, so the residual
hash moves and the change stops being exact. The probe costs a fraction of a second, needs no model
and no lock, and asserting both orders makes it a standing regression test for the convention rather
than a one-off check.

**The rule this earns**, and it belongs beside the addressing rule in
[`docs/activation-scale-axis.md`](activation-scale-axis.md): *replaying* the same cross-lane
operations on independent values is exact by construction and needs no device evidence, because it
never asserts what the operation does. *Reconstructing* what those operations compute requires a
model of them, and the model is the risk. Anything that changes lane placement is the second kind,
and an exactness argument of the second kind is done when the device agrees, not when the algebra
closes.

## The coordinate is the load-bearing half

Lane-per-key on the deployed `[head][pos][256]` cache is a 32-line gather per instruction: it
measures null by construction whatever it does to the instruction census. The key cache therefore
moves to

```
k_leaf_off(pos, d) = (pos/32)*(32*HD) + (d/8)*(32*8) + (pos%32)*8 + (d%8)
```

— the same bytes in the same per-head region, permuted so that the same eight-dimension leaf of 32
consecutive keys is contiguous. One `global_load_b128` then hands a lane a whole leaf of **its own**
key in 512 contiguous bytes: the same instruction count, the same coalescing and the same cache
lines per wave as the deployed loop, with the lane placement the arithmetic wants. The V cache does
not move.

Waves split as (32-key tile, dim half): eight waves cover a 128-key chunk twice over, each wave
owning 16 leaves of 32 keys. The two halves meet through `sc` with one barrier, which is exactly
where `permlanex16` joins the two rows of sixteen lanes, so the join **is** the tree's root.

The second prize is the query. With one lane per key the query is wave-uniform, and naming that
uniformity — `readfirstlane` on the wave index, which the compiler cannot infer from `threadIdx.x` —
moves it onto the scalar unit. Per wave per 128-key chunk, `k_attn_wide<8>`:

| | deployed | first cut | with the scalar query |
|---|---:|---:|---:|
| block instructions | ~3440 work | 1930 | **1444** |
| `v_fmac_f32` | 1024 | 1024 | 1024 |
| `global_load` | 128 | 272 | **16** |
| `s_load` | 0 | 0 | 66 |
| scheduling slots (`s_delay_alu`, `s_waitcnt`) | ~880 | 444 | **68** |
| VGPR / waves per SIMD32 | 92 / 16 | 153 / 9 | **94 / 16** |

The query used to sit in `qr[TT][8]`, 64 registers live across the whole key loop; it is now 66
scalar loads on a pipe that does not compete with the fma. That is why the shape costs no occupancy
despite carrying a five-level tree per row.

**A scalar load of `qrot` is only legal where the kernel does not write it.** Inside the persistent
kernel `ph_attn_pre` writes `qrot` and a grid sync does *not* invalidate the scalar cache, so the
unit is templated on `QINV`: true for the wide launch, which only reads it, and false for `ph_attn`,
which does not. The compiler's own clobber analysis normally refuses the load anyway, but
`__restrict__` on the query pointer is exactly the thing that can talk it into one, and the failure
would be an intermittent stale query on some grids rather than a wrong answer everywhere.

## Both coordinates are in the build, and the pass picks one

`P.k_tiled` selects the arm, `attn_coord()`/`attn_coord_set()` own it as a process setting the way
`gdn_state_format()` owns the state coordinate, and `batch_profile --attn-coord 0,1` walks it as a
case axis shuffled with every other case. Both arms are compiled into the same kernel, so a panel
interleaves them in one process, on one clock, against one warm device — which is the only
instrument this lane has that can read a prefill delta
([`docs/head-rows.md`](head-rows.md) has the cross-process failure that made that rule).

`batch_profile` also stops pinning a 512-token engine context on every non-scan shape: the geometry
check and the engine capacity now follow `--context`, so a case-axis panel can sit at a position
where the phase it measures actually costs something. Attention is the only phase on the position
axis, and every case-axis panel this lane has taken was at the short end of it.

## Results

See [`attn-score-lane/`](../../../data/bonsai2/batch-comparison/attn-score-lane/) for raw samples.

Mode 19, 128-row passes, `--attn-coord 0,1` shuffled with every other case in one process, phase
totals from the tracer. `sequence-core` is the attention phase; every other phase is a control the
change cannot reach.

**Position 1024**, two rounds per arm
([`prefill-1024.json`](../../../data/bonsai2/batch-comparison/attn-score-lane/prefill-1024.json)):

| phase | row-major | leaf-interleaved | ratio |
|---|---:|---:|---:|
| `ffn` | 72.517 | 71.947 | 0.992 |
| `sequence-input-projection` | 43.344 | 43.362 | 1.000 |
| **`sequence-core`** | **26.824** | **23.083** | **0.861** |
| `sequence-output-projection` | 21.252 | 21.131 | 0.994 |
| `gdn-resident-core` | 16.095 | 16.077 | 0.999 |
| `sequence-input-prep` | 3.704 | 3.707 | 1.001 |
| untouched phases (control) | 157.203 | 156.505 | 0.996 |
| device span | 184.999 | 180.557 | **0.976** |

**Position 3072**, one round per arm
([`prefill-3072.json`](../../../data/bonsai2/batch-comparison/attn-score-lane/prefill-3072.json)):

| phase | row-major | leaf-interleaved | ratio |
|---|---:|---:|---:|
| `ffn` | 73.684 | 71.576 | 0.971 |
| **`sequence-core`** | **72.137** | **59.758** | **0.828** |
| `sequence-input-projection` | 43.879 | 42.704 | 0.973 |
| `sequence-output-projection` | 21.904 | 21.506 | 0.982 |
| `gdn-resident-core` | 16.367 | 15.877 | 0.970 |
| untouched phases (control) | 159.796 | 155.497 | 0.973 |
| device span | 232.894 | 216.216 | 0.928 |

**Attention is 14-15% cheaper, normalised by the untouched phases in the same pass** (-13.5% at
1024 against a 0.4% control drift, -14.9% at 3072 against 2.7%), and the device span is **-2.0% and
-4.6%** normalised at the two positions. The change is worth what the phase is worth, and the phase
grows with position: at the 384-token document every other panel in this repository uses, it is
worth almost nothing.

**Bit-identical, in the arms that matter and against the control in the same process.** Residual
FNV-64 `13477932824292311186` in both arms at position 1024 and `2523245940775455799` in both arms
at 3072.

### The census over-promises, and the reason is bytes

The score block went from ~3440 work slots to 1444 — 2.4x — and the phase moved 14%. The units are
not purely issue-bound at this width. A 128-row pass at position 3072 issues one score unit per
(row group, query head, key chunk), and the six query heads of a KV group each request the same key
chunk: about 3.4 GB of K and V requests per pass over sixteen attention layers, which is ~14 ms at
this box's 242 GB/s even if every byte is served at the roof. That is most of what is left. The
instruction census can only spend the part above it, and it did.

This also bounds what any further work on this loop can buy, and says where to look instead: the
remaining term is *requested* bytes, and the lever on it is the query-head axis — one unit serving
the six query heads of a KV group reads the chunk once instead of six times.

## And then it lost, on a build where it costs six wave slots

Re-measured on current main, same panel, same process, same shape
([`prefill-1024-merged.json`](../../../data/bonsai2/batch-comparison/attn-score-lane/prefill-1024-merged.json)):

| phase | row-major | leaf-interleaved | ratio |
|---|---:|---:|---:|
| `ffn` | 74.389 | 74.338 | 0.999 |
| `sequence-input-projection` | 42.212 | 42.104 | 0.997 |
| **`sequence-core`** | **23.545** | **25.822** | **1.097** |
| untouched phases (control) | 154.489 | 154.311 | 0.999 |
| device span | 178.144 | 180.211 | 1.013 |

**+9.7% on the phase, against -13.5% for the same source two hours earlier.** Both numbers are
real and the difference is not noise — the control is 0.1% in this panel. Two things moved
underneath it, and together they are the whole reversal:

1. **The baseline got faster.** `sequence-core` at this shape was 26.824 ms and is now 23.545, a
   12% improvement from the rescheduled reduction that landed in between
   ([`attn-score-schedule.md`](attn-score-schedule.md)). The slots this change removes are the
   slots that body had already learned to hide.
2. **The arm lost six wave slots.** `k_attn_wide<8>` is 92 VGPR / 16 waves on main and **137 / 10**
   with the leaf body compiled in beside it. The register allocator holds the union of two full
   bodies across a runtime branch, and on the earlier tree — one lighter body, no folded head axis —
   the same two arms fitted in 94 / 16. Nothing about the leaf loop changed; what changed is what
   it shares a kernel with.

So the measured cost of the *delivery mechanism* is larger than the measured win of the *change*,
and the change ships selectable and unselected. **This is not a negative about lane-per-key.** The
census is still 2.4x, the arithmetic is still exact, and the one panel that ran it at full occupancy
measured -13.5% and -14.9%. It is a negative about putting a second full body behind a runtime
branch inside a kernel that is already at its register ceiling.

## Where it stands, and the one thing between it and the default

The first panel was taken on a build whose `k_attn_wide<8>` was **94 VGPR / 16 waves per SIMD32**,
against 92 / 16 for the arm it replaces — the score loop's `qr[TT][8]`, 64 registers live across the
whole key loop, is gone, and the query costs scalar registers instead.

On current main it is **137 / 10**. Nothing about the body changed: main's unit now carries the
rescheduled body and the narrow row-group instantiations, and putting a second full body behind a
runtime branch *inside* the unit makes the allocator hold the union of both. Six wave slots is more
than this change wins, which is why `attn_coord()` defaults to **0** and the leaf coordinate ships
selectable rather than selected.

> **Done, and it found a second thing.** [One kernel, one score body](attn-body-dispatch.md)
> moved the dispatch to the launcher: this arm is back to **94 VGPR / 16 waves** and measures
> **-6.1%** at position 1024 and **-4.0%** at 3072 with the fused delivery reproducing the +9.7%
> below at +9.5%. The row-major arm it shares the launcher with gained more than it did, **-18.0%
> and -21.7%**, because it had been paying for this body too — and so had `k_forward_rows`, which
> had been running the deployed route on grid 60 instead of 100 since `04898ea`. The register
> union was never only `k_attn_wide`'s problem.

**The fix is small and named: move the arm dispatch up to the launcher.** `launch_attn_wide` picks
`k_attn_wide` or a `k_attn_wide_leaf` instantiation on `attn_coord()`, each kernel carries one body,
and both go back to their own register counts. That is one dispatch in `kernels/attn_batch.hip` plus
one kernel template — held by another engineer this afternoon, which is why it is written down here
instead of done. The in-process A/B survives it unchanged, because the axis is still a process
setting the tools walk between re-prefills.

## What this does not touch

The drafter's own `DF_HD` layers keep the row-major cache their kernel writes and reads: `KT`
defaults to `(HDh == HD)`, so `kernels/halo_draft.hip` is unchanged and the MTP head — which runs
the target geometry over the shared `qrot` — goes through `ph_attn` with `QINV=false` like every
other persistent-kernel caller. The V cache, `ACHUNK`, the chunk-partial layout, the combine's
numerical map, the softmax and the value loop are all untouched.

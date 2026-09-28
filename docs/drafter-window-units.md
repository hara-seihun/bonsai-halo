# The drafter's window decides its unit list, not just its mask

The DFlash2 drafter attends over a 2048-key sliding window. Until this iteration `attn_window` was
only a *mask*: `ph_attn` dealt one score unit per key chunk from position 0, and each unit read its
keys and values, computed every dot product, and multiplied the whole chunk by zero. At an 11k
context that is 88 chunks a head read to keep 17.

Making the window decide the unit list as well is worth **-51% of the drafter and +12.2% of drafted
generation at 11248 tokens**, and it is exact — the chunks it drops contribute `exp(-inf - M) = 0`
to the fold. The target model is untouched by construction: it has no window, so its floor is chunk
0 and its schedule and its bits are what they were.

[`docs/drafted-verify-route.md`](drafted-verify-route.md) found the waste and measured the fix, and
could not land it: the revision that added the control arm faulted the device on both arms. Rebuilt
on today's master the same design runs clean, with both arms in one binary. Raw:
[`batch-comparison/drafter-window-units/`](../../../data/bonsai2/batch-comparison/drafter-window-units/).

## What moved

One expression, in the two places that have to agree on it:

```
__device__ __host__ int attn_chunk0(int pos, int window) {
    return window > 0 && pos >= window ? (pos - window + 1) / ACHUNK : 0;
}
```

`ph_attn` starts each sequence's chunk range at the floor of its group's **first** row (rows are in
position order, so that is the lowest floor in the group) and deals `HU * (last_chunk + 1 - floor)`
units instead of `HU * (last_chunk + 1)`. `prep_chunk_r`'s `PREP_ATTN_COMBINE` fold starts at each
**row's own** floor, which is at or above the group's. `FwdParams::attn_window_deal` selects the
schedule, `PrepR::comb_window` carries the window to the fold, and `Engine::set_attn_deal` /
`--attn-deal 0|1` walk both arms inside one process. Nothing else in the engine sets a window, so
`win` is 0 everywhere else and both loops compile to the code they had.

## Why it is exact

A chunk every key of which the window excludes has `lo >= vis` in the score body, so its max loop
and its value loop run zero times and its partial is written `m = -inf, l = 0, acc = 0`. Folding
such a partial is the identity: `fmaxf(M, -inf)` is `M`, `exp(-inf - M)` is `0`, `fmaf(0, l, den)`
is `den` and `fmaf(0, acc, v)` is `v`. So the producer stops writing exactly the partials the fold
stops reading, and every surviving term keeps its value, its order and its rounding.

`kernels/attn_window_check` (`make kernels/attn_window_check`, 0.14 s, no GPU) checks that against
the engine's own expressions rather than a restatement of them: 12,900 chunk/row combinations below
the floor are empty and 6,000 at or above are not, the group floor is never above a row's floor over
40,000 positions x four group widths, and 200,000 random folds are bit-identical with and without
their empty chunks. Breaking the floor by one chunk makes it fail.

**The device cannot carry this claim, and that is worth knowing before anyone designs another
drafter acceptance.** `HALO_DF_DUMP` on one arm, twice, at 7371 tokens: the drafter's block hidden
state differs from itself by up to 7.8e-3, `tmpo` by 1e-3, `logits` by 2.4e-6. Four of the drafter's
matvecs run a K-split (`fc` 5, `akp` 5, `o` 2, `down` 2) and `ph_matvec`'s drain is
`if (KS > 1) atomicAdd(out, v)`, so their summation order is the workgroup schedule and its float
state is not reproducible between two runs of one binary. The arms are indistinguishable from
repeats of one arm — over three dumps the *cross-arm* pair is the closest of the three:

| buffer | same arm, run 1 vs run 2 | arm 1 vs arm 0 | arm 1 (2nd) vs arm 0 |
|---|---:|---:|---:|
| `x` (block hidden) | 16339 differ, max 7.8e-3 | 16781, 1.6e-2 | **4306, 7.8e-3** |
| `tmpo` | 15389, 9.8e-4 | 16171, 2.0e-3 | **4206, 2.0e-3** |
| `logits` | 813987, 2.4e-6 | 826280, 3.8e-6 | **294181, 1.9e-6** |

`draft.bin` (the seven drafted tokens), `topi`, `th`, `hcap`, `xq` and both KV caches are identical
in every pair. **`drafted-verify-route.md` used the accepted count as the identity witness for the
drafter's arithmetic; it is not one.** Same binary, same arm, same prompt, adjacent cells of one
process: 9 steps / 63 drafted / 14 accepted, then 10 / 70 / 13. It is a workload-dependent
performance number that moves with the schedule the atomics saw. The greedy digest is the identity
witness for the *product* — a lossless drafter cannot move the output stream whatever it proposes —
and it is identical in all 16 cells of every panel below.

## What it is worth

`--bench --dflash --raw -n 32`, `--context 16384`, Q4 drafter, `--pin-clock`, both arms interleaved
in one process (the two long cells are one arm per process — a 55-second bounded panel does not hold
two 29-second prefills). Chunks are per (row, head, layer) and the window keeps 17 of them forever.

| prompt | chunks dealt: window / deployed | draft ms/step | verify ms/step | drafted tok/s |
|---:|---|---:|---:|---:|
| 11 | 1 / 1 | 6.641, 6.679 vs **6.668, 6.739** | 42.35-42.72 vs 42.67-43.15 | 43.2-43.5 vs 42.8-43.2 |
| 1663 | 13 / 13 | 8.159, 8.159 vs **8.151, 8.117** | 44.96-45.06 vs 44.74-44.82 | 50.10-50.19 vs 50.33-50.44 |
| 7371 | 17 / 58 | 8.419, 8.476 vs **13.394, 13.465** | 55.93-56.20 vs 55.83-56.14 | 49.47-49.73 vs 45.97-46.22 |
| 11248 | 17 / 88 | 8.755 vs **17.901** | 64.51 vs 64.28 | 27.30 vs 24.33 |

- **-37.1% of the drafter at 7371 tokens and -51.1% at 11248**; a drafted step goes 69.42 to 64.51 ms
  and 82.18 to 73.26 ms, which is **+7.6% and +12.2% of drafted generation**.
- **A measured null at and below the window**, which is where the service runs today: 1663 tokens is
  +0.3% and an 11-token prompt is -0.7%, both inside the spread of their own arms. The change buys
  nothing until a session passes 2048 tokens, and then it grows.
- **The verify pass does not move** — +0.13% at 7371 and +0.35% at 11248, in the same processes as
  the drafter's -37% and -51%. Neither does prompt ingestion: 442.5 vs 440.5 tok/s at 7371, 385.9 vs
  383.1 at 11248. The drafter's own prompt ingest computes K and V and never calls `ph_attn`.
- Greedy digest identical in every cell: `1851446955479935831` (11), `4573156307594057558` (1663),
  `8203958597022946936` (7371), `1147699682977340351` (11248). Drafted and accepted counts are
  identical in both arms at 1663, 7371 (84/19, 70/24) and 11248 (112/15).

One line fits every cell of both arms: **draft ms = 6.57 + 0.129 x chunks**. The deployed schedule's
chunk count is `pos/128 + 1` and grows with the context; the window's is `min(that, 17)` and stops.
That is the whole result — the drafter's attention was linear in the context of a model whose own
attention span is fixed.

### The target model's codegen did not move

`ph_attn` is inlined into `k_forward_rows`, whose register count sets the cooperative grid of every
wide pass, so "no bits move" is not enough on its own. `tools/kernel_resources.py kernels/halo_rows.hip
k_forward_rows`, 49 instantiations, canonical against this build: **every instantiation keeps its
occupancy**, and the two deployed shapes are identical in every column —
`k_forward_rows<8,0,0,0,false,1>` 9 waves / 149 VGPR / 107 SGPR / 7204 B LDS and
`k_forward_rows<1,0,0,0,false,1>` 12 waves / 99 VGPR. Four TT=2 and TT=4 instantiations move by one
or two registers inside their wave class and none crosses a granule. Tables are beside the raw.

## What this leaves

1. **The drafter's prompt ingest is the same question on the other side, and it is priced too small
   to take.** A drafted prefill hands the drafter every context row in `RMAX`-row chunks — 1406
   launches at 11248 tokens, each re-streaming `fc` and the five layers' K and V projections — and
   positions only move forward, so once the prompt's end is known every row below `end - 2048` is
   dead the moment it is written. The skip is legal and the engine-side change is small. **The prize
   is not.** The same prompt, same context, same wrapper, with the drafter and without it: 29.15 s
   and 29.36 s against **27.94 s**, so the drafter's *entire* prompt ingest is about 1.2 s of a 29 s
   prefill — 4%, cross-panel, on a box whose panels drift 10%. Skipping 82% of it is a ceiling of
   ~3.4% of an 11k prefill that cannot be separated from drift without an in-process arm, and the
   arm costs more than the win. Someone should take it only as a rider on a change that already
   holds `Engine::prefill`.
2. **The verify pass is now 88% of a drafted step at 11k** (64.5 of 73.3 ms) and it grows with the
   context too — 42.7 ms at 11 tokens, 64.5 at 11248. That is the target's own attention and it has
   no window; [`docs/long-context-serve.md`](long-context-serve.md) holds that ground.
3. **The K-split drain makes the drafter unreproducible**, which costs every future drafter
   experiment its cheapest control. A deterministic drain (a tree reduction over the K parts, or
   `--draft-grid 1` for a check run) would make `HALO_DF_DUMP` an exact instrument again. The same
   atomics are in the target's `ph_matvec`, where `KS > 1` is used by the deployed route.

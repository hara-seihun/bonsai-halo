# A phase opens in lockstep, and that is what a short phase loses

A drafted verify pass runs the same `mv_rows_t` body in seven matvec phases and they read between
113 and 208 GB/s. [The served step's map](drafted-serve-step.md) fitted that spread to the number of
units a phase deals and closed the axis at both ends: retiling to more units is +31%, coarsening to
fewer is +28 to +48%, `KS = 1` is a null. `bench/coop_cost` reproduces the spread with the engine's
own `grid_sync`, `next_unit` and unit body over a cold buffer — **141.9 GB/s over one grid round,
153.9 over three, 184.3 over eleven, 204.2 over seventy-eight** — so it is not the engine's
scaffolding and not its allocations.

This iteration asked what a phase loses at its start. Three answers that sounded right are nulls,
the fourth is **-9.6% of the deployed verify pass and +7.2% drafted generation**, bit-identical, and
it is a per-workgroup *delay*.

Raw samples, the compiled listings and every arm:
[`batch-comparison/phase-ramp-prefetch/`](../../../data/bonsai2/batch-comparison/phase-ramp-prefetch/README.md).

## The mechanism, and the three arms that are not it

`grid_sync` releases every workgroup at the same instant. A phase therefore opens with all 100 of
them at the same offset inside their own units, asking for their first weight block together, and
they de-phase only as far as dynamic dealing lets them drift. A long phase has seventy-eight rounds
of drift; a 320-unit phase on a 100-workgroup grid has three.

| arm, `--tt 8 --pw 5 --grid 100`, GB/s | 100 units | 320 | 512 | 1088 | 7760 |
|---|---:|---:|---:|---:|---:|
| 0, the deployed schedule | 142.0 | 161.1 | 172.5 | 190.5 | 210.9 |
| 1, warm the next phase's first unit before the barrier | 145.0 | 161.1 | 170.8 | 188.7 | 211.4 |
| 2, deal the next unit before the drain and warm it there | 143.9 | 159.1 | 166.4 | 182.3 | 208.0 |
| 8, LDS-scoped unit barriers so a request can survive them | 144.3 | 161.7 | 171.4 | 189.7 | 210.9 |
| 10, arms 2 and 8 together | 144.4 | 160.6 | 169.0 | 183.6 | 210.1 |
| 4, rotate which block range a wave takes by `blockIdx.x` | — | 156.7 / 156.0 | 165.7 / 166.6 | 177.6 / 180.0 | 199.7 / 199.9 |

**Prefetching is not the answer and the unit timeline says why.** `--trace-units` times every unit of
workgroup 0 from its own clock. A 320-unit phase reads `24.04 20.00 20.00 | 8.00` microseconds — the
first unit of a phase costs 20% more than the later ones, and the phase boundary itself costs 8.0 us
— but that first-unit penalty is spread over all forty of its weight blocks, not concentrated in the
first one. Warming one block of forty recovers one fortieth of it, which is what arms 1, 2 and 10
measure.

**The addresses are not colliding either.** Unit bases sit `nb * 896` apart, so the hundred
concurrent streams share their low address bits, and arm 4 breaks that for free by rotating which
block range each wave walks. It is a null at every width and slightly negative at three of five. It
is the **arrival times** that collide, not the addresses.

**A delay breaks it.** `--stagger N` spins workgroup `blockIdx.x % 16` for `N * (blockIdx.x % 16)`
sleep steps at the top of a phase. Palindrome ordered, 320 units, one process, one clock:

| stagger | 0 | 16 | 64 | 16 | 0 |
|---|---:|---:|---:|---:|---:|
| GB/s | 159.0 | 171.4 | **176.8** | 173.3 | 160.7 |

**+8.0% and +7.8% at sixteen, +10.6% at sixty-four**, and the two control cells agree to 1.1%. The
delay is not lost time: dynamic dealing hands a late workgroup fewer units, so a stagger buys the
memory system a smooth demand curve and gives back only the tail.

## What it does to the engine

`ph_matvec` calls `phase_stagger<TT>(total)` before its unit loop. It stays out of any phase that
does not fill the grid twice, because a phase with fewer units than that has no dealing left to
absorb it, and its magnitude rides on the row count: the spread that pays is about one unit wide,
and a unit is eight times cheaper at one row than at eight.

Control and arm are two binaries one `-D` apart, both `HALO_ROWS_TRIM=1`, alternating inside one
lock hold, ten drafter prompts at 24 tokens, `--pin-clock`:

| stagger | verify ms/step | against control | drafted tok/s | plain decode tok/s |
|---|---:|---:|---:|---:|
| 0 (control) | 41.97 | — | 74.24 | 34.06, 34.26 |
| 16, unscaled | 41.42 | -1.3% | — | — |
| 64, unscaled | 38.03 | **-8.1%**, 16 of 16 cases | — | 33.74, 33.77 (**-1.0%**) |
| 192, unscaled | 39.9 | -3.6% (gives 4.8% back against 64) | — | — |
| **64, scaled by rows (ships)** | **37.94** | **-9.6%, 6 of 6 cases** | **79.58 (+7.2%)** | 34.27, 33.45 (null) |

**The greedy digest is identical in every sample of every arm** — `2622622355779879709`,
`12286788372055975269`, `11325551124557423412`, `9165832992999912140`, `11460967368739632265`,
`7652655035396185821`, `2758390290099379723`, `12016378388678097392`, one per prompt, four cells
each. Prompt ingestion is a null in the same panels (346.5 against 347.7 tok/s), which is what the
code says it should be: the wide route reaches `ph_matvec` in no `PART`, so the change cannot touch
mode 19 or 20.

**The unscaled arm's 1.0% on plain decode is the reason the magnitude rides on `TT`.** Sixty-four
steps is 61k clocks against a 20 us eight-row unit and against a 2.5 us one-row unit; scaled, both
routes get a spread of about one unit and the single-row route goes back to its own number.

## What this closes

- **The short-phase deficit is not a cold pipeline.** Four ways of getting bytes in flight earlier -
  across the grid barrier, across the drain, with the barrier's `vmcnt` drain removed, and with the
  addresses de-phased - are nulls or small losses at every width from one grid round to seventy-eight.
- **`__syncthreads()` on this target is `s_waitcnt vmcnt(0) lgkmcnt(0); s_barrier`.** It is a
  seq_cst fence over every address space, so no vector memory request in any of these kernels
  survives a workgroup barrier, and a `ph_matvec` unit boundary carries three of them. The
  LDS-scoped form — `__builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local")` around
  `__builtin_amdgcn_s_barrier()` — compiles to `s_waitcnt lgkmcnt(0); s_barrier` and is all the unit
  deal and the drain need, since every global write they make is published by the `grid_sync` that
  ends the phase. `next_unit` now uses it (`-DHALO_UNIT_BARRIER_LDS=0` restores the predecessor).
  **On its own it is worth nothing** — arm 8 is a null at every width, because the body has no
  request outstanding when it reaches the boundary — and it is in the tree because it is the
  precondition for anything that wants one to be.

## What is next, in the order I would spend a turn on it

1. **The stagger is a magnitude and a shape, and only the magnitude has been swept.** Sixteen
   workgroup classes (`blockIdx.x % 16`) with a linear ramp is the first thing that worked. A
   ramp over all 100, a random permutation, or a stagger proportional to a phase's measured rounds
   are each one constant away and none has been tried.
2. **The probe says +10.6% and the engine gives -9.6%, which is close enough to ask where the rest
   is.** The engine's matvec phases are preceded by prep and attention phases whose units differ in
   cost, so its workgroups arrive at a barrier already partly de-phased; the remaining gap is the
   part of the convoy the barrier re-forms.
3. **Nobody has staggered a phase that is not a matvec.** `ph_prep`, `ph_attn` and `ph_gdn` deal
   units through the same barrier and the same release. Prep phases have fewer units than the grid
   and would only pay the delay, but the attention phases at long context are large enough to
   absorb one.
4. **The first unit of a phase costs 20% more than the later ones and no arm here recovered it.**
   That is 8 us per instance in the trace, ~257 matvec instances in a verify pass. If it is the
   whole unit rather than its first block, the thing that covers it has to be forty blocks wide.

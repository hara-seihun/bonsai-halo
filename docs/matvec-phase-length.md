# A matvec phase is too short to reach its own bandwidth, and neither of the two named causes is why

[`docs/single-stream.md`](single-stream.md) leaves 3.3 ms of a 29.7 ms token in matvec time above
what those phases' bytes cost at 242 GB/s, and names two candidates: **blocks per wave per unit**,
because the achieved bandwidth tracks it, and **round quantisation**, because `G = 60` divides none
of this model's unit counts. This iteration built an instrument that prices the persistent kernel's
scaffolding directly, killed both candidates on the engine, and replaced them with a measured curve.

Held and released: `bench/coop_cost.hip` (new), the `ROWS_GRID_NARROW` constant in
`kernels/halo_rows.hip`. No kernel body changed.

## The instrument

`bench/coop_cost` includes `kernels/phases.hpp` and runs the engine's own `grid_sync`, `next_unit`
and `mv_rows_t` in a cooperative kernel of the same shape, so a number here is the cost of the code
the engine runs rather than of a replica.

    bench/coop_cost --mode 0 --phases 512                       # grid barrier
    bench/coop_cost --mode 1 --phases 128 --pw 5                # unit dealing
    bench/coop_cost --mode 2 --phase-units 320 --pw 3,5,8       # a phase of the engine's length
    bench/coop_cost --mode 3 ...                                # same, two blocks of lookahead

`--phase-units` is the option that matters. Without it the probe deals tens of thousands of units
per phase and measures a machine in a steady state **no phase of this engine ever reaches**; with
it, each phase deals the engine's count and reads a fresh window of the buffer, so it is cold the
way an engine phase is cold.

Two scaffolding costs, at `G = 60`, that nobody had measured:

| what | cost |
|---|---:|
| one grid barrier, no work between barriers | **747.6 ns** |
| one unit dealt to a workgroup (atomic + two `__syncthreads`) | **265.9 ns** |

645 barriers per token is 0.48 ms, 1.6% of the step. That is the floor, not the residue.

## Negative 1: the deeper weight cursor is +44.7% in a long stream and a null in the engine

`mv_rows_t` asks for block `b+1` while it works on block `b`, so a unit of `NBLK` blocks leaves its
first round trip exposed. At three blocks per wave that is a third of them, which is exactly the
shape of the correlation `docs/single-stream.md` reports. Streaming 1 GiB through the real body and
the real unit dealing, the pathology is severe and a second block in flight removes it:

| blocks per wave per unit | one block ahead (ships) | two blocks ahead |
|---|---:|---:|
| 3 | 163.6 GB/s | **236.8** |
| 5 | 235.6 | 235.0 |
| 8 | 233.4 | 220.8 |
| 17 | 228.5 | 228.0 |

So the cursor deepens exactly where the run is too short to cover itself, it is free at five blocks
and it costs at eight. That table justifies a change; the engine refuses it.

Built (`MvCursor<NBLK,Q8>::AHEAD = 2` for `2 <= NBLK <= 4`, HALO tiles only, one extra VGPR on
`k_forward_rows<1,0,0,0,true>` and no occupancy change) and measured on the two phases it reaches,
`mv_ssm_out` and `mv_o`, which are the engine's only `pw = 3` bodies:

| phase | ships | two ahead |
|---|---:|---:|
| `mv_ssm_out` | 1703 us | 1710 us |
| `mv_o` | 573 us | 576 us |
| token | 29961 us | 29784 us |

A null on the phases it targets, and the whole-token difference is the other phases moving. **The
change is not in the tree.**

The probe explains its own over-prediction once it is given the engine's phase length. At 300 units
- five grid rounds, what `mv_ssm_out` and `mv_o` actually deal - the same geometry reads 199.5 GB/s
with one block ahead and 200.1 with two, and 199.5 is within 3% of the engine's own 193.3 and
188.5 GB/s for those phases. **A short phase is faster than a long one at `pw = 3`** (199.5 against
163.6), because sixty workgroups leaving a barrier together put 480 first-block requests in flight
at once and a five-round phase never leaves that burst. Steady state is where a one-block cursor
runs out of memory-level parallelism, and the engine never gets there.

## Negative 2: 64 divides every unit count, and it is worth 0.25%

Every unit count on this path is a multiple of 64 (gate/up 1088, qkv+z 512, q/k/v 448, the three
5120-row matvecs 320, the GDN state 192, the head 7760). At `G = 60` the 320-unit phases spend
six rounds doing 5.33 rounds of work and the ceiling model in `docs/single-stream.md` scores that
92.8% against 99.7% at 64. Nobody had run 64; the grids tried were 48 and 40, which divide nothing.

Four paired in-process panels (`HALO_BENCH_GRID=60,64`, tokens alternating arms inside one process):

| panel | paired median | tokens favouring 64 |
|---|---:|---:|
| 1 | +0.17% | 19 of 30 |
| 2 | +0.37% | 10 of 15 |
| 3 | +0.21% | 18 of 30 |
| 4 | +0.24% | 19 of 30 |

66 of 105 tokens and every panel median favour 64, so the sign is solid and **the size is not**:
+0.25% where the quantisation ceiling predicts +7.7%. 80 and 100 lose 1.43 and 0.97% on the same
instrument, reproducing the -1.6% that document reports at 100.

`ROWS_GRID_NARROW` is now 64, which is worth taking because it is free and bit-exact - grid 60 and
grid 64 produce identical logit dumps, sha256 `a1ba88f9...` - but **the model that motivated it is
dead**. Round quantisation is a correlation, and at the one grid where it should have paid 7.7% it
paid a thirtieth of that.

## What the residue actually is: phase length

Same geometry (`pw = 5`), same code, same cold windows, varying only how many units a phase deals:

| units per phase | grid rounds | GB/s |
|---:|---:|---:|
| 60 | 1 | 175.0 |
| 120 | 2 | 184.3 |
| 300 | 5 | 214.8 |
| 600 | 10 | 223.4 |
| 1200 | 20 | 226.1 |
| 3000 | 50 | 228.5 |
| 12000 | 200 | 233.0 |
| 29940 | 499 | 235.6 |

A phase's achieved bandwidth is a function of its **length**, and the engine's matvec phases live in
the steep part of that curve: 5 rounds for the three 5120-row matvecs, 8.5 for qkv+z, 18 for
gate/up. Their measured rates - 188 to 220 GB/s - are what this curve predicts for their lengths,
and the 3.3 ms residue is the integral of the gap between that curve and its own asymptote.

**That prices the remaining prize at about 2.7 ms of a 29.9 ms token, 9%,** and it says the lever is
the phase structure, not the matvec body: ramp after the barrier and drain at the end, paid roughly
320 times per token.

The curve is close to affine in rounds, which turns it into one number. Fitting the 20-round and
200-round cells gives **9.20 us per grid round plus 6.2 us fixed per phase**, and the fit predicts
52.2 us at five rounds against 50.0 measured. Of that fixed 6.2 us, 0.75 is the barrier this
document measures and the rest is the stream getting up to rate and draining again. A token runs
257 matvec phase calls - 64 gate/up, 64 down, 48 qkv+z, 48 ssm_out, 16 q/k/v, 16 o, one head - so
the fixed term alone is **1.6 ms of a 29.9 ms token**, and the rest of the residue is the first
rounds of each phase running below rate.

That is why the count of phases matters more than anything inside one. Nothing in the matvec body
is worth 1.6 ms; the body is already within 5% of the machine once a phase is long enough to reach
rate.

Two ways to make a phase longer, both already partly explored:

- **More units at the same bytes.** At gate/up's 38.7 MB the probe reads 231.1 GB/s at `pw = 5` with
  1080 units, 234.5 at `pw = 1` with 5400 and 237.6 at `pw = 2` with 2700. `single_map 3` is exactly
  that retile (`KS = 5` takes the 40-block matvecs to `PART = 8`, one block per wave) and it is the
  one single-stream arm that ever gained, at **+1.6%**, which is what this table predicts for it.
  The probe now says where that gain comes from: not the operand map, the round count. It is a
  reassociation, so it stays an explicit alternative and needs its own quality evidence - but the
  next engineer should price the reassociation against +1.5 to +2.8%, not against a map theory.
- **Prefetch across the boundary.** `docs/single-stream.md` experiment 1 took **844 us off the
  matvecs** of a token by reading the next weight stream from idle workgroups and gave 522 us back
  in the preps it padded. That is this ramp, measured from the other side, and the 844 us is the
  upper half of what the curve says is there. The open question is not whether the ramp exists but
  whether it can be paid down from somewhere that is not on the critical path.

## Raw

`batch-comparison/matvec-phase-length/` holds the probe output for every table here, the two engine
panels for the cursor arm and the four grid panels. Reproduce any row with the command in its
section; the probe needs no model and runs in seconds under `tools/run-batch-compare --exec`.

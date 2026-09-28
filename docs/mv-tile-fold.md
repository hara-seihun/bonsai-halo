# Sixteen drains a block is a correlate, and the unit count is the thing: the tile fold, measured and rejected

[`docs/mv-scalar-waits.md`](mv-scalar-waits.md) counted the eight-row matvec's stalls and named the
axis: a `TT = 8` weight block carries **sixteen** `s_waitcnt lgkmcnt(0)` against a `TT = 1` block's
one, and both stream the same 896 bytes. It bought one wait per row pair from the SGPR file and got
4%; four rows a wait needs 128 SGPRs against a 107-SGPR file and loses 12%. **The other side of that
ratio was never tried.** A wait is paid per activation batch; the weight bytes between two waits are
free to grow, and they grow out of the VGPR file, which this body was not using: `mv_rows_t<5, 8,
false>` compiled alone is 61 VGPR where `k_forward_rows<8>` allocates 141 for its other phases.

**The fold**: one wave carries `G` output tiles instead of one and dots each activation batch against
all `G` before letting it go. Drains per weight byte fall by `G` with no new scalar pressure, and the
same 40 kB activation window feeds `G` tiles per read. It is bit-exact by construction — a wave keeps
its own K blocks in the same order, the same int32 chains, the same `acc - xsum`, the same
`fmaf(.., wscale * xs, y)` per (tile, row), and a tile's eight-wave LDS reduction is untouched — and
measured that way: **every arm of every bench panel returns the same output digest
`07701590705348488800`**, and the engine arm reproduces the control's greedy continuation.

**It loses, and what it loses to is not the registers.** The drafted verify pass the service runs goes
from 44.4–48.0 ms to 54.4–55.5 ms per step, **+19%**, and the phase table says the whole of that is
the unit count the fold halves. The arm is not in the runtime; it is
`rejected::mv_rows_fold` in [`bench/mvsched.hip`](../bench/mvsched.hip) beside the two arms
[`mv-scalar-waits`](mv-scalar-waits.md) rejected, with the engine-side integration kept as source in
[`batch-comparison/mv-tile-fold/`](../../../data/bonsai2/batch-comparison/mv-tile-fold/README.md).

## What a drain is worth once you can buy it: 3%, and it is negative where the body is fastest

`bench/mvsched` runs the real body over a real 37.2 MB HALO image at a chosen co-residency, arms in
palindrome order, one `--pin-clock` lock hold per panel. Both passes shown, GB/s of distinct weight
bytes at `TT = 8`:

| LDS per workgroup | 43000 B | 32700 B | 26000 B | 21800 B |
|---|---:|---:|---:|---:|
| co-resident workgroups per WGP (bench calibration) | 3 | 4 | 5 | 6 |
| `deployed` (`mv_rows_deployed`) | 135.5 / 134.4 | 138.2 / 139.8 | 130.3 / 131.1 | 158.2 / 164.3 |
| **`batch-rb2`, what the engine runs** | 149.5 / 146.7 | 150.4 / 151.3 | 135.7 / 135.7 | **181.5 / 180.1** |
| `fold-g2` (8 drains a block) | 148.4 / 152.8 | 151.0 / 155.6 | 137.2 / 143.1 | 166.6 / 169.4 |
| `fold-g4` (4 drains a block) | 147.3 / 147.8 | 145.4 / 145.0 | 145.2 / 144.0 | 143.8 / 143.4 |
| `vec-ar2` (activations on `vmcnt`) | 157.0 / 158.3 | 156.4 / 159.9 | 144.6 / 144.5 | 159.1 / 163.1 |
| `TT = 1 deployed`, the in-panel control | 223.0 | 221.8 | 224.1 | 220.0 |

**Halving the drains per weight byte is worth +1.6% to +3.3%, and at the cell where the body is
fastest it is worth −7%.** Quartering them (`fold-g4`, 208 VGPR) is flat at 143–148 across every
co-residency: it buys exactly as much as it gives back. A second panel taken an hour earlier under a
different clock read `fold-g2` at +5.5% in the 32700 B cell, so the honest band on the mechanism is
about ±5% and panel-dependent, against a 1.9x gap between `TT = 1` and `TT = 8` that it was supposed
to explain. **The sixteen-against-one drain count is a correlate of the row count, not the cause of
the rate.** What the same panel prices at 13% is deleting the trit expansion outright, and
`docs/mv-peel-rows.md` already spent that.

**The other thing in that table is larger than anything in this document.** The body's rate is
**not monotonic in co-residency**: 150.4 at four workgroups a WGP, 135.7 at five, 181.5 at six.
Five is a dip, and an odd block count is the one that splits 3/2 across a WGP's two CUs.
[`docs/rows-grid-occupancy.md`](rows-grid-occupancy.md)'s loader table puts `k_forward_rows<8>` at
141 VGPR in the **four**-block step today, and its register ladder says 120 VGPR still lands on five
and only ≤ 96 reaches six. So a fifteen-register cut in the attention unit lands on the dip and a
forty-five-register one is worth about **+21% of every matvec phase in the pass**. Nobody should
spend registers to buy waves in this kernel without the target cell measured first. (The two lane
documents disagree about whether a WGP's LDS budget is 64 or 128 kB, so the workgroup labels above
are `bench/mvsched`'s own calibration; the ordering and the dip are what the panel measures, and
`tools/rows_occ_probe` is what settles the labels.)

## In the engine it is the unit count, and that is the transferable result

`HALO_MV_FOLD=2` puts the fold on every ternary `ph_matvec` at eight rows. Two binaries from one
tree, one constant apart, both `HALO_ROWS_TRIM=1`, arms alternating `t0 t1 t1 t0` inside one lock
hold, ten prompts, `--pin-clock`:

| per drafted step | control | fold `G = 2` |
|---|---:|---:|
| verify pass | 44.4 – 48.0 ms | 54.4 – 55.5 ms (**+19%**) |
| drafted tokens/s, case 5 | 100.5 / 107.2 | — |
| greedy digest | reproduced in 15 of 15 runs | reproduces the control's |

Traced, one prompt, `HALO_PROFILE=1`, per launch, with each phase's unit count before and after:

| phase | units | control | fold | | units after |
|---|---:|---:|---:|---:|---:|
| `mv_lm_head` | 7760 | 1329.1 us | 1395.2 | **+5.0%** | 3880 |
| `mv_gate_up` | 1088 | 235.2 | 260.9 | +10.9% | 544 |
| `mv_qkv_z` | 512 | 122.9 | 138.4 | +12.6% | 256 |
| `mv_qkv` | 448 | 110.6 | 130.2 | +17.7% | 224 |
| `mv_down` | 320 | 142.6 | 183.3 | +28.5% | 160 |
| `mv_o` | 320 | 60.2 | 86.4 | +43.5% | 160 |
| `mv_ssm_out` | 320 | 60.4 | 89.2 | **+47.7%** | 160 |
| `gdn` | — | 44.8 | 45.9 | +2.5% | unchanged |
| `attn` | — | 36.2 | 35.0 | −3.3% | unchanged |
| whole pass | | 42.49 ms | 50.42 | +18.7% | |

**The loss is ordered by unit count and by nothing else**, and the two phases whose units do not move
are flat. The fold costs `k_forward_rows<8>` 141 → 194 VGPR, a quarter of its co-residency, and that
is worth about 0.6 ms of the 7.9: the recurrence and the attention unit did not care. The other
7.3 ms is seven phases each dealing half as many units over the same grid.

**So the deployed unit count is a local optimum in both directions, measured on the engine.**
[`docs/drafted-serve-step.md`](drafted-serve-step.md) built the arm that multiplies units by 5 to 17
and lost 31% to atomics and unit-boundary work; this one divides them by two with **no atomics, no
numerical change and no boundary added** and loses 19%. Below about 500 units a phase is starved:
`mv_ssm_out` deals 320 units, and taking that to 160 costs 48% of it while taking `mv_lm_head` from
7760 to 3880 costs 5%. Any future change that makes a unit bigger — a wider tile group, more K per
unit, a fused pair of projections — has to buy back forty percent inside the unit before it breaks
even on the 160-tile projections, and roughly five on the head.

## Three things not to rediscover

- **`G = 4` is not the answer to `G = 2` being small.** 208 VGPR, seven waves by the compiler's
  ladder, and flat at 143–148 GB/s across every co-residency in the panel. The fold's return falls
  off exactly as fast as its register cost rises.
- **The engine's drafted decode is not run-to-run token-stable in every arm.** The control
  reproduced its case-0 digest in 15 of 15 runs; the fold arm produced a different continuation in 2
  of about 7, with a different accepted count, on the prompt with the lowest acceptance. Both arms
  are exact on paper, and both routes run `KSV = KSFF = 2`, whose two partial sums reach a
  **preloaded residual** through `atomicAdd`: `(x + a) + b` and `(x + b) + a` are different floats,
  and the order is a scheduling accident. That makes a residual-hash control on this route a strong
  signal, not a proof, and it is worth one instrument: nobody has measured how often the deployed
  eight-row route's own low bits move between two identical runs.
- **A panel taken while a peer is compiling is a different machine.** Two of my panels disagree by
  5% on the same arms and the same build; `268a7387`'s power work has the mechanism — the shader
  clock fell from 2689 to 2190 MHz with host cores busy, which is 12% of a pass.

## Reproducing

```sh
make bench/mvsched
tools/run-batch-compare --pin-clock --exec /bin/sh -c 'for w in 6 8 10 12; do bench/mvsched --waves $w; done'
```

The engine arm is `ph_matvec_fold` in
[`batch-comparison/mv-tile-fold/ph_matvec_fold.hpp.txt`](../../../data/bonsai2/batch-comparison/mv-tile-fold/README.md):
paste it into `kernels/phases.hpp` beside `ph_matvec`, dispatch it from `ph_matvec_auto` under
`HALO_MV_FOLD`, and build the two arms with `tools/split-build` and `-DHALO_ROWS_TRIM=1`.

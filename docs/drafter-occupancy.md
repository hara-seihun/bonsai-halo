# The drafter is not short of waves, and the grid calculator was vetoing the point that proves it

`bonsai-halo.service` serves one stream with the DFlash2 drafter, and a drafted step is one
eight-row target pass plus one `k_dflash` launch ([the map](serving-decode.md)). The target pass has
been measured from every side this lane knows. The drafter carries one number nobody had followed:
`k_dflash` holds **239 VGPRs at six waves per SIMD32**, which is three 256-thread workgroups per WGP
and a cooperative grid of 60, while [the dot4/WMMA crossover](matvec-crossover.md) measured the
eight-row target pass at 62.8 / 55.6 / 52.6 / 49.8 ms across 60 / 72 / 84 / 100 workgroups — a 21%
range on identical instructions, bought entirely with resident waves.

That transfer does not happen. **The drafter's time is flat from 40 workgroups upward.** The
attempt is worth writing down because of what it cost to ask the question honestly: the engine's
own grid calculator refused to launch the arm that answers it.

## The curve

`--bench --dflash`, Q4 drafter, 26-token prompt, 48 generated tokens, one process, arms given as a
palindrome so every point is measured early and late. `verify` is the target pass, which nothing
here touches, and is the in-panel control.

| workgroups | draft ms (early, late) | mean | verify ms | drafted tok/s |
|---|---|---:|---:|---:|
| 20 | 7.685, 7.737 | 7.711 | 45.494, 45.576 | 128.9, 128.6 |
| 40 | 6.483, 6.461 | **6.472** | 45.514, 45.438 | 131.9, 132.1 |
| 60 (deployed) | 6.447, 6.494 | **6.471** | 45.442, 45.846 | 132.1, 131.0 |
| 80 (register-capped body) | 6.482, 6.484 | **6.483** | 45.436, 45.572 | 132.1, 131.7 |

Halving the grid to 20 costs 19%. The third workgroup per WGP is worth **0.02%** and the fourth
is worth **-0.2%**. All eight runs digest to `9384357187189555707` and accept 41 of 49 drafts, so
the arms are the same tokens, the same acceptance and the same output; only the schedule moved.

The drafter moves 1.23 GB per step ([0.93 GB of Q4 tiles plus 278 MB of the target's ternary LM
head](drafter-q4.md)) in 6.47 ms: **190 GB/s of this box's 242 GB/s roof**. The remaining 21% is
not waves. `62e20787` reached the same conclusion from the other side on the Q4 landing — a dot4
drafter body at nine waves per SIMD32 lost to the WMMA body at six — and this is the direct form of
it, same body, same arithmetic, grid as the only variable.

## What the fourth workgroup cost to obtain

`amdgpu_waves_per_eu(8)` fits the Q4 body in 192 registers, which is four workgroups per WGP. The
source does not change: same phases, same operand order, same arithmetic, same weight coordinate.

| body | VGPR | waves/SIMD32 | VGPR spill | scratch | grid |
|---|---:|---:|---:|---:|---:|
| `k_dflash<8, q4>` deployed | 239 | 6 | 0 | 0 | 60 |
| `k_dflash<8, q4>` `waves_per_eu(8)` | 192 | 8 | 3 | 32 B/lane | 80 |
| `k_dflash<8, q4>` `waves_per_eu(9)` | 168 | 9 | 22 | — | 80 |
| `k_dflash<8, q8>` deployed | 247 | 5 | 0 | 0 | 40 |
| `k_dflash<8, q8>` `waves_per_eu(8)` | 192 | 8 | 28 | — | 80 |

Three spilled registers for a third more resident waves is a cheap arm, and in a separate panel
against the deployed body at the same grid it cost **1.5%** of draft time (6.548 against 6.450 ms
per step) — which the extra workgroups then handed back, 6.548 at 60 to 6.460 at 80. The capped
body is not in the tree: it is one attribute on one template and the table above is what it buys.

## The calculator refused the arm, and the reason generalises

The first panel ran the capped body at **60 workgroups**, not 80, and reported it. `coop_grid` asks
the occupancy API for workgroups per WGP and then removes one when the VGPR file would be exactly
full — a repair fitted to a 253-VGPR kernel where the API reported six waves of 256 registers.

The premise was wrong. **gfx11 wave32 allocates VGPRs in granules of 24 out of 1536 per SIMD32**,
so 253 registers is 264 and five waves, not 256 and six; the API's over-report came from
granularity, not from a full file. Every occupancy number the compiler prints follows that rule
(135 → 144 → 10 waves, 111 → 120 → 12, 239 → 240 → 6, 247 → 264 → 5, 192 → 192 → 8), and
`b0d22d99` measured the same granule from the other end of the register table. The margin repaired
one case by luck and vetoed every kernel that fits the file **exactly**, and those are real: 192 × 8
and 96 × 16 are both 1536.

`coop_grid` now rounds the way the hardware rounds. `HALO_COOP_GRID_PRINT=1` prints every
cooperative grid in a process with the previous value beside it, and across a full drafted run —
the persistent kernel at four row counts, both drafter coordinates, the sequence parts — **one
number in the engine changes**:

```
coop_grid: regs  95 api  6 granule  96 waves 16 blocks  8 -> 120 (was 120)
coop_grid: regs 107 api  6 granule 120 waves 12 blocks  6 -> 120 (was 120)
coop_grid: regs 111 api  6 granule 120 waves 12 blocks  6 -> 120 (was 120)
coop_grid: regs 135 api  5 granule 144 waves 10 blocks  5 -> 100 (was 100)
coop_grid: regs 192 api  4 granule 192 waves  8 blocks  4 ->  80 (was  60)
coop_grid: regs 239 api  3 granule 240 waves  6 blocks  3 ->  60 (was  60)
coop_grid: regs 247 api  2 granule 264 waves  5 blocks  2 ->  40 (was  40)
```

The API is already conservative — it never exceeds the granule bound in this engine — so the new
form only removes an incorrect veto, and the 192-register launch it unblocks ran correctly at 80
workgroups with identical digests. The deployed grids (100 for the persistent kernel, 60 and 40 for
the two drafter coordinates) are the values they were.

**This matters to the next engineer more than it mattered to me.** 192 registers is the natural
target for anything trying to reach four workgroups per WGP, and until now asking for it produced a
kernel that spilled to fit a budget it was then denied the benefit of — silently, with the grid
looking exactly like the one before.

## Two things this leaves

- **`k_ffn_slice` still takes one grid for every width, and the widest instantiation sets it.**
  `launch_ffn_slice` takes `std::min` over `<1>`, `<2>`, `<4>`, `<8>`, `<16>`, `<32>`, and those
  compile to 78 / 75 / 79 / 73 / 219 / 238 VGPRs — sixteen waves per SIMD32 for every narrow width
  and six for the two wide ones. So a pass of eight rows or fewer runs at **60 workgroups when its
  own occupancy allows 160**. That is precisely the trap `matvec-crossover` found in the persistent
  launcher and fixed there, still live in this one, and it is now a per-instantiation grid table
  away. Unmeasured: the narrow widths serve the deployed (non-A4) route, so price it on
  `--ffn wide-deployed` and on a deployed-route generation step before believing the register table.
- **The drafter's 21% of roof is bytes and latency, not occupancy.** The remaining levers named in
  [`drafter-q4.md`](drafter-q4.md) — the 278 MB of full-vocabulary LM head for a top-16, and a
  three-bit coordinate — are both about traffic, and this result says nothing stands between them
  and the measurement.

Raw samples: [`batch-comparison/drafter-occupancy/`](../../../data/bonsai2/batch-comparison/drafter-occupancy/README.md).

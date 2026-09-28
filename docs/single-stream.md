# Where a single-token step goes, and two things that do not recover it

Single-stream greedy decode is the engine's oldest published number and the one the interactive
server actually runs. It had no owner in the optimization lane and no exact win: the palette, grid
and state-split maps all lost, and only the reassociated K-split gained 1.6%. This is a current
phase map of that step, a model that predicts each phase's bandwidth, and two exact changes built
against that model and measured down to nothing.

## The map

`HALO_PROFILE=1 tools/run-batch-compare --engine --bench -n 48`, canonical build at `9ad103e`,
idle box, 23-token prompt. Device timestamps at each of the ~645 barriers of the persistent kernel,
grouped by phase. 29.694 ms/token, 33.68 tok/s.

| phase | us | share | calls | bytes/call | GB/s |
|---|---:|---:|---:|---:|---:|
| `mv_gate_up` | 11413 | 38.4% | 64 | 39.0 MB | 218.7 |
| `mv_down` | 6367 | 21.4% | 64 | 19.5 MB | 196.0 |
| `mv_qkv_z` | 3992 | 13.4% | 48 | 18.35 MB | 220.6 |
| `mv_ssm_out` | 1708 | 5.8% | 48 | 6.88 MB | 193.3 |
| `gdn` | 1257 | 4.2% | 48 | 6.29 MB state | 240.2 |
| `mv_qkv` | 1202 | 4.0% | 16 | 16.05 MB | 213.6 |
| `mv_lm_head` | 1188 | 4.0% | 1 | 278 MB | 234.0 |
| `prep_norm` | 860 | 2.9% | 129 | — | — |
| `mv_o` | 584 | 2.0% | 16 | 6.88 MB | 188.5 |
| `attn` | 426 | 1.4% | 16 | KV only | — |
| `prep_silu` 214, `gdn_pre` 192, `prep_gdn` 161, `prep_attn` 65, `attn_pre` 47, `argmax` 13 | | | | | |

5.90 GB at the 242 GB/s `bench/bw` measures is 24.4 ms, so 5.3 ms of the token is not weight
streaming at the roof. It splits two ways: **1.98 ms in phases that move no weight bytes at all**
(the preps, attention, argmax) and **3.3 ms of matvec time above what those phases' bytes cost at
242 GB/s**.

Raw: [`batch-comparison/single-stream/`](../../../data/bonsai2/batch-comparison/single-stream/).

## The model that looked right, and what it actually predicts

A phase deals `U` units to `G` workgroups. `G` is 60 here: `coop_grid` grants three workgroups per
WGP over 20 WGPs, set by the 217-VGPR eight-row instantiation, because the deployed path launches
every row count on the minimum grid over the four of them. Every unit count in this model is a
multiple of 64 — gate/up 1088, qkv+z 512, q/k/v 448, the three 5120-row matvecs 320, GDN state 192,
the head 7760 — and 60 divides none of them. Round the work up to whole rounds and each phase has a
ceiling:

| phase | U | rounds at G=60 | ceiling | measured |
|---|---:|---:|---:|---:|
| `mv_gate_up` | 1088 | 19 | 95.4% | 90.4% |
| `mv_qkv_z` | 512 | 9 | 94.8% | 91.2% |
| `mv_qkv` | 448 | 8 | 93.3% | 88.3% |
| `mv_down`, `mv_ssm_out`, `mv_o` | 320 | 6 | 88.9% | 81.0 / 79.9 / 77.9% |
| `mv_lm_head` | 7760 | 130 | 99.5% | 96.7% |

Every phase lands four to eleven points under its own ceiling, and the ceiling tracks the
measurement across a 10-point range. Weighted by bytes, G = 60 is the **worst** value in its
neighbourhood: 92.8%, against 98.1% at 40 and 97.3% at 48.

That is a real correlation and it is not a causal model. Both experiments below were chosen by it.

## Experiment 1: filling the phase-boundary idle with the next stream (negative)

During a 5120-element prep, five units exist and 55 of 60 workgroups never enter the loop; they sit
at the device barrier while the weight stream is idle. The idea is to spend that wait reading the
head of the next weight stream, which brings it into the device cache for whoever reads it next. It
changes no value anywhere: only loads are issued and every result is discarded, so it is
bit-identical by construction rather than by measurement.

Each workgroup walks exactly the tiles the next phase will hand it — unit `blockIdx.x`, then
`+ gridDim.x` — so nothing it moves is wasted. Three variants, all measured with arms alternating
token by token inside one process (`HALO_BENCH_*`, below):

| variant | what it does | paired result |
|---|---|---:|
| poll | 4 KB steps, checks a completion counter between `__syncthreads` pairs, stops when the last straggler finishes | **+0.25%** (17/30, 24/30, 20/30 over three panels) |
| burst | 8 steps issued back to back into distinct registers, one wait for the lot, no poll | −0.19% to −0.49% |
| per-wave poll | 512 B steps, each wave stops on its own, no workgroup barrier | +0.16% (13/20) |

The poll variant's phase map says the mechanism does what it claims and then pays for it. Arms two
tokens apart in one process:

| | `mv_gate_up` | `mv_down` | `mv_qkv_z` | `mv_qkv` | `prep_norm` | `prep_silu` | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| off | 11484 | 6398 | 4021 | 1200 | 869 | 234 | 29876 |
| on | 11212 | 6147 | 3805 | 1115 | 1227 | 398 | 29595 |
| delta | **-272** | **-251** | **-216** | **-85** | **+358** | **+164** | -281 |

844 us comes off the matvecs and 522 us goes back into the preps the prefetch was padding out. The
burst variant removes the padding and the gain with it: issuing 1.8 MB and waiting once means the
workgroup waits for its own traffic, which is the same trade with fewer instructions.

**The negative, with its scope.** For the persistent single-token kernel on gfx1151, converting
phase-boundary idle into next-stream reads is worth 0 to +0.3%, against 1.98 ms (6.7%) of nominally
idle phase time. It is not in the tree. Two readings are worth carrying:

- Whatever the barrier costs, it is not a stalled memory system waiting for cold misses. A matvec
  phase reaches its steady state within its first microsecond on its own, because 60 workgroups
  issue their first loads together and each wave is already one block deep.
- Filling a *matvec* tail is structurally different from filling a prep, and neither pays. In a
  tail, the stragglers are on the critical path and the prefetcher is competing with them for the
  same bandwidth; the bytes are not wasted but the phase does not end sooner. This is the same
  result PLAN.md recorded on 2026-09-18 for a one-block prefetch across the barrier, reached from
  the other end.

## Experiment 2: moving the workgroup count (small positive, and it refutes the model)

Below the occupancy limit the workgroup count is free. A unit is a whole 32-row tile — and, where
`KS > 1`, one K part of it — computed by one workgroup, whose eight waves split K and reduce
through LDS in wave order. Which workgroup runs a unit changes no operand, no order and no output
bit. `FwdParams::grid_rows` caps it (`HALO_GRID_ROWS`, `Engine::set_grid_rows`).

Paired, alternating token by token in one process:

| grid | median ms | tok/s | paired against 60 |
|---:|---:|---:|---|
| 60 (occupancy) | 29.695 | 33.68 | — |
| 48 | **29.497** | **33.90** | **+0.43%**, 26 of 30 tokens |
| 48 (second panel, loaded box) | 32.046 | 31.21 | +0.44%, 15 of 20 tokens |
| 40 | 32.428 | 30.84 | −0.81%, 2 of 20 tokens |

**Bit identity, measured rather than argued.** The installed canonical build
(`577b5eab8c54f4e61911a3475fbf459d8484c359e33ca5d2cd6fad6469792605`) dumped all 248,320
full-vocabulary logits of a fixed prompt at grids 60, 48 and 40 — a five-token prefill in eight-row
passes plus one decode step, so both the eight-row and the single-token kernels, and the K-split
matvecs that combine two parts per output — and the three files compare byte for byte equal. Single
stream on that build is 33.47 tok/s at the default grid, against 33.66 to 33.71 across the panels
that preceded the change, and the resident server came back on it.

41 of 50 paired tokens favour 48 across two panels, so the effect is real. Its **size** is what
matters: moving from 60 to 48 improves the round-quantisation ceiling by 4.5 points and moves
throughput by 0.43%, a tenth of it. Going to 40 improves the ceiling by 5.3 points and **loses**
0.81%, because 320 waves is four per SIMD32 and the latency hiding goes with it.

**So the round model is not the cause.** Dynamic units already smooth it: `next_unit` hands out the
next unit the moment a workgroup is free, so a phase does not cost `ceil(U/G)` lock-step rounds, it
costs its bytes plus about one unit of straggling. The default is unchanged; 0.43% is smaller than
the drift between two panels on this box, and 48 is untested at eight rows, where the same grid
serves a different kernel with different unit counts.

## What the deficit correlates with instead, and the attack it suggests

Take the two phases with identical unit shape — `mv_gate_up` and `mv_lm_head` are both `NB = 40`,
`KS = 1`, same 896-byte tiles, same five blocks per wave — and they differ only in how many units
they have and therefore how long the phase runs: 234.0 GB/s for the head against 218.7 for gate/up.
Now sort every phase by blocks per wave per unit, `pw`:

| pw | phases | GB/s |
|---:|---|---:|
| 5 | `mv_lm_head`, `mv_qkv_z`, `mv_gate_up` | 234, 221, 219 |
| 8-9 | `mv_down` (K-split, uneven 64/72 parts, residual add, atomics) | 196 |
| 3 | `mv_ssm_out`, `mv_o` | 193, 188 |

`mv_rows_t` prefetches one block ahead. A wave with `pw = 3` spends the first of three iterations
with nothing in flight and never reaches steady state; `pw = 5` amortises it better, and the phases
that run longest do best. The deficit is per-unit cold start, not per-phase tail.

**That reading has since been tested directly and it is wrong, and so is the round model above.**
[`docs/matvec-phase-length.md`](matvec-phase-length.md) prices the scaffolding with the engine's own
`grid_sync`, `next_unit` and `mv_rows_t`: a barrier is 748 ns and a unit deal is 266 ns, a second
block in flight is worth +44.7% at `pw = 3` **in a long stream** and a measured null in this engine
(`mv_ssm_out` 1703 to 1710 us, `mv_o` 573 to 576), and `G = 64`, the one grid in this neighbourhood
that divides every unit count here, is worth +0.25% where the ceiling column above predicts +7.7%.
It is now the default for rows 1..4 anyway, because it is free and bit-exact.

What survives is the variable underneath both correlations. Holding the geometry fixed and varying
only how many units a phase deals, achieved bandwidth runs 175.0 GB/s at one grid round, 214.8 at
five, 226.1 at twenty and 235.6 in steady state — and the phases in the table above sit at 5, 8.5
and 18 rounds, which is exactly where their measured rates fall. **The residue is the ramp and
drain of a short phase, paid about 320 times a token, and it is worth roughly 2.7 ms of 29.9.** The
two-tile unit proposed here would make a phase *shorter*, not longer, so it is not the attack; the
attacks are more units per phase (`single_map 3`'s retile, the one arm that ever gained, +1.6%,
which that document predicts to within 0.1 points) and paying the ramp down from somewhere that is
not on the critical path (experiment 1 above, which already took 844 us off the matvecs).

## The instrument

Single-stream work had no in-process A/B. Two processes cannot resolve a few percent here — a peer
measured 33.19 ms/token on this path with five engineers building, against 29.69 idle — so the
engine now alternates arms token by token inside one `--bench`:

```sh
HALO_BENCH_GRID=60,48 tools/run-batch-compare --engine --bench -n 62          # paired medians
HALO_BENCH_GRID=60,48 HALO_PROFILE=1 tools/run-batch-compare --engine --bench -n 62   # + a phase map per arm
```

It prints each arm's median, minimum and mean, the paired difference and how many tokens favour
each arm, so a 0.4% effect is readable. `HALO_PROFILE=1` alone still prints the single map above.
Any future single-token arm should be wired to `Engine::set_*` the same way rather than to a second
process.

# The served verify pass evaluated one gate thirty-two times, and its state phase is a third smaller

The route `bonsai-halo.service` runs every drafted verify pass on is the persistent eight-row
kernel, and its recurrent state phase was paying a `log1pf` for a number the pass had already
computed. Two changes, both bit-identical by construction, take **29 to 38% off `gdn`** and
**2.3 to 2.9% off a mode-0 eight-row decode step**, with the deployed instantiation's register
report unchanged cell for cell.

Raw samples, per-arm executables and the failed and superseded panels are in
[`batch-comparison/gdn-sliced-gates/`](../../../data/bonsai2/batch-comparison/gdn-sliced-gates/README.md).

## Landed a day late, and re-measured on the tree it landed on

`b4382f13` built, measured and wrote all of this on `362264c` and was stopped before committing it.
The source sat uncommitted in its checkout until `d65a8bb5` saved the diff
(`stranded-b4382f13-source.patch` beside the raw samples), applied it to `7d67de7` - it applied
without a conflict, since nothing that landed in between touches the state phase - and re-took
every acceptance below on that tree. The sections after this one are the original author's and
their numbers are from `362264c`.

| check on `7d67de7` + this change | result |
|---|---|
| `HALO_GDN_COMPARE=1`, mode 20, 128 rows | token records **0 / 1,323,008** differ, recurrent output 0 / 786,432, state 0 / 786,432, conv ring 0 / 30,720 |
| `make -C tools/direct-commit check` | every legal schedule equal to the replay path bit for bit, the illegal one diverges |
| `tools/object_resources.py`, deployed verify kernel `k_forward_rows<8, 0, 0, 0, false, 12>` | 120 VGPR, 8 spilled, 36 B scratch, 7204 B LDS in both objects; every TT = 8 instantiation identical |
| verify shape, `batch_profile --modes 0 --decode-streams 1 --decode-tokens 8`, prompt 1024, A-B-A-B | device span **39.28 -> 38.79 ms (-1.25%)**, eight traced passes an arm |
| served step, `--bench --dflash` Q4, 48 tokens, A-B x 4 | verify **40.04 -> 39.43 ms per step (-1.5%)**, drafted generation **68.43 -> 69.36 tok/s (+1.4%)**, 136 accepted tokens in each arm; `gdn` 3.5 -> 2.2 ms (-39%) and 2.3 -> 1.7 ms (-26%) at the two replay depths the cells landed on |
| single-stream plain decode, `--bench`, A-B x 3 | **29.24 -> 28.79 ms per token (-1.5%)**, every pair faster: the one-row route reads the same block cache and stops paying the gate per unit too |
| eight one-token streams (mode 4, committing), A-B-A-B | 50.07 -> 49.78 ms, a null, as the `CM == 0` width gate intends |

The product's greedy digest came out `14250041415274144751` or `10917796635708187097` in both arms,
each in both: the deployed route's `KS = 2` `atomicAdd` drains decide which one a process gets, as
the instrument section below explains, so neither digest can tell the arms apart and the device
comparison above is the identity witness. Raw:
[`gdn-sliced-gates/rebase/`](../../../data/bonsai2/batch-comparison/gdn-sliced-gates/rebase/).

## What was there

`ph_gdn` deals one unit per (sequence, head, row group) and walks that sequence's tokens, and every
unit called `gdn_load<SPLIT>` with `PREGATE = false`, which derives the token's decay gate from the
raw alpha projection:

```c
t.g    = gdn_decay(tok[2 * CONV_CH + h], L.ssm_dt[h], L.ssm_a[h]);   // softplus through log1pf
t.beta = sigmoid(tok[2 * CONV_CH + HV + h]);
```

The source beside `gdn_decay` already says what that costs - "about 180 scalar-float instructions;
that is why the only place this is allowed to run is once per (row, head)" - and the resident route
obeys it, evaluating the gates once in `resident_conv` and reading them back. **The sliced route
never got that treatment.** At the deployed `SPLIT = 4` a head's 128 state rows are four units, each
of eight waves, and every wave of every unit walks every token: **32 evaluations of one (row, head)
gate per layer**, against the one the quantity needs.

`ph_gdn_pre` was already the right place. It runs immediately before `ph_gdn`, and one thread of it
already writes each of the row's `2 * HV` gate slots:

```c
if (g == 0 && tid < 2 * HV) tok[2 * CONV_CH + tid] = P.ab[row * 2 * HV + tid] * P.ninv[row];
```

## What ships

**The block token cache holds evaluated gates.** `ph_gdn_pre` writes `gdn_decay(...)` and
`sigmoid(...)` where it used to write the raw projections, and every reader of that cache -
`ph_gdn`, the resident kernel's replay prefix, the column arm's replay - takes the `PREGATE` path.
One fp32 number, produced by the same expression from the same inputs, stored instead of recomputed
thirty-two times. `HALO_GDN_BLK_GATE=0` restores the raw form in every producer and reader at once.

**Those writes are dealt one row per unit.** `ph_gdn_pre`'s unit is (sequence, channel group) and a
layer deals 40 groups, so leaving the gate work inside group 0's serial row walk put eight `log1pf`
chains end to end in one workgroup while thirty-nine had none - and a phase ends when its slowest
unit does. It cost **+21% of `gdn_pre`** (629 -> 757 us a pass) in the first build. Keyed on `g`
instead, one row per group, `gdn_pre` is flat (634 -> 641 us in the same panel construction).

**A one-sequence eight-row pass deals its state in 96 units, not 192.** `SPLIT = 2` gives a wave
eight state rows instead of four, which halves how many times a token's gate, `k` and `q` are
re-read, and puts the phase's 96 units inside one round of the 100-workgroup grid instead of 1.92
rounds taking two. This is [`gdn-unit-width`](gdn-unit-width.md)'s interior optimum, which
`gdn_resident_split_for` already applies on the resident route for one sequence; the sliced route
was hard-wired to 4 with a `static_assert` that refused 2.

## The arms, and why the width is gated on `CM == 0`

`tools/batch_profile --modes 0 --decode-streams 1 --decode-tokens 8`, which is the drafted verify
shape with a fixed workload, A-B-A in one lock hold under `--pin-clock`, both traces of each round:

| arm | 2785 MHz panel | against base |
|---|---:|---:|
| base (`HALO_GDN_BLK_GATE=0`, `SPLIT 4`) | 38.93 / 39.11 ms | - |
| gate only | 38.18 | **-2.2%** |
| gate + `SPLIT 2` | 38.01 | **-2.6%** |

Three further A-B-A panels put the shipped arm at 37.95 against 38.84, 37.88 against 38.89 and
37.96 against 39.08/39.10; the two base cells of a panel reproduce to 0.05-0.46%.

**The width is not free at more than one sequence, and that is the same shape rule the resident
route found.** Eight streams of one token each - `forward_batch` routes that to mode 4 - measured in
its own A-B-A panel:

| arm | eight-sequence step | against base |
|---|---:|---:|
| base | 50.53 ms | - |
| gate only | 49.79 | **-1.5%** |
| gate + `SPLIT 2` everywhere | 50.93 | **+0.8%** |

A unit re-reads the whole token however few rows it owns, so a wider unit pays less of that per
token; a second sequence already deals enough units to fill the grid, and then all that is left of
the wider unit is its coarser round. **`CM == 0` is the gate that means one sequence in this
engine**: an eight-row pass that does not commit its state is a drafted verify pass or an
`Engine::forward` slice, both solo, while every multi-sequence eight-row step commits
(`forward_batch` routes 5..31 rows to mode 4). With the gate in, the eight-sequence step is a null
against base in its own panel (50.16 against 50.10 and 50.17, inside a 1.3% within-arm spread) and
the TT = 8 register report is the base one, cell for cell.

## Registers, before the panel rather than after

`tools/kernel_resources.py kernels/halo_rows.hip k_forward_rows`, all 48 instantiations, against the
same report on the base build. **The deployed verify instantiation `<8, 0, 0, 0, false, 12>` is 12
waves per SIMD32, 120 VGPR and 8 spilled dwords in both.** Every TT = 8 cell is identical. Three
`OCC = 1` probe shapes move by 1 to 3 VGPR with no wave step, from the gate expression leaving the
token loop.

The ungated width is what the gate is for: at TT = 1 it costs two occupancy steps (12 waves at 100
VGPR to 10 at 132) and the packed-state TT = 8 instantiation spills 21 dwords to 102.

## The phase, and what it is now

`--bench --dflash` with the Q4 drafter, four prompts, three rounds, `HALO_PROFILE=1`, base and
shipped arm in one lock hold. `gdn` is the traced state phase of one verify pass, so cells are
comparable only at equal tokens per step - the pass's replay prefix is whatever the previous step
accepted:

| tokens/step | base `gdn` | shipped `gdn` | | base `gdn_pre` | shipped |
|---:|---:|---:|---:|---:|---:|
| 4.7 | 3221 / 3231 us | 2080 / 2039 us | **-35%** | 634 / 644 | 641 / 644 |
| 4.9 | 3172 | 2008 | **-37%** | 623 | 635 |
| 3.62 | 2408 / 2367 | 1713 / 1726 | **-29%** | 514 / 523 | 526 / 522 |

The state phase was 8.3% of a verify pass and is now 5.2%.

## What it is worth to the product, honestly

The same panel's drafted single-stream generation, paired only where the two arms accepted the same
mean tokens per step (acceptance moves the workload, and the run-to-run variance below moves
acceptance):

- **case 1: 108.46-109.21 -> 109.60-110.82 tok/s**, six pairings, +0.6% to +1.8%
- **case 3: 84.00-84.44 -> 84.43-84.70 tok/s**, six pairings, -0.0% to +0.8%
- median verify pass over all twelve cells an arm: **37.533 -> 37.140 ms, -1.05%**

The controlled step moves 2.3-2.9% and the product moves about 1%. The difference is not a
discrepancy to explain away: a drafted step is a verify pass plus 6.5 ms of drafter that this change
cannot reach, and the product's own measurement carries an acceptance term that the controlled one
does not.

## Acceptance

**Bit-identical by construction.** The gate is the same expression on the same inputs, stored as the
fp32 it already was. The width is a relabelling: every state row keeps its own columns, its own
lane, its own token order and its own `warp_sum`, and nothing is reduced across units, so only which
(workgroup, wave) walks a row moves.

**The device says so.** `HALO_GDN_COMPARE=1` rebuilds a wide pass's first layer through the sliced
route into private buffers and compares it against the resident kernel's, inside one process:

| | base | shipped |
|---|---:|---:|
| token records | 0 / 1,310,720 differ | **0 / 1,323,008 differ** |
| recurrent output | 0 / 786,432 | 0 / 786,432 |
| state | 0 / 786,432 | 0 / 786,432 |
| conv ring | 0 / 30,720 | 0 / 30,720 |

The shipped arm compares **more columns than the base one can**: the diagnostic only compares the
gate columns when both routes hold the same representation, and now the sliced route's stored gates
and the resident kernel's own evaluation are the same form - and every bit of them agrees.

`tools/direct-commit/commit_equiv`, the host mirror of `ph_gdn_pre` and `ph_gdn`, carries the same
change and every legal schedule still agrees with the replay path bit for bit.

## The instrument this iteration had to replace, which is everyone's

**Two runs of one executable on one fixed workload do not agree on the last bits, so a residual
FNV-64 or a greedy digest compared ACROSS PROCESSES is not an identity witness on this route.**

Measured, not inferred. `tools/batch_profile --modes 0 --decode-streams 1 --decode-tokens 8`, the
same binary in cells 1 and 4 of one panel: residual FNV-64 `3844514681862579886` against
`15516161427475040085`. The product does it too - `--bench --dflash` gave greedy digest
`14250041415274144751` in two cells and `10917796635708187097` in two others, with 3.00 and 2.29
accepted tokens per step, all four cells from the same two executables.

The mechanism is in `ph_matvec`'s drain and it is precise:

```c
if (KS > 1) atomicAdd(out, v); else if (seg.add) *out += v; else *out = v;
```

`next_unit` deals units to whichever workgroup asks next, so the two parts of a K-split arrive in
either order. For `add = 0` that is exactly reproducible - the accumulator starts at zero and
`0 + p0 + p1` equals `0 + p1 + p0` - but **`mv_down`, `mv_ssm_out` and `mv_o` all have `add = 1`**,
so the atomics land on the residual that is already there and the pass computes `(x + p0) + p1` or
`(x + p1) + p0`. At `KSV = KSFF = 2` that is 128 accumulations per verify pass whose rounding is
decided by arrival order.

Two consequences for the lane:

- **A published "residual FNV-64 identical in all samples" is a within-process statement.** It still
  witnesses an in-process case axis, which is what those panels were. It does not witness a
  cross-build arm, and every compile-time arm in this engine is a cross-build arm.
- **A deterministic build looks free.** `k_forward_rows`'s own comment records that "coarsening to
  KSV = KSFF = 1 is a null in the same panel". If that holds, one constant buys a run-to-run
  reproducible engine and the cross-build identity instrument this lane has been assuming it had.
  Nobody has measured it as a determinism change; `kernels/halo_rows.hip`'s K-split constants are
  `bbfca1db`'s ground and this is a note for whoever holds them.

## Where the rest of the state phase is

At the shipped arm a verify pass spends 1.9 ms in `gdn` over 48 layers, 40 us a layer, moving
3.1 MB of state in and 3.1 MB out plus about 0.4 MB of token records: **about 155 GB/s against the
242 GB/s `bench/bw` reaches**, where the resident route's equivalent runs at 224 and
[`gdn-state-pack`](gdn-state-pack.md) calls byte-bound and finished. The remaining gap on this route
is not the gate and not the unit width. The two candidates the numbers leave standing are the eight
waves of a unit each fetching the same `k` and `q` fragments of the same token (the LDS stage the
resident route has never needed because its grid covers the cost with other waves), and the fp32
round trip itself, which the packed coordinate already halves on routes that select it and which
this route cannot take today without the register step measured above.

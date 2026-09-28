# What binds the sequence projections: issue, not the fragment stream

[The operand document](sequence-projection-operands.md) closed by naming one lever as the big
remaining one in this kernel:

> The lever that would halve it is the one the FFN already uses and this kernel does not: **give a
> wave two row tiles against one B fragment** ... If the input projection is fragment-bound, that is
> worth far more than the 0.4% instructions bought.

It is built and it **loses**: 7.3% on one instrument, 2.3% on another, and 15.5% at the shape that
prices the bytes hardest. The input projection is not fragment-bound. It is issue-bound, its
marginal instruction converts to time at about one issue cycle, and the premise that sent two
engineers at the fragment stream was an artifact of the instrument that produced it.

The same panel that killed it found the change that ships: the kernel's **slice scheduling barrier
was never fitted**, and dropping it at four token tiles is worth 5% of the output projection and
**+0.42% of full-model prompt throughput**, bit-identical.

Two engineers claimed this experiment two minutes apart on September 21 (`4b5d8fb5` at 13:57,
`06b91f1f` at 13:59) and the board admitted both. Both panels are below, on different modes and
different clock policies, because two instruments agreeing is worth more than either.

## The arm that lost

`project<>` gains a row-tile count `RT`: the wave holds `RT` adjacent weight row tiles and spends
each loaded activation fragment against all of them, so fragment bytes per matrix instruction fall
from 256 to 256/`RT`. Nothing else moves - the same 128-K blocks in the same order, the same int32
block sums, the same two scales drained in the same order into the same FP32 chain - so it is
bit-identical by construction, and measured so.

`SEQ_SCHED_PAIR` pairs `RT = 2` with half the accumulator width, so the wave covers the same four
tile-pairs per block as the selected shape and lands in the same register class. `RT = 4` with one
token tile is the third point on the same line.

| shape | bytes per output tile | slots/128-block | wmma | global_load | VGPR | waves/SIMD32 |
|---|---:|---:|---:|---:|---:|---:|
| `TT=4, RT=1` selected | 4352 | 382 | 32 | 39 | 154 | 9 |
| `TT=2, RT=2` | 2560 | 456 | 32 | 24 | 154 | 9 |
| `TT=1, RT=4` | 2048 | 695 | 32 | 21 | 184 | 8 |

`RT * TT = 4` throughout, so the wave-block count per SIMD32 is the same 65,536 over a 64-layer
128-row pass in all three. The panel therefore measures the marginal price of one block-loop
instruction directly.

### Measured, twice

**Mode 19, `--pin-clock`, one process, `--seq-sched 1,3` interleaved, two rounds, heads off.**
`ffn`, `gdn-resident-core` and `sequence-core` cannot be reached by this change and are the
in-panel controls.

| phase | selected | `RT = 2` | change |
|---|---:|---:|---:|
| `sequence-input-projection` | 41.567, 41.150 | 44.416, 44.246 | **+7.3%** |
| `sequence-output-projection` | 16.831, 16.743 | 17.775, 18.021 | **+6.8%** |
| `ffn` (control) | 83.602, 82.623 | 81.630, 81.759 | -1.6% |
| `gdn-resident-core` (control) | 16.759, 16.595 | 16.688, 16.825 | +0.1% |
| `sequence-core` (control) | 16.429, 16.252 | 16.205, 16.189 | -0.4% |
| device span | 181.182, 179.334 | 182.632, 182.985 | +1.6% |

Residual FNV-1a `7446865760224376151` in all four samples.
Raw: [`sequence-fragment/phase-m19-128.json`](../../../data/bonsai2/batch-comparison/sequence-fragment/phase-m19-128.json).

**Mode 20, unpinned clock, `--seq-sched 1,3,4`, two rounds reshuffled, rows 32 and 128 in one
panel** (`4b5d8fb5`, git `7d06e28`). Mode 20 prepares no weight image, so this is the same kernel
inside a completely different pass:

| 128 rows | `RT=1` | `RT=2` | `RT=4` |
|---|---:|---:|---:|
| `sequence-input-projection` | 38.734 | 40.414 | 44.657 |
| `sequence-output-projection` | 15.937 | 16.285 | 18.722 |
| four controls together | 1.000 | 1.020 | 0.998 |
| **input after control** | | **+2.3%** | **+15.5%** |

| 32 rows | `RT=1` | `RT=2` | `RT=4` |
|---|---:|---:|---:|
| `sequence-input-projection` | 14.147 | 17.201 | 15.851 |
| `sequence-output-projection` | 7.896 | 9.591 | **15.593** |

Residual FNV-1a `16921095978579329146` at 128 rows and `15659291503748132804` at 32, identical in
all three arms.
Raw: [`seq-row-tiles/phase-m20.json`](../../../data/bonsai2/batch-comparison/seq-row-tiles/phase-m20.json).

### The price of a block-loop instruction

Because `RT * TT` is constant, the three cells give the marginal cost directly:

| step | ms | slots | us per slot |
|---|---:|---:|---:|
| `RT1 -> RT2` | +1.680 | +74 | 22.7 |
| `RT2 -> RT4` | +4.243 | +239 | 17.8 |
| `RT1 -> RT4` | +5.923 | +313 | 18.9 |

One issue cycle is 27.1 us at 2420 MHz and 23.0 at 2848. Solving the serial model - 32 cycles per
`v_wmma_i32_16x16x16_iu8`, one per other slot - for the clock that fits each cell gives 2.33, 2.35
and 2.48 GHz. **The model is not approximately right, it is exact to the clock**, and the entire
value of cutting 53% of the fragment bytes shows up as the `RT = 4` cell implying a 6% higher
clock than `RT = 1`.

**32 rows is the cell worth keeping even though nobody wants `RT` there.** `RT` divides the row-tile
grid by `RT`, and at 32 rows the output projection's 320 row tiles become 160 waves on 80 SIMD32s:
+97.5%. The same shape costs 17% at 128 rows. Row tiles are the scarce resource at small batch,
which is the same asymmetry the barrier result below turns on.

### The correction this forces

The operand document put in bold that "the input projection is not issue-bound", from deleting
21.5% of the block and measuring 0.4%. That figure came from a **cross-build** pair at `TT = 8` and
five waves per SIMD32, normalised by a control ratio of 1.0355. This panel is one build, one
process, one clock, arms interleaved, at the nine-wave shape the rule selects. On a box where two
processes minutes apart differ by 8%, a 5% effect normalised by a 3.5% correction is not
resolvable, and the conclusion drawn from it cost two engineers a turn.

**Do not price this kernel from a cross-build pair.** `--seq-sched` puts every arm in one process.

## The change that ships: the slice barrier was never fitted

`project<>` has carried `__builtin_amdgcn_sched_barrier(0)` after every second K16 slice since the
kernel was written, with nothing recorded about why. [The FFN's
lookahead result](ffn-pair-lookahead.md) found the identical construct in `k_proj_opt` fitted at one
of the two shapes it fired on. The same is true here, and the axis is accumulator width:

| width | barrier | no barrier | what the removal costs |
|---|---:|---:|---|
| `TT = 1` | 82 VGPR, 16 waves | 117, 12 | a block holds 8 fragment loads |
| `TT = 2` | 95, 16 | 134, 10 | 16 loads |
| `TT = 4` | 154, 9 | 162, **9** | 32 loads, **and no wave slot** |
| `TT = 8` | 242, 5 | 205, **7** | unmeasured, and it *frees* registers |

Measured in one process, `--seq-sched 1` against `5`, rows 32 and 128, after the in-panel controls:

| shape | row tiles | waves | round 0 | round 1 |
|---|---:|---:|---:|---:|
| 128 rows, input, `TT=4` | 1024 | 2048 | -0.58% | +0.24% |
| 128 rows, output, `TT=4` | 320 | 640 | **-4.22%** | **-6.35%** |
| 32 rows, input, `TT=2` | 1024 | 1024 | -2.64% | +2.70% |
| 32 rows, output, `TT=1` | 320 | 640 | **+32.8%** | **+38.2%** |

A three-round confirmation at 128 rows on the shipped rule, with the box drifting ±12% between
adjacent cases inside one process and the control ratio catching all of it:

| round | input | output | device span | control |
|---|---:|---:|---:|---:|
| 0 | -0.13% | **-3.62%** | -0.36% | +0.31% |
| 1 | +0.63% | **-7.56%** | -0.62% | +11.90% |
| 2 | -0.62% | **-4.77%** | -0.57% | -4.35% |

Round 1 is the one to read: the output projection got 2.9% *faster* in raw time while all four
controls got 11.9% slower.

**Full model**, 384-token document in 128-row passes, mode 0 interleaved as its own control, three
rounds, `--pin-clock`:

| arm | mode 19 prompt tok/s | mode 0 control |
|---|---|---|
| barrier every second slice | 758.8, 758.9, 760.1 | 157.2, 158.0 |
| **fitted rule** | **764.7, 760.8, 762.1** | 158.1, 158.2, 157.6 |

**+0.42%**, and every round of the fitted arm is above every round of the prior arm.
Raw: [`sequence-fragment/full-barrier/`](../../../data/bonsai2/batch-comparison/sequence-fragment/full-barrier/).

So the rule is the FFN's rule seen from the other side: **keep the barrier at every width except
four token tiles.** Four is where it is dropped because four is where it was measured and the only
width where removing it costs no wave slot. Eight is left alone: `--seq-sched 0` selects it and
every published ownership table used it as the control arm, so re-fitting it is a deliberate act,
not a side effect of this one. `--seq-sched 5` puts the old barrier back and `4` removes it
everywhere, both in-process.

### Why the two stages answer differently, and the fit that said so first

`4b5d8fb5` fitted `t = A/clk + B` over four shader-clock ceilings (`db82126f`'s sweep, busy-sample
p50 994/1581/2165/2848 MHz) and split every phase into clock-proportional and clock-independent
time. That fit **predicted this split before it was measured**:

| phase | clock-proportional at 2848 | clock-independent | issue share |
|---|---:|---:|---:|
| `head-projection` | 8.57 ms | 1.11 ms | 88.5% |
| `ffn` | 60.34 | 11.21 | 84.3% |
| `sequence-input-projection` | 30.09 | 5.95 | **83.5%** |
| `gdn-resident-core` | 11.07 | 3.92 | 73.8% |
| `sequence-core` | 8.11 | 4.57 | 63.9% |
| `sequence-output-projection` | 9.72 | 6.58 | **59.6%** |

The input projection is the second most issue-bound phase in the pass and the output projection is
the least. A scheduling change that buys lookahead can only pay where there is clock-independent
time to recover, and the output stage has 40% of its time there against the input stage's 17%.
**The two stages of this kernel are different machines, and a clock sweep says which is which
before a schedule is fitted.** It is the same reason `RT` lost worse on the input stage than on the
output one.

## Where the kernel stands now

The shader clock under a real pass is not 2900 MHz. Three instruments, all in the tree:

- **2420 MHz** busy-weighted over a mode-19 128-row payload with the performance level forced high
  on an idle box; p50 2399, p90 2502, max 2538, ratio 0.827 against 2900. Package power 125 W with
  the gfx and soc throttle residencies flat and the CPU core counter moving, so the GPU is held down
  by shared package power, not by its own thermal limit.
  Raw: [`sequence-fragment/clock-m19-128.json`](../../../data/bonsai2/batch-comparison/sequence-fragment/clock-m19-128.json).
- **2848 MHz** busy-sample p50 at the unconstrained ceiling in `db82126f`'s sweep.
- **2.33 to 2.48 GHz** fitted independently from the three `RT` cells against the serial issue
  model.

So the honest statement about this stage is that its IU8 matrix-pipe floor is **23.5 ms at 2848 and
28.8 ms at 2420**, against 41.4 measured - not the "24 against 38" the operand document carries.
Every budget in `docs/` divides by 2.9 GHz and is optimistic by 2 to 17%.

Per 128-K block of one wave at the selected `TT = 4, W = 2, SHARE`, from the emitted ISA:

| item | slots | share of the 1373-cycle block |
|---|---:|---:|
| `v_wmma_i32_16x16x16_iu8` x 32, at 32 cycles | 1024 cycles | 74.6% |
| weight expansion (32 `v_perm`, 56 shift/mask) | 88 | 6.4% |
| drain (32 `v_cvt_f32_i32`, 32 VOPD mul/fmac) | 64 | 4.7% |
| `s_waitcnt` and `s_delay_alu` | 51 | 3.7% |
| `global_load` | 39 | 2.8% |
| 64-bit address arithmetic | 28 | 2.0% |
| accumulator rezeroing | 11 | 0.8% |
| `ds_bpermute` for the weight scale | 8 | 0.6% |

The matrix instruction is three quarters of the block, so **deleting every other instruction and
every stall would take the input projection from 41.4 ms to 26.9**. That is the whole remaining
prize inside the deployed numerical map, and every slot converts at about one issue cycle.

## What not to try here, with the reason

- **Fragment reuse across row tiles.** This document. Two instruments, three shapes, negative
  everywhere, worst where the row-tile grid is already the scarce resource.
- **A cheaper weight image.** The 2-bit spread word costs 11 slots per 16 weights. Four bits per
  weight costs 10 for twice the bytes, because the byte permute that maps a code to {0, +1, -1} is
  needed either way and only the mask before it changes. Eight bits per weight costs zero slots and
  7.05 GB of sequence weights, which at two token groups per pass is 542 GB/s against a 242 GB/s
  roof. The 2-bit image is the right corner.
- **More occupancy.** 154 VGPR is nine waves per SIMD32 and the next step needs 144. The two stages
  already disagree about the wave target, and the FFN measured a 6.8% *loss* from buying waves with
  accumulator chains.
- **A ternary-specific matrix instruction.** There is not one. `v_dot4_i32_iu8` runs at the same 128
  MAC/cycle/SIMD32 as the IU8 WMMA, and superposing two ternary rows in one operand does not
  separate: a K16 partial sum spans 12 bits and an 8-bit operand cannot carry a shift that large.

**The one large lever left in this kernel is the matrix instruction itself.**
`v_wmma_i32_16x16x16_iu4` is 16 cycles against IU8's 32 and needs four-bit activations. At the
block census above that takes the input projection from 41.4 ms to about 26 and the output
projection from 16.8 to about 10.5: **roughly 21 ms of a 181 ms pass, and 11 ms of a 126 ms
generation step**, which is more than everything else in this kernel put together. It is a
different numerical map - the FFN's A4 route already made that trade and carries mean KL 0.0704
against mode 0 - so it needs its own quality panel and stays an explicit alternative, not a change
to the deployed one.

## The other way to cut the same traffic also loses, and it is the cleaner test

`RT = 2` halves the fragment bytes and pays an extra expansion per (slice, row tile), so this
document's reading was that the expansion bought the loss.
[Staging the workgroup's shared block in LDS](seq-input-bstage.md) cuts the same traffic with **no**
extra expansion, **no** change to the wave count's denominator and **two fewer** issue slots per
block — 4 memory instructions against 18 — and loses **21.6%** of the input projection in a
32-stream generation step, because the barrier it needs takes the instantiation from 95 VGPR and
sixteen waves per SIMD32 to 157 and nine. Both arms now say the same thing from opposite sides:
**the fragment stream is not what this kernel waits on, and a wave slot here is worth about 3% of
the phase.**

## Knobs

- `--seq-sched 3` walks the `RT = 2` pair arm in-process beside the three ownerships, `4` removes
  the slice barrier at every width and `5` restores it at every width. `HALO_SEQUENCE_SCHED` pins
  any of them, `HALO_SEQUENCE_RT` and `HALO_SEQUENCE_SB` cross the two axes with
  `HALO_SEQUENCE_TT` and `HALO_SEQUENCE_W`. All appear in `halo_env`.
- The `RT` arm is **selected by no rule** and nothing reaches it without being asked. It is kept
  reachable only because the negative is the useful half of it; if the next engineer in this kernel
  needs the registers, delete `RT` and cite `5367d45`.

## The fragment stream is now priced in bytes, and it bounds every repair of it

This document killed the fragment-bound premise on issue grounds. [`cache-policy.md`](cache-policy.md)
puts a number on the traffic itself, which is what anyone proposing LDS staging or a deeper row-tile
group should start from. Calibrated, the input projection fetches **25.97 MiB a layer-dispatch
against a floor of 21.88**, so everything re-fetched — activations included — is **at most 4.09
MiB**, and at 32 rows the same stage runs at **1.01x its floor**. Forcing the fragments to miss
(the non-temporal fragment arm, 59.52 MiB) *adds* 33.6 MiB and costs **8%** of the phase, so
removing 4.09 is worth about **1%**. The re-fetch is real, it is small, and it is not what this
kernel waits on. The same measurement finds the FFN's gate/up stage at **2.11-2.28x its floor**,
which is where that argument does have room.

## The third direction, and what it settles

[The token map](seq-operand-coord.md) took the same fragment stream from the fetch side rather than
the reuse side: which token a (tile, lane) position carries is free to relabel, so a wave's `TT`
fragments can arrive in half the requests as `global_load_b128` with no stored byte moving and no
writer changing. It is a **null** (-0.35% normalised, five rounds), and a second map with the
*identical* instruction mix is **+16.25%** because its sixteen lanes straddle eight cache lines at
half use instead of four at full use.

That closes this document's open question. The premise "the input projection is not fragment-bound,
it is issue-bound" survives in one direction and fails in the other: the loop does pay one issue
cycle per added instruction, as measured here, but *removing* sixteen requests and fourteen waits
from it buys nothing at all. What the fetch costs is the cache lines its wave touches, and the
deployed map is already at one lookup per line.

# The pair-code block's scheduling barrier was fitted at four token tiles

`k_proj_opt`'s pair-code arm puts `__builtin_amdgcn_sched_barrier(0)` after every second K16 slice.
It was added for a reason that is written down beside it: at four token tiles the compiler otherwise
hoists all thirty-two B fragment loads of a 128-block above the first matrix instruction and spills
10-12 registers to scratch.

That shape is one of two the arm runs. `shape_for` gives gate/up four token tiles at 128 rows and
gives the down projection **two**, and the barrier is a total scheduling barrier in both. At two
tiles a block holds sixteen fragment loads rather than thirty-two and allocates 145 VGPRs rather
than 252, so there is nothing to spill and the barrier only forbids the lookahead.

Applying it at four token tiles and not at two takes the A4 FFN region of a 128-row pass from
**85.23 ms to 80.27 ms**, bit-identical, against an in-process null control that moved +1.0% the
other way: **6.7% of the stage** once the control is divided out.

## What the compiler does with the two shapes

`hipcc -S`, gfx1151, per 128-K block of one wave, issue slots:

| shape | barrier | slots | VGPRs | waves/SIMD32 |
|---|---|---:|---:|---:|
| gate/up, TT = 4, W = 2 | kept | 573 | 252 | 5 |
| down, TT = 2, W = 4 | removed | 401 → **387** | 145 → **185** | 9 → **8** |
| 32-row arms, TT = 2 | removed | 408 → 389 | 146 → 186 | 9 → 8 |

The trade is explicit: about fourteen issue slots and a wave slot per SIMD32 bought for a block's
worth of load lookahead. The wave slot is the expensive half and it still wins, which says the
two-tile shape was latency-bound rather than occupancy-bound.

The four-tile shape is untouched by construction, so the published 128-row gate/up schedule and its
acceptance stand unchanged.

## Measured

Two builds of `882a563`, the same source apart from this commit. Eight rounds, 128 rows, logits
off, `ffn` region per pass.

The control is **in-process and in the same kernel**: `--ffn-image 1` runs mode 8's dense five-trit
arm, a different `WM` branch of `k_proj_opt` that has never had a barrier and that this change
cannot reach. If the box drifts between the two processes, the dense arm moves with it.

| arm at 128 rows | barrier everywhere | `TT >= 4` only | change |
|---|---:|---:|---:|
| pair code (selected) | 85.23 ms | **80.27 ms** | **−5.8%** |
| dense five-trit (null control) | 109.04 ms | 110.08 ms | +1.0% |

The control moved against the result, so dividing it out gives −6.7% rather than −5.8%.
Per-round samples, sorted:

```
pair  baseline   82.61 84.80 84.89 85.03 85.43 85.91 86.29 90.48
pair  selected   77.39 80.15 80.22 80.25 80.29 81.49 88.11 88.47
dense baseline  105.20 107.85 108.82 108.95 109.14 109.32 115.51 116.33
dense selected  106.07 109.27 109.80 109.84 110.32 110.49 120.01 120.77
```

Six of the eight selected samples sit below every baseline sample. The residual FNV-1a over all
128 x 5120 FP32 outputs is `7446865760224376151` in both builds and both arms, which is the value
`docs/ffn-schedule.md` records for this shape: a scheduling hint moves no bit.

The same pair of builds at `ce4ba22`, before a peer's operand-addressing change landed, measured
85.24 against 79.37 with the null at +0.5%. The result survived that change to the same kernel.

Raw: [`ffn-small-batch/`](../../data/bonsai2/batch-comparison/ffn-small-batch) —
`m882-base.json` and `m882-new.json` are the panels above, `bar-base-128.json` and
`bar-new-128.json` the `ce4ba22` pair.

## What it does not do

At 32 rows both stages of the pair arm are two tiles, and there removing the barrier **loses**:
51.28 ms to 54.20 ms on the FFN region, against a dense-arm null that moved −0.5% in the same pair
of panels. That shape has 1088 gate/up row tiles and only one token group, so it is already holding
as many independent weight streams as the machine can run and the lost wave slot is not repaid.

It costs nothing in the engine, because 32 rows takes the dense five-trit arm; but it is the reason
the barrier is keyed on the token-tile count rather than simply deleted.

Batched generation is untouched for the same reason: a 32-stream step is a 32-row pass and runs the
dense arm, whose code this does not reach. Single-stream decode never calls `run_ffn_batch`.

The evidence for that null is structural plus the dense arm's own numbers, not a matched decode
panel. The dense arm moved −0.5% at 32 rows and +1.0% at 128 between the two builds, and 32-stream
generation on the selected build measured 265.8 and 264.1 aggregate tok/s against 268.7 documented
on an earlier build and panel. **A matched 32-stream decode panel on the two builds was not taken**
— it was cut by the GPU lock — and is the cheapest thing to add.

## Full model

`tools/batch_compare --modes 0,19 --only prefill --prefill-rows 128 --prefill-tokens 256`, four
rounds, mode 0 interleaved in the same process as its control. Round 0 of each process is the clock
ramp and is excluded; the new build's round 0 read 625.0 with its mode 0 control at 141.3, so the
whole process was still ramping.

| build | prompt tok/s, rounds 1–3 | median | mode 0 control |
|---|---|---:|---:|
| barrier everywhere | 703.7, 719.6, 722.2 | 719.6 | 157.9 |
| `TT >= 4` only | 737.2, 739.0, 733.4 | **737.2** | 156.3 |

**+2.4% prompt throughput**, or +3.5% with the control divided out, which moved 1.0% against the
result. That is what a 5 ms saving on a 194 ms pass should look like.

## The negative that led here, and the model it kills

The route to this was an attempt to give the 32-row arm the pair-code image, on the argument that
the five-trit byte costs 300 more issue slots per block than the pair code for 19% fewer weight
bytes, and that the arm is nowhere near the bandwidth roof that would justify the trade.

Four arms in one process at 32 rows, mode 5 as an in-process control, all bit-identical at residual
`1873717996769712938`:

| arm at 32 rows | `ffn` ms |
|---|---:|
| dense five-trit, tile ownership (shipped) | **33.85** |
| pair code, tile ownership | 71.20 |
| pair code, adaptive ownership | 61.79 |
| pair code, wide ownership | 51.06 |

The prediction was 1.34x the other way. **Issue slots do not predict time in this shape.** The arm
with 1.74x the slots and 27% fewer wave slots is 2.1x faster, and the schedule axis moves it by
40% while the instruction count does not change at all. The dense arm runs at about 92% of its own
issue model and the pair arm at 32%; what separates them is how much independent work stands
between a weight load and its use.

So: before cutting instructions out of a 32-row FFN arm, establish that the arm is issue-bound.
The 128-row arm is, and the census transfers there. The 32-row arm is not, and a peer's landed
`--ffn-image` axis reaches the same 33.8 against 50.5 endpoints from the other direction.

## Operation

```sh
make -j8 bonsai-halo tools/batch_profile tools/batch_compare
# The pair arm against the dense arm as an in-process null, eight rounds.
tools/run-batch-compare --profile-tool --modes 19 --rows 128 --heads 0 --traces 1 \
    --rounds 8 --warmup-ms 1500 --context 128 --ffn-image 1,2 \
    --out ../../data/bonsai2/batch-comparison/ffn-small-batch/bar-new-128.json
```

`--ffn-image` is [the A4 image axis](ffn-schedule.md); the barrier itself is a compile-time property
of the token-tile count and is not selectable at runtime. Two builds are therefore two processes,
which is why the null control is a kernel in the same process rather than a second arm.

## The barrier no longer fires at the default pass width, and its premise is gone where it does

Two shape rules moved under this barrier after it was fitted, and a third change removed the reason
for it. All three are visible in the compiled listing; none needs a GPU.

**Where it fires now.** `PAIR_SCHED` is keyed on `TT >= 4`, and no pair-code shape of a default
128-row pass has four token tiles any more:

- gate/up runs the interchanged order (`a4_order = 2`), whose branch carries no barrier at all.
- the down projection takes `TT = 2` at 128 rows, which the `TT >= 4` guard exists to exclude.

At 256 rows (`--pass-rows`, [pass-width.md](pass-width.md)) `shape_for` gives the down stage
`W = 4, TT = 4` and the barrier fires there. So the +2.4% this document measured is real history,
and the code it selects is reachable today only through a 256-row pass or an `a4_order = 0` control
arm. **Re-fit it at 256 rows, not 128.**

**Its premise is gone.** The barrier is there because at four token tiles the compiler otherwise
hoists thirty-two fragment loads above the first matrix instruction and spills 10-12 registers.
[The operand-address split](ffn-operand-address.md) deleted the 64-bit vector addresses that were
that pressure. Censused on `d9fed5d`, the shape that actually fires, `<IU4, PAIR, LANE, SHARE,
TT=4, WV=4, ADD, ORD=0>`:

| barrier | block instrs | waits | VGPR | scratch |
|---|---:|---:|---:|---:|
| `sched_barrier(0)`, deployed | 452 | 47 | 229-232 | **0** |
| `sched_barrier(2)`, VALU may cross | 460 | 45 | 223 | 0 |
| none | **440** | **34** | 229 | **0** |

**There is no spill to prevent any more**, and removing the barrier is twelve instructions and
thirteen waits cheaper than keeping it. The mask that rescued the dense arm's barrier
(`0x20`, [ffn-dense-loads.md](ffn-dense-loads.md)) has no equivalent win here: mask `0x2` is worse
than both other cells. That asymmetry is the arm's own shape - the dense barriers fence a 321-slot
radix-3 peel, while the pair branch fences 80 slots of expansion, so there is much less collateral
to recover and simply dropping the fence beats tuning it.

Not shipped, because the shape is 16% of a 256-row pass and 2.7% of its block slots, which at this
kernel's measured [one-third conversion](ffn-pair-block.md) is under half a percent of a pass -
below what a panel here resolves. The three cells above are the measurement anyone touching
256-row passes needs, so it is a ten-minute job rather than a rediscovery.

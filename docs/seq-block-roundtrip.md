# The sequence projections' memory round trip is the weight stream, and the activation fragment is worth nothing

Five arms have crossed the activation path of `project` — [cache policy](cache-policy.md),
row-tile reuse and [LDS staging](seq-input-bstage.md), [the pair alphabet](seq-pair-operand.md),
[the token map](seq-operand-coord.md), [fragment reuse](sequence-fragment-reuse.md). It is 32 of the
block's 39 memory requests, 335 MB of re-read per layer against 21 MB of weights, and the largest
single thing anyone can see in the loop.

**Its entire price is 1.4% of the input projection at 128 rows and 0.07% of a 32-stream generation
step.** The weight stream — seven requests, 512 bytes and one 16-bit scale per 128-block — is the
whole round trip: **7.5% of the input projection at 128 rows and 7.1% of an entire batched
generation step.**

Nothing ships. The default build's `kernels/sequence_batch.hip` device code is
instruction-for-instruction what it was, 260,152 instructions byte-identical across the two
listings, and no serving default or numerical map moves. What lands is the instrument, the numbers,
and one warning about how arms are compared in this loop.

Raw samples, per-arm phase tables and the compiled listings:
[`batch-comparison/seq-block-roundtrip/`](../../../data/bonsai2/batch-comparison/seq-block-roundtrip/README.md).

## The instrument

`-DHALO_SEQ_PIN=n` has masked one or both streams' block index since
[the pair alphabet](seq-pair-operand.md) needed it, and that document closes by asking for the
reading this one takes: *"`-DHALO_SEQ_PIN=1|2|3` answers that for this loop in one build and one
panel, and it should be run **before** any further instruction work on `project`."* Nobody ran it,
because a baked-in arm costs a rebuild and a second process, and this box moves 10-15% between
processes.

`-DHALO_SEQ_PIN_CONTROL=1` multiplies each stream's block index by a device word the host writes,
so `--seq-pin 0,1,2,3` walks all four arms in one process against one warmed device and one clock:

```sh
make DEFS='-DHALO_SEQ_PIN_CONTROL=1'
tools/run-batch-compare --pin-clock --profile-tool --modes 19 --rows 128,256 \
  --seq-pin 0,1,2,3 --rounds 3 --heads 0 --traces 1 --out RUN.json
tools/arm_panel.py RUN.json --key seq_pin --phase sequence-input-projection
```

A pinned arm collapses the named stream onto block 0, so it hits in cache with every instruction,
request, expansion and matrix operation of the deployed loop intact. It computes the wrong numbers
by construction — the residual FNV-64 differs per arm, which is how you know the pin took — and
exists only to split a phase into what its instructions cost and what its memory costs.
`sequence_set_pin()` refuses a non-zero mask on a build that does not carry the control.

## The split

Mode 19, `--heads 0`, traced, arms shuffled and interleaved inside one process, three rounds each.
Control column is the five phases `project<>` cannot reach (`embed`, `ffn`, `gdn-resident-core`,
`sequence-core`, `sequence-input-prep`); every figure below is normalised on it.

| 128 rows, 640 waves, `TT=4 W=2` | input projection | output projection |
|---|---:|---:|
| deployed | 28.819 ms | 10.920 ms |
| weight stream pinned | **−7.51%** | **−3.66%** |
| activation fragment pinned | **−1.40%** | **−0.08%** |
| both pinned | **−8.01%** | **−3.79%** |

| 256 rows, 1280 waves, `TT=4 W=4` — the width the engine ingests at | input | output |
|---|---:|---:|
| deployed | 46.383 ms | 21.946 ms |
| weight stream pinned | +0.46% | −0.80% |
| activation fragment pinned | −2.00% | −1.37% |
| both pinned | **−3.82%** | **−0.47%** |

| 32 streams, one token each, `TT=2 W=4` | device span |
|---|---:|
| deployed | 74.949 ms |
| weight stream pinned | **−7.08%** |
| activation fragment pinned | **+0.07%** |
| both pinned | **−7.21%** |

The decode row is the whole step, not a phase: **7.1% of aggregate batched generation is these two
kernels waiting on a 512-byte weight load that nothing covers.** It is the largest uncollected
number this turn found, and it is on the shape the batched product runs.

## What the split says, in order

1. **The activation path is closed and it was never open.** Thirty-two of thirty-nine requests and
   sixteen times the weight bytes cost 1.4%, 2.0% and 0.07% at the three shapes. Every mechanism
   that cuts that re-read has been paying for something the machine was already giving away, which
   is why `RT = 2` (a third less total traffic) measures +7.3% and the LDS stage +21.6%. **Do not
   spend another arm there**, and read [the token map's line-lookup law](seq-operand-coord.md) as
   what it is: a correct account of a term that does not matter here.

2. **The round trip is the weight stream, and it is the dependent head of the block.** The two
   `global_load_b128` of codes and the `global_load_d16_b16` scale are on 64-bit VGPR address pairs,
   the scale's eight `ds_bpermute` broadcasts cannot start until it lands, and the 96-operation
   expansion cannot start until the codes do. The thirty-two fragment loads issue on a scalar base
   with a lane offset and are all in flight before the first matrix instruction. One stream is
   covered; the other is the critical path.

3. **It is not a property of the kernel, it is a property of the launch.** 8.0% at 640 waves, 3.8%
   at 1280, 7.2% at the decode shape where a block has only two token tiles of work to hide behind.
   At 256 rows neither stream alone accounts for the 3.8% — pinning the weights is +0.46% — because
   with 1280 waves the machine covers whichever one is left. **A cover experiment measured at one
   width says nothing about another.**

4. **It re-prices two standing results without re-running them.**
   [The lookahead arm](sequence-weight-address.md#the-lookahead-arm-lost-and-the-shape-of-the-loss-is-the-useful-part)
   measured −7.50% against deployed at 128 rows and was written up as a loser because the
   wave-uniform base arm took −10.50% in the same panel. −7.5% is the weight round trip to two
   decimal places. That arm was not a third of a better idea; **it collected the entire term this
   panel isolates**, and it was rejected against a comparison rather than against a floor.

## The warning, which is worth more than the number

The pair alphabet ([`seq-pair-operand.md`](seq-pair-operand.md)) cuts 57 work slots a block and was
rejected on a +2.48% ingestion cell. Measured again this turn on **two builds that differ by one
`-D`** — `HALO_SEQ_PIN_CONTROL=1` adds two `v_mul_lo_u32` per block **to both arms equally** and
changes nothing else:

| `seq_i4op` 2 against 1, input projection, normalised | 128 rows | 256 rows |
|---|---:|---:|
| build carrying the pin control | **−8.34%** | **−5.17%** |
| default build | **+3.1%** | **−2.7%** |

Same source, same arms, same interleaving, same wrapper, same day, one multiply apart. The sign
flips and the magnitude moves eleven points. Both panels are internally consistent — the pin-control
build's own control column moves 0.1-1.6% — and both reproduce their own cells across widths.

**An arm comparison in this block loop is not robust to a schedule perturbation applied equally to
both arms.** That is the mechanism under the folklore three engineers have now recorded
independently: instruction cuts here convert at somewhere between zero and half of their issue
model, unpredictably, and [`seq-pair-operand.md`](seq-pair-operand.md) had already seen half of it
from the other side ("the wave now reaches its `s_waitcnt` sooner with nothing else to do"). The
block is a scheduling equilibrium between one uncovered dependent load and about three hundred
instructions of cover. Move any of the cover and the equilibrium moves with it.

The practical consequence: **do not flip a default in `project` on a control build.** The
pair-alphabet default is left where it is, at `HALO_SEQ_I4OP = 1`. The −5.17% at 256 rows is real
on that build and means nothing for the shipped one, and the honest reading of the default build's
own panel — +3.1% at 128 rows, −2.7% at 256, with the control column drifting 1.7 to 4.0% between
arms — is that the effect is inside this box's drift at the width that matters.

## Where the phase actually is, converted

With the constants [`ffn-matrix-rate`](ffn-matrix-rate.md) and
[`ffn-slice-issue-order`](ffn-slice-issue-order.md) settled — `v_wmma_i32_16x16x16_iu4` is 17.15
cycles per SIMD32, exactly half of `iu8`'s 34.3, and a slot is about 1.57 cycles — a 128-row pass
issues **162.5M** matrix instructions in the input projection (48 recurrent layers × 1024 row tiles
× 2 token groups × 40 blocks × 32, plus 16 attention layers × 896 row tiles). The block loop
censuses at **371 instructions, 325 work, 32 of them matrix**
(`tools/isa_loop_count.py`, `project<TT=4, ADD=0, PLANAR, WV=2, SHARE, RT=1, SB=1, LA=LANE,
I4=nibble, TM=0>`).

At the 2214 MHz the 128-row panel held, that is **15.7 ms of matrix pipe against 26.4 ms pinned and
28.8 ms deployed**: the phase is 60% matrix at its hardware rate, 8% memory, and 32% the other 293
instructions at about 1.27 cycles each. The fully serialised model — 32 × 17.15 + 293 × 1 cycles per
wave-block — predicts 22.0 ms against 24.1 measured under the pin, so **matrix and VALU issue do not
overlap here**, which is the opposite of what
[the FFN's probe](ffn-matrix-rate.md) found for fourteen *independent* VALU beside a matrix
instruction. The difference is that every non-matrix instruction in this block is on the operands'
dependence path.

**What is left in this kernel is 12.6 ms of a 128-row pass and it is instructions, not bytes** —
and the warning above says that is exactly the currency this loop pays out in unpredictably.

## For whoever takes the weight stream

The target is stated as sharply as this panel can state it: **7.1% of a 32-stream generation step,
7.5% of the input projection at 128 rows, and about nothing at 256.** Three things a successor
should have:

- **The registers are there at the decode shape.** `project` takes 90-95 VGPR at `TT = 2` and the
  budget for twelve waves per SIMD32 is 128, so the nine VGPRs a one-block weight lookahead needs
  (eight code dwords and a scale) cost no wave slot at 32 rows. At `TT = 4` the deployed
  instantiation is 180 VGPR against the 192 that eight waves allow, so it fits there too.
- **The rejected implementation is the right one and its verdict was wrong.** `SEQ_OP_STREAM` is in
  commit history rather than the tree; rebuild it against `SEQ_OP_LANE` (the deployed path) and
  measure it against the pin floor, not against another arm. Its +22.39% at 32 rows was taken on the
  base operand path, which is itself +2.46% at that shape.
- **Measure the arm at every width you intend it to run.** The rule in `sequence_shape` already
  selects `TT` and `W` by row count and can select the operand path the same way; nothing forces one
  cover strategy on both a 256-row ingestion pass and a 32-row step, and this panel shows those two
  shapes disagreeing about which stream is exposed.

## Commands

```sh
make DEFS='-DHALO_SEQ_PIN_CONTROL=1'                      # the runtime ablation
make                                                      # the default build, codegen-identical
tools/run-batch-compare --pin-clock --profile-tool --modes 19 --rows 128,256 \
  --seq-pin 0,1,2,3 --rounds 3 --heads 0 --traces 1 --out RUN.json
tools/run-batch-compare --pin-clock --profile-tool --modes 19 --decode-streams 32 \
  --decode-prompt 64 --rows 32 --seq-pin 0,1,2,3 --rounds 3 --traces 1 --out RUN.json
tools/arm_panel.py RUN.json --key seq_pin --phase sequence-input-projection
```

Every panel here ran on a box the wrapper called NOT QUIET (2.8 to 10.1 cores of peer builds, clock
p50 2091 to 2650 MHz). Arms are interleaved inside one process and read against their own control
column, which is what that instrument is for; absolute milliseconds are not comparable between the
panels above.

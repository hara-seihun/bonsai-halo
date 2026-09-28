# The sequence projection's weight address, and what an instruction census cannot tell you

`project<>` reads its weight image through three loads per 128-block: two `global_load_b128` of
thirty-two code bytes and one `global_load_d16_b16` of the block's scale. All three carried a
64-bit VGPR address pair. `afc9679` gave the *activation* fragment a wave-uniform base and an
unsigned lane offset and left the weight side alone, so the emitted block showed thirty-two
fragment loads on a scalar base beside those three on address pairs.

This is the measurement of converting them. The short version: **it is -8.26% on the output
projection at one instantiation and +1 to +3% at three others, with the same or fewer instructions
in the faster arm every time.** It ships as a named arm, not as the default, and the transferable
part is the last sentence rather than the first.

## The arms

`--seq-sched 6` and `7` on both measurement tools, `HALO_SEQUENCE_SCHED` or `HALO_SEQUENCE_LA` on
the engine. Both carry the selected row-tile ownership and the fitted slice barrier, so the only
difference between them is where the weight address comes from.

| arm | `SeqOperandPath` | weight loads |
|---|---|---|
| 6 `SEQ_SCHED_LANE` | `SEQ_OP_LANE` | the deployed path: one byte pointer per lane per row tile, indexed by the block |
| 7 `SEQ_SCHED_BASE` | `SEQ_OP_BASE` | wave-uniform base, unsigned offset carrying the lane's column and the block walk |

Arm 6 is the deployed kernel, not a re-expression of it. `project` keeps the deployed operand path
written out in its own `if constexpr` branch and `SeqStream<SEQ_OP_LANE>` holds nothing, because
[the ownership work](sequence-projection-operands.md) already paid for the lesson that a control
built inside a new loop nest is not a control. Against canonical `5b5ebf6` the default build emits
209 block instructions at `TT = 1`, 624 at `TT = 8` and 375 against 374 at `TT = 4`, at identical
VGPR counts and identical waves per SIMD32.

## Two things were needed to reach the scalar-base form, and the second one is not obvious

1. **`readfirstlane` the row tile, not just the token group.** `tile` is built from `blockIdx` and,
   off the shared path, from `threadIdx.x >> 5`. Divergence analysis will not split a base out of
   it however the index is written.
2. **Keep the base fixed and advance the offset.** Built base-advancing first — `cb += 512` per
   block with a constant lane offset — the census read -14 instructions and the address-form table
   said nothing had happened: the loads stayed on 64-bit VGPR pairs. Moving the block walk into the
   unsigned offset is what produced `global_load_b128 v_off, s[base]`. That is the opposite of what
   reads naturally, and it is the same shape as the fragment walk's `bo` two loops below.

`tools/isa_loop_count.py` prints the address-form table, which is the only way to see that the
first attempt had not happened. Use it before spending a lock session.

## The panel

`tools/run-batch-compare --pin-clock --profile-tool --modes 19 --heads 0 --traces 1 --seq-sched 6,7`,
arms interleaved inside one process, normalised on the four phases `project<>` cannot touch
(`ffn`, `gdn-resident-core`, `sequence-core`, `sequence-input-prep`). Raw in
`batch-comparison/seq-operand-stream/`.

| rows | output stage shape | output projection | input projection |
|---:|---|---:|---:|
| 32 | `TT=1`, `WV=2`, 640 waves | +2.51% | +2.45% |
| 64 | `TT=2`, `WV=2`, 640 waves | +2.98% | -0.23% |
| 128 | `TT=4`, `WV=2`, 640 waves | **-8.26%** | +0.16% |
| 256 | `TT=4`, `WV=4`, 1280 waves | +1.62% | +1.09% |

The 128-row cell was measured twice on two binaries: -10.50% on `f4f5eaad` and -8.26% on
`353679420`, every round of the base arm below the matched round of the deployed one in both. All
arms are bit-identical — residual FNV-64 `1873717996769712938` at 32 rows,
`14426824474163741251` at 64, `7446865760224376151` at 128 and `18439807992172728808` at 256, the
same value in every arm of every panel.

## Why it does not ship as the default

Prompt ingestion has defaulted to 256-row passes since [the pass-width change](pass-width.md), and
a generation step is 32 rows. **Those are the two shapes where this change is a regression.** The
one shape that gains 8-10% is a benchmark width the engine no longer selects. A rule fitted to it
would buy the product nothing and cost it a point, so the default stays on the deployed path and
the base form stays reachable for whoever re-fits after the next ownership or width change.

## Installed acceptance

Canonical `b58eabff`, installed executable `070560ad24ea0eaf`, mode 19, 256 rows with the head on
and logits taken, both arms in one process under `tools/run-batch-compare`: residual FNV-64
**`18439807992172728808`** in both, the same value four separate checkout panels produced at this
width. Wall 309.229 ms on the deployed path against 313.542 on the base arm, which reproduces the
+1.6% the phase panel measured. Raw:
[`seq-operand-stream/installed-identity.json`](../../../data/bonsai2/batch-comparison/seq-operand-stream/installed-identity.json).

One operational note that cost two attempts. **The canonical working tree's built binaries can be
older than canonical source**, because several engineers run `make` in `.`
at once and a build started from an older tree state leaves its objects behind. The first
acceptance run failed with `batch_profile: seq-sched must be -1 ... or 5`, which is the *previous*
range check, against a source file that already carried the new one. `touch` the source you changed
and rebuild before believing an installed-executable failure.

## The transferable result: the census does not predict the sign

This lane has been converting kernels to the scalar-base form since
[the FFN's operand address](ffn-operand-address.md) measured -5.56% on its `ffn` phase and +3.7%
full model, and that result was read as *the address arithmetic was an eighth of the block loop*.
This panel separates the two claims, because the base arm here removes **no work instructions at
all** at `TT = 4` and **seven** at `TT = 1`:

| output stage instantiation | deployed | base | VGPR | waves/SIMD32 | measured |
|---|---:|---:|---|---:|---:|
| `TT=4, WV=2` | 375 = 319 work + 56 sched | 372 = 319 + 53 | 135 → 132 | 10 → 10 | **-8.26%** |
| `TT=1, WV=2` | 210 = 186 + 24 | 204 = 179 + 25 | 78 → 70 | 16 → 16 | **+2.51%** |

Same work, same occupancy, opposite signs. The arm with *fewer* instructions is the slower one at
`TT = 1`. Rebuilding the offset from the block index instead of carrying it as a running add was
the first suspect for the short-block regression; removing those seven slots did not move the
32-row number at all (+2.51% before and after).

So on this device the addressing mode is worth something that is not its instruction count, and
whatever that is changes sign with the instantiation. **An issue-slot census does not predict the
sign of an addressing change here, only its cost.** That is the same wall
[the activation scale axis](activation-scale-axis.md) hit from the other side — -37 slots in the
head, measured null, address forms byte-identical between builds — and it is consistent with
`043b0486`'s four-bit input projection, which is 1.84x on a census that predicted 1.60x. The block
is not slot-limited; it is limited by how fast operands arrive, and both the sign here and the
extra 24% there live in that gap.

## The lookahead arm lost, and the shape of the loss is the useful part

> **Re-priced, and the verdict here was taken against the wrong reference.**
> [`seq-block-roundtrip.md`](seq-block-roundtrip.md) isolates this loop's memory round trip with a
> cache pin and finds it is **7.5% of the input projection at 128 rows** and **7.1% of a whole
> 32-stream generation step**, essentially all of it the weight stream. The −7.50% below is that
> term to two decimal places: **the lookahead arm collected the entire round trip**, and it reads as
> a loser only because the base arm's −10.50% in the same panel is doing something else (address
> form, not cover). Whoever rebuilds it should measure it against the pin floor rather than against
> another arm, and at each width separately — the same panel finds the weight stream exposed at 128
> rows and 32 streams but not at 256.


A third arm held one 128-block of the weight stream in registers, so a wave always had a block in
flight, in the form [the FFN's dense stream](../kernels/ffn_dense_stream.hpp) uses. Two instruments
in the repository pointed at it: the per-phase clock fit in [clock-power.md](clock-power.md) puts
the output projection at 59.6% clock-proportional against the input stage's 83.5%, so two fifths of
it is time that does not answer to the shader clock, and
[the fragment-reuse panel](sequence-fragment-reuse.md) found the slice barrier's removal a null on
the input stage and -4 to -6% on the output one, for the same reason.

It lost anyway, and not on registers: 133 VGPR against 132 and ten waves per SIMD32 either way.

| rows | lookahead vs deployed | base vs deployed |
|---:|---:|---:|
| 128 | -7.50% | -10.50% |
| 32 | +22.39% | +2.46% |

At 128 rows it gives back a third of what the base form buys; at 32 rows it costs 22%. The block
already issues thirty-two activation fragment loads before its first matrix instruction, so
requesting the next block's thirty-two weight bytes ahead of the ones it needs now makes the wave
wait longer, not less. **Lookahead is the wrong prescription for a block that is already carrying
many outstanding loads**, whatever the occupancy says — which is worth knowing for the dense FFN
arm, where the depth axis is being pushed past two.

The implementation is not in the tree. The measurement is here, and reproducing it needs the
`SEQ_OP_STREAM` specialisation from commit history rather than a fresh design.

## Where a pass actually spends its time at the width it now runs

Nobody had profiled the 256-row shape since it became the default. Mode 19, `--heads 0`, deployed
arm, traced, device phase time:

| phase | 32 rows | 64 rows | 128 rows | 256 rows | ms/row at 256 |
|---|---:|---:|---:|---:|---:|
| `ffn` | 32.562 | 55.098 | 81.635 | 136.437 | 0.5330 |
| `sequence-input-projection` | 14.917 | 33.761 | 47.662 | 72.879 | 0.2847 |
| `sequence-output-projection` | 7.834 | 12.179 | 22.132 | 31.038 | 0.1212 |
| `sequence-core` | 4.166 | 9.393 | 4.757 | 30.336 | 0.1185 |
| `gdn-resident-core` | 5.291 | 10.780 | 17.643 | 29.700 | 0.1160 |
| `sequence-input-prep` | 1.739 | 2.760 | 4.037 | 5.527 | 0.0216 |
| device span | 67.7 | 125.6 | 179.2 | 308.8 | |

The columns come from four different panels on a shared box, so read down a column and not across
a row; only the within-panel arm comparisons above are controlled.

Two things in it are worth someone's turn. **The two projections together are 34% of a 256-row
pass**, second only to the FFN's 44%, and the input stage alone is 24% — which is the ground
`043b0486`'s four-bit input arm is standing on. And `sequence-core` is 9.9% of a 256-row pass
against 6.3% of a 32-row one: it is the one phase whose cost per row grows with the pass, because a
wider pass covers later positions and every one of them rescans more KV. Widening the pass again
trades the projections' amortisation against that.

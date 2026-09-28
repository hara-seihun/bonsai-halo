# The wide sequence projections: operand build and row-tile ownership

`sequence-input-projection` and `sequence-output-projection` are 55 ms of a 178 ms 128-row A4
pass, second only to the FFN. Both run `project<>` in `kernels/sequence_batch.hip`, which had never
been touched by the row-tile ownership work that
[the batched FFN](ffn-schedule.md) landed, and whose IU8 operand build spent more issue slots
moving bits into byte lanes than on the byte selection itself.

Two changes, both exact. Every product, every summation order and every block scale is the
deployed one, and the acceptance below compares 71,516,160 full-vocabulary logits against the
canonical build with zero differing bits.

## The stored word already knew where the bytes had to go

`expand_i8` turns sixteen two-bit codes into sixteen signed bytes. `v_wmma_i32_16x16x16_iu8` wants
one code per byte lane; the deployed word keeps code *k* at bits 2*k*, so four codes share a byte
and each of the four output dwords costs a spread — two shift/or pairs and two masks — before the
`v_perm_b32` that actually selects the weight. Sixteen issue slots per K16 slice, twelve of them
moving bits.

A code's bit position inside the stored word is a free choice of the packer. Put the code of K slot
*k* at

    bit 8 * (k & 3) + 2 * (k >> 2)

and the spread is already done: selector dword *j* is `(word >> 2j) & 0x03030303`, whose byte *i*
is exactly the code of K slot 4*j* + *i*. Eleven slots per slice instead of sixteen, the same 2.000
bits per weight, the same eight dwords per (row, 128-block), and the K order the B operand sees
does not move — so the weights are the deployed weights in the deployed slots.
[`kernels/sequence_operands.hpp`](../kernels/sequence_operands.hpp) holds the permutation, its
inverse and the expansion. The scaled-FP16 control route keeps the deployed order, because
`expand_scaled` folds its own spread into a selector pair it needs anyway.

The second half of the same change is addressing. The fragment and scale strides are wave-uniform
and a lane's own column is a fixed offset, so the block loop can advance a pointer instead of
rebuilding an index per (slice, token tile). That deletes all 64 `v_mul_u32_u24` of a TT = 8 block.

## Row-tile ownership, and why it has to be per stage

A batch of `ntiles` 16-token tiles is spent two ways: `TT` tiles in one wave's accumulators, and
`W` waves of a workgroup owning different token groups of the same weight row tile. `TT` is
accumulator width and costs 16 VGPRs per tile; `W` is grid parallelism that costs no registers and
re-reads no weights, because the `W` waves stream the same bytes together. An output element stays
owned end to end by one wave either way, which is why nothing is reduced across waves and no bit
moves.

The compiler's numbers for this kernel, `-Rpass-analysis=kernel-resource-usage`:

| token tiles per wave | VGPRs | waves/SIMD32 |
|---:|---:|---:|
| 8 | 242 | 5 |
| 4 | 153 | 9 |
| 2 | 95 | 12 |

So the deployed shape at 128 rows, `TT = 8` with a wave on its own row tile, is the bottom of the
occupancy cliff. The rule that replaces it is the FFN's, with the same 640-wave target — eight per
SIMD32 on this device's 80 units — and a budget of four token tiles, which is the widest shape that
still holds eight waves:

    W = min(4, max(ceil(ntiles / 4), ceil(640 / row_tiles)))

Both terms matter, because the two projections are not the same shape. The input matrix owns 1024
row tiles (896 on attention layers) and the output matrix 320:

| rows | stage | row tiles | deployed | selected |
|---:|---|---:|---|---|
| 128 | input | 1024 | TT = 8, own tile | **W = 2, TT = 4** |
| 128 | output | 320 | TT = 8, own tile | **W = 2, TT = 4** |
| 32 | input | 1024 | TT = 2, own tile | unchanged |
| 32 | output | 320 | TT = 2, own tile | **W = 2, TT = 1** |

The 32-row input cell is the one that made the rule necessary rather than optional. Forcing two
waves onto its row tile there measured **+20.6%**: its grid already holds 1024 waves, so sharing
buys no occupancy and only halves the token tiles each block's expansion serves. The output
projection at the same 32 rows holds 320 waves, under the 640 the device wants, and the same
sharing measured **-4.2%**. One target, two answers, exactly the asymmetry the FFN found between
gate/up and the down projection.

## Issue census

Per 128-K block of one wave, counted as issue slots with a `v_dual` line counted once. `main`
is canonical `16c5d16`; the rest is this change.

| shape | slots | wmma | A expansion | drain | VMEM | B index | VGPR | waves/SIMD32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| main, TT = 8, own tile | 647 | 64 | 128 | 130 | 75 | 64 | 252 | 5 |
| TT = 8, own tile | **508** | 64 | **88** | 130 | 75 | **0** | 242 | 5 |
| TT = 4, W = 2 | 330 | 32 | 88 | 64 | 39 | 0 | 153 | 9 |
| TT = 2, W = 4 | 253 | 16 | 88 | 32 | 21 | 0 | 95 | 12 |

At 32 cycles per IU8 WMMA a deployed block is 2048 + 583 = 2631 modelled cycles and the new one
2048 + 444 = 2492, so the operand change is worth 5.3% of the stage if the stage is issue-bound.
Narrowing to `TT = 4` raises the per-token issue cost about 6% and the occupancy from five waves to
nine; that trade is what the measurement below decides.

## Measured

`tools/batch_profile --modes 19 --heads 0 --traces 1`, two rounds, medians, **three ownerships
interleaved inside one process** through `--seq-sched 0,1,2`, so the arms share an executable and a
clock. `ffn`, `gdn-resident-core` and `sequence-core` are untouched by this change and are the
in-panel controls.

128 rows:

| phase | own tile | **selected** | wide (W = 4) |
|---|---:|---:|---:|
| `sequence-input-projection` | 39.605 | **38.368** | 39.051 |
| `sequence-output-projection` | 16.120 | **15.964** | 16.044 |
| `ffn` (control) | 85.259 | 85.901 | 86.651 |
| `gdn-resident-core` (control) | 22.908 | 23.066 | 23.119 |
| `sequence-core` (control) | 14.907 | 15.041 | 15.263 |

32 rows:

| phase | own tile | **selected** | wide (W = 4) |
|---|---:|---:|---:|
| `sequence-input-projection` | 14.135 | 14.227 | 14.153 |
| `sequence-output-projection` | 8.143 | **7.798** | 7.773 |

The controls move at most 0.91% and in the opposite direction, so the input projection's -3.1% and
the output projection's -1.0% at 128 rows, and the output projection's -4.2% at 32, are the change.
A separate panel that forced the ownership instead of deriving it gives the sign of the cell the
rule refuses: 32-row input, own tile 14.062, forced W = 2 **16.961**, forced W = 4 17.189.

The operand change is measured against the canonical `16c5d16` build at the deployed ownership,
normalised by the same untouched phases (control ratio 1.0355 on that pair of panels):

| stage | main | this change, own tile, normalised |
|---|---:|---:|
| `sequence-input-projection` | 38.411 | 38.25 (-0.4%) |
| `sequence-output-projection` | 16.480 | 15.57 (-5.5%) |

**The input projection is not issue-bound, and that is the most useful thing in this document.**
Deleting 21.5% of its block instructions moved it 0.4%. The output projection, whose 320 row tiles
put one workgroup per processor, took 5.5% from the same deletion. Whatever holds the input
projection at 38 ms against a 24 ms IU8 issue budget is not instruction supply.

Full model, 384-token document in 128-row passes with the head on the last pass, `tools/batch_compare
--modes 0,19 --only prefill --prefill-rows 128`, with mode 0 interleaved in the same process as its
own control: **695.2 to 702.2 prompt tok/s, +1.0%**, medians of 12 and 9 rounds, mode 0 at 156.9 and
157.2. The round-to-round spread of mode 19 on this box was 647 to 720 tok/s while five other
engineers were building and measuring, so that panel resolves about 2% and the phase table above is
the stronger evidence. The arithmetic between them agrees: 2.8 ms off a 178 ms device span is 1.6%.

## Numerically identical

The packing is a bit permutation of the stored word and the ownership is a choice of which wave
owns a (row tile, token group). Neither changes a product, a summation order or a scale.

[71,516,160 full-vocabulary logits](../tools/batch-compare/results/sequence-projection-identity.json)
at 32, 40, 88 and 128 prefill rows, canonical `16c5d16` against this build, **zero differing bits**
and `max_abs_diff` 0. The residual FNV-1a over all 128 x 5120 FP32 outputs is
`7446865760224376151` in every ownership arm, the same value the row-tile ownership work published,
and `1873717996769712938` at 32 rows.

## Installed and accepted

Published as `26fd8a6` with this document at `a3b56bc`, built into the canonical executable and
restarted under the resident service. The installed acceptance ran under `tools/run-batch-compare`
on canonical `b78f36c`, which by then also carried the afternoon's FFN and resident-GDN work:

- `--only quality --quality-rows 32,40,88,128` on the installed build against the pre-change
  canonical dump: **71,516,160 logits, zero differing bits**, `max_abs_diff` 0
  ([`installed-identity.json`](../../../data/bonsai2/batch-comparison/sequence-ownership/installed-identity.json)).
  That also re-verifies every other exact change landed between the two dumps.
- `--engine --bench -n 100`: 33.49 tok/s, 29.86 ms/token, with another engineer's `batch_compare`
  and a `qemu` guest on the box. Single-stream decode never calls `sequence_project`; the band for
  that path on a busy machine is 32.95 to 33.65 against 34.0 idle.
- The resident server came back on the new binary.

## A control that was not one

The first attempt at isolating the operand change kept the deployed expansion and the deployed
address arithmetic behind a template flag *inside the new loop nest*. It is 720 issue slots against
the deployed kernel's 647, and it measured 1.40x the input projection and 1.80x the output
projection of the canonical build on a panel whose untouched phases were within 1.1% of it. Eleven
percent more instructions do not explain that; the scheduler simply produced a different, worse
kernel (74 `s_waitcnt` and 27 `s_clause` against 64 and 9), and the arm was deleted rather than
published.

**Do not build a control for an operand change by putting the old operand code inside the new loop
nest.** Compare against the build you are replacing, and normalise on phases the change does not
touch.

## The token group was uniform and the compiler could not know it

`first`, the wave's token group, is built from `threadIdx.x` through `wave`. It is the same value in
every lane, but divergence analysis has to assume otherwise, so every address derived from it was a
per-lane 64-bit value. That is what the block loop was paying for:

- the activation scale `scp[(size_t)t*16*w.nb+b]` cost a 64-bit multiply per (token tile, block) —
  four `v_mad_u64_u32`, eight `v_mul_lo_u32`, four `v_add3_u32` and two `v_lshlrev_b64` per block
  rebuilding four addresses whose only per-block change is four bytes;
- the fragment walk advanced a 64-bit per-lane pointer once per slice, a carry pair each time.

Naming the uniformity with `__builtin_amdgcn_readfirstlane` gives the fragment and scale loads a
scalar base and a 32-bit lane offset, the form
[the vocabulary head](wide-head.md) took from 733 slots to 600. All the 64-bit index arithmetic
goes; eight 32-bit adds and about thirteen scalar ops replace it. The accumulators also stopped
being re-zeroed per block: one zero octet feeds the first slice's `src2`, as in `k_proj_opt`.

| instantiation | block slots | VGPR | waves/SIMD32 |
|---|---:|---:|---:|
| 128-row input and output, `TT = 4`, `W = 2` shared | 382 → **369** | 154 → **142** | 9 → **10** |
| 32-row input, `TT = 2`, own tile | 284 → **257** | 95 | 16 |
| 32-row output, `TT = 1`, `W = 2` shared | 225 → **209** | 78 | 16 |

One cell went the other way and is not in the change: making the *weight* base uniform as well
costs four slots back, because that stream is read once per block and folding the lane's column
into the base is already free.

**It is a wall-clock null, and the slot price says it should be.** At the measured 0.023 ms per slot
per block, 13 slots is 0.35 ms of the input projection and 0.13 of the output — 0.26% of a 128-row
pass, where two panels of the same instrument forty minutes apart disagree by 2.7% on phases
nothing touched. Measured anyway, `--modes 20 --rows 32,128 --traces 1 --rounds 3`, fastest round
of each panel, normalised on `ffn`, `gdn-resident-core`, `sequence-core` and `sequence-input-prep`:
the input projection is +0.5% at 128 rows and +2.1% at 32, the output projection +0.03% and -1.6%.
The signs disagree between the two stages at 32 rows, which is what a null looks like on this box.
Raw: [`seq-row-tiles/phase-addr.json`](../../../data/bonsai2/batch-comparison/seq-row-tiles/README.md).

It ships on the census rather than on the clock: bit-identical, 3.4% fewer instructions in the
128-row block loop and 9.5% fewer at 32 rows, twelve registers and a wave slot back, and no
measured regression at either row count. The residual FNV-1a over every FP32 output of all 64
layers is unchanged across the two builds — `16921095978579329146` at 128 rows,
`15659291503748132804` at 32 — and this change cannot reach the head that reads it.

## The shader clock is 2428 MHz, not 2900

The clock log beside the panel above is the third reading this lane has taken, and the two measured
*under a real payload* agree: `sclk_p50` **2428 MHz**, mean 2365, busy 93%, package 124 W, 4.64 s
from idle to 95% of the ceiling. A peer reading the same counter with the performance level forced
high got 2420. The 2848 MHz in the ceiling sweep is a p50 over samples that were only 27% busy,
which is the governor between passes rather than the clock a matrix instruction retires at.

That moves the budget this document is written against. The IU8 matrix-pipe floor for
`sequence-input-projection` is **28.8 ms, not the 24 quoted above**, against 38-41 measured, so
three quarters of this stage is matrix instructions retiring and the serial issue model accounts
for the rest almost exactly — [the panel that fits it](sequence-fragment-reuse.md#one-issue-cycle-per-slot)
recovers 2.33 to 2.48 GHz from three measured shapes. There is no reservoir of stall here to
attack; the remaining work is instructions.

## Knobs

- `--seq-sched 0,1,2` on `tools/batch_profile` and `tools/batch_compare` interleaves the ownership
  inside one process; `-1` leaves whatever the build selects. `sequence_batch_set_sched()` is the
  setter behind it.
- `HALO_SEQUENCE_SCHED` pins it for a process that cannot use the setter, `HALO_SEQUENCE_W` forces
  the wave count and `HALO_SEQUENCE_TT` the accumulator width. All three appear in `halo_env`.
- `HALO_SEQUENCE_OPERAND=scaled-f16` still selects the FP16 control route, which keeps the deployed
  word order.

## What to attack next here

The input projection is 38 ms against 24 ms of IU8 issue budget and does not respond to
instructions *at the deployed ownership's five waves per SIMD32*. The traffic it does respond to is
the B operand: every row tile streams the whole fragment image for its token coverage, and the
input matrix has 1024 of them, so one layer reads 655 KB of fragments 1024 times — 671 MB per
layer, 43 GB per 128-row pass. Row-tile sharing does not change that number, because the waves
sharing a tile hold *different* token groups.

The lever that would halve it is the one the FFN already uses and this kernel does not: **give a
wave two row tiles against one B fragment**, the way `k_proj_opt` carries gate and up, or the two
halves of the down matrix, on a single fragment read. At `W = 4`, `TT = 2` and two row tiles per
wave the accumulator cost is the same 64 registers as today's selected shape, the expansion doubles
to 176 slots per block, and the fragment loads and their addresses halve. If the input projection
is fragment-bound, that is worth far more than the 0.4% instructions bought.

> **It was built, and it is not.** Two row tiles per wave loses 7.3% on one instrument and 2.3% on
> another, four row tiles lose 15.5%, and the bold sentence above - that this stage does not
> respond to instructions - is an artifact of the cross-build pair that produced it. In process,
> its instructions convert at about one issue cycle each.
> [What binds the sequence projections](sequence-fragment-reuse.md) has both panels, the marginal
> price of a block-loop slot, the measured shader clock the budgets above are missing, and the
> schedule change that came out of the same panel.

The shape family around the selected point is now searched out: `TT` 1, 2, 4, 8 crossed with `W` 1,
2, 4 here, and row tiles per wave 1, 2, 4 there. What is left in this kernel is a cheaper operand
or a cheaper instruction, not a better ownership.

And the operand is already at its floor. Eleven slots per K16 slice turn sixteen two-bit codes into
sixteen signed bytes: three shifts, four masks, four `v_perm`. A nibble image halves the masks but
needs two source dwords per slice instead of one, so it lands at ten slots for twice the bytes —
one slot saved for a 1.96 GB image becoming 3.9 GB. Byte-aligned codes would remove the spread
entirely and cost four times the image. The drain is two flops per output element per block because
the scale is per block. **So the arithmetic here is finished, and the next real step is fewer matrix
instructions rather than fewer slots around them.**

That is worth pricing, because it is large. These projections issue 2.62M IU8 matrix instructions
per layer at 32 cycles each, which is 28.8 ms of the input projection's 38-41 and 75% of its block
cycles. `v_wmma_i32_16x16x16_iu4` retires in half that. The engine already ships four-bit
activations in the FFN as an explicit numerical mode with its own quality evidence (`--ffn a4`,
mode 19), and the same operand build would get cheaper rather than more expensive, since ternary
weights fit a nibble. A four-bit sequence projection is roughly a 45% cut of both stages, about
10% of a prefill pass and a fifth of a generation step — far past anything left in the schedule.
What it needs is not a kernel but a quality panel: these projections feed attention and the
recurrent state, not a residual sum, so the teacher-forced KL and multistep continuation the A4
FFN was accepted on have to be re-run for them.

> **That 45% is matrix cycles, and matrix cycles are not what this kernel is short of.**
> [The output projection's counters](sequence-output-projection.md) put the memory unit at 95.9%
> busy in the output stage and 97.8% in the input one, level with `resident_state` at 224 of
> 242 GB/s and 10 to 19 points above the FFN. And the swap does not relieve that path at all:
> `iu8` asks for 16 activation bytes per lane in 32 cycles, `iu4` for 8 in 16 — the same half byte
> per lane per cycle, because K is 16 either way. Four-bit activations halve the arithmetic and
> leave the operand stream exactly where it is. Pair them with two weight row tiles against one
> fragment, which is what makes `k_proj_opt`'s `iu4` arm work, or measure the memory path first.
> The quality panel is the expensive half of that build; do not spend it on the arithmetic alone.

## The other half of that pricing, measured

> **The operand build is not what this loop is short of either, and the panel that says so is
> cheap.** [The pair alphabet](seq-pair-operand.md) gives the four-bit expansion the A4 FFN's
> nine-valued weight-pair codes at *identical* bytes - 2.000 bits per weight, the same 512 bytes per
> (16-row tile, 128-block) - and takes it from twelve instructions a K16 slice to five: 325 to 267
> work slots a block at four token tiles, 191 to 134 at one, two registers cheaper, bit-identical
> and proved so on the host and on the device. It measures **+1.9 to +2.5% slower** at 128 rows and
> +2.0% / +0.7% at a 32-stream step. Pin both of the block's streams into cache with
> `-DHALO_SEQ_PIN=3` and the same cut becomes **−4.2% and −5.7%**. Those 96 operations were the
> cover between a block's loads and their use. Price any further instruction work on this body
> against that ablation first, including the drain.

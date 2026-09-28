# The sequence projections' ternary expansion is latency cover, not cost

Removing **57 of the 325 work slots** in the two sequence projections' block loop, at zero bytes,
zero extra registers and bit-identical output, makes the phase **1.9 to 2.5% slower** at every
deployed shape. Pin the same loop's two memory streams into cache and the identical change becomes
**4.2 to 5.7% faster**. The instructions were not what the loop was paying for; they were what it
issued while its loads were in flight.

Nothing ships. `HALO_SEQ_I4OP` stays at 1, the deployed expansion, and no shipped kernel contains
the other map. Raw samples, both builds' censuses and the ablation are in
[`batch-comparison/seq-pair-operand/`](../../../data/bonsai2/batch-comparison/seq-pair-operand/README.md).

## What was built

`project` reads one stored dword per K16 slice and `expand_i4_nib` turns it into the
`v_wmma_i32_16x16x16_iu4` operand: mask the two-bit codes out of the nibbles, then walk `0b11` up to
`0b1111` in two shift-ors so code 3 reads as −1. Twelve instructions a slice, **96 of the deployed
128-row block loop's 325 work slots — its largest class, against 32 `v_wmma`**.

Nothing about sixteen weights requires them to be stored one per code.
[`ffn_a4_operands.hpp`](../kernels/ffn_a4_operands.hpp) already carries the alphabet that removes the
arithmetic, for the A4 FFN's own image: a nine-valued code names a **weight pair**, so the code *is*
the source-byte selector of the permute that expands it. Nine values fit a nibble, two codes fit a
byte, four weights fit a byte — **2.000 bits per weight, the same 512 bytes per (16-row tile,
128-block) this image already stores**, the same eight dwords per (row, block), the same two
`global_load_b128` a lane-block. The expansion becomes a mask, a shift and two `v_perm_b32`: five
instructions a slice instead of twelve.

It is a relabeling of which bit carries which weight and the repacker owns that, so the image gains
a third stored order beside spread (for `iu8`) and nibble (for `iu4`). `reorder_codes` now takes
`(from, to)` over all three and `sequence_batch_sync_quant` moves the image on an operand-map change
exactly as it already did on a quantiser change.

### Bit-identical by construction, and proved three ways

The permute table's low nibble is the lower K slot, so operand nibble `2i` of dword 0 is K slot `2i`
and nibble `2i+1` is K slot `2i+1` — exactly the map the nibble order builds by hand. Same operand
bits, same activations, same K order, same matrix instruction, same int32 block sum, same two scales
in the same FP32 drain.

- `make kernels/seq_op_check && kernels/seq_op_check` runs both maps on the host against **the
  weights a deployed word names**, not against each other alone: every code in every one of sixteen
  slots, all 81 four-weight combinations a stored byte can carry, 200,000 random words, plus a round
  trip of all three stored orders. One second, no GPU.
- On the device the residual FNV-64 is `3240780174642567821` in every 128-row sample of every arm and
  `6678137683989106976` in every 32-stream decode sample.
- The shipped build's deployed block loop is the canonical one instruction for instruction: 371
  instructions, 325 work, 96 bitops, 41 `s_waitcnt`, before and after.

## The census, and the issue model it implies

`tools/isa_loop_count.py` on the instantiations the launcher actually selects, no GPU. Everything
outside the expansion is equal in both arms: 39 `global_load`, 32 `v_wmma_i32_16x16x16_iu4`, 33
`v_cvt_f32_i32`, 8 `ds_bpermute` at four token tiles.

| deployed shape | arm 1, nibble | arm 2, pair | work slots | block issue at `iu4` = 11.0 |
|---|---:|---:|---:|---:|
| 128-row pass, `TT = 4`, both projections | 325 | **267** | −17.8% | 645 → 587, **−9.0%** |
| 32-stream step, input, `TT = 2` | 236 | **179** | −24.2% | 396 → 339, **−14.4%** |
| 32-stream step, output, `TT = 1` | 191 | **134** | −29.8% | 271 → 214, **−21.0%** |

The expansion itself goes 96 → 39 (23 bit operations and 16 `v_perm_b32`). The matrix cost is
`2464a25e`'s successor `16259bba`'s measured 11.0 issue slots per SIMD32 for `iu4` on this device,
not a nominal number; converting this census with anything else overstates the cut.

**Registers fall and no arm crosses a granule**, `tools/kernel_resources.py`: 180 → 178 VGPR at eight
waves per SIMD32 at `TT = 4`, 91 → 84 at sixteen waves at `TT = 2`, 74 → 69 at sixteen at `TT = 1`,
zero spill and zero scratch in all six.

## Measured, and it loses at both shapes

Mode 19, `--pin-clock`, both arms compiled into one binary (`-DHALO_SEQ_I4OP_CONTROL=1`) and
interleaved inside one process against one warmed device — this box moves up to 15% between
processes ([row-band-fit](row-band-fit.md)). `ffn`, `gdn-resident-core`, `sequence-core` and
`sequence-input-prep` cannot be reached by this change and are the control column; every number
below is the phase over that column.

| 128-row passes | `sequence-input-projection` | `sequence-output-projection` | samples |
|---|---:|---:|---:|
| arm 1, nibble | 0.25071 | 0.10133 | 3 |
| arm 2, pair | 0.25692 | 0.10363 | 3 |
| | **+2.48%** | **+2.28%** | |
| the same panel at six samples an arm | **+1.92%** | **+1.93%** | 6 |

| 32 streams x 1 token | `sequence-input-projection` | `sequence-output-projection` | samples |
|---|---:|---:|---:|
| arm 1, nibble | 0.10620 | 0.08664 | 4 |
| arm 2, pair | 0.10837 | 0.08722 | 4 |
| | **+2.04%** | **+0.67%** | |

The decode cell is worth reading twice. Its **raw** phase times moved +8.7% and +7.3% between arms
while the control column moved +6.6%, so a panel without that column would have published this as a
7% regression, and a two-sample version of it read −1.5% before the drift averaged out. The control
is the measurement.

**And a control column is the only defence this box gives you against its own power budget.**
`268a7387` measured the mechanism on September 21: this is an APU, sixteen Zen5 cores and the 8060S
share one 120 W socket, and a peer's compile running inside your panel takes the shader clock from
about 2690 to 2190 MHz — roughly 12% of a 128-row pass on
[the elasticity table](clock-power.md). `--pin-clock` asks for 2900 and gets what the budget leaves.
Interleaving arms inside one process does not fix that, because a peer's build starts and stops on
its own schedule. What it does fix is the *common mode*, which is what the phase-over-control ratio
reads. Every arm ratio in this document is that ratio, and the +1.9 to +2.5% reproduces across four
separate panels taken over forty minutes.

## The ablation that explains it

`-DHALO_SEQ_PIN=3` masks both streams' block index with an opaque `__device__` zero: the same
instructions, the same request counts, the same expansion and the same matrix work, with only the
addresses stopping their walk so both streams collapse onto block 0 and hit in cache. The arm is
numerically invalid by construction and exists to separate issue from latency. Three rounds an arm,
same panel shape, same control column.

| 128-row passes | input | output |
|---|---:|---:|
| deployed streams, pair against nibble | **+2.48%** | **+2.28%** |
| both streams pinned into cache, pair against nibble | **−4.24%** | **−5.74%** |

**The sign of a 57-slot cut is decided by the memory, not by the slots.** This is also the one
statement in this document that the clock cannot have manufactured: a shared power budget moves a
magnitude, and it does not flip the sign of the same diff under a cache pin. With the round trip gone,
removing the instructions returns about half the −9.0% the issue model predicts, which is what a
block loop that is genuinely issue-bound looks like. With the round trip present, the same removal
costs 2.3%, because those 96 operations sat between a block's loads and their use and the wave now
reaches its `s_waitcnt` sooner with nothing else to do.

The compiled issue orders say the same thing without a GPU. In the nibble arm the expansion is one
96-operation run at the top of the block and the rest is software-pipelined — `WMxWMWMMWMcWMMMxxx`,
`sLLxxxxsLL`, `WMxxxxLxWMWM` — 41 `s_waitcnt`, each with work in front of it. In the pair arm the
compiler spends the freed slots on a different shape entirely: about 34 `global_load` issued in one
cluster at the top, 19 `s_waitcnt`, and the sixteen permutes interleaved with the matrix
instructions behind them. Fewer instructions, fewer waits, and less in front of each wait.

### And the schedule is not the fix

The kernel's slice barrier is already an axis (`SB`), dropped at four token tiles because
[fragment reuse](sequence-fragment-reuse.md) measured that. Forcing it back constrains both arms'
schedules in the same direction and the two respond oppositely, which is the cover story again from
the other side:

| 128-row passes, phase over control | input | output |
|---|---:|---:|
| nibble, no barrier at `TT = 4` (deployed) | **0.25056** | **0.10151** |
| pair, no barrier | 0.25622 | 0.10368 |
| nibble, barrier every second slice | 0.27504 | 0.11040 |
| pair, barrier every second slice | 0.25092 | 0.10859 |

The barrier costs the nibble arm **9.8%** and buys the pair arm **2.1%**: the arm with the long
expansion is the one that wants the scheduler left alone. The deployed cell is still the best of the
four, on both projections.

## What a later engineer should take from this

1. **A census cut on this kernel predicts nothing without knowing what its waits wait on.** The
   companion statement is already in the tree from the other direction — [the A4 FFN's activation
   operand](ffn-b-operand.md): "request count matters where a wait waits on it". Here: *instruction
   count matters where nothing is outstanding*. `-DHALO_SEQ_PIN=1|2|3` answers that for this loop in
   one build and one panel, and it should be run **before** any further instruction work on
   `project`, including on the drain (32 `v_cvt`, 30 `v_mul`, 18 `v_fmac` a block, the second largest
   class after the expansion and still unclaimed).

   > **It has been run, and the answer is the weight stream.**
   > [`seq-block-roundtrip.md`](seq-block-roundtrip.md) reads all four arms absolutely, in one
   > process, at three shapes. The activation fragment is worth **1.4%** of the input projection at
   > 128 rows and **0.07%** of a 32-stream step; the weight stream is **7.5%** and **7.1%**. That
   > document also measures the arm below on two builds one `-D` apart and gets **−8.34%** and
   > **+3.1%** for the same 128-row cell, so **do not read this document's magnitudes across
   > builds** — the pinned/unpinned sign reversal it reports is real, and so is the reversal a
   > single unrelated multiply produces.
2. **The operand map is a process-wide property of the image**, because an image holds one stored
   order at a time and the reorder costs about 30 ms. So a map that wins at decode and loses at
   ingestion cannot be taken per pass — the choice has to be right for both, which is why a 0.67%
   decode cell could not have rescued a 2.3% ingestion cell even had the sign been the other way.
3. **The map itself is good and is preserved.** `expand_i4_pair`, `pair_codes`/`unpair_codes` and the
   three-order `reorder_codes` are in `kernels/sequence_operands.hpp` with a host proof, and any
   `iu4` ternary consumer whose block loop *is* issue-bound can take the same trade at zero bytes:
   the sign extension only looks cheap, and a nine-valued pair code is the same 2.000 bits per weight
   as two 2-bit codes. The head and the drafter are the two candidates; both should be priced with
   their own pin ablation first.

## What came after this

[The token map](seq-operand-coord.md) asked the complementary question on the same block loop —
what the *fetch* costs rather than what the expansion costs — and found the binding term. Removing
half of this loop's memory requests at constant cache-line traffic is a null; dealing the same
requests across twice the lines is +16.25%. Both results are bit-identical against the same residual
FNV-64 `3240780174642567821` quoted above, and together they say the loop pays for lines, not for
requests and not for slots.

## Reproducing

```sh
make kernels/seq_op_check && kernels/seq_op_check          # the map, host, one second, no GPU
tools/isa_loop_count.py /tmp/seq.s projectILi4ELb1ELb0ELi2ELb1ELi1ELi1ELi1ELi1E 1

# both arms in one process; the default build carries only the selected one
make DEFS='-DHALO_SEQ_I4OP_CONTROL=1'
tools/run-batch-compare --pin-clock --profile-tool --modes 19 --rows 128 \
  --seq-i4op 1,2 --rounds 3 --warmup-ms 400 --heads 0 --traces 1 --out RUN.json
tools/arm_panel.py RUN.json --key seq_i4op --phase sequence-input-projection

# the decode shape, and the ablation that explains the sign
tools/run-batch-compare --pin-clock --profile-tool --modes 19 --decode-streams 32 \
  --decode-prompt 64 --seq-i4op 1,2 --rounds 4 --warmup-ms 400 --rows 32 --traces 1 --out RUN.json
make DEFS='-DHALO_SEQ_I4OP_CONTROL=1 -DHALO_SEQ_PIN=3'    # numerically invalid by construction
```

Build under measurement: `bonsai-halo` `f717515f49340c8d0852c88a...`, `tools/batch_profile`
`846bb9ab986d66ac28ea0c33...`, both from the selected arm. The shipped build's own
`--prefill-identity` at 128 and 256 rows a pass compares 95,354,880 logits, all equal, zero
non-finite, FNV-64 `2067497836734107792`.

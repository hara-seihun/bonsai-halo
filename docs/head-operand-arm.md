# The vocabulary head's operand map, and what the head is actually waiting for

The wide head builds its IU8 operand out of HALO five-trit bytes with the same sixteen lines the
deployed matvec used until this evening: multiply a byte pair by three in two 16-bit lanes, shift
the product down eight bits, keep the low bytes as the next remainder, and move two trits back into
byte lanes with `v_lshl_or_b32`. [`kernels/halo_expand.hpp`](../kernels/halo_expand.hpp) landed the
identity that removes the shift — the trit is already byte 1 and byte 3 of the product, so one
`v_perm_b32` gathers four of them straight into the operand dword — and the FFN slice took it first
([the peel gather](mv-peel-gather.md), −1.66% of that phase). The head is the other consumer, and
[`docs/wide-head.md`](wide-head.md) predicted it would read bigger: "220 of the 600 issue slots its
block loop spends", at 24 GB/s of a 242 GB/s roof.

**It reads smaller, and the reason is the useful part: the head is not issue-bound, and it is not
traffic-bound either.** The arm ships because it is free and bit-identical, but the number it
produces prices the head's real constraint for the next engineer.

## What shipped

`head_tile` takes the operand map as a template parameter. `HALO_HEAD_OP=0` compiles the peel back
in as the control, `tools/batch_profile --head-op 0,1` alternates the arms inside one process, and
`--head-tt 1,2,4` walks the token-group width against them on the same clock. The default is arm 1.

Nothing else moves: the same 26 stored bytes, the same 32 operand dwords in the same byte lanes, the
same accumulator seed, the same two `v_wmma_i32_16x16x16_iu8` per K16 slice, the same wave
reduction and the same `fmaf` drain. **Every panel below carries residual FNV-64
`9171727463267762618` in all of its samples, across both arms and all three widths.**

## The census, and the refactor that ate it

`head_tile<2>` block loop, `hipcc --cuda-device-only -S`, gfx1151:

| | instructions | work | `v_pk_mul_lo_u16` | `v_pk_lshrrev_b16` | `v_lshl_or_b32` | `v_perm_b32` | wmma |
|---|---:|---:|---:|---:|---:|---:|---:|
| canonical, before | 600 | 504 | 64 | 64 | 32 | 0 | 32 |
| arm 0 (control) | 619 | 500 | 64 | 64 | 32 | 0 | 32 |
| **arm 1** | **535** | **433** | 64 | **0** | **0** | **32** | 32 |

−13.4% of the block loop's work at `TT = 2` and −16.3% at `TT = 1` (406 → 340). Registers do not
move enough to matter: 131 → 134 VGPR at `TT = 2`, ten waves per SIMD32 in both arms; 113 → 108 at
`TT = 1`, twelve waves in both; 224 at `TT = 4` in both.

**The first version of this switch cost 89 work instructions before it computed anything**, and the
lesson generalises to every body in this engine with a block lookahead. Carrying the block's
registers in a `HeadOperand<OP>::Regs` struct — so the two arms could differ in what they load —
turned the deployed loop's SSA rename (`nqa = qa; ... qa = nqa;`) into 23 extra `v_mov_b32` and took
arm 0 from 504 work instructions to 593. That is larger than the whole expansion cut this change is
about. The arm is now one call inside the deployed loop, and the control arm's census is the
deployed body's: 500 work against 504, within the compiler's own noise.

## Full model

`tools/batch_profile`, mode 20, 128-row passes, 384-token document, pinned clock, both arms
shuffled inside one process against one warmed device, medians. `head-projection` is the traced
phase; the other phases are the in-process control.

| shape | arm 0 peel | arm 1 gather | |
|---|---:|---:|---:|
| 128-row pass, logits on every row (`TT = 2`) | 11.065 ms | **10.666 ms** | **−3.6%** |
| the same, normalised by the eight untouched phases | 0.05122 | 0.04927 | −3.8% |
| 128-row pass, logits on the tail row only (`TT = 1`) | 1.643 ms | **1.578 ms** | **−4.0%** |
| the whole pass forced to `TT = 1` | 13.800 ms | 12.954 ms | −6.1% |

Every other phase in the same samples is flat: `ffn` +0.20% and −0.23% in the two panels,
`sequence-input-projection` +0.10% and −0.33%, `gdn-resident-core` +0.54% and −0.10%,
`sequence-output-projection` +0.27% and −0.38%. The device span moves +0.17% and −0.38%.

**In full-model terms this is small and it is honest to say so.** With logits on every row the head
is 4.8% of a 128-row pass, so −3.6% of it is −0.17% of the pass; prompt ingestion reads one row of
logits, where the head is 0.7% of the pass and the change is −0.03%. It ships because it is free,
bit-identical, costs no register and no byte, and because the same header is now the one place this
engine's exact IU8 ternary map lives.

## What the head is waiting for

**A −13.4% cut of the block loop's work bought −3.6% of the phase. The conversion is 0.27.** That
is the number to carry, and the width ladder says the rest of it is not bytes either. Same panel,
same process, `--head-tt` against `--head-op`, 128 rows with logits on every row:

| token groups | image reads per pass | arm 0 | arm 1 | samples |
|---|---:|---:|---:|---:|
| `TT = 1` | 8 | 13.800 ms | 12.954 ms | 2 |
| **`TT = 2` (selected)** | 4 | 11.065 | **10.666** | 5 |
| `TT = 4` | 2 | 11.231 | 11.007 | 5 |

Halving the weight traffic from four reads to two buys **nothing** — +1.5% on arm 0 and +3.2% on
arm 1, both inside the spread of their own samples. Halving it again from eight reads to four buys
20%, so the head crosses out of its traffic bound somewhere between those two points and sits, at
the selected width, in a region where neither its bytes nor its issue slots explain most of its
time. The published ladder in [`docs/wide-head.md`](wide-head.md) had `TT = 4` at 13.160 ms against
11.204; on today's build the two widths are 11.2 and 11.1. **The selected width is still `TT = 2`
and this change does not move it** — the reversal that appeared in the first two-sample round
disappeared at five.

### The arm this prices, and why it is now worth building

[`kernels/halo_expand.hpp`](../kernels/halo_expand.hpp) carries a third arm nobody has measured: a
spread two-bit coordinate where `(w >> 2k) & 0x03030303` **is** operand dword `k`, 56 operations per
128-trit block against 168, for 34 stored bytes per (row, block) against 28. On the deployed image
that is +21.4% of the head's 278 MB.

The two numbers above price it for the first time. Arm 2 removes about 168 of arm 1's 433 work
instructions (−38.8%); at a conversion of 0.27 that is about −10% of the phase. The bytes it adds
land on an axis this panel has just measured as flat at the selected width: doubling the image reads
from two to four cost 1.5-3.2%, so +21.4% on one read is plausibly under 2%. **That is a predicted
win of roughly 8% of the head phase, and it was worth building before this panel only as a guess.**
What it needs is an image: `spread_pack` is in the header and runs on the host or the device, so the
builder is a kernel that reads the 896-byte HALO blocks the head already owns and writes 1088-byte
spread blocks into a separate 337 MB allocation inside `HeadBatch`, plus the `HALO_HEAD_OP=2`
instantiation that is already written. Note the trap the same header warns about: this is the trade
the A4 FFN lost 38.6% on at 32 rows, and it lost it on a phase that was priced in bytes. The head is
not, which is the whole reason the arm is worth a build here.

### Where the rest of the time is, and why the next lever is a numerical decision

The conversion is not a mystery about the block loop; it is about how little of the wave's life is
spent in it. Static census of the shipped arm, `head_tile<2>`: 1318 instructions, of which **783 are
outside the block loop**, with **8 `s_barrier`, 149 `ds_` operations and 32 `global_store`**. A wave
owns `PW = NB_D / NW = 5` K blocks, so its dynamic count is `783 + 5 x 535 = 3458` and **22.6% of it
is the per-workgroup prologue and drain**. Arm 0 is `783 + 5 x 619 = 3878`. The arm removes 10.8% of
a wave's dynamic instructions and the phase moves 2.76%, so the kernel is issuing about a quarter of
the time and waiting for the rest — behind eight barriers and a seven-wave LDS reduction that it
reaches after five blocks of work.

That makes the unit shape the next lever on this kernel: how many K blocks a wave owns between
drains. **It is not a schedule change.** `head_tile`'s eight-wave split exists to reproduce the
eight-row kernel's `fmaf` association exactly — wave `w` owns blocks `[5w, 5w+5)` and wave 0 adds
the seven partials in order — so widening a wave's ownership re-associates an FP32 sum and changes
logit bits. The lane now has the instrument that prices such a change: the 1024-prediction
teacher-forced horizon panel at ±0.008 nats that
[the four-bit sequence default](seq-out-a4-default.md) was decided on. Note also what does *not* buy
it back: more token columns per drain is the `TT` axis above, and it is flat.

## Raw

[`batch-comparison/head-operand-arm/`](../../../data/bonsai2/batch-comparison/head-operand-arm/README.md)
holds the three panels, the census listings and the exactness check. The operand identity itself is
proved on the CPU with no GPU by `make kernels/head_op_check`, which runs the deployed peel, the
gather and the spread map against `halo::decode_block` over 3 uniform, 256 byte-sweep and 4000
random blocks.

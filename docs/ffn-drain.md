# The A4 FFN is at 90% of its own block, and the last 10% is the drain

Source `6871cf6`. Raw under
[`batch-comparison/ffn-drain/`](../../../data/bonsai2/batch-comparison/ffn-drain/README.md).

The FFN is the largest phase in this engine on every shape — 136.4 ms of a 308.8 ms 256-row pass,
about half of a prompt pass once the four-bit input projection landed, and 29 to 36 ms of a batched
generation step. Every attack on it so far has been aimed at a particular instruction. Nobody had
priced the whole block against the instruction it exists to issue, so nobody knew how much was left.

**There is 27% left, all of it non-matrix issue, and the largest single item in it is the FP32
drain.** This document measures that, prices each item, and records one change built on a census
that turned out to be of a kernel the launcher never selects.

## The arithmetic ceiling and where the kernel sits against it

`v_wmma_i32_16x16x16_iu4` retires 4096 MACs in 16 cycles on one SIMD32
([the operand arithmetic](sequence-output-projection.md#the-arithmetic-that-makes-this-matter)), so
80 SIMD32 at the 2428 MHz [this box runs at](sequence-projection-operands.md#the-shader-clock-is-2428-mhz-not-2900)
is **49.7 T MAC/s**. A 256-row pass puts `3 x 5120 x 17408 x 64 x 256 = 4.381 T` MACs through the
three FFN matrices, and `ffn` takes 136.437 ms: **32.1 T MAC/s, 64.6% of the pipe.**

The block loop says why, and it agrees. Both stages of the deployed A4 route compile to a 128-block
of **64 matrix instructions and about 388 of everything else**:

| per 128-block, `IU4`/`PAIR`, `TT = 4`, `W = 2` shared | gate/up (`FUSE`, `ORD = 2`) | down (`ORD = 0`) |
|---|---:|---:|
| `v_wmma_i32_16x16x16_iu4` | 64 | 64 |
| `v_cvt_f32_i32` | 64 | 64 |
| `v_dual_mul_f32` + `v_dual_fmac_f32` (VOPD lines) | 27 + 27 | 27 + 28 |
| `v_mul_f32` + `v_fmac_f32` (unpaired) | 10 + 10 | 10 + 8 |
| `s_waitcnt` | 46 | 46 |
| `global_load_b64` (B fragments) | 32 | 32 |
| `v_perm_b32` + `v_and_b32` (pair-code expansion) | 32 + 30 | 32 + 30 |
| `ds_bpermute_b32` + `v_lshrrev_b32` (scale broadcast) | 16 + 16 | 16 + 16 |
| addressing, `s_clause`, `mov`, branch | 60 | 59 |
| **total** | **452** | **451** |

At 16 cycles a matrix instruction and one per slot for the rest, that block is
`64 x 16 + 388 = 1412` cycles with the matrix pipe busy for 1024 of them — **72.6%**. Measured
64.6% against the pipe is **89% of that model**, which for a kernel at six waves per SIMD32 is
about as close as a block census gets.

So the FFN's remaining headroom at prompt width is exactly the 388 non-matrix instructions, worth
**27% of the phase if every one of them vanished**, and it is not memory: the pass streams its
4.28 GB image in 136 ms, 31 GB/s against a 242 GB/s roof.

### What the 388 are, in order

| item | instructions | share of the block's cycles |
|---|---:|---:|
| **FP32 drain** (`cvt` + the scale multiply + the accumulate) | **192** | **13.6%** |
| B activation fragments (`global_load_b64`) | 32 | 2.3% |
| pair-code expansion | 62 | 4.4% |
| per-block scale broadcast (`ds_bpermute` + extract) | 32 | 2.3% |
| addressing, clauses, moves | ~60 | 4.2% |
| `s_waitcnt` | 46 | 3.3% |

**The drain is the FFN's second-largest consumer after the matrix pipe itself**, ahead of the
expansion that three iterations of this lane have attacked. It is one `v_cvt_f32_i32`, one multiply
and one `fmaf` per accumulator element per block — `ya[t][r] += float(acc) * (ws[r] * cs[t])` — and
at `TT = 4` with two matrices that is 64 elements a block, exactly one per matrix instruction,
because the block is eight K-slices deep and drains once.

Two of its three instructions are as cheap as they can be: the compiler already packs the multiply
and the accumulate into 27 VOPD pairs. **The 64 conversions cannot be packed** — `v_cvt_f32_i32` is
not a VOPD opcode on gfx11 — so 4.5% of the block is conversions that no scheduling can remove.

**The only way to shrink the drain is to drain less often or to remove one of its terms, and both
are numerical-map changes.** Draining once per 256 K instead of 128 needs one weight scale per two
blocks. Removing `cs[t]` from the inner expression — worth the 64 multiplies, 4.5% of the block —
needs one activation scale per row instead of one per 128-block, which is a quality question the
[horizon instrument](gdn-state-horizon.md) can answer without writing a kernel, exactly the way
`a4e` [priced the four-bit input projection](sequence-input-a4.md) before it existed.

## The change this built, and why it measured nothing

`run_down` passed `ORD_SLICE` as a literal where `run_gate_up` passes `p->a4_order`, so the down
projection has never been able to take [the operand-order interchange](ffn-order.md) that gate/up
has selected since it was written. The census that motivated giving it one read the down kernel as
**547 instructions with 64 unpaired `v_mul_f32`, 64 unpaired `v_fmac_f32` and 63 `s_delay_alu`** —
95 instructions and a stall-hint storm worse than gate/up for identical matrix work, and exactly
the disease [the dot4 body](mv-dot4-body.md) cured by splitting an accumulator chain.

It is a real kernel. **It is not the one the down stage launches.**

`shape_for` chooses wave ownership at run time. At 128 rows the down stage has `ntiles = 8` and
`cap = 4`, so `w = 2`: `SHARE = true`, two waves per row tile, `WV = 2`, `TT = 4`. The 547-instruction
block is `SHARE = false, WV = 4`, which that width never selects. The instantiation it does select
is **451 instructions and already pairs its drain into 28 + 27 VOPD lines with six `s_delay_alu` in
the whole block** — one instruction away from the gate/up block. There was nothing for `ORD` to buy.

The panel says the same thing. Mode 19, 128 rows, heads off, traced, both orders interleaved in one
process, two rounds:

| phase | round 0, `ORD 0` → `2` | round 1, `ORD 0` → `2` |
|---|---|---|
| `ffn` | 79.128 → 78.170 ms, **-1.21%** | 67.764 → 68.839 ms, **+1.59%** |
| `sequence-input-projection` (control) | -1.13% | +1.37% |
| `sequence-output-projection` (control) | -1.00% | +1.31% |
| `gdn-resident-core` (control) | -1.16% | +1.26% |
| `sequence-core` (control) | -0.42% | +1.64% |
| device span | 176.62 → 174.68 | 153.09 → 155.31 |

Normalised by the four phases the change cannot reach, that is **-0.16% and +0.24%** — opposite
signs, both an order of magnitude inside a round-to-round drift that moves every phase together by
15%. Residual FNV-64 is `7446865760224376151` in every sample of both arms, which is the published
128-row value, so the interchange is the bit-identical change it claims to be and it is worth
nothing here.

**The change is not in the tree.** It costs nine kernel instantiations and buys a null;
`batch-comparison/ffn-drain/dn-order.patch` reproduces it.

## The lesson, which is about measurement rather than about the FFN

**An instruction census prices an instantiation, and `shape_for` decides which one runs.** This
lane has learned three times that a slot count does not predict a sign
([the address-form table](ffn-operand-address.md), [the scheduling-slot split](activation-scale-axis.md),
[the four-bit block that is longer and faster](sequence-input-a4.md)). This is a fourth failure and
it is upstream of all of them: the census was of a kernel the engine does not launch, so it could
not have predicted anything.

`k_proj_opt` carries `SHARE`, `WV`, `TT`, `FUSE`, `ORD`, `DLS` and `MATS`, and the same template at
`SHARE = false, WV = 4` compiles to **547 instructions where `SHARE = true, WV = 2` compiles to
451** — a 21% difference in issue for identical matrix work, from the ownership choice alone. Before
you price a block, print the shape the launcher picks at the width you are measuring:
`HALO_FFN_GRID=1` prints the grid, and `shape_for` is fifteen lines.

That difference is itself worth checking. `w = 1` — the heavy block — is selected whenever
`ntiles <= cap` and the stage's row tiles already fill the device, which at `cap = 4` means
**passes of 64 rows or fewer on the pair arm**. Prompt ingestion defaults to 256 rows and a
generation step takes the dense arm, so nothing this engine serves lands there today; a route that
did would pay 21% more issue than it needs to.

## What is left here, priced

- **The drain, 13.6% of the block.** Irreducible at the current numerical contract, 4.5% removable
  with a per-row activation scale, which needs a quality panel and not a kernel.
- **The scale broadcast, 2.3%.** 16 `ds_bpermute` per block, on a chain rooted in one two-byte load.
  [The handoff](../orchestration/HANDOFF.md) proposes a `[half][r]` scales image for it and says it
  needs a second image because `FFN_IMAGE_SCALES` is shared with the control modes. It does not: a
  lane that loads all sixteen scales of the group as eight dwords — 32 bytes, the same 32 bytes
  every lane wants, one line — already holds scale `2r` in the low half of dword `r` and `2r + 1`
  in the high half, so the shuffle becomes eight field extracts against the same image. The
  replacement is about as many instructions as it removes; what it removes is the LDS round trip
  and its place in a 46-deep wait budget. Unbuilt.
- **`s_waitcnt`, 3.3%.** 46 per block, which is more than the expansion costs.
- **Nothing here is worth more than 2% of a prompt pass on its own.** The FFN is not where the next
  large prompt-side number is; it is where the next large *decode*-side number is, and that is the
  dense arm at `<= 32` rows, which is a different block, a different image and
  [a different diagnosis](ffn-dense-loads.md).

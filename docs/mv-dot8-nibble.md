# The deployed matvec's operand can be nibbles, and it buys 1%

`mv_rows_t` is the body of every matvec the persistent kernel runs: single-token decode, the
eight-row drafted verify pass that is 80% of a served generation step, and eight-row prompt
ingestion. Per 128-K block it spends about 254 issue slots lifting five trits out of a byte and
laying them out as IU8 operand bytes, against 32 `v_dot4_i32_iu8` per row.
[The dot4 body result](mv-dot4-body.md) priced that pass at 34.1 ms of issue against 26.5 ms of
weight bytes and measured 49.8, and named the expansion as the largest single item left in it.

This replaces the operand with ternary **nibbles** for `v_dot8_i32_iu4`. It is exact, it removes
**10.5% of the kernel's instructions**, and on the two shapes the deployed engine runs it is worth
about **1%** — which the second activation pane it requires gives back in prep. The runtime keeps
the byte operand. What stayed in the tree is the operand map (`kernels/mv_nibble.hpp`) and its host
acceptance (`kernels/mv_nibble_check.cpp`), because the map is right and only its consumer is
wrong.

## The map

Three facts compose, and the first two were already in this tree.

**A five-trit byte peels two trits at a time.** `(b * 9) >> 8` is `3*t0 + t1`, re-multiplying the
low byte gives `3*t2 + t3`, and a final `* 3` gives `t4`. The A4 FFN found this
([`kernels/ffn_a4_operands.hpp`](../kernels/ffn_a4_operands.hpp)) and calls the nine-valued result a
code. A code is a whole operand **byte** when the operand is nibbles, so one `v_perm_b32` over a
constant table turns four codes into eight ternary nibbles. As IU8 bytes the same four codes need
two table lookups and two more permutes to interleave, which is where the 254 slots go.

**The HALO element order is already the right one.** `src/halo_format.h` fixes trit order by the
peel: for byte pair `p = 2d + w` of qs dword `d`,

```
e = 8p + {0,1,2,3} <- t0(first), t1(first), t0(second), t1(second)
e = 8p + {4,5,6,7} <- t2(first), t3(first), t2(second), t3(second)
```

so the four codes of one byte pair — `A(first) A(second) B(first) B(second)` — are exactly the four
element pairs of one eight-element group. That is one `dot8` operand, in order, with nibble `j` of
operand dword `i` holding element `8i + j`. Nothing about the stored format moves.

**Eight-bit activations keep the full MAC rate.** IU4 wants four-bit activations and this path has
eight-bit ones, so split them: `a = 16*a_hi + a_lo`, low nibble read unsigned, high nibble read
signed, and

```
sum_k w_k a_k = 16 * sum_k w_k a_hi_k + sum_k w_k a_lo_k
```

Two `dot8` per eight elements retires the same four useful MACs per instruction one `dot4` per four
elements does. Every term is an exact `int32`, so the block's integer sum — and the float that
reaches `fmaf` — is bit for bit the one the byte operand produces. The two halves are independent
accumulator chains by construction, which is the split [the dot4 body](mv-dot4-body.md) had to
introduce by hand.

### The relabel that makes `v_perm_b32` free

A nine-valued code needs nine table entries and `v_perm_b32` offers eight dynamic bytes for
selectors 0..7. Selector 8 returns the sign replication of source byte 1, which is `0x00` when that
byte's top bit is clear. Code 8 is the trit pair `(2,2)`, so store the operand under **`v = 2 - t`**:
code 8's byte becomes `(0,0) = 0x00`, every other byte is at most `0x22`, and no byte in the table
can turn selector 8 into `0xFF`. The dot then computes

```
V = sum_k (2 - t_k) a_k = 2 * xsum - sum_k t_k a_k
```

so the deployed accumulator `sum t a - xsum` is `xsum - V`: the same integer, one subtraction, in
the other direction. The A4 FFN needed the same ninth-code trick and solved it with a different
relabel, because its alphabet is signed; this one is the ternary path's own.

`kernels/mv_nibble_check.cpp` runs the header's own source on the host — the host arm of `nib_perm`
reproduces the gfx1151 selector rule — against `src/halo_format.h`'s encode/decode: operand order
over 4096 random blocks, all 243 packed byte values including the `(2,2)` pair, and the accumulator
identity against `sum (t-1) a` over 4096 random activation blocks. `make kernels/mv_nibble_check`,
no GPU, under a second.

## What it costs and what it buys

Instruction census, `kernels/mv_body_isa.hip` at PW=5, TT=8, ACC=4, `tools/isa_loop_count.py`:

| | expansion block | whole kernel, work slots |
|---|---:|---:|
| IU8 bytes | 254 | 771 |
| IU4 nibbles | **156** | **690** |

The expansion falls 39%; the kernel falls 10.5%, because the nibble arm gives back about 16 slots
folding its two accumulator halves. `v_dot8_i32_iu4` exists on gfx1151 (`llvm-mc`, encoding
`0xcc18`) and the built engine emits 15,680 of them.

Registers do not move where it matters. `k_forward_rows<8>` is **135 VGPR and 10 waves per SIMD32
in both arms** — halving the operand array frees nothing, which says the eight-row kernel's peak is
set somewhere other than the matvec operands. That is a fact worth more than this result: the
[grid-occupancy](../orchestration/HANDOFF.md) search for the seven registers that would buy twelve
waves should not look here.

At `TT = 1` the nibble arm costs occupancy: 98 VGPR against 95, which is the wrong side of the
96-register step, and sixteen waves per SIMD32 become twelve. Single-token decode runs its matvecs
at 240 GB/s of a 242 GB/s roof and has nothing to win, so the arm is gated to `TT > 1` and the
single-token kernel stays untouched code.

### Measured, both shapes, one lock hold per arm

Eight-row prompt ingestion, a 1463-token document through the deployed route, interleaved in both
orders:

| order | byte operand | nibble operand |
|---|---:|---:|
| byte first | 173.6 tok/s | 173.9 |
| nibble first | 171.1 | **174.3** |

Drafted generation, six prompts, 64 tokens each, `--draft-weights q4`: the per-step verify pass is
**46.05 ms with bytes and 45.57 with nibbles**, and generation is inside its own noise — the
drafter's candidate selection uses atomics, so drafted and accepted counts move a few percent
between identical runs and generation tok/s moves with them.

**Bit-identical, measured:** all six greedy digests equal across the arms, 384 generated tokens per
arm. The identity is over `int32` and needs no quality instrument.

So a 10.5% instruction cut is worth about 1% of the pass, in both shapes, in the same direction.

## Why it does not ship

The operand needs the activations as nibbles, which means a second pane in the quantised activation
row: `prep_chunk_r` writes the IU8 codes and then the same activations as IU4 nibble pairs, and
every consumer's row pitch doubles. That pane is not free. The prep phases are 4.7% of an eight-row
pass, the pane adds four ALU operations and two stores per four elements to them, and the estimate
lands within a rounding error of the 1% the matvec gains. It is also a coupling tax that is paid
forever: three separate places compute an `xq` row pitch, and the one in
[`kernels/q4_tiles.hpp`](../kernels/q4_tiles.hpp) that this experiment missed produced a GPU memory
fault rather than a wrong number. The next engineer to add an activation consumer would step on the
same contract.

A 1% gain at the edge of this box's resolution does not earn that. The runtime keeps the byte
operand and the map keeps its test.

## What this measurement is actually worth

**The eight-row pass is not issue-bound at the margin, and no roof in this handoff predicts it.**
It reads 5.9 GB of weights in 46.35 ms — **127 GB/s of a 242 GB/s roof** — while sitting at 74% of
its own issue model. Removing 10.5% of the issue moved it 1%, so the binding resource is neither
instructions nor bandwidth: it is **memory latency at ten waves per SIMD32**, with each wave one
block ahead of its weight stream.

That is the third independent measurement in this kernel saying the same thing, and the first one
large enough to settle it. The two-trit palette cut the expansion 12% and measured null; a peer cut
9.4% of the A4 FFN's block slots and its step did not move; this cut 10.5% and got 1%. Anyone
pricing a change to `mv_rows_t` from an instruction census should multiply by about a tenth, not by
[the third the FFN converts at](ffn-dense-loads.md).

The levers that remain on this pass are the ones that change outstanding memory requests per SIMD:
more waves, or more weight blocks in flight per wave. The first is the grid work; the second is
measured and rejected twice ([phase length](matvec-phase-length.md),
[down waves](ffn-down-waves.md)) — but both of those rejected it at unchanged occupancy, and a
deeper cursor has never been tried on a body whose operand array is half the size, which this map
makes possible for free.

## Where the map would pay

An issue-bound consumer of ternary HALO tiles, if one appears. The one in this engine is the
[vocabulary head](wide-head.md), which reads its 278 MB weight image at 24 GB/s and is explicitly
issue-bound — but it takes WMMA fragments rather than `mv_rows_t`'s scalar-fed operand, so the map
would have to be rebuilt against `head_batch.hip`'s layout. `kernels/mv_nibble.hpp` is the map and
`kernels/mv_nibble_check.cpp` is the proof that its element order and its accumulator identity are
right; what a new consumer needs is its own activation coordinate, not another look at the algebra.

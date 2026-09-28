# What lookahead is worth in the pair-code block loop, and what it costs to find out

The A4 FFN's pair-code arm runs every prompt pass of 64 rows or more: gate/up at four token tiles
under the interchanged order, the down projection at two under the deployed one. At 128 rows it is
about 83 ms of a 183 ms pass. [The dense arm's block cursor](decode-dispatch.md) had just bought
28% on the ≤32-row down stage by keeping its weight stream in flight, and
[`ffn-schedule.md`](ffn-schedule.md) recorded the obvious next question: *at four token tiles on the
pair map the trade is still untested.*

It is tested now. **It loses, at both depths, and the reason is worth more than the cursor would
have been.** This document is that negative, the latent defect the experiment exposed in the cursor
that did win, and a conversion rate between issue slots and time that the next person in this
kernel can price against.

## The hole is real and the pair arm does not care

The deployed block issues its four `global_load_b128` operand loads and waits on them inside the
same block:

```
s_clause 0x3 ; global_load_b128 x4 ; ... ; s_waitcnt vmcnt(32) ; <80 slots of expansion>
```

`PairStream` (kept with this document's data, not in the tree) holds DEPTH blocks of pair codes in
registers and hands out the oldest, exactly as `ffn_dense_stream.hpp` does for the five-trit image.
Mode 19, 128 rows, logits off, three depths interleaved in one process, two rounds each, untouched
phases as the in-panel control:

| depth | VGPR | scratch | `ffn` ms | controls ms | `ffn` / controls | vs deployed |
|---:|---:|---:|---:|---:|---:|---:|
| 0, deployed | 256 | 28 B | 78.99, 80.86 | 88.37, 90.30 | 0.8938, 0.8955 | — |
| 1 | 247 | 0 | 83.65, 91.07 | 87.56, 94.80 | 0.9553, 0.9607 | **+7.2%** |
| 2 | 256 | 116 B | 106.84, 103.72 | 89.35, 87.61 | 1.1957, 1.1839 | **+33%** |

Residual FNV-1a `7446865760224376151` in all six samples, canonical main's value, so the arms are
the same arithmetic.

**Depth 1 pays for itself in registers and still loses.** It removes nine VGPRs and the block's
28-byte spill, the waits fall from 39 to 29, and the loads are issued 141 instructions before the
first one is needed instead of 5 — the mechanism did what it was built to do. What it costs is
visible in the listing: the block goes 493 to 597 instructions, 83 of them `s_delay_alu` against 4,
the drain's VOPD packing collapses from 28 dual lines to 12, and handing out a held block is 19
register moves the deployed arm never pays. Depth 2 spills 116 bytes into the block loop and loses
a third of the stage.

The transferable part is the comparison with the arm where the same cursor won. The dense cursor
bought 28% on a stage the grid holds to **two** waves per SIMD32, and 1.2% on its 1088-wave gate/up
stage. The pair arm at 128 rows launches 2176 waves and keeps five or six resident per SIMD32.

> Lookahead is worth its registers where the grid cannot fill the machine, and only there. Before
> spending registers on a cursor, count the stage's resident waves, not its stalls.

## The cursor that won reads past the end of its image

Both cursors clamp the wrong thing. `DenseStream` advances on the caller's "is there another block"
flag, so the last DEPTH iterations of a tile still request blocks `nb` and `nb+1`. Inside an image
that is the next tile's first block, harmless and unused; for the **last** row tile it is past the
allocation.

This is not hypothetical. The pair version faulted on the first run:

```
Memory access fault by GPU node-1 ... on address 0x76981e600000. Reason: Page not present
```

The pair image is `512 * nb * tiles`, an exact multiple of the page size, so its end is a page
boundary. The dense image is `416 * nb * tiles` and survives on the slack of its last partial page.

The fix is in the allocation rather than the loop, because the over-read values are never consumed
— only their addresses have to be mapped. Every weight image now carries `CURSOR_SLACK_BLOCKS = 2`
blocks past its last tile, under a kilobyte per image. The alternative, clamping the block index
inside the cursor, was measured first and costs eight instructions in a 587-instruction block loop
that two engineers are currently using as a baseline.

## Issue slots convert at about a third, and only downward

Three of us built the same operand-addressing split within an hour; the one that landed is
[`ffn-operand-address.md`](ffn-operand-address.md) (`7ea7bf2`), which goes further than the version
measured here by also giving the scale runs and a block-major `cscale` a uniform base. This
iteration's pair-only version is superseded by it and is not in the tree. Its independent panel is
worth one line as corroboration: on a build where only the pair arm's B fragments moved, nine
rounds interleaved in one process gave 753.9 to 770.1 prompt tok/s, **+2.15% with 8 of 9 paired
rounds favourable**, and 39,731,200 full-vocabulary logits at 32 and 128 prefill rows compared bit
for bit with zero differences.

Read together with the cursor, the two arms of this iteration calibrate the kernel in both
directions on one build:

| change | block slots | measured `ffn` phase |
|---|---:|---:|
| cursor depth 1 (added slots) | +6.5% | +7.2% |
| B address split (removed slots) | -8.6% | -1.6% raw, -3.0% normalised |

**Slot removal converts at roughly a third; slot addition at close to one.** The asymmetry is the
same fact [`clock-power.md`](clock-power.md) measured from the clock side: a 128-row pass is 14%
clock-elastic at the operating clock and its `ffn` phase 30%. Removing an instruction gives back
the fraction of its slot that was on the critical path; adding one on a dependent chain costs the
whole slot plus the schedule around it.

So the ceiling is priced. Deleting *every* non-matrix slot from both pair stages — drain, operand
expansion, addressing, waits — is 29-38% of their issue and about 9-11% of the phase at that
conversion. The two largest items will not go:

- **The drain is structural.** One `v_cvt_f32_i32` and one scale product per accumulator value per
  128-block, with the weight scale indexed by row and the activation scale by token. It factorises
  no further, and it cannot be deferred across blocks without changing the rounding, because both
  scales change every block. Its magic-constant replacement is dead on precision: the bias must be
  2^22 or larger for the mantissa to carry an integer, while the partial sums are under 2^10, so
  subtracting it back cancels thirteen bits.
- **The expansion cannot lose its mask.** Five operations per sixteen weights, and the device's own
  [measured selector table](../kernels/ffn_a4_operands.hpp) forbids the obvious cut: `v_perm_b32`
  returns `0xFF` for every selector byte of 13 and above, so a nine-valued code sharing a byte with
  its neighbour must be masked before it can be a selector.

Nor can the matrix half shrink. Packing two ternary weights into one IU4 A slot would need the two
outputs' partial sums separable out of one int32 accumulator, which needs `|y| < radix/2`; a K16
slice against four-bit activations reaches 112, and the slot holds 4 bits.

## What is left, and where to look next

The remaining time is neither issue nor DRAM bandwidth (4.28 GB per pass at about 59 GB/s against a
242 GB/s roof), it does not respond to the shader clock, and this iteration rules out a per-block
memory round trip as well.

The obvious next suspect is **line sharing between the two waves of a workgroup**: at 128 rows a
workgroup is two waves on the same row tile differing only in token group, so the weight image is
fetched once per pass if they stay close and twice if they drift, and nobody has measured which.
Before spending a counter panel on it, note that the stage sweep in
[`ffn-schedule.md`](ffn-schedule.md) already bounds what the answer can be worth. `W = 4, TT = 2`
runs **four** token groups against `W = 2, TT = 4`'s two — double the walks over the same weight
stream — and measures 85.21 ms against 82.00. Doubling the weight reads on this stage costs about
4%, so even a complete failure of L1 sharing is a few percent, not the missing half.

That makes the weight side a small target and the activation side the one with room: a 128-row
gate/up pass issues 32 B-fragment loads per block against four weight loads, 44.6 GB of L1/L2
traffic per pass against 2.85 GB from DRAM. `BONSAI_PROFILE_COUNTERS=L2CacheHit` on one
`k_proj_opt` dispatch is still the cheapest instrument, but point it at the B image, not the
weights.

## Reproduce

```sh
# the cursor arms: apply the patch, then
tools/run-batch-compare --profile-tool --modes 19 --rows 128 --heads 0 --traces 1 \
    --rounds 2 --ffn-cursor 0,1,2 --out DIR/cursor.json
# the census that chose and then explained both experiments, no GPU needed
hipcc --offload-arch=gfx1151 --cuda-device-only -S -O3 -Isrc -Ikernels kernels/ffn_batch.hip -o ffn.s
tools/isa_loop_count.py ffn.s "k_proj_optILi1ELi1ELi1ELb1ELi4ELi2ELb1ELi2E"
```

`../../data/bonsai2/batch-comparison/ffn-pair-cursor/` holds the three-depth phase panel, the
two full-model panels and the logit dumps of the superseded addressing arm, the rejected cursor as
a patch (`pair-cursor.patch`, against `27cafb1`) with its header, and its compiled listing.

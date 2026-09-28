# The FFN a generation step runs

Every published measurement of this engine's batched FFN is prefill-shaped, and every recent
improvement to it moved the 128-row point. A generation step is 32 rows however many sequences it
advances, and at 32 rows mode 8 takes a different kernel entirely: the dense five-trit image under
tile ownership, which none of the row-tile ownership work touched and which measured an exact null
in aggregate generation TPS. This document is that arm — what it costs, what its cost is made of,
and which of the obvious repairs lose.

## Ordering the two arms on one build

`ffn_batch_set_a4_image()` forces mode 8 onto the dense five-trit image or the pair-code image
between calls, and `--ffn-image` is a case axis in `tools/batch_profile` beside `--ffn-sched`. Both
images decode through the same nine-valued code alphabet to the same weights, so the axis is
bit-identical and the panel checks that rather than assuming it.

Mode 19, one build, device trace, whole-FFN region, medians of two rounds:

| rows | image | ownership | FFN ms | pass ms |
|---:|---|---|---:|---:|
| 32 | dense five-trit | tile | **33.77** | 72.93 |
| 32 | pair code | tile | 71.38 | 109.52 |
| 32 | pair code | adapt | 61.96 | 99.82 |
| 32 | pair code | wide | 50.51 | 88.99 |
| 128 | dense five-trit | tile | 116.32 | 225.42 |
| 128 | dense five-trit | wide | 108.59 | 209.97 |
| 128 | pair code | wide | **89.49** | 196.56 |

The residual FNV-1a over every FP32 output is `1873717996769712938` for all six 32-row cells and
`7446865760224376151` for all six 128-row cells, so nothing in the table is a numerical difference.

Raw: `decode-ffn/arm-cross-r32-r128.json`.

**The rule is right at both ends and wrong in the middle.** A peer measured the cell neither panel
had, 32 streams advancing two tokens each, which is 64 rows: dense 57.272 ms of FFN against pair
68.780, with the three phases the swap does not touch moving 0.55-1.1% as an in-process control
(`decode-step/k2-image.json`). `Npad <= 32` selected pair codes there and gave away 11.5 ms.

## What a 32-row block costs

ISA census of the selected instantiations, gfx1151, per one 128-K block of one wave, counted as
issue slots. `k_proj_opt<OP_IU4, WM_DENSE, LAYOUT_LANE, share=false, TT=2, WV=4, fuse>` is the
gate/up stage and the same without fuse is the down stage.

| | slots | wmma | VGPR | cycles/block |
|---|---:|---:|---:|---:|
| dense TT=2 gate/up, before | 895 | 32 | 200 | 1375 |
| dense TT=2 down, before | 826 | 32 | 198 | 1306 |
| pair TT=2 gate/up | 595 | 32 | 146 | 1075 |
| pair TT=1 W=2 down | 353 | 16 | 110 | 593 |

Cycles count a `v_wmma_i32_16x16x16_iu4` as 16 and everything else as one. Inside the dense gate/up
block: expansion 321, drain 122, SALU 84, waits 83, B addressing 52, movs 39, VMEM 42, LDS 16 for
the scale broadcasts, wmma 32.

The work counts are 1088 row tiles x 40 blocks = 43,520 wave-blocks per layer for gate/up and
`DH/16` = 160 tiles x 136 blocks = 21,760 for down, because the down projection runs its two output
halves as the two matrices of one wave. So the whole-FFN issue model at 32 rows is
43,520 x 1375 + 21,760 x 1306 = 8.82e7 cycles per layer, **24.3 ms over 64 layers** at 80 SIMD32 and
2.9 GHz, against 33.8 ms measured.

Two numbers from that are worth carrying:

- **The matrix instructions are 9.2 ms.** At 32 rows only 27% of the FFN is matrix work; the rest is
  operand build, drain, addressing and stall. Prefill techniques that assume the matrix pipe is the
  scarce resource do not transfer here.
- **The two stages are not limited by the same thing.** Gate/up launches 1088 waves and fills the
  device, so its gap is register occupancy. Down launches 160 waves onto 80 SIMD32, two per SIMD32,
  and measures about 3.5x its issue model: that one is latency with nothing to hide it behind.

## The two stages do not want the same image

`run_opt` forced one weight image and one ownership on both projections, so the choice looked like
a single crossover in the row count. It is two, and they run in opposite directions, because what
decides an arm at a given width is **how many times its stage walks its weight stream**.

- The dense map's accumulator budget caps it at two token tiles. Above 32 rows a stage reads its
  weight stream `ceil(ntiles/2)` times where the pair map, capped at four, reads it
  `ceil(ntiles/4)` times. At 64 rows that is twice against once.
- The pair arm's down stage is under the wide-wave target by then: `ceil(640/160) = 4` waves on one
  row tile at one token tile, which reads the stream four times, because 160 row tiles leave the
  device two thirds empty and the rule buys waves with duplicated work. The dense arm never takes
  that trade, since duplicating 321 slots of five-trit decode costs more than the waves return.

So at 64 rows gate/up wants pair codes and down wants the dense byte. `ffn_batch_set_a4_image()`
takes 3 and 4 for the two mixed arms, and the rule now has one row threshold per stage.

Mode 19, one build, one process, device trace, both samples of two rounds:

| rows | gate/up + down | FFN ms | pass ms |
|---:|---|---|---:|
| 32 | dense + dense (selected) | 32.84, 34.00 | 69.18, 70.21 |
| 32 | pair + pair | 49.39, 50.03 | 84.83, 84.86 |
| 64 | **pair + dense (selected)** | **51.88, 52.13** | **111.68, 116.90** |
| 64 | dense + dense | 53.14, 53.28 | 112.33, 112.63 |
| 64 | pair + pair (what shipped) | 62.58, 62.82 | 118.97, 120.51 |
| 128 | pair + pair (selected) | 83.70, 84.59 | 182.42, 182.91 |
| 128 | dense + dense | 99.10, 99.80 | 198.40, 200.26 |

At 64 rows the selected arm takes **16.9% off the FFN region and 6.0% off the pass**. 32 and 128
keep the arm they had, and their rows are the null that says so. The residual FNV-1a is
`1873717996769712938`, `14426824474163741251` and `7446865760224376151` at 32, 64 and 128 rows, one
value per row count across every arm, so none of this is a numerical choice. Raw:
`decode-ffn/rule-accept.json`.

Full model, 384-token document, mode 19, the two rules interleaved in one process:

| workload | pair + pair above 32 rows | per-stage rule | change |
|---|---:|---:|---:|
| prompt processing, 64 rows/pass | 507.7, 497.3 tok/s | **534.9, 539.9** | **+6.9%** |
| prompt processing, 128 rows/pass | 694.6, 657.3 | 680.0, 658.6 | same cell, null |
| generation, 32 streams | — | 265.1, 268.0 | 268.7 documented, null |

A generation step of one token per sequence is 32 rows and keeps the arm it had, so aggregate
batched generation is a measured null. The gain is in 33-to-64-row passes: prompt processing at
that width, and any step that advances more than one token per sequence.

## Two cuts to the operand build, and an honest null

Both are bit-identical: the same values in the same order, spelled in fewer instructions.

**The peel's shifts were never needed.** `peel_pair` extracts three radix-3 codes from a byte pair
by multiplying in 16-bit lanes, and each code lands in the high byte of its lane. It then shifted
each product right by eight to move the code into the low byte, because that is where the next
`v_perm_b32` selector looked. The selector can read byte 1 and byte 3 instead. All 76
`v_pk_lshrrev_b16` per block disappear.

**The dense branch had no scheduling barrier where the pair branch has four**, so the compiler
hoisted a whole block's peel above the first matrix instruction and kept five peeled words per
source dword live at once. Adding slice-level barriers costs 22 slots and returns 25 registers.

The first attempt at that put a barrier between the block's two load phases as well, which fences
the loads rather than the peel and costs a full memory latency mid-block on a stage running two
waves per SIMD32: **37.33 ms against 33.8 at 32 rows, a 10% regression**, caught because the
untouched pair arm reproduced at 50.45 against 50.51 and ruled out drift. Issuing all six of a
block's weight requests before any peel and fencing only the arithmetic is what the table below
measures.

| | slots | VGPR | cycles/block |
|---|---:|---:|---:|
| gate/up, before | 895 | 200 | 1375 |
| gate/up, shifts removed | 806 | 200 | 1286 |
| gate/up, selected | **809** | **175** | **1289** |
| down, before | 826 | 198 | 1306 |
| down, selected | **750** | **173** | **1230** |

**The wall clock does not resolve it.** At 32 rows the reshaped arm measures 32.84 and 34.00 ms
against 33.76, 33.77 and 35.33 for the identical configuration on the previous build: 10% fewer
issue slots and 12% fewer registers land inside the panel's own spread. The 32-row dense arm is
further from its issue model than the model suggests, which means its remaining time is stall, and
`docs/ffn-decode-shape.md`'s next question is which stall. The change is kept because it is
strictly cheaper on every ISA axis at identical output bits, and because the shapes it helps most
are the ones the per-stage rule now sends to the dense arm at 64 rows; it is not claimed as a
measured speedup.

## Reproduce

```sh
make -j8 tools/batch_profile tools/batch_compare
# the per-stage arms, ordered on one build
tools/run-batch-compare --profile-tool --modes 19 --rows 32,64,128 --heads 0 --traces 0,1 \
  --ffn-image 0,1,2 --rounds 2 \
  --out ../../data/bonsai2/batch-comparison/decode-ffn/rule-accept.json
# the full-model shape the change targets, both rules in one process
tools/run-batch-compare --modes 19 --only prefill --prefill-rows 64 --ffn-image 0,2 --rounds 2 \
  --out ../../data/bonsai2/batch-comparison/decode-ffn/fullmodel-prefill64
```

`--ffn-image` takes 0 for the rule, 1 dense on both stages, 2 pair on both, 3 dense gate/up with
pair down and 4 the reverse.

`HALO_FFN_A4_IMAGE` pins the image for a process that cannot use the setter. The ISA census is
`hipcc --offload-arch=gfx1151 --cuda-device-only -S -O3 -Isrc -Ikernels kernels/ffn_batch.hip`
followed by an instruction count over the innermost backward branch of each instantiation.

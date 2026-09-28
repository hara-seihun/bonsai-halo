# Wave slots are not what the A4 FFN is short of

The batched A4 FFN takes about 85 ms of a 192 ms 128-row prefill pass and issues roughly
5.35e8 IU4 matrix instructions doing it. Those instructions account for about 40 ms. The
remaining 45 ms has been attributed, in this repository and by three engineers working the
kernel on September 21, to memory latency that the gate/up stage cannot hide because it
holds only five waves per SIMD32.

That attribution is wrong. This document is the experiment that rules it out.

## What the block loop actually issues

`hipcc --offload-arch=gfx1151 -O3 --cuda-device-only -S`, symbol
`k_proj_opt<OP_IU4, WM_PAIR, LAYOUT_LANE, SHARE, TT=4, WV=2, FUSE>`, which is the gate/up
stage of a 128-row A4 pass. The inner 128-K block loop is 508 instructions:

| what | count | issue slots |
|---|---:|---:|
| `v_wmma_i32_16x16x16_iu4` | 64 | 1024 |
| `v_cvt_f32_i32` | 64 | 64 |
| `v_perm_b32`, `v_and_b32`, `v_lshrrev_b32` (operand expansion) | 80 | 80 |
| 64 multiplies, VOPD-packed | 36 | 36 |
| 64 fused multiply-adds, VOPD-packed | 34 | 34 |
| `ds_bpermute_b32` (block-scale broadcast) | 16 | 16 |
| `v_add_co_u32` / `v_add_co_ci_u32` (B addressing) | 32 | 32 |
| `global_load_b64` B, `b128` weights, `b32` cscale, `d16` scales | 42 | 42 |
| `v_mov_b32` (accumulator zeroing) | 8 | 4 |

Two things in that table correct estimates the handoff carried. The accumulator zeroing is
already free: the compiler materialises one zero octet and feeds it as `src2` to all eight
first-slice matrix instructions, each writing a different destination, so there are eight
moves per block rather than `16 * TT`. And the epilogue's multiplies and fused multiply-adds
are already dual-issued, so the drain costs about 134 slots rather than 192, of which 64 are
the integer-to-float conversions that VOPD has no form for.

At 16 cycles per IU4 matrix instruction on a SIMD32 and 2.9 GHz, a gate/up block is 1302
cycles and a down block (TT = 2) is 694. Both stages run 87,040 wave-blocks per layer over
64 layers, on 80 SIMD32 units. That is **31.3 ms of issue for gate/up and 16.7 ms for down,
48 ms against the 85 ms the stage measures**. Counting the scalar and wait slots as well,
as a peer's independent census did, raises the model to 55.5 ms. Either way about half the
stage is not instruction issue.

## The change that bought a wave slot

Two operand-addressing cuts, both value-preserving:

- **Half-major block scales.** The WMMA accumulator gives lane `L` the weight rows
  `2r + (L>>4)`, so the eight block scales a lane's epilogue needs are eight of the sixteen
  a tile stores, stride two. Row-major storage made that a broadcast: every lane loaded the
  scale of row `col`, and sixteen `ds_bpermute` redistributed them, which also pinned eight
  VGPRs of lane selectors for the whole kernel. Storing the eight even rows then the eight
  odd ones puts a lane's eight at `half * 8` as one contiguous 16-byte run. The A operand
  still wants the scale of row `col`, which moves to `scale_slot(col)`.
- **Block-major `cscale`.** `cscale[token][block]` makes `cscale[(tg + t*16 + col) * nb + blk]`
  a sixteen-cache-line gather, because consecutive lanes are `nb * 4` bytes apart. Transposed
  to `cscale[block][token]` it is one 64-byte line, and the token tile becomes an immediate
  offset instead of a fourth independent 64-bit address.

`tools/kernel_resources.py` reports what that does to the register file, straight out of
`-Rpass-analysis=kernel-resource-usage`, with no GPU:

| selected shape | main `16c5d16` | with both cuts |
|---|---|---|
| gate/up `<IU4, PAIR, LANE, SHARE, 4, 2, FUSE>` | 252 VGPR, **5 waves/SIMD32** | 233 VGPR, **6 waves** |
| down `<IU4, PAIR, LANE, SHARE, 2, 4, !FUSE>` | 147 VGPR, **9 waves** | 138 VGPR, **10 waves** |

Neither cut crosses the gate/up cliff alone: scales alone give 244 VGPR and `cscale` alone
245, both still five waves. Together they reach 233 and six.

## It buys nothing

**Phase instrument.** `tools/batch_profile` through the measurement runner, mode 19, 128
rows, head off, traces on, eight rounds, two builds. The five phases of the same pass that
the change cannot touch (`sequence-input-projection`, `gdn-resident-core`,
`sequence-output-projection`, `sequence-core`, `sequence-input-prep`, `embed`) are the
in-pass control, so the machine's clock cancels:

| build | `ffn` / untouched phases, eight rounds | median |
|---|---|---:|
| main | 0.85427 0.85157 0.85226 0.85248 0.84905 0.85215 0.84996 0.85033 | 0.85186 |
| six waves | 0.84162 0.84880 0.86708 0.86120 0.85682 0.85453 0.85767 0.85657 | 0.85670 |

The FFN's share of its own pass went **up** by 0.57%. Device span moved between 177.2 and
200.7 ms inside the six-wave run and 190.9 to 199.9 inside the main run, so the panel cannot
resolve better than about 1.5%, but a wave slot worth several percent is not there.

**Full-model instrument.** 384-token document in 128-row passes with the head on the last
pass, mode 0 interleaved in the same process as its own drift control, two panels per build.
Rounds whose mode 0 control fell outside 155.5-156.6 tok/s are listed separately rather than
averaged in:

| build | mode 19 prompt tok/s | median | mode 0 control |
|---|---|---:|---|
| main | 696.50, 691.32, 681.13, 687.74, 690.05 | 690.05 | 155.86, 156.41, 155.86, 155.62, 156.55 |
| six waves | 706.16, 695.92, 703.43, 696.27, 697.13, 679.51 | **696.70** | 156.00, 156.20, 156.44, 156.43, 156.11, 155.71 |

That is +0.96% on the median with the control moving +0.19%. The distributions overlap:
Mann-Whitney U is 23 of 30, two-tailed p about 0.18. Discarded rounds, all with a moved
control: 640.5 (control 144.3) and 540.6 (152.2) on main, 492.1 (127.4) on six waves.

The two instruments disagree in sign. Together they say the occupancy step is worth
somewhere between nothing and one percent, and not the several percent a latency-bound
kernel would show.

**The lane already had the corroborating measurement.** The stage sweep in
[`ffn-schedule.md`](ffn-schedule.md) ran gate/up at `W = 4, TT = 2`, which is nine waves per
SIMD32, against `W = 2, TT = 4`, which is five: 85.21 ms against 82.00 and 83.11. The
higher-occupancy gate/up shape was already the slower one. Nobody read it that way at the
time because the same sweep showed the *down* stage gaining from wave slots, and the two
stages were being explained together.

## Numerically identical

The FNV-1a hash of all 128 x 5120 FP32 residual outputs is **7446865760224376151** on every
one of the sixteen profile samples across both builds, which is the value
[`ffn-schedule.md`](ffn-schedule.md) records for this shape. Both storage orders hold the
same sixteen half-precision values per tile block and the same activation scales; only their
addresses move.

## What it rules out, and what it does not

Ruled out, for the A4 pair-code gate/up stage at 128 rows, `TT = 4`, `W = 2`, on gfx1151:
raising resident occupancy from five to six waves per SIMD32 by freeing 19 VGPRs of operand
addressing does not reduce the stage's time. The gap between the stage's issue model and its
measured time is not wave-slot-limited latency hiding, and spending registers to buy waves is
not the lever. Accumulator width is what the measured shape responds to, which is what the
row-tile ownership result found from the other direction.

Not ruled out: the same trade on the *down* stage alone, which the sweep above suggests does
respond to wave slots and which this experiment could not separate, because both stages moved
together and the down stage is a third of the region. A panel that pins gate/up and moves only
the down stage's occupancy would settle it.

Not addressed at all: what the other half of the stage's time actually is. Instruction issue
accounts for 48 to 55 ms of 85. The remaining 30 to 37 ms is not wave slots, not DRAM
bandwidth (the weight stream is 4.28 GB per pass, 50 GB/s against the 242 GB/s `bench/bw`
measures) and not B-operand bandwidth (33.4 GB per pass out of L1/L2, 393 GB/s). The next
candidate worth a panel is the sustained shader clock under a WMMA-dense load: the 5.99 ns
per matrix instruction behind the 40 ms budget was measured in a tight arithmetic loop, and
if the full FFN runs several hundred megahertz lower, the "missing" time is not missing.
`rocprofv3` counters on a single dispatch would answer it.

## Why it is not in the tree

The change is value-preserving and strictly less work, but its premise is refuted and its
measured effect is inside the panel's noise. It also moves occupancy the wrong way on routes
this experiment never measured: IU8 `TT = 2` goes 9 waves to 8, the A4 dense five-trit
arm that serves 32-row and generation shapes goes 7 to 6, and scaled-FP16 `TT = 4` goes 10 to
9. Landing a null result that silently cuts a wave slot off the shape two other engineers were
measuring that afternoon is worse than not landing it.

The patch is kept whole so a later engineer can revive either half without rebuilding it:

| artifact | contents |
|---|---|
| `scale-layout/half-major-scales.patch` | the complete change against `16c5d16` |
| `scale-layout/kernel-resources-{main,halfmajor,cscale-only,scale-only}.txt` | waves and VGPRs for all 123 instantiations, four builds |
| `scale-layout/{before,after}-phase-r8.json` | the eight-round phase panels, with residual hashes |
| `scale-layout/{before,after}-phase-128.json` | the first pair, with mode 4 as a second control |
| `scale-layout-{before,after}{,2}/run.json` | the four full-model prefill panels |

All under
[`../../data/bonsai2/batch-comparison`](../../../data/bonsai2/batch-comparison/README.md).

`tools/kernel_resources.py` is in the tree, because finding an occupancy cliff without
touching the GPU is worth having regardless of what this experiment concluded:

```sh
tools/kernel_resources.py kernels/ffn_batch.hip "k_proj_opt<1, 1, 1, true"
tools/kernel_resources.py --json kernels/sequence_batch.hip | jq '.[] | select(.waves < 6)'
```

# A stream stride that is a multiple of 4096 costs a weight image a third of its bandwidth

Every weight image in this engine is stored **tile-major**: a 16-row weight tile owns its `nb`
128-blocks contiguously. A wave owns a row tile and walks its blocks, so the hundreds of waves
resident at one instant each sit inside their own stream and the streams stand one **stream
stride** apart — `nb * block bytes`.

That stride is a performance parameter. [The sequence images](weight-stream-order.md) were
reordered block-major for it and gained 19% of their phase, and that work left an open question:
whether the cause is **bank and channel aliasing** between the streams or the **cross-wave
locality** of reading where the other waves are reading. The two have different fixes and only one
of them is cheap.

**It is the aliasing.** Breaking the stride's alignment with one run of padding, which keeps every
wave's own blocks contiguous and moves nothing else, recovers as much as reordering the whole
image — and where there is no aliasing to fix, the reorder is a loss.

## The geometry, with no model in it

[`bench/wstream`](../bench/wstream.hip) reads the same bytes three ways with no matrix
instructions, no scales and no model: N concurrent streams walking their own runs (tile-major),
every wave inside one run (block-major), and tile-major with one extra run of stride (padded). 512
MB buffer, 3 rounds, `WORK` dependent VALU ops per visit standing in for the block loop. GB/s of
distinct bytes read:

| deployed geometry | stride | tile-major | block-major | **padded** |
|---|---:|---:|---:|---:|
| sequence output (320 tiles × 2 waves, nb 48) | 24,576 | 92.4 | 105.4 | **104.1** |
| sequence input (1024 tiles, nb 40) | 20,480 | 140.4 | 202.9 | **199.5** |
| **FFN gate/up (2176 tiles, nb 40)** | **20,480** | **148.3** | 202.3 | **202.2** |
| **FFN down (640 tiles, nb 136)** | **69,632** | **173.5** | 205.4 | **199.1** |
| head (7760 tiles, nb 40) | 20,480 | 184.6 | 215.8 | **218.2** |
| control (320 tiles, nb 47) | 24,064 | 104.2 | 101.9 | 91.2 |
| control (320 tiles, nb 49) | 25,088 | 105.4 | 102.0 | 103.3 |
| control (1024 tiles, nb 41) | 20,992 | 200.4 | 201.2 | 187.5 |

`WORK = 192`; the zero-filler half of the sweep is in
[`wstream-pad.txt`](../../../data/bonsai2/batch-comparison/ffn-image-order/wstream-pad.txt) and is
the same shape with more room between the arms (tile-major 143–180, the other two 210–235).

Two things fall out, and the second is why the change that ships is a rule and not a pad.

- **Every deployed stride is a multiple of 4096 and every one of them is in the collapse band.**
  The padded arm lands within 3% of block-major in all five, at a fraction of the change.
- **Padding a stride that does not alias makes it worse.** 24,064 + 512 is 24,576 — straight into
  the band — and reads 0.88x. A blanket pad is not safe; a pad conditioned on the stride is.

## The run order in the FFN

A `(row tile, 128-block)` pair is one **run**. Its index in the stored image is
`tile * tstep + blk * bstep`, and `kernels/ffn_batch.hip` carries four settings:

| `--ffn-run` | order | `tstep` | `bstep` | |
|---:|---|---:|---:|---|
| 0 | tile-major | `nb` | 1 | the previous image |
| 1 | padded tile-major | `nb + 1` | 1 | unconditional; `1/nb` more bytes on every image |
| 2 | block-major | 1 | `ntiles` | every resident wave inside one contiguous run |
| 3 | **the stride rule** | per matrix | | pad iff `nb * block bytes` is a multiple of 4096 — **the default** |

`HALO_FFN_RUN_ORDER` picks the order a process starts in and `--ffn-run` on either measurement tool
walks them inside one. Only order 1 needs every image larger, so only a process that names it
before the state is created can select it later.

The rule separates the FFN's two A4 images exactly:

| image | block bytes | gate/up stride | down stride | rule |
|---|---:|---:|---:|---|
| two-bit slice-major, two-bit lane-major, pair codes | 512 | 20,480 = 2¹²·5 | 69,632 = 2¹²·17 | **padded** |
| dense five-trit | 416 | 16,640 = 2⁸·65 | 56,576 = 2⁸·221 | tile-major |

**Nothing else moves.** Same bytes, same values, same order of use, the same block loop, and the
scale image is untouched — a run carries 32 bytes of scale, so its stream stride is 1,280 or 4,352
bytes and no order question reaches it. The residual FNV-64 is identical in every arm at every
width, including `7446865760224376151`, which this lane published for a 128-row mode 19 pass
before any of this.

## What it is worth in the engine

Mode 19 at 32 prefill rows with each A4 image forced, three orders walked in one process with the
case order reshuffled every round, three rounds, `--pin-clock`. `ffn / control` is the phase divided
by the four phases the change cannot reach, which is what makes the arms comparable on a box whose
clock moves 15% inside a panel:

| image (stride) | tile-major | padded | block-major |
|---|---:|---:|---:|
| dense five-trit (16,640) | 1.2434 | 1.2469 **(+0.3%)** | 1.2678 **(+2.0%)** |
| pair codes (20,480) | 1.6529 | 1.4598 **(−11.7%)** | 1.4738 **(−10.8%)** |

An earlier process measured the same three arms at −17.0% and −16.7% for the pair image and +0.4% /
+5.6% for the dense one. **The sign and the ordering repeat; the magnitude does not.** The
tile-major pair arm itself read 50.66 ms in one process and 40.36 in another for identical work,
which is what an effect that depends on where the allocator put the image looks like. Quote the
range, not one panel.

### The deployed shapes

Mode 19 reads pair codes on both stages above 32 rows and the dense image at or below, so the rule
reaches the 128-row pass and leaves a generation step alone.

| | tile-major | the rule | padded |
|---|---:|---:|---:|
| `ffn / control`, 128 rows, process A | 0.9951 | **0.9749** | 0.9746 |
| `ffn / control`, 128 rows, process B | 0.9969 | **0.9834** | 0.9781 |
| `ffn` ms, 32 rows | 29.82 | 29.76 | — |
| `ffn` ms, 32-stream generation step | 32.30 | — | 32.14 |

**−1.4% to −2.1% of the FFN phase at 128 rows, and a null at the generation shape**, which is the
dense image and is not aliased. The rule and the unconditional pad agree to a tenth of a point on
the shape they both touch, which is the check that the rule is firing where it should.

Full model, 384-token document in 128-row passes, arms interleaved in one process: **852.7 against
848.1 tok/s** (mode 19 prompt, +0.5%) and **323.0 against 323.1 tok/s** aggregate across 32 streams
(null). Mode 18's A8 prompt reads 530.0 against 519.5 (+2.0%) with arm ranges that overlap. **The
full-model prompt effect is inside this box's spread and is not the evidence for this change**; the
phase panel is.

### Why the phase moves less than the bandwidth

`bench/wstream` says the FFN's stream geometry gives up a third of its achievable read bandwidth,
and the deployed 128-row pass gains 2%. Both are true because a 128-row pass is not waiting on that
stream: it retires 128 rows of matrix work per weight byte and sits at about 90% of its own issue
model. The stream binds where a shape reads the whole image for very little work, which is exactly
the pair image at 32 rows — 4.28 GB for two token tiles — and there the same geometry is worth 12
to 17%.

**A stride fix pays in proportion to how much of the phase is spent waiting for the stream.** That
is also the reason the change ships as a default rather than as a lever: it is free everywhere and
it raises the floor of the shape that is most exposed.

## Reproducing

```sh
make bonsai-halo tools/batch_profile bench/wstream
tools/run-batch-compare --exec ./bench/wstream --mb 512 --rounds 3          # the geometry alone
HALO_FFN_RUN_ORDER=1 tools/run-batch-compare --pin-clock --profile-tool --modes 19 --rows 32 \
    --ffn-image 1,2 --ffn-run 0,1,2 --heads 0 --traces 1 --rounds 3 --out RESULT.json
HALO_FFN_RUN_ORDER=1 tools/run-batch-compare --pin-clock --profile-tool --modes 19 --rows 128 \
    --ffn-run 0,1,3 --heads 0 --traces 1 --rounds 3 --out RESULT.json
```

`HALO_FFN_RUN_ORDER=1` is needed in the environment for the unconditional pad arm, because it is
the only order that needs every image one run per tile larger. Read the panels as `ffn` divided by
the untouched phases in the same sample; a raw millisecond column on this box is a clock reading.

## Installed

Published as `bac8a8a`, `f93c24b` and `11eaa80`; installed on canonical `11eaa80`, binary
`7ee42785713b029f…`, `tools/batch_profile` `68aaee919e9d79ac…`. The installed build reproduces the
panel on its own hardware — `ffn / control` **1.0045 tile-major against 0.9818 under the rule**,
−2.3%, with the residual FNV-64 `7446865760224376151` in every sample, the value published for this
shape before any of this work. The resident service runs on it.

Raw panels, including the two processes that disagree on magnitude:
[`batch-comparison/ffn-image-order/`](../../../data/bonsai2/batch-comparison/ffn-image-order/).

## What this leaves

- **The vocabulary head's image is still tile-major at stride 20,480** and `bench/wstream` prices
  that geometry at 184.6 against 218.2 GB/s. The head is 2.5 ms of a 32-stream generation step and
  more of a prompt pass that writes logits. Nobody holds `kernels/head_batch.hip`.
- **The drafter reads 0.93 GB of Q4 weights at 127 GB/s** ([`drafter-q4.md`](drafter-q4.md)), which
  is the collapse band exactly. Its image's stride has not been checked against this rule.
- **The persistent kernel's HALO tiles** are the last large image nobody has priced this way.
- The rule's threshold is fitted, not derived: 4096 separates the two FFN images and every
  `bench/wstream` row, and the controls at 20,992 and 25,088 say the band is narrower than "any
  power of two". A sweep of `nb` at fixed bytes would map its edges.

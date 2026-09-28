# The weight image's run order, and why a permutation of stored bytes is worth 19%

Every weight image in this engine is stored **tile-major**: a 16-row weight tile owns its `nb`
128-blocks contiguously, `[row tile][K block][512 B]`. A wave owns a row tile and walks its blocks,
so the hundreds of waves resident at any instant are each 512 bytes into their *own* stream, and
those streams are `nb * 512` bytes apart — 20,480 for the sequence input projection and 24,576 for
the output one.

**Block-major** stores the same runs at `[K block][row tile][512 B]`. Same bytes, same values, same
order of use, one wave-uniform multiply-add either way. The only thing that moves is which 512-byte
run a `(row tile, block)` pair lands on, which puts every concurrently resident wave inside one
contiguous run instead of spreading them over the image.

It is worth **19% of the two sequence projections at the 32-row shape and 9% at 128 rows**, with the
residual FNV-64 identical in every arm at both widths and the block loop's instruction census
unchanged to the slot.

## The stream geometry costs up to 45% of DRAM read bandwidth on its own

[`bench/wstream`](../bench/wstream.hip) takes the model out of the picture. `N` concurrent streams,
each walking its own contiguous region 512 bytes per visit, against the same bytes read with every
resident wave inside one run; 64 launches over distinct regions of a 512 MB buffer, so no launch
re-reads what the last one left in the 32 MB MALL; `WORK` dependent VALU ops per visit stand in for
the block loop the real kernel issues between two weight reads.

| run bytes | filler ALU | waves | stream stride | tile-major | block-major |
|---:|---:|---:|---:|---:|---:|
| 512 | 0 | 640 | 12,800 | 211.3 | 214.0 |
| 512 | 0 | 1280 | 6,144 | **143.3** | 213.5 |
| 512 | 192 | 1280 | 6,144 | **116.5** | 187.4 |
| 1024 | 192 | 320 | 25,600 | 199.8 | 191.9 |
| 1024 | 192 | 640 | 12,288 | **125.4** | 200.1 |
| 2048 | 192 | 160 | 51,200 | 194.3 | 189.0 |
| 2048 | 192 | 320 | 24,576 | **119.2** | 206.5 |
| 2048 | 0 | 640 | 12,288 | **116.3** | 217.5 |

GB/s of distinct bytes; the full 40-point sweep is in
[`weight-stream/sweep-512mb.json`](../../../data/bonsai2/batch-comparison/weight-stream/sweep-512mb.json).

**Block-major never leaves the 165-218 band in any cell of the sweep. Tile-major is either the same
number or about half of it, and which one depends on the power-of-two factor of the stream stride.**
Strides of 6,144, 12,288 and 24,576 read 143, 125 and 116-119; strides of 12,800, 25,600, 26,112 and
51,200 read 194-211. That is channel and bank aliasing between concurrent streams: when the stride
carries a large power of two, the streams' 512-byte visits land on a few of the device's banks
instead of spreading across them.

Two things this is **not**. It is not latency — the `WORK` axis moves both arms together and the
collapse survives at zero filler. It is not the run length on its own — 2048-byte runs collapse at
one stride and not at another, so reading more per visit is not the fix; reading it *where the other
waves are reading* is.

The deployed strides are in the collapse zone by accident of the model's shape: the sequence input
matrices have `nb = 40` (stride 20,480 = 2¹²·5) and the output matrices `nb = 48` (24,576 = 2¹³·3).

## What it is worth in the engine

`Matrix` gained two ints. A `(row tile, block)` pair's run index is `tile*tstep + b*bstep`:
tile-major is `(nb, 1)` and block-major `(1, ntiles)`. `repack` writes whichever order the image is
in and `sequence_batch_set_image` permutes an existing image in place through one scratch matrix,
which is how both orders share a process, a clock and a warmed device.

**The emitted code does not change.** The block loop of `project<1,…>` is 209 instructions and
`project<4,…>` 369 in both orders, the same numbers
[the output-projection document](sequence-output-projection.md) published, with the same class
histogram. `nb` was a runtime field before and `tstep` is a runtime field now.

Mode 20, `--seq-image 0,1` shuffled with the other cases in one process, two rounds, the six phases
the change cannot reach as the in-panel control
([`phase-m20.json`](../../../data/bonsai2/batch-comparison/weight-stream/phase-m20.json)):

| | 32 rows | | 128 rows | |
|---|---:|---:|---:|---:|
| phase | tile | block | tile | block |
| `sequence-input-projection` | 14.666 | **10.747** | 41.535 | **39.800** |
| `sequence-output-projection` | 7.751 | **6.650** | 20.209 | **15.349** |
| `ffn` *(control)* | 42.162 | 42.929 | 167.679 | 169.718 |
| `gdn-resident-core` *(control)* | 5.059 | 5.056 | 15.551 | 15.721 |
| `sequence-core` *(control)* | — | — | 4.471 | 4.548 |
| device span | 75.557 | 71.409 | 256.827 | 252.540 |

Second round shown; the first round of the 32-row pair is in the raw file and is not usable on its
own because its controls moved 8-14% while the device was still settling. Normalised on the control
phases, the two projections together are **-19% at 32 rows and -9% at 128**.

The split between the two stages is the interesting part and it reverses with width:

| | 32 rows | 128 rows |
|---|---:|---:|
| input projection, normalised | **-27%** | -5% |
| output projection, normalised | -14% | **-22%** |

**Bit-identical, measured rather than argued:** residual FNV-64 `15659291503748132804` at 32 rows and
`16921095978579329146` at 128, in every arm of every round, and `16921095978579329146` is the hash
[the sequence row-tile work](../orchestration/HANDOFF.md) already published for a 128-row mode 20
pass. The arm that carries the permutation also carries its cache flush — 1.9 GB of copying
immediately before its first measured pass — so the effect is measured against a handicap.

## Full model

Same executable, `--seq-image 0,1` interleaved inside one process, mode 19 (A4 FFN, wide committed
sequence path), fp32 state at defer 1 — the process defaults, not the fastest configuration. Raw
under [`weight-stream/`](../../../data/bonsai2/batch-comparison/weight-stream/).

| workload | tile-major | block-major | |
|---|---:|---:|---:|
| prompt, 384-token document in 128-row passes, round 0 | 736.5 | **773.0** | +5.0% |
| the same, round 1 | 717.4 | **733.4** | +2.2% |
| aggregate generation, 32 streams, round 0 | 299.3 | **318.8** | +6.5% |
| the same, round 1 | 299.1 | **318.3** | +6.4% |

The generation pair is the tighter measurement by a long way — the two arms repeat to 0.2% across
rounds — and it is the shape the phase panel predicted, because a 32-stream step is where the
projections are 23% of the step and where the input stage takes its -27%.

Single-stream generation is **a null by construction**: `--bench` never calls `run_sequence_batch`,
so no part of this change is on that path.

Block-major is now the default image. `HALO_SEQUENCE_IMAGE=tile` restores the previous order for a
process, and `--seq-image 0,1` on either measurement tool walks both.

## Why the prefill shape still wins, which the arithmetic did not predict

A peer's objection is the right starting point and it is correct as far as it goes: four-bit
activations on the input projection measured 1.84x at 128 rows
([`sequence-input-a4.md`](sequence-input-a4.md)), and that change moves no weight byte and no weight
address. A kernel whose weight stream is the wall cannot behave that way. So the 128-row shape is
issue-bound, and the -5% the input stage takes there is about what a second-order effect should look
like.

The output stage at 128 rows is not second-order at -22%, and it is the stage with the larger
power-of-two factor in its stride. It is also the stage with the smaller grid — 320 row tiles where
the input has 1024 — which is the same asymmetry
[`sequence-output-projection.md`](sequence-output-projection.md) found when it measured that stage
at 171% of its issue model against the input stage's 129%, and the same one
[`docs/clock-power.md`](clock-power.md) reads as 59.6% clock-proportional against 83.5%. Three
independent instruments now say the output stage carries clock-independent time that the input stage
does not, and this is the first change that removes some of it.

At 32 rows the ordering reverses because the input stage's demand per weight byte rises by the same
8x the token count fell, while the output stage's grid problem is unchanged.

## What this does not say

- **It is not a general 2x on the weight path.** The microbenchmark's collapse is 45% at the worst
  stride and nothing at the best, and the engine's phases sit at 27-68 GB/s of demand, well under
  even the collapsed rate. What the permutation removes is the part of the stall that DRAM bank
  conflicts contribute at a given instant, not a bandwidth ceiling.
- **It does not retire the issue-model work.** Both projections remain well above their own issue
  models after this change.
- **The FFN, the vocabulary head and the drafter were not measured** here.
  [The FFN's image order](ffn-image-order.md) has since measured the FFN and both remaining
  images' geometries, and it answers the question this document left open. `bench/wstream` now
  carries a **padded** arm — tile-major with one extra run of stride — which keeps each wave's own
  blocks contiguous and changes only the alignment. It matches block-major on every aliased
  geometry, so the cause is the **aliasing** and not the cross-wave locality, and a stride that
  does not alias is made *worse* by padding. The FFN ships the rule that follows (pad iff the
  stride is a multiple of 4096); the head (20,480 → 184.6 against 218.2 GB/s) and the drafter still
  store theirs unpadded.

## Reproducing

```sh
make bench/wstream tools/batch_profile
tools/run-batch-compare --exec ./bench/wstream --mb 512 --rounds 3
tools/run-batch-compare --profile-tool --modes 20 --rows 32,128 --seq-image 0,1 \
    --heads 1 --traces 0,1 --rounds 2 --warmup-ms 1500 --out RESULT.json
```

`HALO_SEQUENCE_IMAGE=tile|block` picks the order a process starts in; `--seq-image 0,1` on either
measurement tool walks both inside one process, which is the only form that resolves a difference
this size on this box.

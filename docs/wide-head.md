# Wide vocabulary head

The output projection was the last part of a batched pass that still ran eight rows at a time. A
128-row prompt pass visited the 278 MB output image sixteen times and filled eight of the sixteen
columns each matrix instruction offers. It now runs once over the whole batch, and the arithmetic
map is unchanged: every logit keeps the bit pattern the sliced head produced.

## Full-model result

Three rounds per configuration, 384-token document, logits on the final pass only, generation
across independent streams from their own prompts. Both arms are the same executable; the control
sets `HALO_WIDE_HEAD=0`. Medians:

| workload | sliced head | wide head | gain |
|---|---:|---:|---:|
| A4 prefill, 128 rows/pass | 600.3 | 614.7 | 2.4% |
| A4 prefill, 32 rows/pass | 441.1 | 444.0 | 0.7% |
| A8 prefill, 128 rows/pass | 485.5 | 497.7 | 2.5% |
| A8 prefill, 32 rows/pass | 344.9 | 346.8 | 0.6% |
| A4 generation, 32 streams | 253.4 | 260.0 | 2.6% |
| A8 generation, 32 streams | 210.4 | 215.0 | 2.2% |
| A4 generation, 8 streams | 147.4 | 147.5 | none |
| A8 generation, 8 streams | 147.5 | 147.5 | none |

Eight streams is the intended null: the route requires more than `RMAX` rows, so that shape still
runs the sliced head and must not move. Raw runs are `wide-head-{a4,a8}-{prefill,decode}-{on,off}/`
in [the comparison directory](../../../data/bonsai2/batch-comparison/README.md).

Gains scale with how much of a pass the head is. At 128 rows/pass the 384-token document runs three
passes and charges the head once, so a 15 ms saving moves 2.4%. At 32 rows/pass the same saving is
spread over twelve passes. Nothing outside the head changed, so a workload that computes logits on
every row gains proportionally more.

## The head's own cost

`tools/batch_profile --heads 0,1` runs the identical pass with and without the output head, so the
difference is the head including its normalization, quantiser, projection and argmax. Medians of
three samples, mode 19, after a 128-token prefix:

| route | 128-row pass | 32-row pass |
|---|---:|---:|
| sliced, eight rows at a time | 26.554 ms | 6.370 ms |
| wide, 16 token columns per workgroup | 13.070 | 2.999 |
| **wide, 32 token columns (selected)** | **11.204** | **2.690** |
| wide, 64 token columns | 13.160 | 5.665 |
| wide, 128 token columns | 20.323 | |

A counter-free kernel trace of one 128-row pass splits the selected route: `head_tile<2>`
10.675 ms, `head_argmax_slices` 0.737 ms, the sixteen `SEQ_PART_HEAD` prep launches 0.065 ms
together, and `head_sums` below the reporting threshold.

The token-group width is a pure scheduling choice - all four widths produce identical bits - and it
trades weight traffic against occupancy. Two 16-column groups per workgroup read the 278 MB image
four times for 128 rows; eight groups would read it once but need 256 VGPRs and spill 45 of them.
The measured optimum is two groups, at 141 VGPRs. At 32 rows the selected route costs 2.69 ms
against a floor of about 1.15 ms of weight traffic plus 1.5 ms of matrix issue, so that shape is
close to the arithmetic and bandwidth it must pay.

## Implementation

`kernels/head_batch.hip` owns the route.

1. Each eight-row slice runs `SEQ_PART_HEAD`, a new `PART` of `k_forward_rows` holding only the
   deployed final-norm, Hadamard and quantiser phase. It writes the wide WMMA operand directly
   through the existing `FwdParams::sequence_*` binding rather than the row-major `xq/xs/xsum`
   triple, exactly as the layer projections already do.
2. `head_sums` recovers the zero-point term the unsigned trit codes need. The wide operand path
   does not produce it, because the layer projections use signed codes with no such term. Integer
   block sums are exact and order-free, so recovering them from the written operand gives the same
   integers the row-major quantiser would have written.
3. `head_tile<TT>` walks the HALO tiles of `output.weight`. One workgroup owns a 32-row tile and
   `TT * 16` token columns; wave *w* owns K-blocks [5w, 5w+5). The tile bytes are peeled into WMMA
   A operands with the deployed `peel`, and the two lane-half orderings cover the tile's 32 rows.
4. `head_argmax_slices` and `head_argmax_final` reuse the deployed argmax device code with the same
   1024-element slice decomposition and tie rule.

The packed tiles are never converted. The head reads the same 26-byte blocks the eight-row kernel
reads and peels them straight into matrix-instruction registers, so it needs no second weight image
and moves 278 MB rather than the 338 MB a two-bit lane-major copy would cost. The only new memory is
the batch's logits, operand and argmax scratch: 132 MB at capacity 128, allocated during
preparation, and nothing at all when a process never prepares a batch.

### Why the bits do not move

For one output logit the sliced head computes eight partial sums, one per wave, each an `fmaf` chain
over five K-blocks, and then adds them in wave order. The wide head keeps that decomposition exactly:
same wave-to-block assignment, same chain, same final order. Integer WMMA results do not depend on
which of the sixteen matrix columns a token occupies, and the quantiser, scales and block sums are
the values the deployed path produces. The only change is that the other eight columns now carry
real tokens instead of duplicates of the first eight.

## Acceptance

[A4 comparison](../tools/batch-compare/results/wide-head-identity-a4.json) and
[A8 comparison](../tools/batch-compare/results/wide-head-identity-a8.json): 71,516,160 finite
full-vocabulary logits each at 32, 40, 88 and 128 rows, zero differing bits, no non-finite values,
and 128 continuation tokens from a four-step 32-stream greedy run identical on both sides. Row
counts 40 and 88 exercise the partial final token group.

## Operation

The route is the default for any batched pass of more than eight rows once a batch is prepared.

```sh
make -j8 bonsai-halo tools/batch_compare tools/batch_profile
tools/run-batch-compare --tag wide-head-a4-prefill-on --modes 19 --only prefill \
  --prefill-rows 32,128 --rounds 3
HALO_WIDE_HEAD=0 tools/run-batch-compare --tag wide-head-a4-prefill-off --modes 19 --only prefill \
  --prefill-rows 32,128 --rounds 3
```

`HALO_WIDE_HEAD=0` selects the sliced control and allocates none of the module's memory.
`HALO_HEAD_TT=1|2|4|8` overrides the token-group width; every value produces the same bits.
Ordinary serving never prepares a batch, so its head, defaults and numerical contract are untouched.

## Deployment

Canonical `f069a08`, reinstalled after the profiling fix below as SHA-256
`71c93e725789d70d83ae9f91bb3d582ee5a88d8bad5e0c6adc054842d0b0abf9`. The installed binary reproduced
all 7,946,240 logit bits at 32 rows in both modes, against the preflight wide-head run and against
the sliced control, recorded in `wide-head-published-a4/deployment.json`. Single-stream `--bench`
gave 34.02 tok/s and 29.40 ms/token, unchanged, run through the new
`tools/run-batch-compare --engine` passthrough so it held the GPU lock and the service mask. The
resident server was restored on the new executable.

The first published binary asked the profiler for one phase stamp on the head-prep launch, but
`SEQ_PART_HEAD` returns without a grid barrier and nothing writes the closing stamp, so any traced
pass computing logits threw `missing or reversed kernel profile stamp`. The bonsai-pass-schedule
engineer found it and proposed the fix taken here: the launch reports no phases and leaves
`P.prof` null, so the HIP event bracket times it and the traced and untraced arms run the same
code. A traced 128-row A4 pass now completes, agrees with its untraced residual hash, and puts
`head-projection` at 11.474 ms and all sixteen `head-prep` launches at 0.195 ms together. The
reinstalled binary reproduces all 7,946,240 logit bits at 32 rows against both the preflight run
and the sliced control.

## What the head kernel actually issues, and the one thing that was free

The wide head reaches about 30 TOPS of issued IU8 where the `project` kernel reaches 37 on the same
instruction. The gap is not memory: at 128 rows the kernel reads its 278 MB weight image at about
24 GB/s against the 242 GB/s `bench/bw` measures. It is issue.

One `head_tile<2>` workgroup walks its 32-row tile one 128-K block at a time. Counting the compiled
block loop on `gfx1151` as issue slots gives the whole story:

| block loop, `head_tile<2>` | slots | of which |
|---|---:|---|
| `v_wmma_i32_16x16x16_iu8` | 32 | the work |
| five-trit peel | 220 | 64 `v_pk_mul_lo_u16`, 64 `v_pk_lshrrev_b16`, 60 `v_and_b32`, 32 `v_lshl_or_b32` |
| epilogue drain | 125 | zero-point subtract, convert, scale product, accumulate |
| operand addressing | 109 | 64-bit address rebuilt per fragment load |
| `swap16` for the odd 16 rows | 32 | `v_permlanex16_b32` |
| other | 32 | |
| **total** | **733** | 518 of them non-WMMA VALU |

An IU8 WMMA occupies the SIMD for 32 cycles, so a block costs about 32 x 32 + 518 = 1542 cycles
against 1024 of arithmetic. Multiplied out over 1.24M wave-blocks that is 10.0 ms against the
11.2 ms the phase measured, which is close enough to treat the count as the model.

**Addressing was a quarter of the overhead and bought nothing.** The fragment a lane wants advances
by exactly `npad * 16` bytes per K16 slice, and eight slices carry it to the next block, so the
whole K walk needs one add per load instead of rebuilding `(kblk * 8 + kb) * npad + token` each
time. Keeping that offset 32-bit matters: the operand is at most `(D / 16) * 128 * 16` bytes, so the
load keeps its scalar base, where a 64-bit running pointer makes the compiler emit `flat_load_b128`
and lose it. In the same edit the epilogue's zero-point subtraction disappears into the accumulator
seed — the A operand carries unsigned trit codes, so every product owes one activation per weight,
and seeding the accumulator with `-xsc` costs the moves the zero seed already cost.

Block loop 733 -> 600 slots, non-WMMA VALU 518 -> 428, and VGPRs 141 -> 133 at unchanged occupancy.

### Measured

`tools/batch_profile`, mode 19, one 128-row pass, logits on every row. `--heads 0,1` is the control:
a head-free pass runs every other phase and never launches this kernel, so `ffn`,
`gdn-resident-core` and the sequence projections say what the box was doing. Both orderings were
run, because the two processes an hour apart differed by 8%.

| order | `head-projection` | untouched phases | head, drift-corrected |
|---|---:|---:|---:|
| control then selected | 13.143 -> 10.449 ms | x0.920 | **-13.6%** |
| selected then control | 13.371 -> 11.610 ms | x1.0006 | **-13.2%** |
| 32 rows, one panel | 3.202 -> 2.965 ms | x1.034 | **-10.5%** |

The second pair is the one to read: its controls moved 0.06%. A 32-row pass runs the same
instantiation and the same one-token-group grid a 32-stream generation step runs, so the decode
shape is covered by that row.

Full model, 384-token document in 128-row passes with the head on the last pass, mode 0 interleaved
in the same process: 717.9 -> 736.0 prompt tok/s with the control at 156.1 -> 158.0, so +1.3% after
the control against +0.33% predicted from the phase. The panel rules out a regression; it cannot
resolve a third of a percent, and the phase measurement is the evidence here.

[Bit identity](../../../data/bonsai2/batch-comparison/head-issue/identity.json): 71,516,160 finite
logits at 32, 40, 88 and 128 prefill rows, zero differing bits, zero non-finite pairs. Raw panels
are in [`head-issue/`](../../../data/bonsai2/batch-comparison/head-issue).

### The rows it produces

[The head's row selection](head-rows.md) is a separate question from its schedule and it was the
larger one at a prompt pass: the head produced `VOCAB` floats for every row and ingestion read the
last. It now takes a half-open row selection, so a prompt pass costs one pass over the output image
at any width - 1.534 ms at 128 rows, 1.535 at 256 - instead of 10.598 and 21.147. The token-group
width follows the selected rows rather than the pass, and is chosen per call rather than cached from
the first wide call for the life of the process.

### What is left in this kernel, priced

The peel is now half of everything that is not a matrix instruction: 220 slots per block, 1.72 VALU
per weight per lane, and independent of the token-tile width. Two ways to make it cheaper both lose
here, for reasons worth keeping:

- **One trit per nibble, expanded by `v_perm` from a three-byte table.** About 5 ops per 8 weights
  against the peel's 14, roughly 150 slots off a block, 10% of the kernel. It needs 4 bits per
  weight, so the head's image goes 278 MB to 636 MB, beside the 278 MB HALO image the deployed
  per-slice head still reads. At 128 rows the extra traffic is free (24 -> 55 GB/s). At 32 rows the
  same kernel already reads 278 MB in 2.97 ms, 94 GB/s, and 636 MB would be 214 GB/s against a
  242 GB/s roof: the arithmetic saved is not recoverable because the kernel lands on its own new
  traffic floor. So it buys about 1 ms of a prefill pass, nothing of a generation step, and costs
  636 MB on a host where modes 18 and 19 together already exceed the device budget.
- **Two-bit codes with `expand_i8`**, the FFN's IU8 operand: 36 ops per 16 weights, 2.25 VALU per
  weight. Worse than the peel it would replace. The head's packed five-trit peel is already the
  cheaper expansion of the two the engine carries.

The register cliff is not where the earlier note put it. Widening the token group is blocked by the
FP32 row accumulators, not by the peel's `tr[32]`: `head_tile` allocates 113 VGPRs at one token
group, 133 at two, **232 at four** and 256 with 44 spilled at eight. The step from two to four is
99 registers for 32 floats of accumulator, and it takes occupancy from 10 waves per SIMD to 6. A
lazy peel returning one A fragment per source dword would move `tr[32]`, which is not what stands
between the kernel and four groups.

# A generation step's attention, and the two things the unit was spending on rows it did not have

Every attention measurement in this repository is prefill-shaped. This is the decode shape: what a
32-stream generation step actually charges for attention, how that changes with context, and one
exact change that comes out of reading the unit against the shape that runs it.

Raw panels and what each settles: [`batch-comparison/attn-row-groups/`](../../../data/bonsai2/batch-comparison/attn-row-groups/README.md).

## The unit is written for eight rows and a generation step has one

`attn_chunk_unit<TT>` computes one (row group, query head, key chunk). A group is up to `TT`
consecutive rows of one sequence, and the row loop clamps its index:

```c
const int row = S.row0 + min(i, S.nrows - 1);
```

So a group of one row runs all `TT` iterations on the same row. It loads that row's query `TT` times
from one address the compiler cannot fold — `min` is a runtime clamp — runs the 256-dimension dot
product against every key `TT` times, runs `block_max_256` and `block_sum_256` over the score array
`TT` times, and does `TT` fused multiply-adds per value. Then `if (i < S.nrows)` writes one partial
and throws the other seven away.

**Prompt passes never see it.** A prompt pass hands this phase groups of exactly eight consecutive
rows, so every prefill panel in the tree measures a full unit. **Generation is made of one-row
groups**: a 32-stream step is four persistent slices of eight sequences carrying one row each, which
is the 64 `sequence-core` launches
[the decode map](decode-map.md) counts; single-stream decode and the context rows of a drafted verify
are the same shape.

`attn_chunk_group` dispatches on the group's own row count, to the narrowest of 1, 2, 4 and `TT`
that covers it. The 2 and 4 rungs exist for a batched drafted step, where `--decode-tokens` gives
each sequence two or four rows.

**This is exact, not approximate.** The narrow instantiation is the same source with the row loop
taken fewer times. Row *i* sees the same query, the same key chunk in the same order, the same
`fmaf` chain and `warp_sum`, the same `block_max_256` / `block_sum_256` over the same 256 threads,
and writes the same partial. Nothing is reassociated because nothing was ever summed across the row
axis. `P.attn_narrow == 0` keeps the wide-only dispatch as the in-process control, and `k_attn_wide`
is unchanged at 92 VGPR and 16 waves per SIMD32 in both arms.

## Attention becomes the largest phase in a generation step at about a thousand tokens

`tools/batch_profile` gains `--decode-context` and a list-valued `--decode-prompt`. This is the
decode counterpart of [`--prefill-scan`](long-context-attention.md): each case re-prefills every slot
to its own prefix, so both dispatch arms meet at the same position in one process, under one clock.

Mode 19, 32 streams, context 1024, one round, arms interleaved
([`decode-pos-1024.json`](../../../data/bonsai2/batch-comparison/attn-row-groups/decode-pos-1024.json)):

| phase | prefix 64 | prefix 960 |
|---|---:|---:|
| `gdn-resident-core` | 48.840 | 48.077 |
| `ffn` | 39.455 | 39.440 |
| **`sequence-core`** | **4.422** | **48.410** |
| `sequence-input-projection` | 16.881 | 17.020 |
| `sequence-output-projection` | 12.831 | 13.052 |
| `head-projection` | 3.423 | 3.319 |

**Five of the six phases are flat to 0.4% across that walk and attention grows by a factor of
eleven**, passing the recurrent state somewhere around a thousand tokens of context. The flat five
are therefore a free in-panel control for anything measured this way, exactly as the position axis
made them in prefill.

[The decode map](decode-map.md) says "at 32 streams the step is dominated by recurrent state traffic",
and it is right at a 64-token prefix and wrong at 960. **Every decode number this lane has published
is one point of that curve at the short end**, including this document's own headline below.

## The decode score units are at their byte roof, and that is a different roof from prefill's

Per step at 32 rows and 961 positions, the score units request

```
961 keys x 1024 bytes (K and V, fp16, HD = 256) x 24 heads x 32 rows x 16 attention layers = 12.1 GB
```

and the measured slope is **0.0491 ms per token of position**, which is 257 GB/s against the
242 GB/s [`bench/bw`](throughput-budgets.md) measures on this box. The narrow dispatch takes the
slope to 0.0355 ms per token — 355 GB/s effective, so it starts getting reuse rather than being
issue-limited at a rate that happens to coincide with the roof.

This is the opposite diagnosis from prefill, where
[the long-context measurement](long-context-attention.md) established that the score units are
latency- and issue-bound and that `TT = 16` loses 18% because wave slots are worth more than halved
key traffic. Both are true: the same kernel is issue-bound where it has eight rows to amortise a key
chunk over, and byte-bound where it has one.

**So a slot census does not price the decode shape.** Anything whose whole mechanism is fewer
instructions per key — including the lane-per-key score loop — is a prefill result, and needs to be
measured there.

## What it costs, measured

### At context 512, the phase moves and the step does not

Mode 19, 32 streams, 64-token prefix, three rounds, arms interleaved in one process
([`decode-512.json`](../../../data/bonsai2/batch-comparison/attn-row-groups/decode-512.json)):

| phase | wide | narrow | ratio |
|---|---:|---:|---:|
| `gdn-resident-core` | 45.202 | 44.904 | 0.993 |
| `ffn` | 33.281 | 33.049 | 0.993 |
| `sequence-input-projection` | 14.037 | 13.991 | 0.997 |
| `sequence-output-projection` | 11.639 | 11.694 | 1.005 |
| **`sequence-core`** | **3.410** | **2.951** | **0.865** |
| `head-projection` | 2.545 | 2.538 | 0.997 |
| device span | 112.729 | 111.721 | 0.991 |
| step wall, untraced | 108.661 | 108.380 | 0.997 |

Six untouched phases sit inside 0.7%, so the phase is **-12.9% normalised** and the step is a null:
`sequence-core` is 3% of a step at this prefix and there is nothing else there to win.

### At context 960 it is the step

| | wide | narrow | ratio |
|---|---:|---:|---:|
| `sequence-core` | 48.410 | 34.910 | 0.721 |
| six untouched phases, summed | 123.025 | 112.304 | 0.913 |
| **`sequence-core`, normalised** | | | **0.790** |
| step wall, untraced | 144.618 | 139.524 | **0.965** |

One round on a box carrying five other engineers, so the untouched control moves 8.7% and the
normalised phase number is the one to read: **-21%**, against -12.9% at the short prefix. The raw
step wall is **-3.5%**, and it grows with context because the term it removes is the only one that
does.

### Bit-identical

Residual FNV-64 over every FP32 output of all 64 layers, every sample of both panels:
`8087160186626784132` at a 64-token prefix — which is the hash
[the FFN load schedule](ffn-dense-loads.md) published for the 32-stream step before this change
existed — and `10606832871390354142` at 960. Both arms, traced and untraced.

## The other reuse axis, and why it is the decode one

A key chunk costs the same bytes whoever reads it. What changes is how many dot products are charged
to those bytes, and the unit has exactly two axes to spend on that:

- **`TT`, consecutive rows of one sequence.** A prompt pass has eight. A generation step has one,
  and no arrangement gives it more, because its rows are *different sequences reading different
  caches*. This is why `TT = 16` is a prefill lever and cannot be a decode one.
- **`HG`, query heads of one KV group.** Six here, at every shape. Nothing crosses the head axis —
  separate query, separate scores, separate softmax, separate partial — so folding a KV group's
  query heads into one unit is a relabeling of which workgroup owns which (row, head) pair, and
  `HG = 1` compiles to exactly the code that ran before the parameter existed.

The registers say they are alternatives rather than two knobs: a unit holds `HG * TT * DPL` query
registers live across the whole key loop. Eight rows by six heads is 384 VGPRs of query alone and
cannot be built; **one row by six heads is 48, which is under the 64 that `qr[8][8]` costs today**,
so the decode shape does not raise the register ceiling of the persistent kernel.

Because it changes the unit count it is a launch-level choice, not a per-group one. `ph_attn` reads
`P.seqs` — at most `MAXSEQ` entries — and takes it only when every group in the slice carries one
row; `launch_attn_wide` takes the same test from the host, which already has the group table.

### It converts, and it converts where the byte argument said it would

Mode 19, 32 streams, context 1024, narrow dispatch on in both arms, one process
([`decode-hg-1024.json`](../../../data/bonsai2/batch-comparison/attn-row-groups/decode-hg-1024.json)):

| | prefix 64 | | prefix 960 | |
|---|---:|---:|---:|---:|
| | `HG = 1` | `HG = 6` | `HG = 1` | `HG = 6` |
| `gdn-resident-core` | 44.680 | 44.831 | 45.458 | 44.843 |
| `ffn` | 33.158 | 33.191 | 33.456 | 33.165 |
| **`sequence-core`** | **2.957** | **1.374** | **32.846** | **12.465** |
| `sequence-input-projection` | 13.973 | 13.925 | 14.206 | 14.031 |
| `sequence-output-projection` | 11.544 | 11.567 | 12.129 | 11.725 |
| eight untouched phases, summed | 107.628 | 107.846 | 109.543 | 108.056 |
| device span | 111.466 | 110.109 | 143.253 | 121.387 |
| step wall, untraced | 107.672 | 107.901 | 139.148 | **123.529** |

The control moves 0.2% at the short prefix and 1.4% at the long one, so the phase is **-53.6%** and
**-61.5%** normalised, and the step is **-11.2%** at a 960-token prefix and a null at 64.

**The prediction and the measurement agree on the mechanism, which is the part worth keeping.** The
byte argument said the phase should fall from 12.1 GB of requested K and V toward 2.0 GB, which at
the 242 GB/s roof is about 8.3 ms plus the fixed 2.9 ms this phase costs at a 64-token prefix. It
landed at 12.465. A six-fold cut in requested bytes bought a 2.6-fold cut in time, which is what a
phase sitting on its byte roof does and is not what a phase sitting on its issue roof does — the
same kernel, measured on the other shape, loses 18% when it halves its key traffic.

Residual FNV-64 `8087160186626784132` at the 64-token prefix and `10606832871390354142` at 960, in
every arm of both widths, traced and untraced.

## The whole ladder, one process, one clock

Both axes walked together at a 960-token prefix, 32 streams, mode 19
([`decode-ladder-960.json`](../../../data/bonsai2/batch-comparison/attn-row-groups/decode-ladder-960.json)).
`HG = 6` runs a one-row unit by construction, so the fourth arm is the third arm's code reached
through a different flag — a free consistency check, and it agrees to 0.16%.

| | deployed | row dispatch | head fold | both |
|---|---:|---:|---:|---:|
| `attn_narrow` / `attn_hg` | 0 / 1 | 1 / 1 | 0 / 6 | 1 / 6 |
| `gdn-resident-core` | 44.576 | 44.734 | 44.557 | 46.260 |
| `ffn` | 32.921 | 33.030 | 33.305 | 35.655 |
| **`sequence-core`** | **37.376** | **32.702** | **12.575** | **13.229** |
| `sequence-input-projection` | 13.892 | 13.989 | 13.965 | 15.219 |
| `sequence-output-projection` | 11.550 | 11.789 | 11.706 | 12.250 |
| eight untouched phases, summed | 107.177 | 107.825 | 107.842 | 114.141 |
| `sequence-core`, normalised | 1.000 | **0.870** | **0.334** | 0.343 |
| step wall, untraced | 142.495 | 139.351 | 118.230 | **118.045** |
| **aggregate generation, tok/s** | **224.6** | 229.6 | 270.7 | **271.1** |

**142.495 to 118.045 ms, -17.2%, and 224.6 to 271.1 aggregate tokens per second, +20.7%**, on a
32-stream generation step at a 960-token prefix. Residual FNV-64 `10606832871390354142` in all eight
samples.

The fourth arm's traced control drifted 6.5% and its phase number is inflated by exactly that; its
untraced wall is the one to read, and it is the third arm's.

## Prefill is a null, by construction and measured

A prompt pass makes groups of eight consecutive rows, so `attn_chunk_group` goes straight to `TT`
and `one_row` is false, so `HG` never fires. Mode 19, four samples per arm, two rounds
([`prefill-null.json`](../../../data/bonsai2/batch-comparison/attn-row-groups/prefill-null.json)):

| | 128 rows | | 256 rows | |
|---|---:|---:|---:|---:|
| | wide | narrow | wide | narrow |
| `ffn` | 81.675 | 85.257 | 154.612 | 165.823 |
| `sequence-input-projection` | 48.620 | 50.650 | 82.775 | 88.836 |
| `gdn-resident-core` | 18.163 | 18.975 | 33.926 | 36.192 |
| `sequence-core` | 4.867 | 5.083 | 11.867 | 12.611 |
| pass wall, untraced | 179.734 | 180.268 | 329.982 | 333.858 |

**Every phase moved by the same amount** — 1.042 to 1.058 at 128 rows and 1.062 to 1.074 at 256 —
which is the clock, not the change. Against the untouched phases `sequence-core` is +0.2% at 128
rows and -0.6% at 256, and the pass wall is +0.3% and +1.2% of the same drift. This is a panel that
would read as a 4% regression without its in-panel control.

Residual FNV-64 `7446865760224376151` at 128 rows and `18439807992172728808` at 256 — both hashes
already in the tree, from [the FFN load schedule](ffn-dense-loads.md) and
[the sequence weight address](sequence-weight-address.md), neither of which had this change.

## What is on by default

Both arms are on: `attn_narrow = 1` and `attn_hg = GQA`. Neither can fire on a shape with a
multi-row group, so a prompt pass and the drafted verify pass reach exactly the code they reached
before; neither changes a bit on any shape. `HALO_ATTN_NARROW=0` and `HALO_ATTN_HG=1` are the
controls, and `tools/batch_profile --attn-narrow`/`--attn-hg` walk them as case axes in one process.

Installed and accepted on the canonical build, 32 streams at a 960-token prefix, `HALO_ATTN_HG`
as the axis:

| phase | `HG = 1` | `HG = 6` |
|---|---:|---:|
| `gdn-resident-core` | 44.922 | 44.761 |
| **`sequence-core`** | **35.530** | **12.540** |
| `ffn` | 32.592 | 32.486 |
| eight untouched phases, summed | 99.280 | 99.034 |
| step wall, untraced | 135.344 | **111.877** |
| **aggregate generation, tok/s** | 236.4 | **286.0** |

Control 0.9975, phase **-64.6%** normalised, step **-17.3%**, aggregate generation **+21.0%**, and
residual FNV-64 `10606832871390354142` — the hash the checkout panels published for this prefix.

## What this leaves

**Ruled out for the decode shape.** Wider row groups, in any form. A generation step has one row per
sequence and merging across sequences is not available: different KV caches, different positions.

**The next axis, and it is the last cheap one.** `HG` is capped at `GQA = 6` because a seventh query
head belongs to another KV group and another chunk of cache. Above that the only remaining reuse is
*across chunks* — one unit walking several key chunks for the same heads — and that is the flash
shape [the long-context measurement](long-context-attention.md) rules out for exactness, because the
combine takes one global maximum and a flat weighted sum. It stays an explicit alternative with its
own quality panel.

**What the head fold did not reach.** At a 960-token prefix `sequence-core` is still 12.5 ms against
a 2.9 ms floor at a 64-token one, so about 9.6 ms of position-priced cost remains against a
shared-bytes roof of roughly 8.3 ms. That is close enough that the next gain here is not in this
phase: at these numbers a 32-stream step is back to being led by `gdn-resident-core` at 44.6 ms and
`ffn` at 32.9.

**The prefix is now part of every decode number.** `--decode-prompt` is a list and `--decode-context`
raises the KV allocation; a slot is 90112 bytes per token per sequence across the 22 cache slots, so
32 streams at 1024 is 2.95 GB of KV and at 2048 is 5.9 GB. That budget, not the engine, is what
bounds a long-context decode panel on this box.

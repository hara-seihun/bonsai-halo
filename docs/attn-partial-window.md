# The chunk partials are a working set, not a buffer

Attention in this engine is split in two kernels. A score unit owns one `(row group, query head,
128-key chunk)` and writes an `AttnPartial` — `{m, l, pad[2], acc[256]}`, **1040 bytes** — for every
row it carries. The combine then walks chunk 0..`pos/128` for each `(row, head)` and folds them into
one attention output.

That split is what makes the phase parallel, and the arithmetic in it has been read from several
sides. What nobody had asked is **how many bytes the split itself moves, and where they live between
the write and the read.**

    partial bytes per layer = rows x NH x chunks x 1040

A 128-row pass at position 3072 has 25 chunks and 24 query heads, so it turns **3.1 MB of attention
output into 79.9 MB of partials per layer** — 1.28 GB across the 16 attention layers, written once
by the score kernel and read once by the combine. The device's last-level cache is 32 MB. The
buffer is two and a half times larger than the cache that is about to be asked for it, so every byte
of it goes to DRAM and comes back.

[The long-context measurement](long-context-attention.md) saw the size of this and filed it under
the arithmetic: "at position 2944 a 128-row pass writes 77 MB per layer to produce 3.1 MB of
attention output — twenty-five times write amplification — and reading all of it back is 16% of the
phase. The score kernel, which reads the keys and values, is 84%." The 16% is right. What it does
not say is that **the 16% is a residency number**, and that the driver already has the knob.

## The driver already runs the loop

`src/batch.cpp` does not launch one score kernel and one combine per pass. It walks row groups in
**pairs**:

    for (g0 = 0; g0 < groups; g0 += gpl) { launch_attn_wide(g0, ng); launch_attn_combine_wide(...); }

so the partial bytes in flight between a write and its read are exactly one launch pair wide, and
`gpl` comes from `attn_batch_groups_per_launch` — the allocated partial budget divided by what a
group costs. The budget is `HALO_ATTN_PART_MB`, default **192 MB**. A 128-row pass at position 3072
needs 79.9 MB, which is comfortably under it, so `gpl` is every group in the pass, the loop runs
once, and the working set between the write and the read is the whole 79.9 MB.

The loop that would keep it in cache is the loop that is already there. It has simply never been
asked to run more than once on any shape this lane measures.

`attn_batch_set_part_window_mb` caps how much of the allocated buffer one launch pair may fill,
without touching the allocation:

    gpl = min(part_bytes / per_group, window_bytes / per_group)

Two arms therefore share one process, one allocation and one clock, and differ only in how many row
groups a score launch carries. `--attn-part-mb` is the in-process axis; `-1` leaves the process
default so no panel written before this exists reads differently.

## Exact by construction

A `(row, head, chunk)` partial is computed by a unit that sees one row group, one head and one
chunk. Nothing about it depends on which other groups share its launch:

- the keys and values it reads are addressed from the KV slot and the chunk, not from `g0`;
- its softmax runs over its own chunk's `count`, set by its own group's last row;
- the partial's address is `((row - rowbase) * NH + hq) * pstride + chunk`, and `rowbase` moves with
  the launch so each row lands where its own combine expects it;
- the combine's expression — a single global maximum over that row's chunks and then a flat weighted
  sum — reads exactly the same chunk list whatever produced it.

So the window changes launch geometry and nothing else. The control is the residual FNV-64 of the
pass, which must not move at any window.

There is a second control the shape gives for free. `per_group = 8 x NH x chunks x 1040` grows with
position, so a window only starts splitting once `16 x per_group` exceeds it — for a 24 MB window
that is about position 1000. **Below that position the arms are the same launch geometry and must
measure the same thing**, which separates a real effect from a clock ramp without any argument.

## What it costs, and the floor that is the whole result

Nothing is free: a window of `W` megabytes turns one score/combine pair per layer into
`ceil(footprint / W)` pairs, and each pair is two more kernel launches. These are plain dispatches,
not cooperative ones, so the boundary is the cheap kind — but the score kernel's grid is
`groups x NH x chunks`, and a launch that carries too few groups has no grid left to amortise its
own tail against.

`--attn-part-mb 1` measures that directly, because it drives `gpl` to one group everywhere:

| whole 2048-token walk, round 1 | score | combine | attention / control | prompt |
|---|---:|---:|---:|---:|
| `mb=0`, the deployed geometry | 199.7 ms | 47.7 | 0.17188 | 922.2 tok/s |
| `mb=12`, no floor | 216.5 (+8.4%) | **20.0 (-58.2%)** | **0.16430 (-4.41%)** | 923.9 |
| `mb=1`, one group per pair | 291.2 (**+45.8%**) | 39.5 (-17.3%) | 0.23346 (+35.8%) | 878.7 |

**At one group a pair the score kernel is +45.8% and the combine gives back two thirds of its own
gain**, because the combine's grid is `rows x 6` and eight rows is 48 workgroups. Both halves want a
grid; only one of them wants a small working set. So the window carries a floor in row groups
(`HALO_ATTN_PART_MIN_GROUPS`, four), which keeps the score grid at `4 x NH x chunks` and caps any
pass at four pairs however long the context grows. Without it a fixed megabyte window splits further
and further as the partials grow with position — at position 3968 a 12 MB window is already down to
one group and the score kernel is +19%.

A second correction the shape forces: a group's row count is not its nominal width. A prompt pass
hands eight-row groups and a generation step hands **one-row** groups, so sizing the window off
`group_rows` would charge a decode step eight times its real partials and split a working set that
was already resident. `attn_batch_begin` records the pass's rows and groups and the window uses
those, which makes it inert on every decode shape by arithmetic rather than by a special case.

## Results

Mode 19, 128-row passes, `HALO_ATTN_TRACE_SPLIT=1`, `tools/run-batch-compare --pin-clock`, both arms
alternating in one process on one clock and one allocation. The control is the eight phases the
change cannot reach. Raw in
[`attn-partial-window/`](../../../data/bonsai2/batch-comparison/attn-partial-window/README.md).

`--prefill-scan 3072`, two rounds, read round 1 (round 0 pays the first walk's page faults):

| position | score `mb=0` | score `mb=16` | combine `mb=0` | combine `mb=16` |
|---:|---:|---:|---:|---:|
| 0 | 1.15 | 1.15 | 0.46 | 0.47 |
| 512 | 7.57 | 7.57 | 0.84 | 0.84 |
| 1024 | 14.19 | 14.84 | 3.74 | **1.38** |
| 1536 | 20.42 | 21.76 | 5.22 | **1.84** |
| 2048 | 26.98 | 28.96 | 6.35 | **2.45** |
| 2560 | 33.21 | 35.71 | 7.59 | **2.96** |
| 2944 | 38.13 | 41.38 | 7.92 | **3.90** |
| **whole walk** | 473.4 | 505.6 (+6.8%) | 106.7 | **45.2 (-57.6%)** |

**Attention is -5.07% normalised over the walk and whole-document prompt throughput is 879.4 ->
886.0 tok/s, +0.75%.** The control sums are 2855.2 ms and 2855.6 ms — 0.01% apart, which is what
makes the attention ratio readable at all.

**Positions 0 through 512 are identical to the last two decimals in both arms**, because until the
pass's partials exceed the window there is nothing to split and the two arms launch the same
geometry. That is the panel's own null control and it costs nothing to take.

### What the two halves say about the phase

The combine is **not an arithmetic phase that happens to read memory; it is a memory phase**. The
same bytes, the same expression and the same grid per row run at 154 GB/s when they were written
79.9 MB ago and at 365 GB/s when they were written 16 MB ago. Two thirds of that kernel was waiting
for DRAM to hand back something the device had produced itself a few hundred microseconds earlier.

The score kernel's +6.8% is not lost cache reuse across row groups, which is what it looks like: the
`mb=1` control separates them. Its cost scales with **how few groups a launch carries**, not with
how many times a key chunk is re-read — one group is +45.8% where four groups is +6.8%, while the
chunk re-read count between those two arms differs by only a factor of four.

## What would take the other half

The floor is what stops this from being bigger: the combine's -57.6% is paid for to the tune of
6.8% by launch tails the split creates. Two things would collect it.

1. **Overlap.** `combine(k)` and `score(k+1)` are independent once the partial region is
   double-buffered, and the allocation is 192 MB against a 20 MB window, so the buffers are already
   paid for. Two streams and an event per pair would hide the tail the floor is currently buying
   back with residency.
2. **A smaller partial.** `AttnPartial` is `{m, l, pad[2], acc[256]}` = **1040 bytes**, and 1040 is
   not a multiple of the 128-byte line: `acc` lands at byte 16 of a partial whose base rotates by 16
   bytes per chunk, so seven chunks in eight have every wave's 128-byte store and 512-byte read
   straddling a line. Splitting the struct into a 1024-byte `acc` array and an 8-byte `{m, l}` array
   indexed the same way is **1032 bytes instead of 1040**, aligns every access, and turns the
   combine's first sweep — 4 bytes at stride 1040, once per chunk — into a contiguous 8-byte run.
   That is a bit-identical change to a phase now known to be memory-bound, and it is not built.

## Installed

Canonical `0a1f82c`, executable `cbba8d83f95d68f4d8df4084`. The installed acceptance is the deployed
drafted path, which this change reaches only through prompt ingestion and which therefore has to
reproduce itself exactly: `--bench --dflash q4`, 1697-token prompt, 48 greedy tokens, under
`tools/run-batch-compare --pin-clock`.

    prompt 1697 tokens 526.4 tok/s; 48 tokens 33.72 tok/s; 27 steps, 189 drafted, 21 accepted,
    1.78 tokens/step; draft 8.441 ms, verify 44.284 ms; digest 15165079467408989842

**Greedy digest `15165079467408989842` and 189 drafted / 21 accepted are the values the deployed
path has published across every build this week.** The prompt and generation rates are above the
last published pair on this panel (510.2 and 31.38) because several peers landed between them; the
digest is the part this change owes.

`bonsai-halo.service` was already stopped when this turn opened - `run-batch-compare` reported
"Unit bonsai-halo.service does not exist" on its first admission, so no hold in this turn had a
service to restore - and it is masked under another engineer's GPU hold as this is written. Every
wrapper invocation here restored exactly the state it found.

## What this does not touch

`ACHUNK`, `AttnPartial`'s layout, the score unit's arithmetic, the combine's arithmetic, the fold,
the coordinate, `attn_batch`'s allocation or budget, and every serving default. The window's own
default is 0, which is the geometry the engine has always launched.

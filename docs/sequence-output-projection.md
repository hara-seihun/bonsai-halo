# The sequence projections are memory-path bound, and four-bit activations will not fix that

> **The title is wrong and this document says why.** Four-bit activations on this exact stage
> measured **−54.4%** of the phase and −7.06% of a 128-row pass; see
> [the output-stage four-bit route](sequence-output-a4.md). The table below is correct — `iu8` and
> `iu4` both ask the operand path for 0.5 bytes per lane per cycle — and the inference from it is
> not. **Bytes per cycle is the invariant at the matrix-pipe boundary, and this kernel is at 158%
> of its issue model, which is the proof that it is below that boundary.** Below it the absolute
> bytes are what a wave waits on, so halving the operand halves the wait as well as the issue and
> the two compound. Everything else here stands, including both measured negatives.

Source `0b759ad`. Raw under
[`batch-comparison/seq-output/`](../../../data/bonsai2/batch-comparison/seq-output/).

`sequence-output-projection` is 21.8 ms of a 183 ms 128-row prefill pass and 7.9 ms of a 32-row one,
which is the same kernel, grid and instantiation a 32-stream generation step runs. Together with the
input stage the two projections are 37% of a prefill pass, second only to the FFN, and the output
half had never been read on its own: [the operand document](sequence-projection-operands.md) was
written from the input stage and closes by proposing four-bit activations for both.

This iteration reads the output stage, builds the two exact changes its shape suggests, and both
lose. The reason they lose is the useful part, because it also prices the four-bit proposal, and the
price is not the one the operand document quotes.

## Where the time is not

A serial issue model — 32 cycles for `v_wmma_i32_16x16x16_iu8`, one for every other slot, at the
2428 MHz [this box actually runs at](sequence-projection-operands.md#the-shader-clock-is-2428-mhz-not-2900):

| stage | shape | block slots | model | measured | over model |
|---|---|---:|---:|---:|---:|
| output, 128 rows | `TT=4`, `W=2` shared, 640 waves | 374 (32 WMMA) | 13.83 ms | 21.8 | **158%** |
| output, 32 rows | `TT=1`, `W=2` shared, 640 waves | 209 (8 WMMA) | 4.63 ms | 7.9 | **171%** |
| input, 128 rows | `TT=4`, `W=2` shared, 2048 waves | 374 | 35.72 ms | 45.9 | 129% |

So the output stage carries 8 ms of non-issue time at 128 rows and 3.3 at 32, and it is the worse of
the two by that measure — which agrees with the clock fit a peer ran over the four pinned ceilings,
where `sequence-output-projection` is the **least** clock-proportional phase in the engine at 59.6%
and `sequence-input-projection` the second most at 83.5%.

Occupancy is not the answer and was already ruled out: the published ownership table has all three
row-tile ownerships inside 1% at 128 rows, and the kernel holds 10 waves per SIMD32 where its grid
only supplies 8. DRAM is not the answer either: the output stage streams 503 MB of weights per pass,
2.1 ms at the 242 GB/s `bench/bw` measures.

## What the memory unit says

`rocprofv3` counters over one traced 128-row mode 19 pass, all kernels in the same run:

| kernel | `MemUnitBusy` | `L2CacheHit` |
|---|---:|---:|
| `project<4, false, true, ...>` sequence input | **97.8%** | 74.8% |
| `project<4, true, false, ...>` sequence output | **95.9%** | 47.2% |
| `resident_state<4, 1>` | 97.7% | 87.1% |
| `k_proj_opt<..., 2, 4, false, 0>` FFN down | 87.9% | 50.7% |
| `k_proj_opt<..., 4, 2, true, 2>` FFN gate/up | **78.9%** | 56.6% |
| `head_tile<2>` | — | 93.0% |

`resident_state` is the calibration point: it is independently known to move 9.36 GB per step at
224 of 242 GB/s, and it reads 97.7%. Both sequence projections sit on top of it. The FFN, the
largest phase in the pass, sits 10 to 19 points below.

**Read `MemUnitBusy` carefully.** It counts stall time as busy, so it says the memory unit has
outstanding work essentially always, not that its return path is saturated. The counter that
separates them, `MemUnitStalled`, appears in `rocprofv3 --list-avail` and is then rejected for this
agent: `Unable to find all counters for agent 1 (gpu-0, gfx1151) in [MemUnitStalled]`. The same is
true of `WriteUnitStalled`. Both names are now accepted by `tools/run-batch-compare` so the next
engineer finds that out in one command instead of reading the runner. **On gfx1151 the busy/stalled
split is not available, and any claim of bandwidth saturation on this device rests on bytes you
counted yourself.**

**The bytes are counted now, and this table's open question is answered.**
[`cache-policy.md`](cache-policy.md) reproduces both hit rates on a later build (74.27 and 51.43)
and adds calibrated `FETCH_SIZE` for the same dispatches. The output stage fetches **11.37 MiB a
layer against a floor of 8.72** and the input stage 25.97 against 21.88 — 1.30x and 1.19x — while a
whole 128-row pass fetches about **8.3 GiB in 176 ms, close to 50 GB/s against a 242 GB/s roof**.
Neither projection is anywhere near a bandwidth limit, so the 8 ms of non-issue time in the table
above is latency and not traffic, and the obvious repair is rejected there as well: a cache-policy
bit on the weight stream, which turns out to be a cache *hit* at this shape rather than a polluter.
Calibrate that counter before quoting it — it undercounts by exactly two on this device.

## The arithmetic that makes this matter

A `v_wmma_i32_16x16x16_iu8_w32` takes a 16-byte B operand per lane and retires in 32 cycles. To keep
one SIMD32's matrix pipe fed, the memory path must return 0.5 bytes per lane per cycle — 256
distinct bytes per wave-instruction, **1.55 TB/s across 80 SIMD32 at 2428 MHz**, for the activation
operand alone. The weight operand is nearly free beside it: 32 bytes per lane per 128-block feed
`8 * TT` matrix instructions, one byte per lane per instruction at four token tiles against sixteen.

Now put `v_wmma_i32_16x16x16_iu4` beside it. Eight bytes per lane, 16 cycles:

| instruction | B bytes/lane | cycles | **B bytes/lane/cycle** | device-wide to saturate |
|---|---:|---:|---:|---:|
| `iu8` | 16 | 32 | **0.5** | 1.55 TB/s |
| `iu4` | 8 | 16 | **0.5** | 1.55 TB/s |

**Four-bit activations halve the matrix cycles and halve the bytes in exactly the same proportion,
so they ask the memory path for precisely what eight-bit activations ask for.** K is 16 either way.
A kernel held by its matrix pipe gets 2x from that swap; a kernel held by its operand path gets
nothing, and these two are the two kernels in this engine closest to their operand path.

**That last sentence is the error, and it was measured wrong by 30 points.** A kernel *at* the
matrix-pipe boundary gets nothing, because there the ratio is what binds. A kernel *below* it — one
whose waves are waiting on bytes that have not arrived, which is what "158% over the issue model"
in the table above means — is waiting on an absolute number of bytes, and halving them halves the
wait. Both terms then improve at once and the measured gain is larger than either model predicts:
[**21.707 to 9.903 ms at 128 rows**](sequence-output-a4.md), against 16.6 from the cycle ratio
alone.

That is a correction to the closing recommendation in
[the operand document](sequence-projection-operands.md#what-to-attack-next-here), which prices four-bit
activations at "roughly a 45% cut of both stages, about 10% of a prefill pass and a fifth of a
generation step". The cut is real in matrix cycles and is not obviously real in wall clock, and the
work it asks for is a quality panel for a numerical map that feeds attention and the recurrent
state. **Do not start that build on the arithmetic alone.**

What the same table says positively: `k_proj_opt` reads 78.9% on the identical instrument while
running `iu4`, because it carries **two weight row tiles against one activation fragment** — gate and
up — which is 0.25 bytes per lane per cycle, half of what `project<>` asks. The bytes per matrix
instruction, not the width of the matrix instruction, is the axis with room on it. Four-bit
activations paired with two row tiles per fragment would move both terms; four-bit activations alone
move one term that is not binding.

## Two exact changes built on this shape, and both lose

Both are bit-identical by construction and measured so: the residual FNV-1a over every FP32 output
of all 64 layers is `1873717996769712938` at 32 rows and `7446865760224376151` at 128 in every arm
of both panels. Both were ordered against the deployed shape **inside one process** through a new
`--seq-sched 6` arm, with five phases the change cannot reach as the in-panel control. Neither is in
the tree. The patches are kept beside the panels.

### The line epilogue: four times fewer epilogue memory accesses, no clock

The matrix instruction leaves output row `2r + half` in lane `(col, half)`, so sixteen consecutive
rows of one token live as eight registers across a lane pair, interleaved. The deployed epilogue
stores them in register order, which compiles to exactly this:

```
global_store_b32 v[9:10], v11, off
global_store_b32 v[9:10], v12, off offset:8
...                                offset:56
```

Eight dword stores at an eight-byte stride, each using four bytes of a 64-byte destination line, and
with `ADD` eight loads of the same line as well. One `permlanex16` per register hands each lane four
*consecutive* rows instead — lane `half` takes rows `8m + 4*half .. +3` — and the pair leaves in two
`b128` accesses per eight rows. Per wave at four token tiles, 32 stores and 32 loads become 8 and 8.
The exchange is a relabeling of which lane holds which float between the drain and the store; the
values, the additions and the destination addresses are untouched.

It costs nothing to hold: 135 VGPR and ten waves per SIMD32 at `TT = 4` against the deployed 135 and
ten, 78 and sixteen at `TT = 1` against 78 and sixteen. (A kernel carrying *both* epilogues is
allocated their union, 161 and nine — which is why the arm is a template parameter and not a kernel
argument, and worth knowing before anyone builds a runtime-switched control into a tight kernel.)

Mode 19, two rounds, arms interleaved, `--seq-sched 1,6`:

| rows | `sequence-output-projection` | untouched phases | corrected |
|---:|---|---|---:|
| 128 | 23.175 → 22.858 ms | `ffn` -1.82%, `gdn` -1.55%, `head` -0.19% | **+0.2%** |
| 32 | 8.165 → 8.155 ms | `ffn` +0.90%, `gdn` +2.03%, `head` +1.68% | **-2.0%** |

The two row counts disagree in sign after their own controls and the whole spread is inside what
this box gives, so it is a null. The mechanism says it should be: the epilogue is about 1% of the
memory accesses the kernel issues, against 32 fragment loads per 128-block in the body.

**What it rules out.** Partial-line write amplification is not a cost in this kernel, and the same
accumulator layout and the same register-order epilogue are in `head_batch.hip` and in every
`k_proj_opt` drain. Nobody needs to build this again for any of them. Patch:
[`seq-output/line-epilogue.patch`](../../../data/bonsai2/batch-comparison/seq-output/line-epilogue.patch),
panel `line-epilogue.json`.

### The block cursor: the medicine that paid 28% on the FFN costs 8 to 28% here

The block loop asks for block `b`'s 32 code bytes and its row scale at the top of block `b` and
waits on them a few instructions later, so a wave pays a memory round trip per block that only the
other waves on its SIMD32 can cover — and the output stage's 320 row tiles give it eight of them.
[`docs/decode-dispatch.md`](decode-dispatch.md) took the FFN's down stage 13.7 → 9.8 ms with exactly
this diagnosis, by holding blocks in flight in registers. Doing the same here, one block of
lookahead, is *cheaper* in registers at four token tiles (132 against 135) and identical in
occupancy at both widths.

| rows | `sequence-output-projection` | `sequence-input-projection` | untouched drift |
|---:|---:|---:|---:|
| 128 | 22.205 → 24.009 ms, **+8.3%** | 46.377 → 45.514, -1.7% | x0.998 |
| 32 | 7.868 → 10.110 ms, **+27.9%** | 14.758 → 17.148, +15.7% | x1.004 |

Five untouched phases move less than 1% in both panels, so the effect is the change.

**Why the same medicine reverses sign.** In `k_proj_opt`'s dense arm the weight stream *is* the
dominant operand — five-trit bytes peeled per lane, with one activation fragment serving both gate
and up — so putting it in flight early is putting the critical operand in flight early. In
`project<>` the mix is inverted: a 128-block issues **32 fragment loads against 2 weight loads** at
four token tiles. The cursor aims at 2 of 34 accesses, and holding their results live across the
whole slice loop disturbs the scheduling of the 32 that matter. At one token tile, where the block
is 209 slots and there is nothing to hide behind, it costs 28%.

The transferable rule: **before putting an operand in flight, count which operand the block is
actually made of.** A lookahead is not free scheduling; it is register liveness spent on one stream
at the expense of the scheduler's freedom with the others. Patch:
[`seq-output/block-cursor.patch`](../../../data/bonsai2/batch-comparison/seq-output/block-cursor.patch),
panel `block-cursor.json`.

## What to attack next here

The fragment stream is the operand this kernel is made of, and three shapes have now tried to make
it cheaper: row-tile pairing within a wave (`RT = 2`, +7.3% at 128 rows and +97.5% on the 32-row
output stage, because it halves the row-tile grid and doubles the expansion), tile ownership (a null
against sharing), and the two changes above. None of them reduced fragment loads per matrix
instruction *without* also paying for it somewhere else.

The one form that has not been tried keeps the grid and keeps the accumulator width: under the
**tile** ownership the `W` waves of a workgroup hold different row tiles and the *same* token group,
so they read the *same* fragments, `W` times, through the memory unit. Staging a block's fragments
into LDS once per workgroup and reading them with `ds_read` would cut those global accesses by `W`
and move the traffic to a pipe the counters say is idle, at 8 KB of LDS per workgroup at four token
tiles and a barrier per block. That is the structure `k_proj_opt` gets for free by carrying two row
tiles, expressed as workgroup sharing rather than register sharing, and it is the only way left to
change bytes per matrix instruction without halving the row-tile grid.

Price it against this: the barrier per block is the thing that killed a 512-thread variant in
[the wide head](wide-head.md), and 48 barriers per layer per workgroup is not obviously cheap.

> Read [the operand document](sequence-projection-operands.md) for the block loop itself, and
> [what binds the sequence projections](sequence-fragment-reuse.md) for the fragment-reuse negative
> and the issue-cycle price of a block-loop slot.

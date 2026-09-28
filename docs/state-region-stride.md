# The packed state's container is still fp32-shaped, and that is what pins the batched width

A sequence slot costs 237 MiB at context 256, and **144 MiB of it is recurrent state that an int8
process never writes**. `GDN_STATE_FLOATS` is the fp32 state's size, and until now it was also the
stride the engine allocated and every kernel addressed with in *every* coordinate. A packed head is
a quarter of an fp32 head at eight bits and a half at sixteen, so an int8 engine reserved 3.000 MiB
per (slot, layer) to hold 1.150.

That reservation, not throughput, is what has been pinning the batched operating point:

- [Sequences per generation step](decode-streams.md) reaches 597 tok/s at 64 streams and names the
  wall - "64 slots with both A4 images is 30.6 GB, which is where `hipMalloc` actually failed".
- [Generation at 64 and 128 streams](generation-128.md) needed a 43 GiB process budget and explicit
  managed allocation to exist, and says why: "INT8 state reduces traffic but the current engine
  still reserves FP32-sized state regions."

## The arithmetic

A region holds the values, one shared exponent per (head, row), and the deferred commit's control
word and pending triples. Only the values change width with the coordinate:

| | value floats | scales | defer ctl + triples | used | region, rounded to 4 KiB |
|---|---:|---:|---:|---:|---:|
| fp32 | 786,432 | - | - | 786,432 | 786,432 = 3.000 MiB |
| 16-bit | 393,216 | 6,144 | 98,692 | 498,052 | 498,688 = 1.902 MiB |
| **8-bit** | **196,608** | **6,144** | **98,692** | **301,444** | **302,080 = 1.152 MiB** |

The old layout put the scales at `GDN_STATE_FLOATS/2` because that is where a 16-bit head ends, so
an int8 region also paid for the quarter between its values and that offset. `gdn_fmt_scale_off`,
`gdn_fmt_defer_ctl`, `gdn_fmt_defer_off` and `gdn_fmt_region_floats` in
[`kernels/halo_kernels.h`](../kernels/halo_kernels.h) are `constexpr` functions of the format, so
inside every codec branch that has already switched on the format they are the same literals they
were before; `r_gdn_region` beside `r_gdn_fmt` carries the stride to the kernels the way this file
has always carried process-wide state settings.

Per sequence slot at context 256, int8: **state 144.0 -> 55.3 MiB, slot total 237 -> 148 MiB.**

## What it buys, measured

`HALO_GDN_REGION=packed` selects it; the default is the fp32-sized region this engine has always
allocated (see the acceptance section below for why). Device bytes are the engine's own tracked
totals, same build, same flags, only the selector moving:

| configuration | full region | packed region |
|---|---:|---:|
| 32 slots, ctx 256, after load | 14.72 GB | **11.37 GB** |
| 64 slots, ctx 256, after load | 22.50 GB | **16.18 GB** |
| 64 slots + **both** A4 images, after prepare | 30.63 GB - **refused** | **26.37 GB - runs** |
| 128 slots, pair-only image, after prepare | 43.9 GB (the published panel) | **32.00 GB** |

**The 64-slot cell is the one that changes what the lane can run today.** Both A4 images at 64 slots
is exactly the configuration `hipMalloc` refused; under the default 28 GiB tracked budget the
control does not get past `prepare_batch` ("would exceed 30064771072 tracked device bytes from a
current 22503820360"), and the packed region runs it inside the same budget with ordinary
allocations. That restores the image rule at 32 rows, which the 64-slot runs had been giving up:

| streams, 64 slots | pair-only image | both images |
|---:|---:|---:|
| 32 | 403.0 / 446.4 | **464.2 / 483.8** |
| 64 | 609.2 / 607.9 | **616.5 / 610.7** |

Mode 19, i8 state, defer 4, a4 sequence input, context 256, 16 greedy steps per timed region, two
rounds, `--pin-clock`, 128 distinct prompts. For scale: the installed canonical build published
440.3 and 597.5 tok/s in those two cells with the pair-only image it had to use.

**128 streams, which had not been re-taken since `1c7d7f2`:** 615.4 tok/s aggregate (4.81 per
stream) against 578.0 at 64 in the same process, with `HALO_MANAGED_ALLOC=1` and a 33 GiB budget.
The published 128-stream number was 517.7. It still needs managed allocation because 32.00 GB is
above the ~30.6 GB this host's ordinary allocator reaches, and the 64-stream cell in that same
managed process reads 578 where an ordinary-allocation process reads 609 - **managed allocation
costs about 5%, so the 128-stream cell above is a floor, not the shape's real rate.** The next
section says what would remove it.

## Bits

The change moves addresses. It moves no value, no operand order, no summation tree and no rounding,
and the two shapes the lane measures say so:

- **Batched generation, 32 streams, i8 + defer 4 + a4, 16 steps:** all **32 of 32** streams'
  generated tokens identical to the canonical build.
- **Prompt ingestion, 128-row passes, mode 19, i8 + defer 4:** `tail_logits_fnv64`
  `7860994932085604082` identical to canonical over 128 rows of full-vocabulary logits, and the
  traced residual `10342843363571155848` identical in all four traced passes of both builds.
- **fp32 is untouched by construction**: `gdn_fmt_region_floats(GDN_STATE_F32)` is
  `GDN_STATE_FLOATS`, so a default serving process allocates and addresses exactly what it did.

Timing at a fixed slot count is a null that this box cannot resolve across processes: the phase
trace makes the packed arm 1.2% slower on the `gdn-resident-core` ratio while a prefill TPS panel
makes it 4.2% faster, and both are cross-process (a stride is frozen per process, so no in-process
pair exists). Every phase in that trace moved together, including the three this change cannot
touch.

## The acceptance that is not green, and it is not this change

`tools/run-batch-compare --state-check` (mixed commit/replay/ragged-slice/rollback) **fails at every
packed coordinate on current main, with or without this change.** The bisect:

| build | fp32 | i8 / i16 packed coordinate |
|---|---|---|
| canonical objects as built 19:35 | pass | **pass** |
| this branch's base `598ff76`, no change | pass | **fail** |
| base + this change, full region (default) | pass | **fail** |
| base + this change, packed region | pass | **fail** |
| current main + this change, either region, `--seq-quant a8` or the default | pass | **fail** |

It fails at the **first committing pass** (`mode 18`, 2 rows, one sequence: logits differ where the
never-switching reference and the commit twin must agree), and it is insensitive to the state
stride, to the region layout, to the deferred commit and to the sequence operand. So the regression
is somewhere in `598ff76`'s own window rather than in this work, and it is worth someone's turn:
that acceptance is the only instrument in the tree that checks the commit/replay contract, the
deployed drafted decode path mixes exactly those routes, and **it is currently red on main for every
packed state coordinate.** Reproduce with

```sh
HALO_GDN_STATE=i16 tools/run-batch-compare --memory-gib 30 --state-check --references 17 \
  --no-rollback --shapes 1:0,2:0,5:0 --out /tmp/sc.json
```

Because that instrument cannot currently certify a packed-coordinate change, **the packed region is
a selector and not the default.** Flipping the default is one line in `gdn_region_for` and wants the
mixed acceptance green first.

## Reproducing

```sh
# capacity and throughput, 64 slots with both A4 images, ordinary allocations
HALO_GDN_REGION=packed tools/run-batch-compare --pin-clock --memory-gib 30 \
  --tag state-region-stride/stride-64-both --only decode --modes 19 --streams 32,64 --slots 64 \
  --context 256 --a4-images both --gen-steps 16 --rounds 2 --gdn-state 3 --gdn-defer 4 \
  --seq-quant a4 --warmup-steps 2 \
  --prompts ../../data/bonsai2/batch-comparison/tps128-20260921-2306/prompts.txt

# 128 streams
HALO_GDN_REGION=packed HALO_MANAGED_ALLOC=1 HALO_SNAPSHOT_KV_GB=0 \
  tools/run-batch-compare --pin-clock --memory-gib 33 --host-reserve-gib 5 \
  --tag state-region-stride/stride-128 --only decode --modes 19 --streams 64,128 --slots 128 \
  --context 256 --a4-images wide --gen-steps 8 --rounds 1 --gdn-state 3 --gdn-defer 4 \
  --seq-quant a4 --warmup-steps 2 --prompts .../prompts.txt
```

A process that sweeps coordinates has to say so before the engine allocates, because a region's
start is an address and the slots already hold state at whichever stride they were written under:
`gdn_state_reserve_format(fmt)` widens the reservation and `gdn_state_boot_format(fmt)` picks the
one the process starts in. Both measurement tools do this from their own `--gdn-state` list; asking
for a wider coordinate after the freeze aborts with the reason instead of addressing the wrong
bytes.

Accepted on the installed canonical build `e8607afbd2fb6df7ed3f19fcf3f97a31` (`a696625`): the same
binary run with and without `HALO_GDN_REGION=packed` generates **32 of 32 streams' tokens
identically** and reports 14.72 GB against 11.37 GB of tracked device memory at 32 slots
(`installed-full/` and `installed-packed/`).

Raw samples, the failing acceptance JSONs and every bisect cell are in
[`batch-comparison/state-region-stride/`](../../../data/bonsai2/batch-comparison/state-region-stride/).

## What is left in a slot, for whoever takes the next cut

At context 256 with the packed region a slot is 148 MiB: **state 55.3, `blk_cache` 60.3, KV 16.0,
conv ring 5.6.** `blk_cache` is now the largest item and it is `[slot][2][48][RMAX][BLK_TOKEN_FLOATS]`
- eight rows per (slot, parity, layer) whatever the pass shape - while a batched generation step
carries **one** row per sequence and the wide route only reads that buffer to replay an uncommitted
prefix. Sizing it at runtime would take a 128-slot engine from 32.0 GB to about 24.3 GB, which is
inside ordinary allocation, which is worth about 5% of the 128-stream cell on this evidence. It
needs a guard: `blk_tok` indexes with `RMAX` as a literal and the eight-row persistent path writes
`S.nrows` rows, so a smaller capacity has to be refused rather than wrapped.

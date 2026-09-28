# The weight stream is a cache hit, not a cache polluter

No load in this engine had ever set a cache policy bit. `grep` for `nontemporal`, `slc`, `glc` or
`dlc` across `kernels/`, `src/`, `tools/` and `docs/` returned nothing: every `global_load` runs the
default policy, which caches a byte that will never be read again and evicts one that will be read
two thousand times. That reading of the sequence projections was wrong in both halves, and the
counters say so to the kilobyte.

**The result is a rejection.** Marking the weight stream non-temporal is a wash at 128 rows and
costs 1.2% of a 32-stream generation step, at a pinned clock. The arms are not in the tree; the
patch is in [the raw directory](../../../data/bonsai2/batch-comparison/seq-cache/README.md) and the
reason is below, because the reason rules out a family of changes rather than one of them.

## What was built

`__builtin_nontemporal_load` emits `slc dlc` on gfx1151 — stream through the device L2, bypass the
shader-array L1, keep the L0 hit. Two arms of `project<>`, reachable as `--seq-sched 8` and `9`:

- **`SEQ_OP_LANE_NT`** puts that bit on the weight stream: the two `global_load_b128` of code words
  and the block scale.
- **`SEQ_OP_LANE_NTB`** puts it on the activation fragments instead. This is the **wrong stream by
  the hypothesis**, and it is what makes the panel readable: an arm that moves nothing in either
  direction is a fact about the device, and an arm that hurts one way and helps the other is a fact
  about the streams.

Both compile to **39 loads and 135 VGPRs**, exactly what the deployed kernel compiles to; only the
policy bit moves. All **420 baseline instantiations** in `sequence_batch.hip` were byte-identical
after basic-block label normalisation, so the control arm is the deployed kernel rather than a
re-expression of it. Residual FNV-64 `7446865760224376151` in every arm at 128 rows.

## Calibrate the counter first, because it is not bytes

**`FETCH_SIZE` on gfx1151 undercounts by exactly two.** `tools/profile_read_check` reads 1 MiB,
256 MiB and 1 GiB three times each; the counter returns 512.0, 131,072.1 and 524,288.1, so
`bytes = value * 1024 * 2.000` with a spread of 1.0009 across all nine dispatches. Every figure
below is calibrated. The first draft of this document was not, and its absolute claims were wrong
by that factor — the geometry is what caught it. A recurrent layer's input projection has to fetch
at least its own image, 16384 rows x 5120 K at two bits is 20.0 MiB, and the raw counter said 13.0.
**A dispatch cannot fetch less than the image it reads exactly once.**

## What the streams actually cost, against the floor their geometry sets

Per layer-dispatch, mode 19, one process, MiB:

| dispatch | floor (image + operand, once) | deployed | over-fetch | NT weights | NT fragments |
|---|---:|---:|---:|---:|---:|
| sequence input, 128 rows | 21.88 | **25.97** | 1.19x | 33.66 | 59.52 |
| sequence output, 128 rows | 8.72 | **11.37** | 1.30x | 14.14 | |
| sequence input, 32 rows | 21.41 | **21.54** | **1.01x** | 21.41 | |
| FFN gate/up, pair, 128 rows | 22.89 | **52.21** | **2.28x** | | |
| FFN down, pair, 128 rows | 23.64 | **26.37** | 1.12x | | |
| FFN gate/up, dense, 32 rows | 18.68 | **39.51** | **2.11x** | | |
| FFN down, dense, 32 rows | 18.87 | **19.78** | 1.05x | | |

**Three facts fall out and they are the whole result.**

1. **The weight block was being fetched once and read twice, and the caches were doing most of that
   for free.** At 128 rows the rule selects `SHARE = true`: two waves of a workgroup walk the same
   row tile and differ only in token group, so each weight block is read by both. Forcing the
   stream to no-allocate adds **7.70 MiB** to the input stage, 38% of a second read of its 20.0 MiB
   image, and 2.77 MiB to the output stage. The stream this change was designed for — one with no
   reuse — does not exist at this shape.
2. **The weight stream was not evicting the activation working set.** The deployed input dispatch
   fetches 25.97 MiB where its weights and scales alone are 21.25, so **everything else in it,
   activation re-fetch included, is at most 4.09 MiB** — and at 32 rows the same stage runs at
   **1.01x its floor**, which is as close to perfect residency as a counter can show. There was
   nothing for the weight stream to evict.
3. **The 32-row row is the control that had to pay and did not.** That shape is `SHARE = false`,
   each weight block is read by exactly one wave, the stream really has no reuse — and the policy
   changes fetched bytes by **-0.6%**, which is nothing.

`L2CacheHit` on the same build, 128 rows, reproducing the published panel almost exactly (74.8 and
47.2 were measured a day and many commits earlier):

| kernel | deployed | NT weights | NT fragments |
|---|---:|---:|---:|
| `project` input | **74.27%** | 66.55 | 87.27 |
| `project` output | **51.43%** | 46.31 | 89.55 |
| `k_proj_opt` FFN gate/up (pair) | 56.25 | | |
| `k_proj_opt` FFN down | 51.28 | | |
| `resident_state<4,1>` | 87.13 | | |
| `k_forward_rows<8>` | 89.20 | | |

## The times, and a cross-phase coupling that survives a pinned clock

32 streams, one token each, `--gdn-state 3 --gdn-defer 0 --decode-prime 1`, arms in the palindrome
order `1,8,9,9,8,1` inside one `decode_setup`, four traced samples per arm, medians:

| phase | deployed | NT weights | NT fragments |
|---|---:|---:|---:|
| `sequence-output-projection` | 7.240 | **6.760 (-6.6%)** | 7.425 |
| `sequence-input-projection` | 10.538 | 10.470 | 10.818 |
| **`ffn`** (cannot be reached by this change) | 32.505 | **33.744 (+3.8%)** | 32.517 |
| `gdn-resident-core` | 17.740 | 17.771 | 17.711 |
| step, wall | **75.58** | 76.49 | 76.06 |

Pinned flat at 2400 MHz, the same ordering reproduces: output projection 7.427 to **7.011**, `ffn`
33.179 to **34.406**, step 77.56 to 78.52. **The FFN penalty is not the DPM governor.** It is the
same sign and the same size in every sample of both panels, in a kernel whose loads, image and
launch this change does not touch, and the only thing crossing between them is the state the memory
system is left in. Nothing available on this agent separates it further: `MemUnitStalled` and
`WriteUnitStalled` are rejected for gfx1151.

At 128 rows the same arms are a wash — device span 162.967 / 158.783 deployed against 163.319 /
155.443, both orders, and the untouched phases move together by 2-3% between rounds.

The wrong-stream arm behaves exactly as a cache arm should: **+8%** on both projections at 128 rows,
+2.5% at decode, for 2.8x the activation traffic. The instrument could see a cache effect of a few
percent. It did not see one in the direction the hypothesis needed.

## What this rules out, and the one place it points at instead

**Do not spend another turn on cache policy in these kernels** unless you first show your stream has
no reuse at the shape you are measuring. Three of the four (stage, width) cells have weight reuse
the caches are already serving, and the fourth is already at its floor.

**The sequence projections are not where the traffic is.** They run at 1.01x to 1.30x of the bytes
their geometry demands, so the family of changes that removes activation re-fetch — LDS staging, a
deeper row-tile group, a wider token group — has **at most 4.09 MiB a dispatch to win on the input
stage and 2.65 on the output stage**. The conversion rate for that currency is measured here too:
the wrong-stream arm *adds* 33.6 MiB a dispatch and costs 8% of the phase, so removing 4.09 is worth
about **1%**. Price any such change against those two numbers before building it.

**The FFN gate/up stage is the exception, at both widths, and it is not its weight stream.** It
fetches **2.28x** its floor with pair codes at 128 rows and **2.11x** with the dense image at 32
rows: 29.3 and 20.8 MiB a layer-dispatch beyond what its image costs, which is **more than the
entire weight image it exists to read**. The 32-row instantiation is the one that names the cause,
because `shape_for` gives it `SHARE = false`: every wave owns its row tile, no weight block is read
twice, and there is no cross-wave weight re-read available to explain the excess. What is left is
the activation image — 0.31 MiB at 128 rows, read once per weight row tile, 356 MiB of requests a
layer — and about 8% of those requests are reaching memory. The down stage, with a quarter of the
row tiles, is at 1.05-1.12x.

So **the activation-residency argument that fails in the sequence projections is live in the FFN**,
with 21 to 29 MiB a layer-dispatch on the table and a measured conversion of roughly 8% of a phase
per 33 MiB. That is `k_proj_opt`'s owner's to take, and the panel that decides it is a `FETCH_SIZE`
pair, not a stride argument: a weight-image *order* change cannot move a number whose cause is the
B operand.

A whole 128-row pass fetches about **8.3 GiB at the L2 boundary in 176 ms, close to 50 GB/s against
the 242 GB/s `bench/bw` measures**: FFN gate/up 3.34 GiB, FFN down 1.69, sequence input 1.66,
sequence output 0.73, `resident_state` 0.43. **The prefill shape is not near the memory roof
anywhere**, which is the frame this iteration started by getting wrong and the reason a policy bit
was never going to be worth much in it.

## Reproduce

```sh
# the two panels, arms interleaved in one process
tools/run-batch-compare --profile-tool --modes 19 --rows 128 --heads 0 --traces 1 --rounds 2 \
    --seq-sched 1,8,9 --out DIR/prefill128.json
tools/run-batch-compare --clock-fix 2400 --profile-tool --modes 19 --decode-streams 32 --heads 1 \
    --traces 1 --rounds 2 --seq-sched 1,8,9,9,8,1 --gdn-state 3 --gdn-defer 0 --decode-prime 1 \
    --decode-prompt 16 --warmup-ms 0 --out DIR/decode32-pinned.json
# the counters, all kernels of the run in one file
BONSAI_PROFILE_COUNTERS=FETCH_SIZE BONSAI_PROFILE_OUTPUT=DIR/fetch \
tools/run-batch-compare --profile-tool --modes 19 --rows 32,128 --heads 0 --traces 0 --rounds 1 \
    --seq-sched 1,8,9 --warmup-ms 0 --out DIR/fetch/run.json
# the calibration, which is not optional: seconds of GPU, three known read sizes
BONSAI_PROFILE_COUNTERS=FETCH_SIZE BONSAI_PROFILE_OUTPUT=DIR/calib \
tools/run-batch-compare --profile-read-check
```

`--seq-sched 8` and `9` need
[`seq-cache-policy.patch`](../../../data/bonsai2/batch-comparison/seq-cache/README.md) applied
first; without it the axis runs 0..7 and the decode loop below still works.

## The instrument this leaves, which is worth more than the arm

`tools/batch_profile` never carried `--seq-sched` into its decode loop, so **no arm of the sequence
projection's schedule axis had ever been run at a generation shape** — the ownership rule, the slice
barrier, the operand path and the two policy arms were all prefill-only facts. It does now, and the
arms run inside one `decode_setup` rather than paying for their own, which puts them seconds apart
on one prefilled batch instead of a full setup apart.

**The trap that costs the first panel, because it cost this one.** With `--gdn-defer 4` consecutive
steps are *different work*: the arms then read the deferred-commit cycle rather than the change, and
the first run of this iteration reported `gdn-resident-core` moving 14.1 to 18.3 ms and three
different residual hashes for a change that cannot touch either. Use `--gdn-defer 0` so every step
commits, `--decode-prime 1` so the timed step is steady state, and a palindrome arm list so each arm
gets an early and a late slot. The residual hash is a control *across arms* only in the prefill
panel, where every arm runs the same pass; at decode each step advances the batch, so bit-identity
has to be established at 128 rows and carried.

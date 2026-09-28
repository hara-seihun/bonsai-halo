# The packed state's memory coordinate, and the term that binds its phase

[The storage coordinate](gdn-state-pack.md) closed the recurrent state's *precision* question and
left a bounded negative behind it:

> within any schedule of the existing gated-delta recurrence that keeps FP32 state in memory between
> layers, nothing you do to its arithmetic, lane ownership, unit split or synchronization can take
> that 48 ms below about 40.

That was true, and it was true **of fp32 only**. Fp32 moved 9.664 GB per 32-stream step at 224 GB/s
of a 242 GB/s roof: byte-bound and finished. int8 moved the same round trip in 2.415 GB and took the
phase to 17.74 ms — **4x fewer bytes and 4x fewer requests for 2.7x the time** — so the packed arm
was no longer byte-bound and nothing had said what it was instead.

It is neither bytes nor issue slots. It is the **number of memory instructions a wave issues**, and
the fix is a permutation of where a row lives.

## What was never counted

Two instruments had already ruled out the usual answers for this phase, and both were pointed at the
token loop. [The recurrence's row batching](gdn-token-reduce.md) removed **24.5% of the token body's
issue slots** for −15.1% in prefill and an exact null at 32 streams, and
[the occupancy probe](../tools/gdn_occ_probe.hip) reports 8 blocks per WGP and 16 waves per SIMD32
for *every* `resident_state` instantiation at 38–61 VGPR, which is this device's workgroup ceiling.

The codec was the part nobody had counted. A generation-shape unit loads `R` rows, walks **one**
token, and stores `R` rows. `tools/gdn_state_isa.py` (new, eight seconds, no GPU lock, no model)
instantiates `gdn_state_load_row` and `gdn_state_store_row` alone with the format as a compile-time
constant:

| SPLIT 4, int8 | instructions | basic blocks | `s_delay_alu` | `s_waitcnt` | state + scale requests |
|---|---:|---:|---:|---:|---|
| `i8c`, the coordinate until today | 511 | 4 | 119 | 8 | 8x `global_load_b32`, 8x `global_store_b32` |
| `i8s`, full-exec scale store | 379 | **1** | 48 | 8 | unchanged |
| `i8`, accepted | **368** | 1 | 53 | **4** | **1x b128 + 1x b96 + 1x b32**, 8 stores |

`gdn_token` is 166 slots. **So at one token per unit the codec is 75% of the unit and the token loop
is 25%**, and removing a quarter of the token loop — the best conversion this lane has measured in
prefill — is 6% of a decode unit. The null was not a fact about the decode shape. It was the wrong
quarter.

## The two changes

Both are inside [`kernels/gdn_state_codec.hpp`](../kernels/gdn_state_codec.hpp), which is the only
thing that has ever seen this order.

**The row order.** A packed head is 128 rows x 32 lane-quads. The old order stored row `j` as one
contiguous 128-byte run, and a wave owns rows `part*RPU + wave + 8r`, so it issued `R` loads of 128
bytes each 1024 bytes apart. The accepted order stores row `j = 32p + w + 8r` at
`(group p*8 + w, slot r)`:

    group(j) = (j >> 5) * 8 + (j & 7)          slot(j) = (j >> 3) & 3
    quad(j, lane) = group(j) * 128 + lane * 4 + slot(j)

One lane's four rows become four adjacent quads — 16 contiguous bytes — so the wave's four state
loads merge into one `global_load_b128` and its 512-byte footprint is one contiguous run. The scale
array is permuted the same way, so its four wave-uniform loads merge too.

**The scale store's predicate.** `gdn_state_store_row` ended with `if (lane == 0) region[...] =
scale`. The scale is wave-uniform, so that is a divergent store, a divergent store is a branch, and
a branch ends a basic block: the caller's `for r` store loop was `R` blocks the scheduler cannot
mix, each covering its own five-deep `warp_max` butterfly with nothing. Every lane writing the same
value to the same address is one instruction with full exec and no branch. This is the same defect
[the token loop](gdn-token-reduce.md) had, reached from the store side.

### Why it cannot move an output bit

`(lane, s) -> column c = lane + 32s` is byte-for-byte the deployed map. Only the **address** of the
`(row, lane, s)` element moves, so every `warp_sum` runs the same butterfly over the same operands in
the same order, and the two-class bit pattern that butterfly leaves in a wave — the quad-permute pair
agrees within a quad, the row-rotate pair sums quad families `j` and `j+2` together, `out_o` reads
lane 0 — is untouched. The int8 values, the `warp_max`, the reciprocal and the rounding are the same
instructions on the same inputs.

Measured rather than argued: the **residual FNV-64 is identical in all three arms of every round** at
every shape below, including `13796160890358595764` at 32 prefill rows and
`17373016486868993645` at 128.

## What it is worth, and where

Mode 20, `--gdn-state 5,6,3` interleaved in one process so the arms share a build, a warmed device
and a clock, accepted arm **last** in each round. `gdn-resident-core` normalised by the five phases
the change cannot reach (`ffn`, both sequence projections, `sequence-core`, `head-projection`):

| workload | `i8c` | `i8s` | `i8` | accepted, normalised |
|---|---:|---:|---:|---:|
| **32-stream generation step**, 1 token/unit | 18.740 ms | 18.827 | **17.091** | **−9.41%** |
| 32-row prefill pass, 32 tokens/unit | 5.374 | 4.827 | **4.749** | **−7.11%** |
| 128-row prefill pass, 128 tokens/unit | 14.188 | 14.464 | 14.301 | −0.74%, a null |
| device span, 32-stream step | 85.581 | 85.688 | **84.351** | −1.44% |
| step wall, 32-stream step | 86.804 | 86.993 | **85.745** | −1.22% |

Three rounds, ranges that do not overlap on the generation step (18.656–18.982 against
17.016–17.105). Raw in
[`batch-comparison/gdn-state-coord/`](../../../data/bonsai2/batch-comparison/gdn-state-coord/README.md).

**The win scales with how few tokens a unit walks**, which is the codec-per-unit model paying out: one
token 9.4%, 32 tokens 7.1%, 128 tokens nothing. It is a generation-shape lever that a prefill panel
would have reported as noise.

### The result that names the bottleneck

**`i8s` is an exact null at 32 streams** (+0.5%, ranges overlapping) and a **−4.9%** win at 32
prefill rows. It removes 132 instructions per unit — 26% of the codec, 19% of the whole unit — and
three basic blocks. So at the generation shape:

- **not bytes**: 2.415 GB in 18.74 ms is 129 GB/s, 53% of the 242 GB/s roof;
- **not issue slots**: −19% of the unit's slots bought nothing;
- **not occupancy**: 16 waves per SIMD32 at every instantiation;
- **requests**: eight loads to three per wave, and `s_waitcnt` eight to four, bought 9.4%.

A phase can sit at half the byte roof and half its issue budget and still be limited by how many
memory instructions it takes to ask for the bytes. That is the transferable half of this result, and
it is the same shape of answer
[the deployed matvec's TT=8 knee](deployed-matvec-groups.md) is waiting for.

## Installed and accepted

Installed on canonical `cd1545b` into `.`: `bonsai-halo`
`55087de3256520ed8381069c`, `tools/batch_profile` `0c412ad15d6e4fbad857f0b1`,
`tools/batch_compare`. The code diff between that build and the tip it was installed under is empty
- the two commits after it are documentation - and it carries
[the four-bit sequence default](seq-a4-default.md) that landed in the same hour, so the acceptance
panel runs at `seq_quant a4` where the panels above ran at `a8`.

Same shape on the installed executable from the canonical tree, `--gdn-state 5,3`, three rounds:
`gdn-resident-core` **18.974 -> 17.198 ms**, ranges 18.921-18.996 against 17.186-17.433,
**-9.54% normalised** by the five untouched phases against -9.41% in the checkout panel. Device span
84.539 -> 82.822 (-2.03%), step wall 85.714 -> 84.285 (-1.67%). Residual FNV-64
`3025822917109219071` equal in both arms - a different hash from the panels above because the
sequence operand is four-bit on this build, which is the point of comparing arms rather than
absolutes. Raw in
[`batch-comparison/gdn-state-coord/install-decode.json`](../../../data/bonsai2/batch-comparison/gdn-state-coord/README.md).

## What this does not do

- **No serving default changes.** `HALO_GDN_STATE` still defaults to `f32`, the fp32 path is
  untouched by construction, and single-stream `--bench` never enters this kernel's packed arm.
  What changed is that every consumer that *already* asks for `i8` — the route
  [the storage coordinate](gdn-state-pack.md) measured at +21.9% aggregate generation, and the
  [128-stream panel](generation-128.md) — gets the grouped coordinate with identical bits.
- **`i8` means the grouped coordinate from 2026-09-21.** Panels published before that measured
  `i8c`. `--gdn-state 3,5,6` and `HALO_GDN_STATE=i8|i8c|i8s` select the arms; the two controls are
  kept so this can be re-measured rather than re-argued.
- **`i16` and `f16` keep the row-major order** and gain only the full-exec scale store, which is
  bit-identical for them as well. They are not deployed and their phase was not re-measured.
- **The merge is SPLIT-dependent and that is not an accident.** It needs a wave's row set to be
  `{w, w+8, w+16, w+24}` inside a 32-row block, so that `group` is `r`-independent and `slot` folds
  to a literal. SPLIT 4 gets one `b128`, SPLIT 2 gets two, and SPLIT 8 and 16 get none — a wave there
  owns two rows or one, and the compiler cannot see that its slots are adjacent. One more reason the
  [SPLIT ladder above 4](resident-gdn.md) loses.

## The next lever, priced and unbuilt

**The store side is still eight requests where two would do.** The four state stores did not merge
and neither did the four scale stores, and the reason is in the assembly rather than in the
coordinate: the store sequence is `[state r][scale r][state r+1][scale r+1]...`, and the scale write
is a `float` store through the same `region` base as the `char4` state store, so alias analysis
cannot move store `r+1` across it. Per wave the load side is now 3 requests and the store side is 8.

Merging them needs a wide entry point — `gdn_state_store_rows<R>` computing all `R` scales and codes,
then storing the `R` state quads, then the `R` scales — which is a two-line change inside
`resident_state`'s and `gdn_defer_flush_k`'s store loops in `kernels/halo_rows.hip`. Those are held
by the row-ownership work; the codec side is written for it. Expect **less** than the load side
bought: a store has no `s_waitcnt` on the wave's critical path, so it costs request slots rather than
exposed latency.

**And the arithmetic the coordinate does not touch.** The store path re-encodes: `warp_max` over the
row, `qmax / a` as a full IEEE division (the value is wave-uniform and computed redundantly in VALU,
which is free in slots but not in instructions), 16 round-to-nearest conversions, 32 clamps, and a
re-decode so the registers match a reload. That is about 184 of the 368 remaining slots and none of
it can change without changing an output bit. If the store request count comes down and the phase is
still above its byte floor, that arithmetic is what is left.

## Reproducing it

```sh
tools/gdn_state_isa.py                                    # the census, 8 s, no lock
tools/run-batch-compare --pin-clock --profile-tool --out OUT.json \
  --modes 20 --decode-streams 32 --decode-context 512 --gdn-state 5,6,3 --traces 1 --rounds 3
tools/run-batch-compare --pin-clock --profile-tool --out OUT.json \
  --modes 20 --rows 32,128 --heads 1 --traces 1 --gdn-state 5,6,3 --rounds 2 --context 512
```

Mode 20 prepares no A4 weight image, so both panels start in seconds and run the same wide schedule
as mode 19. Put the accepted arm last in the case list: within-round warming on this box moves the
untouched phases by up to 5% in the first cell, which is what the normalisation column is for.

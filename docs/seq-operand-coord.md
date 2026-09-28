# A wave's fragment fetch is priced in cache lines, not in requests

The two sequence projections issue **32 of their 39 memory requests per 128-K block** to collect one
wave's activation fragments. Halving that to 16 wider requests, at the same bytes, the same image and
bit-identical output, is worth **nothing** (−0.35% normalised, five rounds). Dealing the same 16
requests across the lanes a different way — same instruction mix, same bytes, same everything the
census counts — costs **+16.25%**.

What separates the two is one number, and it is the only number this experiment found that moves
the phase: **how many distinct 64-byte lines one instruction's sixteen lanes touch.** The deployed
map is already at the floor of one lookup per line, so nothing here ships.

Raw samples, censuses and the register table:
[`batch-comparison/seq-operand-coord/`](../../../data/bonsai2/batch-comparison/seq-operand-coord/README.md).
`HALO_SEQ_TMAP` stays 0. The default build's 405 `project` instantiations are
**instruction-for-instruction identical to their predecessors**; only the mangled name gains the
defaulted parameter.

## The labelling that was free to change

`project`'s block loop hands a lane `TT` activation fragments per K16 slice and the kernel picks
which token each one is:

```c++
x[t] = *(const Frag *) (bbase + bo + t * 16 * FB);   // token = first + t*16 + col
...
int token = first + t * 16 + col;                    // and the same map on the way out
```

`v_wmma_i32_16x16x16` reduces over K alone and never across its sixteen B columns, so **which token
sits in column `col` of tile `t` is a labelling, not a constraint**. Any bijection from (tile, lane)
onto a group's `TT * 16` tokens computes the same outputs from the same products in the same order —
the same int32 block sums over the same eight slices, the same `fmaf(float(acc), wr[r] * a, y)` chain
over the same blocks — with only the lane that holds a token moving. Nothing about the image, the
weights, the scales or any writer has to change, which is why this was worth asking: `capture`, the
wide prep, the attention combine and the head all keep writing `[K16 slice][token]`.

At four-bit activations, which both projections take by default, a fragment is **eight bytes**, so
the deployed map spreads a lane's `TT` of them `16 * FB` = 128 bytes apart and the wave spends one
`global_load_b64` on each. Put a lane's tiles side by side and two of them arrive in one
`global_load_b128`.

| map | token of (tile `t`, lane `col`) | requests/block at `TT=4` |
|---|---|---:|
| 0, tile-major (deployed) | `first + t*16 + col` | 32 |
| 1, lane-major | `first + col*TT + t` | 16 |
| 2, pair-major | `first + (t>>1)*32 + col*2 + (t&1)` | 16 |

`make kernels/seq_tmap_check && kernels/seq_tmap_check` proves all three are bijections onto the
same tokens in under a second with no GPU, and prints the line geometry below. At `TT = 1` all three
are `token = first + col`, so that width is untouched code.

## The census, and why it predicted the wrong winner

`tools/isa_loop_count.py` on the instantiation a 128-row pass launches,
`project<TT=4, PLANAR, WV=2, SHARE, RT=1, SB=1, LA=LANE, I4=nibble>`, plus the register file out of
the same listing's metadata. No GPU.

| block loop, 128-row shape | map 0 | map 1 | map 2 |
|---|---:|---:|---:|
| instructions | 371 | 358 | 358 |
| work slots | 325 | 323 | 323 |
| `global_load` | **39** | **23** | **23** |
| `s_waitcnt` | 41 | 20 | 27 |
| `salu` | 25 | 41 | 20 |
| `v_wmma_i32_16x16x16_iu4` | 32 | 32 | 32 |
| operand expansion (`bitops`) | 96 | 96 | 96 |
| VGPR / waves per SIMD32 | 180 / **8** | 121 / **10** | 186 / **8** |
| spill | 0 | 0 | 0 |

Read on its own, that table says map 1 should win twice over: sixteen fewer requests, twenty-one
fewer waits, two fewer work slots and **two extra wave slots**, on a kernel whose last two owners
both released it saying "a wave slot in `project` is worth about 3% of the phase, the largest
per-unit rate this lane has measured on any kernel". It is the worst of the three.

## What the wave actually asks the cache for

Sixteen lanes issue one address each and the coalescer turns them into line lookups. The three maps
fetch the identical 512 bytes per (slice, token group) and deal them differently
(`kernels/seq_tmap_check`, `TT = 4`, `FB = 8`):

| map | one fragment instruction | lines it touches | of each line it uses | lookups per slice |
|---|---|---:|---:|---:|
| 0 | 16 lanes x 8 B over 128 contiguous bytes | 2 | 100% | 4 x 2 = **8** |
| 1 | 16 lanes x 16 B at stride 32 | **8** | **50%** | 2 x 8 = **16** |
| 2 | 16 lanes x 16 B over 256 contiguous bytes | 4 | 100% | 2 x 4 = **8** |

**512 bytes is eight lines, so eight lookups is the floor and the deployed map is on it.** Map 2
reaches the same floor with half the instructions. Map 1 straddles: each of its two instructions
walks eight lines and uses half of each, and the pair of them pays twice.

## Measured

Mode 19, 128-row passes, four-bit on both projections, arms shuffled and interleaved inside one
process on one build and one clock, head off. `ffn`, `gdn-resident-core`, `sequence-core`,
`sequence-input-prep` and `embed` are phases no map can reach and are the control column.

**Five rounds an arm** (`m19-128-three.json`):

| map | `sequence-input-projection` | `sequence-output-projection` | control | normalised |
|---|---:|---:|---:|---:|
| 0, deployed | 25.730 ms | 10.427 ms | 102.083 | — |
| 1, lane-major | 29.567 | 11.734 | 100.304 | **+16.25%** |
| 2, pair-major | 25.953 | 10.506 | 103.294 | **−0.35%** |

**Maps 1 and 2 issue the same instructions.** Same 23 `global_load`, same 16 `global_load_b128` for
the fragments, same 32 `v_wmma`, same 96 expansion operations, same bytes out of the same image. The
only difference between them is which lane gets which token, and it is worth **16.6% of the phase**.

Residual FNV-64 `3240780174642567821` — the value
[`seq-pair-operand.md`](seq-pair-operand.md) publishes for this shape — in every sample of every arm
of every panel below, so all of this is bit-identical on the device as well as by construction.

### The schedule is already the right one, and the token-tile width is a null

The same instrument, walking `sequence_shape`'s own ownership rule against the two it does not pick
(four rounds an arm, `sched-tmap.json`; `sum/ctl` is the two projections over the control column):

| ownership | map | `seq-in` | `seq-out` | `sum/ctl` | against deployed |
|---|---|---:|---:|---:|---:|
| 1, shared row tile (**deployed**) | 0 | 24.441 | 9.955 | 0.3549 | — |
| 1, shared row tile | 2 | 24.399 | 9.942 | 0.3538 | −0.31% |
| 0, per-wave row tile | 0 | 23.809 | 10.592 | 0.3559 | +0.29% |
| 0, per-wave row tile | 2 | 25.323 | 11.328 | 0.3803 | +7.16% |
| 2, wide | 0 | 28.425 | 11.626 | 0.4162 | +17.3% |
| 2, wide | 2 | 28.490 | 11.637 | 0.4163 | +17.3% |

Two things fall out. **The deployed ownership is the best of the three** at this shape, by 0.3% over
per-wave tiles and by 17% over the wide one, so there is no default to move. And map 2 is a null
where the workgroup shares a row tile and **costs 7.2% where each wave owns its own** — four waves
that read byte-identical fragments ([`seq-input-bstage.md`](seq-input-bstage.md)) apparently want
them requested the way the other three waves are requesting them.

**Eight token tiles, the rung `kernels/sequence_batch.hip` records as never measured, is also a
null.** `HALO_SEQUENCE_TT=8`, four rounds (`tt8-tile.json`, `tt8.json`), normalised by the same
control column: per-wave tile ownership reads `sum/ctl` **0.3569** against `TT = 4`'s 0.3559, and the
shared-row-tile ownership reads 0.6724 — because at 128 rows `TT = 8` leaves one token group and the
shared form gives its second wave nothing to do. The width trades what it should: `TT = 8`
amortises the 96-slot expansion and the weight block over twice the tokens, about −10% of the block's
issue per token, and pays a wave slot for it (213 VGPR at seven waves against 180 at eight). The two
cancel.

## What this closes, and the law it hands over

Five engineers have now attacked this kernel's activation path and every arm has landed between
null and −18%:

| arm | what it changed | result |
|---|---|---:|
| cache policy (`ea7a09b2`) | where the fragments live | null |
| row-tile reuse `RT` ([`sequence-fragment-reuse.md`](sequence-fragment-reuse.md)) | halves fragment bytes, doubles the expansion | +7.3% |
| LDS staging ([`seq-input-bstage.md`](seq-input-bstage.md)) | 4x fewer requests, −7 wave slots | +21.6% |
| pair alphabet ([`seq-pair-operand.md`](seq-pair-operand.md)) | −57 work slots of expansion | +2.3% |
| **token map 2, here** | **−16 requests, −14 waits, same registers** | **−0.35%** |
| **token map 1, here** | **the same, dealt across 8 lines instead of 4** | **+16.25%** |

Stated as the law the lane can spend elsewhere: **in this block loop a fragment fetch costs one L0
tag lookup per distinct line its wave touches, and nothing else.** Not its request count — sixteen
requests and thirty-two requests for the same lines measure the same. Not its issue slots — two
fewer measured nothing, fifty-seven fewer measured +2.3%. Not its wave slots — the arm that bought
two of them is the arm that lost. A map is worth exactly the lines it straddles.

Three consequences worth carrying:

1. **A lane-contiguous layout is not automatically a cheaper one.** Making a lane's own bytes
   adjacent makes the *wave's* instruction stride, and a strided 16-byte-per-lane load touches four
   times the lines of the contiguous 8-byte one it replaced. `kernels/seq_tmap_check` prints that
   number for a candidate map in a second, before anyone builds it.
2. **[`ffn-b-operand.md`](ffn-b-operand.md)'s law does not generalise.** "The B fragments' cost is
   their request count, not their locality" is true of `k_proj_opt`, where moving the fetch to LDS
   bought −4%; it is false here, where removing half the requests at constant lines bought nothing.
   The difference the two kernels do not share is what else is outstanding: the A4 block carries 26
   requests and a wait every twenty slots, and this one reaches its floor on lines first.
3. **`head_batch` and `halo_draft` read the same `[K16 slice][token]` image the same way.** The
   relabelling is as free for them as it was here, and this document says what to expect from it:
   nothing, unless their current deal is off the line floor. Check that with the check program
   before spending a panel.

## Reproducing

```sh
make DEFS='-DHALO_SEQ_TMAP_CONTROL=1' bonsai-halo tools/batch_profile
tools/run-batch-compare --pin-clock --profile-tool --modes 19 --rows 128 --heads 0 \
    --traces 1 --rounds 5 --seq-tmap 0,1,2 --out RUN.json
tools/arm_panel.py RUN.json --key seq_tmap --phase sequence-input-projection
make kernels/seq_tmap_check && kernels/seq_tmap_check          # the map model, no GPU
```

`HALO_SEQ_TMAP=0|1|2` pins one map for a whole process. A default build carries only map 0 and
refuses the others; `sequence_batch_tmap_control()` says which build is in hand.

**Every panel here ran on a box that was not quiet** — the wrapper's own sampler reported 4.2 to 7.9
cores of peer host work and a delivered clock of 2100 to 2470 MHz against a nominal 2900, package
power limited for 45 to 74% of each window. That is exactly the drift
[`power-budget.md`](power-budget.md) measured, and it is why every number above is an in-process
arm ratio against a control column rather than a millisecond compared across panels.

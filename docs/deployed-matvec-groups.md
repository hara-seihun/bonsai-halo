# A second column group for the deployed matvec

[The sixteen-row FFN slice](ffn-slice-width.md) ends by naming what was left in the same body:

> Sixteen is one column group. A second group would run against weights already peeled into
> registers, so it would amortise the ~112 issue slots of five-trit expansion per 128-K block and the
> weight load itself over 32 rows [...] and it is what stands between a 218 ms `ffn` phase and the
> ~120 ms its weight traffic alone would take.

That group is built. `mvw_rows` now takes `GROUPS` sixteen-column groups per unit, `k_ffn_slice`
takes 32 rows, and a 128-row wide-deployed pass runs four FFN slices where it ran eight. The `ffn`
phase of that pass falls **216.3 to 173.2 ms** with the untouched phases moving 3.5% the other way,
full-model prompt processing on the wide-deployed route goes **407.2 to 458.5 tok/s**, and the
engine's own drafted prompt ingestion goes **377.4 to 421.0 tok/s** against an in-process deployed
control that does not move. It is bit-identical: 381,419,520 full-vocabulary logit values across
four pass widths with zero differing bits.

Source `d3a3606` and `4e1fa98`. Held: `kernels/phases.hpp` (`mvw_rows`, `ph_matvec_w`, the `COLS` and
`GROUPS` arguments in `ph_matvec_auto`), the `FMAX` line of `kernels/halo_kernels.h`, the
`launch_ffn_slice` dispatch in `kernels/halo_rows.hip`.

## What a block costs, and what of it is per row

A unit of `ph_matvec_w` owns one 32-row weight tile and walks its `NBLK` 128-K blocks. Per block a
wave loads 896 bytes, peels 128 trits a lane out of radix-3 into `tr[32]`, half-swaps each of the
eight A fragments into its second 16-row matrix, and then issues sixteen `wmma_i32_16x16x16_iu8`
against the activation fragment. Only the last of those four depends on how many activation rows the
unit serves. The compiled block loop says so directly:

| per block | one group | two groups |
|---|---:|---:|
| instructions | 432 | 652 |
| of which `v_wmma_i32_16x16x16_iu8` | 16 | 32 |
| activation rows served | 16 | 32 |
| **non-matrix instructions per 16 rows** | **416** | **310** |
| issue cycles per row, 32 cycles a WMMA | 58.0 | 51.4 |

So the second group buys **25% of the block's non-matrix instructions** and half of the weight
stream, and costs sixteen more matrix instructions, eight more activation fragment loads and two
more accumulator pairs.

`tools/isa_loop_count.py /tmp/x.s k_ffn_sliceILi32ELi0EE` reproduces the right column from a
`hipcc --cuda-device-only -S` listing; no GPU, no lock.

## The shape that fits, and the two that do not

Groups run **outer**, K slices inner, with `__builtin_amdgcn_sched_barrier(0)` between them.

- **K outer, groups inner** is the obvious shape and it is the wrong one. It saves the 32 half swaps
  per block by sharing one A fragment across both groups, and holds both groups' accumulator pairs
  and both activation fragments across the whole K loop: 256 VGPR, a spill, and five waves per
  SIMD32. Thirty-two instructions out of a 1644-cycle block is not worth a wave slot.
- **Groups outer without the fence** is the same trap by another route: the scheduler hoists every
  group's eight `global_load_b128` above the first matrix instruction, which is 32 registers a
  group, and lands at 256 VGPR with a spill again.
- **Groups outer, fenced** holds one group's operands at a time: 242 VGPR, no spill.

242 is four registers above this device's 240-register, six-wave cliff, and the allocator had no
reason to look for them. `__attribute__((amdgpu_waves_per_eu(6)))` on the instantiations above
sixteen rows finds them: **238 VGPR, no spill, six waves per SIMD32**, and the one-group
instantiation the control arm runs is untouched at 217. That is worth about four points of the
result; the same panel shape measured `ffn` at -16% before the attribute and -20% after.

**Four groups lose.** `k_ffn_slice<64>` is 256 VGPR with a 10-byte spill and stays at five waves, and
on a three-arm panel it measured no better than two groups normalised and 4% worse in absolute
terms. The peel it saves is already mostly saved; what it adds is 32 activation fragment loads per
block, each of which touches sixteen distinct cache lines.

## Measured

All panels: mode 20 (`wide-deployed`, the exact numerical map on the wide schedule), one build, arms
selected by `HALO_FFN_SLICE` inside one lock hold, untouched phases summed as the in-panel control.

**The phase, pinned at 2900 MHz, two rounds an arm**
(`mv-column-groups/pin6-{16,32}.json`):

| arm | `ffn` | untouched phases | `ffn` / control | device span |
|---|---:|---:|---:|---:|
| 16-row slices, one group | 213.431, 219.215 | 96.973, 99.318 | 2.2009, 2.2072 | 312.714, 320.832 |
| **32-row slices, two groups** | **173.262, 173.062** | 101.505, 101.233 | **1.7069, 1.7095** | **276.368, 275.898** |

-19.9% on the phase, -22.5% after the control, -12.8% on the pass. The control moves *against* the
change, so the raw figure is the conservative one. An unpinned pair of the same shape on the
previous build read 221.907 to 187.021 ms with its control flat to 0.2%.

**Three widths, one process each, unpinned** (`mv-column-groups/w-{16,32,64}.json`): `ffn` 249.2,
196.3, 204.8 ms at 16, 32 and 64 rows a slice, with controls 102.6, 96.4, 101.4. Normalised: 2.429,
2.036, 2.020. Launches per pass: 512, 256, 128.

**Full model**, 384-token document in 128-row passes, three rounds, mode 0 interleaved as a control
that never calls `k_ffn_slice` (`mv-column-groups/doc-{16,32}`):

| arm | mode 20 prompt tok/s | mode 0 control |
|---|---:|---:|
| one group | 405.6, 407.2, 408.6 | 156.1, 156.5, 157.3 |
| **two groups** | **458.2, 458.5, 467.9** | 157.3, 157.0, 157.4 |

**+12.6%, +12.0% after the control.**

**The product path**, the engine binary on a 1610-token raw prompt in its own 256-row passes, DFlash2
loaded, both routes walked inside one process by `--prefill-sweep off,wide-deployed`
(`mv-column-groups/dsweep-{16,32}.txt`):

| arm | deployed route | `wide-deployed` | generation | drafted / accepted | token digest |
|---|---:|---:|---:|---:|---|
| one group | 150.1 tok/s | 377.4 | 20.29, 20.12 | 63 / 3 | 3349235032644853533 |
| **two groups** | 151.6 tok/s | **421.0** | 20.25, 20.36 | 63 / 3 | 3349235032644853533 |

+11.6% with the in-process deployed control at +1.0%, and the wide route is now **2.78x** the
deployed one on this machine's own serving configuration. Generation, the drafted token stream and
the acceptance counts are unchanged, which is what a change to prompt ingestion should do.

## Bit-identity

Every output element sees the same weights in the same K order, the same per-block int32
accumulation and the same `fmaf(C - xsum, wscale * xscale, y)` fold. Which group carried a row moves
only which lane holds its column.

`tools/batch_compare --prefill-identity` hashes every logit byte of the whole document at a given
pass width. Mode 20, four pass widths, both arms
(`mv-column-groups/ident-{16,32}/g{16,32}/run.json`):

| rows per pass | passes | logit values | one group | two groups |
|---:|---:|---:|---|---|
| 32 | 12 | 95,354,880 | 15043080294956075508 | 15043080294956075508 |
| 40 | 10 | 95,354,880 | 86164060768772125 | 86164060768772125 |
| 88 | 5 | 95,354,880 | 15043080294956075508 | 15043080294956075508 |
| 128 | 3 | 95,354,880 | 15043080294956075508 | 15043080294956075508 |

**381,419,520 full-vocabulary logit values an arm, zero differing bits, identical argmax streams, no
non-finite value.** 40 and 88 rows matter beyond the count: they leave a slice with 8 and 24 rows, so
the second group is partial and the epilogue's `g * 16 + col < nrows` guard is what keeps its
columns out of the output. Residual FNV-1a is `16921095978579329146` in every traced sample of every
panel above, which is the hash `docs/ffn-slice-width.md` published.

## What this says about the stage

The issue model says the second group should be worth 11.5% of the phase. It measured 20-22%. The
difference is the half of the change the instruction count cannot see: the pass reads the FFN weight
image four times instead of eight, 15.0 GB instead of 29.9 per 128 rows.

That is the useful negative inside the positive result. **This stage is not the issue-bound machine
the 128-row A4 FFN is**, where `docs/ffn-order.md` and `docs/resident-gdn.md` found instruction
counts converting at close to 1:1. At 86 GB/s after the change it is not bandwidth-bound either;
what it is short of is *memory in flight*, the same diagnosis `docs/decode-dispatch.md` reached from
the other side of the tree. Price a change to it on traffic and latency before pricing it on slots.

## Next in this body

- ~~**The activation fragment is on the wrong axis and it is now the biggest uncounted cost.**~~
  **Built, and it loses.** `xq` is row-major, so the sixteen columns of one `global_load_b128` are
  sixteen different token rows, 5120 or 17408 bytes apart: eight loads a group, sixteen distinct
  cache lines each, 128 line requests per block where 16 would do. The `[block][K slice][row]`
  operand does exactly that - 320 line requests a block become 36, bit-identically over 381,419,520
  logit values - and the `ffn` phase gets **3.6 to 4.2% slower** in two panels.
  [The FFN slice's activation coordinate](ffn-slice-operand.md) has the census: the per-lane address
  arithmetic that the scattered operand needed was separating the radix-3 peel's dependent pairs,
  and removing it closes that chain up for 62 more `s_delay_alu` in the gate/up block loop while
  every load, store and matrix instruction stays the same. **Price a change to this loop on issue
  slots and its dependence chain, not on cache lines.**
- **Sixteen more rows are free in the persistent kernel too, and nobody can spend them.** The same
  body serves `ph_matvec_w` inside `k_forward_rows`, which is bounded by `RMAX = 8` because the pass
  replays, rolls back and attends. `docs/serving-decode.md` prices what a wider drafted verify would
  be worth; the row-scaled 20.6% of a pass is what stands in the way, not this kernel.
- The `s_delay_alu` count in the block loop goes 13 to 112 between the two shapes. That is the
  scheduler paying for a longer dependence chain, and it is 17% of the two-group block. Nobody has
  looked at whether the fence between groups can be narrower than `sched_barrier(0)`.

## Installed

Canonical `c48a369`, installed build `bonsai-halo`
`7f0e0bd6300a4373e3bcd884bcf6c9ee24d3b336a43cd667812050975073e6b0`, `tools/batch_compare`
`4f5d14fe415c68310d8f2415c0472e2056548cca6c734e422f306e531f12a111`.

Throughput on the installed binary, 384-token document in 128-row passes, two rounds, both arms in
one lock hold with mode 0 interleaved (`mv-column-groups/inst-{16,32}`):

| arm | mode 20 prompt tok/s | mode 0 control |
|---|---:|---:|
| one group | 412.3, 451.5 | 162.6, 165.0 |
| **two groups** | **507.6, 501.8** | 164.7, 165.1 |

Medians 431.9 to 504.7, **+16.9%**, control +0.7%. The one-group arm's own two rounds differ by 9.5%
on a box carrying three other engineers, so read the direction and the identity from this panel and
the size from the pinned one above.

Identity on the installed binary, mode 20 at 40, 88 and 128 rows per pass
(`mv-column-groups/instid-{16,32}`): `86164060768772125`, `15043080294956075508`,
`15043080294956075508` in **both** arms, 286,064,640 logit values an arm, no non-finite value. Those
are the same three hashes the pre-install panel produced.

The two paths this change must not move, on the installed binary and under the same wrapper:

- **The drafted product path**, 1610-token raw prompt with DFlash2 loaded and `--prefill-ffn
  wide-deployed`: 414.1 tok/s of prompt ingestion, greedy token digest `3349235032644853533`, 63
  drafted and 3 accepted, generation 20.74 tok/s. The digest and both counts are the ones the
  pre-install panel produced in both of its arms. This is also the check that covers the Q8 drafter,
  which reaches `mvw_rows` through `ph_matvec_auto` at eight rows and one group.
- **Single-stream `--bench`**: 31.44 tok/s, inside the band this box gives while several engineers
  are measuring, and untouched by construction since a one-row step takes the dot4 body.

The resident service came back on the new binary and answers `/health`.
## The grid is the occupancy number, and four groups still lose when it is level

A second engineer reached this kernel from the same sentence in
[the sixteen-row slice](ffn-slice-width.md) and built the same two-group body independently, within
twenty minutes of the version above. That duplicate is not published; what came out of it that this
document did not already have is below, and the first item explains the confound in the four-group
negative.

**Waves per SIMD32 is not what `coop_grid` spends.** A workgroup here is 256 threads, eight waves
over a WGP's four SIMDs, so the cooperative grid is `workgroups per WGP x 20` and occupancy is
quantised in whole workgroups: **six and seven waves per SIMD32 both give three workgroups and a
grid of 60; five gives two and a grid of 40.** That is why `amdgpu_waves_per_eu(6)` was worth four
points here - it crossed the 5-to-6 boundary - and why freeing registers past six waves buys
nothing. The independent build measured the cliff directly: the same two-group source at 241 VGPR
and 40 workgroups read **-5.7% and -6.4%** on the `ffn` phase, and at 215 VGPR and 60 workgroups
**-17.4% and -17.9%**, with `sequence-input-projection` as the in-process control. A phase time
alone reads that as a worse schedule. `HALO_FFN_GRID=1` now prints the grid once per slice width.

**Dropping the block lookahead pays for four groups' registers, and four groups still lose.** The
`nqa`/`nqb`/`ntail` copy of the next weight block costs 26 registers. Gating it off above one group
takes `k_ffn_slice<32>` to 214 VGPR and seven waves and `k_ffn_slice<64>` to **240 VGPR, six waves
and no spill**, where the four-group shape above was 256 with a spill and five waves - so the
four-group negative can be re-run without its grid handicap. One process, one build, three widths
against a common clock through `--ffn-slice 16,32,64`, `ffn` over `sequence-input-projection`, two
rounds:

| rows a slice | groups | launches per 128-row pass | `ffn` / control |
|---:|---:|---:|---:|
| 16 | 1 | 512 | 5.0025, 4.9598 |
| **32** | **2** | **256** | **4.1186, 4.1189** |
| 64 | 4 | 128 | 4.2649, 4.2658 |

Four groups is **3.6% worse than two** at equal grid, and halving the launch count again does not
help it. The negative in the section above stands on its own merits: what the fourth group adds is
32 activation fragment loads a block across sixteen cache lines each, and that costs more than the
weight stream it saves.

**The lookahead earns its 26 registers at two groups.** Since 214 and 240 VGPR are the same grid,
the only question left is whether the prefetch hides the weight load, and it does. Two builds
differing only in the gate, with the one-group body - identical in both - as the cross-build anchor:

| | `ffn` at 32 rows / `ffn` at 16 rows | normalised |
|---|---:|---:|
| lookahead on (published) | 0.8014 | 0.8182 |
| lookahead off above one group | 0.8243 | 0.8269 |

1.1 to 2.9% in favour of keeping it. The gate is not in the tree; this table is why. A future
engineer who wants the four-group shape at six waves has the recipe and the reason not to bother.

## One free instruction at every width

The drain computed `(float) (C1[r] - xsc)` for every output element, where `xsc` is the activation
block sum the unsigned trit codes owe back. **Seeding both int32 accumulators with `-xsc` costs the
same eight `v_mov` the zero seed cost** and removes sixteen integer subtracts per group per block -
32 of a two-group block's 652 instructions. Integer addition is associative and wraps identically,
so the drained word is the same. [The wide head](wide-head.md) found it first; this is the same
trick in the deployed matvec, and because `mvw_rows` is also the persistent kernel's matrix body it
reaches the drafted verify pass, taking `k_forward_rows<8>` from 217 VGPR to 216 and from six
resident waves per SIMD32 to seven.

It is about 2% of the block, which is inside what a cross-build pair resolves on this box, so the
census and the register report are the evidence and the phase panel only rules out a regression.
Bit-identity is not: `--prefill-identity` at both pass widths hashes **95,354,880 full-vocabulary
logits to `15043080294956075508`**, zero nonfinite, the same value the build without it produces.

`--ffn-slice N` on `tools/batch_profile` walks the slice width as a case axis inside one process,
which is how the three-width table above was taken; `batch_set_ffn_slice_rows` is the setter and
`HALO_FFN_SLICE` still pins it for a process that cannot call one. Raw samples are in
[`batch-comparison/ffn-slice-tiles/`](../../../data/bonsai2/batch-comparison/ffn-slice-tiles/).

### Installed acceptance of the accumulator seed

Canonical `40d39df`, installed `bonsai-halo`
`7262d905c5070c4ede48bcdfe5f28503bc2812aca7a5d0210d61999a98aba17b`, `tools/batch_compare`
`254d0f23f098be95c8fc479a94c071547962624174bdf5bae95239fe97762c87`.

- **Bits.** `--prefill-identity`, 384-token document at 128- and 256-row pass widths: 95,354,880
  full-vocabulary logits, FNV `15043080294956075508`, zero nonfinite - the value this kernel
  produced before the seed and before the column groups.
- **Prompt.** 462.5 tok/s at 128 rows a pass and 467.5 at 256, mode 20.
- **Served path.** 1289-token drafted prompt, `--prefill-sweep off,wide-deployed`: **469.9 tok/s**
  on route 20 against 159.5 on the eight-row deployed control in the same process, greedy digest
  `7796914616836249716` in both. Drafted verify is 51.1 and 51.3 ms a step against 52.5-53.6 ms in
  the panels before the seed; that is a cross-panel comparison of a 2% change, so read it as no
  regression on the shape the server actually generates.

The resident service was left inactive and `masked-runtime` with the measurement lock held by
another engineer, which is the correct state while anyone is measuring.

## What this block waits on, measured

[The request order](ffn-slice-issue-order.md) prices the block loop this document built. The three
weight loads are drained 264 slots into a 476-slot block by the accumulator seed's wait for one
`xsum` word - the in-order `vmcnt` shape [the A4 cursor](ffn-decode-schedule.md) found binding in
`k_proj_opt` - but pinning the weight stream into cache is only **8%** of the phase here and
requesting the operands first is a bit-identical **+0.55%** null. With `bench/wmma_cost`'s measured
**21.9 issue slots** for `v_wmma_i32_16x16x16_iu8`, a two-group block is 701 slots of matrix inside
1145 counted, so this kernel is 61% matrix, a tenth memory, and the rest is the peel and the fold.

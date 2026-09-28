# The FFN slice's column-group ladder ends at two, and the wall is operand cover

[Sixteen rows per slice](ffn-slice-width.md) was -47.1% of the deployed FFN phase, and
[the second column group](deployed-matvec-groups.md) took it to thirty-two for another -20% and
+51 prompt tok/s. Both rungs did the same thing: a unit loads its 896 weight bytes once, peels them
into `tr[32]` once and swaps them once, and every further sixteen rows ride along for the price of
their own matrix instructions. `mvw_rows` already carries `GROUPS` as a template parameter and
`ph_matvec_auto` already derives `MV_GROUPS = (TT + 15) / 16`, so the third and fourth rungs need
no kernel body at all — only `FMAX`, the width table and the launcher.

**They are built, they are bit-identical, and they buy nothing.** Sixty-four rows per slice halves
the weight stream and the radix-3 expansion of a 128-row prompt pass and measures **+0.18% per
row**. Forty-eight rows — the same registers, the same grid, three groups instead of four — is
**+29%**. This document is why, and the mechanism is not the one the ladder was priced against.

`-DHALO_SLICE_WIDE=1` builds the rung (`FMAX` 32 → 64). **The default build is unchanged**: `FMAX`
is 32, `slice_wi` returns exactly the index it returned before for every admissible row count, and
the deployed panel reproduces `residual_fnv64 9171727463267762618` with 256 FFN launches per
128-row pass.

## What the arms cost, per launch, in one process

Mode 20 — the route `bonsai-halo.service` ingests prompts on — 128-row passes with logits on the
tail row, `--ffn-slice` as an in-process case axis on one build and one clock, two rounds with the
cases interleaved. Per-launch times come straight out of the trace, so a 32-row launch and a 64-row
launch are compared at the layer they both do:

| slice width | groups | launches per pass | ms per launch | **ms per row per layer** |
|---|---:|---:|---|---:|
| **32 (deployed)** | 2 | 256 | 0.7133 / 0.7108 | **0.02229 / 0.02221** |
| 48 | 3 | 192 (48+48+32) | 1.3841 / 1.3743 | 0.02884 / 0.02863 (**+29%**) |
| 64 | 4 | 128 | 1.4289 / 1.4287 | 0.02233 / 0.02232 (**+0.2%**) |

Whole-phase, same samples: `ffn` **182.60 / 181.96 ms** at four launches of 32 against **182.89 /
182.87** at two launches of 64, with `sequence-input-projection + gdn-resident-core +
head-projection` as the in-panel control at 36.34 / 36.25 against 35.47 / 35.53. Every sample of
every arm returns residual FNV-64 `9171727463267762618`, which is the value canonical published
before this change existed: **the rung is a relabeling of which column group carries a row, and the
measurement says so rather than the argument.**

A 64-row launch costs 2.004x a 32-row launch. That is the null in its sharpest form — two 32-row
launches and one 64-row launch do the same arithmetic, and the 64-row one reads **half the weight
bytes** to do it.

## What it halves, and what it does not

Per (unit, block) a wave reads 896 weight bytes and, per column group, 2048 activation bytes. Over a
128-row pass at 64 layers:

| | 4 launches of 32 rows | 2 launches of 64 rows |
|---|---:|---:|
| weight stream | **15.0 GB** | **7.5 GB** |
| activation fragments | 68.4 GB | 68.4 GB |
| `v_wmma_i32_16x16x16_iu8` | 534 M | 534 M |
| radix-3 expansions | 16.7 M blocks | 8.4 M blocks |

The activation stream is invariant under grouping — a row's fragments are read once per weight tile
whatever carries them — and it is **4.6x the weight stream**. Halving the smaller of the two moved
nothing.

## The census says the rung works exactly as designed

`tools/isa_loop_count.py` on the block loop of each width, from a three-second
`-D HALO_SLICE_PROBE=N` compile:

| | 32 rows (2 groups) | 48 rows (3) | 64 rows (4) |
|---|---:|---:|---:|
| block instructions | 476 | 657 | 778 |
| `v_wmma` | 32 | 48 | 64 |
| `v_perm` (peel gather + half swaps) | **64** | **64** | **64** |
| bitops | 71 | 74 | 75 |
| `global_load` | 23 | 33 | 43 |
| `s_delay_alu` | 32 | 113 | 130 |
| `s_waitcnt` | 22 | 34 | 42 |

The peel is shared across groups exactly as the design claims — 64 permutes and ~71 bitops whatever
the group count — and everything that scales is per-group accumulator and operand work.

Converted with the **measured** instruction cost rather than a nominal one
(superseded by [`ffn-matrix-rate.md`](ffn-matrix-rate.md), which measures the same instruction at
34.3 cycles per SIMD32 and the slots beside it at zero - the ladder's conclusion does not change,
but its "7.6% cut that converts at zero" is now expected rather than surprising;
[`bench/wmma_cost`](ffn-slice-issue-order.md): `v_wmma_i32_16x16x16_iu8` is 21.9 issue slots per
SIMD32, not 32), a block is 1145 slots at two groups, 1660 at three and 2116 at four:

| | slots per row | measured, relative |
|---|---:|---:|
| 2 groups | 35.8 | 1.000 |
| 3 groups | 34.6 (-3.4%) | **1.299** |
| 4 groups | 33.1 (-7.6%) | 1.005 |

**The issue model explains none of the ordering.** Matrix instructions are 61% of the deployed
block, so amortising the peel over more rows is worth less than it looks, and what the wider bodies
do to the memory schedule is worth more.

## The mechanism: cover, not registers, bytes or slots

`tools/vmcnt_cover.py` walks the block as the loop it is and reports how many issue slots stand
between a request and the `s_waitcnt` that drains it. On the same three listings:

| | requests per block | mean cover | shortest cover |
|---|---:|---:|---:|
| 2 groups | 23 | **167.7 slots** | 30 |
| 3 groups | 33 | 33.8 | **1** |
| 4 groups | 43 | **27.9 slots** | **1** |

At two groups the block's first operand set is requested with 250-288 slots of cover and the second
with 30-47. At four groups the tail of the distribution is 1, 1, 1, 1, 4, 4, 5, 5, 6 slots **with
zero matrix instructions inside the cover**: the wave issues a fragment load and waits for it
immediately. Every group added is another operand set requested inside the loop that consumes it,
and the stall it buys is the size of the issue the shared peel saves.

That is also why three groups is a 29% loss while four is a wash: both pay the collapsed cover, and
only four amortises enough peel to pay it back. Per block-visit, converting time to machine
capacity at the clock the run held, the idle is about 480 slots at two groups, 1495 at three and
1143 at four.

**The register file is not the wall, which is worth stating because that is where this was expected
to stop.** `tools/kernel_resources.py -D HALO_SLICE_PROBE=N`, no GPU, three seconds each:

| rows | groups | VGPR | waves/SIMD32 | spill | LDS | grid |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 2 | 238 | 6 | 0 | 8228 | 60 |
| 48 | 3 | 240 | 6 | 1 | 8228 | — |
| **64** | **4** | **240** | **6** | **2** | 8228 | **60** |
| 96 | 6 | 240 | 6 | 49 | 8228 | — |
| 128 | 8 | 240 | 6 | 141 | 8228 | — |

Four groups costs two spilled dwords and no wave slot, and `HALO_FFN_GRID=1` confirms both widths
launch on 60 workgroups. Six and eight groups spill `y1[GROUPS][8]`/`y2[GROUPS][8]` badly and are
the register wall — but nothing reaches it, because the cover wall is two rungs earlier.

## The arm the cover table asks for, built and measured: it loses at both widths

`16259bba` read the same cover table from the other side and built the cheap form of the fix: only
a group's **first** fragment matters, because once its two matrix instructions are issued every
later fragment of that group is covered by the ones before it. Arm 1 of `mvw_rows` hoists one
`int4` and the two scale words per group after the first to the top of the block - six VGPRs at two
groups, eighteen at four - and buys them the whole peel plus sixteen matrix instructions of cover.
Their body, their patch; the width plumbing (`k_ffn_slice`'s `MVORD`, `slice_ord`, `ffn_slice_order`,
the `--mvw-order` case axis) and the panel are this iteration's.

**Both widths reject it.** Mode 20, 128-row passes, arms interleaved in one process, `ffn`
normalised by the phases the change cannot reach (`tools/arm_panel.py`):

| arm | `ffn` | control | normalised |
|---|---:|---:|---:|
| two groups, operands in the group loop (deployed) | 177.843 ms | 54.636 | — |
| two groups, first operand per group hoisted | 179.518 | 54.018 | **+2.09%** |
| four groups, hoisted, against four groups plain | — | — | **+8.4 / +8.3 / +7.9%**, three rounds of three |

Six rounds per arm at two groups, one residual FNV-64 `9171727463267762618` across every sample of
every arm. The static census said this before the panel did: the hoist's six registers are paid for
by the allocator **sinking group 0's operand loads from slot 16 to slot 247**, which takes the
block's shortest cover from 30 slots to 8 and its mean over 20 requests to 72. Hand-placing a
request in this body buys one cover and sells another.

`-DHALO_MVW_ORDER_ARM=1` compiles the arm; the default build carries exactly the instantiations it
carried before, and `ffn_slice_order()` reports what was asked either way.

### A measurement trap this panel walked into, which is worth more than the arm

The first reading of the hoist was **-2.14%**, from the median of 768 per-launch times per arm
pooled over three rounds. Per round the deltas were **+0.37%, -3.19%, +5.23%**: the shader clock
ramped during the hold - every arm's third round is 13% faster than its first, control phases
included - and an arm that ran later in a round collects a different clock. **The pooled median of
a per-launch instrument is not protected from clock drift by its sample count.** The normalised
ratio is, because the control phases move with it in the same sample. Only the ratio column above
should be read as a result, and the same panel's `ffn` totals in `order-cross.json` disagree with
it by 2-3% in both directions.

`268a7387` measured what moves that clock while this panel was running and it is not thermal: the
CPU and the 8060S share one 120 W socket, and a twelve-way build beside a lock hold takes the
delivered shader clock from 2689 MHz to 2190, which is about 12% of a 128-row pass. `--pin-clock`
asks for 2900 and gets what the power budget leaves. **Two consequences for anything read here:**
compile outside a peer's lock hold, and trust the normalised column over the raw one. This
iteration's byte cut is also a counter-example to the cheerful version of that finding - halving
the FFN weight stream hands watts back to the shader clock and still measures a null, because this
phase was not waiting on those bytes.

## What this rules out, and what it hands over

- **The weight stream is not what the deployed FFN phase pays for.** Halving it is a null here, and
  [pinning it entirely into cache](ffn-slice-issue-order.md) is -8.0% of the phase. Two independent
  arms, same conclusion: this kernel is about a tenth memory on the weight side.
- **Neither is the block loop's instruction count.** A 7.6% cut in issue slots per row converts at
  zero. Do not spend a turn shortening this loop without first showing what stall the shortening
  removes.
- **The binding term is operand cover in the group loop, and moving a request by hand does not fix
  it.** Group `g`'s fragments are requested inside the `kb` loop that consumes them, behind the
  `sched_barrier(0)` that stops the scheduler hoisting 32 VGPRs of them per group. Hoisting the one
  request that matters is built, costs six registers, and loses 2.1% at two groups and 8% at four,
  because the allocator sinks group 0's operands to pay for it. **The compiler is defending a local
  optimum and the next arm here has to change what is in flight, not when it is asked for** - the
  A4 FFN's answer to the same problem was to move the operand to a different pipe entirely
  ([the activation stage](ffn-b-operand.md), `global_load` to `ds_load`, -4% of its phase), and
  that is the shape this kernel has not tried. If something does raise group 1's cover to group 0's
  260 slots, re-asking the rung is one build: `make DEFS=-DHALO_SLICE_WIDE=1` and
  `--ffn-slice 32,64`.
- **The activation stream is 4.6x the weight stream and nothing has moved it.**
  [Straightening its address](ffn-slice-operand.md) cost 4%, so it is not cache-line count; the
  cover table says it is when the request is issued.

## Reproducing

```sh
# the rung, one build, and it changes nothing at FMAX 32
make DEFS='-DHALO_SLICE_WIDE=1' -j8 tools/batch_profile

# both arms in one process, one clock; per-launch times are in the trace
tools/run-batch-compare --pin-clock --profile-tool \
  --out ../../data/bonsai2/batch-comparison/ffn-slice-rows/m20-grid.json \
  --modes 20 --rows 128 --heads 1 --traces 1 --rounds 2 --warmup-ms 600 --context 128 \
  --ffn-slice 32,48,64

# offline: registers, block census and cover, no GPU and no lock
tools/kernel_resources.py -D HALO_SLICE_PROBE=64 kernels/halo_rows.hip k_ffn_slice
hipcc --offload-arch=gfx1151 -O3 -std=c++17 -Isrc -Ikernels \
  -I../bonsai-hip/include -I../bonsai-hip/ggml/include \
  -D HALO_SLICE_PROBE=64 -S --cuda-device-only kernels/halo_rows.hip -o /tmp/s64.s
tools/isa_loop_count.py /tmp/s64.s k_ffn_slice
tools/vmcnt_cover.py /tmp/s64.s k_ffn_slice
```

Raw samples, both listings and the cover tables are in
[`batch-comparison/ffn-slice-rows/`](../../../data/bonsai2/batch-comparison/ffn-slice-rows/README.md).

**A note on cross-run comparison in this panel.** The box had four to nine engineers compiling
during these runs and the shader clock moved with them: the same default build measured `ffn` at
182 ms in one hold and 201 ms in another, with every untouched phase scaling by the same factor.
Only the in-process paired arms above are comparable, which is the discipline
[the state horizon](gdn-state-horizon.md#throughput-on-one-build-in-one-process) already wrote down
for a 1% effect and which a 0.2% null needs even more.

# The FFN slice's scattered activation operand is free, and straightening it costs 4%

[The second column group](deployed-matvec-groups.md) closed by naming what it thought was the next
thing in that body, and the prediction was specific:

> **The activation fragment is on the wrong axis and it is now the biggest uncounted cost.** `xq` is
> row-major, so the sixteen columns of one `global_load_b128` are sixteen different token rows, 5120
> or 17408 bytes apart: **eight loads a group, sixteen distinct cache lines each, 128 line requests
> per block where 16 would do.** [...] This is the change that would let four groups pay, since what
> stopped them was fragment loads rather than the peel.

The arithmetic is right and the fix works exactly as designed. It is a **3.6 to 4.2% loss** on the
phase, reproducibly, in two independent panels, and the mechanism is the opposite of the prediction:
the scattered addresses were not costing anything, and the address arithmetic that produced them was
*earning* its slots by filling gaps in the radix-3 peel's dependence chain.

Nothing shipped. The implementation, the patch against `598ff76`, the compiled listings and the raw
samples are in
[`batch-comparison/ffn-slice-operand/`](../../../data/bonsai2/batch-comparison/ffn-slice-operand/README.md).

## The coordinate

A `v_wmma_i32_16x16x16_iu8` B fragment wants lane `l < 16` to supply matrix column `l` — activation
row `l` — as that row's sixteen K values in its own sixteen bytes. Lanes 16..31 repeat lanes 0..15.
So in `xq[row][K]` one fragment load addresses sixteen rows one row stride apart, and in
`xq[k-group][row][16 B]` it addresses 256 contiguous bytes.

Per 128-K block at two column groups:

| | `[row][K]` (deployed) | `[k-group][row][16 B]` |
|---|---:|---:|
| fragment loads | 16 | 16 |
| cache lines touched per load | 16 | 2 |
| block-scale and block-sum loads | 4 | 4 |
| lines per scale load | 16 | 1 |
| **line requests a block** | **320** | **36** |
| distinct lines a block | 32 | 32 |

The distinct-line count is equal because it is the same 4 kB of activations either way. What the
coordinate removes is 87% of the *(instruction, line)* pairs the address unit has to resolve.

The map is not speculative: `prep_chunk_r` already writes it for the wide sequence projections
(`seq_q`, element `e` of row `r` at `((e >> 4) * npad + r) * 16 + (e & 15)`), three lines above the
row-major store this experiment replaced. The experiment gave `prep_chunk_r` and `ph_prep` an `XR`
row-stride parameter, gave `k_ffn_slice` a `bool XF` that drives both the quantiser's store and the
matvec's read, and put the reading body in `mv_frag.hpp`. `XF = false` compiles to the code the
kernel compiles today, instruction for instruction.

## It is an exact relabeling, and that part is worth keeping

Same quantiser, same `amax`, same level, same rounding, same warp block sum, same 128-K block
boundaries, same K order inside a block, same lane-to-column map, same `int32` accumulation seeded
with the negated block sum, same `fmaf(C, wscale * xscale, y)` drain in the same order. Only the
address a value lives at moves.

`tools/batch_compare --prefill-identity --ffn-op 0,1 --prefill-rows 128,256`, mode 20, a 384-token
document:

| rows a pass | passes | logit values | `row` | `frag` |
|---:|---:|---:|---|---|
| 128 | 3 | 95,354,880 | `9524621175407926160` | `9524621175407926160` |
| 256 | 2 | 95,354,880 | `9524621175407926160` | `9524621175407926160` |

**381,419,520 full-vocabulary logit values, one hash, zero non-finite.** The residual FNV-64 of
every traced 128-row pass in both timing panels is `8785572807677303713` in all ten samples. A
future engineer who wants this operand for a different consumer can take the map knowing it is
exact; what follows is only about whether it pays here.

## Measured: it loses

Mode 20 (`wide-deployed`, the route the resident service ingests prompts on), 128-row passes, both
coordinates alternating inside one process on one build and one pinned clock, `--ffn-op 0,1`. The
five untouched phases are the in-panel control.

| panel | arm | `ffn` ms | control ms | `ffn` / control | device span |
|---|---|---:|---:|---:|---:|
| A, 2 rounds | row | 183.241, 175.519 | 59.607, 57.557 | **3.0620** | 239.557 |
| A | frag | 179.735, 178.680 | 56.564, 56.419 | **3.1723** | 237.222 |
| B, 3 rounds | row | 183.924, 160.943, 167.650 | 60.061, 53.210, 55.405 | **3.0385** | 228.587 |
| B | frag | 188.476, 169.055, 171.943 | 59.140, 53.504, 54.555 | **3.1667** | 233.741 |

**+3.6% and +4.2% against the control, the same direction in two panels taken twenty minutes
apart.** The raw `ffn` column is noisier than the effect — one arm's own rounds differ by 12% on a
box carrying five other engineers — which is why the ratio is the result and the span is only a
sanity check.

Both arms ran on **grid 60 workgroups**, printed per coordinate by `HALO_FFN_GRID=1`:

    [ffn slice] rows<=32 map 0 operand row: grid 60 workgroups
    [ffn slice] rows<=32 map 0 operand frag: grid 60 workgroups

## Why: the scattered addresses were paying rent

The whole-kernel census, `hipcc --cuda-device-only -S` on the probe, `k_slice_probe<32, *, 6>`:

| | instructions | `global_load` | `global_store` | `v_wmma` | `s_delay_alu` |
|---|---:|---:|---:|---:|---:|
| row | 5160 | 124 | 38 | 96 | 541 |
| frag | 5237 | 124 | 38 | 96 | 593 |

Same loads, same stores, same matrix instructions. **Every added instruction is a scheduling hint**,
and they land in one place — the gate/up matvec's block loop, 561 → 620 instructions, of which
`s_delay_alu` goes 50 → 112 while *work* goes 488 → 483. The down projection's loop does not move
(550 → 548).

What the hints sit in front of says the rest: `v_pk_mul_lo_u16` 26, `v_pk_lshrrev_b16` 19,
`v_lshl_or_b32` 16, `v_and_b32` 15 — the radix-3 peel, which this change does not touch a line of.
In the deployed arm the peel's dependent pairs are separated by the per-lane address arithmetic that
`xrow + b * 128 + g * (16 * 128 * NB) + kb * 16` needs. Take that arithmetic away, as a single base
register with immediate offsets 0, 512 … 3584 does, and the peel chain closes up and the compiler
has to tell the hardware about interlocks it used to cover for free.

So the block loop's currency is **issue slots on the peel's dependence chain**, and a `b128` load
whose sixteen lanes hit sixteen *resident* lines costs what one that hits two costs, as long as the
eight loads are independent and the wait is at the end. The census predicts +10.5% on the two thirds
of the weight stream that gate and up carry, or about +7% of the phase; it measured +4%, so issue is
most but not all of what that loop spends.

This does not contradict [the packed state's request result](gdn-state-coord.md), which won 9.4% by
taking a wave's state loads from 8 requests to 3. That is a per-unit prologue whose `s_waitcnt` is
on the critical path of every unit. Here the same instruction count covers the same bytes with the
loads already in flight. **Request count matters where a wait waits on it.**

## It does not enable the occupancy lever either

`2464a25e` measured the register ladder of the same body and asked whether the coordinate would pay
for the eighth wave. It does the opposite: `tools/kernel_resources.py` on the probe, one compile,

| `waves_per_eu` | row VGPR / spill | frag VGPR / spill |
|---:|---:|---:|
| 6 | 238 / 0 | 239 / 0 |
| 7 | **205 / 0** | 216 / 1 |
| 8 | 192 / **18** | 192 / **39** |
| 10 | 144 / 162 | 144 / 176 |

Eight independent loads from one base are eight live destination quads; the deployed arm's address
chain lets the allocator stagger them. At the deployed six-wave budget both arms sit inside the same
240-register granule and take the same grid, which is why the panel above is a clean comparison —
but there is no register dividend to spend on occupancy.

At sixteen rows the frag arm is 216 VGPR against 219 and the compiler reports seven waves against
six. `coop_grid` gives both 60 workgroups, so that reading buys nothing on this device.

## What this closes, and what it points at

- **The `[block][K slice][row]` operand is measured and it loses.** Nobody needs to build it again
  for this kernel. The four-group shape it was supposed to rescue was already
  [3.6% behind two groups at equal grid](deployed-matvec-groups.md#the-grid-is-the-occupancy-number-and-four-groups-still-lose-when-it-is-level);
  both halves of that prediction are now answered.
- **The lever in this body is the peel, and it landed the same evening.**
  [The trit was already a byte of the product](mv-peel-gather.md): `2f17d1b2`'s `halo_expand.hpp`
  keeps the `r * 3` product instead of shifting it and gathers four trits from two products with one
  `v_perm_b32`, 232 → 168 operations per 128-weight block. `mvw_rows` spent about 230 of its 488
  work instructions there. Measured on this loop with this panel's own instrument: block loop 561 →
  476, registers unmoved, `ffn` **-1.66%** normalised, full-model prompt **+1.40%** median paired.
  The prediction this section makes is the one that held.
- **Do not price a change to this loop on cache lines.** Price it on issue slots and on what the
  dependence chain can hide.

## Reproduction

The patch, the probe and the samples are beside this document's raw data. With the patch applied:

```sh
# the census: registers, occupancy and the block loop, 1.4 s, no GPU and no lock
tools/kernel_resources.py kernels/mv_frag_isa.hip k_slice_probe
hipcc --offload-arch=gfx1151 -O3 -std=c++17 -Isrc -Ikernels --cuda-device-only -S \
      -o /tmp/frag.s kernels/mv_frag_isa.hip
tools/isa_loop_count.py /tmp/frag.s k_slice_probeILi32ELb1EE

# the phase panel, both coordinates in one process
HALO_FFN_GRID=1 tools/run-batch-compare --pin-clock --profile-tool \
  --modes 20 --rows 128 --heads 0 --traces 1 --rounds 3 --ffn-op 0,1 --out .../phase.json

# the bits
tools/run-batch-compare --modes 20 --prefill-rows 128,256 --prefill-identity \
  --ffn-op 0,1 --rounds 0 --out ... --tag ident
```

Without the patch, `2464a25e`'s `-DHALO_SLICE_PROBE=32` compiles the real `k_ffn_slice` alone in
three seconds and is the better instrument for any further work on this kernel; the probe here
carries a structural copy only because it predates that flag.

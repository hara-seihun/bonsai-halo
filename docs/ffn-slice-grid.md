# The deployed FFN's launcher gave every width the widest body's grid

`launch_ffn_slice` computed one cooperative grid and used it for the whole kernel ladder:

```c
static int grid = std::min({coop_grid(k_ffn_slice<1>), ..., coop_grid(k_ffn_slice<FMAX>)});
```

The minimum is set by the two-group body, which needs 238 VGPRs and gets three workgroups per WGP.
Every narrower width inherited that. `k_ffn_slice<8>` is the scalar-fed dot4 body at **73 VGPRs and
sixteen waves per SIMD32**, and it was launching on 60 workgroups where its own registers and shared
block allow 120. Twenty lines above, `launch_sequence_part` already keeps a per-shape table for
exactly this reason - "each part is a short single-layer body that shares its launch with nothing,
so it takes the whole grid its own registers allow" - and the FFN slice never got the same
treatment. `--single-map 4` exists because somebody hit the symptom and worked around it for the
one-row body alone: it selects the same kernel as map 0 to get a grid of its own.

A per-width table is **-18.1% raw and -21.4% normalised on the eight-row FFN slice phase, and +4.0%
to +6.9% on eight-stream aggregate generation**, with the same executable, the same bytes, the same
K order and identical output tokens.

The second half of this iteration is a negative on the same axis: **buying the fourth workgroup per
WGP with registers loses at both widths that can reach it**, and the width that spills nothing is
what turns that from a spill story into a mechanism.

## What runs on which grid now

| rows a slice | body | VGPR | LDS bytes | grid before | grid now |
|---:|---|---:|---:|---:|---:|
| 1 | dot4 | 78 | 4132 | 60 | 160 |
| 2, 4 | dot4 | 79 | 4132 | 60 | 160 |
| 5..8 | dot4 | 73 | 8228 | 60 | **140** |
| 9..16 | WMMA, one column group | 219 | 8228 | 60 | 60 |
| 17..32 | WMMA, two column groups | 238 | 8228 | 60 | 60 |

The two widths that do not move are the point of the table: 219 and 238 VGPRs are both three
workgroups per WGP, so **the served prompt route cannot change**, and it does not - the 32-row cell
is a null by construction and a measured null below.

`coop_grid`'s model (`kernels/phases.hpp`) is granule 24, 1536 registers per SIMD32, blocks =
waves/2, times 20 WGPs, capped by what the occupancy API reports for the shared block. That cap is
why the eight-row width stopped at 120 and not 160: at 9508 bytes a block, 65536 bytes of LDS is six
workgroups. This kernel runs prep, matvec, prep, matvec and never touches the attention score array
that `LDS_FLOATS` is sized for, so `slice_lds_floats(TT)` now asks for what its own phases use -
1024 floats of `prep_chunk_r` staging, `[8][TT][32]` for the dot4 reduction or `[7][8][32]` plus a
32-float scale table per wave for the WMMA one. 9508 bytes becomes 8228, which is seven workgroups,
and 4132 at four rows, which reaches the register ceiling of eight.

**The extra blocks past 120 are a measured null**, and that is the more useful half of it: 422.795
and 418.749 ms at 140 against 419.491 and 419.659 at 120. A 128-row pass sliced eight rows at a time
reads 54.7 GB of weights, so 419 ms is **130 GB/s** - the same wall `docs/mv-dot8-nibble.md`'s
successor measured on the eight-row matvec body inside the persistent kernel (127 GB/s at 74% of its
own issue model). The eight-row machine has one roof and block count is no longer what holds it
down.

## Measured

Build `36eb87a0f75795e6201ebfe0` (and `357e3b5e34206d1e0427ee32` before the shared-block change),
one executable per panel, arms selected by `HALO_SLICE_GRID` inside `tools/run-batch-compare
--pin-clock`, which holds the GPU lock and pauses the resident service. `HALO_SLICE_GRID=0` is the
predecessor: one grid for the whole ladder.

**The phase**, mode 20, 128-row passes, `--ffn-slice` walking the width inside each process, `ffn`
over the sum of the five untouched phases:

| arm | rows a slice | grid | `ffn` ms | `ffn` / control |
|---|---:|---:|---:|---:|
| shared minimum | 8 | 60 | 512.062, 512.382, 510.916 | 9.0848, 9.0780, 9.0616 |
| **per width** | **8** | **120** | **419.491, 419.659** | **7.1474, 7.1591** |
| per width + own block | 8 | 140 | 422.795, 418.749 | 7.1135, 7.0884 |
| shared minimum | 32 | 60 | 178.992, 168.738, 165.692 | 2.7774, 2.7398, 2.7435 |
| per width | 32 | 60 | 176.316, 167.460, 168.573, 168.866 | 2.7846, 2.7544, 2.7506, 2.7504 |

The 32-row cell is the in-panel null: same kernel, same grid, seven samples across four processes
and two builds, 2.740 to 2.785.

**Full model**, `tools/batch_compare --only decode --modes 4 --streams 8`, which is the shape that
routes 5..31 rows to this kernel (`mode = total <= 4 ? 0 : total < 32 ? 4 : wide`), two rounds an arm:

| arm | aggregate tok/s | build |
|---|---:|---|
| shared minimum | 132.34, 132.80 | `357e3b5e` |
| **per width** | **137.13, 138.52** | `357e3b5e` |
| shared minimum | 131.41, 132.62 | `36eb87a0` |
| **per width + own block** | **140.68, 141.53** | `36eb87a0` |

**+4.0% and +6.9%.** The two builds bracket the same change; read the direction from both panels and
the size as "four to seven percent".

**The paths this must not move.** Mode 20 prompt ingestion at 128 rows per pass, four processes in
palindrome order on the same binary: `551.9/549.3` median for the shared minimum and
`524.7/552.2` for the per-width table. One of those four processes reads 5% low and the arms
otherwise sit on top of each other; this box's process-to-process spread tonight is 521.5 to 553.2
inside a single arm. **The traced phase is the resolving instrument here, and it says null.** The
32-stream wide route never calls this kernel with a width the change touches: 297.3 against 301.2
tok/s, one round each, inside the same spread.

## Installed

Canonical `cbde40d`, rebased onto [the peel gather](mv-peel-gather.md) and rebuilt after it, so the
accepted binary carries both changes. `bonsai-halo`
`7a6ad4fde92266362a22997a872f2d7e...`, `tools/batch_compare`
`24263ec03fa57ea1b74afb8188f4025e...`, `tools/batch_profile`
`14703392df611da09df6c85efef8708f...`.

| arm | 8-stream aggregate tok/s, installed binary |
|---|---:|
| shared minimum | 135.6, 135.8 |
| **per width (the default)** | **141.3, 142.8** |

**+4.7%**, and the 192-token continuation is the same sha256
`c256dc970ea6024100e684ed` it was on both pre-install builds and on the build before the peel
gather - three executables, two independent bit-identical changes, one token stream.

Single-stream decode on the installed engine is untouched by construction, because a one-row step
never reaches this launcher: `--bench` reads 33.11 tok/s and 248.6 tok/s of prompt on a 23-token
prompt, inside the band this box gives while other engineers are measuring.

## Bit-identity

Nothing about a numerical map moves. The same weights are read in the same K order, every output
element is accumulated by one block in the same FP32 chain, and only the number of blocks the work
is spread over changes.

- **The matvec unit is the atom.** `ph_matvec_w` and `ph_matvec` both give a whole `(tile, part)`
  unit to one block, and a block reduces its eight waves through LDS in a fixed order.
- **`ph_prep`'s norm is computed inside one block**, striding the whole row per workgroup, so the
  RMS reduction cannot see the grid.
- **The only cross-block accumulation is the down projection's `KS = 2` atomic pair**, and two
  addends commute exactly in FP32: `0 + a + b` and `0 + b + a` are the same word.

`residual_fnv64` is `315472146097739578` in all 14 traced passes across every arm, width and build.
The eight-stream decode arms emit token-identical continuations: 8 streams x 24 tokens, sha256
`c256dc970ea6024100e684ed6cf5fb82` in both.

## The fourth workgroup is not worth its registers (bounded negative)

`amdgpu_waves_per_eu` was a literal on this kernel - 6 above sixteen rows, no ask below - and it is
now the template parameter `OCC`, so a build can carry both budgets. The ladder, from
`tools/kernel_resources.py -D HALO_SLICE_PROBE=N -D HALO_SLICE_WPE=W`, three seconds each:

| body | ask | VGPR | spill | waves | workgroups per WGP | grid |
|---|---:|---:|---:|---:|---:|---:|
| two groups (32 rows) | 6 | 238 | 0 | 6 | 3 | 60 |
| two groups | 7 | 205 | 0 | 7 | 3 | 60 |
| two groups | **8** | 192 | 18 | 8 | 4 | 80 |
| two groups | 9 | 168 | 76 | 9 | 4 | 80 |
| one group (16 rows) | none | 219 | 0 | 6 | 3 | 60 |
| one group | **8** | 187 | **0** | 8 | 4 | 80 |

Seven waves is free and buys nothing, because three workgroups is three workgroups. Eight waves buys
a fourth, and both widths get slower:

| rows a slice | arm | `ffn` / control |
|---:|---|---:|
| 16 | six waves, grid 60 | 3.4027, 3.3446 |
| 16 | eight waves, grid 80 | 3.4795, 3.4689 |
| 32 | six waves, grid 60 | 2.7774, 2.7398, 2.7846, 2.7544, 2.7506, 2.7504 |
| 32 | eight waves, grid 80 | 3.0499, 3.0411, 3.0316, 3.0013 |

**+2.6% and +9.8%.** The one-group cell is the one that names the mechanism, because it **spills
nothing**: the allocator pays for the smaller register file inside the block loop, 445 -> 518
instructions at one group and 565 -> 614 at two, with `v_wmma` at 16 and 32 and `global_load` at 10
and 20 unchanged in both. A third more blocks does not cover a sixth more issue in a loop that is
already issue-bound. The 18 spilled registers at two groups are a red herring - they land in the
drain, and the hot blocks contain no `scratch_` instruction at all.

This is the same trade `docs/ffn-decode-bytes.md` measured on `k_proj_opt`, where asking for eight
waves cost the dense arm 54.7%, and the opposite of what "more waves hide more latency" predicts.
Two kernels, two shapes, one answer: on gfx1151 an occupancy ask is priced in issue slots, and it
has to be read as such.

The losing arm is **not in the shipped build** - `slice_fn` compiles one instantiation per width -
and `-DHALO_SLICE_OCC_CONTROL=1` rebuilds both for a re-fit. That re-fit is worth running after
anything that shortens this block loop, because the whole negative is 8 to 16% of issue against 33%
of blocks.

## Instruments this leaves

- **`-DHALO_SLICE_PROBE=N` compiles `k_ffn_slice<N>` alone**, cutting the fifty `k_forward_rows`
  shapes that make this translation unit a minute of `hipcc`. Registers, spills, LDS bytes and the
  compiler's own occupancy for any width in three seconds and no GPU:
  `tools/kernel_resources.py -D HALO_SLICE_PROBE=8 kernels/halo_rows.hip k_ffn_slice`.
- **`HALO_SLICE_GRID=0`** restores the predecessor's shared minimum, so the block count stays an
  in-process axis rather than a second build.
- **`HALO_FFN_GRID=1`** now prints the grid per (width, single map, occupancy arm) instead of per
  width, which is how every grid in this document was read.
- **`HALO_SLICE_OCC_ARM=0|1`** and `ffn_slice_set_occ()` pin the budget in a control build.

## What is left here

- **The eight-row band is at 130 GB/s and block count will not move it.** The same roof shows up on
  the persistent kernel's eight-row matvec. Whatever raises it is one change for both.
- **9..31 rows is stuck at three workgroups per WGP** and the only way to a fourth that this
  iteration found costs more issue than it buys. A change that removes real work from the block loop
  - the radix-3 peel is about 230 of its 488 work instructions - would be worth re-reading the
  ladder against, since 187 VGPRs at one group already spills nothing.
- **`--single-map 4` is now redundant**: map 0 and map 4 select the same kernel and now also the
  same grid. It costs nothing to keep and one line to drop when somebody re-fits the single-row
  maps.

Raw samples, the register ladder, the block-loop census and the compiled listings are in
[`batch-comparison/ffn-slice-grid/`](../../../data/bonsai2/batch-comparison/ffn-slice-grid/README.md).

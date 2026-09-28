# One kernel, one score body: the coordinate is an address

[The score unit's lane coordinate](attn-score-lane.md) measured **-13.5%** on attention at position
1024 and **-14.9%** at 3072, then measured **+9.7%** on the next build and shipped switched off. Its
author diagnosed the reversal exactly and could not act on it, because the file was held:

> The register allocator holds the union of two full bodies across a runtime branch ... **The fix is
> small and named: move the arm dispatch up to the launcher.**

That was half of it. Doing it found the other half, which nobody had looked for and which was
costing the **deployed** path rather than the experimental one.

## The persistent kernel had been carrying the leaf body for three commits

`ph_attn` is not a launch. It is inlined into `k_forward_rows`, the persistent kernel whose register
count sets [the cooperative grid](rows-grid-occupancy.md) of every wide pass in this engine. A
runtime branch inside `attn_chunk_unit` therefore reached far past the kernel it was written for.

`tools/kernel_resources.py -D HALO_ROWS_PROBE=8` compiles one instantiation and needs no GPU:

The number that decides a grid is the **loader's**, not the compiler's — `5b7c49d1`'s rule, and
`tools/rows_occ_probe` is what asks. Both trees below were probed on this device within one minute
of each other, canonical at `dc331a5`:

| `k_forward_rows<8, ...>` | canonical `dc331a5` | this change |
|---|---:|---:|
| fp32, allocator's budget — **what the deployed route launches** | 206 regs, 3 blocks/WGP, **grid 60** | 141 regs, 4 blocks/WGP, **grid 80** |
| packed, allocator's budget | 211, 3 blocks, grid 60 | 146, 4 blocks, grid 80 |
| either, `waves_per_eu(12)` | 120, 5 blocks, grid 100 | 120, 5 blocks, grid 100 |
| spills on that 120-register arm (compiler) | **108** | **12** |

`56b4448` measured the fp32 row at **135 registers, 5 blocks, grid 100** and bought +22.8% prompt
and +20.0% drafted generation by taking the packed arm from 80 workgroups to 100. `04898ea` then put
a second score body behind `if (P.k_tiled)` inside the shared unit, and the allocator did what
allocators do. **The deployed persistent kernel has been running on three fifths of its grid since**,
and the arm that holds its register count by asking for twelve waves paid in spills instead: 108 of
them, against the five that `rows-grid-occupancy.md` measured at 3.5% of a verify pass.

Nothing in the leaf coordinate's own panel could see it. That panel measures `k_attn_wide`.

**The 141 is not a residue of this change.** Forcing the coordinate to a compile-time false through
the same probe also reads 141, so the address select below costs nothing; the six registers over the
published 135 arrived with the narrow row-group ladder in `5bf19ce`, which landed in the same window.

## The coordinate is an address, not a body

The obvious repair — template `k_forward_rows` on the coordinate — doubles a translation unit that
already takes 42 seconds of every engineer's turn, and it treats the storage format as if it implied
an algorithm. It does not.

A whole eight-dimension leaf is contiguous in **both** forms. Row-major keeps a key's 256 dimensions
together; leaf-interleaved keeps one leaf of 32 consecutive keys together. Either way the sixteen
bytes a lane wants are sixteen contiguous bytes, so a row-major score unit reads a leaf-ordered
cache by changing one offset expression:

```cpp
__host__ __device__ __forceinline__ size_t k_pos_off(bool tiled, int pos, int d, int hd) {
    return tiled ? k_leaf_off(pos, d, hd) : (size_t) pos * hd + d;
}
```

Same bytes, same values, same expansion, same `fmaf` chain, same `warp_sum`, same store. The write
side in `ph_attn_pre` has always selected this way; the read side now does too. So *correctness* in
either coordinate costs no second body, and what stays a compile-time choice is only the
**algorithm** — lane-per-key, which one kernel wants and the persistent kernel never did.

That separation is what makes the dispatch cheap. `ATTN_KM_ROW` is correct everywhere, so
`k_forward_rows` compiles it alone and the process setting stays free to move under it.

## What each kernel is worth now

`tools/kernel_resources.py kernels/attn_batch.hip`, same build, KM as the last template argument
(0 row-major, 1 leaf, 2 the fused control):

| kernel | on `cd47cf6` | split | |
|---|---:|---:|---|
| `k_attn_wide<8, 0/1, 1, ROW>` | 130 / 10 | **92 / 16** | the deployed prefill arm |
| `k_attn_wide<8, 0, 1, LEAF>` | — | **94 / 16** | the lane-per-key arm |
| `k_attn_wide<16, 0/1, 1, ROW>` | 251 / 5 | **180 / 8** | |
| `k_attn_wide<16, 0, 1, LEAF>` | — | 182 / 8 | |
| `k_attn_wide<8, 1, 1, DYN>` | 130 / 10 | 130 / 10 | the control reproduces `cd47cf6` exactly |
| `k_attn_wide<1, 0, 6, ROW>` | 70 / 16 | 70 / 16 | the folded-head unit never had a leaf body |

94 VGPR and sixteen waves is the register count `attn-score-lane.md`'s **-13.5%** panel was taken at,
recovered on a tree that has the rescheduled body, the narrow ladder and the head fold in it.
180 / 8 also repairs the 16-row arm, which [long-context-attention.md](long-context-attention.md)
measured at 180 / 8 before it drifted to 251 / 5.

### Four bodies is the same union as two

Splitting the coordinate branch alone left the leaf arm at **130 / 10**, which is where the census
stops being about the coordinate at all. `attn_chunk_group`'s narrow ladder inlines the unit at
TT = 1, 2, 4 and 8, so a fused kernel compiles *eight* score bodies and a leaf kernel still compiled
four. `ATTN_KM_LEAF` therefore takes the group width alone, and that is the 130 -> 94 step.

The cost is bounded and it is not on a default. A pass whose groups all carry one row never reaches
the ladder — `launch_attn_wide` sends it to the folded-head unit, which has no leaf body — so the
absence is visible only on mixed group widths in the leaf coordinate. Giving it back is a
launcher-level `TT = 1` instantiation, not a second body in the unit.

## The control is in the build, on the axis that already existed

`--attn-sched 2` runs the rescheduled body through the **fused** kernel: same coordinate, same
schedule, same everything except that the kernel also compiles the body it will not run. That is the
delivery priced directly, in one process, against `--attn-sched 1`, and its register table above is
identical to `cd47cf6`'s, which is what makes it a faithful control rather than an approximation of
one. Nothing selects it by default.

`--attn-coord 0,1` crossed with `--attn-sched 1,2` is a 2x2: both coordinates, both deliveries, one
clock.

## Results

Raw samples in
[`attn-coord-dispatch/`](../../../data/bonsai2/batch-comparison/attn-coord-dispatch/).

### The wide route: the delivery was worth a fifth of the phase

Mode 19, 128-row passes, `--attn-coord 0,1` crossed with `--attn-sched 1,2`, every cell in one
process on one clock. `sequence-core` is attention; the control is the six phases the change cannot
reach. Two rounds per cell at position 1024
([`prefill-1024-r2.json`](../../../data/bonsai2/batch-comparison/attn-coord-dispatch/prefill-1024-r2.json)),
one at 3072 ([`prefill-3072.json`](../../../data/bonsai2/batch-comparison/attn-coord-dispatch/prefill-3072.json)).

| position 1024, ms per pass | row / split | row / fused | leaf / split | leaf / fused |
|---|---:|---:|---:|---:|
| `ffn` | 70.720 | 71.832 | 71.901 | 70.594 |
| `sequence-input-projection` | 39.794 | 40.489 | 40.624 | 39.944 |
| **`sequence-core`** | **18.398** | 22.820 | **17.606** | 24.627 |
| `sequence-output-projection` | 15.316 | 15.590 | 15.627 | 15.380 |
| `gdn-resident-core` | 13.629 | 13.883 | 13.940 | 13.808 |
| control (six untouched phases) | 143.788 | 146.174 | 146.505 | 144.070 |
| device span | **163.604** | 170.415 | 165.502 | 170.094 |

Normalised on the control, which drifts 1.9% across the whole panel:

| | position 1024 | position 3072 |
|---|---:|---:|
| **split vs fused, row-major — the default coordinate** | **-18.0%** | **-21.7%** |
| split vs fused, leaf coordinate | -29.7% | -30.3% |
| leaf vs row, both split — the arm itself | **-6.1%** | **-4.0%** |
| leaf vs row, both fused — the published reversal | +9.5% | +7.8% |

The last row is the load-bearing one for trusting the rest. `attn-score-lane.md` measured that cell
at **+9.7%** on a different tree and a different afternoon; the fused control reads **+9.5%** here.
It is reproducing a known number, which is what a control is for.

**The default route gets the larger half.** Nothing about the coordinate changes for it — the split
is worth 18-22% of the attention phase in the coordinate this engine actually serves, which at
position 1024 is **-4.0% of the device span of a 128-row pass**, 163.604 against 170.415 ms, or
+4.2% of prompt throughput at that width. Attention grows with position, so does this.

**And the lane-per-key arm is worth something again.** -6.1% and -4.0% normalised, against the
-13.5% its first panel measured: the baseline body got 12% faster in between, exactly as its author
predicted. It is no longer a negative, and it still is not a default, because
`attn_coord()` is a whole-process setting and nobody has yet measured the persistent kernel reading
an interleaved cache through `k_pos_off`.

**Bit-identical, every cell, both positions.** Residual FNV-64 `13477932824292311186` in all
sixteen samples at 1024 and `2523245940775455799` in all twelve at 3072 — the hashes
`attn-score-lane.md` published for these shapes.

### The deployed route: the grid came back

`tools/run-batch-compare --engine --bench`, DFlash2 q4, deployed eight-row prompt route
(`--prefill-sweep off`), 949-token prompt, 64 greedy tokens. The engine's own `[rows grid]` line
confirms which instantiation runs: *fp32 state, budget allocator* — the row that lost two of its
five blocks.

| | prompt tok/s | drafted tok/s | verify ms/step |
|---|---:|---:|---:|
| canonical `dc331a5`, grid 60 | 141.1 | 41.95 | 58.137 |
| this change, **grid 80** | 145.1, 145.5 | 42.66, 42.93 | 57.013, 56.648 |

Greedy digest `18433580762981189499` and 161 drafted / 41 accepted in every run of both builds.
This pair is cross-process and cross-build — a register count is not an in-process axis — so read
it as +3% prompt and -2% on the verify pass with the clock as an uncontrolled term, and read the
loader table above as the part that is certain.

## What the coordinate is worth now, and what still gates it

[The reuse fold](attn-reuse-fold.md) re-measured this arm on a later build and took the one
measurement this document names as missing. The leaf coordinate is **-6.7%** of the score kernel at
position 3072 with the kernel traced separately, and folding two query heads into a unit - which
only this coordinate can afford - is a further **-4.06%**, so the pair is -10.4% against today's
default. On the deployed drafted path the coordinate is neutral and exact: greedy digest
`15165079467408989842` and 189 drafted / 21 accepted in both coordinates, prompt -0.9%, drafted
generation -0.1%, verify +0.2% across one cross-process pair. What is still missing is an
in-process instrument for the eight-row route, because `tools/batch_profile`'s decode path rejects
mode 0.

## What this does not touch

The leaf coordinate's arithmetic, `kernels/attn_leaf.hpp`, `leaf_of_pos`, the relabeled tree and its
device probe are exactly as `attn-score-lane.md` published them. The V cache, `ACHUNK`, the
chunk-partial layout, the softmax, the value loop and the combine are untouched. `k_attn` on the v1
per-op path keeps its own inline leaf arm. No serving default moves: `attn_coord()` still defaults
to 0, and every kernel the default route launches computes the bits it computed before.

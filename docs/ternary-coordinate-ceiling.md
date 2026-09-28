# The ternary coordinate is not what the eight-row pass is short of, and a wider image cannot pay

[Three weight coordinates in one kernel](drafted-step.md) measured the target model's ternary HALO
stream at **137 GB/s** inside `k_dflash` while the Q4 and Q8 streams beside it reached 200-220, fitted
`rate = 0.62 x (bytes per instruction) x 232 GB/s` to two shapes of that coordinate, and priced the
consequence:

> Either way **there is about 15 ms per drafted step in the coordinate, which is 24% of a 63 ms
> served step** — larger than everything inside the drafter put together.

That is the largest single number any document in this repository advertises, and the next engineer
to read the handoff would spend a turn building a 7 GB image to collect it. **It is not there.** The
0.62 is a property of the phase, not of the coordinate, and the two-bit image loses on arithmetic:
it is a 21.4% byte increase competing for a 16% rate headroom.

The panel below is a fresh map of the served step on canonical `f7b680d`, taken because the
correction needed both of its numbers from **one build, one process and one clock**.
Raw: [`ternary-coordinate-ceiling/`](../../../data/bonsai2/batch-comparison/ternary-coordinate-ceiling/).

## The same body, the same coordinate, the same eight rows, 20% apart in one pass

`k_forward_rows` computes the FFN gate/up projection and the vocabulary head with **the same
instantiation**: `ph_matvec_auto<NB_D, 1, TT, false, SM>`, `NB_D = 40` blocks, `KS = 1`, `TT = 8`,
through `mv_rows_t` over 32-row tiles of 896-byte HALO blocks. A unit is one tile — 35,840 weight
bytes against eight activation rows — and it is the same unit in both. They run in the same
cooperative launch, on the same grid of 100 workgroups, at the same occupancy, microseconds apart.

`HALO_PROFILE=1`, `--bench --dflash`, Q4 drafter, `--pin-clock`, **2812 MHz p50 (97% of nominal),
host 3.4 cores busy**, 48 tokens, medians of the two grid-100 cells of an A-B-A arm list:

| phase | units | tiles/instance | bytes/instance | ms/instance | GB/s | rounds | ideal rounds |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mv_lm_head` | 7760 | 7760 | 278.1 MB | 1.3358 | **208.2** | 78 | 77.6 |
| `mv_gate_up` | 1088 | 1088 | 39.0 MB | 0.2241 | **174.0** | 11 | 10.88 |
| `mv_qkv_z` | 512 | 512 | 18.4 MB | 0.1172 | 156.7 | 6 | 5.12 |
| `mv_down` | 320 | 160 | 19.5 MB | 0.1361 | 143.3 | 4 | 3.2 |
| `mv_qkv` | 416 | 416 | 14.9 MB | 0.1073 | 138.6 | 5 | 4.16 |
| `mv_ssm_out` | 320 | 160 | 6.9 MB | 0.0539 | 127.7 | 4 | 3.2 |
| `mv_o` | 320 | 160 | 6.9 MB | 0.0544 | 126.5 | 4 | 3.2 |

Whole verify pass 41.01 ms for 5.582 GB, **136.2 GB/s**; drafted generation **63.39 and 62.84 tok/s**,
digest `14250041415274144751`, 3.00 tokens/step in every cell.

**A property of a weight coordinate cannot differ by 20% between two phases of one pass that share
its instruction stream.** Per unit — the same 35,840 bytes of the same image through the same block
loop — the head costs **17.13 us** and gate/up costs **20.37 us**.

The drafter's own follow-up says the same thing from the other side. On the build
[the head-column window](drafter-head-columns.md) measured, `mv-lm-head` inside `k_dflash` is
**1.525 ms for 278.1 MB — 182 GB/s**, not the 137 the constant was fitted to, on the same coordinate
in the same kernel. The efficiency factor moved from 0.63 to 0.84 while the image, the expansion and
the census stayed where they were.

The census behind the fit is not in question — the ternary expansion really does deliver about 0.94
bytes per issued instruction at eight rows. But **an issue ceiling is a ceiling, and the head phase
is against it.** Converted at the clock this panel actually held rather than the nominal one,
`0.94 x 80 SIMD32 x 2.812 GHz` is **211 GB/s**, and `mv_lm_head` measured **208.2**: the deployed
ternary coordinate runs at 98.7% of its own issue model when the phase is long enough. There is
nothing left in it to collect.

## What a wider image would have to beat

Grant the two-bit arm everything: assume its expansion is free and it reaches the memory system's own
roof. Then price the bytes. A HALO block row is 26 code bytes plus a 2-byte fp16 scale per 128 trits,
28 bytes. A two-bit code is 32 plus the same scale, 34. The scale does not shrink, so the image grows
by **34/28 = 1.214** and the model's 5,877,784,576 bytes become 7.14 GB.

| arm | bytes | rate | time |
|---|---:|---:|---:|
| HALO, at the rate its own head phase reaches in this panel | 5.878 GB | 208.2 GB/s | **28.2 ms** |
| two-bit, granted the whole 242 GB/s `bench/bw` roof | 7.14 GB | 242 GB/s | **29.5 ms** |
| two-bit, at the 216 GB/s a Q4 stream actually reaches on this box | 7.14 GB | 216 GB/s | 33.1 ms |

**Break-even needs `208.2 x 1.214 = 252.8 GB/s`, which is above the DRAM roof of this machine.** The
conversion cannot win in `k_forward_rows` at any scheduling quality. In `k_dflash`, where the
coordinate is slowest and the comparison is friendliest to the arm, HALO at 182 GB/s is 32.3 ms
against a two-bit image at the 216 GB/s its own Q4 neighbour reaches in that kernel: 33.1 ms. It
loses in the kernel chosen to make it win.

### The general form, which closes the family rather than one arm

For any coordinate that trades bytes for expansion instructions, the most the trade can return is
the headroom between the current ternary rate and the rate a cheap coordinate reaches in the same
kernel:

```
k_forward_rows :  242 (bench/bw roof) / 208.2 = 1.16
k_dflash       :  216 (its own Q4)    / 182   = 1.19
```

**Both kernels are near 1.17, and two bits per trit costs 1.214.** Every wider fixed-length ternary
image starts at or past that line: two bits is 1.214, a nibble 2.4x, one spread byte per trit 4.6x.
What stays open is the direction that buys no bytes at all — cutting the expansion at 1.75 bits per
weight, which [the perm gather](mv-peel-gather.md) already did once, 232 operations a block to 168.

## Where the 15 ms actually is, and 9% of it is now named

The verify pass averages 136.2 GB/s while one of its own phases runs the same body at 208.2. If every
phase ran at the rate the head already demonstrates, the pass would be **26.8 ms instead of 41.0** —
the same 15 ms the coordinate document priced, in the same place, with a different owner.

**Nine per cent of it is the partial last round, and this panel is the first to price it.** A phase
ends when its slowest workgroup does, so it costs `ceil(units / grid)` rounds while doing
`units / grid` rounds of work. At the deployed grid of 100 that quantisation is 0.5% of the head,
1.1% of gate/up and **20% of every 320-unit projection**:

| phase | rounds | ideal | waste | ms lost per pass |
|---|---:|---:|---:|---:|
| `mv_down` | 4 | 3.2 | 20.0% | 1.74 |
| `mv_qkv_z` | 6 | 5.12 | 14.7% | 0.83 |
| `mv_ssm_out` | 4 | 3.2 | 20.0% | 0.52 |
| `mv_qkv` | 5 | 4.16 | 16.8% | 0.29 |
| `mv_o` | 4 | 3.2 | 20.0% | 0.17 |
| `mv_gate_up` | 11 | 10.88 | 1.1% | 0.16 |
| `mv_lm_head` | 78 | 77.6 | 0.5% | 0.01 |
| **total** | | | | **3.71 ms of 41.01, 9.0%** |

**It cannot be fixed by choosing a friendlier grid, and the arm that proves it is in this panel.**
Grid 60 divides the unit counts better — the same table gives 5.4% waste instead of 9.0% — and the
pass costs **51.93 ms against 41.01, 26.6% worse**, because a unit's own time only falls by 12 to
44% while a quarter of the machine's workgroups leave. Nor is there a better grid below 100: the
320-unit phases need 4 rounds at anything in [80, 100], gate/up needs 11 at anything in [99, 100],
and the head needs 78 at 100, so the deployed grid is already the joint minimum of the ladder it has
to serve. The remaining routes to those 3.71 ms are a finer K split on the short phases, which
[the retile arm](drafted-serve-step.md) measured at +31% on the pass and which changes the FP32
association and the served digest, or a unit that can be subdivided in its tail without an
`atomicAdd`, which nothing in this engine currently does.

## What is left, stated as sharply as this panel can state it

**The same unit costs 20.37 us in an eleven-round phase and 17.13 us in a seventy-eight-round one, in
one kernel, at one grid, in one launch, microseconds apart.** Not the coordinate: one instantiation
computes both. Not occupancy or the grid: identical. Not the activation working set: both read the
same 40 kB. Not the tail: that is 1.1% and 0.5% respectively, and it is subtracted above. Not the
phase boundary: `coop_cost --gap-units` bounds a cold restart at 1.3 us, and 257 matvec instances of
that is 0.33 ms over the pass. `bench/coop_cost` reproduces the direction with no model behind it —
141.9 GB/s at one round, 184.3 at eleven, 204.2 at seventy-eight — so it is a property of the unit
loop under the cooperative grid rather than of this model's weights.

That is the whole of the remaining 15 ms, it is the first open question of
[the served step's map](drafted-serve-step.md), and after this panel three more axes are closed
around it.

## What to keep from the document this corrects

Everything except the constant and the arm it priced: `HALO_PROFILE_DRAFT`, the three-coordinate
control inside one kernel, the census (686 instructions for 896 bytes, 594 of them the radix-3 peel),
the device's 1.04 bytes per issued instruction, and the finding that the Q4 and Q8 streams sit at the
memory wall while the ternary one does not. All of it stands, and it is why this correction could be
made from measurements already in custody. What does not stand is reading one kernel's efficiency
factor as a coordinate constant and extrapolating a 15 ms prize from it.

# Four-bit drafter weights, and why approximation is free there

The DFlash2 drafter is 10.7 ms of a 60 ms drafted generation step, and
[its cost is bytes and nothing else](matvec-crossover.md#next-questions-this-leaves): a Q8 tile
block is 4160 bytes with no peel, the body is about five times oversupplied with issue, and the
phase moves 2.16 GB at roughly 190 GB/s of this box's 242 GB/s roof. The only lever in it is fewer
bytes.

Fewer bytes means a coarser weight coordinate, which everywhere else in this engine is a quality
argument. Here it is not one. **A lossless drafter proposes and the target verifies every proposal,
so the drafted greedy stream is identical whatever the drafter computes.** Drafter precision cannot
move a single output bit; it moves only how many drafts survive, and `--bench --dflash` prints that
as an exact integer. This is the cheapest place in the engine to test a representation end to end.

Q4 tiles hold the same 32 rows x 128 K per block as packed nibbles with one fp16 scale per row:
**2112 bytes against 4160**, and the drafter image falls from **1.825 GB to 0.927 GB**.

## Result

Ten prompts spanning the acceptance range this drafter sees (prose, code, arithmetic, proof,
explanation), both arms and both rounds in one process against one loaded model, 3840 generated
tokens per arm, under `tools/run-batch-compare`:

| | Q8 | Q4 |
|---|---:|---:|
| `draft` per step | 11.35 ms | **7.65 ms** |
| `verify` per step | 49.75 ms | 49.80 ms |
| drafted generation, median | 59.34 tok/s | **65.50 tok/s** (+10.4%) |
| accepted, pooled | 2816 / 7224 = 38.98% | **2855 / 7021 = 40.66%** |
| tokens per step, pooled | 3.729 | **3.846** |
| drafted prompt ingestion, 2663 tokens, `wide-deployed` | 401.9 tok/s | **410.1 tok/s** (+2.0%) |
| drafter image on device | 1.825 GB | **0.927 GB** |

Per case, the gain runs from +2.6% to +25.0% and **no case is slower**. The greedy digest is equal
between arms in all ten cases, which is the whole output contract of a drafted run.

Plain single-stream decode is 33.12 tok/s on the same binary, inside this box's range, and shares
no code with the change. Batched generation has no drafter.

Raw samples, prompt set and binaries:
[`batch-comparison/drafter-q4/`](../../../data/bonsai2/batch-comparison/drafter-q4/README.md).

### Installed acceptance

Re-run on the canonical build `a3cbc21` (`599bc0b1b4dd9264`), whose verify pass is 45.2 ms rather
than 49.8 because [the dot4 body's accumulator chains](mv-dot4-body.md) landed in between. Ten
prompts, both arms, one process:

| | Q8 | Q4 |
|---|---:|---:|
| `draft` per step | 11.00 ms | **7.30 ms** |
| `verify` per step | 45.20 ms | 45.35 ms |
| drafted generation, median | 64.31 tok/s | **73.58 tok/s** |
| paired ratio, per case | | **median 1.143**, mean 1.135, range 0.983 to 1.311 |
| accepted, pooled | 699 / 1897 = 36.85% | **713 / 1827 = 39.03%** |
| tokens per step | 3.579 | **3.732** |

The gain is larger here than in the decision panel for a good reason and not a suspicious one: a
shorter verify pass makes the drafter a larger share of the step, so the same 3.7 ms is worth more.
One case of ten is 1.7% slower, and it is an acceptance difference (37 steps against 34), not a
phase difference. Digest equal between arms in all ten cases.

## Acceptance did not pay for the bytes

This is the number the experiment existed to find, and it came out the wrong way round: pooled
acceptance **rose** from 38.98% to 40.66%. Per case:

| case | workload | Q8 accepted | Q4 accepted |
|---|---|---:|---:|
| 0 | prose, two sentences | 36.5% | 35.6% |
| 1 | Python + complexity | 47.9% | 52.4% |
| 2 | arithmetic, step by step | 55.7% | 68.4% |
| 3 | C++ ring buffer | 37.7% | 40.3% |
| 4 | word problem | 56.0% | 56.0% |
| 5 | B-tree vs B+ tree | 24.0% | 24.0% |
| 6 | bash one-liner | 37.7% | 39.8% |
| 7 | irrationality proof | 54.3% | 61.4% |
| 8 | URL walkthrough | 33.6% | 33.0% |
| 9 | Rust refactor | 28.3% | 28.3% |

Three cases are identical - the coarser weights changed no draft at all. Two are slightly worse,
five are better. **Do not read this as "four-bit weights improve a drafter".** The defensible claim
is the one the table supports: over ten workloads spanning 24% to 68% acceptance, a 0.107 relative
RMS perturbation of the drafter's weights cost nothing measurable, and the byte saving is not
bought back at the acceptance end. A drafter picks a top-16 candidate list and then a chain through
it; the perturbation has to cross a ranking boundary to matter, and mostly it does not.

Acceptance is deterministic for a fixed binary - every case reproduced its accepted count exactly
across rounds - but it is **not invariant across schedule changes**, because the K-split segments
(`fc`, `akp`, `mkp` at KS=5, `o` and `down` at KS=2) accumulate across workgroups with float
`atomicAdd`, so a different cooperative grid reorders that sum. Two Q4 builds that differ only in a
crossover constant drafted 119 and 126 tokens on the same prompt. That is pre-existing behaviour of
every K-split matvec here and it is harmless exactly because the drafter is lossless, but it means
**an accepted count is a property of a build, not of a coordinate**, and two drafter builds cannot
be compared on drafts alone.

## The coordinate

`kernels/q4_format.h` owns the layout; `kernels/q4_tiles.hpp` owns the bodies.

    run + row * 64                    64 bytes of packed nibbles, K 0..127 of that row
    run + TILE_ROWS * 64 + row * 2    fp16 scale for that row and block

Nibble placement is chosen so that **unpacking is two instructions per four weights and needs no
permute**. Within a row, K is cut into groups of eight and each group is one dword; the low nibble
of byte `4g+j` is K `8g+j` and the high nibble is K `8g+4+j`. Then

    P & 0x0f0f0f0f          = the four codes of K 8g..8g+3, in byte order
    (P >> 4) & 0x0f0f0f0f   = the four codes of K 8g+4..8g+7

which are exactly the two operand words both matvec bodies want, in the K order the activation row
already has. No byte of the packed block is ever materialised as a weight: the packed form *is* the
operand form, up to an AND and a shift.

**The bias is free.** Codes are offset binary, `c = q + 8`, and the matvec already carries `xsum`,
the int32 sum of a block's activation bytes, because the ternary path needs it for its own `+1`
offset. The Q4 path subtracts `8 * xsum` from the same int32 accumulator, so no weight is ever
unbiased individually and the WMMA `A` fragment is simply declared unsigned - the same flag the
ternary path already passes.

**The unpack is amortised over rows.** At the eight-row block shape the drafter actually runs, one
64-byte load feeds 16 matrix instructions; the expansion is 32 VALU against about 512 cycles of
matrix work, and it is paid once per block rather than once per row.

**It cost fewer registers, not more.** `k_dflash` goes from 247 VGPR and 5 waves per SIMD32 to
**239 and 6**, because the block-lookahead cursor holds `uint4 qa[4]` where Q8 holds `qa[8]` and the
nibble expansion is transient. Halving the bytes bought occupancy as well.

`kernels/q4_pack_check` establishes placement, the operand map and the contraction on the host with
no GPU: every element lands in the nibble the format names, the two AND/shift words are exactly the
codes of K `4m..4m+3` in byte order, and the kernel's int32 minus `8 * xsum` equals the exact
integer contraction of the dequantised weights. All of Q4's error is in the codes; the coordinate
itself adds no arithmetic.

## The scale is not amax/7

Four bits leave the largest element of a block worth 14% of its own magnitude in rounding error,
and a block's amax is a poor place to spend the step: shrinking it clips a few extremes and rounds
everything else finer. The cache build searches sixteen steps between 0.53 and 1.00 of `amax/7` and
keeps the least squared error over the block, scored **through the fp16 scale the kernel will
actually read** rather than an exact-scale fiction. Relative RMS weight error over the whole
drafter is **0.1072**; on a Gaussian control the search is worth about 12% of the error. It is a
one-time host cost and changes no kernel instruction.

## The crossover, measured rather than inherited

`MV_DOT4_MAX_Q8 = 4` was fitted on the Q8 drafter, and copying it to Q4 would have been exactly the
`#ifndef` with no definition that [the matvec crossover](matvec-crossover.md) warns about. So it was
measured, two binaries off one tree, same three prompts, the Q8 arm carried in both processes as
the cross-process control:

| Q4 body at eight rows | `k_dflash` | waves/SIMD32 | `draft` ms | paired Q4/Q8 |
|---|---:|---:|---:|---:|
| WMMA (`MV_DOT4_MAX_Q4 = 4`) | 239 VGPR | 6 | **7.40** | **1.1915** |
| dot4 (`MV_DOT4_MAX_Q4 = 8`) | 145 VGPR | 9 | 8.10 | 1.1406 |

**Fifty percent more resident waves and it is 9% slower.** This is the same shape as the ternary
crossover from the other side: there, dot4 won at eight rows *because* of registers, taking the
deployed grid from 60 workgroups to 100. Here the WMMA arm is already at six waves, the phase is
byte-bound rather than latency-bound at that occupancy, and the extra waves buy nothing while the
dot4 body's worse bytes-per-issue costs 0.7 ms. Occupancy is worth what the phase is short of, and
this one is short of bytes. `MV_DOT4_MAX_Q4` stays at 4, now on a measurement.

## What this leaves

- **The drafter is now 7.65 ms of a 57 ms step, and 1.23 GB of it is still traffic.** 278 MB of
  that is the target's ternary LM head, which the drafter runs at full vocabulary for seven rows to
  take a top-16 per row. Nothing has asked whether a drafter needs all 248,320 columns.
- **The Q4 arm moves its bytes at 161 GB/s where Q8 moved them at 190.** Halving a byte-bound
  phase's traffic did not halve its time; the phase is drifting toward latency-bound and there is
  perhaps 1.5 ms in closing that, which is where a deeper block cursor would be priced.
- **Occupancy is not one of them, and that is now measured directly.** The Q4 body runs at six
  waves per SIMD32 and three workgroups per WGP; forcing four changes the drafted step by -0.2%,
  and the drafter's time is flat from 40 workgroups upward. [`drafter-occupancy.md`](drafter-occupancy.md)
  has the curve, and the grid calculator repair it took to launch the endpoint at all.
- **The same lever is untouched on the MTP head**, whose weights are still Q8 and which shares this
  quantiser and cache.
- **Q4 is not the floor.** Three bits with a per-32 sub-scale is the same trick one step further,
  and the instrument to judge it already exists: the prompt set and the accepted count.
- **The generalisation is the point.** Any representation idea that cannot yet afford a quality
  defence on the target model can be built on the drafter first, where the output contract is
  untouched by construction and the verdict is an exact integer.

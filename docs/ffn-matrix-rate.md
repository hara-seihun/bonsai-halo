# A matrix instruction costs 34 cycles, and the slots beside it are free

Two constants decide what every block census in this repository is worth, and both were wrong in the
same direction. `v_wmma_i32_16x16x16_iu8` costs **34.3 cycles per SIMD32**, not the 21.9 issue slots
[`bench/wmma_cost`](../bench/wmma_cost.hip) reports and not the 17.1
[`bench/wmma_chain`](../bench/wmma_chain.hip) reported when it landed. And the non-matrix
instructions in the same loop are **free to first order**: fourteen independent VALU operations per
matrix instruction - the deployed FFN block's own ratio - cost 0.5%.

Those two facts together explain every null this lane has measured in `k_ffn_slice`. The peel gather
took 85 slots out for -1.66%. The fourth column group cut 7.6% of slots per row for +0.2%. The
request reorder was +0.55%. [The straightened activation operand](ffn-slice-operand.md) removed 87%
of the block's line requests and *lost* 4%. They were all trading instructions that the matrix pipe
was already hiding.

Raw samples for both probes, in one panel and on one clock, are in
[`batch-comparison/ffn-matrix-rate/`](../../../data/bonsai2/batch-comparison/ffn-matrix-rate/README.md).
**Nothing in the runtime changed**: this iteration ships `bench/mvblock`, a one-line correction to
`bench/wmma_chain`'s SIMD count, and a `-DHALO_SLICE_CYCLES=1` diagnostic inside `ph_matvec_w`.

## Two probes disagreed by 2x, and the machine is 80 SIMD32

`bench/wmma_chain` landed with `simds = multiProcessorCount * 2`, "two SIMD32 per reported CU".
`multiProcessorCount` returns **20** on this part, which is its WGP count, not its 40 CUs, so that
line reported half the machine and halved every constant it printed. The lane's own arithmetic says
80 everywhere else: `coop_grid` multiplies per-WGP workgroups by `multiProcessorCount`,
`bench/wmma_cost` uses `* 4`, the deployed FFN slice's 60 workgroups are 480 waves at the six per
SIMD32 the register file allows, and 480/6 = 80. The count is corrected in place, with the reason
in the source.

With that fixed both probes were run in one lock hold, on one clock:

| | `iu8`, 1 / 2 / 4 / 8 accumulator chains |
|---|---|
| `wmma_chain`, grid 60, cycles at an **assumed** 2900 MHz, loud box | 39.9 / 45.4 / 45.4 / 46.5 |
| `mvblock`, six waves, cycles at the clock **it read for itself**, same panel | 36.4 / 34.9 / 35.8 / 36.0 |
| `wmma_chain` again, grid 60, on a box quiet enough to hold 2900 | **34.30 / 34.25 / 34.23 / 34.23** |

The last row is the two probes closing on each other: an assumed clock is right when the part is
holding it, and `mvblock` reads 34.3-34.9 whether it is or not.

The remaining gap is the clock. This panel ran with sixteen host cores busy and 81 W on the CPU
rail; `mvblock`'s in-kernel read saw 2961 MHz falling to 2199 MHz across the arms, and 45.4 cycles
at a nominal 2900 is 34.4 at 2200. **A probe that assumes its frequency reports the socket, not the
instruction** - which is [the power budget's](power-budget.md) result applied to a probe. `mvblock`'s
own elapsed time moved 39% across that panel while its cycles per matrix instruction moved 9%.

34.3 cycles is this part's hardware rate: 512 int8 ops per clock per CU is 256 MACs per SIMD32, and
a 16x16x16 instruction is 4096 MACs, so the peak is 32.2 cycles. **The matrix instruction is at 94%
of peak and there is nothing to schedule for.** Chain depth is flat, which is the result
`wmma_chain` already published and this reproduces: the accumulator form pipelines, and splitting
`C1`/`C2` into more chains - bit-identical, cheap, and the obvious first arm - buys nothing.

## What the slots beside it cost

`mvblock` adds independent VALU work to the same chain at the ratio the deployed block runs, 444
non-matrix instructions against 32 matrix ones:

| arm, six waves per SIMD32 | cycles per matrix instruction |
|---|---:|
| two chains, nothing else | 34.85 |
| **two chains + fourteen VALU per matrix instruction** | **34.40** |
| four chains + fourteen VALU | 34.36 |
| two chains, A operand written by `v_perm` and `v_permlanex16` one slot earlier | 37.51 |
| VALU only, no matrix | 1.32 cycles per VALU operation |

Filler is free; the operand hazard costs **9%**, about three cycles, and it is the one
census-shaped term left in that block. The VALU line also settles where `wmma_cost`'s 21.9 came
from: its calibrator asserts that `v_fmac_f32` is one cycle per SIMD32, and it is **1.32** here
(`wmma_chain`'s dependent control is 1.5 to 2.2). 34.3 / 1.57 = 21.9. **A "slot" in this
repository's censuses is about 1.5 cycles, and the matrix constant was 1.5x low.**

## Reconverting the phase this came from

[The FFN slice's unit boundary](ffn-slice-activation.md) is the result these constants were measured
for, and they do not move its conclusion - they sharpen it. That panel priced a 128-row mode-20
pass's `ffn` phase as 106.5 ms of counted issue slots, 14.5 ms of weight round trip and 59.7 ms of
neither, then found the 59.7 in the unit boundary. At 34.3 cycles per matrix instruction and zero
for the slots beside it, the same block is **1098 cycles of matrix and nothing else**, against 1145
"slots" - the total was accidentally right and its composition was not:

| | slots model | measured constants |
|---|---:|---:|
| matrix | 701 slots, 61% of the block | **1098 cycles, ~100%** |
| peel, swap, drain, addressing (444 slots) | 39% of the block | **~0** |

A pass issues 535M matrix instructions, so the phase's floor is **78 ms at 2.9 GHz and 101 at
2.25**, against 180.7 measured. Everything between those two numbers is the unit boundary, the
memory the probe does not have, and the clock - and none of it is the instruction stream.

An independent reading of the same phase agrees. `-DHALO_SLICE_CYCLES=1` makes `ph_matvec_w` read
`SHADER_CYCLES` around its own block loop and around the whole unit and print for one wave of one
workgroup; on the drain this lane shipped that morning it reported a unit tail of 14,042 cycles
against the 47,085 of the five-block gate/up loop it drains, and 44k-64k against 73k on the K-split
down projection - 23% and 38-47% of a unit, **32% of the phase**, which is the same third the arm
panel found by difference. The instrument is in the tree because the next question needs it: with
the flat drain shipped, the down projection's per-unit cost fell 14.4% while gate/up's did not move,
and the difference between them is that down's stores are `atomicAdd` and gate/up's are not.

## What to do with this

1. **Convert `iu8` at 34.3 cycles and a VALU slot at 1.3**, and only when the VALU is not running
   beside matrix work - when it is, price it at zero and check.
2. **Do not cut instructions out of `k_ffn_slice`'s block loop.** Five arms have now tried; the
   probe says why. The operand hazard is the only instruction-shaped term left and it is 9% of the
   matrix pipe, not of the phase.
3. **The gate/up unit tail is the open number**: 12,000 cycles a unit, 21% of that projection, and
   it did not respond to distributing the drain. It is the barriers and the skew they expose, plus a
   store whose lanes are `N` floats apart. `-DHALO_SLICE_CYCLES=1` prices any arm against it
   directly, without a clock in the comparison.
4. **`next_unit` is in none of these numbers.** It runs between units, outside both brackets: a
   `__syncthreads()`, a device-scope `atomicAdd` on one address contended by 60 workgroups, and
   another `__syncthreads()`. A launch has about 290k cycles beside the 1.50M the instrument
   accounts for. Claiming the next unit before the block loop instead of after it hides that latency
   behind 47,000 cycles of work and keeps the dynamic balance.

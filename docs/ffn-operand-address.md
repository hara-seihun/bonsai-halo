# The FFN projection's operand addresses

`k_proj_opt` is the A4 FFN's whole inner loop and the largest single phase of a prefill pass. This
document covers what it costs to *reach* its operands, which turned out to be about an eighth of
the instructions in its block loop, and none of which is arithmetic.

Three facts about the compiled loop drove the change. All three are visible without a GPU, through
`tools/isa_loop_count.py` on a `hipcc -S` listing.

1. **Every operand load used a 64-bit VGPR address pair.** `global_load_b64 v[..], v[58:59], off`
   is the expensive addressing form: the wave builds a full 64-bit address per load with a
   `v_add_co_u32`/`v_add_co_ci_u32_e64` pair, and holds two VGPRs per live address. The cheap form,
   `global_load_b64 v[..], v74, s[34:35] offset:128`, keeps the base in scalar registers and pays
   one 32-bit lane offset for the whole wave.
2. **The activation scale array was stored token-major.** The drain read
   `cscale[(tg + t*16 + col) * nb + blk]`, and `col` is the lane. Sixteen distinct lanes therefore
   read addresses `nb * 4` bytes apart — 160 bytes on gate/up, 544 on down — so one instruction
   asked for sixteen separate 64-byte lines. A 128-row A4 pass issues 33.4M of those loads.
3. **The gate/up shape spilled.** `k_proj_opt<IU4, PAIR, LANE, SHARE, TT=4, WV=2, FUSE, ORD=2>`,
   the shape that carries gate and up at 128 rows, allocated all 256 VGPRs and spilled 52 bytes to
   scratch, which is five waves per SIMD32.

The three are the same fact. The kernel indexed its operands with `size_t` expressions that mixed
the wave-uniform part (block, slice, row tile, token group) with the lane (`col`), so the
uniformity analysis could not put anything in a scalar register, and the resulting live 64-bit
addresses were what the allocator could not fit.

## What changed

Nothing in the arithmetic, the accumulation order, the scales, the residual add or the output
layout. Three address maps:

- **Operand cursors.** Each block computes `bblk`, `sab`, `sbb` and `csb`: wave-uniform base
  pointers for the B fragments, the two weight-scale runs and the activation scales. Loads index
  them with an `unsigned` lane offset, which is what the `GlobalSAddr` form needs. The compiler
  then emits one scalar base per K16 slice, one VGPR offset (`col`) for the whole block, and the
  token tile as a load immediate.
- **`wave` is stated to be wave-uniform.** `threadIdx.x >> 5` is constant across a wave, but the
  analysis only knows that `threadIdx.x` is divergent, so the row tile and the token group
  inherited its divergence. `__builtin_amdgcn_readfirstlane` costs one instruction outside the
  loop. On its own it changes nothing — the `size_t` index still forces the 64-bit add — which is
  why the two halves had to land together.
- **`cscale` is block-major.** `k_prep` writes `cscale[block * csstride + token]` and the drain
  reads sixteen consecutive tokens of one block: one aligned 64-byte line per load, and the four
  token tiles of a `TT = 4` shape become four immediates off one address. The stride is fixed at
  `npad_max + ROW_SLACK` for the module's lifetime, so a narrow pass addresses the same slots a
  wide one does. Same values, same order, same allocation size.

## Static effect, per 128-K block of one wave

`tools/isa_loop_count.py`, counting the block loop of each selected instantiation.

| shape | where it runs | before | after |
|---|---|---:|---:|
| `<IU4, PAIR, LANE, SHARE, 4, 2, FUSE, ORD=2>` | gate/up at 128 rows | 559 | **487** (-12.9%) |
| `<IU4, PAIR, LANE, SHARE, 2, 4, ADD, ORD=0>` | down at 128 rows | 389 | **325** (-16.5%) |
| `<IU4, DENSE, LANE, TILE, 2, 4, FUSE, ORD=0>` | the <=32-row arm, every generation step | 648 | **587** (-9.4%) |

What left the gate/up block: 26 of its 38 `v_add` (the 64-bit address pairs), 16 of its 63 SALU,
and the whole 52-byte spill. The 64 matrix instructions, 64 conversions, 32 `v_perm` of operand
expansion and 16 `ds_bpermute` of scale broadcast are untouched, which is the check that this is an
addressing change and not an arithmetic one.

Registers, from `tools/kernel_resources.py`:

| instantiation | VGPR | waves/SIMD32 | scratch |
|---|---:|---:|---:|
| gate/up `TT=4, ORD=2` | 256 -> **229** | 5 -> **6** | 52 -> **0** |
| down `TT=2` | 187 -> 180 | 8 -> 8 | 0 |
| dense `TT=2` | 213 -> 200 | 7 -> 7 | 0 |

47 of 150 instantiations gain registers or lose a spill. Three lose a wave slot: the scaled-FP16
`<F16, TWO, SLICE, *, 4, *>` arms go 134 -> 149 VGPR and ten waves to nine, while their block loops
get slightly shorter (236 -> 233). Those are mode 3/7 control routes, not a serving path, and this
device has repeatedly measured wave slots as a non-binding resource in this kernel
([`ffn-order.md`](ffn-order.md), [`ffn-occupancy.md`](ffn-occupancy.md)).

## Measured

Source `3abc5d1` against its own parent `035cbfe`, both built in one checkout from the same
toolchain so the only difference is this file. Raw: `batch-comparison/ffn-addr/`.

**The FFN phase of a traced 128-row pass, normalised on the phases this change cannot touch.**
Cross-process panels on this box drift several percent, so the statistic is the ratio of the `ffn`
phase to the sum of the other seven phases of the same run. That ratio is stable to 0.2% inside
each process and moves 5.6% between builds:

| build | ffn ms | untouched ms | ffn / untouched |
|---|---:|---:|---:|
| `035cbfe` round 0 | 76.706 | 102.173 | 0.7507 |
| `035cbfe` round 1 | 72.063 | 95.964 | 0.7509 |
| selected round 0 | 76.065 | 107.190 | **0.7096** |
| selected round 1 | 76.759 | 108.335 | **0.7085** |

**-5.56% of the FFN phase.** The untouched phases ran 5-7% slower in the second process, which is
why the absolute `ffn` column looks flat; that is the drift the ratio removes.

**Full model, 384-token document, mode 20 `wide-deployed` interleaved in the same process as the
control.** Mode 20 runs the identical wide schedule with the deployed eight-row FFN, so it executes
none of this kernel:

| workload | `035cbfe` | selected |
|---|---:|---:|
| prefill 128 rows/pass, mode 19 | 771.3, 773.8 tok/s | **808.5, 794.3** |
| prefill 32 rows/pass, mode 19 | 523.4, 518.5 | 518.9, 530.1 |
| mode 20 control, 128 rows | 268.9, 268.5 | 268.8, 247.1 |
| 32-stream generation, mode 19 | 280.2, 279.9 | 277.6, 279.7 |

**+3.7% on 128-row prompt medians**, every sample above every base sample. Three of the four control
samples agree to 0.15%; the fourth is 8% low, so the box degraded during the last round of the
treatment panel and the +3.7% is the conservative reading. The 32-row prefill and the 32-stream
generation step are both nulls.

The generation null is the informative one. The `<=32-row` dense arm every generation step runs
loses 9.4% of its block instructions here and gives back nothing, which is the same thing
[`ffn-pair-lookahead.md`](ffn-pair-lookahead.md) measured from the other side: below 32 rows this
kernel runs at about a third of its issue model and is waiting on operands, not on issue. Cutting
instructions converts to time only in the 128-row shape.

**Acceptance.** 71,516,160 full-vocabulary logits at 32, 40, 88 and 128 prefill rows, byte-compared
against the canonical dump in `batch-comparison/gdn-gate-rule-installed/20260921-135830/`, zero
differing bits; that reference is itself identical to the one taken an hour earlier on a different
build, so it is the numerical map's invariant rather than one run's. Residual FNV-1a
`7446865760224376151` in all four 128-row profile samples.

## The permutation on its own is a null, and that is the more useful half

`f5054351` built the cscale transpose independently within the same hour and measured it alone,
with both layouts interleaved in one process through an `--ffn-cscale` axis:

| shape | token-major | block-major | normalised |
|---|---:|---:|---:|
| 32 rows, `ffn` | 33.93, 31.88 | 30.85, 32.14 | +1.4% |
| 128 rows, `ffn` | 80.00, 72.84 | 81.13, 76.33 | -0.4% |

Opposite signs, both inside a panel whose own control phases disagree by 2.8 points. Their residual
FNV-1a matched at both shapes, so the permutation is bit-exact on device and not merely by
construction.

So a **32x cut in cache-line requests** - 534M lines per 128-row pass down to about 17M - buys
nothing measurable. Two things follow. The L0/L1 line-request path is not a bottleneck in this
kernel at any width, which retires "uncoalesced scale gather" as a candidate for the FFN's
unexplained 30 ms. And the transpose earns its place here only as the *enabler*: a token-major
scale array cannot be addressed from a wave-uniform base, because the lane is the axis carrying the
`nb` stride, so without it the cscale load keeps its 64-bit VGPR address pair and its four token
tiles cannot collapse onto one base with immediate offsets.

## What this does not touch

`expand_pair` is five operations per fragment — two masks, a shift and two `v_perm_b32` against
the two constant selector registers — and the only way to reach four is a nibble-per-weight image,
which doubles the 4.28 GB weight stream. [`wide-head.md`](wide-head.md) priced that same trade in
the head and it lost. The 64 `v_cvt_f32_i32` of the drain stay: see below.

## The magic-constant conversion, priced and rejected

The drain's 64 `v_cvt_f32_i32` per block have no VOPD form and are 13% of the block, so the
standard bit trick keeps coming up: `as_float(bits(C) + acc) = C + acc` for `C = 1.5 * 2^23`, then
`f += (m - C) * s`, and the `-C * s` term is folded into a running sum of scales.

It is not an instruction-count question, it is a precision one. `ulp(C) = 1` forces `C` near 8.4e6,
while the true per-block term `acc * s` is order 1: the accumulated `g` and its correction
`C * sum(s)` are both order 1e5-1e6 and cancel to a value of order 10, which costs about four
decimal digits of an FP32 mantissa that only has seven. No choice of `C` avoids this, because the
trick needs the fixed exponent that makes the mantissa the integer. Prior notes recorded that the
correction costs the FMAs it saves; the precision argument is the stronger reason and closes it.

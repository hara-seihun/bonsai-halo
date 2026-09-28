# The deployed matvec takes the gather, and the ablation that says what its wall is not

`mv_rows_t` is the body of every matvec the persistent kernel runs: single-token decode, eight-row
prompt ingestion, and the eight-row verify pass that is
[87% of a drafted step](drafted-serve-step.md) — what `bonsai-halo.service` generates on. It was the
last consumer of the five-trit peel after [the peel gather](mv-peel-gather.md) landed in the FFN
slice and the head took its own copy (0dad0d38 holds `head_tile`; the header is shared).

`bdf1e3f` put it on `hx_expand_perm` from [`kernels/halo_expand.hpp`](../kernels/halo_expand.hpp) —
**168 operations per 128-weight block against 232**, the same 32 operand dwords out of the same 26
stored bytes — and shipped it "for the single map rather than for the milliseconds" against a null
in one pair. **This document is what the map is worth, measured on the kernel itself**: **3.0-3.8%
of the isolated eight-row body**, **+1.8% and +2.2%** on single-token decode, and a small,
order-1% improvement on the drafted verify pass that this lane's full-model instrument cannot
separate from its own drift. `mv_rows_pre`, the `--v1` per-op path, was a third copy of the peel
and takes the header here.

**The ablation beside it is the part worth keeping, and it is why the arm is a defaulted template
parameter rather than a plain `#if`**: one process holds both arms plus a third that is not an arm
at all — it deletes the expansion and computes a wrong answer on purpose. It buys 12.2-13.2%. So the whole decode of five trits out
of a byte — every multiply, mask, shift and gather this engine has argued about for two days — is
an eighth of this kernel, and **removing all of it still leaves the eight-row body at 150 GB/s
where the same image, the same loads and the same block order at one row read 226.**

## The panels

`bench/mvsched`, real 37.2 MB weight image, palindrome order, one process, one pinned clock, ten
resident waves per SIMD32 (the LDS ballast `k_forward_rows<8>` gets). Two panels, different builds
of the same source, means of the forward and reverse samples:

| arm | slots/block | TT = 8 | | TT = 1 | |
|---|---:|---:|---:|---:|---:|
| | | panel 1 | panel 2 | panel 1 | panel 2 |
| peel (the deployed expansion) | 751 | 308.6 us | 295.9 | 178.1 | 172.9 |
| **perm gather (selected)** | **687** | **296.9** | **287.0** | **176.7** | **171.7** |
| no expansion at all (ablation) | 519 | 267.8 | 259.7 | 175.6 | 172.5 |
| | | **-3.79% / -13.22%** | **-3.01% / -12.24%** | -0.8% / -1.4% | -0.7% / -0.2% |

**Both arms hash to `07701590705348488800`** at both row counts in both panels; the ablation hashes
to `14489737127402071171`, which is what an arm that computes the wrong thing is supposed to do.

**The eight-row body is issue-proportional in this range, and that is a number you can spend.**
64 slots bought 3.0-3.8% and 232 slots bought 12.2-13.2%: **0.053% and 0.055% of the phase per
issue slot per block.** A change to this loop can be priced before it is built.

**One row is flat across all three arms** because it is already at 224-228 GB/s of the 242 GB/s
`bench/bw` measures. [The dot4 body](mv-dot4-body.md) said this two days ago and it is still true:
the single-token kernel has 232 slots of expansion to interleave with 32 dot products and no
schedule change has ever moved it.

## What the ablation rules out

At eight rows the body reads 896 weight bytes per block and issues 751 slots to consume them. Delete
the 232 slots of trit decoding and it issues 519 — and it gets **150 GB/s against the 226 the same
bytes reach at one row, and the 242 the device delivers**. The expansion is not what this pass waits
for. Neither is the block count ([the slice-width panel](ffn-slice-width.md)), the image geometry
([`bench/wstream`](weight-stream-order.md) gives this geometry 202 GB/s with no model), the
operand's cache lines ([the fragment coordinate lost 4%](ffn-slice-operand.md)) or the scalar-wait
depth ([two blocks in flight is a null, three costs 2.9%](mv-scalar-waits.md)).

What is left, in the order this panel ranks them:

1. **The activation path, which wants a different kernel rather than a different budget.**
   `bench/mvsched`'s `vec-ar2` arm — the rows' activations on `vmcnt` instead of `lgkmcnt`, which is
   the counter that can be waited part way — measures **134-136 GB/s at ten waves**, the best any
   arm has reached on this body, against 129-131 for the deployed schedule. It is in the `rejected`
   namespace because at `HALO_ROWS_WPE = 12` it spilled 370 dwords, and the obvious next thought is
   that the deployed launcher now selects `k_forward_rows<8, 0, 0, 0, false, 1>` at 141 VGPR and ten
   waves with no spill, so the arm deserves another look at that budget. **It does not.** The
   listing in this panel's directory puts `kmv<8, ..., MVS = 2>` at **128 VGPR against 62** for the
   deployed body — +66 for the two rows it keeps in flight — and 141 sits inside a 144 granule. The
   arm costs three wave slots wherever it lands inside that kernel. What it is really asking is
   whether this phase should be **its own kernel**, where 128 registers is still ten waves; that is
   the same question [the verify pass map](drafted-serve-step.md) asks from the other side with a
   2.7 us unit boundary and a 1.8x spread across identical bodies.
2. **The unit boundary.** [The verify pass map](drafted-serve-step.md) prices a unit at 2.7 us of
   workgroup time and finds `mv_gate_up` 26% off its own probe cell at 1088 units where
   `mv_lm_head` matches its cell exactly at 7760. That is 3.8 ms of a 45 ms pass and it is not in
   this body at all.

## Full model

`--bench --dflash` with the DFlash2 q4 drafter the service runs, a 932-token prompt, 48 greedy
tokens, 12 drafted steps. Both arms are the same source; the control arm is built with
`HALO_MV_EXPAND=0` and is [canonical's body instruction for instruction](#the-control-arm-is-the-deployed-body).
Five pairs, three of them with `HALO_PROFILE=1` so the phases the change cannot reach — `gdn`,
`attn`, `prep_*`, `argmax` — normalise the machine drift between the two halves of a pair.

| pair | control drift | matvec phases, normalised | verify ms/step | drafted tok/s |
|---|---:|---:|---:|---:|
| 1 | x0.985 | **+0.07%** | 47.278 -> 46.088, **-2.52%** | 72.98 -> 74.63 |
| 2 | x1.008 | **-1.32%** | 46.171 -> 45.961, -0.45% | 74.60 -> 74.82 |
| 3 | x1.034 | **-2.80%** | 47.638 -> 48.305, +1.40% | 72.36 -> 71.24 |
| 4 (no tracer) | | | 44.562 -> 44.260, -0.68% | |
| 5 (no tracer) | | | 44.426 -> 45.188, +1.72% | |

Median of the five verify deltas **-0.45%**, median of the three normalised matvec deltas
**-1.32%**. **The instrument's own control moves 1.5-3.4% between the halves of a pair**, so this
panel bounds the change rather than resolving it: it is not a regression and it is smaller than the
isolated kernel's 3.0-3.8%. That gap is the expected one —
[the peel gather in the FFN slice](mv-peel-gather.md) cut 15% of its block loop for 1.66% of its
phase — and the reason is in the ablation above: a phase whose slots are 64% of its issued cycles
returns a fraction of what its census says.

Single-token decode, the shape with no drafter, same prompt, 40 tokens, two rounds each:

| | round 1 | round 2 |
|---|---:|---:|
| peel | 31.17 tok/s | 31.39 |
| **perm gather** | **31.72** | **32.08** |
| | +1.8% | +2.2% |

That shape is the one where the census promised the most (232 of 345 work slots) and the bandwidth
roof promised nothing, and it is the one where the full-model panel is cleanest.

## Bit identity

Nothing about the values moves: the same 26 bytes produce the same 32 operand dwords in the same
byte lanes, feeding the same `v_dot4_i32_iu8` in the same order into the same `int32` accumulators
and the same `fmaf`.

- **The map is proved on the CPU.** `make kernels/head_op_check && ./kernels/head_op_check` runs
  the peel, the gather and the spread coordinate against `halo::decode_block` over 3 uniform, 256
  byte-sweep and 4000 random blocks in a second, with no GPU. (From
  [the peel gather](mv-peel-gather.md); it is the acceptance for this change too.)
- **The kernel digests agree.** `07701590705348488800` for peel and perm at TT=8 and TT=1 in both
  panels of `bench/mvsched`.
- **The product's own output agrees.** Greedy digest `1494572377574161652` in all ten drafted runs
  above, with 84 drafted and 37 accepted (44.0%) and 4.08 tokens/step in every one. A matvec that
  changed a logit bit would change which drafted tokens are accepted.

### The control arm is the deployed body

`HALO_MV_EXPAND=0` is not an approximation of the old code, it is the old code: the expansion block
of `kmv<8, 5, 1, 1, 1, 26000, 2>` compiled from canonical is **269 instructions, 263 work**, and
from this tree with `EXP = 0` it is **269 / 263**; at one row both are **290 / 284**. With the
gather it is **205 / 199** and **228 / 219**. (`tools/isa_loop_count.py` on a
`hipcc -S --cuda-device-only` listing, no GPU.) 0dad0d38 lost 89 work instructions of their control
arm to a traits struct in the head and warned the lane; this body's control was checked against
canonical before the delta was believed.

## Registers

`tools/kernel_resources.py kernels/halo_rows.hip -D HALO_ROWS_TRIM=1`, the deployed
instantiations:

| kernel | peel | perm gather |
|---|---|---|
| `k_forward_rows<8, 0, 0, 0, false, 1>` (the deployed verify pass) | 141 VGPR, 10 waves | **141, 10** |
| `k_forward_rows<8, 0, 0, 1, true, 1>` | 142, 10 | 143, 10 |
| `k_forward_rows<8, 0, 0, 0, true, 1>` | 153, 9 | 153, 9 |
| `k_forward_rows<1, 0, 0, 0, true, 1>` | 144, 10 | **145, 9** |

One instantiation crosses a granule boundary for one register — the gather's selector is a constant
the compiler materialises — and loses a wave slot. **It measured faster anyway** (the single-token
table above), which is what a body at 98% of its bandwidth roof does with a wave slot. Reordering
the gather to hold four products live instead of ten does not recover the register; that arm was
built, measured at 145/9, and deleted rather than left in the tree.

## Operation

```sh
make bench/mvsched
tools/run-batch-compare --pin-clock --exec ./bench/mvsched --reps 24      # the axis, seconds
make DEFS='-DHALO_MV_EXPAND=0' bonsai-halo                                # the control build
```

`HALO_MV_EXPAND=0` restores the peel in every `mv_rows_t` shape and is the arm the panels above
call `peel`. The ablation arm is compiled only by `bench/mvsched`, which passes
`-DHALO_MV_EXPAND_PROBE`; no runtime translation unit defines it.

Raw samples, both binaries, the census listings and the pair files are in
[`batch-comparison/mv-peel-rows/`](../../../data/bonsai2/batch-comparison/mv-peel-rows/README.md).

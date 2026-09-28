# What is alive inside one 128-block of the A4 gate/up projection

`k_proj_opt` walks a weight block slice-outer: expand one K16 slice of the A operand, spend it
immediately against every token tile the wave holds, repeat for eight slices, then drain. That
keeps all `TT` int32 accumulator pairs live for the whole block. At the gate/up shape — four token
tiles, two waves sharing a row tile — those accumulators are 64 of the kernel's 252 VGPRs, and 252
VGPRs is five waves per SIMD32.

`ORD` is the interchange. Expand the block's sixteen fragments once into 32 registers, then run
`ORD` token tiles at a time and drain each group. Every output element still accumulates slices
0..7 of a block in that order, and still drains once per block into the same `ya`/`yb` FP32 chain.
Nothing is reassociated, so this is the deployed arithmetic element for element.

`HALO_FFN_A4_ORDER` and `ffn_batch_set_a4_order()` select it; `--ffn-order` is a case axis in both
measurement tools. **`ORD = 2` is selected.** The down projection keeps the slice-outer order, and
[the block census](ffn-drain.md#the-change-this-built-and-why-it-measured-nothing) says it should:
at the ownership `shape_for` gives it, its drain is already paired into VOPD and giving it `ORD = 2`
measures a null with the arms interleaved in one process.

## What the compiler does with it

gate/up, `k_proj_opt<IU4, PAIR, LANE, SHARE, TT=4, WV=2, FUSE>`, from
`-Rpass-analysis=kernel-resource-usage` and a census of the 128-block loop:

| `ORD` | tiles per group | matrix chains | VGPR | waves/SIMD32 | scratch | block-loop slots |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | — (slice-outer) | 8 | 252 | 5 | 0 | 508 |
| 1 | 1 | 2 | 209 | **7** | 0 | 510 |
| **2** | **2** | **4** | 256 | 5 | 52 | **489** |
| 4 | 4 | 8 | 256 | 5 | 52 | not measured on device |

A slot is one issue cycle: a `v_dual` pair counts once. The 52 bytes of scratch at `ORD >= 2` are
**not in the block loop** — that loop has zero `scratch_*` ops. They sit in the enclosing
token-group loop, which runs once per row tile against the block loop's forty iterations.

Where the 19 slots go, per block: waits fall from 65 to 39 and scalar ops from 69 to 60, while VALU
slots rise from 252 to 268. Hoisting the expansion above the matrix work removes most of the
`s_waitcnt`/`s_delay_alu` the slice-outer order needs between a slice's expansion and its first
`v_wmma`, and costs a few VALU slots for the wider fragment file.

The down projection, `<IU4, PAIR, LANE, SHARE, TT=2, WV=4, !FUSE>`, goes the wrong way: 147 VGPRs
and nine waves per SIMD32 become 158 and still nine. It is left alone.

## Measured

One build, `ffn_batch.hip` at this commit, every arm selected inside one process by `--ffn-order`
and walked in a reshuffled order each round, so the arms share an executable and a clock.
**Bit-identical throughout**: the FNV-1a over all 128 x 5120 FP32 residual outputs is
`7446865760224376151` in every arm of every panel, the same value the row-tile ownership change
recorded.

Mode 19, 128 rows, no head, four rounds, whole pass:

| arm | median pass | against slice-outer |
|---|---:|---:|
| `ORD = 0`, slice-outer | 186.603 ms | — |
| `ORD = 1`, one tile per group | 199.209 ms | **+6.8%** |
| `ORD = 2`, two tiles per group | **183.535 ms** | **-1.6%** |

Same shape with device traces, six rounds, the `ffn` region only:

| arm | median `ffn` | min | rounds favourable |
|---|---:|---:|---:|
| `ORD = 0` | 86.934 ms | 85.022 | — |
| `ORD = 2` | **82.940 ms** | **81.909** | 5 of 6 |

Full model, 384-token document in 128-row passes with the head on the last pass, four rounds, the
two arms interleaved in one process:

| round | `ORD = 0` | `ORD = 2` |
|---:|---:|---:|
| 0 | 690.5 tok/s | 702.9 |
| 1 | 656.7 | 696.8 |
| 2 | 652.6 | 658.2 |
| 3 | 652.2 | 635.5 |

Mean of the four paired differences: **+1.55% prompt throughput**. Across every paired prefill
round of every panel here, 14 of 17 favour `ORD = 2`; a sign test on that is p = 0.013 two-sided.

**Read the round-to-round spread before quoting a single number.** The box carried a dozen
concurrent lane engineers and a qemu guest during these panels, and absolute throughput drifted
about 8% downward across panel C's four rounds. That is why every arm here is interleaved inside
one process and why the paired differences, not the levels, carry the result.

### Nulls, by construction and measured

At 32 rows and below, mode 8 takes the dense five-trit arm, and `launch_opt_one` only offers the
interchanged order to `OP_IU4` with `WM_PAIR`. So no shape at or below 32 rows can reach it.

| workload | `ORD = 0` | `ORD = 2` |
|---|---:|---:|
| 32-row prefill pass, head on | 74.141 ms | 74.193 ms |
| 32-stream generation, aggregate | 265.05 tok/s | 265.25 tok/s |

This is a prompt-processing change, like the row-tile ownership before it.

### Through the head

[39,731,200 full-vocabulary logits](../tools/batch-compare/results/ffn-order-identity.json) at 32
and 128 prefill rows, `ORD = 0` against `ORD = 2`, **zero differing bits and zero differing rows**.
One executable, two processes whose recorded `HALO_*` route flags differ, so
`sequence_layout_compare.py` accepted the pairing rather than comparing a build against itself.

Raw samples, per-round values and the `halo_env` each process saw are under
[`../../data/bonsai2/batch-comparison/ffn-order`](../../../data/bonsai2/batch-comparison/ffn-order),
with the identity pair in `ffn-order-ident-slice` and `ffn-order-ident-group2`.

## The bounded negative, and why it is the more useful half

`ORD = 1` is the shape the occupancy argument predicts you should want: 209 VGPRs, seven waves per
SIMD32 against five, no spill, the same 510 issue slots. It is **6.8% slower**.

So, for this kernel, on this device, in this instruction grammar: **wave slots are not the binding
resource, independent matrix chains are.** Five waves each holding eight independent accumulator
chains beat seven waves holding two, by more than the 40% occupancy gain was ever going to be
worth. The gate/up stage measures 1.53x its WMMA issue budget and that gap is real, but it is not
a latency-hiding gap that more resident waves close.

The scope of that negative: IU4 pair-code operands, 16x16x16 WMMA, one weight row tile shared
across two waves, four token tiles, 128 rows per pass, `-O3` on the current ROCm. It says nothing
about a shape whose chains stay at four or more while its registers fall — which is exactly what
the next experiment should be, because `ORD = 2` won at *unchanged* occupancy and the 19 slots it
saved are only a third of the 30 ms that separates this stage from its issue model.

## Installed

Canonical `097a54a30791a7f29cb8a13fcbf0723e85e453982b84bff78a6d83fcf19c8579` reproduced the
ordering on its own panel at 128 rows, 186.847 ms against 189.856 median with all three paired
rounds favourable and the residual FNV unchanged. Single-stream `--bench` read 33.65 tok/s, the
figure this box gives under load; that path never calls `run_ffn_batch`. The resident server came
back on the new binary.

## What to try next here

- **Compose with the scale-broadcast and `cscale` layout change.** Removing the sixteen
  `ds_bpermute` frees the eight VGPRs of lane selectors they pin; on the slice-outer order that was
  measured at 252 to 233 VGPRs and five waves to six. Applied to `ORD = 2` it may buy the 4-chain
  shape a wave slot it currently does not have, which is the one cell this panel could not reach.
- **`ORD = 4` on device.** It keeps all eight chains and changes only where the expansion happens,
  so it separates "hoisted expansion" from "fewer chains" in the `ORD = 2` result. It is built and
  selectable; it was never run.
- The 30 ms this stage spends above its issue model is still unexplained. Neither occupancy nor
  issue count accounts for it, and weight DRAM traffic (4.28 GB per pass, 50 GB/s) and B re-reads
  (33.4 GB per pass of L1/L2, 393 GB/s) are both far from a wall.

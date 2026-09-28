# The generation-shape FFN waits for its activations, not for its instructions

> **Read [`ffn-decode-bytes.md`](ffn-decode-bytes.md) with this.** The open question at the bottom of
> this document - the pair arm has no operand cursor - is now measured at the two-token-tile shape:
> the cursor is worth -3.9% on the pair arm and the pair arm is still 15-18% behind the dense one,
> because at 32 rows this phase is **stream-rate bound at about 110 GB/s and the two images' times
> are their byte counts divided by the same rate**. The pair block is 303 instructions against the
> dense block's 602 for identical matrix work and it is slower, so the census is slack at this shape.
> `bench/wstream` gives the same geometry 202 GB/s with no model; the kernel gets 112.

A generation step of 32 streams spends 34.3 ms of its 118 ms in the FFN, second only to the
recurrent state. Every attempt to make that arm faster this week has moved instruction counts, and
every one of them has measured a null. This is why, and what does move it.

The arm in question is `k_proj_opt<IU4, WM_DENSE, LAY, false, 2, 4, FUSE>`: the dense five-trit
weight image, two token tiles, one row tile per wave. It serves every pass of 32 rows or fewer,
which means a 32-stream generation step, a 32-row prefill pass, and the down projection up to 64
rows. Above that the row-count rule hands the stage pair codes and this arm does not run.

## Two measurements that agree, from opposite directions

**Cutting instructions does nothing.** A peer removed 9.4% of the block's issue slots by
re-addressing its operands (648 to 587, bit-identical) and the 32-stream step moved 280.2, 279.9 to
277.6, 279.7. A null.

**Cutting 37% of them and paying 23% more bytes costs 38%.** The pair-code image decodes to the
same weights through the same nine-valued alphabet, so it is bit-identical to the dense image by
construction and `--ffn-image` orders the two inside one process. Mode 19, 32 streams, one build,
two rounds:

| phase | dense five-trit | pair codes | change |
|---|---:|---:|---:|
| `ffn` | 34.349 | 47.594 | **+38.6%** |
| `gdn-resident-core` | 49.236 | 49.001 | -0.5% |
| `sequence-input-projection` | 13.802 | 13.856 | +0.4% |
| `sequence-output-projection` | 11.026 | 11.166 | +1.3% |
| `sequence-core` | 5.421 | 5.433 | |
| `head-projection` | 2.535 | 2.539 | |
| device span | 119.06 | 132.26 | |

Residual FNV-64 is `8087160186626784132` in all eight samples, so the two coordinates are the same
map on device and the `ffn` row is the entire difference. Pair codes are 408 issue slots per block
against 648 and read 4.28 GB per step against 3.47. At the 242 GB/s the memory system delivers,
that extra 0.81 GB buys 3.3 ms of the 13.2 ms it lost.

In a cold round of the same process the two images measured 132.76 and 133.15 ms, level. The
penalty only appears once the clock is up, which is what a latency term looks like and not what an
issue term looks like.

So this arm is not issue-bound and not byte-bound. What separates the two images is that the dense
arm has a weight cursor (`ffn_dense_stream.hpp`) that requests a block's operand bytes a block
before it hands them out, and the pair arm has none.

## What the compiled loop was actually waiting on

Reading the emitted ISA for the decode instantiation, the weight loads were never the problem after
the cursor landed. The *activation* fragments were:

```
global_load_b64 v[189:190], v[187:188], off
global_load_b64 v[187:188], v[187:188], off offset:128
global_load_b64 v[193:194], v[191:192], off
global_load_b64 v[191:192], v[191:192], off offset:128
s_waitcnt vmcnt(3)
v_wmma_i32_16x16x16_iu4 ...
```

Four `global_load_b64` issued, then a wait on the first of them four instructions later, four times
per block. The only work covering that round trip is the previous slice pair's four matrix
instructions.

The cause is the block body's own scheduling barriers. They were placed to stop the compiler
hoisting a whole block's radix-3 peel above the first matrix instruction, which is a register
pressure problem: five peeled words per source dword live at once runs the allocation past what the
wave slots pay for. But `__builtin_amdgcn_sched_barrier(0)` fences every instruction class, so the
barrier that pinned the peel also pinned the loads.

Mask `0x20` lets VMEM reads cross and keeps fencing VALU. The peel stays where it was put; the
loads may move.

| dense block, decode shape | fenced (`0`) | loads may cross (`0x20`) |
|---|---:|---:|
| issue slots | 587 | 601 |
| `s_waitcnt` per block | 46 | **39** |
| largest load cluster | 4 | **11** |
| VGPRs | 200 | 206 |
| waves per SIMD32 | 7 | 7 |

Six registers, no wave slot, seven fewer waits.

## What it is worth

`--ffn-dense` selects the load schedule at runtime, so the arms interleave inside one process:
0 fences everything with a two-block cursor (the shape this replaced), 1 lets VMEM reads cross,
2 and 3 add a third block to the cursor.

**A 32-stream generation step**, mode 19, two rounds, traced and untraced cells:

| phase | fenced | loads may cross | change |
|---|---:|---:|---:|
| `ffn` | 34.326 | **32.918** | **-4.10%** |
| `gdn-resident-core` | 49.361 | 48.855 | -1.03% |
| `sequence-input-projection` | 14.122 | 13.964 | -1.12% |
| `sequence-output-projection` | 11.827 | 11.652 | -1.48% |
| `sequence-core` | 3.252 | 3.254 | +0.06% |
| `head-projection` | 2.526 | 2.526 | 0.00% |
| device span | 117.98 | 115.73 | -1.91% |
| step, untraced | 113.48 | 112.85 | -0.56% |

The three projection and state phases this cannot reach moved 1.0 to 1.5% in the same direction, so
read the FFN row as about 3% net. The two rounds reproduce the `ffn` row to 0.07%.

**A 32-row prefill pass**, same build, shuffled cases, two rounds, `ffn` normalised against the
phases the change cannot touch:

| rows | fenced | loads may cross | cursor three deep |
|---:|---:|---:|---:|
| 32, round 0 | 33.138 | **29.015** | 29.159 |
| 32, round 1 | 31.042 | **29.530** | 30.475 |
| 32, normalised | 1.000 | **0.941** | 0.969 |
| 128, normalised | 1.000 | 1.000 | 0.997 |

**-5.9% on the FFN phase at 32 rows**, and a null at 128 rows, which is the control: above 32 rows
the row-count rule gives gate/up pair codes and this arm does not execute. Residual FNV-64 is
`1873717996769712938` at 32 rows and `7446865760224376151` at 128 in every cell of every schedule.

### A third block in the cursor is worse than a wave slot

Depth three is the obvious next step from `ca9d83a`'s 13.705 to 11.445 to 9.835 ms series on the
down stage, and it loses at both shapes this arm serves: 30.475 against 29.530 ms at 32 rows,
and the same ordering in the generation step. It allocates 235 VGPRs against 206 and drops the
block from seven waves per SIMD32 to six. Once the block's own activation fragments are allowed to
move, the weight stream is not what the arm is waiting for, and buying more of it with a wave slot
is a bad trade.

## For the next kernel you put a barrier in

A scheduling barrier is not a scheduling hint, it is a wall, and `0` is a wall against every class.
Before writing one, name which pressure you are containing:

- The dense arm's barrier contains **VALU** pressure (the peel), so it should not fence memory.
  Mask `0x20`.
- The pair arm's barrier at four token tiles contains **VMEM** pressure: without it the compiler
  hoists all thirty-two fragment loads of a block above the first matrix instruction and spills
  10-12 registers. That one has to keep fencing loads, and `docs/ffn-pair-lookahead.md` measured
  what it is worth.

Two barriers, the same builtin, opposite requirements. The lane has three more `sched_barrier(0)`
calls in kernels that have never been read for this: the pair arm's `PAIR_SCHED`, and the sequence
projection's slice barriers.

## Reproduce

```sh
tools/run-batch-compare --profile-tool --out DATA/dense-sched-dec32.json \
    --modes 19 --decode-streams 32 --heads 1 --traces 1 --rounds 2 --ffn-dense 0,1
tools/run-batch-compare --profile-tool --out DATA/dense-sched-prefill.json \
    --modes 19 --rows 32,128 --heads 0 --traces 1 --rounds 2 --ffn-dense 0,1,2
tools/run-batch-compare --profile-tool --out DATA/image-order-dec32.json \
    --modes 19 --decode-streams 32 --heads 1 --traces 1 --rounds 2 --ffn-image 1,2
```

`HALO_FFN_A4_DENSE` pins the schedule for a process that cannot call the setter.
Raw samples are in `../../data/bonsai2/batch-comparison/ffn-decode-dense/`.

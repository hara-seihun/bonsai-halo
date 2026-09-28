# The deployed FFN slice is not waiting on its weight request, and here is what it is waiting on

[The A4 FFN's weight cursor](ffn-decode-schedule.md) closed an hour before this iteration with a
mechanism that reads like it belongs to every streaming kernel in this engine:

> gfx11 gives a wave **one in-order `vmcnt` counter**, so a block that requests the next block's
> bytes and then waits for anything of its own drains its own lookahead at that wait.

`mvw_rows` is built in exactly that order, and it is the block loop of `k_ffn_slice` — the FFN of
the **default** prompt route, mode 20, which this panel measures at **180.7 ms of a 242.6 ms
128-row pass, 74% of everything**. So the question was whether the same counter costs the same
thing here. It does not, and the measurement that says so also prices the phase properly for the
first time.

**Nothing in the runtime changed.** What is in the tree is three instruments, a comment in
`mvw_rows` naming what was ruled out, and this document. The losing arm is
[a patch](../../../data/bonsai2/batch-comparison/ffn-slice-issue-order/issue-order-arm.patch), not
a second code path.

## The drain is real, and now it is a number rather than a reading

`tools/vmcnt_cover.py` walks a compiled block twice — so the second pass sees a steady state with
the previous iteration's requests still outstanding — simulates the in-order counter, and reports
how far each request gets before a wait drains it. Deployed `k_ffn_slice<32, 0, 6>`, 476-instruction
block, 23 requests, 32 matrix instructions:

| request | issued at slot | drained at | cover | matrix in the cover |
|---|---:|---:|---:|---:|
| the three weight loads for block `b+1` | 0, 1, 2 | 264 | **264, 263, 262** | **0** |
| `xsum`, the accumulator seed | 14 | 264 | 250 | 0 |
| group 0's eight activation fragments | 16..28 | 276..316 | 260..288 | 0..14 |
| group 1's eight activation fragments | 362..376 | 396..423 | **34..47** | 0..10 |

The weight request is drained a fifth of the way into the block by **the accumulator seed's wait for
one `xsum` word**, which is issued four loads later — the exact shape the A4 cursor document
describes, and the reason depth is the wrong knob: an in-order counter does not care how much is in
flight, only what was issued before the thing you want.

It also shows what a slot census cannot: **group 1's operands have 34 to 47 slots of cover**, the
shortest in the block, because they are requested inside the group loop that immediately consumes
them.

## The reorder is bit-identical, and it is a null

The arm requests group 0's fragments and its two scale words **before** the weight cursor, behind a
`sched_barrier` that pins VMEM order and lets ALU, matrix and LDS cross, and fences the matrix work
away from the peel so the seed's wait cannot float to the top of the block. Same bytes, same
addresses, same peel, same `tr[32]`, same half swap, same `wmma` order, same int32 block sums, same
`-xsc` seed, same `fmaf` fold. 237 VGPR against 238, six waves per SIMD32 in both.

Mode 20, 128 rows, tail head, traced, `--pin-clock`, both arms interleaved in one process, two
samples each
([`order-128.json`](../../../data/bonsai2/batch-comparison/ffn-slice-issue-order/order-128.json)):

| arm | `ffn` | eight untouched phases | `ffn`/control | pass wall |
|---|---:|---:|---:|---:|
| deployed order | **180.684** | 60.281 | 2.997 | 243.899 |
| operands first | 182.452 | 60.536 | 3.014 | 246.013 |

**+0.55% normalised, +0.87% on the pass.** Residual FNV-64 `2482349408765731952` in all four
samples, traced and untraced, which is the arms agreeing to the last bit rather than being asserted
to.

## Why it cannot pay: the weight round trip is under a tenth of this phase

`-DHALO_MVW_PIN=1` holds the weight cursor on block 0, so every weight request hits cache and the
stream disappears. It is numerically invalid by construction, it ships as nothing, and it is the
same instrument the A4 cursor document used to find **7.9 ms of 32.4** on `k_proj_opt`.

| build | `ffn` | control | `ffn`/control | against deployed |
|---|---:|---:|---:|---:|
| deployed | 183.928 | 61.528 | 2.989 | |
| weight cursor pinned | **152.110** | 55.307 | **2.750** | **−8.0%** |

**Read the pinned build's census before spending that number**: pinning removes an address cursor
and the block goes 476 instructions to **519**, so the arm pays 9% more issue for its own diagnosis
and the weight round trip is worth *at least* those 8%. Call it a tenth of the phase. That is the
ceiling on every schedule change aimed at this stream, and the reorder captured none of it.

**The activation half of the same probe is confounded and is reported as such.** `-DHALO_MVW_PIN=2`
pins the fragments and measures **+18.6%** — with a block of 518 instructions carrying **124**
scheduling slots against the deployed block's 54. That is a different kernel, not a measurement of
the activation stream, and it is in the raw directory so nobody repeats it.

## The second arm: the request this body does stall on, and why it is not free either

`tools/vmcnt_cover.py` says the weight cursor is not the short pole in this block - **group 1's
activation fragments are**, at 34 to 47 slots of cover with zero matrix instructions inside, against
group 0's 260 to 288. Only the *first* fragment of a group matters: after its two matrix
instructions are issued, every later fragment of that group is covered by the ones before it, which
is exactly the shape of the drain table (34 slots on the first, then 1 to 10 matrix instructions of
cover, each worth 21.9 slots).

So the cheap arm is six VGPRs: hoist one `int4` and the two scale words of every group after the
first to the top of the block, and they get the whole peel plus sixteen matrix instructions of
cover. Built, bit-identical by construction, `k_ffn_slice<32, 0, 6, 1>` at **240 VGPR, six waves,
no spill**.

**It is worse on the census and it is not measured to a conclusion.** The block goes 476
instructions to **518**, because the allocator pays for the six registers by sinking group 0's
operand loads from slot 16 to slot 247 - the block's shortest cover goes 30 slots to **8**, and the
mean over 20 requests to 72. The compiler is defending a local optimum: every hand-placed request in
this body is paid for by a sunk one somewhere else, which is the same answer the first arm gave from
the other side. Its panel did not run: the box was saturated by peers' holds for the rest of this
turn, and three attempts returned one arm's cells. **The arm is
[a patch](../../../data/bonsai2/batch-comparison/ffn-slice-issue-order/group-operand-arm.patch) and
its listing is beside it**, so the next engineer spends a build and a panel rather than a design.
[A peer's four-group ladder](ffn-slice-rows.md) needs that answer before its own rung can pay: at
64 rows a slice the same body reads **mean cover 27.9 slots over 43 requests**, because four groups
are four sets of the same late request.

## What the phase is made of, with the constant measured instead of quoted

Every block census in this repository converts instructions into time with a matrix-instruction
constant, and [the A4 schedule](ffn-decode-schedule.md) had already caught one of them wrong by 2x.
`bench/wmma_cost` measures them on this device: four independent accumulators so nothing waits on a
result, `v_fmac_f32` as the harness control at one issue slot by construction, eight waves per
SIMD32, `--pin-clock`.

| instruction | issue slots per SIMD32 | MAC/cycle/SIMD32 |
|---|---:|---:|
| `v_fmac_f32` | 1.00 (the calibration) | 1.0 |
| `v_dot4_i32_iu8` | **1.34** | 3.0 |
| `v_wmma_i32_16x16x16_iu8` | **21.91** | 5.8 |
| `v_wmma_i32_16x16x16_iu4` | **11.00** | 11.6 |
| `v_wmma_f32_16x16x16_f16` | 21.95 | 5.8 |

Two waves per SIMD32 gives the same ratios (21.1 / 10.7 / 21.1), and the VALU arm reports the clock
the run actually held — 1860 MHz of a nominal 2900 in a probe this small — which is why the table is
in slots. **`iu4` is exactly half of `iu8`, so splitting an eight-bit activation into two nibbles to
reach the cheaper instruction buys nothing**: the 2x on this kernel belongs to the operand width
itself, which is the A4 route and its own numerical map, not a schedule anyone can rearrange.

Against that constant the deployed 32-row block is **32 x 21.9 = 701 slots of matrix work inside
1145 counted slots — 61% matrix**. So the whole non-matrix body of this kernel, peel and half swap
and scale fold and addressing together, is 39% of its counted issue, and the weight stream is a
tenth of its measured time.

## What this leaves, in the order I would take it

1. **A third of this phase is neither its bytes nor its counted instructions, and that is now a
   number rather than a ratio.** A 128-row pass runs 16.71M block iterations (64 layers x [1088
   gate/up tiles x 40 blocks + 160 down tiles x 136 blocks] x 4 slices), and `--clock-log` says the
   device holds **2246 MHz median (2217-2421, 103 busy samples, 122 W)** through it - 0.77 of the
   nominal 2900 this lane's models keep using. So:

   | term of the deployed 180.7 ms | ms | share |
   |---|---:|---:|
   | 1145 counted slots at 2246 MHz over 80 SIMD32 | **106.5** | 59% |
   | the weight round trip, from the pinned arm | **~14.5** | 8% |
   | **neither** | **~59.7** | **33%** |

   Candidates for the 33%, in the order the census suggests: the 22 `s_waitcnt` and 32 `s_delay_alu`
   a block already carries, the nine LDS operations of the scale table on `lgkmcnt`, and whatever a
   `v_wmma` costs beyond its issue slots when its operands are produced two instructions earlier.
   None of them is the memory stream, and none of them is what an instruction cut addresses.
2. **Group 1's operands have a tenth of the cover group 0's have** (34 slots against 264) and they
   are the requests a fourth column group multiplies. Anyone widening the slice should re-run
   `tools/vmcnt_cover.py` on the wider body before assuming the extra groups are free.
3. **The 39% that is not matrix is where an instruction cut can still pay**, and the peel is most of
   it — [the peel gather](mv-peel-gather.md) took 85 slots out of this same block for -1.66% of the
   phase, which is about what 85 of 1145 slots predicts. Price the next one against 1145, not
   against 476.

## Reproduce

```sh
# the census and the cover, no GPU, three seconds each
hipcc --offload-arch=gfx1151 --cuda-device-only -S -O3 -std=c++17 -Isrc -Ikernels \
      -I$LLAMA/include -DHALO_SLICE_PROBE=32 kernels/halo_rows.hip -o /tmp/slice32.s
tools/isa_loop_count.py /tmp/slice32.s k_ffn_slice
tools/vmcnt_cover.py /tmp/slice32.s k_ffn_slice

# what a matrix instruction costs this device
make bench/wmma_cost
tools/run-batch-compare --pin-clock --exec bench/wmma_cost --grid 80 --iters 4000

# the phase, and the same phase with its weight stream pinned into cache
tools/run-batch-compare --pin-clock --profile-tool --modes 20 --rows 128 --heads 1 --traces 1 \
  --rounds 2 --context 512 --warmup-ms 400 --out OUT.json
rm -f kernels/halo_rows.o && make -j8 DEFS='-DHALO_MVW_PIN=1' tools/batch_profile   # then the same panel
tools/arm_panel.py OUT.json --key mvw_order --phase ffn
```

`rm kernels/halo_rows.o` before and after a `DEFS=` build: `make` cannot see a changed `-D`, and a
pinned object must never be linked into a binary that produces tokens.

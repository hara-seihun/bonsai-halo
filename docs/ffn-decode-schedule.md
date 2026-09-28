# One in-order counter decides what a weight cursor is worth

[The byte law](ffn-decode-bytes.md) left the A4 FFN's generation-shape phase with a gap nobody had
explained: 3.476 GB read in 32.425 ms is 107 GB/s where `bench/wstream` gives the same padded
geometry 202 GB/s and `bench/bw` says 242. A peer then priced the gap directly by pinning the weight
address so every request hits cache — `ffn` 32.4 → 24.5 ms — so **7.9 ms of a 32-stream generation
step is the weight stream's round trip and the other 24.5 ms is issue**.

This iteration went after those 7.9 ms with the one lever that owns them, the block cursor in
[`ffn_dense_stream.hpp`](../kernels/ffn_dense_stream.hpp), and **all three arms lose**. The reason is
worth more than the arms: on gfx11 a wave has **one in-order `vmcnt` counter**, so a block that
requests the next block's bytes and then waits for anything of its own drains its own lookahead at
that wait. The compiled down-stage block says it in one line — `s_waitcnt vmcnt(0)` at slot **128 of
363**, with the cursor's six requests at slots 1..6. The bytes are in flight for a third of a block,
not for the block the cursor was built to buy.

Everything below is bit-identical by construction and measured so: residual FNV-64
`8099343800456739472` in **all eleven samples of all three panels**. Only when a load is issued
moves.

## What the cursor cannot buy

Mode 19, 32 streams, one token each, `--decode-prompt 64`, traced, `--pin-clock`, arms interleaved in
one process. `ffn` is normalised by the eight phases the cursor cannot reach.

| arm | `ffn` ms | control ms | `ffn`/control | against deployed |
|---|---:|---:|---:|---:|
| **deployed**, cursor 2 deep, requested at the top | 32.758 | 66.466 | **0.4928** | — |
| down cursor 4 deep (`--ffn-dense 4`) | 32.895 | 65.850 | 0.4995 | +1.4% |
| down cursor 6 deep (`--ffn-dense 5`) | 32.579 | 65.902 | 0.4943 | +0.3% |
| requested late, shifting cursor | 34.702 / 34.398 | 67.173 / 66.753 | 0.5166 / 0.5153 | **+4.6% / +2.3%** |
| requested late, ring cursor, body unrolled | 43.775 / 43.863 | 66.565 / 66.573 | 0.6576 / 0.6589 | **+33.4% / +34.2%** |

The deployed arm reproduces to 0.4908-0.5038 across the three panels, so the depth ladder is inside
its own noise and the two late arms are not.

**Depth is a null because the drain does not care how much is in flight.** Depth 4 and 6 put two and
four more blocks of weight bytes in the air and the same `s_waitcnt vmcnt(0)` — now at slot 136 and
143 — waits for every one of them. `docs/ffn-down-waves.md` bought depth 2 on a stage that had 160
waves; at 320 waves and this drain the ladder above it is flat.

**A late request is undone by the buffer shift.** Issuing the fill after the block's last consumption
is the obvious repair and it costs 2-5%. `hand()` shifts the ring down by one slot, and a shift is a
register copy that *depends on the load that filled the slot above it*, so the compiler places the
copy immediately after the fill and covers it with `s_waitcnt vmcnt(0)` **36 slots later**. The arm
issues the request in the right place and then waits for it in the wrong one. Registers fall (gate/up
205 → 194, down 118 → 109) because the fill's destinations live for a shorter span; that is the only
thing it wins.

**Removing the copy costs more than the copy.** The ring form — block `b` reads slot `b % 2` and
refills that same slot for block `b + 2`, so nothing is ever copied and the fill is a whole block
from its first wait — needs the block loop unrolled by the cursor's depth to make the slot index a
constant. Unrolled, `k_proj_opt` allocates 180 VGPR instead of 205 on gate/up and **the phase is a
third slower**. Two copies of a 600-slot body is not a schedule this kernel wants, and the register
dividend does not pay for it.

## What the memory system is not short of

Two new arms in [`bench/wstream`](../bench/wstream.hip), both with no model and no lock beyond the
probe itself. They were built to test two explanations of the 107 GB/s and they kill both.

**Half of a 32-lane request can be duplicate addresses for free.** Every WMMA operand load in this
engine is addressed by `lane & 15` — the fragment is replicated across the wave's halves — so a
`global_load_b128` covers 256 distinct bytes where a full-lane load covers 512, and a kernel issues
twice as many instructions for the same run. The `half-lane` arm is exactly that change and nothing
else:

| geometry | padded, full lane | padded, half lane | ratio |
|---|---:|---:|---:|
| ffn gate/up (2176 tiles, nb 40) | 208.3 | 205.0 | 0.98x |
| ffn down (640 tiles, nb 136) | 194.3 | 193.0 | 0.99x |
| sequence input (1024 tiles, nb 40) | 192.7 | 194.6 | 1.01x |
| head (7760 tiles, nb 40) | 216.1 | 213.9 | 0.99x |

Ten geometries, two work levels, **0.96 to 1.02x everywhere**. The texture unit coalesces the
duplicate half for nothing. *No redesign that spends instructions or cross-lane operations to give
every lane distinct bytes can win on bandwidth* — that closes the same question for `head_tile`,
`project` and `mvw_rows`, all of which read their weights the same way.

**The dense arm's own geometry streams at 203 GB/s at the dense arm's own occupancy.** The
`dense_arm_shape` probe reads what `k_proj_opt` reads — two 416-byte runs per block through
`dense_load0/1/2` off `lane & 15`, tile-major, 1088 waves of 40 blocks — with the LDS knob holding
the machine at the kernel's resident wave count:

| stage | waves/SIMD32 | tile-major | block-major |
|---|---:|---:|---:|
| gate/up, 1088 waves, nb 40, ~600 slots of work a block | 16 | 187.0 | 197.5 |
| gate/up | **7** (the kernel's own) | **203.6** | 213.4 |
| gate/up | 4 | 165.3 | 163.9 |
| down, 320 waves, nb 136, ~300 slots | 7 | 206.6 | 204.9 |
| down | **4** (the kernel's own) | **205.7** | 204.8 |

So at the shape, the order, the occupancy and the work density the kernel actually runs, the stream
delivers about twice what the kernel gets. **The 107 GB/s is not the geometry, not the stride, not
the lane duplication and not the wave count.** It is issue plus an uncovered round trip, and the
round trip is uncovered because of the counter, not because of the memory.

The general instrument is worth more than this use of it. `bench/wstream` now walks resident waves
from 16 down to 1 at two work levels, with `hipOccupancyMaxActiveBlocksPerMultiprocessor` printing
what the device actually admitted: at 192 dependent ALU operations per 512-byte visit the stream
reads 204.7 GB/s at 16 waves, 202.9 at 7, 148.0 at 4, 97.2 at 2 and 59.5 at 1. **Four waves per
SIMD32 is where a streaming phase starts paying for its occupancy**, and with no work in the loop at
all the same walk is flat at 204-227 from 16 waves down to 2. Any claim that a phase is short of
memory parallelism can be priced against that curve in ten seconds.

## The arithmetic this leaves in the tree

Per wave-block of the dense gate/up arm at 32 rows: **602 issue slots, 26 `global_load`, 32
`v_wmma_i32_16x16x16_iu4`, 16 `ds_bpermute_b32`**, and 1490 SIMD32 cycles measured (3.476 GB at
832 distinct bytes a wave-block, 32.425 ms, 80 SIMD32 at 2.4 GHz). The pair arm is 303 slots and
**1856** cycles.

**`v_wmma_i32_16x16x16_iu4` is 16 cycles per SIMD32, not 32.** The dense arm measures *below* the
32-cycle model (1490 against 1594), which settles the constant: the iu4 model is 602 + 32·16 = 1082
cycles and the arm runs at 73% of it, the pair arm's is 815 and it runs at 44%. Anyone pricing this
kernel against `wmma_iu8`'s 32 cycles is off by a factor of two on half the block.

## What is left, and who owns it

**The only issue order that lets a weight request survive its own block is `[every B fragment this
block consumes][the weight request][the block's work]`.** In-order `vmcnt` gives no other way: a wait
for the last B fragment allows exactly the loads issued after it to stay outstanding, and the cursor
has to be among them. That needs all sixteen B fragments of the block in registers before the first
matrix instruction — 32 VGPRs at two token tiles, which the down arm has room for at 118 — and it is
the **B-operand side of this kernel**, held by `815433b1`, whose own pin says B locality and B
latency are already free. This is the one arm of that claim that is not about request count.

**Built, and the issue order was not the whole answer.** With the activation stage in the tree the
block's fragments are `lgkmcnt`, so the order this section asks for already exists and the cursor's
request already survives its block. [The block pipeline](ffn-block-pipeline.md) put the same
treatment on what was left - the weight-scale pair and the token scales, 3 to 24 slots of cover
each - for -2.40% of this phase and -0.99% of a 32-stream step, bit-identical.

Two things not to spend a turn on again: the depth ladder (`--ffn-dense 2,4,5` are in the tree and
are nulls at this shape), and any argument that this phase is short of stream bandwidth.

## Reproducing

```sh
make bench/wstream
tools/run-batch-compare --pin-clock --exec ./bench/wstream --mb 1024 --rounds 2
tools/run-batch-compare --pin-clock --profile-tool --modes 19 --decode-streams 32 \
  --decode-prompt 64 --ffn-dense 1,4,5 --traces 1 --rounds 1 --warmup-ms 400 --rows 32 \
  --out ../../data/bonsai2/batch-comparison/ffn-decode-schedule/dense-ladder.json
python3 tools/phase_totals.py .../dense-ladder.json
```

The two late arms are not in the runtime. `late-cursor-arms.patch`, beside the raw samples in
[`batch-comparison/ffn-decode-schedule/`](../../../data/bonsai2/batch-comparison/ffn-decode-schedule/README.md),
applies them to `ffn_batch.hip`, `ffn_dense_stream.hpp` and `tools/batch_profile.cpp` as
`--ffn-dense 6` and `7`.

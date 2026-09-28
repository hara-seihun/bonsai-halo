# The eight-row matvec was gated behind sixteen scalar drains a weight block

Held and released: `mv_rows_t` and its bodies in `kernels/device.hpp`, the `mv_rows_t` call in
`kernels/phases.hpp:ph_matvec`, `kernels/mv_body_isa.hip`, and two new instruments,
`tools/mv_sched_census.py` and `bench/mvsched.hip`. Raw samples, the census tables and the
executable hashes are in
[`batch-comparison/mv-scalar-waits/`](../../../data/bonsai2/batch-comparison/mv-scalar-waits/README.md).

The deployed drafted step moves the target's weights at about 127 GB/s. The same kernel on the same
image at one row moves them at 200. Three engineers have cut instructions out of this body for a
null or 1% — the two-trit palette, the IU4 nibble operand, the A4 block — and
[`docs/mv-dot4-body.md`](mv-dot4-body.md) closed with "price an instruction-census change to
`mv_rows_t` at a tenth of its census". This is not an instruction-census change. **The two shapes
differ by a factor of sixteen in how often the wave stops.**

## What the assembly says

Every scalar wait on gfx11 is `s_waitcnt lgkmcnt(0)`. SMEM returns out of order, so the counter
cannot be waited part way: one wait drains **every** scalar load the wave has outstanding. Counted
in `kernels/mv_body_isa.hip` with `tools/mv_sched_census.py`, per instantiation:

| `mv_rows_t<5, TT, false, 4>` | `lgkmcnt(0)` waits | in the block loop | block-loop slots | `v_dot4` |
|---|---:|---:|---:|---:|
| TT = 1 | 3 | **1** | 310 | 32 |
| TT = 8, deployed schedule | 19 | **16** | 746 | 256 |
| TT = 8, row batch (shipped) | 18 | 15 | 852 | 256 |

Both stream the same 896 weight bytes per block. The one-row body stops once for them; the
eight-row body stops sixteen times — twice a row, once for the activation pair and once for the
`xs`/`xsum` drain scalars loaded inside the row.

**The deployed loop's own prefetch is what drains it.** It asks for row `r + 1`'s thirty-two dwords
and then waits for row `r` one to three instructions later, so the request it has just issued is
drained by the wait it was already paying. The double buffer it holds costs 32 SGPRs and buys
nothing. That diagnosis is `d8f6882a`'s, left in a checkout at 17:57 and never measured; the census
above is its confirmation.

## What shipped

`mv_rows_batched` requests **`RB` rows of activations and their drain scalars together**, waits
once, and then retires `RB * 32` dot products whose `RB * ACC` accumulator chains are independent.
The first batch of a block is requested above the trit expansion, so its round trip has 229 slots of
cover; every later batch's wait lands after the previous batch's dot products.

**`RB = 2` is the knee, and the reason is cover rather than count.** One wait per pair is only one
fewer wait per block than the deployed schedule, and it is worth 4%; one wait per row is the same
count as the pair and is worth nothing, because a single row has nothing to put between the request
and the wait. Four rows per batch needs 128 SGPRs of activations against a 107-SGPR file and loses
12%. Measured on `bench/mvsched`, 37.2 MB HALO image, ten waves per SIMD32, palindrome arm order,
both passes shown:

| TT = 8 arm | GB/s | | TT = 8 arm | GB/s |
|---|---:|---|---|---:|
| deployed | 115.3 / 114.3 | | rows per batch = 4 | 99.2 / 102.7 |
| rows per batch = 1 | 115.1 / 113.6 | | weight blocks in flight = 2 | 115.7 / 115.5 |
| **rows per batch = 2** | **119.2 / 119.6** | | activations on the vector path | 124.6 / 123.3 |

Controls in the same panel, all on the deployed body: TT = 1 at 216.7 GB/s, TT = 2 at 216.2,
TT = 4 at 168.7. **The one- and two-row bodies are already at the image's stream roof**, which is
where the sixteen-against-one drain count comes from and why this change cannot reach them.

`TT = 1` and the drafter's `Q8` image keep `mv_rows_deployed` instruction for instruction, and
`HALO_MV_SCHED=0` restores the deployed schedule everywhere as the control arm.

## Full model

Two binaries from one checkout on `b4e5365`, differing only in `kernels/halo_rows.o` and
`kernels/halo_draft.o`, alternated in one clock state; 1227-token prompt, 96 generated tokens,
Q4 drafter, sliced route:

| | deployed schedule | row batch |
|---|---|---|
| drafted verify pass, per step | 47.775 / 47.783 ms | **45.764 / 45.299 ms (−4.7%)** |
| drafted generation | 47.21 / 47.23 tok/s | **49.40 / 49.91 tok/s (+5.2%)** |
| drafter pass, per step | 8.696 / 8.674 ms | **8.212 / 8.124 ms (−6.0%)** |
| prompt ingestion, sliced route | 172.9 / 171.9 tok/s | 172.2 / 181.3 tok/s |
| greedy digest | `10538716321276340354` | `10538716321276340354` |
| drafted / accepted / steps | 252 / 60 / 36 | 252 / 60 / 36 |

**Bit-identical, and the drafter's counts are the second witness.** A lossless drafter verifies
every token against the target, so identical acceptance over 252 drafts means the target's argmax
agreed at all 252 positions. The change moves no value: same bytes, same blocks, same K order, and a
row's `int32` sum is exact under any grouping — the argument `ACC` already ships on.

The drafter pass moving 6% is not a second effect. `halo_draft.hip` runs the target's ternary
vocabulary head through this same body at the drafted row count.

### Installed acceptance

Installed into `.` twice, because a peer's own install landed between
the two. The first pair is the acceptance — the canonical binary before, and this change on
`da668b9` as `e6d386703a572bed0bdd11b3` — and the last column is the build canonical master carries
now, `af2d93f` as `9e40e8cde3a587265cd2c045`, which is this change on top of that peer's:

| | canonical, before | installed on `da668b9` | installed on `af2d93f` |
|---|---:|---:|---:|
| drafted verify pass | 47.761 ms | **45.291 ms (−5.2%)** | **45.159 ms** |
| drafted generation | 47.25 tok/s | **49.93 tok/s (+5.7%)** | **50.11 tok/s** |
| prompt ingestion, sliced route | 168.4 tok/s | 182.3 tok/s | 181.8 tok/s |
| greedy digest | `10538716321276340354` | `10538716321276340354` | `10538716321276340354` |

The resident service runs the last of those and answers `/v1/models` and a completion. **An install
race is not a measurement:** the middle column is the pair taken against the binary it replaced, and
the third is there to say the number survived a rebuild on a moved tree. Prompt ingestion here is the sliced route, which the four-bit sequence default landing
the same hour cannot reach — the equal digest across all four builds is what says so.

## Two rejected arms, kept executable in the bench

Both are in `bench/mvsched.hip` under `namespace rejected`, out of the runtime header.

**Activations on the vector path is the fastest body measured and cannot be paid for.** `vmcnt` is
in order and partially waitable, so a uniform `global_load_b128` pipeline never drains what it does
not need: the compiler emits `vmcnt(7)` down to `vmcnt(0)` between groups of four dot products, and
with two rows in flight the arm is **+8.0%**, the best number in the panel. One row of activations
is 32 VGPRs. Under `HALO_ROWS_WPE = 12` the deployed `k_forward_rows<8, 0, 0, 0, false, 12>` is
already 120 registers with **7 spilled**, and this arm spills **370**. It becomes shippable the day
that kernel gets its registers back, or the day the matvec phase is its own kernel; the body and its
bench arm are ready for that day.

**A deeper weight cursor is a measured negative, which refutes the model that picked it.** Two
blocks of weights in flight is a null (115.6 against 114.8) and three costs 2.9% at eight rows and
8% at four. The hypothesis was Little's law on the wave: at TT = 1 sixteen waves each hold 896 bytes
in flight and the body reaches the roof, at TT = 8 ten waves hold the same 896 and reach 57% of it,
and the ratio of in-flight bytes per SIMD32 fits the ratio of achieved bandwidth to within 3%. **The
fit is real and the prediction is wrong.** Adding bytes in flight per wave does not buy bandwidth
here, so whatever that ratio is measuring, it is not a request-parallelism deficit. The runtime
carries no depth parameter at all.

## What this leaves

**The scalar drains were not the whole gap, and now neither candidate explanation is left.** After
the change, a TT = 8 block period at ten waves per SIMD32 is about 13,600 cycles against 7,460
cycles of issue for its 852 slots, so 45% of the pass is still waiting, and it is not the scalar
counter (removing 45% of the stops bought 4%) and not weight-request parallelism (measured
negative). The candidate nobody has priced is **the grid tail**: `ph_matvec` hands out one row tile
per unit through `next_unit`, and the D-output projections are 160 tiles at `KS = 2`, which is 320
units on a 100-workgroup grid — 3.2 rounds, so the last round leaves 40% of the grid idle and the
phase costs four rounds for 3.2 rounds of work. `gate/up` at 1088 units and the head at 7760 pay
almost nothing for the same reason. That is arithmetic against the tracer's per-phase table, and it
predicts which phases a unit-splitting change would move. It also predicts that the whole-pass grid
sweep in [`docs/matvec-crossover.md`](matvec-crossover.md) mixed phases whose tails peak at
different grids.

**One image in the handoff's stride list is closed without a panel.** A HALO tile run strides
`nb * 896` bytes and `896 = 2^7 * 7`, so it is a multiple of 4096 only when `nb` is a multiple of
32; this geometry's `nb` is one of 8, 16, 40, 48, 80, 136. The persistent kernel's weight image
cannot be in the aliasing band [the stride rule](ffn-image-order.md) found, whatever the pass
measures.

## The instruments

```sh
tools/mv_sched_census.py                     # waits, slots, registers, spills per arm; 8 s, no GPU
tools/mv_sched_census.py --rb 1,2,4 --tt 8   # the batch width axis
make bench/mvsched && tools/run-batch-compare --pin-clock --exec bench/mvsched
bench/mvsched --waves 8|10|12|16             # resident waves, set by LDS ballast
```

`bench/mvsched` prints an FNV-64 of its outputs per arm, and every arm in every panel above
digested to `07701590705348488800`: the arms are exact against each other on device before any
model runs. Two traps it was built around, both paid for here:

- **`red[8][TT][32]` is itself 1024 * TT bytes of LDS**, so a fixed ballast measures TT = 8 at eight
  waves per SIMD32 and TT = 1 at ten. The first panel taken that way ranked the arms differently
  from the corrected one — the vector arm and the row batch swapped places. Ask
  `tools/kernel_resources.py bench/mvsched.hip kmv` what occupancy each instantiation actually got.
- **A wave count is not an occupancy.** 26,000 bytes of LDS per 256-thread workgroup is five
  workgroups a WGP and ten waves a SIMD32, which is what the deployed cooperative grid of 100 gives
  `k_forward_rows<8>`. The arms rank differently at eight.

## The other side of the ratio was bought, and the drain count is a correlate

[`docs/mv-tile-fold.md`](mv-tile-fold.md) spent the VGPR file where this document spent the SGPR
file: one wave carrying `G` output tiles dots every activation batch against all of them, so the
weight bytes between two `lgkmcnt(0)` grow by `G` with no new scalar pressure. It is bit-exact —
same digest in every arm of every panel — and **halving the drains per weight byte is worth +1.6% to
+3.3%, and −7% in the cell where this body is fastest**; quartering them is flat. So the sixteen
waits a block counted above are a correlate of the row count rather than the cause of the 1.9x, and
the same `bench/mvsched` panel run across four co-residencies found the body's rate **non-monotonic**
in workgroups per WGP (150.4 at four, 135.7 at five, 181.5 at six), which is worth more than either
arm. The fold is `rejected::mv_rows_fold` in `bench/mvsched.hip`, beside the two arms here.

# The FFN slice's missing third is its unit boundary, not its bytes (-2.8% of a prompt pass, bit-identical, installed)

[The issue-order panel](ffn-slice-issue-order.md) closed the largest phase in this engine with a
term nobody could name: of the 180.7 ms a 128-row mode-20 pass spends in `k_ffn_slice`, 106.5 ms is
its counted issue slots and about 14.5 ms is its weight round trip, and **59.7 ms is neither**. The
obvious suspect was written into the handoff as the lane's largest unexplained number:

> **the activation stream is 4.6x the weight stream** (68.4 GB against 15.0 per 128-row pass) and is
> invariant under everything tried so far.

**It is worth 1.6%.** The 33% is the *unit boundary* - the barriers, the partial publish and the
wave-0 reduction that end every one of the 360,000 units a pass runs - and it is worth **12%**. A
drain that owns an eighth LDS slot instead of a wave's registers takes 3.9% of the phase back, and
it is bit-identical: **381,419,520 full-vocabulary logit values, one hash**.

Raw samples, the four ISA listings and the losing arms are in
[`batch-comparison/ffn-slice-activation/`](../../../data/bonsai2/batch-comparison/ffn-slice-activation/README.md).

## The instrument: collapse the footprint, do not pin the cursor

`-DHALO_MVW_PIN=2` was already in the tree and its answer was published as confounded: pinning the
activation cursor on block 0 constant-folds an address, the block goes **476 instructions to 518
with 124 scheduling slots against 54**, and it measured +18.6%. That is a different kernel.

This drops `wb` - the wave's own block offset - from a base pointer computed **once per unit**, and
changes nothing else. Same loads, same sixteen lines per load, same `b * 128` cursor, same peel,
same waits. What moves is only *which* lines the machine is asked for: every wave of every workgroup
now reads the same `pw` blocks, so a per-CU footprint of about 240 kB becomes 20 kB and the stream
is served out of L0 instead of L1/L2. The difference is that stream's delivery and nothing else.

| arm | block loop | work + scheduling |
|---|---:|---|
| 0, deployed | **476** | 422 + 54 |
| 2, activation footprint collapsed | **477** | 422 + 55 |
| 3, weight footprint collapsed | **477** | 422 + 55 |
| 4, both collapsed | **477** | 419 + 58 |
| 1, spread drain | 525 | 413 + **112** |
| 7, flat drain (**shipped**) | **477** | 423 + 54 |

Arms 2..4 compute wrong numbers by design, only a `-DHALO_MVW_ABL=1` build carries them, and
`ffn_slice_arm()` refuses to report an arm the build is not running.

## What the phase is actually made of

Mode 20, 128-row passes, tail head, traced, `--pin-clock`, arms interleaved in one process on one
clock, `HALO_FFN_GRID=1` printing **grid 60 workgroups in every arm**. Normalised against the five
phases the change cannot reach.

| arm | `ffn` ms | control ms | normalised |
|---|---:|---:|---:|
| 0, deployed | 169.179 | 51.930 | |
| 2, the 68.4 GB activation stream removed | 159.235 | 49.667 | **-1.59%** |
| 3, the 15.0 GB weight stream removed | 147.402 | 50.245 | **-9.95%** |
| 4, both removed | 151.209 | 51.492 | -9.86% |

and, in a second panel against its own control,

| arm | `ffn` ms | control ms | normalised |
|---|---:|---:|---:|
| 0, deployed | 191.212 | 58.304 | |
| 5, the whole unit epilogue removed | 160.857 | 55.687 | **-11.92%** |

So the deployed phase is **59% counted issue slots, 10% weight round trip, 1.6% activation stream,
12% unit boundary**, and the arms are not additive: removing both streams is what removing the
weights alone is, because the activation stream is already hiding behind it.

**Three nulls in this kernel are now one fact.** [Four column groups](ffn-slice-rows.md) halved the
weight stream for +0.18% per row; [straightening the operand](ffn-slice-operand.md) took 87% of the
line requests off the address unit and lost 4%; [the request reorder](ffn-slice-issue-order.md) was
+0.55%. None of them was ever going to pay, because a change that halves the activation stream is
playing for 1.6% and a change that reorders requests is playing for a tenth of a tenth. **The
activation coordinate is closed: do not build the `[block][k-slice][row]` layout, do not build two
weight tiles per wave for the sake of one shared B fragment, and do not price this kernel on cache
lines.**

## What ships: give the reduction buffer its eighth slot

A unit ends with `2 * GROUPS` sequential passes through one `[7][8][32]` buffer. Each pass has seven
waves publish eight floats, a barrier, **wave 0 read 56 floats, add 56 and write 8 outputs**, and a
second barrier that exists only to keep the next pass from overwriting what wave 0 is reading. At
128 rows that is four passes, eight barriers and 224 serial LDS reads per unit, and a `gate/up` unit
is five blocks of work per wave.

`86551bc8` shipped the fix for exactly this shape in `ph_matvec` an hour earlier - row `r` belongs
to wave `r`, eight partials in seven slots because the owner splices its own partial in from a
register - and measured -3.9% on the drafted verify pass. **That form loses 15% here**, and the
reason is the sharpest number in this document: the register splice costs **two VGPRs**, 238 to 240,
and two VGPRs in this body bought **54 more `s_delay_alu` inside the block loop**. Same waves, same
grid, same bits, 15% slower.

So arm 7 buys the eighth slot with **LDS**, which this kernel has and does not spend:

- every wave publishes all eight of its partials and reads all eight back, so `y` dies at the
  publish, no wave is special, and the reduce is eight ways parallel;
- the buffer is doubled, so pass `p+1` publishes into the copy pass `p` is reading and the
  write-after-read barrier disappears: `2 * GROUPS + 1` barriers a unit instead of `4 * GROUPS`;
- the sum is still `y_0[r] + y_1[r] + ... + y_7[r]` left to right.

**It gives back 21 registers**: `k_ffn_slice<32, 0, 6, 0, 7>` is **217 VGPR** against 238, six waves
per SIMD32, no spill, 17444 bytes of LDS - three workgroups per WGP either way, so the grid does not
move. The block loop is 477 instructions against 476 with 33 `s_delay_alu` against 32.

| mode 20, 128 rows, arms interleaved in one process | arm 0 | arm 7 | |
|---|---:|---:|---:|
| `ffn` phase, traced, six rounds | 178.547 ms | 172.253 ms | **-3.53%, -3.91% normalised** |
| device span, same panel | 234.532 ms | 228.437 ms | -2.60% |
| step wall, untraced, eight rounds, median | 231.295 ms | 224.792 ms | **-2.81%** |
| the same paired by round | | | **-2.34% median, 7 of 8 rounds** |
| 32-stream mode-4 decode step, three rounds each | 174.098 ms | 172.228 ms | -1.08% |
| full-model prompt ingestion, 1289 tokens, four runs each | 516.2 tok/s | **527.3 tok/s** | +2.2% |

**Bit-identical, and measured rather than asserted.** Residual FNV-64 `9171727463267762618` in every
sample of every panel above - sixteen untraced, twelve traced, and both arms of the decode step. The
full-vocabulary control, `--prefill-identity` over a 384-token document at 128 and 256 rows a pass:

| arm | rows a pass | logit values | non-finite | logits FNV-64 |
|---|---:|---:|---:|---|
| 0 | 128, 256 | 190,709,760 | 0 | `2067497836734107792` |
| 7 | 128, 256 | 190,709,760 | 0 | `2067497836734107792` |

**381,419,520 compared logit values, one hash.**

## A constant the lane was missing: a dependent matrix chain is free

`bench/wmma_cost` measures 21.91 issue slots per `v_wmma_i32_16x16x16_iu8` over **four independent
accumulators**, and [the issue-order document](ffn-slice-issue-order.md) left the obvious question
about the other 33%: *what does a `v_wmma` cost beyond its issue slots when its operands are produced
two instructions earlier?* `mvw_rows` rotates through **two** accumulators, and its compiled block
carries eight `s_delay_alu` hints in front of matrix instructions with `VALU_DEP_1` and `VALU_DEP_2`.

`bench/wmma_chain` walks the chain count with everything else fixed. Slots per SIMD32:

| workgroups | waves/SIMD32 | 1 chain | 2 chains | 4 chains | 8 chains |
|---:|---:|---:|---:|---:|---:|
| 20 | 4 | 17.18 | 17.17 | 17.18 | 17.17 |
| 40 | 8 | 17.12 | 17.12 | 17.12 | 17.11 |
| **60** (this kernel's grid) | 12 | **17.14** | **17.12** | **17.10** | **17.10** |
| 80 | 16 | 17.09 | 17.09 | 17.15 | 17.08 |

**Flat to 0.2%.** A wholly serial chain of matrix instructions issues at the same rate as eight
independent ones, at every occupancy. The harness control says the instrument can see a chain when
there is one: `v_fmac_f32` costs **1.04** slots at one chain, **0.81** at two and **0.71** at four,
because an ordinary VALU needs independent neighbours to reach dual issue.

So nobody should split an accumulator in this engine to shorten a matrix chain, and every census in
this repository can keep converting `v_wmma` at its throughput constant. (The absolute scale here is
a nominal-clock reading like `wmma_cost`'s; the ratio across chain counts is the result.)

## Reproducing

```sh
# the register and block-loop census of any arm, three seconds, no GPU and no lock
tools/kernel_resources.py -D HALO_SLICE_PROBE=32 -D HALO_MVW_ABL=1 -D HALO_SLICE_PROBE_ARM=7 \
    kernels/halo_rows.hip k_ffn_slice
hipcc --offload-arch=gfx1151 -O3 -std=c++17 -Isrc -Ikernels -DHALO_SLICE_PROBE=32 \
    -DHALO_MVW_ABL=1 -DHALO_SLICE_PROBE_ARM=7 --cuda-device-only -S -o /tmp/sl7.s kernels/halo_rows.hip
tools/isa_loop_count.py /tmp/sl7.s k_ffn_slice

# the decomposition: the ablations need their own build and must never produce tokens
make -j12 DEFS='-DHALO_MVW_ABL=1' tools/batch_profile
HALO_FFN_GRID=1 tools/run-batch-compare --pin-clock --profile-tool --modes 20 --rows 128 \
    --heads 1 --traces 1 --rounds 3 --mvw-arm 0,2,3,4,5 --out .../arms-128.json
tools/arm_panel.py .../arms-128.json --key mvw_arm --phase ffn

# what ships, on an ordinary build
make -j12 tools/batch_profile
tools/run-batch-compare --pin-clock --profile-tool --modes 20 --rows 128 --heads 1 --traces 0 \
    --rounds 8 --mvw-arm 0,7 --out .../untraced-ab.json
HALO_MVW_ARM=0 tools/run-batch-compare --modes 20 --prefill-rows 128,256 --prefill-identity \
    --rounds 0 --out .../ident-arm0        # then HALO_MVW_ARM=7
tools/run-batch-compare --pin-clock --exec bench/wmma_chain --iters 3000
```

`HALO_MVW_ARM=0` restores the predecessor's drain in any build, which is the control every number
above was taken against.

## What this leaves

1. **The unit boundary is 12% and this took 3.9% of it.** What is left is the publish itself and the
   output write: a unit stores 4 kB and publishes 7 kB through LDS to do it, four times, and the
   store traffic alone is about 1.3 GB a pass. The shape that removes the publish entirely is a
   **wave that owns a whole tile's K range**, which needs no reduction at all - and it is not
   bit-identical, because the eight partial sums would become one `fmaf` chain over forty blocks.
   That is a reassociation with its own quality evidence to earn, and it is the one remaining
   structural change to this unit that is worth a turn.
2. **59% of the phase is counted issue and 61% of that is the matrix instruction**, so 36% of this
   phase is `v_wmma_i32_16x16x16_iu8` and nothing but a different numerical map moves it. `iu4` is
   exactly half the cost and splitting an eight-bit activation into nibbles is a wash, so that half
   belongs to the A4 route.
3. **The remaining 23% is the peel, the drain fold and the addressing**, and
   [instruction cuts there convert at about 0.22](mv-peel-gather.md) - 85 work instructions removed
   bought 1.66%. The asymmetry this document adds is that *added* scheduling slots convert at 3.5x:
   54 `s_delay_alu` cost 15%. **In this block loop, price a change by what it does to the register
   file and the dependence chain, not by its instruction count.**

# A generation-shape A4 FFN is priced in weight bytes, and a full block cursor cannot buy that back

Raw samples, the compiled loops and the losing arm's patch are in
[`batch-comparison/ffn-pair-stream/`](../../../data/bonsai2/batch-comparison/ffn-pair-stream/README.md).
**Nothing here ships in the runtime**; the source change this turn produced was a duplicate and is
not in the bundle.

1. **The pair-code arm's missing weight cursor is not why the pair image loses at a generation
   width, and no schedule of that phase is.** A block cursor that removes every in-block `vmcnt`
   wait - verified in the compiled loop, at no register cost - is **+3.5% of the phase**, which is
   its instruction cost and nothing else. Four weight images spanning 23% of byte count put the
   phase on one line: **`ffn` tracks weight bytes at about 107 GB/s**, while a 4.4x difference in
   operand work between two images that decode to the same weights is invisible. That prices the
   peel cut this phase is queued for, at this width, at about nothing.
2. **The activation stage that `08284b3` selected had never run**, because `create_ffn_batch`
   overwrote its default with an unset `env_int("HALO_FFN_BSTAGE")`. Two engineers found this within
   twenty minutes; [`ffn-block-pipeline.md`](ffn-block-pipeline.md) owns the repair and shipped it,
   and the panel here is the independent confirmation on a different instrument: **`ffn`/control
   -4.29% median paired over four rounds, four of four**, prompt widths under the deployed rule null
   to -2.8%, residual FNV-64 unchanged. It also names the one cell where the stage *loses*.
3. **And where the bytes live is not the answer either**: one allocation per image instead of one
   per matrix - a gate/up wave's two streams made virtually contiguous in read order - is a
   **null** (-0.31% and +0.63%). Every term a wave or an allocator controls is now ruled out for
   this phase.

## What the compiled loop said, and why the cursor looked free

The dense five-trit arm has had a block cursor since [`ffn_dense_stream.hpp`](../kernels/ffn_dense_stream.hpp);
the pair arm never did, because `DenseStream` is templated `ON = (WM == WM_DENSE)`. The compiled
pair block at two token tiles, `BSTAGE_WAVE`, issues in this order:

```
    4 x global_load_b128    the activation stage's fetch for block b+1
    2 x global_load_d16_b16 this block's two fp16 weight scales
    4 x global_load_b128    this block's 512 code bytes
    s_waitcnt vmcnt(5)      <- retires the staging loads, eleven instructions after the request
    s_waitcnt vmcnt(4)      <- and then this block's own weight bytes
```

gfx11 counts vector memory with one in-order counter, so a wait that exists only to read a two-byte
scale also retires the four staging loads issued ahead of it. The stage's documented claim - its
fetch for block b+1 is "issued at the top of block b" - is defeated inside the same block, and the
block then waits a second time for its own weight bytes. Two exposed round trips per block, on a
loop whose whole census is 298 instructions.

The cursor built for this (`kernels/ffn_pair_stream.hpp` in the patch) holds block b+1's four
`uint4` and both scale words in registers and orders a block as

```
    [this block's activation scales]     consumed by this block's drain
    [the stage's fetch for b+1]          consumed by this block's commit
    [the cursor's request for b+1]       consumed by the NEXT block
```

which an in-order counter can honour. The compiled loop does exactly that: the commit's wait is
`s_waitcnt vmcnt(6)`, leaving the cursor's four code loads and its scale word in flight across the
commit and across all 32 matrix instructions, and **there is no `vmcnt` wait at all before the
expansion** - the block reads registers. Both listings are beside the raw samples
(`pair-deployed.isa`, `pair-cursor.isa`).

**It costs no occupancy, because the scale broadcast paid for it.** Moving the sixteen `ds_bpermute`
that build `asr[8]`/`bsr[8]` from the top of the block to the drain that uses them frees sixteen
registers across the matrix section - more than the eighteen the cursor holds:

| `k_proj_opt<IU4, PAIR, ..., TT=2, WV=4, BSTAGE_WAVE>` | VGPR | waves/SIMD32 | spill | block loop |
|---|---:|---:|---:|---:|
| deployed | 200 | 7 | 0 | 298 instructions |
| stage fetch issued behind this block's loads | 190 | **8** | 0 | 304 |
| the block cursor | **195** | 7 | 0 | 327 (15 `v_mov`) |

## And it loses, by its instruction count

Mode 19, 32-stream generation step, traced, `ffn` normalised by the four phases no arm here can
reach (`gdn-resident-core`, `sequence-core`, `sequence-input-prep`, `head-projection`), pair image
forced at 32 rows, one process, `--pin-clock`:

| arm | `ffn` ms | `ffn`/control | against deployed |
|---|---:|---:|---:|
| deployed pair block | 38.939 | 0.7708 | — |
| stage fetch behind this block's loads | 40.298 | 0.7985 | **+3.6%** |
| the block cursor | 40.843 | 0.7974 | **+3.5%** |
| *the dense image in the same process, three cells* | 31.419 / 31.313 / 31.316 | 0.6191 / 0.6196 / 0.6190 | the panel's resolution: **0.1%** |

Residual FNV-64 `6678137683989106976` in every sample of every arm, which is the value this lane
published for this shape: the image, the schedule and the cursor are all bit-identical, as
constructed.

**+9.7% of the block's instructions bought +3.5% of the phase and recovered nothing.** A wave that
waits for no load it issued itself runs at the speed of a wave that waits for two per block. The
weight stream is not what this kernel waits for at this width.

## What it is: four byte counts on one line

The A4 module holds two images that decode to the same weights, and the image can be chosen per
stage, so a panel can walk the byte count with the arithmetic, the grid and the instruction mix
held fixed. Mode 19, 32-stream step, two samples per cell, stage off, one process:

| image | weight bytes | `ffn` ms | GB/s |
|---|---:|---:|---:|
| dense five-trit, both stages | 3.476 GB | 32.499 / 32.576 | 107.0 / 106.7 |
| dense gate/up + **pair down** | 3.743 GB | 36.275 / 36.312 | 103.2 / 103.1 |
| **pair gate/up** + dense down | 4.011 GB | 36.060 / 36.095 | 111.2 / 111.1 |
| pair codes, both stages | 4.278 GB | 39.851 / 40.093 | 107.4 / 106.7 |

**103 to 111 GB/s across a 23% span of byte count**, and the same 107 GB/s appears in the staged
arms, the cursor arms and the four arms of [the ablation ladder](ffn-b-operand.md). The two images
differ by 4.4x in operand work - the dense byte costs about 140 operations per 128 weights against
the pair code's 32 - and by 36% in block-loop instructions, and neither difference is visible in the
time. The phase is bytes at a fixed rate.

That rate is not the device's. `bench/wstream` reads *this image, this geometry, this lane
duplication, at this kernel's own seven waves per SIMD32* at **203.6 GB/s**
([`ffn-decode-schedule.md`](ffn-decode-schedule.md)), and `bench/bw` measures the roof at 242.

### What the law is, and what it is not

It is not a wall that nothing can move: two schedule changes have moved the *delivered rate* by a
few percent this evening - the activation stage by -4.3% and [the scalar
stream](ffn-block-pipeline.md) by -2.4%, both by removing waits rather than work. What the four
image points say is that the **byte count orders this phase and the operand work does not**, over a
range far wider than any of those changes:

- **The pair image cannot win at a generation width, and no schedule change to it can.** It reads
  23% more bytes at the same delivered rate, so it must cost about 23% more time; it measures 22.9%.
  The rule in `run_ffn_batch` - dense at 32 rows and below, pair above - is right for the reason the
  byte law gives, not the reason its comment gives.
- **A slot cut in this body is worth about nothing at this width, and the two images are the
  evidence.** The dense byte costs about 140 operations per 128 weights and the pair code about 32,
  a difference of roughly 250 issue slots on a block of 629 - **four times the peel gather's
  cut** - and it appears in the time only through the bytes. [`ffn-b-operand.md`](ffn-b-operand.md)
  queued the gather here "because this kernel runs at 100% of its issue model once its memory is
  free"; it does, and that is the wrong condition, because the memory is not free and the 24.5 ms
  issue floor sits under a 32.5 ms byte term. Take the gather to the 128-row shape, where the same
  phase is four token groups deep over the same bytes.
- **The exposed-latency explanation for the 107 is dead.** A block that waits for nothing it
  requested runs at the same rate as one that waits twice. Whatever separates 107 from `wstream`'s
  203 is not something a wave can prefetch its way out of - which is also why the three-deep
  `ScalarStream` win is worth what it is and not more.

## The stage default: the same defect, found twice, and the cell where the stage loses

`08284b3` gave `FfnBatch` the selected default `int b_stage = ffnb::BSTAGE_WAVE` and then, two lines
into `create_ffn_batch`, wrote `p->b_stage = env_int("HALO_FFN_BSTAGE")` - and `env_int` returns
zero for an unset variable. Every process that did not set the environment variable or pass
`--ffn-dense 16/17` therefore ran the body the change replaced. This panel found it by accident: a
default build reported `ffn_b_stage 0` in all eight samples. It went to the board at 21:22 and
[`ffn-block-pipeline.md`](ffn-block-pipeline.md) landed the repair - an `env_int(name, default)`
overload, which is the better fix because it repairs the shape rather than the instance - with its
own step-wall measurement of -1.46%, seven of eight rounds.

What this panel adds is a traced phase measurement of the same repair and the width sweep it needs,
because **the stage is not a win everywhere it is instantiated**.

**Mode 19, 32-stream generation step, four rounds, both arms in one process, paired within a round**
(`stage32b.json`):

| round | `ffn` ms, stage off -> wave stage | `ffn`/control | device span | aggregate tok/s |
|---|---:|---:|---:|---:|
| 0 | 32.475 -> 31.207 | 0.6467 -> 0.6179 (**-4.46%**) | 98.99 -> 98.15 | 323.3 -> 326.0 |
| 1 | 32.519 -> 31.471 | 0.6447 -> 0.6181 (**-4.13%**) | 99.29 -> 98.97 | 322.3 -> 323.3 |
| 2 | 32.841 -> 32.517 | 0.6456 -> 0.6238 (**-3.39%**) | 100.23 -> 101.71 | 319.3 -> 314.6 |
| 3 | 36.105 -> 33.377 | 0.6795 -> 0.6390 (**-5.96%**) | 107.51 -> 103.14 | 297.7 -> 310.3 |

Median paired: **`ffn`/control -4.29%**, four of four rounds, which reproduces the -4.06% the change
published. Device span -0.58% median, aggregate generation +0.59%; the step-level number is small
because this phase is a third of the step and the box moved 8% inside this panel. Residual FNV-64
`6678137683989106976` in all eight samples.

**Prompt widths under the deployed rule are a null, and one forced cell is not.** 32, 64 and 128-row
passes with `--ffn-image 0` (the rule): -0.5%/-2.8%, -0.1%/-0.4%, +0.1%/+0.8% - no regression, and
the 128-row cell is null by construction because the rule takes the pair image above 32 rows, where
four token tiles leave the stage uninstantiated. **Forcing the dense image at 128 rows** - a route
the engine never selects - runs the staged `TT = 2` body four token groups deep and costs
**+3.5%** (`ffn` 108.271 -> 112.691 and 107.480 -> 111.471, `prompt128.json`). The stage is a
request cut, and it pays where requests are what a block waits for, not where the same phase has
become four passes over the weight stream.

Single-stream decode, the eight-row verify pass and the served decode route do not reach this
kernel at all: they run the engine's own ternary FFN slice.

## Reproduce

Panels were taken on canonical `df44665` plus the arms in the patch, before
[`ffn-block-pipeline`](ffn-block-pipeline.md) landed its scalar stream; the byte law and the cursor
arms are ratios within their own process, so the base build shifts every cell together.

```sh
make -j8 bonsai-halo tools/batch_profile
# the stage default, both arms in one process
tools/run-batch-compare --pin-clock --profile-tool --out .../stage32b.json \
    --modes 19 --decode-streams 32 --heads 1 --traces 1 --rounds 4 --ffn-image 1 --ffn-dense 15,17
# the byte law
tools/run-batch-compare --pin-clock --profile-tool --out .../imagebytes.json \
    --modes 19 --decode-streams 32 --heads 1 --traces 1 --rounds 1 --ffn-image 1,2,3,4 --ffn-dense 15,20
# the prompt widths under the rule
tools/run-batch-compare --pin-clock --profile-tool --out .../prompt-widths.json \
    --modes 19 --rows 32,64 --heads 1 --traces 1 --rounds 2 --ffn-image 0 --ffn-dense 15,17
```

The cursor arms are `pair-cursor-arms.patch` beside the samples. They apply to `kernels/ffn_batch.hip`
and add `kernels/ffn_pair_stream.hpp`; `--ffn-dense 20,21,22` walks them once applied.

## Where 203 GB/s becomes 107: one more suspect ruled out

Everything a wave controls is now ruled out: the stride ([`ffn-image-order.md`](ffn-image-order.md)),
the lane duplication and the wave count ([`ffn-decode-schedule.md`](ffn-decode-schedule.md)), the
request count and the activation stream ([`ffn-b-operand.md`](ffn-b-operand.md)), the scalar waits
([`ffn-block-pipeline.md`](ffn-block-pipeline.md)), and the instruction mix and the block's own
memory latency (here).

So this turn also asked **where the bytes are**, because `create_ffn_batch` gives every matrix of
every image its own `hipMalloc` - `Arena::zeroed` is one allocation per call - and a gate/up wave
walks two of those buffers at once, 0.6 to 1.4 GB each, from two unrelated regions, where
`bench/wstream` walks one. `HALO_FFN_SLAB=1` puts an image's four matrices in **one** allocation in
the order a pass reads them, 4 kB aligned, same bytes and same order of use.

**It is a null.** Two processes, same panel, two rounds each, `ffn` normalised by the four phases
that cannot move:

| image | weight bytes | four buffers | one slab | |
|---|---:|---:|---:|---:|
| dense five-trit | 3.476 GB | 30.722 ms, 0.6074 | 30.576 ms, 0.6056 | **-0.31%** |
| pair codes | 4.278 GB | 37.974 ms, 0.7554 | 38.148 ms, 0.7602 | **+0.63%** |

113.1 -> 113.7 GB/s and 112.7 -> 112.1 GB/s, residual FNV-64 unchanged. (Both arms read faster here
than in the byte-law panel above - 113 against 107 - because the box was quieter; see
[`power-budget`](../orchestration/HANDOFF.md). Within a panel the law holds; across panels the
delivered rate follows the socket.) The arm is `image-slab-arm.patch` beside the samples and is not
in the runtime.

**Scope, so nobody over-reads it:** this moved four allocations to one, not four hundred, and it
moved *virtual* contiguity, which is all a process can ask for. The
[`weight-placement`](../orchestration/HANDOFF.md) probe on the HALO tensors is a different magnitude
on different kernels and this does not answer it. What it does say is that for *this* kernel, the
number of buffers a wave's two streams live in is not worth 0.5%, and the next explanation for the
gap has to be something neither the wave nor the allocator chooses: the delivered rate at the socket
(the power budget), or a property of `bench/wstream` that this kernel does not share. **The cheapest
next step is to point `wstream` at this kernel's real pointers and its real concurrent stream count
under the same power sampler**, rather than to build another schedule.

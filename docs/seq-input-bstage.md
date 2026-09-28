# Four waves of one workgroup fetch the same activation block, and fixing that loses

`sequence_shape` builds exactly two ownerships, and each makes a workgroup agree on a different
operand. Under `SHARE` the waves take one row tile and a token group each, so they agree on the
**weights**. Under per-wave tile ownership — which is what the input projection takes at every
width up to 64 rows, including a generation step — the waves take one token group and a row tile
each, so they agree on the **activations**: `first` comes from `blockIdx.y` alone, `bbase` is
`B + first`, and every `x[t]` address in the block loop is byte-identical in all four waves.

The kernel fetched it four times anyway: 64 `global_load_b128` per 128-K block for 4096 unique
bytes. A wave of the input stage reads 160 kB of activations per layer against 20 kB of weights,
and all of that reuse lives inside one workgroup.

Staging the block in LDS once per workgroup removes three quarters of those requests and leaves the
issue count alone. **It costs seven of sixteen wave slots and loses 21.6% of the phase.** Nothing
shipped; the implementation is kept beside the raw samples and is not in the tree.

## The change, and why the issue count is the interesting part

One buffer in LDS, one block of lookahead held in registers, two `__syncthreads()` a block:

    prime:   barrier; fetch block 0 -> registers; store; fetch block 1 -> registers; barrier
    block b: read the staged block ... ; barrier; store block b+1; barrier; fetch block b+2

The fetch for block `b+2` is issued a whole block before the store that consumes it, so the global
round trip is covered by a block of expansion and matrix work — the same trick
[the dense FFN weight cursor](ffn-dense-loads.md) uses. The bytes, the fragments, the expansion,
the matrix order and the per-block drain are the deployed ones. Only the path from memory to the
VGPR moves.

ISA census of the innermost block loop, one 128-K block of one wave, gfx1151, the instantiation a
32-row pass launches (`project<TT=2, WV=4, SHARE=false, RT=1, SB=1, LA=LANE>`):

| | slots | `global_load_b128` | `ds_load_b128` | `ds_store_b128` | `s_barrier` | scratch ops | VGPR | waves/SIMD32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deployed | 239 | 18 | 0 | 0 | 0 | 0 | 95 | **16** |
| staged | **237** | **4** | 16 | 2 | 2 | 0 | 157 | **9** |
| staged, `waves_per_eu(12)` | 290 | 4 | 16 | 2 | 2 | 10 | 120 | 12 |
| staged, `waves_per_eu(16)` | 331 | 4 | 16 | 2 | 2 | 31 | 96 | 16 |

The first two rows are the experiment. The staged arm issues **two fewer instructions** and
**4.5x fewer memory instructions** than the arm it replaces, at the same sixteen matrix
instructions. The only thing that moved against it is the register allocation.

## Measured

`--seq-bstage` orders the arms inside one process on one build, one warmed device and one clock.
Arm 0 is the deployed fetch; arms 1, 2 and 3 are the three rows above in order.

**A 32-stream generation step**, mode 19, 64-token prefix, two rounds, traced and untraced:

| phase | deployed | staged | change |
|---|---:|---:|---:|
| `sequence-input-projection` | 10.024 ms | **12.194** | **+21.6%** |
| `ffn` | 32.358 | 32.291 | -0.2% |
| `gdn-resident-core` | 44.575 | 44.552 | -0.1% |
| `sequence-output-projection` | 7.336 | 7.321 | -0.2% |
| `sequence-input-prep` | 1.565 | 1.567 | +0.1% |
| `sequence-core` | 1.361 | 1.361 | -0.0% |
| `head-projection` | 2.514 | 2.527 | +0.5% |
| device span | 100.732 | 102.823 | +2.1% |
| step wall, untraced | 99.681 | 101.732 | +2.1% |

Six phases the change cannot reach move by at most 0.5%, so the phase row is the change and not the
clock. Residual FNV-64 `8087160186626784132` in every sample — the value
[the dense FFN load schedule](ffn-dense-loads.md) and
[the score-unit schedule](attn-score-schedule.md) both published for a 32-stream decode step before
this code existed.

**A prefill pass at 32 and 128 rows**, three rounds, cases shuffled, medians of rounds 1-2:

| rows 32 | deployed | staged | at 12 waves | at 16 waves |
|---|---:|---:|---:|---:|
| `sequence-input-projection` | 11.449 | **14.390** | **18.807** | **28.805** |
| `ffn` (control) | 31.006 | 31.007 | 30.116 | 30.467 |
| `gdn-resident-core` (control) | 5.339 | 5.329 | 5.119 | 4.969 |
| `sequence-output-projection` (control) | 6.987 | 6.995 | 6.719 | 6.635 |
| device span | 59.088 | 62.010 | 64.909 | 74.896 |
| | | **+25.7%** | **+64.3%** | **+151.6%** |

**128 rows is the null this design predicted, and it measures as one.** There the shape is
`SHARE = true`, the waves hold different token groups, there is nothing to agree on and the staged
arm is never launched. Every phase in that column moves together — `ffn` -7.7%,
`gdn-resident-core` -8.0%, `sequence-core` -7.6%, `sequence-input-prep` -9.4%,
`sequence-input-projection` -7.8% — which is this box's clock across a walk and not a result.

Residual FNV-64 `1873717996769712938` at 32 rows and `7446865760224376151` at 128 in **every arm**,
both hashes this repository published before this code existed. The transformation is what it
claimed to be; it is simply slower.

## What this prices, and it is a number the lane did not have

The staged arm removes 78% of the block loop's memory instructions and two issue slots, keeps every
byte, every value and every accumulation order, and gives up seven wave slots out of sixteen. It
loses 21.6% of the phase. **On this kernel at this shape a wave slot is worth about 3% of the
phase, and the entire within-workgroup request redundancy is worth less than seven of them.**

That bounds a rule this lane has been extending. [The four-bit output
projection](sequence-output-a4.md) established that a kernel above its issue model waits on
absolute operand traffic, and halving the activation operand converted 1:1. **It does not follow
that the request count converts.** The a4 arm halved the bytes at every level *and cost no wave
slot*; staging removes requests that a workgroup's four waves were already coalescing in L0 and
pays a wave slot for it. The two are not the same lever, and the redundancy that looked like the
obvious waste is the cheapest part of this kernel's memory behaviour.

It also closes, from the other side, the question
[the row-tile pair arm](sequence-fragment-reuse.md) left open. `RT = 2` halves the same activation
traffic and lost 7.3%, and the reading then was that the extra expansion per (slice, row tile) paid
for it. This arm cuts the traffic *without* adding an expansion and without changing the wave
count's denominator, and still loses. The fragment stream is not what this kernel waits on.

## The register law, which is the reusable part

Writing this cost three failed attempts at the register allocation before the cause was found, and
the cause is general enough that the next engineer reaching for LDS in a block loop should read it
first.

**A `__syncthreads()` inside a block loop is a whole-memory fence, so every operand load in the
body has to have landed by the end of the block.** The compiler therefore issues all sixteen of
them at the top to overlap them, and sixteen live `global_load_b128` or `ds_load_b128` results are
**sixty-four VGPRs**. That single effect is the whole difference: this instantiation is 95 VGPR and
sixteen waves per SIMD32 deployed, and 157 and nine staged.

It was isolated by compiling the staged arm with the LDS *reads* removed and only the fetch, the
store and the barriers left — no LDS read anywhere in the kernel — which still reads **143 VGPR and
ten waves**. The barriers cost 48 of the 62 registers; the sixteen `ds_load_b128` cost the other
14.

Four things do **not** move it, and each was tried and measured:

- `__builtin_amdgcn_sched_barrier(0)` after every slice instead of every second one: 157, unchanged.
- `__builtin_amdgcn_sched_group_barrier` naming the pipeline explicitly, `DS_READ` then `MFMA` then
  `VALU` per slice: 157, unchanged.
- `__asm__ __volatile__("" ::: "memory")` after every slice — a full compiler memory barrier: 157,
  unchanged.
- An LDS-scoped fence in place of `__syncthreads()`,
  `__builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local")` around
  `__builtin_amdgcn_s_barrier()`, which emits `s_waitcnt lgkmcnt(0)` without `vmcnt(0)`: 157,
  unchanged.

None of them works because this is not the scheduler disobeying a hint. It is a correctness
constraint the barrier asked for, and no hint outranks it.

`__attribute__((amdgpu_waves_per_eu(N)))` *does* move it — 120 VGPR at twelve waves, 96 at sixteen
— and the table above is what that costs: the registers come back as **scratch inside the block
loop**, 10 and 31 memory instructions a block, and the arm that recovers the deployed occupancy
exactly is the slowest of the four. Buying occupancy from the allocator in a loop this wide is
buying it with private-memory traffic.

## Reproduce

The implementation, the `--seq-bstage` axis and the patch that applies them to `7e608ee` are in
[`batch-comparison/seq-input-bstage/`](../../../data/bonsai2/batch-comparison/seq-input-bstage/README.md)
with the two raw panels.

```sh
git apply ../../data/bonsai2/batch-comparison/seq-input-bstage/seq-input-bstage.patch
cp ../../data/bonsai2/batch-comparison/seq-input-bstage/seq_lds.hpp kernels/
make -j8 bonsai-halo tools/batch_profile
tools/run-batch-compare --pin-clock --profile-tool --modes 19 --decode-streams 32 \
    --decode-prompt 64 --seq-bstage 0,1 --rounds 2 \
    --out DATA/seq-input-bstage/decode-arms.json
tools/run-batch-compare --pin-clock --profile-tool --modes 19 --rows 32,128 --heads 0 \
    --traces 1 --seq-bstage 0,1,2,3 --rounds 3 \
    --out DATA/seq-input-bstage/prefill-arms.json
tools/kernel_resources.py kernels/sequence_batch.hip project
```

## What is still open here

The staged arm's own census says where the remaining headroom is, and it is not in the operand
path. A 32-row block is 239 issue slots of which 16 are `v_wmma_i32_16x16x16_iu8` at 32 cycles
each, so **512 of about 735 cycles a block are the matrix instruction**, and the phase measures
2.3x that model. The gap is neither issue nor requests; with sixteen waves resident and the request
count shown to be slack, the next question is what those waves are actually waiting on, and the
counter that would answer it directly — an L0 miss or request rate — is
[not available on this agent](sequence-output-projection.md).

The one lever this panel does point at: **wave slots convert here at about 3% of the phase each**,
which is the largest per-unit conversion rate this lane has measured on any kernel. Anything that
returns registers to `project` without adding scratch is worth more than anything that returns
memory instructions to it.

## The cheap version of this change, and it is also a null

Staging the block in LDS removes three quarters of the fragment requests and costs seven wave slots.
[The token map](seq-operand-coord.md) removes half of them for **no** wave slots, no LDS, no barrier
and two fewer issue slots — just by relabelling which token a (tile, lane) position carries, which
`v_wmma` leaves free — and the phase does not move (-0.35% normalised, five rounds, bit-identical).

So the request count was never the thing. Under the shared-row-tile ownership this document
contrasts with, the same map is a null; under the per-wave tile ownership it measures, it is
**+7.2%**. A wave's fetch is priced in the cache lines it touches, and both ownerships were already
at one lookup per line.

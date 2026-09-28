# Who owns a weight row tile in the batched FFN

The batched FFN spends a batch of 16-token tiles three ways: `TT` tiles held in one
wave's accumulators, `W` waves of a workgroup sharing one weight row tile, and
whatever is left, walked serially or split across workgroups at the price of reading
the weight stream again. `shape_for` in `kernels/ffn_batch.hip` picks `W` and `TT` per
projection stage.

The rule it used targeted 320 resident waves, four per SIMD32 on this device's 80
units. The integer maps cannot reach four at full width. The compiler's own numbers
for the IU4 pair-code kernel, `-Rpass-analysis=kernel-resource-usage`:

| token tiles per wave | VGPRs | waves/SIMD32 |
|---:|---:|---:|
| 4 | 249-252 | 5 |
| 2 | 145-147 | 9 |
| 1 | 109-112 | 12 |

So the target was set below what the narrower shape can actually hold, and the rule
never traded width for wave slots. Raising the target to 640 makes it trade, and only
where the stage has slots to gain: the down projection owns 160 row tiles, two
workgroups per processor, while gate/up owns 1088 and already fills the machine.

`SCHED_WIDE` is the existing adaptive rule against the new target. The A4 pair-code
arm and the scaled-FP16 arm above 32 rows take it. Nothing else moves: the same A
operand values come from the same codes, the same WMMA instructions accumulate in the
same order, the same block scales apply, and each output element is still owned end to
end by one wave.

## What it selects

At the 128-row prefill pass, `cap` being 4 token tiles for the integer maps and 8 for
scaled FP16:

| stage | row tiles | prior | selected |
|---|---:|---|---|
| gate/up, A4 | 1088 | tile ownership, 4 tiles | W = 2, 4 tiles |
| down, A4 | 160 | tile ownership, 4 tiles | **W = 4, 2 tiles** |
| gate/up, scaled A8 | 1088 | tile ownership, 8 tiles | unchanged |
| down, scaled A8 | 160 | W = 2, 4 tiles | **W = 4, 2 tiles** |

At 32 rows and below, A4 runs its dense five-trit arm and scaled A8 its own small-batch
arm. Neither is touched: one wave already covers the batch there, so sharing a tile
only idles wave slots. Measured below.

The two targets first differ again at 256 rows, which `forward_batch`'s 128-row cap
never reaches.

## Measured

Every arm below runs one build, `7b0d037`, binary `7d02abe8ca6fcb22`. Row-tile
ownership is selected per stage by `HALO_FFN_W_GU` / `HALO_FFN_W_DN`, which the process
reads once, so the two arms are two processes. To see machine drift between them, each
panel runs **mode 5 in the same process**: its FFN is `k_proj_int`, which this change
does not touch, so if the box drifts, mode 5 moves with it.

128 rows, two rounds, logits off, `tools/batch_profile`, whole-FFN region per pass:

| numerical map | schedule | FFN ms | control mode 5 | pass ms |
|---|---|---:|---:|---:|
| A4, mode 19 | prior | 98.12 | 143.50 | 197.15 |
| A4, mode 19 | **selected** | **84.86** | 143.57 | **186.41** |
| scaled A8, mode 18 | tile ownership | 167.69 | 142.70 | 268.09 |
| scaled A8, mode 18 | prior (W = 2) | 146.40 | 141.62 | 249.33 |
| scaled A8, mode 18 | **selected** | **142.38** | 141.65 | **244.90** |

The control moves 0.05% across the A4 pair and 0.02% across the scaled-A8 pair, so
those FFN differences are the change: 1.156x on A4 and 1.028x on scaled A8. The A4
pass is 5.4% shorter and the scaled-A8 pass 1.8%.

Full model, `tools/batch_compare --modes 0,19 --only prefill --prefill-rows 128`, a
384-token document in three 128-row passes with the head on the last pass, mode 0
interleaved **inside the same process** as its own control:

| schedule | prompt tok/s | rounds | mode 0 control |
|---|---:|---|---:|
| prior | 662.6 | 666.2, 659.1 | 155.3 |
| **selected** | **702.3** | 706.2, 698.3 | 155.4 |

Mode 0 moves 0.06% between the two processes. **+5.97% full-model prompt throughput.**

At 32 rows, where the code path is identical by construction, it measures identical:
A4 FFN 33.37 ms prior against 32.88 selected, controls 70.98 and 70.72.

Raw samples, per-round values, the `halo_env` each process saw and both executable
hashes are in
[`../../data/bonsai2/batch-comparison/ffn-schedule`](../../../data/bonsai2/batch-comparison/ffn-schedule),
with `summary.json` as the index.

## Numerically identical

The schedule changes which workgroup owns a row tile, nothing a lane computes. Every
panel above hashes all 128 x 5120 (or 32 x 5120) FP32 residual outputs of the pass with
FNV-1a and gets the same value in both arms:

| shape | residual FNV-1a, both arms |
|---|---|
| A4, 128 rows | 7446865760224376151 |
| A4, 32 rows | 1873717996769712938 |
| scaled A8, 128 rows | 5923795314807283696 |
| control mode 5, 128 rows | 11759156880888867700 |

The residual is the FFN's own output summed into, and the head reads it unchanged, so
equal residual bits at every row is equality of the whole forward computation.

## How the shape was found, and what that sweep cannot settle

The stage-by-stage sweep that located `W = 2` for gate/up and `W = 4` for down ran as
separate processes without an in-process control, at 128 rows on the pre-wide-prep
base, whole-FFN region:

| gate/up | down | FFN ms |
|---|---|---:|
| tile, 4 tiles | tile, 4 tiles | 97.1 |
| tile, 4 tiles | W = 4, 2 tiles | 88.7 |
| W = 4, 2 tiles | W = 2, 4 tiles | 92.8 |
| W = 4, 2 tiles | W = 4, 2 tiles | 85.2 |
| W = 2, 4 tiles | W = 2, 4 tiles | 91.2 |
| **W = 2, 4 tiles** | **W = 4, 2 tiles** | **82.0 / 83.1** |
| W = 2, 4 tiles | W = 4, 1 tile | 88.4 |
| W = 2, 2 tiles | W = 4, 2 tiles | 89.9 |
| tile, 2 tiles | tile, 2 tiles | 118.1 |

Two readings survive the controlled re-measurement. Narrowing a stage to two token
tiles only pays when the waves it buys share a row tile: at tile ownership it is the
worst cell in the table, because each wave then re-reads its own tile's whole weight
stream per token group. And the two stages want opposite things, which is why one
global target cannot serve both.

Cross-process steps in that table are inflated. A peer measured the middle step,
tile ownership to `W = 2` on A4, in one process with a runtime selector and randomised
case order: 205.6 ms to 202.1 ms, 1.76% of a pass, where the cross-process pair above
put it near 2.9%. Their raw file is
`../../data/bonsai2/batch-comparison/wide-prep-launch/a4-share-inproc.json`.
Treat the sweep as the search that found the shape and the controlled panels above as
its size.

Row counts between 33 and 127 are interpolation. They were not measured, and both arms
are correct at every row count.

## Operation

```sh
make -j8 bonsai-halo tools/batch_profile tools/batch_compare
# One bounded panel per command. Mode 5 is the in-process control; its FFN is untouched.
HALO_FFN_W_GU=1 HALO_FFN_W_DN=1 tools/run-batch-compare --profile-tool \
    --modes 19,5 --rows 128 --heads 0 --traces 1 --rounds 2 --warmup-ms 1200 \
    --context 128 --out ../../data/bonsai2/batch-comparison/ffn-schedule/prior.json
tools/run-batch-compare --profile-tool --modes 19,5 --rows 128 --heads 0 --traces 1 \
    --rounds 2 --warmup-ms 1200 --context 128 \
    --out ../../data/bonsai2/batch-comparison/ffn-schedule/selected.json
tools/run-batch-compare --tag ffn-sched-new --modes 0,19 --only prefill \
    --prefill-rows 128 --rounds 2
```

`HALO_FFN_W_GU` and `HALO_FFN_W_DN` force a stage's ownership: 1 is tile ownership,
2 or 4 share a row tile across that many waves, and 0 or unset takes the rule. They sit
beside the existing `HALO_FFN_TT_GU` and `HALO_FFN_TT_DN` width overrides, and neither
can reach a shape the accumulator budget forbids. Both are recorded in every run's
`halo_env`.

`HALO_FFN_A4_SCHED=0|1|2` pins mode 8's pair-code arm to tile, adapt or wide ownership
for a whole process, and `ffn_batch_set_a4_sched()` switches the same choice between
calls. Prefer `--ffn-sched` on either measurement tool, which drives the setter and
keeps every arm in one process.

## The same three shapes inside one process

The panels above pin ownership with `HALO_FFN_W_GU` / `HALO_FFN_W_DN`, which a process reads once,
so each arm is its own process and mode 5 or mode 0 carries the drift. `ffn_batch_set_a4_sched()`
switches mode 8's pair-code arm between calls instead, and `--ffn-sched` makes it a case axis in
`tools/batch_profile` and `tools/batch_compare` beside `--modes`. One process, one executable, the
case order reshuffled every round, no control needed because there is nothing to drift between.

That also reaches the middle shape, `SCHED_ADAPT`: the existing rule's sharing against the old
320-wave target. Separating it from `SCHED_WIDE` says how much of the gain is sharing a row tile at
all and how much is aiming the sharing at eight waves per SIMD32.

| mode 19, whole pass | TILE | ADAPT | WIDE |
|---|---:|---:|---:|
| 128 rows | 203.563 ms | 199.633 ms (+1.97%) | **193.884 ms (+4.99%)** |
| 32 rows | 72.647 ms | 72.804 ms | 71.628 ms |

So about two fifths of the 128-row gain is sharing the tile and three fifths is the wave target.
The 128-row groups do not overlap. The 32-row row is the null this section can offer: all three
arms run the dense five-trit kernel there, so their 1.4% spread is the panel's resolution at that
shape rather than an effect.

Full model with the ownership interleaved rather than controlled, `--modes 19 --ffn-sched 0,2`,
256-token document in 128-row passes, generation aggregated across 32 streams, three rounds:

| workload | TILE | WIDE | change |
|---|---:|---:|---:|
| prefill, 128 rows/pass | 672.1 tok/s | **720.4 tok/s** | **+7.19%** |
| generation, 32 streams | 268.7 tok/s | 268.9 tok/s | +0.07% |

Batched generation does not move, and the change should not be read as though it did. A 32-stream
step is 32 rows, which takes the dense arm. This is a prompt-processing change with a measured null
in the other batched outcome.

[Full-vocabulary comparison](../tools/batch-compare/results/ffn-sched-identity.json): 71,516,160
finite logits at 32, 40, 88 and 128 prefill rows, TILE against WIDE, zero differing bits and zero
non-finite pairs. That carries the residual equality above through the output head, which is a
separate kernel reading the residual, so the comparison covers the logits a caller actually sees.

[Panel detail, per-round samples and raw paths](../tools/batch-compare/results/ffn-sched.md).

## Installed

The schedule is `7b0d037`, its controlled panels `fc5bef5`, its runtime selector and three-way
ordering `3dc1784`. The canonical build at `fc5bef5`
(`7f1e2d63c55442603c3a0da7a22d2a4c0085bfb3e1c9e25655472bf0a3aa2037`) reproduced the result under
`tools/run-batch-compare`: 705.5 and 702.0 prompt tok/s on the 384-token document, in-process mode 0
control 155.6 and 155.4.

Serving never calls `prepare_batch`, so nothing this change touches runs in that path, and
single-stream `--bench` on the installed binary measured 33.65 then 33.83 tok/s across the two
installs. The resident server is on the current build and answering `/v1/models`.

## The gap is not wave slots

The stage sweep above holds a measurement nobody read at the time. Gate/up at `W = 4, TT = 2`
is nine waves per SIMD32 and measured 85.21 ms; at `W = 2, TT = 4` it is five waves and
measured 82.00 and 83.11. The higher-occupancy gate/up shape was the slower one, and the
sweep only looked like a wave-slot result because the *down* stage gains from wave slots and
the two stages were explained together.

[Wave slots are not what the A4 FFN is short of](ffn-occupancy.md) tests that directly.
Freeing 19 VGPRs of operand addressing takes gate/up from 252 VGPR and five waves to 233 and
six, and down from 147 and nine to 138 and ten, bit-identically. The stage does not get
faster: the `ffn` phase's share of its own pass moves from 0.85186 to 0.85670 over eight
rounds a build, and full-model prompt processing moves 690.1 to 696.7 tok/s, inside the
panel's resolution. Do not spend registers to buy waves on this stage. The change is not in
the tree; the document says where its patch and panels are.

## What is still open in this kernel

The selected A4 FFN issues about 5.35e8 IU4 WMMA instructions for a 64-layer 128-row
pass. At the 5.99 ns per WMMA per SIMD32 the [native rate
table](../../kelana/research/ffn/batched/arithmetic/README.md) achieved, those occupy
about 40 ms of the 84.9 ms the stage now takes. That gap is not explained here.

One piece of it has since been found and taken: the block's `sched_barrier(0)` was fitted at four
token tiles and was also firing on the down stage's two, where nothing spills and it only forbade
the lookahead. [Load lookahead in the pair-code block](ffn-pair-lookahead.md) keys it on the tile
count and takes the stage to 80.3 ms, bit-identically. The gate/up shape is untouched, so every
measurement in this document stands.

Remaining candidates, in the order their size suggests:

- The per-128-block integer drain. Each block's weight and activation scales force the
  int32 accumulators out to FP32: at four token tiles and two matrices that is 64
  conversions, 64 multiplies and 64 fused multiply-adds per 64 WMMA, about 25% of the
  block's issue budget before the operand build. The scaled-FP16 map has no drain at
  all and runs at 1.75x its own budget where A4 runs at 2.1x.
- Weight-load latency at each block boundary, which is what the extra wave slots just
  bought the down stage. Whether the remaining gap is the same thing on gate/up, where
  the register cliff blocks the same trade, is untested; software-pipelining the block's
  weight words needs about 16 VGPRs it does not have at four token tiles.

  Answered for the dense five-trit arm at two token tiles, where the registers are
  there: [holding two blocks in flight](decode-dispatch.md) costs 54 VGPRs and takes its
  down stage from 13.7 to 9.8 ms in a 32-stream step, 2.0x to 1.44x its block-loop issue
  model. Gate/up took 1.5% from the same cursor and one of its seven wave slots per
  SIMD32 for the registers, so the pipelining is worth what the stage's occupancy can
  spare and no more.

  Answered for the pair map as well, and the answer is no:
  [block lookahead on the pair arm](ffn-pair-block.md) costs 7.2% of the `ffn` phase at
  one block and 33% at two, at 128 rows, with the deployed arm and both depths
  interleaved in one process. It frees nine VGPRs and the shape's 28-byte spill and
  still loses, because 2176 waves at five or six resident per SIMD32 already cover the
  round trip the cursor removes. That document also prices slot removal in this kernel
  at about a third of its issue count, in both directions on one build.
  The same document fixes a serialized residual epilogue that costs every `k_proj_opt`
  down stage, this map and the pair map both, a memory round trip per output element.
- The B operand, which every wave of a workgroup loads separately even when the waves
  hold the same token group. Staging it through LDS trades global requests for
  `ds_read`, and was not tried.

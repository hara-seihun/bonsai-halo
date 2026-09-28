# Four-bit activations on the sequence input projection

The QKV and GDN input projection is the second largest phase of a prompt pass. It reads sixteen
signed activation bytes per token per K16 slice and multiplies them with
`v_wmma_i32_16x16x16_iu8`, whose matrix instruction is three quarters of its block loop. The
four-bit instruction on the same device retires the same 4096 MACs in half the cycles and wants
half the activation bytes, and ternary weights fit a nibble with room left over.

**Measured: `sequence-input-projection` 40.494 to 22.010 ms at 128 rows (1.84x), a 128-row pass
163.2 to 144.7 ms, and full-model prompt +11.0% at 256 rows and +11.4% at 128.** A 32-stream
generation step moves +1.4%. Teacher-forced negative log likelihood does not move: the four-bit
input projection is −0.0086 ± 0.0380 nats against the eight-bit one over 127 predictions, against
the +0.0415 ± 0.0228 the A4 FFN this engine already serves costs on the same instrument.

**It is the default.** [`docs/seq-a4-default.md`](seq-a4-default.md) owns the acceptance: the
widened instrument this document asked for puts its quality cost at +0.00770 ± 0.00798 nats against
the +0.01866 ± 0.00676 the A4 FFN already serves, over 1024 paired predictions in the generation
shape, and the error does not accumulate with horizon depth. `HALO_SEQ_QUANT=a8` is the control and
restores the previous default exactly.

## The coordinate, and why the operand got cheaper rather than more expensive

`v_wmma_i32_16x16x16_iu4` takes sixteen signed nibbles per lane in two dwords. The obvious way to
feed it is more storage: four bits per weight doubles the 1.30 GB input image and, as
[the vocabulary head measured](wide-head.md), buying an expansion with bytes stops paying the
moment a shape reads the image faster than it computes on it.

Nothing forced that. The stored word is 2.000 bits per weight and **which bit a code sits on is the
packer's free choice**, the same freedom [the eight-bit operand](../kernels/sequence_operands.hpp)
already spends on its byte lanes. A nibble's bottom two bits are exactly a code, so put the code of
K slot `k` at

    bit 4 * (k & 7) + 2 * (k >> 3)

and `w & 0x33333333` is the first operand dword's codes sitting in their nibbles already, with
`(w >> 2) & 0x33333333` the second's. What remains is a 2-to-4 bit sign extension, and the ternary
alphabet makes that three instructions instead of the six a general one costs: bit 1 of a nibble is
set exactly on code 3, the only negative weight, so ORing that bit back in at positions 2 and 3
turns 3 into `0xf` and leaves 0 and 1 alone.

Nine slots per K16 slice against the eight-bit operand's eleven, for the same 2.000 bits of
storage. The K order inside a 128-block is permuted relative to the eight-bit operand and that is
exact rather than approximate: a block accumulates in int32, and the prep writes the activation
operand in the same order.

### One instruction the compiler will take away from you

Written from a common `h = x & 0x22222222` as `x | (h << 1) | (h << 2)`, LLVM recognises
`(h << 1) | (h << 2)` as `h * 6` and emits `v_mul_lo_u32`, which is **quarter rate** on this
device: sixteen of them per block loop, three cycles each of pure loss. Taking the second mask from
the first result instead —

```c++
const unsigned a = x | ((x & 0x22222222u) << 1);
return a | ((a & 0x44444444u) << 1);
```

— keeps every step full rate. The census sees 311 instructions for the multiply form and 372 for
this one and prefers the wrong one; counted in cycles they are 359 and 340.
[The census split](activation-scale-axis.md) warns that scheduling slots are not work; this is the
other half of the same warning, that work slots are not all one cycle either.

## What the numbers say and what they do not

Block loop of the selected 128-row instantiation `project<4,false,true,2,true,1,1>`, `hipcc -S`,
gfx1151, counted with `tools/isa_loop_count.py`:

| | eight-bit | four-bit |
|---|---:|---:|
| block loop instructions | 374 (318 work) | 372 (326 work) |
| matrix | 32 x `v_wmma_i32_16x16x16_iu8` | 32 x `v_wmma_i32_16x16x16_iu4` |
| activation loads | 34 x `global_load_b128` | 32 x `global_load_b64` |
| A expansion per K16 slice | 11 (4 `v_perm_b32`) | 9 |
| cycle equivalents at 32 / 16 per matrix instruction | 1366 | 852 |

**The instruction count is the same and the time is not.** At the 32-row shape (`TT = 2`) the
four-bit block is ten instructions *longer*, 257 to 267, and 1.48x cheaper in cycles. Whatever a
census predicts here, it predicts from the instruction's width, not its count.

It does not win on registers either, at the width it is selected on. `tools/kernel_resources.py`,
the input arm's ownership, VGPRs and waves per SIMD32:

| `TT` | eight-bit | four-bit |
|---:|---|---|
| 1 | 78, 16 waves | 74, 16 |
| 2 | 95, 16 | 91, 16 |
| **4 (selected at 128 rows)** | **135, 10 waves** | **180, 8 waves** |
| 8 | 240, 6 | **213, 7** |

At `TT = 4` the four-bit arm spends 45 more registers and **two waves per SIMD32** — the compiler
buys pipelining with the registers the narrower operand freed — and still wins by 1.84x. At
`TT = 8` the sign flips: 27 registers back and a wave gained, where the eight-bit arm has been
stuck at six since it was written. That is the first concrete reason to re-fit `sequence_shape`
under this instruction rather than inherit a rule fitted for the other one.

The measured phase beat even that: 1.84x against the 1.60x the cycle model gives at `TT = 4`. The
missing quarter is the operand that does not appear in a slot count. A wave reads one whole token
tile's K row per weight row tile — [43 GB per 128-row pass](sequence-fragment-reuse.md), 671 MB per
recurrent layer — and four-bit activations halve every byte of it.

## Quality

The engine already ships four-bit activations in the FFN (mode 19) with published quality
evidence, but the FFN's activations feed a residual sum, while these feed attention scores and the
recurrent state. That is a different question and it needed its own answer.

`tools/batch_compare --seq-quant a8,a4e,a4`, mode 19, teacher-forced from a fixed 64-token context,
analysed by `tools/seq_quant_quality.py`:

| rows | arm | NLL | next-token top-1 | ΔNLL vs the exact engine |
|---|---|---:|---:|---:|
| 128 | mode 0 (exact FFN, eight-bit projections) | 1.71498 | 0.598 | — |
| 128 | mode 19, eight-bit projections | 1.75645 | 0.591 | +0.0415 ± 0.0228 |
| 128 | **mode 19, four-bit input projection** | **1.74785** | **0.591** | **+0.0329 ± 0.0286** |

Paired per prediction, within mode 19, four-bit against eight-bit:

| rows | predictions | ΔNLL (a4 − a8) | predictions improved |
|---|---:|---:|---:|
| 32 | 31 | −0.0074 ± 0.0982 | 18/31 |
| 128 | 127 | −0.0086 ± 0.0380 | 63/127 |

A null, with the point estimate on the favourable side at both widths, and one order of magnitude
inside the change the shipped A4 FFN already makes. Greedy agreement with the exact engine is
116/128 for the four-bit arm against 115/128 for the eight-bit one.

**Do not read the KL column as the ranking.** Against mode 0 the four-bit arm is KL 0.0448 where
the eight-bit arm is 0.0338, and it predicts slightly better. Every activation quantiser here
scales a 128-block by its own `amax`, so a perturbation large enough to move an `amax` produces a
fixed-size logit difference whatever its quality; [the state coordinate work](gdn-state-pack.md)
hit the same saturation and reached the same conclusion. NLL on the document is the number that
ranks arms.

The sample is one 128-token window of one document. It is the same instrument the A4 FFN and the
packed state were accepted on, and it is not enough to certify a serving default on its own — see
the open questions.

### Free-running, which is the instrument a teacher-forced window cannot replace

32 streams from distinct prompts, eight greedy steps each, every arm against the exact engine
(`--only multistep --modes 0,19 --seq-quant a8,a4 --multistep-streams 32 --multistep-steps 8`):

| arm | tokens equal to the exact engine | streams that diverge |
|---|---:|---:|
| mode 19, eight-bit projections (what this engine serves) | 221/256 | 7/32 |
| **mode 19, four-bit input projection** | **235/256** | **5/32** |
| mode 0, four-bit selected | 256/256 | 0/32 |

The four-bit arm tracks the exact engine *further* than the route it is added to. Greedy
continuations diverge chaotically and eight steps of 32 streams cannot resolve a small difference,
so read this as a second instrument failing to find a cost rather than as a gain. The mode-0 row is
the route guard again: that route never reaches the wide projection and its two arms are identical.

One process, one build, `53c8c97`.

## The kernel is proved against the map, not only against itself

`HALO_SEQ_QUANT` has three values, and the middle one exists to make the acceptance possible:

- `a8` — the deployed map, 127 levels, sixteen bytes, `iu8`.
- `a4e` — 7 levels, **still stored as bytes and still multiplied by `iu8`**. Every product, every
  int32 accumulator and every FP32 drain is what `a4` computes.
- `a4` — 7 levels in nibbles, `iu4`.

So `a4e` measures the numerical map with no new kernel, and afterwards it proves the kernel:

| shape | logits | a4e vs a4 |
|---|---:|---|
| prefill 32 rows, mode 0 | 7,946,240 | identical, 0 differing bits |
| prefill 32 rows, mode 19 | 7,946,240 | identical, 0 differing bits |
| prefill 128 rows, mode 0 | 31,784,960 | identical, 0 differing bits |
| prefill 128 rows, mode 19 | 31,784,960 | identical, 0 differing bits |

79,462,400 logit values, every bit equal. The mode-0 rows are also the route guard's own check:
mode 0 never reaches the wide sequence projection, and its three arms are identical to each other.

`tools/seq_a4_pack_check.py` checks the operand pair on the host in a second and needs no GPU: 4000
random K16 slices agree on every dot product across both storage orders, and both orders round-trip.

## Timing

`tools/run-batch-compare --pin-clock`, one process, arms interleaved and reshuffled per round,
1024-token document, three rounds.

| shape | eight-bit | four-bit | |
|---|---:|---:|---:|
| prompt, 128-row passes | 741.3 tok/s | 825.7 | **+11.4%** |
| prompt, 256-row passes | 767.8 | 852.4 | **+11.0%** |
| generation, 32 streams | 299.1 | 303.4 | +1.4% |
| generation, 8 streams (route not taken) | 119.8 | 118.7 | −0.9% |

The eight-stream row is the control: below 32 rows the engine leaves the wide sequence path
entirely, so that arm must not move, and it does not.

Device phases, traced in the same process, mode 19, 128 rows, the warm round:

| phase | a8 | a4 | |
|---|---:|---:|---:|
| `ffn` | 68.178 | 68.118 | −0.1% |
| **`sequence-input-projection`** | **40.494** | **22.010** | **−45.6%** |
| `sequence-output-projection` | 20.439 | 20.621 | +0.9% |
| `gdn-resident-core` | 15.227 | 14.840 | −2.5% |
| `head-projection` | 10.126 | 10.195 | +0.7% |
| `sequence-core` | 4.020 | 4.041 | +0.5% |
| `sequence-input-prep` | 3.323 | 3.434 | +3.3% |
| device span | 163.239 | 144.667 | −11.4% |

Four phases this change cannot reach agree within 0.9%, so the pair needs no normalisation, and the
span matches the full-model prompt figure.

### Why generation gets +1.4% and prompt gets +11%, measured rather than argued

The same trace on a 32-stream step, one token per stream:

| phase | a8 | a4 | |
|---|---:|---:|---:|
| `gdn-resident-core` | 44.603 | 44.417 | −0.4% |
| `ffn` | 32.955 | 32.978 | +0.1% |
| **`sequence-input-projection`** | **13.883** | **12.134** | **−12.6%** |
| `sequence-output-projection` | 11.568 | 11.507 | −0.5% |
| `head-projection` | 2.523 | 2.518 | −0.2% |
| `sequence-core` | 2.110 | 2.112 | +0.1% |
| `sequence-input-prep` | 1.583 | 1.503 | −5.1% |
| device span | 110.215 | 108.191 | −1.8% |

**The phase falls 1.84x at 128 rows and 1.14x at 32, on the same kernel and the same instruction.**
The cycle model says 1.48x at this shape, so the decode arm is not merely a smaller slice of a
bigger step — halving the matrix instruction's cost recovers a quarter of what it should, which
means **at 32 rows the input projection is not issue-bound at all.** At `TT = 2` a wave spends one
128-block of weights on two token tiles; the same diagnosis [the A4 FFN reached at its generation
shape](ffn-decode-shape.md) — instructions free, bytes cheap, the distance between a weight load
and its first matrix instruction expensive — applies to this kernel at this width, and it is what
[the weight-stream arm](sequence-projection-operands.md) attacks rather than this one. Neither
number is wrong. They are different machines.

## Reproducing

## Installed acceptance

Canonical `53c8c97`, built in the canonical tree with a clean worktree, measured by
`tools/batch_compare` `b8582901256100d9` under `tools/run-batch-compare --pin-clock`:
256-row prompt passes over a 768-token document, `a8`/`a4e`/`a4` interleaved, two rounds, paired
within each round: **+12.6% and +12.3%** for `a4` over `a8`. `a4e` is the control that separates
the two halves of the change — same numerical map as `a4`, same instruction as `a8` — and it reads
+5.4% and −10.4%, a noisy null, which is what it must be: **the gain is the instruction, not the
quantiser.** That panel ran while several engineers were compiling and its arms spread 11%, so the
paired ratios are the number and the absolute rates are not. `a4e` against `a4` on the installed
binary is 31,784,960 logits, every bit equal.

## Reproducing

```sh
# quality and the kernel's acceptance against its own map
tools/run-batch-compare --tag seq-input-a4-quality --only quality --modes 0,19 \
  --seq-quant a8,a4e,a4 --quality-rows 32,128 --quality-shapes prefill --slots 8
tools/seq_quant_quality.py ../../data/bonsai2/batch-comparison/seq-input-a4-quality

# full model, both pass widths
tools/run-batch-compare --pin-clock --tag seq-input-a4-prefill2 --only prefill --modes 19 \
  --seq-quant a8,a4 --prefill-rows 128,256 --prefill-tokens 1024 --context 1280 --rounds 3 --slots 8

# device phases
tools/run-batch-compare --pin-clock --profile-tool --modes 19 --rows 128 --heads 1 --traces 1 \
  --seq-quant 0,2 --rounds 2 --out .../seq-input-a4-phase/prefill.json
tools/run-batch-compare --pin-clock --profile-tool --modes 19 --decode-streams 32 \
  --decode-prompt 32 --seq-quant 0,2 --rounds 1 --warmup-ms 500 --out .../seq-input-a4-phase/decode.json
tools/phase_totals.py .../seq-input-a4-phase/prefill.json

# free-running continuations
tools/run-batch-compare --tag seq-input-a4-multistep --only multistep --modes 0,19 \
  --seq-quant a8,a4 --multistep-streams 32 --multistep-steps 8 --slots 32

python3 tools/seq_a4_pack_check.py          # host, no GPU
```

Raw: `../../data/bonsai2/batch-comparison/seq-input-a4-{quality,prefill,prefill2,decode,phase}/`.

## What it does not do, and what is open

- **It is reachable only through the wide input prep.** That prep is the quantiser; the per-slice
  route inside the persistent kernel has no four-bit store, so `sequence_project` refuses
  `HALO_SEQ_QUANT=a4` without `HALO_WIDE_PREP=1` and the direct layout rather than reading a
  mismatched operand as numbers. Single-stream decode and any pass under 32 rows are untouched by
  construction.
- **The output projection is still eight-bit.** It is now 14.3% of a 128-row pass and the largest
  phase after the FFN. Its input is the attention and SSM output rather than a normed residual, and
  its quantiser lives in three places (`ph_prep`, the wide attention combine, the resident GDN)
  rather than one, so it is a bigger change with its own quality question.
- **The shape rule was fitted for the eight-bit instruction.** `sequence_shape` picks `TT = 4` at
  128 rows partly on a register budget the four-bit operand relaxes: the B fragments halve, so
  eight token tiles may now fit where they did not. Nobody has swept `HALO_SEQUENCE_TT` under `a4`.
- **The quality sample wanted widening before this became a default. It was widened and it
  passed.** The horizon instrument now carries the precision axis:
  [`docs/seq-a4-default.md`](seq-a4-default.md). The prediction in this list was right about which
  instrument to use and wrong about which way it would read the risk — depth was the thing the
  127-prediction window could not see, and depth is where an activation quantiser turns out to be
  free, because the next token's operand is re-derived from the residual stream. A packed recurrent
  state has the opposite property, which is why the two axes needed the same instrument and
  different arguments.
- **The weight order follows the precision.** Switching `HALO_SEQ_QUANT` permutes the 1.30 GB input
  image in place, which is reversible and loses nothing, but it is about 20 ms: a measurement that
  walks the axis calls `sequence_batch_sync_quant` between arms, which both measurement tools do.

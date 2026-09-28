# Wide attention

`sequence-core` was the last phase of a wide pass still launched once per eight-row slice. A
128-row prompt pass made **256 cooperative launches** for its sixteen attention layers, and each of
them scanned that layer's whole KV cache for its own eight rows. The phase cost 14.7 ms of a
204.7 ms pass: 2.5 ms of it was dispatch, and the rest was sixteen passes over key and value data
that one pass could have read.

This module runs the same three phases once for the whole batch: rope and the cache write, then
scores and softmax over row groups, then the combine. A **row group** is what a slice handed the
phase — at most eight consecutive rows of one sequence — so the groups, their chunk boundaries,
their reduction order and their partials are the ones the per-slice path produced. Only the launch
shape changes.

## Why the row axis was the expensive one

`ph_attn`'s unit is `(sequence, q head, key chunk)`, and a unit loads `ACHUNK = 128` keys and
values for the `TT` rows it holds. Neither the unit count nor the traffic per unit depends on how
many rows the *pass* has, only on how many rows the *slice* has, so a 128-row pass paid the key and
value stream sixteen times:

| | units per launch | launches per layer | K+V read per layer |
|---|---:|---:|---:|
| per slice, 128 rows, 384-token context | 72 | 16 | 151 MB |
| wide, same pass | 1152 | 1 | 151 MB |
| wide, sixteen-row groups | 576 | 1 | 75 MB |

Two separate effects sit in that table.

- **Occupancy.** 72 units is less than half of a cooperative grid on this device, so the per-slice
  phase left most of the machine idle and paid a launch for the privilege sixteen times. One
  launch with 1152 units fills it.
- **Traffic.** Only a larger row group reduces the number of times a key chunk is read.
  `HALO_ATTN_TT=16` merges pairs of groups that are consecutive rows of the same sequence, which
  halves the stream at the cost of doubling the q registers a unit holds.

Both are bit-preserving. The group's rows, their per-row causal limit and their per-row reduction
order do not change with the group size: a row whose position falls inside a chunk sees exactly the
keys it saw before, and a row that shares a chunk with a later row simply has more masked entries,
which enter `l` as exact zeros and the value accumulation as `fma(0, v, acc)`.

## Chunk partials are what this costs

The deployed path keeps eight rows of partials with the compile-time `AMAX_CHUNKS` stride. A wide
pass needs `rows x heads x chunks`, and chunks grow with the context: 9.5 MB for a 128-row pass at
384 tokens, 101 MB at 4096, 811 MB at the full 32768. The module allocates against a byte budget
(`HALO_ATTN_PART_MB`, default 192) and reports how many row groups fit one launch, so a long
context splits into a few launches rather than demanding a gigabyte. The stride is the pass's own
chunk count, not `AMAX_CHUNKS`, so a short context pays for a short stride.

## Operation

The route is the default for any batched pass of more than eight rows on the direct wide sequence
layout. Ordinary serving never prepares a batch and is untouched.

```sh
make -j8 bonsai-halo tools/batch_compare tools/batch_profile
tools/run-batch-compare --tag wide-attn-on --modes 19 --only prefill --prefill-rows 32,128 --rounds 2
HALO_WIDE_ATTN=0 tools/run-batch-compare --tag wide-attn-off --modes 19 --only prefill \
  --prefill-rows 32,128 --rounds 2
```

- `HALO_WIDE_ATTN=0` selects the per-slice phase and allocates none of the module's memory.
- `HALO_ATTN_TT=8|16` sets the rows per unit group.
- `HALO_ATTN_PART_MB` sets the chunk-partial budget in megabytes.

## Results

Measured on gfx1151 under `tools/run-batch-compare`, one build, `HALO_WIDE_ATTN` the only
difference. Raw panels under `../../data/bonsai2/batch-comparison/wide-attn/`.

### The phase, and the pass around it

Four traced pairs, mode 19, logits on. `sequence-core` is the phase this changes; everything else
is an in-panel control, and the ratio of the untouched phases is how much the two processes differ
before the change is read at all.

| pair | `sequence-core` | normalised | device span | normalised | untouched |
|---|---|---:|---|---:|---:|
| 128 rows, clock pinned | 17.047 → 4.729 ms | −72.1% | 200.303 → 186.436 ms | **−6.52%** | ×0.9957 |
| 128 rows, free clock | 15.753 → 4.714 | −72.0% | 180.554 → 180.038 | −6.68% | ×1.0686 |
| 32-stream step, pair 2 | 5.384 → 3.243 | −39.9% | 119.620 → 117.681 | **−1.90%** | ×1.0029 |
| 32-stream step, pair 1 | 5.395 → 3.488 | −38.6% | 119.221 → 123.220 | −1.85% | ×1.0530 |

The launch count per pass goes 256 → 16 at 128 rows and 64 → 16 at 32 streams. The residual
FNV-1a is `7446865760224376151` in every 128-row arm and equal across every decode arm.

Two of those four pairs have honest controls (untouched phases within 0.43%) and two do not, and
they agree to a tenth of a percent once the control ratio is divided out. **Read the control before
the result**: the free-clock 128-row pair looks like an exact null on the span, because in that
pair every phase the change cannot reach — `ffn`, `sequence-input-projection`,
`gdn-resident-core`, `head-projection` — was 6.9% slower on one side. A pair of processes on this
APU drifts by more than this change is worth.

### Full model

384-token document in 128-row passes, logits on the last pass, three rounds, `--pin-clock`, mode 0
interleaved in the same process as the control (mode 0 never prepares a batch and cannot reach this
code):

| arm | prompt tok/s | mode 0 control |
|---|---:|---:|
| per-slice attention | 765.2, 764.1, 767.6 | 157.3, 157.5, 158.2 |
| **wide attention** | **815.8, 804.9, 805.5** | 154.5, 158.2, 157.9 |

**+5.3%** on the median, with the control inside 0.3% and the two groups of samples not
overlapping. Raw: `full-on/`, `full-off/`.

### Acceptance

[Bit identity](../tools/batch-compare/results/wide-attn-identity-a4.json): 71,516,160 finite
full-vocabulary logits at 32, 40, 88 and 128 rows, **zero differing bits**, no non-finite values,
`max_abs_diff` 0.0 — the wide route against the per-slice control on one executable. The residual
FNV-1a agrees in every timed panel as well.

### Deployment

Canonical `7cb04fd`, installed as SHA-256
`75a468ebac9e9ec3a4687fd189da0a7ebf936c993ceed7c80c8c0f26784ec332`. That binary carries peer work
landed after the panels above, so its absolute rates are higher; the arms were re-run on it:

| arm | prompt tok/s | mode 0 control |
|---|---:|---:|
| per-slice attention | 783.6, 800.7 | 157.7, 158.4 |
| **wide attention** | **835.5, 836.7, 833.7** | 158.9, 158.3, 158.3 |

**+5.5%**, control inside 0.4%. [The installed binary reproduced all 71,516,160 logit
bits](../tools/batch-compare/results/wide-attn-installed-a4.json) at 32, 40, 88 and 128 rows
against its own per-slice control. Single-stream `--bench` gave 33.55 tok/s and 29.81 ms/token,
unchanged and untouched by construction: single-stream decode never prepares a batch, so
`batch_attn` is null and the persistent kernel runs its own attention phase. The resident server
came back on the new binary.

### What is left in it

- **The partials budget is sized in whole groups, not rows.** A 32-stream step has thirty-two
  one-row groups, so the budget reserves eight rows for each of them and splits the layer into two
  launches where one would do. Prefill groups are full, so this only costs the decode shape.
Both of those questions are answered in
[the long-context measurement](long-context-attention.md), and the answers are not what this
section expected.

- **Sixteen-row groups lose by 18%.** `HALO_ATTN_TT=16` is bit-identical by the argument above and
  it does halve the key stream. It also takes `k_attn_wide` from 92 VGPR and sixteen waves per
  SIMD32 to 180 and eight, and the wave slots are worth more: the score units are +18.7% and +18.3%
  at two positions, normalised on untouched phases. The q registers took the wave slot exactly as
  feared, and the reason it is not recoverable is that this kernel is not key-bandwidth-bound — it
  moves 3.1 GB of the 19.3 GB it requests, 76 GB/s against a 242 GB/s roof. `--attn-tt 8,16` walks
  both widths in one process; two processes read this backwards, because the raw wall time favours
  TT=16 by 8.5% and all of that is the clock ramp.
- **Long prompts were unmeasured, and this phase is the whole growth term.** 0.0166 ms per token of
  position per 128-row pass, every other phase flat to 1%. That is 1.7% of a 384-token prefill and
  28% of an 8192-token one, and inside a single pass it passes the whole A4 FFN at position 4500.
  `tools/batch_profile --prefill-scan N` is the axis.
- **The capacity bound was 128 while the default pass became 256**, so this module was not running
  at all in the engine's default prompt shape. Fixed in `012e7a0`; the bound tracks `PASSMAX` now.

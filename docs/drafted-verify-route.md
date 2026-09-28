# The drafted step's shape is right, and the context it runs at is what costs it

The resident service generates with the DFlash2 drafter: one 8-row verify pass of the target plus
one drafter pass, per step. Three things about that step had never been measured — which route the
verify pass should take, how much a row of it costs, and what any of it does at a context longer
than a paragraph. This iteration measures all three. Two of them close as negatives. The third
found that **drafted generation falls from 68 to 24 tok/s by 11k tokens of context, and 60% of the
drafter is attention over keys its own 2048-token window excludes.**

Raw: [`batch-comparison/drafted-verify-route/`](../../../data/bonsai2/batch-comparison/drafted-verify-route/).

## 1. The batch route loses at the verify shape (bounded negative)

`generate_dflash` calls `Engine::forward` — mode 0, the eight-row persistent kernel. Every route
comparison this lane owns was taken at sixteen rows or more, where mode 0 already pays two weight
streams; at eight rows both routes stream the weights exactly once, so the comparison is the FFN
body and the head alone. `tools/batch_profile` could not measure it because it refused mode 0; it
takes it now (`--decode-streams 1 --decode-tokens 8 --modes 0,4`).

One sequence, eight tokens, logits on every row, one process, `--pin-clock`, two rounds:

| route | step | | phases (traced) |
|---|---:|---:|---|
| **mode 0**, persistent | **43.75, 43.51 ms** | | one span, 43.64 ms |
| mode 4, sliced + batch FFN + wide head | 46.00, 44.90 ms | +4.0% | `ffn` 25.79, `sequence` 18.07, `head` 1.42 |

**Mode 4 is 4 to 6% slower and its FFN is not faster.** `k_ffn_slice<8>` in its own launch reads the
same 3.74 GB in 25.8 ms where the dot4 FFN inside the persistent kernel reads it in about 26.4; the
64 extra launches and the per-slice sequence stages cost the difference. That is the answer to the
question two engineers left from the other side — *the matvec wants to be its own kernel* — at this
width: taking the FFN out of the persistent kernel buys nothing that its launches do not spend.

## 2. Every drafted row pays for itself, and RMAX is what caps the depth (bounded negative)

The cost of a row of the deployed verify pass, same panel, medians of two rounds:

| rows | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| step, ms | 31.11 | 32.75 | 34.44 | 39.00 | 40.45 | 41.48 | 43.10 | 43.31 |

A verify pass is **31 ms of fixed weight stream and 12.2 ms spread over seven rows**, and the last
row costs 0.21 ms. The acceptance profile against it, measured over 94 steps of ten prompts on the
installed binary (`--bench --dflash --prompts bench/drafter-prompts.txt`, `HALO_SPEC_DEBUG=1`):

| j | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `P(accept >= j)` | 0.777 | 0.585 | 0.447 | 0.319 | 0.181 | 0.149 | 0.085 |

3.543 tokens a step, 36.3% of drafts accepted, 6.39 ms of drafter and 44.5 ms of verify. **The tail
is fat** — 8.5% of steps accept all seven — because real text has runs that a drafter gets right.
Priced against the row costs, with a token worth 14.0 ms:

| drafts used | 1 | 2 | 3 | 4 | 5 | 6 | **7** |
|---|---:|---:|---:|---:|---:|---:|---:|
| tok/s | 45.4 | 57.8 | 61.9 | 66.8 | 69.1 | 69.9 | **71.3** |

**Seven is optimal and the seventh row returns 5.7x its own cost.** So the depth is capped by
`RMAX = 8`, not by economics — and widening past it is worse, because a ninth row is a second
weight stream (+31 ms) for a draft worth about 0.06 tokens.

**Adaptive draft width is dead at this cost curve, and that is worth writing down** because it is
the obvious next idea. An oracle that knew the accepted count in advance and paid only `C(a+1)`
would be +16.4%. Conditioning on the previous step's outcome — lag-1 correlation `r = 0.325`, real
signal — captures **+0.6%**: seven is the best width in *every* history bucket. Cutting the seventh
row saves 0.21 ms and risks `q_7 x 14 ms = 1.19 ms`, so a predictor would have to be better than
98.5% certain before the trade turns. Anything in this family needs a cost curve steeper than this
one, not a better predictor.

## 3. What long context does to a drafted step, and the defect inside it

Nobody had generated at a context longer than a few hundred tokens. The service advertises 32768.
`--bench --dflash --raw --prompts`, one prompt per length, `--pin-clock`, 32 tokens:

| prompt | drafted tok/s | tokens/step | draft ms | verify ms |
|---:|---:|---:|---:|---:|
| ~64 | 69.6 | 3.54 | 6.39 | 44.5 |
| 2891 | 68.4 | 4.00 | 9.21 | 49.2 |
| **11248** | **24.3** | 2.25 | **17.88** | **64.4** |

Both halves grow, and the drafter grows faster than the target it is meant to be cheap against.
`HALO_PROFILE_DRAFT=1` at 11248 tokens says why:

| drafter segment | ms/step | share |
|---|---:|---:|
| **`attn`** | **11.640** | **60.0%** |
| `mv-gate-up` | 2.259 | 11.6% |
| `mv-lm-head` | 1.578 | 8.1% |
| `prep-attn-combine` | 0.746 | 3.8% |

At position 64 that phase is 0.145 ms and 1.9% of the drafter. **The drafter is a five-layer model
with a 2048-key sliding window** (`DF_WINDOW`), so its attention should not grow past 2048 at all.
It grows because `attn_window` was only ever a *mask*: `ph_attn` deals one unit per key chunk from
position 0, and each unit reads its keys, computes every dot product and then multiplies the whole
chunk by zero. At 11248 tokens that is 88 chunks a head to keep 17 — **81% of the drafter's
attention is provably zero-contribution work**, and it scales with context, not with the window.

### The fix, what it measured, and why it is not in main

Make the window decide the unit list as well as the mask: no unit for a chunk every key of which the
window excludes, and the fold skips the partials those units no longer write. Both sides read one
function, `attn_chunk0(pos, window)`, and a window of zero returns zero, so every target-model pass
keeps its schedule and its bits.

Built and measured (`11248`-token prompt, `-n 8`, `HALO_PROFILE_DRAFT=1`), against the canonical
binary on the same prompt:

| | drafter `attn` | `prep-attn-combine` | drafter total | draft ms/step | verify ms/step |
|---|---:|---:|---:|---:|---:|
| deployed | 11.640 | 0.746 | 19.397 | 13.033 | 65.517 |
| **windowed units** | **2.414** | **0.162** | **9.401** | **6.389** | 64.776 |

**-79% of the drafter's attention, -51% of the drafter, and the target is untouched.** Identity:
greedy digest `236184251123154896` and 21 drafted / 5 accepted in both arms. That digest is the
weaker half of the check — a lossless drafter cannot move the output stream whatever it proposes —
so the *accepted count* is the evidence that the drafter's own arithmetic did not move, and it did
not.

**It is not in main because a later revision of the same change faults the device.** Adding the
control arm (`attn_window_deal`, so both schedules live in one binary) produced a build that takes a
`Memory access fault ... Page not present` during prompt ingestion on **both** arms — including
`HALO_ATTN_DEAL=0`, which restores the deployed unit list exactly — while the canonical binary runs
the same prompt and the same context cleanly. Two candidate causes were tested and excluded: a stale
`FwdParams` layout (every linked object was rebuilt after the field was added) and the per-group
minimum that the first control revision walked over `P.rows` (removed; the fault survives). The
measured version above, which has neither the field nor the guard, ran clean on the same workload
at two lengths. The work is on `agent/drafted-verify-route`; the repro is
`./bonsai-halo --bench --context 16384 --dflash DRAFTER --raw --prompts long-11k.txt -n 8`.

**The finding does not depend on the patch.** The waste is arithmetic: 88 chunks read to keep 17,
five layers, every step, growing linearly with context on the one shape the product serves.

> **Finished and landed.** [`docs/drafter-window-units.md`](drafter-window-units.md) carries this to
> the end: rebuilt on `89633a2` the same design does not fault, with both arms in one binary and
> `--attn-deal 0|1` walking them in one process. Measured -37.1% of the drafter at 7371 tokens and
> -51.1% at 11248, +7.6% / +12.2% of drafted generation, a null below the window, the verify pass
> and prompt ingestion unmoved. The likely cause of the fault was a build-system defect and is fixed
> at that layer: `make` could not see a changed `-D` set and linked objects compiled under two
> different ones. **One claim in this document is withdrawn there**: the accepted count is not an
> identity witness for the drafter's arithmetic. Four of the drafter's matvecs K-split and
> `ph_matvec` drains those with `atomicAdd`, so its float state is not reproducible between two runs
> of one binary - the same arm differs from itself by up to 7.8e-3 in the block hidden state, and
> adjacent cells of one process gave 63 drafted / 14 accepted and then 70 / 13.

## What this leaves

1. **Finish the skip.** The value is measured and large on the deployed serving path. What it needs
   is the fault isolated — build the measured revision first (no `attn_window_deal`, no guard),
   confirm it clean, then add the control arm one field at a time.
2. **The target's own attention at long context is the bigger half and nobody holds it.** The verify
   pass goes 44.5 to 64.4 ms between position 64 and 11248. The target has no window, so none of
   this applies to it: that is 20 ms of real key traffic and it is what a long-context serving
   number is made of.
3. **Acceptance falls with context** — 3.54 tokens a step at 64, 2.25 at 11248 — and nobody knows
   whether that is the drafter losing its window, the text, or both. It is worth separating, because
   at 11k a token is worth 36.6 ms and every point of acceptance is worth more than at position 64.

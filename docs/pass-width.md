# Rows per pass

`forward_batch` refused more than 128 rows. It now takes 256, prompt ingestion defaults to that
width, and a generation step can carry 256 rows. The change is bit-identical and its prefill gain is
small; the useful part is what the measurement says about *why* it is small, because the lane has
been pricing a per-pass fixed cost that is four times larger than the real one.

Source `7408b84`. Held: `kernels/halo_kernels.h` (the `PASSMAX` line), `kernels/head_batch.hip`,
`kernels/prep_batch.hip`, `src/main.cpp`, both measurement tools. The `PASSMAX` constant itself was
put there by the wide-drafter engineer so `CAPMAX` could follow it.

## What stopped a pass at 128 rows

Four guards, and two of them were bounding rows by the number of independent sequence states:

| site | test | what it meant |
|---|---|---|
| `src/batch.cpp` | `max_rows > PASSMAX`, `total > PASSMAX` | the real cap |
| `kernels/sequence_batch.hip` | `capacity > 128` | a literal |
| `kernels/halo_rows.hip` | `a.nrows > 128` in `launch_gdn_resident` | a literal, and it `abort()`s |
| `kernels/head_batch.hip` | `capacity > MAXSLOTS` | slots, not rows |
| `kernels/prep_batch.hip` | `a.rows > MAXSLOTS` | slots, not rows |

The head one was the dangerous shape. `create_head_batch` returns `nullptr` above its bound and
`forward_batch` then falls back to the per-slice head instead of failing, so a wider pass would have
quietly lost the wide head and reported a regression that was not the width. Check
`head_batch_capacity()` before believing any pass-width number.

Nothing narrower changes. Every `npad` is computed from the rows in the pass, not from the prepared
capacity — `sequence_project` and `run_mode` both do `(total + 15) / 16 * 16` — so a larger capacity
costs device bytes and nothing else. About 147 MB at 256 rows: 127 MB of it is the head's logit
buffer at `VOCAB` floats per row, 15 MB the FFN activation workspace, the rest the residual buffer
and the sequence operand. Only a process that arms wide ingestion pays it; `prepare_batch` is not
called otherwise.

## The per-pass fixed cost is 9.1 ms, not 37

Traced, mode 19, one build, one process, two rounds, logits on every row. Raw:
`batch-comparison/pass-width/phase-m19-128-256.json`.

| phase | 128 rows | 256 rows | 2 x 128 | saved | ms/row at 128 | ms/row at 256 |
|---|---:|---:|---:|---:|---:|---:|
| device span | 181.489 | 353.850 | 362.979 | **9.129** | 1.4179 | 1.3822 |
| `ffn` | 77.461 | 155.410 | 154.923 | -0.487 | 0.6052 | 0.6071 |
| `sequence-input-projection` | 39.514 | 73.352 | 79.028 | 5.676 | 0.3087 | 0.2865 |
| `sequence-output-projection` | 16.337 | 30.239 | 32.674 | 2.436 | 0.1276 | 0.1181 |
| `gdn-resident-core` | 16.088 | 31.619 | 32.176 | 0.557 | 0.1257 | 0.1235 |
| `sequence-core` | 15.836 | 32.638 | 31.673 | -0.965 | 0.1237 | 0.1275 |
| `head-projection` | 10.472 | 21.031 | 20.945 | -0.086 | 0.0818 | 0.0822 |
| `sequence-input-prep` | 3.605 | 6.094 | 7.209 | 1.116 | 0.0282 | 0.0238 |

A doubled pass pays each phase's fixed cost once instead of twice, so the `saved` column *is* the
per-pass fixed cost of each phase, measured directly rather than extrapolated.

**The FFN has none.** 0.6052 ms/row at 128 rows and 0.6071 at 256, and it re-streams its whole
4.28 GB weight image in both. That stream runs at 55 GB/s at 128 rows and 27 GB/s at 256 against the
242 GB/s `bench/bw` measures, so re-reading the image was never costing anything: the stage is
issue-bound and issue work is per row. The `11.5 ms fixed + 0.669 ms/row` fit that has been quoted
for this kernel came from the 32-row and 128-row endpoints, and the FFN **changes weight image**
between them — dense five-trit at 32 rows, pair codes at 128. Fitting a line across an arm boundary
invented a fixed cost that does not exist. A change to `k_proj_opt` justified as "it saves a weight
re-read" is justified against nothing.

What does amortize is the three stages nearer their bandwidth: the two sequence projections
(5.68 and 2.44 ms per pass) and the wide input prep (1.12).

`sequence-core` is the one row that is not a saving. The profile runs one pass after a fixed
128-token context, so a 256-row pass covers positions 128..383 where a 128-row pass covers
128..255, and attention work per row grows with position. It is the only position-dependent phase in
the table; everything else is flat, which is why the rest of the column is readable at all.

## The document is 1.7% faster and every bit is the same

Full model, 384-token document, mode 19, both widths interleaved in one process across two panels
and five paired rounds (`batch-comparison/pass-width/doc384-m19`, `.../identity-m19`):

| round | 128 rows/pass | 256 rows/pass | change |
|---|---:|---:|---:|
| A0 | 694.5 | 706.2 | +1.7% |
| A1 | 739.9 | 782.8 | +5.8% |
| A2 | 768.0 | 780.3 | +1.6% |
| B0 | 698.9 | 692.6 | -0.9% |
| B1 | 768.2 | 769.8 | +0.2% |

Mean +1.7%, four of five favourable. That is what the phase table predicts and the panel cannot
resolve it on its own: a 384-token document is three passes at 128 rows and two at 256, so it saves
exactly one pass's 9.1 ms out of 543, and the box's own spread across these rounds is 10%. **The
device-span measurement is the evidence here; the throughput panel only rules out a regression.**
The asymptote is 9.1 / 181.8 = 5.0% of prompt processing, and 256-row passes take half of it.

`--prefill-identity` is the acceptance, and it is new: it walks the `--prefill-rows` widths outside
every timed region, asks for logits on *every* pass rather than the last, and hashes all of them.

```
identity mode 19 rows 128: 95354880 logits in 3 passes, MATCH nonfinite 0, fnv 17618767336181543518
identity mode 19 rows 256: 95354880 logits in 2 passes, MATCH nonfinite 0, fnv 17618767336181543518
```

95,354,880 full-vocabulary logits — every token of the document, not the last row of the last pass —
identical between the two widths. That is the contract a width change has to meet: the quantisers
are per row, the FFN arms are published bit-identical across widths, and a row's attention reads the
same keys whether the earlier rows arrived in this pass or the previous one.

## The larger consequence is in the generation step

`--decode-streams 32 --decode-tokens 8` is 256 rows and was not a legal shape before this. It
extends the step-width table past the K=4 wall the lane stopped at. One process, mode 19, 32
streams, both cases interleaved; K=4 reproduces the documented 555.6 tok/s exactly, which makes it
the in-panel control. Raw: `batch-comparison/pass-width/decode-k4-k8.json`.

| K | rows | step, untraced | aggregate tok/s at full acceptance |
|---:|---:|---:|---:|
| 1 | 32 | 121.074 ms | 264.3 |
| 2 | 64 | 181.352 ms | 352.9 |
| 4 | 128 | 230.240 ms | 555.9 |
| **8** | **256** | **373.762 ms** | **684.9** |

+23.2% on the ceiling over K=4. The mechanism is the same one the decode map found and it holds at
the new width: `gdn-resident-core` goes 53.207 to 63.268 ms for twice the rows, 2.52 ms per extra
token per sequence against 1.347 ms per extra *stream*, because the state round trip is indexed by
(sequence, layer) and a token rides along. `sequence-output-projection` also amortizes hard here,
24.570 to 31.356 ms.

**This is a step cost, not a shipped throughput.** Nothing drafts eight tokens. What it changes is
the arithmetic a batched drafter has to clear, and it does not obviously favour K=8:

- K=8 beats one-token generation when a sequence lands **3.09 accepted tokens per step** (38.6% of
  eight), against 1.90 at K=4 and 1.50 at K=2.
- K=8 beats **K=4** only when it accepts `1.623 x` what K=4 accepts. A drafter landing 3.6 at K=4
  has to land 5.84 at K=8 to be worth the wider step.

So the decision for a batched drafter is now priced at both widths, and it turns on whether
acceptance scales with draft depth — which nobody here has measured. DFlash2's 3.6-4.5 tokens per
step is a *single-stream* figure at its own depth.

### Tokens against streams at constant rows, which the handoff had open

"Tokens and streams were never separated at constant rows" has stood as the decode map's unfinished
question. A 256-row pass makes it a 2x2 instead of a single cell. Two processes (stream count is
fixed at engine load), mode 19, one round, both token widths interleaved inside each process. Raw:
`batch-comparison/pass-width/decode-k4-k8.json` and `.../decode-s16.json`.

| | 32 x 4 | 16 x 8 | 32 x 8 | 16 x 16 |
|---|---:|---:|---:|---:|
| rows | 128 | 128 | 256 | 256 |
| device span | 219.888 | 188.828 | 374.259 | 340.708 |
| `gdn-resident-core` | **53.207** | **27.748** | **63.268** | **41.391** |
| `ffn` | 75.321 | 72.046 | 151.825 | 140.530 |
| `sequence-input-projection` | 37.659 | 37.397 | 71.032 | 72.193 |
| `sequence-output-projection` | 24.570 | 22.846 | 31.356 | 31.297 |
| `sequence-core` | 13.172 | 12.903 | 26.610 | 25.106 |
| `head-projection` | 10.467 | 10.368 | 20.993 | 21.113 |
| step, untraced | 230.240 | 188.135 | 373.762 | 337.946 |
| aggregate tok/s ceiling | 555.9 | 680.4 | 684.9 | 757.5 |

**Halving the streams and doubling the tokens, at the same row count, moves one phase.**
`gdn-resident-core` goes 53.207 to 27.748 ms at 128 rows, a factor of 0.52 for half the state, while
the five phases that are priced per row and cannot depend on which sequence a row came from sum to
161.189 against 155.560 — 3.5%, which is the two processes' drift and the right normalizer here.
Normalized on those five, a 16-stream step is 11% faster at 128 rows and 5.3% at 256.

The cost model in the decode map predicted 16 x 8 would land near 209 ms against 230.375. It lands
at 188.135. A sequence is more expensive than that model said.

**The serving decision is the opposite of what the ceiling column suggests.** Those ceilings assume
every drafted token is accepted. Put acceptance in and a fixed 256-row budget compares as
`32 a8 / 0.373762` against `16 a16 / 0.337946`: sixteen streams of sixteen drafted tokens beat
thirty-two of eight only when a sequence accepts **1.81x** as many tokens from a draft twice as
deep. Acceptance per token falls with depth, so the wide-token shape is the worse bet at equal rows
even though it holds the faster step. What the table actually establishes is the *price*: 25.5 ms
per sixteen sequences per step at 128 rows, which is what a serving designer pays for concurrency
and can now quote.

### One number in that table that does not belong to this change

`sequence-output-projection` is 24.570 ms in a 128-row generation step and 16.337 ms in a 128-row
prefill pass. Same kernel, same row count, same `npad`, 50% apart — far outside the 2.8% and 4.7%
that `ffn` and `sequence-input-projection` differ by between the same two panels. Whatever that is,
it is worth 8 ms of a generation step and it belongs to whoever holds `project<TT, ADD, PLANAR>`.

## Installed and accepted

Canonical `e8923f0`, binary
`3a38366b1bc0f341845eeedceae16109acc0faaaf3f96155449d2a33f01f2080` (`tools/batch_compare`
`70b1000d8f0c1aecd5feeedcbe70843b1a7100d7ad6ec120c2ca5bf5116bdc35`). The installed executable
reproduced the identity exactly — 95,354,880 logits, FNV `17618767336181543518` at both widths, zero
nonfinite, the same digest the pre-install panel produced — and ingested the 384-token document at
707.1 and 704.7 tok/s in 128-row passes against 711.2 and 734.4 in 256-row passes, both rounds
favourable. Raw: `batch-comparison/pass-width/installed-e8923f0/run.json`.

The resident service was down and masked through this work because the lane was measuring
continuously; the restore marker at `/tmp/kelana-gpu-gpu-measure.restore` is set, which is what
hands the restart to whichever panel drains the queue.

## What is left here

- **A wider pass is monotonically better and 256 is not a limit, it is where the memory trade stops
  paying.** 512-row passes would save another 4.5 ms per 512 rows, 1.2%, for another 254 MB of logit
  buffer. The next width is not worth it unless the head stops writing rows nobody reads.
- **The head writes `VOCAB` floats for every row of the final prompt pass and prompt ingestion reads
  one of them.** `Engine::prefill` takes `am.back()` and `publish_last_logits(k)`; every other row's
  127 MB is computed, copied and dropped. At 256 rows that is 21 ms of head projection per prompt
  where 0.08 ms would do. It is exact — no consumer reads those rows — and it removes the only way a
  wider pass can cost anything: a prompt whose last pass is wide pays up to 10.5 ms more head than
  it would at 128 rows. Nobody holds `head_batch.hip` right now.
- **The 5.68 ms fixed cost of `sequence-input-projection` is a weight stream and it is the largest
  one left.** Whether it is reducible at all is the projections' question, not the width's.

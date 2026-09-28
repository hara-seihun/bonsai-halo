# The engine at the context it advertises

Every prefill number this repository publishes comes from a 384-token document, and
[long-context attention](long-context-attention.md) extended that to 3072 for one phase of one
shape. The engine advertises `MAXCTX 32768`, the resident service admits 1200-1400 token prompts,
and nothing here had ever run a **generation step** past a 64-token prefix. This is what both
shapes do out to 8192, taken on the routes the service runs, and the three questions it settles.

Absolute step times here were taken while other work shared the socket: the clock sampler that
landed mid-panel reports 2213 MHz p50 of a nominal 2900 with the host at its own limit, so read the
ratios, which are in-panel and interleaved, rather than the milliseconds. Raw samples, per-file, in
[`batch-comparison/long-context-serve/`](../../../data/bonsai2/batch-comparison/long-context-serve/README.md).
Nothing here changes a serving default or a numerical map. One instrument landed, below.

## The headline: a generation step spends its context at 30 GB/s

`tools/batch_profile --modes 0 --decode-streams 1 --decode-prompt N` now runs the eight-row
persistent route — the one `Engine::generate`, `generate_dflash` and every solo served request take.
One row, one stream, mode 0, prefixes ingested through the same wide route so the arms share their
state, medians of the samples in `m0-kvslope.json`:

| context | step | tok/s | context term |
|---:|---:|---:|---:|
| 1024 | 31.157 ms | 32.1 | 2.3 ms |
| 4096 | 37.79 | 26.5 | 8.9 |
| 8192 | 46.95 | 21.3 | **18.1** |
| 16384 | 65.59 | 15.2 | 36.7 |
| 32768 *(fit)* | 101.1 | 9.9 | 72.2 |

It is a straight line: **2.204 ms per 1000 tokens of context**, intercept 28.9 ms, which is the
context-free single-stream step this lane has published for weeks. The line is fitted on 1024,
4096 and 8192 and then **predicts 16384 to 0.9%** — 65.0 ms against 65.587 measured — so the 32768
row is an extrapolation of a relation that has held across a sixteen-fold range, not a guess. The eight-row verify pass the
drafted path runs carries the same term — 58.3 ms at 8192 against about 46 at a short prefix.

**A token of context is 65536 bytes of K and V** (16 attention layers x 4 KV heads x 256 dims x
2 bytes x 2). So the marginal cost is 65.5 MB per 1000 tokens against 2.204 ms: **29.7 GB/s, 12% of
the 242 GB/s `bench/bw` measures.** At the roof that term would be 0.27 ms per 1000 tokens, an 8192
step would be 31.1 ms instead of 46.9 (**+51%**) and a 32768 step 37.8 instead of 101 (**2.7x**).

### It is not bandwidth, and one arm proves it

`--attn-hg 1,6` changes how many query heads of a KV group one score unit carries. At six, a key
chunk is read once for the whole group; at one, the same chunk is read six times. Same arithmetic,
same output, **six times the requested K and V bytes**. Mode 0, one row, 8192-token prefix,
`m0-hg-8192.json`:

| heads per key read | step | requested K+V per 1000 tokens |
|---|---:|---:|
| 6 (deployed) | 46.965 ms | 65.5 MB |
| 1 | 47.329 ms (**+0.8%**) | 393 MB |

Six times the traffic for 0.8% of the time. The phase is nowhere near a byte roof — unfolded it is
already moving 393 MB in 18 ms, 178 GB/s of requests — and the folded arm is spending the same 18 ms
on a sixth of that. **Whatever binds this phase costs the same whether it reads 65 MB or 393 MB**,
which rules out every fix whose mechanism is fewer KV bytes at this shape: a smaller cache
coordinate, an fp8 cache, more head folding, wider row groups. They are all playing for 0.8%.

What is left is the shape of the walk itself, and the census points at one loop. At **TT = 1** the
value accumulation in `attn_chunk_unit_ref` is, per key, one `global_load` of a single `__half` per
thread and **one** `fmaf` to hide its latency, in a `#pragma unroll 1` loop with no lookahead — the
score loop got a one-block cursor in the schedule work and the value loop never did. At the prefill
shape a load covers eight rows of FMAs and many resident units; at one row it covers one. That is
consistent with every number above: arithmetic is 4% of the machine's issue capacity at this shape,
bytes are 12% of its bandwidth, and the phase still takes 18 ms.

**And the deployed route runs the reference body.** `ph_attn` defaults `SCHED = 0` deliberately —
its comment says so, because the body is inlined into `k_forward_rows` whose VGPR count sets
`rows_grid_size` for every wide pass — so the four schedule changes in `attn_chunk_unit_fast`
(key-block cursor, breadth-first trees, 32 barriers to 4, batched value reads) have **never run on
the route that serves solo requests**. They were measured on `k_attn_wide` at 128 rows, where this
loop is not what the unit waits on. Somebody should measure them where it is.

## Prompt ingestion at 8192, on the served width

`--prefill-scan 8192 --rows 256`, mode 20, the deployed prompt route at the width the server
ingests with, `HALO_ATTN_TRACE_SPLIT=1` for the three attention launches
(`scan8192-split.json`):

| phase | whole 8192-token document | share |
|---|---:|---:|
| `ffn` | 10746.5 ms | 59.9% |
| `sequence-core-score` | 3949.6 | **22.0%** |
| `sequence-input-projection` | 1386.0 | 7.7% |
| `gdn-resident-core` | 621.7 | 3.5% |
| `sequence-core-combine` | 601.3 | 3.3% |
| `sequence-output-projection` | 571.6 | 3.2% |
| `sequence-input-prep` | 217.2 | 1.2% |
| `sequence-core-pre` | 46.3 | 0.3% |

By position, the same walk — a pass is charged where its first row sits:

| position | score | combine | pass | score share |
|---:|---:|---:|---:|---:|
| 0 | 3.6 ms | 1.4 | 449.5 | 0.8% |
| 2048 | 52.6 | 11.5 | 479.6 | 11.0% |
| 4096 | 109.2 | 18.3 | 549.4 | 19.9% |
| 6144 | 204.0 | 26.5 | 645.5 | 31.6% |
| 7168 | 253.1 | 31.1 | 698.3 | **36.2%** |

Three things fall out of it.

- **The chunk-partial round trip is not the problem.** A 256-row pass at 8192 writes 409 MB of
  partials per layer and reads them back, 13 GB a pass against 5.9 GB of weights, and the whole
  combine is **3.3%** of the document. [long-context-attention](long-context-attention.md) ruled the
  exact alternatives out for flash-style folding; this says the prize for breaking that exactness is
  small anyway. Optimise the score loop, not the buffer.
- **The partial budget's launch split is free at this length.** `HALO_ATTN_PART_MB` (192 MB) caps
  the buffer, so the phase runs 16 launches a pass to position 3072 and 32 from 4096 — visible in
  `launches` — and the score curve steps straight through that boundary (80.0 ms at 3072, 109.2 at
  4096, against a position ratio of 1.33). At 32768 the same rule would cut the phase into about
  eleven launches a layer and each one re-reads the layer's keys; nobody has measured there.
- **Prefill and decode disagree about what attention costs.** A 256-row pass amortises one key read
  over 256 rows and still spends a fifth of itself there; a generation step reads the same keys for
  one row. The two shapes want different fixes and this lane has been fitting both at 384 tokens.

## Three questions closed

### 1. The K cache coordinate is bit-identical on the deployed route, and the two routes pull opposite ways

[attn-reuse-fold](attn-reuse-fold.md) left the leaf coordinate switched off with the reason written
down: nobody had measured the persistent kernel reading an interleaved cache, and
`tools/batch_profile` could not run mode 0 at all. It can now.

Mode 0, 2048-token prefix, `--attn-coord 0,1` alternating in one process, two rounds
(`m0-coord-2048.json`): **the residual FNV-64 is `6988584762389117466` at one row and
`4823451459947689174` at eight in every arm of every round.** The coordinate moves no bit on the
route that serves solo requests, which is the half of its exactness argument that was missing.

It costs that route about **1%** at one row (33.34 against 33.78 ms, mean of two) and nothing
measurable at eight (48.81 against 48.86, discarding a cold first sample), and the cost grows with
context because it is a property of the strided read the row-major body does against an interleaved
cache.

It buys the wide route **−8.6% of the whole attention phase** over a 3072-token document — 596.9 to
545.7 ms, with `ffn`, both sequence projections, `gdn-resident-core` and the head flat within 0.3%
in the same samples — worth +1.1% of whole-document prompt throughput (536.4 to 542.3 tok/s) at
3072, and about +2.2% at 8192 where attention is a quarter of ingestion.

**So it is a wash and it should stay off.** The service ingests and generates on the same K cache;
the coordinate cannot be per-route, because the cache holds one layout at a time. A deployment that
only ingests would want it on, and `HALO_ATTN_COORD=1` is how. That question does not need
reopening without a route that decides the layout per sequence.

### 2. Sixteen-row groups lose in the leaf coordinate too

[The 16-row group](long-context-attention.md#sixteen-row-groups-lose-and-the-raw-wall-time-says-the-opposite)
lost 18% in the row-major body and the reason given was registers: `k_attn_wide` goes 92 VGPR and 16
waves to 180 and 8, because a folded row costs `qr[TT][DPL]` = 64 registers. In the leaf body the
query is wave-uniform and sits on the scalar unit, so that argument does not apply there, and the
corner had never been run.

One panel, four arms, one process (`scan3072-coord-tt.json`), attention phase over a 3072-token
document:

| | coord 0 (row-major) | coord 1 (leaf) |
|---|---:|---:|
| 8-row groups | 596.9 ms | **545.7** |
| 16-row groups | 643.0 (+7.7%) | 674.5 (+13.0%) |

**Sixteen-row groups are worse in the leaf coordinate than in the row-major one**, so the register
explanation was not the whole story and halving the key traffic is not worth having in either. With
the `--attn-hg` result above, the reuse rectangle is now closed on both axes at both shapes: more
reuse per key read is not what this phase wants.

### 3. The prompt-route ladder at length, and what the approximate route is actually worth

The service ingests through mode 20. At 3072 tokens in 128-row passes, one process, the three wide
routes (`scan3072-modes18.json`), with the `ffn` row carrying the entire difference:

| route | `ffn` over the document | prompt tok/s |
|---|---:|---:|
| mode 20, the engine's own FFN slice (deployed) | 3957.1 ms | 537.4 |
| mode 18, the A8 FFN module | 3559.2 | 567.8 (+5.7%) |
| mode 19, the A4 FFN module | **1710.1** | **887.2 (+65%)** |

At 8192 in 256-row passes, the width the server uses: mode 19 is **670.2 tok/s against 454.6**,
+47%, and the gap narrows only because attention — which both routes share — grows into a quarter of
the work.

**The A4 route is the largest unclaimed number in prompt ingestion and it is already built**
(`--prefill-ffn wide-commit-a4`). Its cost is measured and it is not zero:
[seq-a4-default](seq-a4-default.md) puts the A4 FFN at **+0.01866 ± 0.00676 nats** against the A8
module over 1024 paired predictions, 2.8 sigma, on a metric that resolved the change this lane did
adopt at less than half that size. **That is a product decision, not a lane experiment**, and it is
written down here rather than taken: +47 to +65% of prompt ingestion for +0.019 nats on the served
path. The exact middle arm, mode 18, is +5.7% at the same activation precision as the deployed
route, and is likely to be erased by the FFN slice's own column-group work now in flight.

## The instrument, which the lane needed for other reasons

`tools/batch_profile` admits `--modes 0` in its decode path. Until now the eight-row persistent
route — the route the service generates on and every drafted verify pass runs — could not be
compared with any other route inside one process, one clock and one prefilled state; a route
question had to be answered across two builds. Three pieces:

- `--modes 0` passes validation and reserves no FFN image, like modes 4 and 20.
- The residual FNV-64 comes from the engine's own `x` for mode 0. The persistent route leaves its
  residual there rather than in `batch_x`, and a step of at most `RMAX` rows is one slice, so the
  hash covers every row of it. `residual_rows` records the count.
- `--decode-setup-mode M` ingests the context through a fast route and runs the timed step on the
  measured one. The eight-row route ingests at about 155 tok/s, so a 4096-token prefix is half a
  minute of GPU lock per case; through the A4 wide route it is three seconds. The ingested values
  differ between routes and a decode step's cost does not depend on them — no phase branches on a
  cache or state value — so arms sharing a setup mode compare on identical state. `setup_mode` is in
  every sample.

`--attn-coord` is also an axis of the `--prefill-scan` loop now, applied where the walk resets the
sequence and re-ingests from position 0, which is the only point at which the K cache layout can
move.

## What to do next here

1. **Put the value loop's latency on the clock at one row.** `ph_attn` compiles `SCHED = 0`, so the
   key-block cursor and the batched reductions in `attn_chunk_unit_fast` have never run on the
   deployed decode route. The cheap arm is a lookahead in the value loop specifically: at TT = 1 it
   is one load and one FMA per key. The register trap is named in `ph_attn`'s own comment — this
   body is inlined into `k_forward_rows` and its VGPR count sets `rows_grid_size` for every wide
   pass — so the honest form is a launcher-level instantiation, not a schedule switch that moves
   everybody's grid.

   **It cannot be reached by a flag, and the null that proves it is in the tree.** `--attn-sched
   0,1` on a one-row pass forced onto the wide route (`--wide-min 1`, 8192-token prefix) gives
   45.600 against 45.731 ms, `m20-sched-1row-8192.json` — because `launch_attn_wide` sends every
   all-one-row pass to the folded-head unit, and **both of its one-row launches hardcode the
   schedule template argument at 0**. The axis exists, the kernels it selects do not, and a panel
   that does not read those two launch lines will publish a null about a body it never ran. The same
   panel says the wide route is 2.8% *faster* than the persistent one at one row and 8192 context
   (45.6 against 46.9 ms), on a different numerical map, which belongs to whoever owns the decode
   route floor.
2. **18 ms of an 8192-token step and 72 of a 32768 one are worth a rocprof panel**, not another
   arithmetic argument. `BONSAI_PROFILE_COUNTERS=MemUnitStalled` and `OccupancyPercent` through
   `run-batch-compare --profile-tool` will say whether this walk is short of memory parallelism or
   short of units.
3. **Nobody has run a decode step past 8192**, and the partial-budget split at 32768 is an
   arithmetic prediction, not a measurement. A 32768-token prefix costs about 40 s of ingestion
   through the A4 route, which is one bounded panel.

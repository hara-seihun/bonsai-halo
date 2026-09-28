# The served decode route takes the schedule the server already ingests with

Raw samples, the four logit dumps and a file-by-file index are in
[`serve-decode-wide/`](../../../data/bonsai2/batch-comparison/serve-decode-wide/README.md).

[The served decode route](serve-decode-route.md) gave a concurrent server batched steps and stopped
one step short of the wide schedule. [The wide route's floor](serve-wide-decode.md) then measured
what that step is worth — mode 20 at 301.8 tok/s against 220.4 at 32 rows — and left the decision
with one sentence: *the question wants one horizon panel on mode 20 + a8 against mode 0.* Both
authors released without taking it, and the route stayed off behind `decode_batch_mode`.

It is on. A served decode step now runs `batch_mode` 20, which is **two different things** depending
on how many requests share it, and both halves are measured here:

| active requests | what the step does | against the route it replaces |
|---|---|---|
| 1..8 | `Engine::forward`, untouched | identical code |
| 9..31 | the same sliced route, plus the direct state commit | **bit-identical**, +3.1% of a decode step, a served null |
| >=32 | the wide schedule: wide prep, wide sequence projections, resident GDN state | **+52.1%** of a decode step, **+19.9%** of an aggregate served wave, and a different numerical map worth **+0.0064 +- 0.0072 nats** |

The floor between the last two rows is `batch_wide_min()`, which is 32 and which this change does
not move. The server prints which regime it is in at startup. The two percentages are different
measurements of the same thing and the smaller one is the honest headline: a served wave spends a
large share of its time ingesting, and this route carries only the generation half of it.

## The argument that decides it, before any panel

**A served request's prompt is already ingested by the wide map.** `prefill_batch_mode` defaults to
20 for `serve`, chat and `-p` alike, so the wide prep, the wide sequence projections and the
resident state have produced every served prompt's hidden state since
[the served prompt route](serve-prefill-route.md) landed. Only the generation step was still on the
sliced map. One request was being carried by two different maps in sequence, and the one it spent
the most tokens in was the slower one.

## Below the floor: no numerical question at all

`forward_batch` routes a pass under `batch_wide_min()` rows to the sliced schedule whatever mode
asked, so mode 20 at 9..31 rows differs from mode 4 in exactly one thing: `commit_state`, the direct
state commit.

`tools/batch_compare --only quality --quality-shapes decode --quality-rows 32` with
`HALO_WIDE_MIN=33`, which routes a 32-row decode pass to the sliced schedule in both arms:

| pair | differing logit bits |
|---|---:|
| mode 4 vs **mode 20**, 32-row decode pass, sliced in both | **0 of 7,946,240** |
| mode 4 against itself, separate process | 0 of 7,946,240 |

So the direct commit is free of the output, the instrument's floor is zero, and the whole 9..31 band
is a schedule change with an identity proof rather than a quality argument.

## Above the floor: the map, measured on both sides

The same instrument with the floor at its default, so the 32-row decode pass takes the wide route in
mode 20 and the sliced one in mode 4. Every arm is fed identical tokens; prompts are short enough
([`bench/serve-wide-prompts.txt`](../bench/serve-wide-prompts.txt)) that their ingestion passes stay
under the floor and are bit-identical in every arm, so **only the decode pass differs**.

| pair | differing values | max abs | rms | argmax |
|---|---:|---:|---:|---:|
| wide `a8` vs sliced | 7,946,193 / 7,946,240 | 0.115 | 0.0182 | 32/32 equal |
| wide `a4` (the default) vs sliced | 7,946,227 / 7,946,240 | 0.488 | 0.0587 | 32/32 equal |
| sliced `a4` vs sliced `a8` | **0** | 0 | 0 | 32/32 |

Two things fall out. **The sliced route is inert to the sequence activation width** - the third row
is a zero, which is what makes the first two readable - and **`a8` is 3.2x closer to it than `a4`**,
which is the mechanism under `serve-wide-decode.md`'s 27/32 against 31/32 greedy continuations. It is
not bit-identity, at either width: the wide schedule reduces FP32 in a different order.

### The horizon panel

The question a logit difference cannot answer is whether the difference is *worse*.
`tools/batch_compare --only horizon` walks 32 teacher-forced document windows one token per step and
scores the distribution every eighth step, so both arms see identical inputs and the pairing is
exact. Both arms are **mode 20** and differ only in `HALO_WIDE_MIN`: 33 routes the 32-row decode step
to the sliced schedule (today's server), 32 takes the wide one (the new default). Ingestion is the
wide map in both, at 128 rows a pass, exactly as the product does it.

512 paired predictions, 32 streams, 128 tokens of context, 128 steps, `--pin-clock`:

| arm | NLL | perplexity | top-1 vs the document |
|---|---:|---:|---:|
| sliced decode (today) | 2.99602 | 20.006 | 0.4121 |
| sliced decode, **repeated in a separate process** | **2.99602** | **20.006** | **0.4121** |
| wide decode (shipped) | 3.00238 | 20.133 | **0.4199** |

**dNLL = +0.00636 +- 0.00716**, clustered by stream because a stream's state error persists across
its own steps (the naive per-prediction error is 0.00752, so the clustering barely moves it). At two
sigma the wide decode map costs **at most +0.021 nats**, and the repeat control reproduces all 512
predictions *bit for bit* in a separate process, so the instrument has a zero floor and everything
in that interval is the map.

For scale, on the same kind of instrument: the A4 FFN map this repository ships as its batched
headline costs **+0.0615 nats**, and [the int8 recurrent state](gdn-state-horizon.md) scored
-0.0165 +- 0.0082. This change is a third of the first and inside the second's error bar.

There is no trend across the horizon either, which is what would show a map whose error compounds
through the recurrence:

| steps | 7-31 | 39-63 | 71-95 | 103-127 |
|---|---:|---:|---:|---:|
| dNLL | +0.0058 +- 0.0150 | +0.0191 +- 0.0127 | -0.0037 +- 0.0153 | +0.0042 +- 0.0143 |

6.25% of argmaxes move, which is the same trajectory decorrelation every perturbation of this model
produces ([the state horizon](gdn-state-horizon.md) measured 10.6% for a 257x range of perturbation
sizes), and the top-1 agreement with the document goes **up**, 0.4199 against 0.4121. The honest
reading is that this map is not distinguishable from the sliced one at 512 paired predictions, not
that it is better.

## Throughput

**Engine, `tools/batch_compare --only decode`, three rounds with the arm order reshuffled, one
process, `--pin-clock`, 2809 MHz p50:**

| streams | sliced (mode 4) | wide (mode 20) | change |
|---|---|---|---:|
| 16 | 206.9 / 207.3 / 206.6 | 213.1 / 213.6 / 213.8 | **+3.1%**, bit-identical |
| 32 | 223.8 / 223.6 / 223.8 | 340.5 / 340.5 / 338.7 | **+52.1%** |

Every cell repeats to better than 0.5% and the ordering survives the reshuffle.

**The product, `tools/serve_load.py` over HTTP against the real server**, both arms started in one
lock hold, `--arm sliced:32:HALO_SERVE_DECODE_MODE=4 --arm wide:32`:

| clients x tokens | sliced | wide | change | identical completions |
|---|---:|---:|---:|---:|
| 16 x 16 | 90.5 tok/s | 91.6 | +1.3% | 16 / 16 |
| 32 x 16 | 91.9 | 103.9 | +13.0% | 22 / 32 |
| 32 x 64 | 166.0 | 199.0 | **+19.9%** | 11 / 32 |
| 16 x 32, installed | 129.0 | 128.9 | -0.0% | 13 / 16 |
| 32 x 32, installed | 131.9 | 159.1 | **+20.7%** | 12 / 32 |

The served figures are smaller than the engine's because a wave of short requests is mostly
ingestion: 850 prompt tokens against 512 generated ones in the 16-token wave, and the route this
changes only carries the second. At 64 tokens a request - an ordinary chat answer - it is +19.9%,
and per-request rate goes 5.2 to 6.3 tok/s, which is what a client waiting behind 31 others feels.

### A served completion is not a reproducible object, and that is measured here

The 13 of 16 in the installed row is not this change. **Two servers running the identical binary in
the identical configuration agree on 13 of 16 completions at 16 clients and 29 of 32 at 32 clients**
(`served-selfcontrol.json`, both arms `HALO_SERVE_DECODE_MODE=4`). Greedy requests, same prompts,
same slots.

So the honest reading of the served identity columns is against that floor, not against 100%:

| pair, 32 tokens a request | 16 clients | 32 clients |
|---|---:|---:|
| sliced against **itself** | 13 / 16 | 29 / 32 |
| sliced against wide | 13 / 16 | 12 / 32 |

**At 16 clients the new default is indistinguishable from its predecessor at the server's own
reproducibility floor**, which is what the bit-identical logit dump predicts and what a served panel
can actually resolve. At 32 clients it is far below that floor, which is the map change, visible
exactly where the panel above says it should be.

Two mechanisms are already known to make a served step non-reproducible, and this run cannot
separate them: the cooperative `KS = 2` matvec drains two partial sums with `atomicAdd`, so a phase
is not bitwise reproducible run to run, and **which requests share a step is decided by admission
timing**, so a wave under different host load composes different passes. The self-control ran with
15.3 host cores busy against 3.3 for the wide arm, so its floor may be the looser of the two. Either
way it is the floor a served acceptance has to be read against, and
[the chunked admission panel](serve-prefill-admission.md)'s "22 of 22 identical completions" was a
shorter wave than this one.

## What ships

`ServeBatch::decode_route()` defaults to 20 instead of 4 when the process has sequence modules, which
a served process always builds for ingestion. Nothing else moves: `route_min_rows()` is still
`RMAX + 1`, `batch_wide_min()` is still 32, `WIDE_MIN_DEFAULT` is untouched, and an explicit
`decode_batch_mode` keeps the exact contract it had.

- **`HALO_SERVE_DECODE_MODE=4` restores the predecessor exactly**, and is what the sliced arm of
  every panel above ran.
- A process with no sequence modules gets mode 4, not the one-row persistent pass it used to fall
  back to when a wide mode was asked for and could not be served.
- The startup line now prints the wide floor beside the route, so a panel can see which of the two
  regimes it is measuring.

## The resident service cannot reach this, and that is now two routes deep

`bonsai-halo.service` runs `serve --slots 4` at the full 32768-token context, so **a step never
reaches nine rows** and neither batched route exists for it: not this one, and not
[the sliced batched route](serve-decode-route.md) that landed +38%/+50% before it. The startup line
says which regime a process is in, and the service's says `mode 20 above 9 rows, wide floor 32 rows`
with four slots underneath it.

What stops it is memory, and the arithmetic is small enough to settle here. A slot costs about 88 KiB
per token of context plus 220 MB of recurrent state: **2.97 GB at the full 32768 tokens**, 0.91 GB at
8192, 0.57 GB at 4096. This machine has **60 GB of memory total**, the service already holds 23.1 GB
of it, and an agent's measurement panel asks for 28 GiB more under
`tools/run-batch-compare`. So the two ways to reach nine concurrent rows are not equal:

| configuration | slot memory | reaches the batched route | costs |
|---|---:|---|---|
| `--slots 4 --context 32768` (today) | 11.9 GB | never | - |
| `--slots 9`, full context | 26.7 GB | at 9 clients | +14.8 GB on a 60 GB box that also runs 28 GiB panels |
| `--slots 12 --context 8192` | 10.9 GB | at 9 clients | **less memory than today**, and a request may not exceed 8192 tokens |
| `--slots 32 --context 8192` | 29.1 GB | the wide floor, at 32 clients | +17.2 GB and the same context cap |

The third row is the interesting one and it is a **product policy question, not an optimisation**:
it buys the batched routes for less memory than the service spends today, and it pays for them with
the advertised 32K context. Nobody has measured what this household's requests actually ask for, and
that measurement - not another kernel - is what decides whether any batched serving work reaches a
user. [The int8 recurrent state](gdn-state-horizon.md) cuts the fixed 220 MB per slot to 55 MiB if it
becomes the default, which moves every row of that table.

## What this leaves

- **The slot count is the binding constraint on every batched serving result in this repository**,
  and it is one line of the service's command. Price the trade before optimising another batched
  step: a route the deployed configuration cannot enter is worth exactly zero to the product.
- **`a8` on the wide decode route is not priced.** It is 3.2x closer to the sliced map and
  `serve-wide-decode.md` measured it as free of time at a generation width (106.87/108.67 ms against
  108.94/108.47). It is not free of *prompt* throughput, because `sequence_quant()` is process-wide
  and `sequence_batch_sync_quant` reorders stored codes, so pinning it for decode pins it for
  ingestion at -12.2% of prompt rate. Deciding it needs one horizon pair at `a8` against this one.
- **The prompt floor is the same question at a different width.** `WIDE_MIN_DEFAULT = RMAX + 1` is
  worth +21 to +34% on every 9..31-row prompt pass, and the map it crosses into is the one measured
  here. This panel is evidence about that map, not about that width: nobody has run the horizon with
  ingestion as the axis, because the horizon's own prefill is hardcoded to 128-row passes.
- **Above 32 clients the wide route's own width ladder is unmeasured.** Every number here is at 32
  rows; `PASSMAX` is 256 and `batch_capacity` decides what a step may carry.

# A prompt stopped every other request, and the throughput was never where that looked

Raw samples and a file-by-file index are in
[`serve-prefill-admission/`](../../../data/bonsai2/batch-comparison/serve-prefill-admission/README.md).

[Concurrent serving](served-concurrency.md) and [the served decode route](serve-decode-route.md)
both closed with the same open item, in almost the same words:

> A prompt blocks every decode, which `served-concurrency.md` named and nobody has taken. Ingestion
> is one call, so a 2000-token prompt stalls every other request for its duration, and chunked
> admission is a change inside `ServeBatch::loop` with no kernel in it.

It is taken. A prompt is admitted a pass at a time now, and the requests that are already
generating take a decode step between its chunks. **The longest gap between two tokens of one
response falls from 12.97 s to 0.75 s at six concurrent requests, for −0.8% of aggregate rate
against a ±2% panel spread.** Twenty-two of twenty-two completions across four panels are identical
to the server this replaces.

**And the throughput was never where the claim that opened this work said it was.** Serializing
admission does not narrow the decode steps; it *synchronises* the requests, which widens them. That
result is at the bottom of this page and it is the part to carry.

## What the defect was

`ServeBatch::loop` picked the first request whose prompt was not yet in its sequence and ran the
whole prompt:

```cpp
Req * pre = nullptr;
for (Req * r : batch) if (!r->started) { pre = r; break; }
if (pre) { ...; prefill(pre); }          // every other active request waits here
```

`Engine::prefill` returns when the last prompt token is ingested, so a 1387-token prompt held the
engine thread for 2.5 s and every other request emitted nothing for that time. The cost compounds
with arrivals: six requests arriving 0.8 s apart gave the first one a **12.97 s** gap in the middle
of its response, because it had to wait out five other prompts.

## The change

`prefill` becomes `prefill_begin` plus a resumable `prefill_step(r, budget)`. The budget is one wide
pass of prompt tokens (`prefill_width()`, 256 on a served process, floored at 256 tokens so a
server on the eight-row ingestion route does not interleave every eight tokens), and between chunks
the loop runs one ordinary decode step for the requests that have already started.

**A chunk boundary is a boundary `Engine::prefill` already had.** It takes what the sequence has
ingested as `from` and walks `prefill_width()` rows a pass from there; the prompt-prefix snapshot
points in this same function have always cut a prompt into several such calls. Rounding the budget
down to whole passes keeps the pass composition an unchunked ingestion would have had, so what a
chunk adds is the state commit its last pass performs and nothing else.

Three properties are by construction rather than by measurement:

- **One prompt ingests at a time.** Two prompts are never interleaved, so the prefix cache keeps the
  property [concurrent serving](served-concurrency.md) was careful about: the second request behind
  a shared preamble still looks for its snapshot after the first request saved it. Round two of the
  paired panel re-sends the same four prompts and is served from snapshots in both arms, 0.07 s to
  first token.
- **A solo request is untouched code.** With nothing else started the budget is zero and the call is
  the single call the server has always made, so interactive latency and the drafted single-stream
  path cannot move.
- **An interleaved decode step carries no row of the ingesting sequence.** A row's arithmetic
  depends on its own sequence's state and position; which rows share a pass is a schedule, which is
  the property every batched result in this lane rests on.

## Measured

Three panels, each with all arms inside one `tools/run-batch-compare --pin-clock` hold, servers
started by `tools/serve_mixed.py` on one clock. `HALO_SERVE_PREFILL_CHUNK=-1` is the control arm: it
is the admission this replaces, call for call.

| panel | arm | window | aggregate | **worst stall** | TTFT median | completions |
|---|---|---:|---:|---:|---:|---|
| 6 requests, 1387-token prompts, 0.8 s apart, 32 tokens each | blocking | 17.31 s | 11.09 tok/s | **12.97 s** | 8.24 s | — |
| | chunked | 17.59 s | 10.91 (−1.6%) | **0.75 s** | 8.35 s | 6/6 identical |
| 4 requests, 1200-token prompts, 1.5 s apart, 96 tokens each | blocking | 13.09 s | 22.1 | **7.05 s** | 4.24 s | — |
| | chunk 256 | 13.16 s | 22.0 (−0.5%) | **0.96 s** | 4.41 s | 4/4 identical |
| | chunk 1024 | 12.86 s | 22.5 (+1.8%) | 2.28 s | 4.04 s | 4/4 identical |
| 4 requests, 1200-token prompts, 1.0 s apart, 64 tokens each | blocking | 12.40 s | 19.8 | **7.20 s** | 5.36 s | — |
| | chunked | 12.22 s | 20.1 (+1.5%) | **0.89 s** | 5.12 s | 8/8 identical |
| **installed acceptance**, canonical `cca7f58`, same shape | blocking | 12.34 s | 19.9 | **7.15 s** | 5.21 s | — |
| | chunked | 12.70 s | 19.4 (−2.5%) | **0.95 s** | 5.42 s | 4/4 identical |

**The worst stall falls 7.3x, 7.5x, 8.1x and 17.3x. Throughput is a null with a slight cost:**
−1.6%, −0.5%, +1.5%, −2.5% over four cold panels, **mean −0.8%** against a spread that reaches ±2%
between arms running identical code (the second round of the paired panel, fully cached, is 47.4
against 47.5 tok/s). Every request's median inter-token gap during someone else's ingestion becomes
the chunk period (0.49 s) instead of the whole prompt.

The cost has a mechanism and a knob. An interleaved decode step pays a whole weight stream for the
rows that have started; the step it replaces later would have carried the rows of every request,
because blocking admission synchronises them. So the price of interleaving is the narrow steps it
adds, and the chunk size sets how many: **chunk 1024 measures +1.8% against blocking with a 2.28 s
stall**, chunk 256 measures −0.8% with a 0.95 s one. One wide pass is the default because a
sub-second gap is not visible in a streaming response and a 2.3 s one is, and both are inside the
panel's own spread on rate. `HALO_SERVE_PREFILL_CHUNK` takes the other point in one environment
variable.

## The result worth more than the stall: serialized admission is not a throughput defect

The claim that opened this work was that serialization costs throughput by *de-synchronizing*
requests - a request held out of the batch finishes later, so later steps carry fewer rows, and a
served step is priced per weight stream. **The measurement says the sign is the other way, and the
reason is worth carrying.**

Blocking admission *synchronizes* requests. Everybody waits out every prompt, so when decoding
finally runs, all N requests step together at the widest row count the workload can produce.
Chunked admission spreads the same decode work across the ingestion window, where the started set is
smaller, so it runs **more steps at narrower widths** for the same tokens. The two effects nearly
cancel, which is exactly what the three panels show.

**So the throughput money in this part of the server is not in *when* a decode step runs. It is in
*which pass carries it*.** A 256-row prompt pass has already paid for the whole 5.9 GB weight
stream; a decode row riding inside it costs one row of a pass that is running anyway. During the
chunked panel's 8.8 s of ingestion the server ran about 18 separate decode steps, roughly 1 s of
weight streaming, which is **7.6% of that window** and is the size of the prize — and it is also
exactly the −0.8% this change pays, from the other side of the same ledger. It is continuous
batching, and on this engine it is **a numerical change, not a schedule change**: prompt ingestion
runs mode 20 and [the wide route is not bit-identical to the sliced route at any activation
coordinate](wide-sequence.md) - mode 0 hashes `7241142880296265908` where mode 20 hashes
`15043080294956075508`. A decode row moved into a prompt pass changes what the server emits, so it
needs [the horizon instrument](seq-a4-default.md) at 9..32 rows and not a token-identity check. The
exact alternative does not exist: mode 4 keeps the eight-row route for prep, the projections and
attention, so a 256-row mixed pass on it would be 32 slices of the non-FFN weights - far more
expensive than the two passes it replaces.

## What this does not change

- **Aggregate decode throughput at a fixed width.** `batched_step`, `decode_route` and the route
  floor are untouched.
- **The prefix cache.** Cached prompts are ingested by the same calls in both arms, and a fully
  cached round measures 47.4 against 47.5 tok/s with 0.07 s to first token in both.
- **Any kernel, image, weight or numerical map.** The engine calls are the calls
  `Engine::generate` makes, in the same order, at different offsets.
- **`--slots 4`, which the resident service runs.** With four slots a step never reaches the nine
  rows [the served decode route](serve-decode-route.md) needs, and this change is about the four
  that do share the server rather than about how wide a step is. The line that document is missing:
  `route_min_rows()` is `RMAX + 1 = 9` and a step carries at most `max_active` rows, so **the route
  is inert at any slot count of eight or fewer, whatever the context length** - eight slots at 16384
  tokens costs 0.6 GB more than four at 32768 and buys nothing there. Nine slots is the floor, and
  that is a footprint decision rather than a lane one.

## Commands

```sh
tools/run-batch-compare --pin-clock --exec python3 tools/serve_mixed.py \
    --exec ./bonsai-halo --dflash ~/data/bonsai2/drafters/dflash2.safetensors --context 4096 \
    --arm blocking:8:HALO_SERVE_PREFILL_CHUNK=-1 --arm chunked:8 \
    --requests 4 --stagger 1.5 --prompt-tokens 700 --max-tokens 96 --check --out RESULT.json
```

`tools/serve_mixed.py` is the instrument this needed and the lane did not have.
`tools/serve_load.py` fires N requests together and waits for all of them, which is the right shape
for pricing a route and the one shape in which this defect is invisible: in a synchronised burst
there is never a request generating while another one's prompt is read. `serve_mixed` posts requests
on a schedule, each with its own distinct long prompt, streams every response and keeps the arrival
time of every chunk, so it reports the window's aggregate rate, time to first token, and the
**longest gap between two tokens of one response** - which is the number a blocking admission moves
by an order of magnitude and an aggregate rate cannot see.

`HALO_SERVE_PREFILL_CHUNK` is the axis: `-1` restores the blocking admission exactly, `0` or unset
takes one wide pass, and any token count is rounded down to whole passes.

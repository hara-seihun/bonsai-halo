# Concurrent requests decode together

The product could not reach the throughput this engine has. `src/server.cpp` ran every request
under one `std::mutex`, so two clients each got half of one single-stream decode and eight clients
each got an eighth, while the same box carries eight sequences through one weight stream for about
the cost of one. Everything needed was already in the engine: `nslots` sequence state slots,
`Engine::forward` over a mix of sequences, per-row argmax and logits. The server called none of it.

`ServeBatch` (`src/serve_batch.h`, `src/serve_batch.cpp`) owns the engine on its own thread.
Requests are submitted by HTTP handler threads, admitted into state slots, and decoded together:
one pass per step carrying one row per active request. `serve --slots N` allocates the slots;
`--slots 1` is the serialized server this replaces and is the control arm below.

## What it is worth

One process per arm, one lock hold, pinned clock, 16 distinct prompts of 27 tokens, 48 greedy
tokens per request, DFlash2 loaded, the deployed 32768-token context. Aggregate is every request's
completion tokens over the wall time of the wave. Build `543c657`, executable
`db441d7917c3adf01186cc19`:

| concurrent requests | `--slots 1` (serialized) | `--slots 4` (together) | |
|---|---:|---:|---:|
| 1 | 42.8 tok/s | 42.9 tok/s | null, same route |
| 2 | 56.5 | 59.0 | +4% |
| 4 | 50.5 | **82.7** | **1.64x** |
| 8 | 47.8 | **81.2** | **1.70x** |

The serialized arm is flat in concurrency because it is one stream: adding clients adds queue, not
throughput. Slowest-request latency at eight clients falls with it, 8.03 s to 4.73 s, because a
queued request waits for a shared step rather than for three whole generations.

Eight slots at a shorter context is the same memory and more throughput (`--context 8192`, same
workload, separate panel):

| concurrent requests | serialized | `--slots 8` | |
|---|---:|---:|---:|
| 4 | 49.0 tok/s | 83.1 | 1.70x |
| 8 | 45.1 | **115.7** | **2.57x** |

Raw and per-request samples: [`batch-comparison/serve-concurrency/`](../../../data/bonsai2/batch-comparison/serve-concurrency/).

### An agent-shaped workload

The requests this machine actually serves share a long system prefix and differ at the end, which is
what the prompt-prefix snapshots exist for. Four concurrent requests, a 900-token shared preamble,
distinct questions, 48 tokens each, installed executable:

| | serialized | `--slots 4` |
|---|---:|---:|
| aggregate | 29.7 tok/s | **38.5 tok/s** |
| wall | 6.46 s | 4.99 s |
| prompt tokens served from a snapshot | 3864 of 5229 | 3864 of 5229 |

**The prefix cache survives concurrency**, which was not obvious: requests are admitted together but
ingested one at a time, so the first request's snapshot is saved before the second one looks for it,
and three of the four resume instead of re-prefilling. The ratio is lower here than at short prompts
because 48 generated tokens against 1307-token prompts is mostly ingestion, and ingestion is
already a wide pass that batching does not change.

### On the resident service

The same driver against the running `bonsai-halo.service` on the installed executable
`db441d7917c3adf01186cc19`, `serve --port 8471 --slots 4 --dflash ...`, 23.06 GB on device:

| concurrent requests | resident service |
|---|---:|
| 1 | 40.9 tok/s |
| 2 | 55.0 |
| 4 | **81.1** |
| 8 | **79.2** |

That is the product, not a panel: `~/.pi/agent/local-models.json` carries `--slots 4`, so the unit
the machine restarts serves four requests at once.

## Why this needs no numerical argument

A row of a pass carries its own token, its own state slot and its own position. The recurrent state
and the K/V cache are indexed by slot, the weights are read-only, and `Engine::forward` builds
`fwd.rows[i] = { token, slot, position }` from each sequence independently. Which rows share a pass
is a schedule, not a map — the same property the speculative verify path has relied on since it
started sending eight rows through one pass.

The acceptance is the product's own output: greedy completions of the same prompts, compared
between the arms. **30 completions at concurrency 1, 2, 4 and 8, in two independent builds, zero
divergent characters.** The streaming path is compared against the buffered one in the same run and
matches, so the SSE deltas carry the same text the non-streaming body does.

## Where it saturates, and what is next

At the deployed FFN's arithmetic a pass carries eight rows, so a step with more than eight requests
runs `ceil(N/8)` passes and each pass re-reads the whole weight stream. Measured with 32 slots at a
2048-token context, 64 tokens per request:

> **This was the route, not the arithmetic, and it is gone.** The step called `Engine::forward`,
> which *is* the eight-row persistent pass, so sixteen requests were two complete forwards and
> thirty-two were four. [The batched decode route](serve-decode-route.md) sends a step above eight
> rows through `forward_batch` at mode 4, where the deployed FFN body and the vocabulary head run
> once for the whole step: **146.6 -> 220.4 tok/s aggregate at thirty-two rows**, with the served
> completions identical. The table below is the shape that measured the problem.

| concurrent requests | aggregate |
|---|---:|
| 8 | 114.2 tok/s |
| 16 | 111.3 |
| 32 | 111.4 |

Flat, as the row count predicts. Everything above eight rows per pass lives on the wide route
(`batch_mode` 19 or 20), which [decode-streams](decode-streams.md) and
[generation-128](generation-128.md) measure at 327 and 518 aggregate tok/s for 32 and 128 streams.
That route is the obvious next step for this scheduler, and it is a numerical decision rather than
a scheduling one: mode 20 is the wide schedule at the deployed arithmetic and mode 19 adds the A4
FFN map. The serialized server never had to make that choice; a batching one does.

Two more things this leaves:

- ~~**A request that shares a step loses speculation for the rest of its life.**~~ **Done** with the
  cheap re-arm. A shared persistent-kernel step now captures the drafter features for its rows and
  `Engine::dflash_ingest` feeds each sequence its row (one ingest launch per step, no draft), and
  prompt ingestion always fed the drafter. Every greedy request therefore keeps a current drafter
  and drafts again as soon as it is alone. A wide pass (above the wide floor) still captures
  nothing this route feeds, so a request that rides one stays on plain decode. September 26
  panel, 400-token code completion with a 16-token naming request arriving 0.6 s in: the
  previous binary fell to 35.1 tok/s (361 steps, 38 accepted drafts); the re-arm kept 121.6 tok/s
  (69 steps, 330 accepted) against 122.6 alone, with identical text. Four concurrent greedy
  requests plus one sampled request matched their solo completions exactly, and the longest
  resumed drafting when the others finished. A batched drafter remains the route to speculation
  *while* sharing.
- ~~**A prompt blocks every decode.**~~ **Done**, and the throughput was not where this predicted:
  [chunked admission](serve-prefill-admission.md) takes the worst inter-token gap from 12.97 s to
  0.75 s at six concurrent requests with aggregate throughput unchanged, because blocking admission
  *synchronises* requests and a server's decode steps are priced per weight stream. The money is in
  a decode row riding the prompt pass, which is a numerical change and is written up there.

## What a slot costs

Per slot, per the allocation in `Engine::load`: `2 x KV_SLOTS x NKV x HD x 2` bytes per token of
context (88 KiB), plus 220 MB of recurrent state, conv ring and pending block cache that does not
depend on the context.

| context | device bytes per slot |
|---|---:|
| 8192 | 0.96 GB |
| 16384 | 1.70 GB |
| 32768 | 3.17 GB |

The resident service runs `--slots 4` at the full 32768 tokens, 23.3 GB on device against 13.8 GB
for one slot. `--slots 8 --context 16384` is the same memory and 1.4x the ceiling; it is one flag
in `~/.pi/agent/local-models.json`, and the trade is context window against aggregate rate.

## Driving it

`tools/serve_load.py` is the first measurement in this repository that speaks to the product over
HTTP rather than to the engine directly, which is why this bug survived every panel:

```sh
tools/run-batch-compare --pin-clock --exec python3 tools/serve_load.py \
    --exec ./bonsai-halo --dflash ~/data/bonsai2/drafters/dflash2.safetensors \
    --arm serial:1 --arm batch:4 --concurrency 1,2,4,8 --max-tokens 48 \
    --out RESULT.json --check --stream-check
```

Each arm starts its own server, so both share a lock hold and a clock. `--check` compares every
arm's completions against the first arm's and prints the first divergence; `--stream-check` adds a
streaming request and compares it with the buffered one.

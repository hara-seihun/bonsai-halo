# A concurrent server was decoding on the one route with no batching in it

Raw samples, both instruments and a file-by-file index are in
[`serve-decode-route/`](../../../data/bonsai2/batch-comparison/serve-decode-route/README.md).

[Concurrent serving](served-concurrency.md) gave the HTTP server a scheduler: requests are admitted
into sequence slots and one step carries one row per active request. Its own table then said
something it could not explain.

> The ceiling is the eight-row pass, and it is measured. 32 slots, 2048 ctx: 114.2 tok/s at 8
> clients, 111.3 at 16, 111.4 at 32. Every eight rows is one whole weight stream, so concurrency
> above eight buys latency fairness and nothing else at the deployed arithmetic.

It is not the deployed arithmetic that does this. `ServeBatch::batched_step` called
`Engine::forward`, and that call *is* the eight-row persistent pass: `MAXSEQ` sequences, `RMAX`
rows, one complete forward over every weight in the model. Sixteen requests were two of those.
Thirty-two were four. `Engine::batch_mode` is zero unless something sets it, `ServeBatch` never did,
and `forward_batch` - the entry with every batched schedule behind it - was never called from
generation at all.

The same process already builds the wide modules. A served run defaults `prefill_batch_mode` to 20
and calls `prepare_batch` and `prepare_sequence`, so the batched FFN slice, the wide vocabulary head
and the wide sequence path are allocated on device while the step loop reaches none of them. The
comment on `prefill_batch_mode` says so in as many words: *"Generation is untouched: the route ends
at the last prompt token."*

## The route ladder, one process, one clock

`--only decode --modes 0,4,20 --streams 16,32 --slots 32 --context 256`, `--pin-clock`, six
generation steps after two warmup steps, aggregate tokens per second
([`route-ladder/run.json`](../../../data/bonsai2/batch-comparison/serve-decode-route/route-ladder/run.json),
and [`engine-routes/run.json`](../../../data/bonsai2/batch-comparison/serve-decode-route/engine-routes/run.json)
for the 0/20 pair taken separately):

| streams | mode 0 — what the server ran | **mode 4 — what it runs now** | mode 20 — the wide schedule |
|---:|---:|---:|---:|
| 16 | 146.7 | **203.0** (+38.4%) | 205.9 |
| 32 | 146.6 | **220.4** (+50.3%) | 301.8 |

**Mode 0 is flat and that is the whole diagnosis.** 146.7 at sixteen rows and 146.6 at thirty-two:
twice the rows for twice the time, because the second sixteen rows are a second complete pass over
the weights. Every route that shares a weight stream between rows rises instead.

## What mode 4 is, and why it is exact

Mode 4 keeps the persistent kernel. Prep, the input projection, attention or the recurrence and the
output projection run exactly as they do today, eight rows to a slice, and only the two phases that
were never bound by that slice leave it:

- **the FFN**, whose deployed body reads no row metadata, no block cache and no recurrent state, so
  one launch serves up to `FMAX` rows of the step instead of one launch per slice;
- **the vocabulary head**, which runs once over the batch instead of once per slice.

At thirty-two rows that is one pass over the 3.42 GB FFN stream instead of four, and one pass over
the 278 MB head image instead of four. Nothing else about the step moves. Same weights, same K
order, same per-block int32 accumulation, same FP32 fold - which rows share a launch is a schedule.

**Measured on the product, over HTTP, not argued.** `tools/serve_load.py` starts each arm's own
server inside one lock hold, drives them with concurrent chat completions and compares every arm's
completions against the first
([`served-mode4-32.json`](../../../data/bonsai2/batch-comparison/serve-decode-route/served-mode4-32.json),
[`served-16.json`](../../../data/bonsai2/batch-comparison/serve-decode-route/served-16.json)):

| concurrent requests | slots | sliced | batched | completions |
|---:|---:|---:|---:|---|
| 8 | 16 | 86.2 tok/s | 89.9 | 8/8 identical — *both arms run the same code here* |
| 16 | 16 | 86.7 | **103.5** (+19.4%) | 16/16 identical |
| 32 | 32 | 83.5 | **107.1** (+28.3%) | **32/32 identical** |

The eight-request row is the panel's own noise floor: below nine rows the route is not taken, so
both arms execute identical code and still differ by 4.3%. Read the served ratios with that in
mind; the ladder above is the precise instrument and this one is the product's own output.

Served throughput is lower than the ladder's because a served point pays prompt ingestion, HTTP and
per-request host work over 24 to 32 generated tokens. The route is what changed, and the identity
column is what makes it a schedule.

## The wide schedule is no longer bit-identical, and that is worth more than the 1.37x

The first version of this change routed served decode through mode 20, which crosses into the wide
schedule at 32 rows. It is faster - 158.9 against 107.3 tok/s, **+48.1%** at 32 concurrent requests
([`served-32.json`](../../../data/bonsai2/batch-comparison/serve-decode-route/served-32.json)) - and
**18 of its 32 completions diverged from the sliced arm**, where the same panel at 16 rows had 24 of
24 identical.

The schedule is not what did it. [Four-bit activations became the default for the wide sequence
input projection](seq-a4-default.md) at 19:26 on September 21 (`c131fe2`), so **every pass that
crosses `wide_sequence` now carries an approximate map that the eight-row route does not**.
[The wide sequence document's](wide-sequence.md) "direct commit preserves its predecessor's output
bits" predates that default by a day and no longer describes the default build.

Two consequences, and the second is for anyone touching the admission rule:

- A served generation route must not acquire an approximate map by reaching a row count. Mode 4 is
  the exact route and it is what ships; mode 20 stays available through `Engine::decode_batch_mode`,
  priced here at 1.37x more, and it needs the [horizon instrument](seq-a4-default.md) rather than a
  token-identity check.
- **Lowering the wide floor is a numerical change at every width it newly admits**, not a schedule
  change. `b6c614a9` holds that floor and is measuring the exact arm (`HALO_SEQ_QUANT=a8` on the
  generation pass alone); if a8 restores identity, the lane has an exact 301.8 against 146.6 and
  this switch takes it as a one-line constant.

## What it does not change

- **One request is untouched code.** A solo request still takes the drafted single-stream path
  through `Engine::generate`'s step, and interactive latency cannot move.
- **Two to eight requests are untouched code.** `HALO_SERVE_ROUTE_MIN` is 9 by default, which is the
  first width that would have needed a second eight-row pass. Below it `batched_step` makes exactly
  the `Engine::forward` call it always made. `b6c614a9` measured the crossover from the other side
  and found it in the same place: at eight rows the wide route loses 1.6%, at nine it wins 12.6%.
- **No kernel, no image, no weight, no serving default.** The FFN slice width is `FMAX` as it was
  and the head is the head.

## It is inert on the installed service, and that is a slot count

`~/.pi/agent/local-models.json` starts the service with `--slots 4`, so at most four requests decode
together and a step never reaches nine rows. **This change is worth exactly zero there**, and the
fifth concurrent request queues rather than joining a pass.

The arithmetic of the alternative, which is an operator decision rather than a lane one: a slot
costs its KV cache plus its recurrent state, 90112 bytes per token of context and 151 MB of state.
Four slots at the 32768-token default is 23.1 GB. **Sixteen slots at 8192 tokens is 21.8 GB** - the
same footprint, four times the concurrent decode width, and the served model already advertises
`maxTokens: 8192`. It is not free: a client that sends a 12000-token prompt today would be refused.
That trade is named here with its numbers rather than taken.

## Commands

```sh
tools/run-batch-compare --pin-clock --tag serve-decode-route/route-ladder --only decode \
    --modes 0,4,20 --streams 16,32 --slots 32 --context 256 --gen-steps 6 --rounds 1 --warmup-steps 2

tools/run-batch-compare --pin-clock --exec python3 tools/serve_load.py \
    --exec ./bonsai-halo --dflash ~/data/bonsai2/drafters/dflash2.safetensors --context 2048 \
    --arm sliced:32:HALO_SERVE_ROUTE_MIN=999 --arm batched:32 \
    --concurrency 32 --max-tokens 24 --check --out RESULT.json
```

`tools/serve_load.py` arms are now `NAME:SLOTS[:VAR=VALUE,...]`, so one lock hold prices two routes
of one executable against one clock. `HALO_SERVE_ROUTE_MIN=999` is the control arm: it is the server
this replaced, instruction for instruction.

The other open item here - *a prompt still blocks every decode* - is answered by
[chunked admission](serve-prefill-admission.md): the worst inter-token gap falls 7.3x to 17.3x with
aggregate throughput unchanged, and the panel that found it is `tools/serve_mixed.py`, which posts
requests on a schedule with distinct prompts. A synchronised burst cannot see that defect, and it
also turns out to be the pattern in which serialized admission costs nothing: it makes fewer and
wider steps.

# The resident service was ingesting prompts through the eight-row route

This one is not a kernel. It is the default that decides which of two routes the product runs, and
it had been pointing at the slow one for three route improvements in a row.

Measured on the **live service**, not on a panel. One `POST /v1/chat/completions` to the running
`bonsai-halo.service`, 1941 prompt tokens, `max_tokens` 8:

    "prefill_seconds": 13.935507580001286     ->  139.3 tok/s

The same executable has a **bit-identical** prompt route that does about three times that.
[Sixteen-row FFN slices](ffn-slice-width.md) published 408.3 tok/s on it with the DFlash2 drafter
loaded and the identical greedy token digest; [a drafter no longer turns wide prompt ingestion
off](wide-drafter.md) exists because that route used to refuse a loaded drafter at all, and closed
with "the resident server ingested prompts at about 145 tok/s while the route that does 250 sat
behind a capability gap rather than behind a measurement".

The capability gap closed. The default did not move.

## The mechanism, in three lines

`src/main.cpp` had `int prefill_ffn = 0`, and nothing but `--prefill-ffn` ever wrote it. So:

- `Engine::prefill_batch_mode` stayed `0`,
- `Engine::prefill_width()` returns `RMAX` when `prefill_batch_mode` is 0,
- and `Engine::prefill` therefore fed every served prompt to the persistent kernel eight rows at a
  time, re-reading the whole weight stream once per eight tokens.

`bonsai-halo.service` starts as `serve --port 8471 --dflash ...` with no route flag, so the resident
engine took that path every time, and so did every `./bonsai-halo -p ...` invocation.

## What the route is, and why this is not a numerical change

Mode 20 (`wide-deployed`) is the wide **schedule** over the **deployed** FFN arithmetic: wide
sequence projections, resident GDN state with direct commit, wide attention, wide prep, and the
engine's own five-trit ternary FFN through `k_ffn_slice`. There is no A4 map anywhere in it and no
approximation of any kind. [Wide prompt ingestion](wide-prefill.md) separates the two halves of the
wide route's gain for exactly this reason: the schedule is worth 1.70x at the deployed FFN's own
arithmetic, and the approximate FFN map is a separate 2.74x that this route does not take.

Prompt ingestion is also the one phase where a committed pass costs nothing to commit: a prompt's
tokens are never rejected, so the direct state commit that makes a wide pass illegal for a
speculative verify pass is free here.

## What changed

`prefill_ffn` starts at `-1`, meaning "nobody asked", and the default is resolved after parsing:

```c++
if (!prefill_ffn_asked)
    prefill_ffn = (batch > 0 || v1 || !mtp_path.empty() || ffn_mode || !prefill_sweep.empty()
                   || !dump_dir.empty()) ? 0 : 20;
```

Every path that **cannot** take the route keeps the eight-row one, and each for its own reason:
`--batch` drives `batch_mode` itself, `--v1` is the per-op reference path, `--mtp` has no per-row
final hidden on a wide pass, `--ffn` is a whole-run route selection, `--prefill-sweep` sets the
route per arm, and `--dump` writes per-op reference activations. `--prefill-ffn off` pins the old
route explicitly, which is what every historic eight-row ingestion number should be reproduced with.

The guard that rejects the flag on unsupported paths now fires only when the flag was actually
passed, so `--batch`, `--v1` and `--mtp` keep working without it.

Nothing else moved: not `prefill()`, not `prefill_width()`, not `forward_batch`, not one kernel.
The startup work is the same two calls `--prefill-ffn` already made, `prepare_batch(pass_rows,
BATCH_WORKSPACE_ONLY)` and `prepare_sequence()`.

## Measured

One process, one loaded model, one clock, the DFlash2 drafter loaded exactly as the service loads
it, both routes walked as a case axis by `--prefill-sweep`. 932-token prompt, greedy, eight
generated tokens, on canonical `b4e5365`:

| | `--prefill-ffn off` (was the default) | `wide-deployed` (is the default) | |
|---|---:|---:|---:|
| **prompt ingestion** | 5.451 s, **171.0 tok/s** | 2.001 s, **465.7 tok/s** | **2.72x** |
| drafted generation | 50.42 tok/s | 50.91 tok/s | null |
| drafted / accepted | 21 / 5 | 21 / 5 | equal |
| verify per step | 47.243 ms | 46.729 ms | null |
| greedy token digest | `17821512331856351063` | `17821512331856351063` | **equal** |

The same pair on the previous canonical build (`8d87e77`, before `b4e5365` gave the eight-row route
its block of cooperative grid back) read 144.7 against 488.9 tok/s with the same digest, so the
baseline arm moved 18% under me and the wide arm did not. **Re-take a pair that straddles an
install.**

What an unconfigured run does now, and the control, one lock acquisition, same binary:

```
arm 1, no route flag:  wide prompt ingestion: mode 20 (default), 256 rows per pass, +2.25 GB
                       932 tokens in 2.392 s = 389.6 tok/s, digest 17821512331856351063
arm 2, --prefill-ffn off:  932 tokens in 5.691 s = 163.8 tok/s, digest 17821512331856351063
```

Cross-process spread on this box is larger than some of the effects this lane measures - the same
route 20 reads 465.7 in the pair and 389.6 as the first process of the second panel - which is why
the pair is the result and the single-arm runs only show that the default resolves and the control
restores the old route.

**Bit identity: one greedy token digest, `17821512331856351063`, across both routes, both builds and
four processes**, with identical draft and acceptance counts in every cell, so the drafter sees the
same captured features on either route.

### On the live service

The product number is `prefill_seconds` from `POST /v1/chat/completions`, which needs no GPU lock
because it is the service doing its job. Before the change, on the binary the service was running:

    1941 prompt tokens, prefill_seconds 13.936  ->  139.3 tok/s

**The post-change HTTP sample was not taken, and the reason is worth knowing.** Fourteen engineers
were running panels, every measurement wrapper stops and restarts the resident service, and it was
never up for the four seconds the request needs; two forty-second waits found no window. What is
certain is that the service now *runs* the route: `journalctl --user -u bonsai-halo` shows every
start carrying

    model ready: 10.11 GB on device, 0.9 s
    wide prompt ingestion: mode 20 (default), 256 rows per pass, +2.25 GB on device
    dflash2 drafter loaded, weights q4 (13.55 GB on device total)

and the route pair above is the same work on the same binary through the same `Engine::prefill`.
Take the HTTP sample on a quiet box and add it here; the request is in the reproduction block.

### Installed acceptance

Canonical `5eb5fcb`, installed `bonsai-halo`
`9bfa228cc3a29829747244d78d864b8eb672b688c5a40bb99e7c93b70a582b3d`, same panel:

| | `--prefill-ffn off` | `wide-deployed` (default) | |
|---|---:|---:|---:|
| prompt ingestion | 167.8 tok/s | **485.8 tok/s** | **2.90x** |
| drafted generation | 50.60 tok/s | 50.98 tok/s | null |
| verify per step | 46.985 ms | 46.692 ms | null |
| greedy token digest | `17821512331856351063` | `17821512331856351063` | **equal** |

## Costs

**+2.25 GB of device memory** and about a second of startup, both paid once when the process opens:
`prepare_batch(256, BATCH_WORKSPACE_ONLY)` takes the shared pass workspace and no weight image, and
`prepare_sequence()` builds the sequence projections' code images. The resident engine goes from
about 11.3 GB to 13.55 GB on device with the drafter loaded, on a host with 128 GB of unified
memory, and the whole of an unconfigured process - load, drafter, prepare, 932-token prompt and
eight tokens - is 4.59 s, less than the old route spent on the prompt alone.

The allocation happens after the weight images are in place, so it does not move them; but it is a
new 2.25 GB in every default process, and this lane has measured the allocator's placement moving a
kernel's rate by double digits. A panel that wants the historic allocation pattern should pass
`--prefill-ffn off`, which skips both calls.

## What this does not change

- **Generation does not move**, by construction and by measurement: the route covers prompt
  ingestion only, and a drafted step still runs the eight-row verify pass.
- **`tools/batch_compare` and `tools/batch_profile` are unaffected.** They select their own mode and
  never read this default, so every published panel keeps its route.
- **Every published eight-row ingestion number is still reproducible** with `--prefill-ffn off`.

## Controls and reproduction

```sh
# the pair, one process, one loaded model, one clock, with the drafter the service runs
tools/run-batch-compare --engine --bench --dflash ~/data/bonsai2/drafters/dflash2.safetensors \
  --prefill-sweep off,wide-deployed \
  -p "$(cat ../../data/bonsai2/batch-comparison/_inputs/serve-prompt-500w.txt)" -n 8

# what an unconfigured run now does: the route line names the default it resolved
tools/run-batch-compare --engine --dflash ~/data/bonsai2/drafters/dflash2.safetensors \
  -p "$(cat ../../data/bonsai2/batch-comparison/_inputs/serve-prompt-500w.txt)" -n 8

# the product number, through the service, no lock: prefill_seconds in the halo block
curl -s localhost:8471/v1/chat/completions -H 'content-type: application/json' \
  -d "$(python3 - <<'PY'
import json
print(json.dumps({"model":"bonsai-2-27b","max_tokens":8,"temperature":0,
 "messages":[{"role":"user","content":' '.join(open('PLAN.md').read().split()[:1000])}]}))
PY
)" | python3 -c 'import json,sys; print(json.load(sys.stdin)["usage"]["halo"])'
```

Raw samples and the panel output are in
[`batch-comparison/serve-prefill-route/`](../../../data/bonsai2/batch-comparison/serve-prefill-route/).

## What is left on this path, with its price

The route this default now selects spends about 60% of a 128-row pass in `k_ffn_slice`, and that
kernel is capped at 32 rows per launch, so a 128-row pass reads the whole 3.42 GB five-trit FFN
stream **four times**. [The second column group](deployed-matvec-groups.md) took it from 16 rows to
32 for -20% on the phase and stopped at the register wall (238 VGPR with `waves_per_eu(6)`, against
this device's 240-register six-wave cliff).

`MV_GROUPS = TT > 16 ? (TT + 15) / 16 : 1` already derives the group count from the row count, so
`FMAX` 32 to 64 is what asks for four groups. The published census gives the price without a run:
432 instructions per 128-K block at one group, 652 at two, so +220 per group, and 1092 at four
groups for 64 rows — **17.1 instructions per row against 20.4**, plus half the weight stream again.
Against that, the operands are fenced per group so the added liveness is the accumulators only,
8 VGPRs a group: 254 registers, which is past the 240 cliff and buys five waves per SIMD32 where
two groups get six.

**A 10% issue-cycle cut against a sixth of the occupancy is not obviously positive, and it is one
compile away from being known.** The compile is the obstacle: `kernels/halo_rows.o` device codegen
is past 55 seconds since `b4e5365` added two `k_forward_rows` instantiations, and `tools/split-build`
step 0 no longer fits a bounded tool call, so this needs either a quiet box or `k_ffn_slice` in a
probe instantiation of its own. `tools/kernel_resources.py kernels/halo_rows.hip k_ffn_slice` is the
reading that decides it.

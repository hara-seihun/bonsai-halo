# The exact state coordinate can defer its write-back too, and it is worth 31% of a batched step

A 32-stream generation step spent **45% of itself in `gdn-resident-core`** — 45.2 ms of a 99.5 ms
device span, against the FFN's 32.5 — and that phase was the one part of this engine already
finished by its own diagnosis. It moves 9.664 GB of fp32 recurrent state per step and
[the storage coordinate](gdn-state-pack.md) measured it at **224 GB/s of the 242 GB/s `bench/bw`
reaches**. No schedule, census or occupancy change can move a phase at its byte roof. The only
lever is bytes, and **half of those bytes are the write-back**.

[The deferred commit](gdn-state-defer.md) already built the write-side lever and shipped it for
packed coordinates only, for a reason that is address arithmetic rather than arithmetic:

> fp32 keeps no room for them and so cannot defer, and does not need to: it is the coordinate this
> is composed with rather than an alternative to it.

`gdn_defer_effective()` returned 1 whenever the state was fp32 and `gdn_region_for()` handed every
format `GDN_STATE_FLOATS`, so an fp32 region ended exactly where its values ended. That is a
**reservation**, not a numerical fact. This iteration gives the exact coordinate the room, replays
the pending terms in a form that reproduces the eager chain **bit for bit**, and turns it on by
default.

| 32 streams, mode 19, f32 state, traced, arms interleaved in one process, two rounds | `gdn-resident-core` | device span | aggregate |
|---|---:|---:|---:|
| `--gdn-defer 1`, the eager path this engine shipped | 45.184 / 44.930 ms | 99.48 / 99.13 ms | 319.6 / 319.7 tok/s |
| `--gdn-defer 0`, deferred kernels committing every step | 44.399 / 44.448 | 98.52 / 98.49 | 321.9 / 322.0 |
| **`--gdn-defer 2`, the new default** | **26.316 / 26.593** | **75.24 / 76.05** | **420.1 / 414.7** |
| `--gdn-defer 4` | 26.319 / 26.442 | 75.23 / 75.60 | 419.1 / 414.5 |

**-41.3% of the phase, -23.9% of the step, +31% aggregate batched generation, and the residual
FNV-64 is `6678137683989106976` in all eight samples of all four arms** — the hash this lane
published for this shape before the code existed.

## Why it can be exact here, and could not be there

`gdn_token` advances one state element as one rounded multiply and one fused multiply-add:

```c++
m[r][s2] *= t.g;                                  // decay
m[r][s2] = fmaf(d[r], t.kk[s2], m[r][s2]);        // this token's rank-1 term
```

An fp32 store loses nothing, so the sequence of values an element takes through C eager steps is
exactly the sequence the same C `(g, d, k)` triples produce when they are replayed **in
chronological order** with that same pair of operations. Everything downstream follows by
induction: the k-side contraction sees the same state, so the delta is the same, so the next term
is the same, so the logits are the same bits.

The shipped rebuild does not do that. It sums the suffix-product form

    S_C = (prod_t g_t) B + sum_i (prod_{t>i} g_t) a_i (x) k_i

which is algebraically equal, one multiply per element per term cheaper, and **the right choice for
a packed coordinate** — there the eager path rounds the state to eight or sixteen bits on every
commit and the deferred path does not, so bit-identity is unreachable in either form.
`gdn_defer_rebuild` now carries both: the chain form at an exact coordinate, the suffix form at a
packed one, selected by `gdn_fmt_is_exact` inside the function that already switches on the format.

### `warp_sum` leaves a wave two deltas, and a pending term has to keep both

The chain form alone measured **23 of 32 identical greedy continuations**, not 32. The cause is a
property of this recurrence that [the lane layout](gdn-lane-layout.md) established and this is the
first change to be caught by:

> `warp_sum` does not leave one value in a wave, it leaves two ... a wave ends a reduction holding
> exactly two bit patterns of the same mathematical sum, selected by bit 2 of the lane index

`gdn_token` applies **lane `l`'s** delta to the columns lane `l` owns, so column `c` is updated with
the delta of class `((c & 31) >> 2) & 1`. A pending term that keeps only lane 0's delta replays one
class across all 128 columns. At a packed coordinate that difference is far inside the state's own
rounding; at fp32 it is the whole difference, and eight steps of a 32-stream trajectory turned it
into nine different continuations.

An exact term therefore stores **both classes** — one lane of each writes its own — and the rebuild
reads the one its lane's columns were updated with. The delta block per term goes from `SS` floats
to `2 * SS`, the term from 49.3 kB to 73.9 kB per (slot, layer) against a 3.146 MB state, and the
timing is unchanged to within the panel's spread. With both classes the same workload is **32 of 32
identical**.

That is also the shape of the general warning: **a change that replays a reduction's result rather
than the reduction has to replay it per lane class.** Nothing else in this engine stores a
cross-lane reduction's output for later reuse today; the next thing that does will meet this.

## What it costs, which is memory rather than time

An exact region grows from `GDN_STATE_FLOATS` to 866,816 floats per (slot, layer), **+10.2%**, and
a 32-slot engine's tracked device memory goes 14.722 GB to 15.281 GB after load. The pending list
is reserved only when a process may defer (`gdn_defer_reserve_depth`, or the default), and a packed
region is unchanged down to the byte — `gdn_fmt_defer_head(GDN_STATE_I8)` still evaluates to the
`GDN_DEFER_HEAD` it shipped with, and a `static_assert` says so.

The exact list is capped at `GDN_DEFER_EXACT_MAX = 4` rather than 8, for two reasons: the measured
ladder is flat from 2, and four terms of two classes stage through exactly the arrays eight terms of
one class already needed, so the LDS of the state kernel and the flush kernel does not move.
`gdn_defer_effective()` reports the depth that ran and both tools record it per sample.

## The routes that read a state without rebuilding it

A deferred region holds its head's state as a base plus up to four rank-1 terms. Every route that
reads the state directly has to settle first, and all of them now do:

- the eight-row sliced route and the wide sequence parts already called `gdn_defer_settle`, which is
  format-blind and so covers fp32 unchanged;
- **the column arm** (`resident_state_cols`, [the lane layout](gdn-lane-layout.md)) owns a row
  across four lanes and has no rebuild. It used to be excluded from deferring processes outright,
  which would have cost every prompt pass its 19.7%. It is chosen only when a pass carries at least
  eight rows per sequence — a shape that commits anyway — so the launcher flushes the pass's slots
  and then takes it;
- **the single-sequence reference route** (`Engine::forward_token`) reads the region in place. It
  now asks `gdn_defer_flush_pending`, which is a no-op for a slot with nothing pending, and its
  state pointer walks `gdn_region_floats()` rather than `GDN_STATE_FLOATS`, which was a latent
  stride bug the moment any region grew;
- the `--state-dump` path in `batch_compare` and the `HALO_GDN_COMPARE` diagnostic, which compare
  durable images, settle and are gated respectively.

## Acceptance

**Bit-identical, three ways.** Residual FNV-64 `6678137683989106976` in every sample of the decode
panel above. A 128-row prompt pass hashes `3240780174642567821` and a 32-row pass
`8216813404010667075` under both arms — the hashes the four-bit sequence default published before
this code existed. 32 streams × 8 greedy steps, whole continuations compared as text: **32 of 32
identical** against the eager arm, and 32 of 32 for the `--gdn-defer 0` control.

**Prefill is a null by construction and by measurement.** A pass that walks more than one token per
sequence has nothing to defer and writes its state in full, so the only thing the deferred build
changes there is which instantiation runs. 128 rows: `gdn-resident-core` 10.943 / 11.139 ms eager
against 10.816 / 11.130 deferred, device span 136.07 / 140.61 against 134.90 / 139.32. 32 rows:
3.973 / 4.089 against 4.036 / 4.037.

**The mixed commit/replay acceptance passes at every depth.**
`tools/run-batch-compare --state-check` is green at depths 0, 1, 2 and 4 on this build, seven runs.
One earlier run at depth 4 reported a failing scenario and **did not reproduce** on six later runs
of the same executable. It is worth recording what that run was, because it is evidence about the
instrument rather than about this change: the failing scenario is `modes 10 and 16 mixed`, and
`forward_batch` gives a pass the resident route only when `batch_mode >= 17`, so **that scenario
never launches the kernel the deferral lives in**. Its signature — one row group of logits out of
seventeen differing by up to 0.07, with every argmax still matching and the durable state bitwise
equal — is the signature of a summation order changing, not of a state being rebuilt wrongly. The
cooperative route's `KS = 2` matvec `atomicAdd`s two partial sums into one residual float and is
deterministic only under a bound on units in flight. **Anyone bisecting a `--state-check` verdict
should run each cell more than once**; a single red cell is not yet a fact.

That is not a hypothesis about someone else's result any more. The same instrument, the same
canonical executable `0a25c4c7b164a480aa5ffb4f`, `HALO_GDN_STATE=i8`, four runs: **green, green,
and one run with scenario 1 red** — one row group of 17 differing, `max_abs_diff` 0.063, every
argmax matching, which is the fp32 flake's signature exactly.
[The region stride](state-region-stride.md#the-acceptance-that-is-not-green-and-it-is-not-this-change)
bisected "fails at every packed coordinate on current main" one run per cell and left the packed
region a selector on that evidence. **The packed coordinate may not be red at all**, which turns
the +37% aggregate that [the storage coordinate](gdn-state-pack.md) measured back into a default
question with a quality panel attached rather than a blocked one. The instrument's own next step is
to decide what it asserts: raw bits cannot separate a wrong rebuild from the cooperative route's
jitter, and argmax plus a tolerance can. Raw runs are in
[`batch-comparison/gdn-defer-exact/state-check-flake/`](../../../data/bonsai2/batch-comparison/gdn-defer-exact/README.md).

## Where the phase went, and what it did to its neighbours

The step is 23.9% shorter and only 18.7 points of that are the state phase itself. Two phases this
change cannot reach moved with it, and in opposite directions:

| phase, 32 streams | eager | deferred | |
|---|---:|---:|---|
| `gdn-resident-core` | 45.184 | 26.316 | **-41.8%** |
| `ffn` | 32.549 | 27.953 | -14.1% |
| `sequence-output-projection` | 6.851 | 5.870 | -14.3% |
| `sequence-input-projection` | 8.381 | 8.457 | +0.9% |
| everything else | 5.67 | 5.74 | +1.2% |

The two that fall are the two that **follow** the state kernel in a layer; the one that rises
precedes it. A 32-stream step's eager write-back is 100 MB per layer and it is still draining when
the next kernel starts, so the phase after the recurrence has been paying for it. That makes the
"untouched phases" control unusable for this particular change — the honest headline is the device
span and the aggregate rate, both of which are in the table above — and it is a fact worth having
for anyone who cuts a large write anywhere in this engine: **the phase that benefits may not be
yours.**

## What this does not reach

A pending term exists only where a pass carries **one token per sequence through the resident
route**, so the change is exactly as wide as that route:

- **The service as it runs today is untouched.** `bonsai-halo serve --slots 4` decodes on the
  sliced route, which never launches the resident kernel, and its drafted single-stream path
  reproduces its digests and its drafted/accepted counts exactly under both arms.
- **Prompt ingestion is untouched**, by construction and by measurement: a pass with more than one
  token per sequence has nothing to defer.
- **A step below the wide floor is untouched**: `forward_batch` gives a pass the resident route only
  at 32 rows or more.
- **64 fp32 slots still do not fit**, with or without this change: `prepare_batch` refuses at 23.98
  GB tracked in the eager arm and 25.04 in the deferred one, both against a 28 GiB budget that the
  A4 images then exceed. That cell was already closed by
  [the region stride](state-region-stride.md) and this makes it 1.06 GB tighter.

What it does reach is the shape every aggregate-generation number in this lane is quoted at, and
the route a served batch will take the moment it decodes 32 rows at once.

## The depth ladder is flat from two, and that is the interesting number

2, 4 and 8 are one measurement (26.316 / 26.319 ms at 2 and 4). The arithmetic says why: at depth 2
the write is already a third of the traffic instead of a half, and the phase becomes its **read**,
which no depth can touch — 4.83 GB of state read per step is 22.4 ms at the 216 GB/s this phase
sustains, and the deferred arm measures 26.3. So the remaining 4 ms is the rebuild, the triples and
the ring, and **the next lever on this phase is the read, not the write**: a packed coordinate cuts
it 4x and has its own (approximate) map, and nothing else in sight does.

That also re-prices the packed default this lane ships: [the deferred commit](gdn-state-defer.md)
selected depth 4 there, and the same flat ladder may well apply, with a shorter pending list and a
smaller flush. It is one panel and nobody has run it.

## Using it

`HALO_GDN_DEFER=N` sets the depth for a process, `--gdn-defer` walks it in either measurement tool,
and `HALO_GDN_DEFER=1` is the off switch that restores the eager write-back exactly. The default is
**2 at an exact coordinate and 1 at a packed one**: the packed map was measured eager and no
published packed number moves without someone asking for it. A process that sweeps the depth must
reserve it before the engine allocates — both tools do — because the region stride freezes at the
first allocation and a told depth is a contract.

```sh
tools/run-batch-compare --pin-clock --profile-tool --modes 19 --decode-streams 32 \
  --decode-prompt 64 --rows 32 --gdn-state 0 --gdn-defer 1,0,2,4 --traces 1 --rounds 2 \
  --out DATA/depth-ladder2.json
python3 tools/phase_totals.py DATA/depth-ladder2.json

HALO_GDN_DEFER=4 tools/run-batch-compare --pin-clock --state-check --out DATA/state-check.json
```

Raw samples, the multistep continuations of every arm and the state-check verdicts are in
[`batch-comparison/gdn-defer-exact/`](../../../data/bonsai2/batch-comparison/gdn-defer-exact/README.md).

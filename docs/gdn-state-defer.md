# The recurrent state does not have to be written every step

> **The exact coordinate takes this too, and there it is bit-identical.**
> [`gdn-defer-exact.md`](gdn-defer-exact.md) gives an fp32 region the room for a pending list and
> replays the triples in the order `gdn_token` applied them, which is worth **-41% of the state
> phase and +32% aggregate batched generation with no output bit moved**. Two corrections to this
> document follow from it: an exact term has to keep **both** of `warp_sum`'s roundings, not lane
> 0's alone; and the depth ladder is **flat from two** at fp32, where this document selected four,
> which is worth re-measuring at the packed coordinate.

[The state's storage coordinate](gdn-state-pack.md) halved the bytes of a round trip that a
generation step makes twice, and closed by naming the other half of the same lever: the step reads
the state and writes it back, and only the read is forced. This is that half. At depth 4 a 32-stream
step goes from 88.30 ms to 79.57 ms on the device, **+11.0% on the step and +7.4% on full-model
aggregate generation**, on top of the coordinate's own +22%, with the packed state rounded four
times less often rather than more.

## What a step actually does to a head

Both consumers of a state row contract all 128 of its columns, so a step cannot produce its output
without reading every element. The write-back is a different matter. What one token does to a head
is scale it and add one rank-1 term:

    S_t = g_t S_{t-1} + a_t (x) k_t

for the token's decay `g`, the per-row delta `a` the delta rule produced and the token's L2
normalised key `k`. So C steps of a stream are C such triples — 1 kB per head against the 32 kB row
image they modify — and the state C steps later is

    S_C = (prod_t g_t) B  +  sum_i (prod_{t>i} g_t) a_i (x) k_i

with `B` the last committed base. Keeping the triples and rebuilding that sum at load lets the full
write happen once per C steps. Traffic per step goes from 2 to 1 + 1/C plus the triples.

**Where the triples live.** A packed head uses at most half of the `GDN_STATE_FLOATS` its
(slot, layer) region reserves, so they sit above the scales in the same region. Every reset, slot
copy, snapshot and rollback already moves that region, so they move with the state they belong to,
and a zeroed region is an empty pending list over a zero base in every format — no allocation, no
new lifecycle, nothing else to keep in step. fp32 fills its region and therefore cannot defer, which
costs nothing: deferring is the thing you compose *with* a packed coordinate, not an alternative
to it.

**What the rebuild does not touch.** It is element-wise — one fma per element per pending term — so
the token loop, its summation tree, its `warp_sum` butterfly and its fused multiply-adds are the
code they were. What changes numerically is that the decayed base and the pending terms are summed
in a different order than the step-by-step chain sums them, and that in a packed coordinate the
state now crosses memory once per C steps instead of every step. The second runs opposite to the
first, and the quality panel below measures the pair.

## Measured: the cycle, not a step

A deferred run has two kinds of step and they cost different things, so a single timed step measures
whichever one it lands on. `tools/batch_profile --decode-prime N` runs N untimed steps first, which
is what picks. Mode 19, 32 streams, one token each, int16, one process, one build:

| step | commit every step | defer 4 |
|---|---:|---:|
| pending 0 (append) | 88.930 ms | **74.145** |
| pending 1 (append) | 88.129 | 75.235 |
| pending 2 (append) | 87.962 | 76.972 |
| pending 3 (commit) | 88.160 | 91.904 |
| **cycle mean** | **88.30** | **79.57** (-9.9%) |

Full model, same shape, 16 generation steps across 32 streams through `tools/batch_compare`:
**366.6 tok/s aggregate committing every step, 393.7 deferring at depth 4, +7.4%.** The gap between
+9.9% on the device and +7.4% end to end is the host's own per-step work, which this change does not
touch. Raw: `batch-comparison/gdn-defer/decode-cycle.json`, `gdn-defer-final-i16/`.

**Depth 4 is the best depth, and the reason is in the table.** In one process, 16 steps each:
**398.1 tok/s at depth 2, 410.0 at depth 4, 404.7 at depth 8.** The per-term rebuild cost (+1.1 ms
for the first pending term, +1.7 for the second) eventually eats the write it saves, and the write
is only 7.5 ms of a 24.4 ms phase, so nothing much past a handful of steps can pay. The same
ordering held on the first version of the rebuild (383.1 / 393.1 / 389.6), with every arm lower.

**Compare inside a panel, not across them.** This box runs eight engineers against one GPU, and the
same arm reads 393.7 in one process and 410.0 in the next. Every pair in this document is
interleaved inside one process on one build; the spread across processes is about 4%.

**Installed acceptance**, canonical `a81a842`, `bonsai-halo`
`b578c72bea93d01430f646dd607092c377925c3c4786116944aef8ef7f71f049`, `tools/batch_compare`
`80a575817f8dcf92006c9e4564b5857a8b30bd1961e75996a5a5c90ec798b797`, under
`tools/run-batch-compare`:

| on the installed build | commit every step | defer 4 |
|---|---:|---:|
| append step (pending 0) | 89.674 ms | **73.761** |
| commit step (pending 3) | 87.865 | 93.081 |
| aggregate generation, 32 streams, 16 steps | 361.0 tok/s | **409.3** (+13.4%) |

The residual FNV-64 of every zero-pending sample is `14246133829617533007`, the value three
development processes and the published int16 arm produce. Single-stream `--bench -n 60` on the same
binary is **33.39 tok/s**, inside this box's 33.05-33.80 band and untouched by construction: the
default coordinate is fp32, which cannot defer, so the single-token path runs the instantiations it
always ran. Raw: `batch-comparison/gdn-defer/installed-acceptance.json`,
`installed-acceptance-flush.json`, `gdn-defer-installed/`.

Phases of a traced step, same process (`tools/phase_totals.py` groups them):

| phase | commit every step | defer 4, append step |
|---|---:|---:|
| `gdn-resident-core` | 24.389 ms | **16.930** |
| `ffn` | 33.126 | 28.972 |
| `sequence-output-projection` | 11.373 | 8.114 |
| `sequence-input-projection` | 14.058 | 14.119 |
| device span | 91.38 | 76.86 |

Two phases that cannot see the state move with it. They are downstream of the state kernel in the
same stream, and a write-back that is still draining is charged to whoever runs next: removing it
speeds up the FFN and the output projection as well as the phase that owned it. That is also why the
end-to-end gain tracks the step time rather than the phase.

## Bit-identity, and the control that isolates the kernels

At zero pending the rebuild is `m = 1.0 * B` and the step is the plain packed path. It measures that
way: residual FNV-64 `14246133829617533007` — the value the published int16 arm produces — from the
deferred kernels at `--decode-prime 0`, in every panel. From the first pending term on, the arms
diverge by construction, and the per-step hashes differ.

`--gdn-defer 0` is the control that separates the instantiation from the deferral, the way `f32pk`
separates the packed kernels from the packed coordinate: the deferred kernels run, with their extra
registers, their LDS and their rebuild path, and commit every step anyway. Prefill at 128 rows per
pass, warm round: **778.1 tok/s at depth 0, 779.4 committing every step, 782.4 at depth 4** — a dead
heat, and a null for prompt processing by construction (a prefill pass pays the round trip once for
128 tokens, so there is nothing to defer; the control rule commits any pass that walks more than one
token per sequence).

**Read a cold round as a cold round.** In round 0 of that same panel the three arms read 690.7,
794.0 and 704.7 — the first cases in a process are on a clock that takes about 4.6 s to reach its
ceiling and on instantiations nothing has touched. The deferred arms happened to be first. Round 1
is the measurement.

Registers, `tools/kernel_resources.py`: the non-deferred instantiations are byte-identical to the
ones canonical ships (45/42/60/60/46/45 VGPR), the deferred ones cost 2-4 more and 4132 bytes of
LDS, and all of them stay at 16 waves per SIMD32.

## Quality

**The shape matters more than the metric here.** The prefill and decode quality shapes both feed
their context in wide passes, so the state is read back once per pass rather than once per token and
a deferred commit never engages at all. The measurement has to be the generation shape, walked long
enough for rounding to accumulate — which is exactly what [the horizon
workload](gdn-state-pack.md) was built for by the engineer chasing int8's accumulation, and
`--gdn-defer` is a second axis over it. `tools/gdn_state_quality.py` keys an arm on the coordinate
*and* the depth, so `i16` and `i16+d4` are separate arms rather than one overwriting the other.

`tools/run-batch-compare --only horizon --modes 20 --gdn-state 0,2 --gdn-defer 1,4
--horizon-tokens 32 --horizon-ctx 128 --horizon-streams 32`, mode 20 so the exact FFN does not mask
the state, 32 streams each walking their own window, 256 scored predictions per arm, every arm
paired per (stream, step) against the fp32 arm of the same process. The int8 arms are a second run
merged by the tool, which is safe for a quality panel and not for a timing one: the engine is
deterministic on fixed inputs, and the fp32 arm reproduces `2.45995` in both processes.

| state | ΔNLL vs fp32 (nats) | greedy agreement | NLL |
|---|---:|---:|---:|
| int16 | -0.00035 ± 0.00101 | 0.9961 | 2.45959 |
| int16, defer 4 | -0.00019 ± 0.00104 | 0.9961 | 2.45976 |
| int8 | +0.00021 ± 0.00151 | 0.9922 | 2.46016 |
| int8, defer 4 | +0.00060 ± 0.00164 | 1.0000 | 2.46055 |
| fp32 reference | | | 2.45995 |

Four arms inside one standard error of each other and of zero, with stream-clustered errors. Per
bucket of accumulated rounding the deferred arms track their own coordinate rather than drifting
from it: int16 runs `+0.0020, -0.0007, -0.0027, +0.0000` across the four quarters of the walk and
int16 at depth 4 runs `+0.0017, -0.0028, -0.0021, +0.0024`, each bucket inside its own error bar.
Deferring the commit at depth 4 costs nothing this instrument can resolve, against the +0.0615 nats
the A4 FFN map this repository serves as its batched headline costs.

A longer panel in this change's own checkout agreed, on a workload that is not in the tree because
the `horizon` one landed the same shape while this was in flight: 768 paired predictions over 24
steps gave int16 +0.00099 ± 0.00071, int16 at depth 4 +0.00107 ± 0.00072, int8 +0.00067 ± 0.00082
and int8 at depth 4 +0.00028 ± 0.00087. Two instruments, two sign conventions on a difference that
is smaller than either can resolve, and the same conclusion.

**The panel earns its keep.** The rebuild's first fast version handed a float to
`__builtin_amdgcn_readlane`, whose operand is an integer, so the value was converted instead of the
bits: every pending delta became 0 and the state quietly lost its updates. Step time did not move,
the residual hash of a zero-pending step was still bit-identical, and nothing else in the engine
noticed. NLL went 2.406 to 2.671. **A rebuild that produces plausible numbers is not a rebuild that
produces the right ones, and only an absolute quality metric on the shape that uses it will say.**

**int8 composes with this, and int8 is the coordinate the lane now recommends.** [The horizon
panel](gdn-state-horizon.md) settled its precision over 256 roundings per layer: `-0.0165 ± 0.0082`
nats, no drift across the horizon. Deferring the commit rounds a packed state once per C steps
instead of every step, so it can only move that number in the safe direction. int8 at depth 4 is
the fastest arm measured here — **427.9 tok/s aggregate against 410.9 committing every step**, in
one process, where fp32 reads 295.2 — and on 768 paired predictions it is the closest packed arm to
fp32 of the four.

## What the rebuild costs, and why it is shaped this way

The first working version put a global load and a serial multiply inside the per-row loop, and cost
**8.3 ms per step at three pending terms — as much as writing the whole state**. The ISA says why:
the loop body waited on `vmcnt(0)` for a per-lane load of a wave-uniform value, and carried the
decay product across iterations, so every term was a dependent memory round trip.

Three changes took it to 3.6 ms, and each is worth carrying:

- **The keys go through LDS.** Every row of a head multiplies the same 512-byte key vector and a
  head's rows are 32 waves, so reading it per row asks for the same bytes 32 times.
- **The weights are precomputed once per block.** `w_i = prod_{t>i} g_t` is serial and eight long;
  computed inside the row loop it separates every term from the next by a dependent multiply. One
  thread computes them into LDS while the block is staging keys.
- **The loop runs over terms, not rows.** A term's delta column is then one coalesced load for the
  whole wave — the wave owns rows `part*RPU + wave + 8r`, which is lane `wave + 8r` of that load —
  and `readlane` moves it into place. Nothing in the body depends on the previous iteration.

What is left is close to the traffic floor: per pending term a step moves the delta column once per
head and the key once per block, 2.5 kB per head, which is 0.77 ms per term at this device's read
rate against the 1.1 ms measured for the first term.

## Contract

**Defaults are unchanged.** fp32 commits every step, which is what the resident service and every
batch mode run unless asked otherwise.

    HALO_GDN_DEFER=1                    commit every step (default)
    HALO_GDN_DEFER=0                    the deferred kernels, committing every step (control)
    HALO_GDN_DEFER=2..8                 commit once per that many steps
    --gdn-defer 0,1,4                   case axis in both measurement tools

The depth a process runs at is the requested depth in a packed coordinate and 1 otherwise; both
tools record the requested and the effective value, and a logits dump is named for the effective
one so two arms cannot share a file.

**A route that cannot rebuild a deferred state commits it first.** The persistent kernel's
whole-pass route, a sliced recurrent layer and the sliced state route all read a state region
directly, and a served workload reaches them constantly — a narrow pass (under 32 rows) drops out of
the wide route, which is how both batch drivers prefill a stream before stepping it. Each of those
three entry points now settles the slots it is about to read, with a rebuild-and-store kernel that is
idempotent: at zero pending it re-stores what it decoded, and re-rounding an already rounded value
in a packed coordinate returns it. Which slots can hold pending updates is host state, because the
host is what has to decide; `reset_seq` empties a slot's list and `copy_slot_state` moves it.

That makes a mixed workload correct and not free. A process that alternates a wide deferred step
with a narrow pass on the same slot pays a full state round trip at every transition, which is worse
than not deferring at all — the engine's own decode is one such workload, since it runs the
persistent kernel, and it is why the depth is opt-in rather than a default the server inherits. The
shape this pays off on is the batched one, where every recurrent layer of every step is resident.

**Switching depth invalidates live sequences**, for the same reason switching format does: the
pending count is interpreted against the depth. The setter synchronises and the caller resets or
re-prefills, which both tools do per case.

## The next questions

- **An append step rebuilds a state it never stores** - answered, and the answer is no.
  [`gdn-append-gram.md`](gdn-append-gram.md) built the contraction, measured it at 1.14 -> 0.69 ms
  per pending term and 0.34 ms off the base append body, and rejected it: that is 0.9% of a depth-4
  cycle for a reassociation that reassociates at every depth including zero. The premise that the
  per-term cost is what stops the depth from paying is also wrong - the commit step has to
  materialise the state to store it, so its rank-`P` update grows linearly in `C` while the write it
  amortises shrinks as `1/C`, and they cross at four to six terms however cheaply an append runs.
  What did come out of it is exact and shipped: the rebuild's delta column is staged in LDS with the
  keys, bit-identically, which is worth 2.0% of the commit step and a null on an append step.
- **The keys are fp32 and need not be.** They are L2-normalised and enter a rank-1 update; half
  precision halves the dominant half of the rebuild's traffic. It is an approximation of a different
  kind from the state's own coordinate — the key that rebuilds a term would no longer be the key
  that made it — so it needs the step panel above, which now exists.
- **The depth axis belongs in the horizon panel.** [That panel](gdn-state-horizon.md) closed the
  int8 question over 256 roundings per layer while this was in flight, and it is the instrument that
  can also say whether a deferred commit's reassociation drifts: the arms are already paired per
  (stream, step) and bucketed by how many roundings have happened. `--gdn-defer` walks it today.
  The 24-step panel above found no drift and no cost, on a quarter of the walk.
- **Deeper than 8 needs a different pending layout.** At depth 8 the rebuild reads seven terms on
  the step before the commit. Staging the whole pending list once per head instead of once per block
  would cut its key traffic fourfold, and a block that owned a whole head (`SPLIT` 1) would remove
  the redundancy entirely, at a work decomposition the rest of the kernel was tuned against.

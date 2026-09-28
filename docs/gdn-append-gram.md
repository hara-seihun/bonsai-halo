# The pending terms of a deferred step are not what bounds its depth

[Deferring the state commit](gdn-state-defer.md) closed by naming the next lever: an append step
rebuilds a state it never stores, it needs that state for only two dot products, and both distribute
over the pending rank-1 terms, so a term could cost two fused multiply-adds per state row instead of
a hundred and twenty-eight. That was worth measuring and it is measured here.

**It works, it is consistently faster, and it is not worth a numerical-map change.** Contracting the
pending terms cuts a term from 1.14 ms to 0.69 ms per 32-stream step and the append body by another
0.34 ms, which is 0.8% of a depth-4 cycle. It also does not move the deferral depth, because the
step that bounds the depth is the *commit* step, which has to materialise the state to store it and
therefore pays the full rank-P update whichever way the append steps are written.

What did land from this work is smaller and exact: the rebuild's delta column is staged in LDS with
the keys instead of being read from the state region inside the term loop, which is **bit-identical**
and worth **2.0% of the commit step**.

## Occupancy is not an axis on this kernel, and that is worth knowing first

[`tools/gdn_occ_probe`](../tools/gdn_occ_probe.hip) is [`rows_occ_probe`](rows-grid-occupancy.md)
pointed at the recurrent state. It compiles in eight seconds, needs no model and no measurement lock,
and asks the runtime what it will co-schedule:

| instantiation | VGPR | spill | LDS/block | blocks/WGP | waves/SIMD32 |
|---|---:|---:|---:|---:|---:|
| `resident_state<4, LOOP, commit>` | 60 | 0 | 0 | 8 | 16 |
| `resident_state<4, HOIST, commit>` | 60 | 0 | 0 | 8 | 16 |
| `resident_state<4, LOOP, defer>` | 61 | 0 | 4132 | 8 | 16 |
| `resident_state<4, HOIST, defer>` | 59 | 0 | 4132 | 8 | 16 |
| `resident_state<8, HOIST, defer>` | 46 | 0 | 4132 | 8 | 16 |
| `resident_state<16, HOIST, defer>` | 41 | 0 | 4132 | 8 | 16 |
| `gdn_defer_flush_k<4>` | 38 | 0 | 4132 | 8 | 16 |

Eight blocks per workgroup processor is this device's ceiling for 256-thread blocks. Every
instantiation reaches it with room to spare, so **no register change to this kernel can buy a wave**,
and the first thing this claim proposed — an append-only instantiation that stops being charged for
the commit path's token loop and store quantiser — cannot pay and was not built.

This is the opposite of `k_forward_rows`, where 140 registers against 135 is four blocks against
five. The difference is workgroup size: that kernel runs 1024-thread blocks, so its register file
divides into few blocks and every granule matters. Ask the probe which axis a kernel is on before
spending a turn on registers.

## The identity

For a head with committed base `B`, pending terms `(g_i, a_i, k_i)` and weights
`w_i = prod_{t>i} g_t`, the effective state before a token is
`M = w_base B + sum_i w_i a_i (x) k_i`. An append step consumes `M` only through

    part_j = g <M_j, k>
    d_j    = (v_j - part_j) beta
    o_j    = <g M_j + d_j k, q>

and both contractions distribute:

    <M_j, k> = w_base <B_j, k> + sum_i w_i a_i[j] (k_i . k)
    <M_j, q> = w_base <B_j, q> + sum_i w_i a_i[j] (k_i . q)
    o_j      = g <M_j, q> + d_j (k . q)

`k_i . k`, `k_i . q` and `k . q` belong to the head, not to the row, so they are computed once per
block in the staging barrier. The row body also loses the materialised update itself: sixteen vector
operations per state row become eight, because nothing downstream of an append step ever reads the
state it would have built.

[`tools/gdn_gram_check.py`](../tools/gdn_gram_check.py) mirrors both paths in double precision from
the source expressions, with the region offsets the device uses, and agrees to 9e-16 over 200 random
cases at every pending depth, including the staged delta index at `SPLIT` 4, 8 and 16. **The negative
below is about cost, not correctness.**

It is a reassociation at every depth *including zero*: the deployed body rounds `g * m[s]` per element
before the fused multiply-add and `g <M_j, k>` does not. So the deferred arm's residual hash cannot
be its acceptance instrument, and it would have needed the horizon quality panel to ship.

## Measured

All panels: mode 19, 32 streams of one token, 64-token prefix, int16 state, depth 4, arms interleaved
in one process under `tools/run-batch-compare --pin-clock --profile-tool`. Raw in
[`batch-comparison/gdn-append-gram/`](../../../data/bonsai2/batch-comparison/gdn-append-gram/).

### Staging the delta column is exact and a null on an append step

`gdn_defer_rebuild` read a term's delta column as `A[i*SS + part*RPU + lane]` *inside* a
runtime-trip-count loop and then did four `v_readlane` on the result, so every pending term put a
memory round trip on the loop's critical path with nothing to cover it. Staged in LDS with the keys,
behind the barrier that function already needs, the term loop reads a wave-uniform LDS address.

| arm | pending 0 | pending 2 |
|---|---:|---:|
| read from the region (what shipped) | 69.375, 69.347 | 71.748, 71.578 |
| staged in LDS | 69.469, 69.785 | 71.386, 72.245 |

Residual FNV-64 `14246133829617533007` at zero pending and `7937632179187010320` at two, **equal in
both arms**: the change is exact on device, not just in argument. It is also a null here, inside a
noise floor the panel itself shows — two cells that are equal by construction read 69.375 and 69.469.

### On the commit step the same change is worth 2.0%

| round | read from the region | staged in LDS |
|---:|---:|---:|
| 0 | 85.464 | **84.346** |
| 1 | 85.835 | **84.421** |
| 2 | 86.827 | **84.736** |
| 3 | 87.132 | **85.273** |
| median | 86.33 | **84.58** |

Staged wins every round. The asymmetry is the interesting part: the same removed loads are free on an
append step and worth 1.75 ms on the step that also writes 2.42 GB of state back. An append step is
not saturating the memory return path, so its extra requests cost nothing; a commit step is, and they
compete with the store. **This is what ships.**

### Contracting the terms

| arm | pending 0 | pending 2 | per pending term |
|---|---:|---:|---:|
| rebuild the state | 70.219, 70.006 | 72.120, 72.664 | 1.14 ms |
| contract the terms | 70.000, 69.547 | 71.265, 71.022 | **0.69 ms** |

Medians 70.113 / 72.392 against 69.774 / 71.144. The Gram arm is faster in all four cells across both
rounds: 0.34 ms on the base append body and a 40% cut of the per-term cost.

### The loop nesting is most of the arithmetic

The first implementation put the term loop *inside* the row loop, which is where the algebra puts it
and where it re-reads the term's weight and its two head-uniform scalars once per state row and pays
the loop branch `R` times. It measured **1.344 and 1.807 ms per term against the staged rebuild's
0.363 and 0.620** in the same process — four times worse than the thing it replaces, on correct
arithmetic. Hoisting the term loop over the rows, with `dk[R]` and `dq[R]` carried in registers, is
what produces the 0.69 ms above. If a change turns a per-element cost into a per-row one, check which
loop ends up outside before believing the census.

## Why none of this moves the deferral depth

A cycle at depth `C` is `C-1` append steps and one commit. Only the commit writes, and only the
commit has to materialise the state, so it pays the full rank-`P` update whatever the append steps do.
From the measured cells — append base 69.774 ms, per-term 0.69 ms contracted and 1.14 ms materialised,
commit at three pending terms 84.58 ms, hence a write worth 11.4 ms:

| depth | rebuild, staged | contracted, staged |
|---:|---:|---:|
| 2 | 76.37 | 76.04 |
| 4 | **74.67** | 73.99 |
| 5 | 74.67 | **73.79** |
| 6 | 74.86 | 73.77 |
| 8 | 75.60 | 74.01 |

The materialising column reproduces the published result that depth 4 is the best depth and 8 is
worse than 4. The contracted column flattens the curve and moves its minimum to five or six, and the
minimum is 0.2 ms lower than depth 4's. **The per-term cost of an append step was never what stopped
the depth from growing.** What stops it is that the commit step's rank-`P` update grows linearly in
`C` while the write it amortises shrinks as `1/C`, and those cross at four to six terms however
cheaply the append steps run.

Taken together the whole contracted arm is worth 74.67 to 73.99 ms, **0.9% of a 32-stream generation
step**, for a reassociation that would need its own horizon panel and its own opt-in switch forever
after. That is not a trade worth making, so the kernel, its switch and its `--gdn-gram` axis are not
in the tree. The patch is
[`gram-arm.patch`](../../../data/bonsai2/batch-comparison/gdn-append-gram/gram-arm.patch) beside the
raw samples, and the identity it rests on is executable in `tools/gdn_gram_check.py`.

## Installed acceptance

Canonical `80b547f`, `bonsai-halo` `f5f0015e92eb7e2e3092`, `tools/batch_profile`
`93d699821ac4dd503181aa57`, under `tools/run-batch-compare --pin-clock`, two rounds:

| step | ms |
|---|---:|
| append, pending 0 | 66.975, 67.766 |
| commit, pending 3 | 81.615, 81.962 |

Residual FNV-64 `14246133829617533007` at zero pending and `7119616587439629066` on the commit step
- the values the published deferred arm produces - on the installed build, which is the whole
acceptance for an exact change. Raw in `installed-acceptance.json`; the same pair on the development
build before the rebase is `rebased-commit.json`.

These absolute numbers sit below every panel above because the recurrence's
[row batching](gdn-token-reduce.md) landed on master in between. Compare only within a panel.

## What a pending term actually costs, which nobody has explained

A term is `R` rows times one broadcast read, one multiply and four fused multiply-adds, plus four
shared key reads: 28 issue slots per wave at `SPLIT` 4. Across 48 layers, 32 sequences, 48 heads and
4 parts that is 2.36 M waves, and at 80 SIMD32 and 2.428 GHz it prices at **0.34 ms**. It measures
1.14. The contracted form is 19 slots and prices at 0.10 ms; it measures 0.69.

Both are about 3.4 times their census, and the constant overhead is roughly 38 slots per term per
wave in either arm. It is not the global load — removing that is the staging arm above, and it is a
null on an append step. The remaining suspects are the loop's own branch and address arithmetic at a
trip count the compiler cannot see, and the `s_delay_alu` chain on its LDS reads. A term loop unrolled
against `GDN_DEFER_MAX` with a predicated tail would settle it, and at 0.69 ms per term the whole
prize is about 0.4% of a step.

## Reproduce

```sh
make tools/gdn_occ_probe && tools/gdn_occ_probe          # no GPU lock, no model
tools/gdn_gram_check.py                                   # the identity, no GPU at all

tools/run-batch-compare --pin-clock --profile-tool \
  --out .../gdn-append-gram/commit-staged.json \
  --modes 19 --decode-streams 32 --decode-prompt 64 \
  --gdn-state 2 --gdn-defer 4 --decode-prime 3 --rounds 4 --traces 0
```

`--traces` was hardcoded to `{false, true}` in `batch_profile`'s decode loop and is now honoured, so
a decode panel costs half what every previous one did and a case list is the size it says it is.

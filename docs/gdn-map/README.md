# The GDN map over a block of tokens

`gdn_token` runs one token through one head's 128x128 state. This directory
derives the same map over a whole pass of `m` tokens as block algebra, checks
every form against the source recurrence in float64, and counts the work each
form needs against the resident serial path rather than against a straw
baseline that reloads the state per token.

Two things to carry into the rest of it. The algebra is exact and checked. The
costs are counted work at declared peak rates, and the one measured kernel time
here sits 4.0x above its own budget, so no count in this directory predicts a
runtime or a speedup.

What the counts do say: the cross-lane reductions the block form was supposed to
remove belong to the state's lane layout, not to the recurrence, so a layout
change and a scalar decay identity remove more counted work than the block form
does. In the proposed dense-base representation, decode can defer the base
write while still reading that base for each token.

## What the kernel computes

From `gdn_load` and `gdn_token` in `kernels/halo_rows.hip`, with `M[j][i]`
holding value row `j` and key column `i`, `SS = 128`, and `ph_gdn_pre`
L2-normalising q and k per 128-channel head:

    g_t     = exp(ssm_a[h] * softplus(alpha_t + ssm_dt[h]))      scalar per token and head
    beta_t  = sigmoid(beta_raw_t)                                in (0, 1)
    S_t     = g_t S_{t-1} + delta_t k_t^T,   delta_t = beta_t (v_t - g_t S_{t-1} k_t)
    o_t     = S_t q_t / sqrt(128)                                POST-update state

Two source facts the algebra leans on. `ssm_a` is negative at every recurrent
layer and head of the deployed model (`results/gate-stats.txt`: min -139.4, max
-0.0038), so `g_t` is in (0, 1] and every decay ratio `gamma_t / gamma_s` for
`s <= t` is at most 1. The `g` values printed beside it are gates for a
stand-in `alpha`, including the bias-only `alpha = 0` case; the real per-token
`alpha` is a projection of the hidden state, so those are ranges the gate can
take, not measured token gates.

And `ph_gdn_pre` divides by `max(sqrt(sum of squares), NORM_EPS)`, so over the
reals `||k_t|| <= 1`, equal to 1 whenever the pre-norm magnitude clears 1e-6.
With `beta` in (0, 1) the correction `I - beta_t k_t k_t^T` then has singular
values in (0, 1]. That bounds the per-step operator; it does not by itself bound
the condition number of the triangular systems below, which is not established
here. `results/equivalence.txt` includes non-normalised k, where the explicitly
inverted form does lose everything at m = 32 while the substitution forms hold.

## The block form

Write `gamma_t = prod_{s<=t} g_s`, stack `K, Q` (m x 128, rows `k_t, q_t`),
`V` (m x 128), and let `S_0` be the incoming state. The per-token scalar decay
factors out of the rank-one corrections entirely, so the chunk map is the pure
DeltaNet map conjugated by `diag(gamma)`:

    Ntil[t,s] = (gamma_t/gamma_s) (k_s . k_t)     s <  t     strictly lower
    Ctil[t,s] = (gamma_t/gamma_s) (q_t . k_s)     s <= t     lower, inclusive
    (I + diag(beta) Ntil) D = diag(beta) (V - diag(gamma) K S_0^T)
    O   = (diag(gamma) Q S_0^T + Ctil D) / sqrt(128)
    S_m = gamma_m S_0 + sum_t (gamma_m/gamma_t) delta_t k_t^T

`D` is the m x 128 matrix of the same `delta_t` the kernel computes; the unit
lower triangular system is one forward substitution. No intermediate 128x128
state is formed: `S_0` is read, `S_m` is written, and everything between is
m x 128 or m x m.

The decay ratios have to be accumulated as segment products, or as differences
of log-domain sums, never as `gamma_t / gamma_s` from two cumulative products:
for a fast head both underflow and the quotient is 0/0. `decay_ratios` in
`gdn_block.py` builds them with no division at all, and sub-chunking bounds each
product to c terms. Underflow of a segment product to zero is the contribution
vanishing, which is the right answer.

Solving the system instead of substituting gives the fully materialised form,
with `T = (I + diag(beta) Ntil)^-1 diag(beta)` lower triangular:

    A    = Ctil T                                   causal m x m token mixing
    Qeff = diag(gamma) Q - A diag(gamma) K          modified queries
    O    = (Qeff S_0^T + A V) / sqrt(128)
    Psi  = T^T diag(gamma_m/gamma)
    S_m  = S_0 (gamma_m I - K^T diag(gamma) Psi K) + V^T Psi K

The homogeneous part is the WY representation of the product of corrections,
`prod_t (I - beta_t k_t k_t^T) = I - K^T T_0^T K` with `T_0` the undecayed `T`,
scaled by `gamma_m`. `gdn_block.py` checks that identity directly against the
128x128 matrix product, and checks that the map splits linearly into its
`S_0`-driven and `V`-driven parts.

Sub-chunking by `c` applies the same formulas to `c` tokens at a time against
the running state, reducing the m x m objects to c x c. Register residency
still needs an implementation. The base-state contractions for different tokens
can run independently; the triangular solve, output mixing and final update
retain their data dependencies. The base contractions become GEMMs of shape
(c x 128) x (128 x 128), rather than individual matrix-vector products.

### Two identities that need no block machinery

Both fall out of `S_t = g_t S_{t-1} + delta_t k_t^T` and apply to the current
kernel layout unchanged.

**A. The output does not need the updated state.**
`o_t = g_t (S_{t-1} q_t) + (k_t . q_t) delta_t`. The k-side and q-side
contractions then read the same registers and are independent, instead of the
q-side waiting for the rank-one update. `k_t . q_t` is one 128-dot per token per
key head, cheap enough to compute in the conv phase alongside the gates.

**B. The decay does not need to touch the state.**
Carry `S_hat = S_t / gamma_t` and the scalar `gamma_t`; the per-element multiply
disappears, `delta_t / gamma_t` goes into the state, and outputs scale by
`gamma_t`. This removes one of the four passes over the state. It needs a
rescale before `1/gamma_t` overflows, and the rescale frequency is data
dependent: at the bias-only stand-in gate in `results/gate-stats.txt` the median
head sits at `g = 0.973`, which would not rescale inside 128 tokens, while the
fastest sits at `1e-4`, which would rescale every few. Those are gates at a
stand-in `alpha`, so they bracket the behaviour rather than measure it; the real
frequency is whatever the token projections produce. The test is a scalar
comparison uniform across the head and the rescale is one multiply per element,
so the worst case degrades to the current cost rather than breaking. The block
form has no such hazard: it only forms segment products `gamma_t/gamma_s <= 1`,
where a fast head underflows to zero, which is the contribution vanishing.

## What is checked

`gdn_block.py` compares every form against a transcription of `gdn_token`, in
float64, over slow and fast decay, normalised and raw k/q, m in {8, 32, 128},
reporting max absolute difference relative to the largest reference entry. Full
output in `results/equivalence.txt`; unit k/q, slow decay, m = 128, with the
fast-decay rows alongside it in the file:

| form | outputs O | final state |
|---|---:|---:|
| identity A, pre-update output | 4.2e-16 | 0 |
| identity B, deferred decay | 3.9e-16 | 4.4e-16 |
| block chunk, WY/triangular | 7.8e-16 | 3.3e-16 |
| block sub-chunked, c = 8 | 7.8e-16 | 3.3e-16 |
| attention form (Qeff, A, Psi) | 6.0e-16 | 3.8e-16 |
| factored state, materialise every 8 | 4.8e-16 | 3.3e-16 |

The fast-decay head at m = 128 reaches `gamma_m = 2.6e-133` and holds the same
1e-16 agreement, with identity B taking 4 rescales to get there.

With raw, non-normalised k the explicitly inverted attention form reaches 7.6e-1
relative error at m = 32 while the substitution forms stay at 5e-15. That is an
observation at `||k|| >> 1`, where `I - beta k k^T` is expanding; it does not
establish the conditioning of the same inverse at `||k|| <= 1`.

## What the forms cost in counted work

`gdn_cost.py` counts VALU instruction slots per token per head, with the rules
stated in its header, and the lane rate is the same 7.424e12 that
`docs/throughput-budgets.md` uses to price vector work. Full output in
`results/cost-model.txt`. Everything it reports in microseconds is counted work
divided by a declared peak rate. It is a budget, and a ratio of two budgets is
not a predicted speedup.

The measured anchor says how far that is from time. One native 128-row A4 pass,
empty prefix, no warmup, rocprof kernel durations
(`tools/batch-compare/results/resident-kernel-times.json`): wall 232.172 ms,
`resident_state<4>` 18.294 ms summed over 48 layers, `resident_conv` 1.826 ms,
`resident_output` 0.938 ms. The state kernel is 7.88% of the pass and 143
us/token, against a 35.8 us/token work budget and a smaller traffic budget:
4.0x above both. Neither counted resource explains the measured time at these
rates, and what does is not established here.

The first thing the count settles is that cross-lane reduction cost is a
property of the state's lane layout, not of the algebra. Both forms contract the
state over the key index twice per row per token: the serial form for `delta`
and the output, the block form for `K_c S^T` and `Q_c S^T`. Writing L for the
lanes a state row is spread over and R for the rows a lane holds, a reduction
costs the `warp_sum` DPP stages that span L, and a lane pays it R times because
its rows sit in different registers. The deployed SPLIT = 4 layout has L = 32,
R = 4: six instructions per stage, four times, twice per token.

| form | L, R | state VGPRs | waves/head | lane-ops/token/head | vs deployed | budget us/token |
|---|---:|---:|---:|---:|---:|---:|
| serial, as in `gdn_token` | 32, 4 | 16 | 32 | 115,200 | 1.00x | 35.8 |
| serial + identities A, B | 32, 4 | 16 | 32 | 100,096 | 1.15x | 31.1 |
| block WY, c = 8 | 32, 4 | 16 | 32 | 102,536 | 1.12x | 31.8 |
| serial, as written | 4, 1 | 32 | 16 | 68,096 | 1.69x | 21.1 |
| serial + identities A, B | 4, 1 | 32 | 16 | 52,992 | 2.17x | 16.4 |
| block WY, c = 8 | 4, 1 | 32 | 16 | 55,432 | 2.08x | 17.2 |
| serial + identities A, B | 1, 1 | 128 | 4 | 50,944 | 2.26x | 15.8 |
| three state contractions, floor | | | | 49,152 | 2.34x | 15.3 |

Which comparator matters. Against the unmodified deployed form the block form
counts *lower*, 102,536 against 115,200, because it drops the decay pass. The
only comparator that supports a negative is serial with identities A and B at
the same layout: 55,432 against 52,992 at L = 4, where the block form pays 2,184
lane-ops per token of Gram, solve and intra-chunk output on top of the same
three contractions. That is more counted arithmetic in these formulas. It is not
a runtime claim: block scheduling, the shorter dependence between independent
sub-chunk contractions, and a matrix unit are all outside this model and any of
them can reverse it.

The floor is structural for these formulas: the delta needs `g S_{t-1} k_t`, the
output needs a q-side contraction, and the final state needs a rank-m update.
Deferring the rank-m update across passes does not remove it, it only adds
correction terms, so three passes over the state per token stands.

What the deployed form actually spends above that floor is one redundant pass
(the decay multiply, which identity B folds into a scalar) and 49,152 lane-ops
of DPP reduction, 43% of its instruction stream. Splitting a row over 4 lanes
instead of 32, with one row per lane, cuts the reduction to 2,048 and doubles
state registers to 32 per lane; a row entirely inside one lane removes it and
costs 128. The narrow layout still leaves 16 waves per head, 768 per layer for
one sequence.

It does change the state load pattern: consecutive lanes no longer read
consecutive columns, so each lane fetches a 128-byte run and the wave's request
is strided. Every byte is still used and the load happens once per pass rather
than once per token, but the cost of the strided pattern is not counted or
measured here, and at m = 1 the load is the phase. Any trial belongs in the
resident prompt kernel, with the deployed layout left in `ph_gdn`, which is
already a separate kernel with a separate grid.

The fully materialised intermediate count is 12.8 kB per head at c = 8 and
327.7 kB at c = 128. The latter exceeds one workgroup's LDS budget. Tiling,
register allocation, recomputation and global-memory storage have not been
implemented or priced; this count alone does not prove spilling.

Where the traffic budget does land near a measurement, it is in the regimes that
are traffic-heavy rather than work-heavy. Decode, m = 1: 1,180 us/token of state
traffic budget against the 1,270 us `PLAN.md` reports for the GDN recurrence
phase. Prefill in eight-row slices: 160 us/token against the 184 us/token
implied by the 23.5 ms phase stamp in `docs/wide-sequence.md`. At m = 128 the
traffic budget falls to 21.6 us/token while the measured kernel is 143, so the
agreement in those two regimes does not carry over, and none of the three is a
like-for-like isolation.

The counts say nothing about the measured SPLIT ordering. SPLIT changes R but
leaves two reductions per row per token, so the instruction count is invariant
to it; an invariant count cannot predict that 4 beats 8 and 16. Whatever orders
them is outside this model. The parameter that does move counted work is L, the
lanes a row is spread over.

## A third of the gap was the gate, not the recurrence

September 21. The anchor below is `resident_state` at 18.294 ms and 4.0x its own
work budget, and this directory says nothing here locates that gap. An ISA census
of the kernel located a third of it outside every formula counted here.

The token loop spent 83 of its 545 issue slots, plus roughly as many again in
dependent-issue delay, on `softplus(alpha_t + ssm_dt[h])`. `log1pf` inlines as a
compensated double-double reciprocal, it is wave-uniform so the compiler
scalarised it, and every one of the 1536 waves that share a layer evaluated it
once per token: 32 copies of a per-(row, head) scalar. Hoisting it into the
convolution kernel took `gdn-resident-core` from 24.205 to 16.302 ms at 128 rows,
bit-identical. [`docs/resident-gdn.md`](../resident-gdn.md) owns the measurement.

What that says about the counts in this directory: `gdn_cost.py` prices the state
algebra in lane-ops and the gate is a scalar the formulas treat as given, so a
term worth a third of the measured time was outside the model entirely. The
remaining measured-against-budget gap is now about 2.7x rather than 4.0x, and the
lane-layout and identity numbers below are unchanged, since none of them touched
the gate either. Before trusting a counted floor again, census the kernel: the
expensive instruction may not be in the algebra you are counting.

## Verdict for the resident prefill path

The declared peak-rate budget for the state phase is about 2% of a 128-row pass.
Its measured share is 7.88%, 143 us/token against a 35.8 us/token work budget and
a 21.6 us/token traffic budget. Measured time was four times the declared work
budget when this was written; a third of that has since been removed by hoisting
the gate, which no count here priced. The counts do not explain the rest, and the
budget is not an Amdahl ceiling on possible improvement.

That changes which question is open. The declared work budget falls by
19.4 us/token between the deployed form and the cheapest counted one, 1.1% of
the pass if it converted one for one. This is not an upper bound, and it need not
convert at all. The 4.0x gap is the larger object and nothing in this directory
locates it. Things a measurement could separate: the dependence between a token's
reduction, its delta and the next token's contraction; DPP latency; occupancy
across 192 workgroups; the state load and the operand stream. The block form's
per-sub-chunk independence attacks the first of those, which is why its higher
counted arithmetic does not settle it.

What this directory supports is narrower than a recommendation. The identities
and the layout are exact and cheap to try. The block form is exact and would
need a scheduling argument rather than an arithmetic one. The counted floor for
the family is 15.3 us/token, which is a statement about these formulas at a
declared rate and not about achievable time.

Scope: fp32 state, gfx1151 vector lanes, m <= 128, this decomposition of the
engine, counted work plus one measured kernel total. If the state moves to bf16,
the three contractions become 16x16x16 WMMA GEMMs at the 4x rate
`docs/throughput-budgets.md` declares, which a per-token matrix-vector shape
cannot use at all; that is the one realization where the chunk algebra is the
precondition rather than an alternative. It is a changed numerical map, not a
reordering, and would need the same acceptance evidence as any other.

## The write side of a decode token

For a dense state array and an unconstrained `q_t`, `o_t = S_t q_t` touches
every element, so within that representation the read cannot move: 64 kB per
head per token. That is a property of the representation, not of the map. The
write can move, because nothing outside the head reads the state between
tokens. Keep it factored, `S_t = gamma S_base + sum_i c_i u_i k_i^T` with all
coefficients at most 1, append `(delta_t, k_t)` per token, and absorb the
pending pairs with one rank-r update every r tokens. This is the block map
applied across passes instead of within one.

The pending list holds 0 to r-1 pairs over the cycle, so a token reads (r-1)/2
of them on average. The materialisation lands on the token that fills the list,
assuming the implementation retains the base and pending pairs from that
token's contractions. Under that residency assumption, materialisation adds a
base write and no second read. A separate materialisation kernel would need
another base read, which this table does not charge.

| r | bytes/token/head | vs read+write | MB/token/seq | traffic budget us/token/seq |
|---:|---:|---:|---:|---:|
| current | 131,072 | 1.00x | 302 | 1,180 |
| 4 | 84,480 | 1.55x | 195 | 760 |
| 8 | 78,336 | 1.67x | 180 | 705 |
| 16 | 78,336 | 1.67x | 180 | 705 |
| 8, bf16 pairs | 76,032 | 1.72x | 175 | 684 |

Counted work goes down rather than up: 32,768 for the two base contractions,
16,384 for the amortised rank-r update, 1,792 for the pending corrections and
644 of scalars, 51,588 against the current 65,536 lane-ops per token per head.

Verified exactly as `factored_state` in the equivalence table above. The last
column is a declared-rate traffic budget for one phase, not a tokens/s
prediction, and the decode phase's own time is not isolated anywhere in this
directory. For scale, at 32 streams the state array moves 9.7 GB per step
against the 126.5 ms that the measured 252.9 tok/s A4 aggregate implies; what
fraction of that step the phase actually occupies is unmeasured here. It
composes with bf16 state, which halves the base read independently.

The cost is the durable state format, which is the engine's most carefully
specified contract (`tools/direct-commit/README.md`,
`tools/sequence_state_check.cpp`). Two things make it less bad than it sounds.
The k side of each pending pair is already cached for replay in `blk_cache`.
And rollback becomes truncation of the pending list rather than replay of the
accepted prefix, which is the machinery `ph_gdn` currently exists to run.
Per-sequence storage is r x 256 floats per head per layer, 18.9 MB at r = 8.

## Floating point

Every form here is an exact identity over the reals and none of them is
bit-identical to the kernel. The reorderings are specific:

* The serial dot sums four per-lane products in lane order, then a five-stage
  DPP butterfly. A block contraction sums 128 terms in whatever order the
  matmul schedule picks.
* In the block form a token's output arrives as two summands, one through the
  state and one through the c x c Gram and the triangular solve, where the
  serial form had a single running sum.
* Identity A replaces "update the state, then contract with q" by "contract with
  q, then add `(k.q) delta`".
* Identity B rounds `delta_t / gamma_t` into the state and rescales on a
  schedule that depends on the head's gates.
* The factored state rounds the base once per r tokens instead of every token.

Float32 against float64, m = 32, the same operands for both: kernel summation
order 1.2e-7 on outputs, block c = 8 1.9e-7, and 1.7e-7 between the two forms.
For `||k|| <= 1` and beta in (0, 1) the per-token operator has norm at most 1,
so an error injected at one token is not amplified by later ones; errors are
still injected at every token and accumulate, and these figures are observations
on random operands rather than a bound. numpy has no FMA, so this measures
reordering sensitivity and not the kernel's bits; a real implementation would
need the trajectory and logit comparison `docs/wide-sequence.md` applies to the
scaled-FP16 operand experiment.

## Files

    gdn_block.py    every form, checked against the source recurrence in float64
    gdn_cost.py     counted work and traffic per form, per m, per r, plus the
                    measured resident_state anchor they are compared against
    gate_stats.py   ssm_a and dt.bias from the GGUF, with gate ranges for
                    stand-in alpha values
    results/        captured output of all three

```sh
python3 docs/gdn-map/gdn_block.py --m 128 --f32-m 32
python3 docs/gdn-map/gdn_cost.py --m 128
python3 docs/gdn-map/gate_stats.py --model ../../data/bonsai2/PTQ1_0.gguf
```

Numpy only, no GPU, about two seconds for all three.

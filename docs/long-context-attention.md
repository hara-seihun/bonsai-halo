# Prompt length is an axis, and attention is the only phase on it

> **Followed up.** [Folding a KV group's query heads into one score unit](attn-reuse-fold.md)
> prices the reuse rectangle this document's `TT=16` row measures one corner of: in the leaf
> coordinate a head costs one register per row instead of 64, and the fold is -4.06% of the score
> kernel at position 3072, a null at 1024, and -3.8% over a 4096-token document. The same panel
> says this phase is not request-bound below about a thousand positions, which is the part worth
> carrying.

> **Carried to 8192 and to the other shape.** [The engine at the context it
> advertises](long-context-serve.md) takes this curve out to 8192 on the served 256-row width, adds
> the generation step this document never measured (**2.204 ms per 1000 tokens of context, 29.7
> GB/s**, and six times the requested K and V costs 0.8%), and closes the 16-row group question in
> the leaf coordinate as well as this one.

Every prefill number this repository publishes is a 384-token document. The engine advertises
32768. This is the first measurement of what happens between those two numbers, the instrument that
takes it, one exact change that came out of it, and one candidate the measurement kills.

## The curve

`tools/batch_profile --prefill-scan N` ingests an `N`-token document in `--rows`-row passes with the
tracer on and emits one sample per pass, tagged with the position its first row sat at. The engine
context follows the scan instead of the 512 every other shape in that tool uses; a KV slot is 90112
bytes per token across the 22 cache slots, so a scan to 8192 costs 738 MB on one sequence.

Attention cost is set by position, not by token value — nothing in the pass branches on a value, and
the score loop runs its `count` keys whatever they contain — so a scan longer than the document
tiles it. The shape stays exactly reproducible instead of depending on which file is longest in the
tree this hour.

Mode 20, 128-row passes, 3072 tokens, one process
([`long-context/scan-tt-3072.json`](../../../data/bonsai2/batch-comparison/long-context/scan-tt-3072.json)):

| position | attention | everything else | attention share |
|---:|---:|---:|---:|
| 0 | 2.85 ms | 330.6 ms | 0.9% |
| 512 | 11.3 | 320.2 | 3.4% |
| 1024 | 23.4 | 338.3 | 6.5% |
| 2048 | 35.7 | 278.4 | 11.4% |
| 2944 | 51.1 | 298.7 | 14.6% |

**0.0166 ms per token of position per 128-row pass.** Every other phase is flat to 1% across the
walk; `ffn`, the two sequence projections and `gdn-resident-core` are priced per row and do not care
where the row is. That makes them a free in-panel control for anything measured this way, which is
how the two results below were read.

Integrated over a whole prompt of `N` tokens in 128-row passes, the phase costs `2.75 ms` per pass
plus `0.0166 ms` per token of position, which is about `6.5e-5 x N^2` ms once the quadratic term
takes over. Against the rest of a pass — 290 ms at mode 20, about 180 ms at mode 19:

| prompt | attention | share of a mode 20 prefill | share of a mode 19 prefill |
|---:|---:|---:|---:|
| 384 | 0.015 s | 1.7% | 2.7% |
| 2048 | 0.30 s | 6.1% | 9.4% |
| 8192 | 4.5 s | 19% | 28% |
| 16384 | 17.6 s | 32% | 43% |

The 8192 and 16384 rows extrapolate a fit measured to 2944; 384 and 2048 are measured. Two crossings
fall out of it. Within a single pass, attention passes the whole A4 FFN at position **4500** — the
FFN is 77 ms at 128 rows and does not move. Over a whole prompt, attention passes the FFN's total at
about **9000 tokens**. **A prefill optimisation fitted at 384 tokens is fitted where this phase does
not exist.**

## What the phase is made of

`HALO_ATTN_TRACE_SPLIT=1` traces an attention layer's three launches as separate phases. It is off
by default so every panel this lane has already taken keeps reading the same way.

At 2944 tokens, 128 rows: `sequence-core-score` 42.376 ms, `sequence-core-combine` 7.994,
`sequence-core-pre` 0.706.

So the chunk-partial buffer is the *smaller* half of this phase. That is worth saying plainly
because it is counter-intuitive: at position 2944 a 128-row pass writes 128 x 24 x 24 partials of
1040 bytes per layer, 77 MB, to produce 3.1 MB of attention output — twenty-five times write
amplification, growing linearly with context — and reading all of it back is 16% of the phase. The
score kernel, which reads the keys and values, is 84%.

> **Followed up, and the 16% was a residency number.** [The chunk-partial
> window](attn-partial-window.md) shrank the working set between the write and the read from 79.9 MB
> to 16 MB, and the same bytes through the same combine expression went from 154 GB/s to 365:
> **-57.6% of the combine**, -5.07% of attention normalised, and 879.4 -> 886.0 tok/s on a
> 3072-token document, bit-identical. Two thirds of that kernel was waiting for DRAM to hand back
> something the device had produced itself a few hundred microseconds earlier.

Per score unit — one row group, one query head, one 128-key chunk — the ISA census of
`k_attn_wide<8>`'s key loop (`tools/isa_loop_count.py`, no GPU):

| per key, all 8 rows | slots |
|---|---:|
| `v_fmac_f32`, the dot products | 64 |
| `v_add_f32` DPP + `permlanex16`, the cross-lane reduction | 47 |
| `s_delay_alu` | 63 |
| `ds_write`, `v_mov`, address and loop | 41 |

**64 slots of arithmetic need 88 slots of work and 63 of scheduling to deliver.** The reduction is
a five-deep dependent DPP chain per row, and with eight rows in flight the compiler still could not
hide it: `s_delay_alu` is 29% of the block. The value loop is in better shape than it looks —
the compiler already folds the FP16 conversion into `v_fma_mix_f32` and batches the score reads as
`ds_load_2addr_stride64_b32`, four instructions for eight values — and its remaining waste is two
slots of 64-bit pointer increment per key, the same addressing form
[`docs/ffn-operand-address.md`](ffn-operand-address.md) removed from `k_proj_opt`.

## A 256-row pass had no wide attention at all

[The wide attention](wide-attention.md) gave every attention layer one launch per pass;
[the pass width](pass-width.md) made 256 rows the default prompt pass. They landed in the same
minute and did not meet. `create_attn_batch` refused any capacity above 128, `forward_batch` has no
error path for a missing module, and a 256-row pass fell back to the per-slice attention that the
wide module exists to replace — 512 cooperative launches per pass instead of 16, on the engine
default and on every `--rows 256` panel taken since.

`attn_batch_set_enabled` makes the route a case axis, so both arms share a process, a clock and a
warm device with the module's buffers already allocated. `HALO_WIDE_ATTN=0` refuses the module
outright and allocates nothing, which answers a memory question and is the wrong switch for a
timing pair.

Mode 20, 256 rows, 2048 tokens, one process
([`long-context/scan-wide256.json`](../../../data/bonsai2/batch-comparison/long-context/scan-wide256.json)):

| position | per-slice | wide | untouched-phase ratio |
|---:|---:|---:|---:|
| 0 | 26.611 ms | **6.577** | 0.983 |
| 512 | 40.616 | **25.003** | 0.992 |
| 1024 | 57.565 | **41.433** | 1.005 |
| 1792 | 84.726 | **65.058** | 0.998 |

Whole document, 256-row passes: 405.1 to **421.3 tok/s, +4.0%**.

**It is a constant, not a ratio, and that is the useful part.** At 256 rows both routes make the
same eight-row groups and read the same keys; the wide launch removes 496 cooperative launches and
nothing else. The difference is 20.0 ms at position 0 and 19.7 ms at position 1792. So a slice of
this phase costs about 40 us that is not its keys — four times the 9 us
[the wide prep](wide-prep.md) measured for a cooperative launch, because a slice carries the
persistent kernel's grid, its own rope and cache write, and its own combine rather than a plain
dispatch.

Bit-identical, against a hash already in the tree: 95,354,880 full-vocabulary logits over every
token of the document, FNV `17618767336181543518` at 128 rows and at 256, which is exactly what
[the pass-width acceptance](pass-width.md) published for the *sliced* route at 256. The wide
attention at the new width reproduces the per-slice attention it replaces.

## Sixteen-row groups lose, and the raw wall time says the opposite

`HALO_ATTN_TT=16` merges neighbouring eight-row groups so a key chunk is read once for sixteen rows
instead of twice for eight. [The wide attention document](wide-attention.md) left it "built and
untested ... one panel settles it", with a correct argument that it is bit-identical: a row that
shares a chunk with a later row only gains masked entries, which enter `l` as exact zeros through a
fixed 256-slot block sum and the value sum as `fma(0, v, acc)`.

It is 18% slower on the score units.

Mode 20, 128 rows, 3072 tokens, both widths in one process through `--attn-tt 8,16`:

| position | score, TT=8 | score, TT=16 | untouched ratio | normalised |
|---:|---:|---:|---:|---:|
| 2048 | 28.665 ms | 34.407 | 1.011 | **+18.7%** |
| 2944 | 42.376 | 49.303 | 0.983 | **+18.3%** |

`k_attn_wide` goes 92 VGPR and 16 waves per SIMD32 to 180 and 8. Halving the key traffic is worth
less than the wave slots it costs, which says the score units are **not** key-bandwidth-bound: at
position 2944 they request 18.9 GB per pass and, with the six query heads of a KV group adjacent in
the dispatch, move about 3.1 GB of it — 74 GB/s against a 242 GB/s roof. They are latency- and
issue-bound, and the medicine for that is wave slots, not fewer bytes.

**The raw wall time favours TT=16 by 8.5% and every bit of that is the clock ramp.** The TT=8 arm
ran first and its untouched phases fell 330 to 286 ms across its own walk while the TT=16 arm ran
flat at 282-285. A panel without an in-process control would have published a win here.
[`docs/wide-attention.md`](wide-attention.md) made exactly this point about this exact kernel two
hours earlier; this is the second time the same instrument caught the same trap, so treat the
untouched-phase ratio as mandatory rather than careful.

## What this rules out, and what it leaves

**Ruled out.** Sixteen-row groups, at both widths measured, for the reason above. The score units
are not at a bandwidth roof, so nothing whose whole mechanism is fewer key bytes will pay: that
includes wider groups, more query heads per unit, and any KV cache layout change argued from traffic
alone.

**Not available exactly.** The obvious structural fix — one workgroup walks all chunks for a
(group, head) and never writes chunk partials, which is what flash attention does — cannot reproduce
these bits. The combine takes a single global maximum over chunks and then a flat weighted sum,
`v = sum_cc exp(m_cc - M) . acc_cc`. Folding chunks inside the unit needs either a running maximum,
which inserts rescalings the flat sum does not have, or ranged partials, where
`exp(M_g - M) . exp(m_cc - M_g)` is not `exp(m_cc - M)` in floating point. The same argument rules
out a larger `ACHUNK`. Any of these is a legitimate alternative numerical map — it is a
reassociation of a softmax reduction, not a precision reduction — but it needs its own quality panel
and stays an explicit alternative. The domain of that negative is exact FP32 reproduction of the
deployed two-kernel split; it says nothing about whether the alternative is good.

**The largest remaining target, priced.** 64 slots of arithmetic per key against 47 of cross-lane
reduction and 63 of scheduling. A shape where each lane owns a *key* and sums all 256 dimensions in
its own register needs no cross-lane reduction at all, and the query values are wave-uniform in that
layout so they can sit in scalar registers and feed `v_fmac_f32` directly: about 97% of slots become
arithmetic against 53% today, roughly 1.8x on a phase that is 84% of attention.

**That target is exact, and this document said otherwise.** It was filed here as "not
bit-identical — a 256-element dot product summed in one register is a different tree from eight
per-lane terms and a five-level butterfly", and that is wrong: the butterfly *is* a tree, and one
lane can hold it. `sc[]` only ever takes lane 0, and after the two `quad_perm` steps lane *i* holds
its quad's sum `Q(i>>2)`, so the last three steps deliver a balanced binary tree over the 32
per-lane granule sums. A lane that owns a key reproduces the score bit for bit, and the quality
panel is not on this change's critical path. (The observation that only lane 0 survives, and that
the butterfly is therefore a tree a single lane can hold, is from the September 21
`attn-score-coordinate` claim.)

**The leaves of that tree are not in index order, and three engineers got it wrong the same way.**
`row_ror:N` moves data toward *higher* lanes — destination lane *n* reads source lane `n - N`, the
same direction that makes `row_shr:1` the inclusive-scan idiom. So `row_ror 4` gives lane 0 `Q0+Q3`
and lane 8 `Q2+Q1`, `row_ror 8` gives lane 0 `(Q0+Q3)+(Q2+Q1)`, and `permlanex16` adds lane 16's
`(Q4+Q7)+(Q6+Q5)`. The tree is balanced and its leaves are contiguous, but the four quads of a
16-lane row pair as 0-3 and 2-1. An in-lane reproduction in naive index order differs in the last
bits of the score, which then goes through `__expf`, and reads as a reassociation nobody intended.
The relabeling is compile-time and is its own inverse, so one kernel can carry both orders:
`leaf_of_pos(p) = (p & ~15) | ((((4 - ((p>>2) & 3)) & 3) << 2) | (p & 3))`, and
`kernels/attn_tree_check.hip` settles it against the real `warp_sum` on random data in a fraction of
a second with no model and no lock. (From the September 21 `attn-score-lane` claim, which checked
the rotation direction instead of the mnemonic.)

**Four engineers derived the index-ordered tree from the same mnemonic inside one hour, and three of
them confirmed each other.** So: an exactness argument about cross-lane hardware is not finished
when the algebra closes, only when the device agrees. A DPP control name gives the shape of a
permutation and not its direction, and a summation tree is bit-identical only up to that direction.
This is [the addressing rule](activation-scale-axis.md) in a second place — a change is done when
the emitted behaviour changes, not when the source says it should.

So the flash-shaped chunk folding above still needs its quality panel and the lane-per-key score
loop does not. They are two projects, not one.

**The operand cursor does not reach this kernel, and it cost two attempts to find out.** Both key
and value addresses are `global_load` on a 64-bit VGPR pair advanced by a
`v_add_co_u32`/`v_add_co_ci_u32_e64` couple per key, which is exactly the form
[`docs/ffn-operand-address.md`](ffn-operand-address.md) removed from `k_proj_opt` for -12.9%. It
does not come off the same way here.

- Reading the finished pointer's two halves through `readfirstlane` and rebuilding it **loses the
  address space**: every load in the block came back as `flat_load_b128` and `flat_load_d16_b16`.
  [`docs/wide-head.md`](wide-head.md) hit the same wall from the other side and its rule is the
  one to keep — an offset that leaves the scalar base costs you the `global_` form.
- Naming the uniformity at its source instead — `slot` comes from `RB[S.row0].seq`, one row every
  lane of the workgroup reads — and indexing both cursors with 32-bit element offsets compiles to
  **byte-identical address forms**: 17 `global_load_b128` and 1 `global_load_d16_b16` on 64-bit
  VGPR pairs, before and after, with the block three work slots *larger*.

So the cursor here is **untested, not refuted**, and it is the third instance of the rule
[`docs/activation-scale-axis.md`](activation-scale-axis.md) published: an addressing change is done
when the emitted form changes, not when the source index does. Neither attempt is in the tree.
Whatever still forces the wide address in this kernel has not been found, and the census costs
seconds, so the next attempt should start from the address-form table rather than from the source.

## Installed and accepted

Canonical build `7f0e0bd6300a4373e3bcd884bcf6c9ee24d3b336a43cd667812050975073e6b0` at `c48a369`,
which carries this change under a peer's column-group work, reproduced 95,354,880 full-vocabulary
logits at **both** 128 and 256 rows with FNV `17618767336181543518` and zero nonfinite — the hash
[the pass-width acceptance](pass-width.md) published before a 256-row pass could reach this module.
The resident server came back on it and answers on 8471.

Raw panels: [`long-context/`](../../../data/bonsai2/batch-comparison/long-context/),
[`longctx-installed/run.json`](../../../data/bonsai2/batch-comparison/longctx-installed/run.json).

**The decode shape is now measured, and it is a different machine.**
[`docs/attn-row-groups.md`](attn-row-groups.md) walks a generation step's position axis: attention
goes 4.4 to 48.4 ms between a 64-token and a 960-token prefix at 32 streams while five other phases
stay flat to 0.4%, so it becomes the largest phase in the step at about a thousand tokens of
context. The two documents do not agree about what the score units are short of, and both are
right: they are latency- and issue-bound here, where eight rows amortise a key chunk, and at their
**byte** roof in a generation step, where every row is a different sequence and only one row reads
each chunk. **A slot census does not price the decode shape**, which is the domain this document's
negatives and targets belong to.

The decode shape also has a known geometry defect neither measurement covered: a 32-stream step has
thirty-two one-row groups and the chunk-partial budget reserves eight rows for each, so
`attn_batch_groups_per_launch` splits a layer into more launches than it needs to.

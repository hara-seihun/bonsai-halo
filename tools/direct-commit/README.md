# Direct-commit GDN prefill

Prompt ingestion accepts every token it processes. The deployed recurrent path does not: each pass
leaves the committed GDN state and conv ring where its predecessor's *accepted* prefix left them,
caches its own rows, and lets the next pass replay them. That contract exists for speculative
decoding, where rows can be rejected. During prefill it costs a second pass over every token's
state update, measured at [about 60 ms of the 413 ms A4 device span for 128
rows](../../docs/prefill-gap.md#the-recurrent-engine-pays-for-speculative-rollback-during-prefill).

`FwdParams::commit_state` removes that second pass without changing a single arithmetic operation.

## What it does

With `commit_state = 0`, nothing changes anywhere.

With `commit_state` set, for every sequence in the pass:

- `ph_gdn_pre` stores the conv ring after its last row instead of after the replayed prefix.
- `ph_gdn` stores the GDN state after its last token instead of after the replayed prefix.

That is the whole change. The token loop, the operand loads, the cached rows in `blk_cache`, the
per-row outputs and the order of every floating-point operation are what they were. Only the point
at which state becomes durable moves, from the start of the next pass to the end of this one.

Incoming `SeqCtl::n_replay` is still honoured, so a committing pass can absorb an uncommitted
predecessor and its own rows in one go. That is the transition from the deployed path into this
one, and it needs no separate flush.

## Using it

```c++
p.commit_state = 1;          // applies to every sequence in the pass
launch_forward_rows(p, stream);
// next pass for these sequences: SeqCtl::n_replay = 0  (Engine::Seq::keep = 0)
```

The obligations on the caller:

- **The next pass must arrive with `n_replay = 0`.** Committing a pass and then replaying it
  applies its tokens twice. The kernel cannot detect this; the state is simply wrong afterwards.
- **A committed pass must not be rolled back.** Its rows are already in the state, and
  `Engine::rollback` only rewinds `len`, `keep` and `last_n`. Restrict rollback to uncommitted
  passes, or restore a snapshot.
- Parity still alternates every pass, and rows are still written into `blk_cache`, so an
  uncommitted pass that follows a committed one is replayed in the usual way. Only the committed
  pass itself is off limits.
- A launch commits all of its sequences or none. A pass that mixes prefill with speculative
  decoding has to be split into two launches. Making this per sequence is a bitmask over
  `SeqCtl` indices and three lines in `ph_gdn_pre`/`ph_gdn`; it is deliberately not built.
- `single_map` is ignored while `commit_state` is set: commit selects the deployed arithmetic map.

A committing pass keeps the deployed occupancy grid unless its own kernel allows less, so a
commit-versus-replay comparison has one variable in it.

## Why the result is identical, not merely close

Consider a sequence whose tokens are cut into passes. Take pass A, followed by pass B.

Under replay, A computes its rows from the fresh conv outputs, writes `blk_cache[parity]`, and
leaves the state alone. B loads the committed state, reloads A's conv outputs and gates from
`blk_cache`, runs A's tokens through `gdn_token`, writes the state back, then runs its own rows.

Under commit, A does the same arithmetic on the same operands, in the same order, and writes the
state it already holds in registers. The values B would have reconstructed are the values A stored.
Both schedules apply one sequence of operations to one set of inputs; nothing is reassociated,
re-rounded or reordered. The difference is which pass performs the store.

The conv ring is a pure shift register over pre-activations, so the same argument holds there
without any arithmetic at all.

## The probe

`commit_equiv.cpp` is a host mirror of `ph_gdn_pre` and `ph_gdn` with the structure that decides
results: the ring recurrence, the per-head L2 normalisation with its four-wave reduction, the gate
expressions, the delta rule with its 32-lane column split and `warp_sum`'s DPP association, the
`blk_cache` parity slots, and the shared row buffers that several sequences index through
`SeqCtl::row0`. It runs four value heads of 128 state rows instead of forty-eight, which exercises
the same code in a third of a second.

It runs each schedule against the pure replay schedule and compares row outputs, GDN state and conv
ring bitwise. A committed pass is compared against the reference state one pass later, which is the
point where the reference has replayed exactly those tokens.

```
make -C tools/direct-commit check
```

```
Direct-commit GDN equivalence, host mirror of ph_gdn_pre and ph_gdn
seed 20260921, 4 heads of 128 state rows, 1024 conv channels, comparison is bitwise

scenario                                     rows    state   ring     verdict
first pass commits from empty state          equal   equal   equal    matches the replay path
eight-row prefill, four passes               equal   equal   equal    matches the replay path
ragged pass sizes with a short tail          equal   equal   equal    matches the replay path
one-row passes, shorter than the ring        equal   equal   equal    matches the replay path
uncommitted pass, then a committing pass     equal   equal   equal    matches the replay path
old, commit, old, old                        equal   equal   equal    matches the replay path
commit, old, commit, commit                  equal   equal   equal    matches the replay path
one-row commit between eight-row passes      equal   equal   equal    matches the replay path
violation: a committed pass replayed again   differ  differ  equal    diverges, as the contract says it must  [69505 floats, max |d| 0.000582]
five passes, two sequences, shared buffers   equal   equal   equal    sequences do not interact
```

An argument to the binary changes the input seed.

Note the violation row: the conv ring is *equal* there. Shifting the same eight pre-activations
into a three-deep ring twice leaves the same three values, so a duplicated replay is invisible in
the ring and shows up only in the state and the outputs. Do not use ring agreement as evidence that
a sequence's replay bookkeeping is right.

The probe says nothing about the GPU, the decomposition of units across workgroups, the input prep
and projections that feed these phases, or attention. Those belong to the full-model checks.

## The acceptance tool on the real model

`tools/sequence_state_check.cpp` asks the same question of the loaded 64-layer model, where the
host probe cannot go: does mixing commit into a sequence's history change that sequence?

It runs two copies of one token stream in one engine, in separate slots. The reference copy runs
every pass in a replaying mode; the mixed copy runs some passes in that mode's committing twin.
After every pass it compares full-vocabulary logits, per-row argmax, the entire GDN state of each
slot and the entire conv ring, counting differing floats, the largest absolute and ULP gap, and
non-finite values on both sides. Nothing is asserted to be equal in advance.

### The two copies are not durable in the same place at the same time

A durable image covers tokens `[0, len - keep)`. A replaying pass leaves its own rows in
`blk_cache` for its successor to commit; a committing pass has already stored them and sets `keep`
to 0. So immediately after a committing pass the mixed copy is durable at a later prefix than the
reference, by exactly that pass's rows, and raw `gdn_state` and `conv_ring` **differ for a correct
implementation**. Their logical states agree; their durable images have not met yet.

The tool follows that:

- logits and argmax are required equal after every pass, being a function of the logical state,
  which never lags;
- raw state and ring are compared and reported after every pass for every slot, including one that
  sat the pass out, but equality is required only where both copies of that slot are durable at the
  same prefix;
- each scenario ends with one identical committing pass on both copies, which flushes both deferred
  prefixes to `len`, and there the raw state and ring must match.

The synchronising pass is appended after the trajectory, never inserted into it, so the history
under test keeps its original commit points. A difference at an unaligned prefix is reported with
both prefixes and the gap between them, under `state_asserted: false`, and is never relabelled as a
pass; the run-level `deferred_state_differences_reported` counts them. The inverse case is reported
too: a slot that is unaligned yet identical raises `unaligned_but_identical`, which is what a
`commit_state` that stored nothing would look like at that point.

Under the default plan the state assertion still bites in the middle of the trajectory, not only at
the end: after a pass the mixed copy runs uncommitted, both copies are durable at `len - keep`
again, so the two passes that follow committing ones are full bitwise state comparisons, including
the one after the 40-row wide pass.

```sh
make -C tools/direct-commit sequence_state_check
tools/run-batch-compare --state-check --model ../../data/bonsai2/PTQ1_0.gguf \
  --out ../../data/bonsai2/sequence-state-check/run.json
```

It pairs 16 against 10 and 18 against 17, which differ only in `commit_state`. Mode 19 is wide,
committing and A4 at once and has no uncommitted wide A4 twin, so it is not a state comparison and
is not run; pairing it with 11 would measure the wide projection map instead, which is
`tools/batch_compare`'s job.

The default pass plan is two sequences with ragged row counts, 1, 3, 8, 17, 40 and back down, so
one sequence crosses every dispatch boundary in `Engine::forward_batch`: the deployed whole-pass
kernel at four rows and under, the sliced schedule from five to thirty-one, and the batched FFN and
wide sequence projections at thirty-two and above. Sequence rows share and straddle the eight-row
slices. The commit pattern visits every transition: uncommitted first, into commit, commit to
commit, back out, and in again across the wide pass. `--shapes` and `--commit-pattern` replace both.

Three further checks cover the rollback boundary:

- a committed pass refuses to have tokens rejected, and the `Seq` is unchanged after the refusal,
  field by field;
- accepting a committed pass whole stays legal and leaves nothing to replay;
- an uncommitted pass that follows a committed one still rolls back exactly, both when part of it
  is rejected and when all of it is. Rejecting all of it leaves parity one flip ahead of the
  reference, which must not matter because nothing replays from the other half of `blk_cache`; the
  tool records that the parities really did differ, so the case cannot pass vacuously.

The rejected tokens are checked to differ from the accepted ones before the comparison runs, for
the same reason. The pass that gets rolled back carries one sequence alone: when every token of it
is rejected the reference has no pass to match it with, and letting the second sequence ride along
would leave it one token further on in one copy than the other. The rollback checks compare state
under the same prefix rule and end with the same synchronising pass.

The JSON result carries every count, both sides' `Seq` bookkeeping per pass, and which schedule each
pass actually took. `keep` and `rollbackable` legitimately differ between the two copies after a
committing pass; `len`, `last_n` and `parity` must agree, and the tool checks that separately.
Exit status is non-zero when anything differs.

## Evidence from the device

The native preflight on the full 64-layer model (main's run, same checkout) produced bit-identical
finite full-vocabulary logits between mode 16 (commit) and mode 10, and between modes 18 and 17, at
32, 40 and 128 rows after a 64-token context.

## Compiled resources

`hipcc --offload-arch=gfx1151 -O3 -Rpass-analysis=kernel-resource-usage`, against the same file at
`19c9f83`:

| kernel | VGPRs | private B/lane | SGPR spills | waves/SIMD |
|---|---:|---:|---:|---:|
| `k_forward_rows<8>` deployed | 217 | 20 | 89 | 6 |
| `k_forward_rows<8>` commit | 217 | 20 | 89 | 6 |
| `k_forward_rows<1>` deployed | 94 | 20 | 138 | 16 |
| `k_forward_rows<1>` commit | 95 | 20 | 134 | 16 |
| part 1, any row count | 34 | 0 | 0 | 16 |
| part 2, 1/2/4 rows | 62 | 0 | 0 | 16 |
| part 2, 8 rows | 103 | 0 | 0 | 12 |

Every pre-existing instantiation, including the single-token maps and every `k_ffn_slice`, is
unchanged in all of these fields. No new spills and no new scratch.

## The split-layer launcher

`launch_sequence_part(p, layer, part, stream)` exists for the same reason and lives in the same
files, so it is documented here rather than twice.

It runs one layer in two halves so a wider projection can own the layer's matvecs, without
reimplementing the prep or the state phases:

- `SEQ_PART_PREP` runs the layer's input prep and returns after its barrier. It fills
  `P.xq`/`P.xs`/`P.xsum`, and on recurrent layers `P.ab` (stride `2 * HV` per row) and `P.ninv`.
  It zeroes none of the projection outputs.
- `SEQ_PART_STATE` expects the projection outputs restored into the deployed buffers, along with
  `P.ab` and `P.ninv`: recurrent layers read `P.big` (stride `QKV_OUT`) and `P.zbuf` (stride
  `VDIM`); attention layers read `P.qfull` (stride `Q_OUT`, q and gate interleaved per head),
  `P.kbuf` and `P.vbuf` (stride `KV_OUT`). It runs the conv ring, the state and the output prep, or
  the rope/KV write, attention and the combine, and returns with `P.xq`/`P.xs`/`P.xsum` ready for
  the `ssm_out`/`o` projection. It never touches the residual `P.x`; the projection that follows
  must accumulate into it.

Both parts are `PART` template instantiations of `k_forward_rows`, so they execute the same phase
code as the whole-layer kernel, and each gets its own occupancy grid. `layer` must be in
`[0, NLAYER)`, rows in `[1, RMAX]`, sequences in `[1, MAXSEQ]`; anything else aborts with the
offending values instead of launching. Each part is an independent cooperative launch, so `P.bar`
and `P.work` must be zeroed before each one; part 1 uses one phase counter and part 2 three.
`P.commit_state` applies to part 2, which is where the state lives, and is ignored by part 1.

With `P.prof` set, part 1 writes `prof[0..1]` (one interval) and part 2 `prof[0..3]` (three
intervals: conv ring, state, output prep; or rope/KV, attention, combine).

# Resident GDN state

The committed wide sequence route now loads each sequence's recurrent state once per layer, processes all its tokens in registers, and writes the state once. The previous route did that once per eight-row slice. This changes storage and scheduling, not the recurrence.

[The state's storage precision](gdn-state-pack.md) and [its memory coordinate](gdn-state-coord.md)
are separate levers on this path, both selected by `HALO_GDN_STATE`, and neither changes the fp32
default this document measures.

## Full-model result

Three rounds, 64 layers, the same executable on both sides. Each configuration ran separately with original mode 0 interleaved. Prefill processes 256 tokens and computes logits on every row of the final pass. Generation excludes prompt setup and reports aggregate output TPS across 32 streams.

| workload | eight-row state | resident state | gain |
|---|---:|---:|---:|
| A8 prefill, 128 rows/pass | 453.2 | 477.9 | 5.5% |
| A4 prefill, 128 rows/pass | 541.4 | 589.7 | 8.9% |
| A8 prefill, 32 rows/pass | 326.9 | 345.2 | 5.6% |
| A4 prefill, 32 rows/pass | 411.4 | 440.6 | 7.1% |
| A8 generation, 32 streams | 205.3 | 210.1 | 2.3% |
| A4 generation, 32 streams | 245.8 | 252.9 | 2.9% |

The A8 128-row control includes a slow 331.8 TPS sample, retained in its 29.4% spread. A separate five-round, one-slot repeat gave medians 451.4 versus 480.2 TPS, with spreads 3.3% and 3.0%. Eight-stream generation does not select this path and stayed about 147 TPS.

Reports: [A8 control](../tools/batch-compare/results/resident-tps-r0-m18.md), [A8 resident](../tools/batch-compare/results/resident-tps-r1-m18.md), [A4 control](../tools/batch-compare/results/resident-tps-r0-m19.md), [A4 resident](../tools/batch-compare/results/resident-tps-selected-m19.md), [A8 repeat control](../tools/batch-compare/results/resident-a8-repeat-r0.md), [A8 repeat resident](../tools/batch-compare/results/resident-a8-repeat-r1.md).

Against original mode 0 in the selected A4 panel, prefill is 3.81x faster and 32-stream generation is 1.90x faster. A4 remains the existing lower-precision FFN map. This change introduces no additional logit difference.

## Implementation

`launch_gdn_resident` in `kernels/halo_rows.hip` runs three kernels:

1. Compute convolution, SiLU, normalized q/k and gates across all tokens. Convolution has only three preceding raw inputs, so tokens can run independently. Reads before the current pass come from the incoming replay prefix or the durable convolution ring.
2. Load each head's state into registers, run the existing `gdn_load` and `gdn_token` helpers over the incoming replay and every current token, then store once. This kernel also commits the convolution ring after all convolution readers have finished.
3. Run the existing output normalization, gating, Hadamard and quantization helper directly into the next projection's operand layout.

The state loop remains token-serial. No triangular solve or changed state algebra is selected. `ResidentSeq` carries full-sequence row ranges, slots and entry replay/parity separately from the eight-row control descriptors. Current tokens need no replay-cache copy because this path always commits them. Subsequent uncommitted passes still write their own replay cache normally.

At capacity 128, token and output workspaces add 8,437,760 bytes. They are shared across layers and allocated during preparation. Normal execution allocates nothing. Attention layers, uncommitted passes and batches below 32 total rows retain the existing routes.

The first native comparison found a rounding difference in convolution. The deployed compiler output rounds `w.y*r1`, then fuses `w.x*r0` into it; the new loop initially rounded the other product. The resident kernel explicitly uses the deployed multiply/FMA ordering. The first discrepancy appeared at token 3, when all three history values became nonzero. Matching that order removed the discrepancy in processed tokens, recurrent outputs, state and logits.

## Acceptance

[Full-vocabulary comparison](../tools/batch-compare/results/resident-selected-identity.json): 143,032,320 finite logits, modes 18/19 at 32, 40, 88 and 128 rows, zero differing bits against the preceding selected implementation.

Source-matched timing panels also compare [A8](../tools/batch-compare/results/resident-m18-exact.json) and [A4](../tools/batch-compare/results/resident-m19-exact.json) prefill/decode logits at 32 rows and every recorded continuation token. Both pass. These comparisons use the same binary with resident state enabled or disabled.

[Mixed-state acceptance](../tools/direct-commit/resident-result.json) passes multi-sequence, ragged-slice, commit/replay transitions and rollback checks. Outputs agree along the trajectory. Durable state and ring agree at matching prefix positions, including the final common commit. It also exercises entering a resident pass with pending replay.

## Where the gain comes from

A separate native phase workload uses a 128-token prefix followed by one 128-row pass, with logits on all rows. A4 medians:

| schedule | 32-row pass ms | 128-row pass ms |
|---|---:|---:|
| eight-row state slices | 84.001 | 251.755 |
| resident, split 4, serial convolution | 78.798 | 235.638 |
| resident, split 8, serial convolution | 79.954 | 242.814 |
| resident, split 16, serial convolution | 83.617 | 253.369 |
| resident, split 4, parallel convolution | 78.168 | 232.506 |

Every residual hash agrees at each width. Raw files are `resident-phase-s{0,4,8,16}.json` and `resident-phase-parallel.json` in `../../data/bonsai2/batch-comparison/`. Each has three native and three instrumented samples per width. Split 4 is selected. More workgroups did not pay for the duplicated operand work and changed scheduling in these variants; this experiment does not isolate which cost dominates.

A [kernel trace](../tools/batch-compare/results/resident-kernel-times.json) isolates the three kernels on one 128-row pass with an empty prefix:

| kernel, summed over 48 layers | ms |
|---|---:|
| parallel convolution | 1.826 |
| resident state recurrence | 18.294 |
| output preparation | 0.938 |

Wall time was 232.172 ms. These are one traced sample, not medians from the prefixed workload. All three kernels report zero scratch. The recurrence is still 7.9% of this pass, well above a peak-throughput arithmetic budget. The [block-map derivation and cost experiments](gdn-map/README.md) examine that gap. They do not establish a runtime lower bound.

A third of that gap has since been located and removed, and it was not in the recurrence at all: see
[where the decay gate is evaluated](#where-the-decay-gate-is-evaluated) below. The cost model in
`gdn-map/` counts lane-ops of the state algebra and never counted the gate, which is why it could
not see it.

Another sixth of it was the token loop's *schedule* rather than its work, and that is also something
a lane-op count cannot see: the `lane == 0` store at the end of each state row is a branch, so the
compiler emitted the row loop as separate basic blocks and covered each row's cross-lane butterfly
with nothing while the other rows' independent chains sat where the scheduler could not reach them.
[Batching the row reductions](gdn-token-reduce.md) is 220 to 166 issue slots and takes this phase to
**14.697 ms**, bit-identical, with a measured null on the batched decode shape.

A third thing a lane-op count cannot see is how many rows one wave should own.
[The unit width](gdn-unit-width.md) walks that axis to both ends: at one sequence a wave of eight
rows beats today's four by 6.4% of the phase because the per-token fixed cost is paid half as often,
sixteen rows loses 5% because 48 blocks cannot fill the machine, and everything with two or more
sequences wants the width it already has. The split is chosen by `nseq` now; `SPLIT` 8 and 16 remain
selectable and remain losses.

A fourth is which *columns* a lane owns rather than which rows.
[The lane layout](gdn-lane-layout.md) turns the ownership 90 degrees - one lane, 32 columns, one row
- which removes the cross-lane reduction from the per-row cost entirely and is worth another 19.7% of
the phase on top of the width above, bit-identical, at prompt shapes. It loses 10.6% at one token per
sequence, so `launch_gdn_resident` picks it by rows per sequence.

## Where the decay gate is evaluated

The state kernel used to derive its own gates. Every token loop iteration computed

    g_t = exp(ssm_a[h] * softplus(alpha_t + ssm_dt[h]))

and `softplus` goes through the full-precision `log1pf`, whose compensated arithmetic compiles to
83 scalar-float instructions. The value is uniform over a (row, head), so all 1536 waves that share
a layer derived the same number, once per token each: 32 copies of it per token, 48 layers deep.

An ISA census of the committed path, one token iteration of `resident_state<4>` on gfx1151, in
issue slots:

| | slots | VALU | wait/delay | SALU | SALU-float | VMEM | kernel bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| gate in the token loop | 545 | 204 | 135 | 100 | 83 | 23 | 8364 |
| **gate hoisted** | **355** | 169 | 83 | 81 | **0** | 22 | **6640** |
| hoisted, replay prefix split off | 325 | 162 | 80 | 53 | 0 | 30 | 7736 |

Count the loop per basic block, not whole. The static loop is 719 slots, but 174 of them are the
replay branch a committed pass never takes, so the whole-loop figure overstates it by a third.

`gdn_decay()` is now the single expression both producers call. `resident_conv` already ran once per
row and already wrote the scaled alpha and beta into the token record; it now writes the evaluated
gates into the same two slots instead, and `gdn_load` reads them. The deployed block cache keeps raw
projections, so the replay prefix and the sliced `ph_gdn` path are unchanged. 56 VGPRs and occupancy
16 in every arm, no scratch.

### Measured

One build, four rounds, mode 19 at 128 rows with logits off, all three arms interleaved in one
process with the case order reshuffled every round. Medians of bracketed launch time:

| phase | gate in loop | hoisted | hoisted + replay split |
|---|---:|---:|---:|
| `ffn` | 88.855 | 88.509 | 89.979 |
| `sequence-input-projection` | 42.235 | 42.175 | 42.649 |
| **`gdn-resident-core`** | **24.205** | **16.302** | **17.802** |
| `sequence-output-projection` | 17.722 | 17.668 | 17.818 |
| `sequence-core` | 16.170 | 16.068 | 16.404 |
| `sequence-input-prep` | 3.896 | 3.896 | 3.979 |
| device span | 195.062 | **186.647** | 190.525 |

The state phase falls 32.65% and nothing else moves by more than 0.7%, which is this panel's own
resolution. The pass is 4.31% shorter. Every arm hashes the same residual, FNV-1a
7446865760224376151 over all 128 x 5120 FP32 outputs.

Full model, `--modes 19 --only prefill --prefill-rows 128 --gdn-pregate 0,1`, a 384-token document
in three 128-row passes with the head on the last pass, four rounds with the arms interleaved:

| arm | rounds | median |
|---|---|---:|
| gate in the token loop | 683.6, 662.9, 691.1, 699.0 | 687.4 tok/s |
| **gate hoisted** | 715.2, 707.4, 733.0, 718.1 | **716.7 tok/s** |

**+4.26% full-model prompt throughput.** These absolute numbers sit below the 702-720 tok/s the
[FFN schedule](ffn-schedule.md) recorded for the same workload because several engineers were
compiling and measuring on the host throughout; that is exactly why both arms share one process.

### In a generation step the hoist loses, so the shape picks

A state unit loads its head's 64 kB of state once and then walks that sequence's tokens. What the
hoist is worth therefore depends on **rows per sequence**, not rows per pass. At 128 rows in one
sequence it removes 190 issue slots from each of 128 tokens. At four rows in each of 32 sequences it
removes them from four, against a state round trip that dominates - and it measures *worse*:

| rows per sequence | gate in the token loop | gate hoisted | by the rule |
|---|---:|---:|---:|
| 128 rows, 1 sequence | 25.132 ms | **17.305** | 17.176 |
| 4 rows, 32 sequences | **52.777** | 54.653 | 52.738 |
| 1 row, 32 sequences | 48.748, 49.890 | 50.365 | |

The four-row row is three processes per arm with the case order reversed between them, so it is not
an ordering artefact: the control repeats within 0.3% across processes and the hoist is 3.55% above
it every time. The rule cell lands on the control it is supposed to choose, within 0.1%.

What that costs at the step level is small - device span 217.170 against 219.892, and the phase is a
quarter of the step - but it is consistently the wrong sign, and it is the shape
[the decode map](decode-map.md) identifies as the large remaining lever, so it is not a good place to
carry a regression.

The mechanism is latency cover. A short token loop next to a 64 kB state round trip is waiting on
memory, and the softplus was independent work the wave could issue while its state loads were in
flight. Removing it shortens the instruction stream without shortening the wait. That shows up in
the issue model below: at four rows per sequence the hoisted arm sits almost exactly at
memory-plus-issue serialised while the control sits well under it. It is not occupancy -
`resident_conv` is 13 VGPRs without the gate and 15 with it, occupancy 16 either way.

So `launch_gdn_resident` picks by shape: hoist when a sequence contributes at least
`GATE_HOIST_MIN_ROWS_PER_SEQ` = 8 rows, otherwise leave the gate in the loop. Measured at 1, 4 and
128 rows per sequence; 5 to 127 is interpolation, both arms are correct everywhere, and
`--gdn-pregate 0` or `1` pins either one. Every arm produces the same bits, so the rule is a
scheduling choice and no caller can observe which side it took.

### The replay split lost, and why that is worth knowing

The hot loop still branches per token on `t < S.n_replay`, so the raw-gate transform is compiled
into the body of the loop that runs once per row. Walking the replay prefix in its own loop removes
it: 355 issue slots to 325, order-preserving, same tokens and same operands.

It measured 1.5 ms slower, 17.802 against 16.302. What it saves in the loop body it pays for in
footprint: the unrolled token body is then emitted twice and the kernel goes 6640 to 7736 bytes,
against 8364 before any of this. **Fewer static issue slots in a hot loop is not less work when the
reduction is bought by duplicating the body.** It is not in the runtime; this section is its record.

### The phase is an instruction-issue machine, and the count says so

One wave issues at most one instruction per cycle. A 128-row pass runs 1 x 48 x 4 workgroups of
eight waves for 48 layers, 128 tokens each, so the whole phase is
`1536 x 128 x slots x 48 / 80 SIMD32` cycles. At the 2.9 GHz this device boosts to:

| arm | slots/token | predicted | measured | issue explains |
|---|---:|---:|---:|---:|
| gate in the token loop | 545 | 22.17 ms | 24.205 ms | 91.6% |
| **gate hoisted** | 355 | 14.44 ms | 16.302 ms | 88.6% |
| hoisted, replay split | 325 | 13.22 ms | 17.802 ms | **74.3%** |

Two things fall out. The gap this phase was known for - 4.0x a counted work budget - is essentially
all instruction issue, in both of the arms that fit in the instruction cache, so further time comes
out of it only by removing instructions, not by rescheduling them. And the rejected arm is the one
row that does not fit the model: it issues the fewest instructions of the three and is the furthest
from its own issue time, which is what a fetch-bound loop looks like and is independent evidence for
why it lost.

What the 355 are, per token, with the four state rows a wave owns unrolled: 50 slots of cross-lane
reduction (eight `warp_sum` butterflies, `v_add_f32_dpp` x32 + `v_permlanex16_b32` x8 + the final
adds), 50 fused multiply-adds, 16 multiplies that apply the decay to the resident state, 18 loads,
30 slots of 64-bit token-pointer arithmetic, and 70 `s_delay_alu`. The reduction and its dependent
delay are now the largest item, which is what `gdn-map/`'s L = 4 lane layout attacks and what its
identity B would remove the 16 decay multiplies from. Both reassociate FP32; this change did not.

**The lane layout has since been taken, and it did not have to reassociate anything.**
[Giving one lane 32 columns of one row](gdn-lane-layout.md) instead of four columns of each of four
rows puts a whole row's contraction inside four lanes, so one pair of two-stage quad butterflies
reduces all eight of a wave's rows at once: -19.7% of this phase and +2.4% full-model prompt,
bit-identical. What made it exact is a fact about this file's own reduction that nothing here had
noticed - `warp_sum` leaves **two** distinct roundings in a wave, and `gdn_token` has always applied
both of them to different columns of the same state row.

### Acceptance

[Full-vocabulary comparison](../tools/batch-compare/results/gdn-gate-identity.json): 71,516,160
finite logits at 32, 40, 88 and 128 prefill rows, one executable with `HALO_GDN_PREGATE` 0 against
1, zero differing bits and zero non-finite pairs.

`HALO_GDN_COMPARE=1` needed a repair for this change, because it compared the resident token records
against the sliced reference over their whole width and the last 2*HV columns now hold a different
representation of the same quantity. It compares the convolution columns row-strided, and the gate
columns only when both sides use the same placement; state, ring and recurrent output stay
whole-width and catch a wrong gate one step later. Both arms pass: 0/655360 and 0/661504 token
floats, 0/393216 output, 0/786432 state, 0/30720 ring.

### Installed

Commits `1da2fdc`, `d0aad29` and `43753dd`. The canonical build carrying the shape rule
(`d8ce69811fbb827b73f7a09c28fa875b...`) reproduced
[all 71,516,160 logit bits](../tools/batch-compare/results/gdn-gate-installed-identity.json)
against the pinned-control dump, single-stream `--bench` measured 33.93 tok/s against 33.63, 33.65
and 33.83 on the three preceding installs, and the resident server runs on it. The build before the
rule (`600d8f1f878d6dab9596495b1da14baec8a023ad41cc2b6ccd71db8f47586e87`) reproduced the same
71,516,160 bits and answered `/v1/models`.

Serving never calls `prepare_batch`, so nothing this change touches runs in that path, which is why
`--bench` is flat across all four.
Raw samples, per-round values and the `halo_env` each process saw are in
[`../../data/bonsai2/batch-comparison/gdn-pregate`](../../../data/bonsai2/batch-comparison/gdn-pregate).

## Operation

Resident state is the default for direct integer sequence layouts in committed wide modes 18 and 19. Ordinary serving defaults are unchanged. `HALO_GDN_RESIDENT=0` selects the eight-row control. `HALO_GDN_SPLIT=4|8|16` controls state-row ownership, default 4. Staged layouts default to the sliced state route; explicitly combining staged layout with resident state is rejected.

`HALO_GDN_PREGATE=0` pins the decay gate inside the state kernel's token loop for a whole process
and `=1` pins it to once per (row, head); unset, the shape rule above picks. `--gdn-pregate -1,0,1`
on either measurement tool walks rule, pinned-loop and pinned-hoist inside one process, which is how
both rows of the table above were taken; a pair of processes cannot resolve an effect this size on a
shared host.

```sh
make -j8 bonsai-halo tools/batch_compare tools/batch_profile
# Run each separately so its timeout covers one bounded panel.
HALO_GDN_RESIDENT=0 tools/run-batch-compare --tag resident-control \
  --modes 0,19 --streams 8,32 --prefill-rows 32,128 --rounds 3
HALO_GDN_RESIDENT=1 tools/run-batch-compare --tag resident-selected \
  --modes 0,19 --streams 8,32 --prefill-rows 32,128 --rounds 3
make -C tools/direct-commit sequence_state_check
tools/run-batch-compare --state-check --out ../../data/bonsai2/sequence-state-check/resident.json
```

`HALO_GDN_COMPARE=1` is a diagnostic, never a timing option. On the first resident layer-0 pass containing one sequence in slot 0 with no incoming replay, it clones entry state, runs the sliced reference, and compares every processed token, recurrent output, state and ring bit. A mismatch throws. It allocates temporary buffers and synchronizes. Full-model and mixed-state checks remain necessary because this probe covers only that first layer and entry shape.

`BONSAI_PROFILE_TRACE=1 BONSAI_PROFILE_OUTPUT=DIR tools/run-batch-compare --profile-tool ...` collects kernel durations without performance counters, under the same GPU reservation and service-restoration wrapper.

# Full-model batch comparison

Measures what the batched FFN work is actually worth: end-to-end tokens per second and output
agreement for the whole 27B model, across the execution modes below.

## Measured result

The current [wide-sequence comparison](../../docs/wide-sequence.md) reaches
457 tok/s prompt processing and 205 tok/s aggregate generation with A8, or
541 and 246 with A4 FFNs. The direct producer/consumer layout is the selected
wide path; `HALO_SEQUENCE_LAYOUT=staged` reproduces the capture/restore control. Generation uses 32 streams; prefill uses 128-row
passes over a 256-token prompt. Direct commit preserves its matched control's
logits exactly in the recorded checks; widening changes FP32 reduction order.
All new routes are opt-in and A4 has larger numerical differences.

The [ramped automatic-A8 comparison](results/auto-a8-ramped.md) is the recommended
path's full-model result. Three randomized rounds, Radeon 8060S, 512-token KV
capacity, all 64 layers:

| Workload | Original | `--ffn auto` | Gain |
|---|---:|---:|---:|
| 8-stream generation, aggregate tok/s | 133.1 | 142.5 | 7.1% |
| 32-stream generation, aggregate tok/s | 132.8 | 159.4 | 20.0% |
| Prompt processing, 128 rows/pass, tok/s | 154.9 | 291.2 | 88.0% |

Generation measures eight steps per stream with real, distinct prompts. Prompt
processing measures a 256-token document and includes the LM head on all 128
rows of the final pass for both paths. Loading and static weight preparation are
excluded; all input-dependent inference is timed. The runner pauses the resident
server and records clocks and every timing sample. Its memory scope bounds ordinary anonymous,
file-backed and charged kernel memory. AMD driver allocations are not guaranteed to be charged to
the allocating cgroup, so the device-allocation ceiling and host-headroom admission are separate
requirements rather than claims that `MemoryMax` governs GTT by itself.

Automatic A8 is not bit-identical. It agreed on 32/32 fixed-input decode argmaxes
and 31/32 prompt-processing argmaxes; its four-step continuation matched 125/128
tokens. The [all-mode comparison](results/full-model.md) includes the split
original-FFN mode, whose recorded logits were bit-identical and whose 32-stream
generation reached 148.4 tok/s versus 132.6. A4 reached 164.8 tok/s, but changed
4/32 prompt-processing argmaxes and had mean KL 0.065 on those rows. A4 is an
explicit approximation choice, not the automatic default.

These measurements establish throughput and the recorded numerical differences,
not task-quality equivalence. Raw samples, logits and input provenance are in
[`../../data/bonsai2/batch-comparison`](../../../../data/bonsai2/batch-comparison/README.md).

The [packed-map comparison](results/map-summary.md) records modes 6 through 11.
The [wide sequence results](../../docs/wide-sequence.md) record modes 16 through 19,
with source-matched state checks and separate precision comparisons.

## Modes

| mode | name | `--ffn` | what it is |
|---|---|---|---|
| 0 | deployed | `deployed` | `forward_batch` slices the pass into the original 8-row `forward`; the path we ship today |
| 1 | integer-a8 | `a8` | native batched FFN, integer activations at 8 bits |
| 2 | integer-a4 | `a4` | native batched FFN, integer activations at 4 bits |
| 3 | scaled-a8 | `scaled-a8` | native batched FFN, scaled activations at 8 bits |
| 4 | sliced-deployed | `sliced` | layer-wise scheduling with the original FFN computation |
| 5 | auto-a8 | `auto` | original at 1–4 rows, sliced at 5–31, batched integer A8 at 32–128 |
| 6 | optimized-iu8-a8 | `map-a8` | the newly adopted FFN module, IU8/A8, direct |
| 7 | optimized-scaled-a8 | `map-scaled-a8` | the module, scaled A8, direct |
| 8 | optimized-a4 | `map-a4` | the module, A4, direct |
| 9 | auto-optimized-a8 | `auto-map-a8` | original at 1–4 rows, sliced at 5–31, mode 6 at 32–128 |
| 10 | auto-optimized-scaled-a8 | `auto-map-scaled-a8` | likewise, mode 7 at 32–128 |
| 11 | auto-optimized-a4 | `auto-map-a4` | likewise, mode 8 at 32–128 |
| 12 | single-map | `single-map` | palette operand single-token path |
| 13 | single-state | `single-state` | mode 12 plus finer GDN work units |
| 14 | single-retile | `single-retile` | mode 13 plus finer K splits; FP32 reassociation |
| 15 | single-grid | `single-grid` | original single-token kernel on its own occupancy grid |
| 16 | commit-scaled | `commit-scaled` | mode 10 with direct GDN commit |
| 17 | wide-sequence | `wide-sequence` | mode 10 with wide non-FFN projections at 32 or more rows |
| 18 | wide-commit | `wide-commit` | mode 17 with direct GDN commit |
| 19 | wide-commit-a4 | `wide-commit-a4` | mode 18 with optimized A4 FFNs at 32 or more rows |

Modes 6, 7 and 8 are written to reproduce 1, 3 and 2 exactly. Whether they do is a question this
driver answers rather than assumes: pass `--reference-pairs 6:1,7:3,8:2` to the analysis and read the
count of logit values whose bit patterns differ. Nothing pairs modes on its own, because below 32
rows an automatic mode runs the deployed or sliced path and a family pairing would compare the
wrong two things.

No FFN microbenchmark appears anywhere in this driver. Every timed region is
`Engine::forward_batch` on the loaded model with real weights, real tokenizer output and real
generated tokens. A mode that wins the FFN kernel benchmark and loses here has lost.

## Files

- `tools/batch_compare.cpp` — the driver, built by `make tools/batch_compare`
- `tools/run-batch-compare` — the maintained wrapper (engine owner): takes the GPU lock, pauses the
  resident `bonsai-halo` unit, restores it on exit, and sets `BONSAI_BENCH_SERVER_PAUSED=1`, which
  the driver records in `run.json`. Every dispatch path runs in one process-tree scope with
  `MemoryHigh=26G`, `MemoryMax=28G`, no swap and a 20-minute deadline. Admission also keeps 8 GiB
  outside the scope for the host. The wrapper sets a 26 GiB tracked device-allocation ceiling, so
  the batch engine refuses an oversized set of weight images before repacking them.
- `tools/batch_compare.py` — reads a run and prints the report, including `--reference-pairs`
- `tools/batch-compare/prompts.txt` — 40 distinct short prompts for the decode workload
- `../../data/bonsai2/batch-comparison/<tag>/run.json`: raw timings, telemetry, quality manifest
- `../../data/bonsai2/batch-comparison/<tag>/logits-<shape>-r<rows>-mode<N>.f32`: full-vocabulary logits
- `tools/batch-compare/results/`: small derived reports committed with the engine

## Engine API it uses

```c++
void load(const std::string & path, int nthreads, int n_slots = 1, int context_capacity = 32768);
void prepare_batch(int max_rows = 128, unsigned modes = 0);   // once, never timed; 0 means every mode
void forward_batch(std::vector<Seq *> &, const std::vector<std::vector<int>> &, bool with_logits,
                   std::vector<int> * argmax = nullptr, std::vector<float> * all_logits = nullptr);
void prepare_sequence();                 // after prepare_batch, before modes 17 through 19
int batch_mode;                           // modes 0 through 19 as listed above
size_t device_bytes;                      // resident device memory, read after load and after prepare
```

```c++
size_t ffn_batch_bytes(const FfnBatch *);            // kernels/ffn_batch.h, owned by the kernel worker
size_t ffn_batch_weight_bytes(unsigned modes);
size_t ffn_batch_image_bytes(int image);
size_t ffn_batch_resident_image_bytes(const FfnBatch *, int image);
bool   ffn_batch_supports(const FfnBatch *, int mode);
```

Assumptions the driver makes, which are worth checking if a run looks wrong:

- One `Engine` holds one set of weights and one set of slot state; `batch_mode` can change between
  passes without reloading anything.
- `forward_batch` accepts up to 128 rows per pass, including many rows of one sequence, and
  advances each `Seq` through its eight-row slices. `len` advances by all supplied tokens;
  `keep`, `last_n` and parity describe the final slice, which is what recurrent replay consumes.
- `all_logits` comes back as `nrows * VOCAB` float32, row-major, when `with_logits` is set.
- `reset_seq(s)` zeroes that slot's GDN state and conv ring and resets `len/keep/parity/last_n`.
  The KV cache is positional and gets overwritten from position 0, so no separate clear is needed.
  The driver calls `reset_seq` on every slot it is about to use, before every timed region.
- `prepare_batch` is always called with 128, whatever shapes a run measures: prompt prefill during
  decode and quality setup uses passes wider than the measured shapes, and an optimized mode refuses
  a pass wider than the prepared capacity.
- `device_bytes` accumulates every device allocation the engine makes, so the difference across
  `prepare_batch` is the module's cost. `run.json` records both sides and the module's own
  accounting, because a repacked weight image is a second full-model representation held next to
  the deployed weights for as long as an optimized mode is reachable.
- `prepare_batch`'s second argument is a bit per module mode. The driver derives it from `--modes`,
  so a run builds only the images it will use, and a mode whose image was not built is refused
  before anything is measured rather than throwing part way through a sweep.

## Weight images and what a run costs

The module holds up to five weight images, several of them shared between modes. Every measured
mode maps to the module mode it actually enters: 1, 2, 3, 6, 7 and 8 to themselves, 5 to 1, and
9, 10 and 11 to 6, 7 and 8, because an automatic mode only leaves the deployed path at 32 rows and
above. Modes 0, 4 and 12 need no image at all: mode 12 keeps the original scheduling and puts the optimized
kernel inside it, so nothing repacked is read.

When no measured mode reads an image the driver asks for `1u << 30`, the workspace-only mask, rather
than passing zero and getting all 16.58 GB. A run of modes 0 and 12 therefore repacks nothing.

That mask is the difference between a cheap run and an expensive one: every mode at once is 16.58 GB
of weight images, the three original controls alone are 4.55 GB, and the A4 wide-only restriction
brings the full set to 13.10 GB. `run.json` records the mask, the budget for it, the budget for every
mode, and each image's size and residency by name (`SCALES`, `SLICE2`, `LANE2`, `PAIR`, `DENSE5`).


Mode 8 normally keeps two images and picks per call. `--a4-images wide` or `--a4-images dense` keeps
one and uses it everywhere, which is a throughput-for-bytes trade worth measuring rather than
assuming: the kernel owner puts it at about 13% at exactly 32 rows for wide-only and 35-40% at 128
rows and above for dense-only.

```sh
--a4-images both|wide|dense       # both is the default and the fastest
```

The driver refuses to run when `HALO_STOP` or `HALO_LAYERS` is set nonzero. Those truncate the
forward pass, which would let the baseline time a partial model against a full 64-layer mode.

## Build and run

```sh
make tools/batch_compare                          # or tools/batch-compare/build.sh
tools/run-batch-compare --only quality --rounds 0 --modes 0,4,1 --quality-rows 8 --slots 8
tools/run-batch-compare --tag first-native        # full default sweep
python3 tools/batch_compare.py tools/batch-compare/results/first-native
```

For the adoption of modes 6–11, numerics first and timing second:

```sh
# 1. matched numerical families, including row counts the tile width does not divide.
#    Prefill-shaped only, so 128 rows needs one slot instead of 128.
tools/run-batch-compare --tag ffn-adopt-quality \
  --only quality --quality-shapes prefill --quality-rows 32,40,88,128 \
  --modes 1,2,3,6,7,8 --reference-pairs 6:1,7:3,8:2
python3 tools/batch_compare.py ../../data/bonsai2/batch-comparison/ffn-adopt-quality \
  --out tools/batch-compare/results/ffn-adopt-quality.md

# 1b. single token: original scheduling against the optimized kernel at one row
tools/run-batch-compare --tag single-map \
  --modes 0,12 --only decode,quality,multistep --streams 1 --quality-rows 1 \
  --slots 1 --multistep-streams 1 --rounds 3 --gen-steps 32 \
  --reference-pairs 12:0
python3 tools/batch_compare.py ../../data/bonsai2/batch-comparison/single-map \
  --out tools/batch-compare/results/single-map.md

# 2. throughput of the deployed path against the automatic routes
tools/run-batch-compare --tag ffn-adopt-tps \
  --modes 0,5,9,10,11 --streams 8,32 --prefill-rows 32,128 --prefill-tokens 256 \
  --rounds 3 --gen-steps 8 --quality-rows 32 --multistep-streams 32 --multistep-steps 4 \
  --reference-pairs '9:5,10:3@rows>=32,11:2@rows>=32'
python3 tools/batch_compare.py ../../data/bonsai2/batch-comparison/ffn-adopt-tps \
  --out tools/batch-compare/results/ffn-adopt-tps.md
```

The driver records `--reference-pairs` in `run.json` and never interprets it; the analysis applies
what the run recorded unless you override with its own `--reference-pairs` or ignore it with
`--no-run-pairs`. When no pairs are given, the report prints the pair string that fits the modes the
run measured, and applies none of it.

Smoke first: `--rounds 1 --steps 2 --modes 0,4,1` costs a couple of minutes and catches a broken
mode before an expensive sweep. Mode 4 next to mode 0 answers the first question a bad result
raises — schedule or FFN.

Defaults: context 512, 32 slots, 384 document tokens, prefill at 32 and 128 rows per pass, decode at
8 and 32 streams, 3 rounds, 8 generated steps, quality at 8 and 32 rows, multistep 32 streams x 4
steps, modes 0,4,1,2,3. Deliberately small: a first bounded run, not the evidence for a decision.

Useful flags:

```sh
--modes 0,4,1        --prefill-rows 32,128    --streams 8,32
--rounds 5           --gen-steps 32 (--steps) --quality-rows 8,32 --quality-ctx 64
--context 512        --slots 32               --prefill-tokens 1024
--quality-shapes prefill,decode               --reference-pairs 6:1,7:3,8:2
--multistep-streams 32 --multistep-steps 4
--prefill-logits last|all|none                --hip-events
--telemetry-ms 5     --telemetry-raw          --telemetry-cmd "kelana-probe --json"
--only prefill,decode,quality,multistep       --no-logits-dump
```

A decode-shaped quality group needs one slot and one distinct prompt per row, so 128 decode rows
would mean 128 slots and 128 prompts. `--quality-shapes prefill` measures wide row counts without
that: a prefill-shaped group needs one slot, and 128 rows after a 64-token context fit the default
512-token capacity. A group the run cannot serve is skipped with its reason in `quality_skipped`,
printed to stderr and repeated in the report, rather than aborting the run or vanishing.

`--only quality` runs no timed region, so the clock ramp is skipped and no empty rounds are
recorded. Numerics do not care whether the GPU reached its boost clock.

`--prefill-tokens` must not exceed the document's token count, and `prompt + gen steps` must fit in
`--context`. Quality dumps cost disk: `rows * 248320 * 4` bytes each, so 32 rows is 31.8 MB per mode
per shape. `results/.gitignore` keeps them out of git; keep a run's `run.json` when it matters.

## The workloads

**Prefill.** One sequence, one real document (`PLAN.md` by default), fed in passes of 32 or 128
rows. Mode 0 goes through the same entry point and slices internally by 8, so the token accounting
is identical across modes. The last pass asks for logits and the earlier ones do not, which is what
a real prefill does; `logits_policy` and `lm_head_cost_included` record this, and
`--prefill-logits all|none` moves the lm_head cost in or out explicitly.

"Last pass" is not "last token". The final pass carries a full row block and the lm_head runs on
every row of it: 128 logit rows at 128 rows per pass, 32 at 32 rows per pass. Every mode pays the
same cost, so the mode comparison is fair, but the 32-row and 128-row configurations carry different
lm_head work and are not directly comparable to each other. The report states the row count for
each configuration.

**Decode.** 8 or 32 independent sequences, each with its own distinct prompt from `prompts.txt`,
one row per stream per step. Prompt prefill and slot reset happen outside the timed region and are
reported separately as `setup_s`. The timed loop runs a fixed number of greedy steps, so
`tokens_emitted = streams * steps` and every one of those is a token the model produced. Streams
that emit EOS keep going: `eos_tokens_emitted` counts them and `eos_policy` says so in the JSON. The
headline number is `aggregate_tokens_per_s`.

**Multistep.** 32 sequences carried several greedy steps from their own prompts, untimed, token
matrix compared against mode 0 step by step. Two facts shape how to read it: step 0's token is the
argmax of the prompt prefill, so divergence there is a prefill-pass difference rather than a decode
one. Mode 4 holds the original FFN computation fixed; a difference there calls for inspection
of state, replay, parity and floating evaluation order before blaming a new FFN map.
An approximate mode carries a numeric difference into every step, so its
divergence at any step, early or late, can be numerics alone. The report names the first divergent
step and stream and prints both texts.

**Quality.** Full-vocabulary logits on fixed teacher-forced tokens, in two shapes: `prefill`
(rows from one sequence, after `--quality-ctx` tokens of context) and `decode` (one row per stream,
after their real prompts), at 8 and 32 rows. Every mode sees byte-identical token inputs, and the
comparison is mode 0 against every other mode including the mode 4 control. The report gives, per row,
top-1 agreement with mode 0, KL in both directions, the largest centred logit difference, top-5
overlap, where mode 0's chosen token ranks under the other mode, and the exact count of rows whose
argmax changed. Output text alone would hide a mode that is subtly wrong; this does not.

Row counts the kernel's token-tile width does not divide are worth measuring deliberately: 40 and 88
rows exercise the fragment slack past the batch that the row guard is supposed to discard.

## Matched reference pairs

Mode 0 answers "what does this cost us". It does not answer "did this reimplementation reproduce the
arithmetic it was written to reproduce", and for an approximate mode the two questions have
different answers: mode 8 can match mode 2 to the bit and still move logits away from the deployed
path, because mode 2 already did. Both tables are printed, and the mode 0 tables stay first.

Pass pairs to the analysis as `MODE:REF`, comma separated, each optionally scoped:

```sh
python3 tools/batch_compare.py RUN --reference-pairs 6:1,7:3,8:2
python3 tools/batch_compare.py RUN --reference-pairs '9:5,10:3@rows>=32,11:2@rows>=32'
python3 tools/batch_compare.py RUN --reference-pairs '10:7@section=quality+shape=prefill'
```

Conditions attach with `@` and join with `+`: `rows` against `>= <= > < =`, `shape=prefill|decode|multistep`,
and `section=timing|quality|multistep`. `rows` is rows per pass for prefill, streams for decode and
multistep, and the row count for quality, which is exactly the number that decides how an automatic
mode dispatches. That is why the scoping exists: mode 10 only reaches the module at 32 rows and
above, so `10:3` unscoped would compare the sliced deployed path against scaled A8 at 8 streams and
report a difference that means nothing.

`9:5` needs no scope, and for a stronger reason than convenience: both modes are automatic and
switch on the same row counts, so they agree pass for pass including the narrow ones.

**A scope selects a reported shape, not a route.** It says which table the pair is applied to. It
does not certify that every pass behind that number took the matched path: teacher-forced context,
per-stream prompt prefill and the tail pass of a document can each carry fewer than 32 rows, and an
automatic mode runs the deployed or sliced path there however wide the reported rows are. Read
`10:3@rows>=32` as a claim about the 32-row shape it names, not about the whole run. The timing run
uses `9:5` alone for this reason; the direct pairs `6:1,7:3,8:2` have no such ambiguity, because
those modes take one path at every row count.

For each pair the report gives top-1 agreement, the number of logit values whose float32 bit
patterns differ, how many rows are affected, RMS and maximum absolute difference, non-finite counts
on both sides, the multistep token match, and the throughput ratio at each shape.

**Non-finite logits never earn a pass.** A NaN reproduces its own bit pattern perfectly, so two
identically broken dumps would otherwise score as a flawless reimplementation, and a NaN difference
never reaches an RMS because the subtraction is NaN too. The driver counts non-finite logits per
quality entry and warns as it runs; the analysis counts them per side, computes RMS over the finite
elements only, drops affected rows out of the mode 0 aggregates and says how many it dropped, and
reports a bit-for-bit match on a dump containing non-finite values as agreement about a broken
result rather than a reproduction.

## Measurement hygiene

- Everything runs in one process against one loaded model. Mode order is reshuffled each round from
  the recorded seed, so a warming or thermal trend does not land on one mode.
- Every mode runs the decode shape untimed for at least two seconds before the first round (`warmup_s`). A count of short passes alone did not guarantee a completed clock ramp.
- New records include Git revision, executable and input hashes, and a hash of uncommitted source
  differences. Source hashing enumerates `src`, `kernels`, `vendor`, the `Makefile` and this driver
  through `git ls-files --cached --others`, so a header added since the last run is hashed without
  anyone remembering to list it; `source_sha256_rollup` reduces that to one digest.
- `device_memory` records engine `device_bytes` before and after `prepare_batch`, the module's own
  byte accounting and its weight-image size. An optimized mode keeps a second representation of all
  64 layers resident, which is a deployment cost whatever the throughput says.
- Sequence state is reset before every timed region; prepare and prefill sit outside it.
- Each timed region ends with a stream synchronise before the clock stops, so a pass that copies
  nothing back is still fully accounted.
- Individual wall times survive into the JSON: `pass_wall_s` per prefill pass, `step_wall_s` per
  decode step, and the per-round list in the report. The report prints spread, not just a median.
- `--hip-events` adds device-side elapsed time around each region for cross-checking the wall clock.
- Telemetry is host sysfs from the amdgpu hwmon node (gfx clock, edge temperature, board power,
  busy percent), sampled every 5 ms during timed regions. It is context for a slow round, not a
  measurement of the kernel, and it does not prove the GPU was otherwise idle. `--telemetry-cmd`
  captures an external probe's stdout before and after the run.
- Host counters are captured around every timed region and stored as `host`: CPU user and system
  time, major and minor faults, voluntary and involuntary context switches, PSI stall seconds for
  cpu, memory and io, and MemAvailable and SwapFree either side. The reads happen outside the clock.

## Stalled samples

A timed region can stall on the host while the GPU sits idle. Clocks and busy percent show the
symptom and cannot name the cause: an idle, cooling GPU looks identical whether the process was
waiting on memory reclaim, faulting pages back from disk, preempted by another job, or blocked on
I/O. That is what the host counters are for.

The report flags any timed sample more than 20% above the median for its mode and shape, and keeps
it in every aggregate. A stall is a real thing that happened to this machine, and dropping it would
make the benchmark look steadier than the machine is. The median is the headline statistic precisely
so one stall cannot move it, and the spread figure is where it shows.

For each flagged sample the report names the round, the excess over the median, which individual
pass or step absorbed it and what share of the region that was, then reads the host counters:

- low CPU busy fraction means the process was blocked rather than computing;
- a memory or io PSI stall covering a quarter or more of the excess names that resource;
- major page faults mean pages came back from disk or swap, with MemAvailable and SwapFree as
  corroboration;
- blocked with no host resource stall at all points at the device, its queue or the driver, which
  these counters cannot see into.

When nothing accounts for the excess the report says so rather than inventing a cause. Records
written before these counters existed are still flagged, and say the stall cannot be attributed.
- `run.json` is rewritten after every round, so a crash in round 3 keeps rounds 1 and 2.

## What a run does not establish

Three rounds of eight steps and eight teacher-forced rows are a smoke test. They catch a mode that
is broken, badly slower, or numerically off. They do not establish that a mode's quality is
acceptable for deployment, and they do not resolve throughput differences smaller than the measured
run-to-run spread, which the report prints for exactly this reason. If a result matters, raise
`--rounds`, `--gen-steps` and `--quality-rows` and say which numbers came from which run.

## Ownership

This directory plus `tools/batch_compare.cpp` and `tools/batch_compare.py` are the comparison
driver only. The engine, the build and `forward_batch` itself belong to the engine owner; the
batched FFN kernels belong to the kernel owner. The driver never touches GPU services or
configuration, and it only reads sysfs.

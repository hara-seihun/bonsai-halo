# Bonsai Halo

A from-scratch HIP inference engine for Ternary Bonsai 2 27B (Qwen3.8 hybrid GDN/attention), specialized for one Radeon 8060S (gfx1151) in a Ryzen AI MAX+ 395 with LPDDR5X-8000. It repacks PTQ1_0 weights into wave-friendly ternary tiles and executes the model with custom kernels; llama.cpp's `libllama` supplies the tokenizer only. The target has 64 layers; device footprint is about 6.6 GiB. ROCm 7.2.3 is recorded in later MoE measurements; the original decode comparison does not pin a runtime version.

**Measured throughput:** 34.5 tok/s plain single-stream greedy decode versus 21.0 tok/s for llama.cpp HIP on PTQ1_0 (1.64×); 65–99 tok/s with DFlash2 target-verified speculation, workload dependent; 517.7 tok/s aggregate at 128 independent streams on the *approximate* A4/INT8 route. These are different workload and precision coordinates, not one common speedup factor. The engine's plain-decode example is `./bonsai-halo --bench -n 100` after a short prompt; the recorded llama.cpp baseline command is `llama-bench -ngl 99 -fa 1 -n 64` from the `bonsai-hip/build-hip` HIP build. The baseline's context, batch/ubatch, threads, exact build revision and prompt/depth settings were not retained with that 21.0 tok/s figure. A separate two-chunk perplexity comparison *does* record llama.cpp `-c 512 -b 512 -ub 512 --chunks 2`, but those are not documented as the decode-baseline settings.

[PLAN.md](PLAN.md) holds the design and measured history. [The M1 engine shootout](docs/engine-shootout.md) owns the max preset and quality path. [Kelana's catalogue](https://github.com/hara-seihun/kelana) classifies cross-project research. This file is the operating manual.

## Code, measurements and proofs

- **Single-stream inference and target-verified drafting:** [build and run](#build-and-run), [combined full-model measurements](docs/full-tps-20260921.md).
- **Batch inference, prefill and concurrent serving:** [128-stream generation](docs/generation-128.md), [wide prompt ingestion](#wide-prompt-ingestion), [served concurrency](docs/served-concurrency.md).
- **Original measurement evidence:** [public receipt bundle](evidence/README.md), retaining timings, numerical comparisons and output digests with private text removed.
- **Mathematical proofs:** [Kelana's Lean sources](https://github.com/hara-seihun/kelana/tree/main/Kelana), [ternary 2×2 proof and exhaustive checker](https://github.com/hara-seihun/kelana/blob/main/research/toy2/PROOF.md), [Bonsai proof problems](https://github.com/hara-seihun/kelana/blob/main/BONSAI.md). All 162 tracked Lean files from the recovered Kelana source are public. Native operand-map checkers also live in this repository's [`kernels/`](kernels/) directory.

These repositories publish the inference implementation, research source, formal proofs and reports. Model weights, calibration datasets and large raw arrays are separately acquired inputs, not included here. The receipt bundle is recovered historical evidence, not a new benchmark on the replacement server. External programme results linked from the research catalogue retain their own source ownership; a catalogue citation does not claim their entire workspace is mirrored here.

## Status

The [September 21 combined full-model measurement](docs/full-tps-20260921.md) records
single-stream and batched generation with the available optimizations combined, including
where enabling approximate modes loses rather than gains. The subsequent
[128-stream generation measurement](docs/generation-128.md) reaches 517.7 aggregate
tokens/s on the approximate A4/INT8 route and documents the explicit managed-allocation
policy needed to fit 128 streams on this host.

Single stream, greedy, short prompt (`--bench`):

| | tok/s | ms/token or step |
|---|---:|---:|
| llama.cpp HIP, PTQ1_0 (`build-hip` in the bonsai-hip clone) | 21.0 | 47.6 |
| **bonsai-halo, plain decode** | **34.5** | 29.0 |
| bonsai-halo + Qwen3.8 MTP head (`--mtp`, 2 drafts) | 57-60 | 41 per step, 2.3-2.5 tokens/step |
| bonsai-halo + DFlash2 (`--dflash`, block 8, Q4 drafter weights) | **65-99** | 57 per step, 3.8-5.8 tokens/step |
| bonsai-halo `--batch 8` | 126-134 aggregate | 60-64 per step |
| bonsai-halo prefill, `wide-deployed` (the default) | 250-489 | |
| bonsai-halo prefill, `--prefill-ffn off` (8-row passes) | 145-155 | |
| bonsai-halo prefill, `--prefill-ffn wide-commit-a4` | 567 | |

Both drafters use target verification; acceptance is workload dependent, with code and reasoning higher than prose. The [native proposal comparison](docs/speculative-compare.md) found that the target itself could change greedy output across repeated serial runs because split floating residual updates raced. Ordered per-tile K parts now define one reduction order. Six serial reproductions have identical logits, and all ten tested prompt paths and final next decisions match across serial, DFlash2 and the experimental proposal arms. Copy arbitration does not establish a speedup over DFlash2; repaired suffix maps lose. The installed drafter is DFlash2, not DSpark. Earlier plain-decode logits matched llama.cpp's CPU reference at cosine 0.99999 with the same argmax and top-5 on 5-token and 282-token prompts; those measurements predate the ordered reduction.

## Build and run

```sh
# Place a HIP-enabled llama.cpp checkout at ../bonsai-hip, or set LLAMA=/path/to/checkout.
# Build it in build-hip first; provide the PTQ1_0 GGUF separately.
make -j LLAMA=../bonsai-hip   # hipcc, gfx1151; libllama is used only for tokenization
./bonsai-halo -p "Why is the sky blue?" -n 200            # chat template, no thinking
./bonsai-halo -p "What is 17*23?" --think -n 400          # chat template with <think>
./bonsai-halo -p "The capital of France is" --raw -n 40   # raw completion
./bonsai-halo --bench -n 100                              # timing only
./bonsai-halo --temp 0.7 --top-k 20 --top-p 0.95 --seed 1 -p "..."   # sampling (default greedy; not combined with drafters)
./bonsai-halo -p "$(cat long_prompt.txt)" --raw -n 200 --prefill-ffn wide-deployed   # wide prompt ingestion
./bonsai-halo -p "..." --think --dflash ~/data/bonsai2/drafters/dflash2.safetensors            # DFlash2 speculative decode
./bonsai-halo --bench --dflash ... --prompts bench/drafter-prompts.txt --draft-weights q8,q4      # drafter coordinates, one process
./bonsai-halo -p "..." --think --mtp ~/data/bonsai2/drafters/qwen3.8-27b-shard18.safetensors --draft-n 2   # MTP head
./bonsai-halo -p "..." --batch 8 [--prompts FILE]                     # 8 sequences per pass (one prompt per line, else the same prompt)
```

Drafters are bf16 safetensors quantised to tiles on first use and cached beside the file. DFlash2 uses [four-bit tiles](docs/drafter-q4.md) (`.q4`, 0.93 GB) by default and `--draft-weights q8` selects the eight-bit control (`.q8`, 1.83 GB); the MTP head is Q8. `tools/draft_quant DRAFTER [q8|q4|both]` builds a cache without a GPU. `dflash2.safetensors` is `incoai/Qwen3.8-27B-DFlash2` (3.8 GB); the MTP head is shard 18 of `Qwen/Qwen3.8-27B` (its `mtp.*` tensors; the Qwen3-Next style norms store weight-1, the loader adds 1). Both were trained against Qwen3.8-27B and work on Bonsai 2 because it is a ternary finetune of that model.

Options: `-m` model (default `PTQ1_0.gguf` in the working directory; PQ2_0 decodes to the same trits and works too), `-n` tokens, `--v1` per-op kernel path, `--no-graph`, `-t` repack threads.

First run repacks the GGUF into HALO tiles (3 s on 32 threads) and caches them next to the model as `PTQ1_0.gguf.halo` (5.9 GB, rebuilt when the GGUF changes). Warm start loads in about 1 s. Device footprint is 6.6 GB, all in GTT.

The maximum context is 32768 tokens (`MAXCTX`). The persistent kernel handles eight rows per slice (`RMAX`), while the wide-batch path gathers up to 128 rows and sequence slots. The original prefill schedule uses eight-row passes.

The [throughput budgets](docs/throughput-budgets.md) separate perfect use of the
current ISA from hypothetical ternary-native hardware, with reproducible work
and traffic counts in `tools/prefill_budget.py`.
[What each phase does when the shader clock moves](docs/clock-power.md) measures
those budgets against the clock that actually executed them. A 128-row prefill pass
is 89% clock-elastic at 2.0 GHz and 14% at 2.9 GHz, and the recurrent phase of a
32-stream generation step does not respond to the clock at all.
[What sets that clock](docs/power-budget.md) is the socket this GPU shares with
sixteen CPU cores, the fabric and LPDDR5X: the package sustains 120 W, is never
thermally limited, and gives the shader array 52-56 W of it. Host CPU work costs a
prompt pass 7.6% and a 32-stream step 2.9% by taking that budget, so
`tools/run-batch-compare` records the clock, the power rails and the host load of
every panel and says when the machine was not quiet.
The [measured prefill gap](docs/prefill-gap.md) locates the remaining time with
per-launch events and device phase timestamps.

## Continuing optimization

The project owner's September 23 priority is [Qwen3.6-35B-A3B MoE inference](docs/qwen-moe.md):
a pinned runnable reference on this GPU, packed expert computation and new Kelana
representations for the routed expert sum. Its data and acquisition receipts live
in `../../data/qwen-moe/`; the existing joint lane continues this work.
[Real layer-0 route traffic](docs/qwen-moe-route-traffic.md) finds prompt-window
expert reuse but almost no adjacent-token reuse. Its byte counts are conditional
on tile capacity, not native speed measurements.

The [SGLang AWQ comparison](docs/sglang-awq.md) traces its official Bonsai route and
records draft upstream PR #40863. Connecting direct packed GEMM on AMD reduces the
measured real-weight linear operations by 6.65x to 7.48x, with floating-point
activations unchanged. This is not a whole-model or NVIDIA speed comparison.

The joint Bonsai and Kelana work pursues cheaper complete computations and faster inference on existing hardware. The direct M1 shootout is described in [the engine-shootout study](docs/engine-shootout.md). The first round's [sequence weight-cover result](docs/seq-weight-cover.md)
rejects a numerically unchanged prefetch because its extra registers reduce occupancy. The linked
[Kelana GPU consumer study](https://github.com/hara-seihun/kelana/tree/main/research/quantization-discovery/gpu-direct)
records why the CPU sign-folded lookup win does not transfer to its tested GPU lowering.
Kelana's [post-O-aware mixed Q allocation](https://github.com/hara-seihun/kelana/tree/main/research/quantization-discovery/subbit/post-o-allocation) solves the frozen eight-of-sixteen head choice against the complete attention/O response. At identical .40778 matrix BPW, held post-O squared error falls from .024280 under attention-KL allocation to .022787; no engine map or native timing changed.
The [mode-4 drafter capture repair](docs/serve-drafted-batch.md) restores all five feature
planes and has a replay regression check. The associated concurrent speculative-serving
prototype was not accepted because of an output divergence and a four-client slowdown.

## Wide sequence execution

[Wide sequence projections and direct state commit](docs/wide-sequence.md) raise
128-row prompt processing from 291 to 478 tok/s with A8, or 324 to 590 with A4
FFNs. At 32 generation streams, the corresponding aggregate rates are 210 and
253 tok/s. The selected route writes consumer layouts directly and keeps
[GDN state resident](docs/resident-gdn.md) across prompt tokens. Both changes
preserve their predecessor's output bits. The new routes remain opt-in: `--ffn wide-commit` or
`--ffn wide-commit-a4`. Both widen the non-FFN projections and stop replaying
already accepted rows. A4 remains approximate; see the linked numerical results.

`--ffn commit-scaled` isolates direct commit and `--ffn wide-sequence` isolates
widening. Committed passes cannot reject tokens afterward. The ordinary serving
and speculative paths keep their current state contract and defaults.

The recurrent [decay gate is now evaluated once per token and head](docs/resident-gdn.md#where-the-decay-gate-is-evaluated)
rather than once per token inside every state unit. `softplus` goes through
full-precision `log1pf`, 83 scalar-float instructions, and the value is uniform
over a row and head, so 32 waves were computing the same number. The state phase
of a 128-row pass falls from 24.2 to 16.3 ms and full-model prompt processing
rises from 687.4 to 716.7 tok/s, with all 71,516,160 compared logit bits equal.

What it is worth depends on how many rows a *sequence* contributes, not how many
the pass has: a state unit loads 64 kB of head state once and then walks that
sequence's tokens. In a generation step of one to four rows per sequence the
hoist measures 3.6% worse, because the gate was independent work the wave issued
while its state loads were in flight. `launch_gdn_resident` picks by shape, and
both arms produce the same bits. `HALO_GDN_PREGATE=0` or `1` pins one.

The token loop's [row reductions now issue as one block](docs/gdn-token-reduce.md)
instead of one per state row. The `lane == 0` store at the end of a row is a
branch, so the compiler had been emitting the rows as separate basic blocks and
covering each row's cross-lane butterfly with nothing while the other rows' work
sat where the scheduler could not reach it. The state phase of a 128-row pass
falls from 17.5 to 14.7 ms and full-model prompt processing rises from 796.0 to
809.0 tok/s, bit-identical. The same change is a measured null on a 32-stream
generation step, which is [the decode map](docs/decode-map.md)'s memory bound
being paid rather than a defect in the instrument.

The A4 FFN's block loop now [requests its two scalar operands a block ahead](docs/ffn-block-pipeline.md)
of the block that spends them, which is what its weight cursor and activation stage already did. The
compiled block had been waiting 3 to 24 issue slots after asking for a weight scale or a token
scale, three times a block. The 32-stream generation step falls from 69.77 to 69.12 ms (458.6 to
463.0 tok/s aggregate) with every logit bit unchanged, and a 128-row prompt pass is a measured null.
The same panel repaired the activation stage's default, which `env_int` had been overwriting with
the off arm since it landed: another -1.46% of the step. `HALO_FFN_SPIPE=0` and `HALO_FFN_BSTAGE=0`
restore the two predecessors.

The deployed FFN slice's unit now [drains through an eighth LDS slot instead of a wave's
registers](docs/ffn-slice-activation.md). Every unit had ended with four sequential passes in which
wave 0 read 56 LDS floats, added 56 and wrote 8 outputs between two barriers, and a 128-row pass
runs 360,000 of them. Publishing all eight partials and reducing eight ways gives back 21 registers
(217 VGPR against 238, same waves, same grid), and a second buffer removes the write-after-read
barrier. The `ffn` phase of a 128-row pass falls 3.9% and full-model prompt ingestion goes 516 to
527 tok/s, with residual FNV-64 `9171727463267762618` in every sample and one hash over 381,419,520
compared logit values. `HALO_MVW_ARM=0` restores the predecessor.

The same panels decompose that phase, which is 74% of a prompt pass, by collapsing one stream's
footprint at a time: its **activation stream is 4.6x its weight stream and worth 1.6%**, the weight
stream is 10%, the unit boundary is 12% and the rest is counted issue. That closes the activation
coordinate and explains three published nulls. `bench/wmma_chain` adds the constant those censuses
were missing: a *dependent* `v_wmma_i32_16x16x16_iu8` chain issues at the same rate as eight
independent ones, at every occupancy.

[That constant is 34.3 cycles per SIMD32](docs/ffn-matrix-rate.md), not the 21.9 issue slots this
repository converted censuses with, and the slots beside it are free: fourteen independent VALU
operations per matrix instruction — the deployed FFN block's own ratio — cost 0.5%. `bench/mvblock`
measures both, reading the shader clock inside the kernel so a busy socket cannot move the answer,
and two probe errors are corrected with it: `bench/wmma_chain` counted 40 SIMD32 where this part has
80, and `bench/wmma_cost` normalises by a `v_fmac_f32` it assumes is one cycle and which measures
1.32. Reconverted, the deployed block is 1098 cycles of matrix and about zero of everything else,
which is why five instruction cuts in it converted at a fifth or worse, and the phase's floor is 78
to 101 ms against 180.7 measured. `-DHALO_SLICE_CYCLES=1` carries the same instrument into
`ph_matvec_w` and prices a unit's tail against its own block loop with no clock in the comparison.

The head's ternary operand map is [one `v_perm_b32` per operand dword](docs/head-operand-arm.md)
rather than a shift and an or, which is −13.4% of its block loop and −3.6% of the phase, bit for
bit the same logits. The same panel prices what the head is really waiting for: halving its weight
traffic at the selected width buys nothing, and a −13.4% issue cut converts at 0.27, so neither
bound explains most of its time. `HALO_HEAD_OP=0` restores the peel;
`tools/batch_profile --head-op 0,1 --head-tt 1,2,4` walks both axes in one process.

The same trade on the sequence projections is a
[measured loss, and the ablation says why](docs/seq-pair-operand.md). Their four-bit expansion can
read the A4 FFN's nine-valued weight-pair codes at identical bytes, which is 325 to 267 work slots
a block and bit-identical output, and it runs **1.9 to 2.5% slower** at every deployed shape. Pin
the block's two streams into cache and the same cut becomes 4.2 to 5.7% faster: those instructions
were the cover between a block's loads and their use, not its cost. Nothing shipped; the map, its
host proof (`make kernels/seq_op_check`) and the `-DHALO_SEQ_PIN` ablation stay for the next
consumer whose block loop really is issue-bound.

A state unit owns [twice as many rows when one sequence owns the grid](docs/gdn-unit-width.md).
The split of a head's 128 rows had only ever been compiled at 4, 8 and 16, and
its two wider points answer the question the narrow ones asked: a wave that owns
eight rows instead of four pays the per-token gate, `k`/`q` fragments and loop
once for twice the work, which is -6.4% of the state phase on a prompt pass. It
pays only at one sequence, where 192 units of a 160-block machine leave wave
slots idle; at two sequences and above the same change costs 1 to 5%, so
`gdn_resident_split_for` picks by `nseq` and every arm is bit-identical.
`HALO_GDN_SPLIT=2,4,8,16` pins one.
A state unit's [lane ownership is now the pass's choice](docs/gdn-lane-layout.md).
Giving one lane 32 columns of one row instead of four columns of each of four
rows puts a whole row's contraction inside four lanes, so the two five-stage
`warp_sum` butterflies a row used to cost per token become one pair of two-stage
quad butterflies for all eight of a wave's rows. The state phase of a 128-row
pass falls 19.7% to 10.8 ms and full-model prompt processing rises from 883.6 to
906.6 tok/s over eight paired rounds, bit-identical. It is 10.6% *worse* at one token per sequence, where
the phase is a state round trip rather than an instruction stream, so
`launch_gdn_resident` picks by rows per sequence and `HALO_GDN_COLS=0` or `1`
pins one. Bit-identity here rests on a fact worth knowing before touching any
cross-lane reduction in this engine: `warp_sum` leaves **two** distinct roundings
in a wave, and `gdn_token` has always applied both of them to different columns
of the same row.

The persistent route's state phase [stopped evaluating one gate thirty-two times](docs/gdn-sliced-gates.md).
`ph_gdn` deals (sequence, head, row group) units and every wave of every unit re-derived each
token's decay gate through a full-precision `log1pf`; the block token cache now holds the evaluated
gates, written once per (row, head) by `ph_gdn_pre`, and a one-sequence verify pass deals its state
in 96 units instead of 192. The drafted verify pass falls 1.5% (`gdn` -26 to -39%), drafted generation
rises 1.4%, single-stream plain decode 1.5%, with the device's own sliced-against-resident comparison
at zero differing bits. `make DEFS='-DHALO_GDN_BLK_GATE=0 -DHALO_GDN_PK_SPLIT=4'` builds the predecessor.

A packed state can now be allocated [the region it actually occupies](docs/state-region-stride.md)
instead of an fp32-sized one. At eight bits a (slot, layer) region needs 1.152 MiB of the 3.000 it
reserved, so a sequence slot's state falls from 144 to 55 MiB and a 64-slot engine holds **both** A4
weight images inside the default budget - the configuration `hipMalloc` used to refuse - which is
worth 483.8 tok/s at 32 streams against 446.4, and 616.5 at 64. 128 slots fit in 32.0 GB rather
than 43.9. It moves addresses and no values: 32 of 32 streams' tokens and the prompt path's logit
hashes are identical to the build before it. `HALO_GDN_REGION=packed` selects it and the default
stays the fp32-sized region, because the mixed commit/replay acceptance that would certify a packed
coordinate is currently failing on main for an unrelated reason the same document bisects.

## Wide prompt ingestion

Prompt ingestion runs the wide schedule over the deployed FFN (mode 20) **by default**, for `serve`,
chat and `-p` alike; [the served prompt route](docs/serve-prefill-route.md) records what that is
worth and what keeps the old path. `--prefill-ffn off` pins the eight-row route that every historic
deployed-ingestion number was measured on, and `--batch`, `--v1`, `--mtp`, `--ffn`, `--dump` and
`--prefill-sweep` keep it because they cannot take the wide one. Generation is untouched either way.

Every wide-batch result here was measured through `tools/batch_compare`; the engine's own chat,
`--bench` and `serve` paths used to feed prompts to the persistent kernel eight rows at a time.
`--prefill-ffn` gives them the wide schedule for prompt ingestion only, leaving generation on the
code it always ran.

[Wide prompt ingestion](docs/wide-prefill.md) also separates the wide route's gain into its two
parts, which nothing had done: the wide **schedule** is worth 1.70x at the deployed FFN's own
arithmetic (156.4 to 265.4 tok/s at 128 rows), and the A4 FFN **map** is worth a further 2.74x on
top of that. The new `wide-deployed` route (`--ffn wide-deployed`, batch mode 20) is the first one
that takes the schedule without the approximate FFN: against the deployed path it reproduces every
token of a 32-stream, four-step continuation and every top-1 choice of 192 teacher-forced rows, at
mean KL 0.0002-0.0006 against mode 19's 0.0704.

On the product path a 2663-token prompt ingests at 249.6 tok/s against 146.6, or 566.5 with
`--prefill-ffn wide-commit-a4`, with generation a measured null in both.

[DFlash2 now rides that route](docs/wide-drafter.md), which is what the machine's own server runs.

The row count at which a pass takes that route is [a runtime setting](docs/serve-wide-decode.md)
rather than a literal 32 (`HALO_WIDE_MIN`, `--wide-min` in `tools/batch_profile`). The band under
the shipped floor is measured on both shapes: the wide route loses 1.5% at eight rows, wins from
nine — the first row count at which the sliced route needs a second pass over the whole weight
stream — and is worth -25.3% on a 32-row generation step and -21 to -34% on a 9..31-row prompt
pass. The default does not move, because crossing the floor gives a pass the wide route's map: it
matches the deployed route's greedy continuations on 27 of 32 streams under the current four-bit
activation default and 31 of 32 with `HALO_SEQ_QUANT=a8`, which costs nothing measurable at a
generation width.

[A pass now carries 256 rows](docs/pass-width.md) rather than 128, and prompt ingestion defaults to
that width. Four guards held the old cap, two of them bounding rows by the count of independent
sequence *slots*. It is bit-identical — 95,354,880 full-vocabulary logits over every token of a
document, same FNV at both widths — and worth 9.1 ms per doubled pass, 1.7% of a 384-token document
against a 5.0% asymptote. The measurement's more useful half is a correction: the FFN has **no**
per-pass fixed cost at all (0.6052 ms/row at 128 rows, 0.6071 at 256, re-streaming its whole 4.28 GB
image in both), so the widely quoted "11.5 ms fixed plus 0.669 ms per row" was an artifact of
fitting a line across the 32-row/128-row boundary where the kernel changes weight image. The wider
pass also makes `--decode-tokens 8` a legal shape: a 32-stream generation step of 256 rows runs at
**684.9 aggregate tok/s** at full acceptance against 555.9 at K=4.

[The FFN slice inside that route now takes sixteen rows](docs/ffn-slice-width.md) instead of eight.
`mvw_rows` feeds a sixteen-column `iu8` WMMA and picked its activation row with `col & 7`, so an
eight-row slice issued exactly the matrix instructions a sixteen-row slice issues and discarded half
the result; a 128-row pass therefore visited the whole FFN weight stream sixteen times. The `ffn`
phase of that pass falls from 412.6 to 218.1 ms and the pass from 522.2 to 326.8, with the six
phases the change cannot reach inside 1.1%. With the drafter loaded, a 1289-token prompt ingests at
**413.0 tok/s against 249.4**, and against the 151.9 tok/s of the eight-row deployed route in the
same process. It is bit-identical: same weights, same K order, same accumulation, same residual
FNV-64, same greedy token stream. `HALO_FFN_SLICE=8` selects the old width.

[That slice now takes thirty-two rows in two column groups](docs/deployed-matvec-groups.md). A block
of weights is loaded, peeled out of radix-3 and half-swapped once whatever the row count, so a
second sixteen-column group costs matrix instructions and accumulators and nothing else: 25% fewer
non-matrix instructions per row and half the weight stream. The `ffn` phase of a 128-row
wide-deployed pass falls 216.3 to 173.2 ms, that route's prompt processing goes 407.2 to 458.5
tok/s, and the engine's own drafted ingestion of a 1610-token prompt goes 377.4 to 421.0 tok/s with
the deployed route in the same process flat. Bit-identical over 381,419,520 compared logit values.
`HALO_FFN_SLICE=16` selects one group.

[The trit was already a byte of the radix-3 product](docs/mv-peel-gather.md), so the deployed
matvec stopped shifting it out and back. One `v_perm_b32` gathers four trits from two products:
168 operations per 128-trit block instead of 232, the block loop 561 to 476 instructions at the same
238 registers and the same grid, the `ffn` phase of a 128-row wide-deployed pass -1.66% against its
in-panel control and full-model prompt ingestion +1.40% median over seven paired rounds. Same image,
same bytes, same operand dwords, same residual hash; `kernels/head_op_check` proves the map against
`halo::decode_block` on the CPU.

[Straightening that slice's activation operand loses](docs/ffn-slice-operand.md), which closes the
largest open question that document left. Giving the quantiser the coordinate the matrix instruction
reads takes a 128-K block from 320 cache-line requests to 36, exactly as predicted, and costs 3.6 to
4.2% of the `ffn` phase: the per-lane address arithmetic the scattered operand needed was separating
the radix-3 peel's dependent pairs. Bit-identical over 381,419,520 logit values, and not in the tree.

[Each width of that slice now launches on its own grid](docs/ffn-slice-grid.md). The launcher took
one cooperative grid for the whole kernel ladder — the minimum, set by the two-group body's 238
registers — so the narrow widths ran on three workgroups per WGP when their own 73 registers allow
six. A pass of 5..31 rows takes those widths, which is `--batch 8` and every small batched decode
step: the eight-row slice phase falls 512.1 to 419.5 ms (-21.4% against the untouched phases) and
eight-stream aggregate generation goes 132.3 to 137.1 and 131.4 to 140.7 tok/s across two builds.
The 32-row width is a null by construction and measured so, so prompt ingestion does not move. Same
weights, same K order, same FP32 chains, identical `residual_fnv64` and identical tokens. The
document also records the negative beside it: asking the allocator for a fourth workgroup per WGP
costs 8-16% more instructions in the block loop and loses at both widths that can reach it.

[Where a drafted generation step goes](docs/serving-decode.md) measures the other half of serving,
which this does not move: 62.5 ms per step, 82.7% of it one eight-row verify pass, and why a longer
draft chain or a two-chain tree loses at the acceptance rates this drafter reaches.

[The drafted step's route, its rows and its context](docs/drafted-verify-route.md) prices the shape
the service generates on: the batch route loses 4-6% at one sequence x eight tokens, a verify pass
is 31 ms of fixed weight stream plus 12.2 ms over seven rows so seven drafts is optimal and
adaptive width is worth +0.6% against a +16.4% oracle - and drafted generation falls 68.4 to 24.3
tok/s by 11k tokens of context, where 60% of the drafter is attention over keys its own 2048-key
window excludes.

[That window decides the drafter's unit list now](docs/drafter-window-units.md), not just its mask.
`ph_attn` dealt one score unit per key chunk from position 0 and each unit read its keys, scored
them and multiplied the chunk by zero — 88 chunks a head to keep 17 at an 11k context. Both the
producer and the fold read one expression, `attn_chunk0(pos, window)`, and a window of 0 returns
chunk 0, so every target-model pass keeps its schedule, its registers and its bits. The drafter goes
**13.43 to 8.45 ms a step at 7371 tokens and 17.90 to 8.76 at 11248**, which is **+7.6% and +12.2%
of drafted generation**, with the verify pass and prompt ingestion inside 0.7% in the same
processes and the greedy digest identical in every cell. Below the window it is a measured null.
The exactness is a host probe (`make kernels/attn_window_check`) rather than a device diff, because
the drafter's own float state is not reproducible run to run: four of its matvecs K-split and
`ph_matvec` drains those with `atomicAdd`, so the same arm differs from itself by 7.8e-3 in the
block hidden state and **an accepted count is not an identity witness**.

[Inside the drafter](docs/drafted-step.md) three weight coordinates stream in one launch — the
drafter's own Q4 or Q8 tiles at 200-220 GB/s and the target's ternary HALO head at 137 — which is the
only control in this engine that prices a weight format against another under one clock and one grid.
[Reading fewer vocabulary columns](docs/drafter-head-columns.md) in that head is free of the output
by construction and costs only an accepted count: it is worth about +3% on English and code down to
a quarter of the vocabulary and **-15.8% on five non-English prompts**, so it is measured, published
and not shipped. The same panel found that DFlash2 accepts 10.2% of its proposals outside English
and code against 38.0% inside, for 34.9 tok/s against 72.0.
[The verify pass beside it](docs/drafted-serve-step.md) is traced phase by phase: seven matvec phases
run one body at 113 to 203 GB/s, the spread is how many units each phase deals, and the retiling that
fact asks for is built and measured as a 31% loss because the unit boundary costs more than the
longer phase wins.

[The packed state was a block short of grid](docs/rows-grid-occupancy.md), and the compiler's
occupancy table could not see it. The cooperative grid of a wide pass comes from the eight-row
`k_forward_rows`, and the int8/int16 state kernels need twenty registers more than their fp32 twins
for the codec: 140 against 135, which this runtime calls **four** blocks per WGP against five. So
the packed coordinates ran the deployed route on 80 workgroups against fp32's 100 from the day they
landed. Asking the allocator for a 120-register budget on that one instantiation gives the block
back: deployed prompt ingestion **140.7 to 172.9 tok/s**, drafted generation **37.34 to 44.80**,
per-step verify **57.5 to 46.75 ms**, same greedy digest and same accepted counts. `tools/rows_occ_probe`
reports what the device will really co-schedule, and LDS at 9508 bytes a block caps this kernel at
120 workgroups whatever its registers.

[The block above that one is reachable and it loses](docs/pk-register-peak.md). Every register the
eight-row kernel spills at the sixteen-wave budget is the attention phase's - 149 VGPRs with that
phase and 107 without - and grouping the score unit's rows hands back 21 of them bit-identically.
The sixth block then costs **18.6%** of the verify pass, the loss is the grid rather than the spills,
and a grid ladder taken in one process reads **54.665 / 46.288 / 43.458 / 45.481 / 51.987 ms** at 80 /
90 / 100 / 110 / 120 workgroups: the engine already runs the maximum. On this kernel a register buys
nothing until it buys a block, so the phases spending registers should spend them.

[That cap is gone and the curve above it is the wrong way up](docs/rows-lds-occupancy.md). The
forward kernel's shared block held a kilobyte the matvec drain never writes; sizing it to its own
phases takes it to 7204 bytes, and with `waves_per_eu(13)` the eight-row kernel reaches 96 registers
and a base grid of **160**. Every point above 100 loses: **80 +24.0%, 100 the deployed point, 120
+6.1%, 140 +5.8%, 160 +14.9%** on the verify pass, bit-identical at every grid. The barrier is not
the reason (862.9 ns at grid 64 to 1759.6 at 160 is 0.30 ms of the 6.7 ms), and the register ladder
has no reachable point between 120 and 143, so **the deployed configuration is the best one this
kernel can address on either axis.** Do not spend registers here to buy workgroups.

**fp32 kept the allocator's choice, and then the body moved under that rule.** The attention split
above left the fp32 eight-row shape at 141 registers where it had been 135, which is four blocks per
WGP instead of five, so the default state coordinate spent an hour and a half serving on 80
workgroups. The same 120-register ask now buys fp32 the block too: **deployed prompt ingestion 143.5
to 172.2 tok/s (+20.0%)**, drafted generation 42.24 to 49.07 (+16.2%), per-step verify 57.81 to
48.57 ms, with the greedy digest and the exact drafted/accepted integers equal in every cell and the
drafter's own phase flat as the in-process control. Both coordinates now take the wide budget, and
the rule records a register count rather than a state format. The sixth block is reachable through
`waves_per_eu(13)` at 96 registers and **loses** — 31 spilled registers against 7, 4.9% behind grid
100 — so the largest grid this kernel can hold is not the fastest one. `--pk-occ 0,1` walks both
budgets in one process for either coordinate, which for fp32 it could not do before.

[One wave was draining that pass while seven waited](docs/matvec-drain.md). A `ph_matvec` unit ends
with eight waves publishing partial sums through LDS and wave 0 summing all of them and writing the
output row: fifty-six `ds_read` and eight writes in one wave at eight rows, once per unit, about
150,000 units per pass. Row `r` belongs to wave `r` now, and the block's seven slots still hold
eight partials, because a row's owner reads its own from a register and wave 0 writes into the slot
that frees. The sum stays `y_0[r] + y_1[r] + ... + y_7[r]` left to right, so every output bit is
unchanged, and the LDS block does not grow. The drafted verify pass falls **-3.9% (mean over three
ABBA blocks, all negative)** with the same greedy digest and the same drafted and accepted counts;
an earlier form of the same change measured -4.5% median paired on the base before it.
`bench/coop_cost --drain` prices the wave-0 drain at 4.4% of a 1088-unit phase and 8.9% of a
320-unit K-split one. The same document records the four things the engine-against-probe gap is
*not* - where the weights are allocated, a tiny phase between two streams, an extra phase body in the
kernel, the K-split stream stride - and that the probe drifts 10% between panels while its arms agree
to 0.6% inside one.

[That verify pass takes the wrong matvec body](docs/matvec-crossover.md), and had done since the
WMMA body was written: `MV_DOT4_MAX` was set to 4 then and never set again by anyone, so five to
eight rows have always taken WMMA without a measurement. Ternary tiles want the dot4 body at eight
rows and Q8 drafter tiles want WMMA, so the row limit now comes from the weight representation.
The drafted verify pass falls from 51.3 to 49.5 ms and drafted generation rises from 51.1 to
52.4 tok/s, with all 248,320 logit floats, the greedy digest and the 133 drafted / 42 accepted
unchanged. The reason is not the instruction count, which says the gain should be four times
larger: dot4 holds `k_forward_rows<8>` in 136 VGPRs instead of 217, which takes the deployed
cooperative grid from 60 workgroups to 100, and at a fixed grid the dot4 body is the slower of the
two. That grid was one number serving two shapes that want opposite things — 21% at eight rows,
∓0.4% the other way at one — so it is now chosen by row count.

[Asked at the same grid, the matrix body loses anyway](docs/mv-wmma-eight.md), by 15.6%. The
"at a fixed grid the dot4 body is slower" above was measured at grid 60, which neither body chooses;
a matrix body built for the deployed register budget reaches 193 VGPR unconstrained and 144 with 27
bytes of scratch, so both arms launch grid 100 and the comparison is finally body against body.
Verify 46.4 ms against 40.4, drafted generation −11.4%, bit-identical in every arm. At eight rows a
`16x16x16` matrix instruction carries eight duplicate columns, so its 34.3 cycles buy what 256
`v_dot4` buy in 256 — and the same 549 cycles would serve sixteen rows, which is the shape that
flips it and the number to put against widening the verify pass.

[Inside that body a row's dot products were one dependent chain](docs/mv-dot4-body.md), and the
compiler was fencing every pair of them: each row's 32 `v_dot4` accumulate into one register, so an
eight-row block carried 128 `s_delay_alu VALU_DEP_1` hints and 174 scheduling slots against 751 of
work. Four independent `int32` chains per row remove 90 of them for seven work slots and no
register. The verify pass falls from 52.6 to 48.65 ms, drafted generation rises from 66.6 to
71.0 tok/s and deployed prompt ingestion from 160.5 to 171.6, with 248,320 logit floats byte for
byte equal and the same greedy digest and acceptance counts. Integer addition over this range is
exact and associative, so the chains are a schedule and not a numerical choice. Single-token decode
keeps one chain and is untouched code: it runs its matvecs at 240 GB/s of a 242 GB/s roof and has
no slack for a chain to take.

[The rest of that body's expansion can be deleted, and it is worth 1%](docs/mv-dot8-nibble.md). A
five-trit byte peels two trits at a time, and a nine-valued two-trit code is a whole operand *byte*
when the operand is nibbles, so one `v_perm_b32` builds eight ternary nibbles for
`v_dot8_i32_iu4` — where IU8 bytes need two lookups and two interleaves. The HALO element order
already groups the four codes of a byte pair into one eight-element operand, eight-bit activations
split exactly into `16*a_hi + a_lo` for two dot8 at the same MAC rate, and the relabel `v = 2 - t`
puts code 8 on `v_perm`'s ninth selector for free. The expansion falls from 254 to 156 issue slots
and the kernel by 10.5%, bit-identically — and the eight-row pass moves 1%. That pass reads 5.9 GB
in 46.35 ms, **127 GB/s of a 242 GB/s roof at 74% of its own issue model**, so what binds it is
memory latency at ten waves per SIMD32 rather than instructions: price a census change to this body
at about a tenth, not at the third the FFN converts at. The map and its host acceptance
(`make kernels/mv_nibble_check`) stay in the tree; the runtime keeps the byte operand, because the
nibble activation pane costs in prep about what the matvec gains.

[The perm gather reaches this body too, and reads null](docs/rows-lds-occupancy.md#the-second-arm-the-perm-gather-in-the-deployed-matvec-body-null):
168 operations per 128-trit block against 232, the same operand dwords, the same registers, verify
42.628 against 42.617 ms. Four instruction cuts on this loop have now measured nothing.

[What binds that body is not memory latency either, it is the wait](docs/mv-scalar-waits.md), and the
ISA says so in one number: every scalar wait on gfx11 is `s_waitcnt lgkmcnt(0)`, SMEM returns out of
order so the counter cannot be waited part way, and an eight-row block stops **sixteen times** per
896 weight bytes where a one-row block stops once. The deployed loop's own prefetch is what drains
it — it asks for the next row's activations and then waits for the current row three instructions
later. Requesting two rows and their drain scalars behind one wait, so each wait lands after the
previous pair's 64 dot products, is bit-identical and takes the drafted verify pass from 47.78 to
45.53 ms and drafted generation from 47.22 to 49.66 tok/s, with the greedy digest and the 252
drafted / 60 accepted equal in every cell. Two rows is the knee: one row is the same wait count with
nothing to cover it and measures null, four rows needs 128 SGPRs against a 107-SGPR file and loses
12%. The faster arm is the one that cannot ship — activations on the in-order `vmcnt` counter is
+8.0% in isolation and spills 370 registers inside `k_forward_rows<8>` — and a deeper weight cursor
is a measured negative, which refutes the in-flight-bytes model that predicted it.

[The drafter's own weights are four-bit now](docs/drafter-q4.md), which is the one approximation in
this engine that is free of the output contract: the target verifies every drafted token, so the
greedy stream is identical whatever the drafter computes and only the accepted count can move. Its
image falls from 1.825 to 0.927 GB and the `draft` phase from 11.35 to 7.65 ms, for **+10.4% on
drafted generation** (59.34 to 65.50 tok/s median over ten workloads, no case slower) and +2.0% on
drafted prompt ingestion, with the greedy digest equal between arms in every case. Acceptance did
not pay for it: pooled 38.98% to 40.66% over 3840 generated tokens per arm. Packed nibbles are the
operand form — `P & 0x0f0f0f0f` and `(P >> 4) & 0x0f0f0f0f` are the two words the matvec bodies
want, and the offset-binary bias comes out of the activation sum the ternary path already carries.

The features the drafter ingests are the raw residual stream entering five layers, so a wide pass
captures them with a strided copy of its shared residual buffer, and `Engine::prefill` hands the
drafter eight rows of it at a time. A **drafted** 2663-token prompt ingests in 10.51 s instead of
18.39 s, **144.8 to 253.3 tok/s**, with the same generated tokens, the same 70 drafts and the same
7 accepted; generation is a measured null. `--prefill-sweep off,wide-deployed[,wide-commit-a4]`
walks the routes inside one process and prints a digest of each arm's token stream. The MTP head
still takes the eight-row route, because nothing writes its per-row final hidden on a wide pass.

## Wide vocabulary head

[The output projection](docs/wide-head.md) now runs once over every row of a
batched pass instead of once per eight-row slice, which takes its cost at 128
rows from 26.6 to 11.2 ms. Prompt processing reaches 614.7 tok/s with A4 FFNs
and 497.7 with A8, up from 600.3 and 485.5; 32-stream generation reaches 260.0
and 215.0, up from 253.4 and 210.4. It reads the same packed HALO tiles as the
eight-row head and keeps its reduction order, so all 71,516,160 compared logit
bits agree at 32, 40, 88 and 128 rows. `HALO_WIDE_HEAD=0` selects the sliced
control.

[It now produces the rows a caller reads](docs/head-rows.md). Prompt ingestion
takes one logit row out of a pass and the head was computing every row of it, so
the phase falls from 10.598 to 1.534 ms at 128 rows and from 21.147 to 1.535 at
256 - one pass over the 278 MB output image whatever the pass width, where it
used to be the one term that grew with it. Prompt throughput rises 1.95% on a
384-token document, arms interleaved in one process, and the 248,320 logits of
the row ingestion reads are byte-for-byte identical. A generation step still asks
for every row, because there every row is a different sequence's next token.
`--head-rows all` is the control.

That kernel is issue-bound, not memory-bound: it reads its 278 MB weight image at
24 GB/s, and a quarter of its non-matrix issue slots were rebuilding a fragment address
that advances by a constant. Walking the operand instead, and seeding the accumulator
with the zero point the epilogue used to subtract, puts `head-projection` at 11.6 ms for
128 rows and 2.8 for 32, bit-identically. The document carries the instruction census
and prices the peel replacement that loses.

## What binds the sequence projections

`sequence-output-projection` is 21.8 ms of a 183 ms 128-row pass and 7.9 ms of a 32-row one against
a 13.8 and 4.6 ms serial issue model, and it is the least clock-proportional phase in the engine.
[Counters say why](docs/sequence-output-projection.md): the memory unit is 95.9% busy in the output
stage and 97.8% in the input one, level with `resident_state`, which is known to run at 224 of
242 GB/s, while the FFN's two stages read 78.9 and 87.9% on the same instrument.

That prices the four-bit-activation proposal these projections carry, and the price is not the one
the operand document quotes. `iu8` takes a 16-byte activation operand per lane in 32 cycles and
`iu4` takes 8 bytes in 16 — **the same half byte per lane per cycle**, so four-bit activations halve
the matrix cycles without asking the operand path for anything less. A kernel held by its matrix
pipe gets 2x; a kernel held by its operand path gets nothing. What `k_proj_opt` does and this kernel
does not is carry *two weight row tiles against one activation fragment*, which is the term with
room on it.

The same document records two bit-identical changes that lose, with their patches and panels: a
line epilogue that cuts epilogue memory accesses fourfold and measures a null, and the block-weight
cursor that paid 28% on the FFN's down stage and costs 8 to 28% here, because a 128-block of this
kernel is 32 fragment loads against 2 weight loads.

## The order the weight image is stored in

The answer to some of that clock-independent time was not in the kernel at all.
[The weight image's run order](docs/weight-stream-order.md) decides where the hundreds of
concurrently resident waves are reading *at the same instant*. Tile-major, which is how every weight
image here was stored, gives each wave its own stream and puts those streams `nb * 512` bytes apart:
20,480 for the input matrices and 24,576 for the output ones. Block-major puts every resident wave's
current block inside one contiguous run. Same bytes, same values, same order of use, and the block
loop's instruction census does not move by a slot.

It is worth **+6.4% aggregate generation at 32 streams and +2 to +5% prompt** on the full model,
-27% on `sequence-input-projection` at the 32-row shape and -22% on `sequence-output-projection` at
128 rows, bit-identical at both widths. `bench/wstream` prices the mechanism with no model in the
picture: the same bytes read as hundreds of concurrent 512-byte streams deliver 113-143 GB/s when
the stride carries a large power of two and 194-218 GB/s read contiguously. The FFN, head and
drafter images have the same shape and the same kind of stride, and that probe costs seconds.
`HALO_SEQUENCE_IMAGE=tile` selects the old order; `--seq-image 0,1` walks both in one process.

**Why it works is now measured, and the cheap fix is one run of padding.**
[The FFN's image order](docs/ffn-image-order.md) added a third arm to that probe: tile-major with
one extra run of stride, which keeps each wave's own blocks contiguous and changes only the
alignment. It matches block-major on every aliased geometry — 202.2 against 202.3 GB/s on the FFN's
gate/up stream, where tile-major reads 148.3 — so the term is **bank and channel aliasing between
the streams, not where the waves are relative to each other**. Padding a stride that does *not*
alias makes it worse, so the FFN stores its runs under a rule: pad a matrix whose stream stride is a
multiple of 4096, leave the others tile-major. That is `--ffn-run 3`, the default, bit-identical,
worth 12-17% of the phase at the shape that waits on the stream and 1.4-2.1% of the deployed 128-row
pass. `kernels/head_batch.hip` and the drafter still store theirs unpadded.

## The activation scale's axis

All three integer matrix kernels store the per-(token, K block) activation scale with the token
contiguous and read it with the token on the *lane* axis, so each drain load's sixteen lanes land in
sixteen different cache lines. A 128-row A4 pass asks for 534 M cache lines where the transposed
layout asks for 17 M. [Transposing it is bit-exact and measures a null](docs/activation-scale-axis.md),
at both 32 and 128 rows; it reached master only as the enabling half of the operand cursors, which
need the wave-uniform base it makes possible.

A second null in the same document prices the census everyone here runs. Two bit-identical changes
moved a selected block loop's instruction total by -6.2% and +2.0% while moving its **work** total
by four slots and zero, and both measured null. `s_delay_alu` and `s_waitcnt` are not work;
`tools/isa_loop_count.py` now separates them.

## Wide attention

[The attention layers](docs/wide-attention.md) were the last phase of a wide pass still launched
once per eight-row slice: 256 cooperative launches per 128-row pass, each scanning that layer's
whole KV cache for eight rows. They now run once for the batch, over row groups that are exactly
what a slice handed the phase, so the chunk boundaries and reduction order are unchanged and all
71,516,160 compared logit bits agree. `sequence-core` falls from 17.0 to 4.7 ms at 128 rows and
from 5.4 to 3.2 ms on a 32-stream generation step; prompt processing reaches 835.5 tok/s against
792.2 with a mode 0 control inside 0.4%. `HALO_WIDE_ATTN=0` selects the per-slice phase.

## Prompt length

[Attention is the only phase priced by position](docs/long-context-attention.md), and every other
prefill number here is a 384-token document. It costs 0.0166 ms per token of position per 128-row
pass while everything else stays flat to 1%, so it is 1.7% of a 384-token prefill, 9.4% of a
2048-token one at mode 19, and passes the whole A4 FFN inside a single pass at position 4500.
`tools/batch_profile --prefill-scan N` walks that axis. The measurement found a 256-row pass
running with no wide attention at all — `create_attn_batch` capped at 128 while the default pass
became 256 — and fixing it is 26.6 to 6.6 ms of attention at position 0 and +4.0% on a 2048-token
document, bit-identical. It also rules out sixteen-row groups, which cost 18% of the score units.

[The other half of that phase is a working set, not a buffer](docs/attn-partial-window.md). A score
unit writes 1040 bytes of chunk partial per `(row, head, chunk)` and the combine reads every one
back, so a 128-row pass at position 3072 turns 3.1 MB of attention output into 79.9 MB of partials
per layer and sends all of it to DRAM through a 32 MB cache. The driver already walks row groups in
score/combine pairs; capping what a pair may leave in flight at 16 MB makes the same bytes resident
and runs the combine at 365 GB/s instead of 154 — **-57.6% of the combine, -5.07% of attention and
879.4 → 886.0 tok/s on a 3072-token document**, with every one of 95,354,880 logits reproducing
`16232333786579957879`. It is inert by arithmetic on any pass whose partials already fit, which is
every short prompt and every decode shape. `HALO_ATTN_PART_WINDOW_MB=0` restores the one-pair
geometry.

[The engine at the context it advertises](docs/long-context-serve.md) carries that to 8192 on the
served 256-row width, where the score phase is 22% of a whole document and 36% of its last pass, and
measures the shape the curve above never covered: a **generation** step costs 2.204 ms per 1000
tokens of context — 29.7 GB/s on a 65536-byte-per-token K and V stream, 12% of the machine's roof —
so single-stream generation falls from 32.1 tok/s at 1024 to 21.3 at 8192. Reading each key chunk
six times instead of once costs 0.8%, so that term is latency, not traffic.
`tools/batch_profile --modes 0` measures the persistent route in-process.

## Wide sequence input prep

[The layer input prep](docs/wide-prep.md) was the last part of a wide pass still
running eight rows at a time: 1024 cooperative launches per 128-row pass, each
recomputing a row's RMS norm once per chunk, with every alpha/beta projection row
re-reading every activation row of its slice. It now runs in two ordinary
launches per layer, taking 19.9 ms to 3.6 ms at 128 rows. A4 prompt processing
reaches 661.5 tok/s at 128 rows per pass and 467.5 at 32, up from 604.7 and
441.4; 32-stream generation reaches 268.7, up from 260.6. The units and their
accumulation chains are the deployed ones, so all 71,516,160 compared logit bits
agree. `HALO_WIDE_PREP=0` selects the per-slice control.

## One kernel carries one score body

[The score unit's coordinate is an address, not a body](docs/attn-body-dispatch.md). Reaching the
lane-per-key arm through a runtime branch made every caller allocate the union of two full score
bodies, and `ph_attn` is inlined into the persistent kernel, so the deployed route had been running
on **grid 60 instead of 100** since that arm landed. The dispatch is a launcher decision now and the
row-major body places its key through `k_pos_off`, so one body stays correct in either coordinate.
`k_attn_wide<8>` goes 130 VGPR / 10 waves to **92 / 16**, the attention phase of a 128-row pass
**-18.0% at position 1024 and -21.7% at 3072** in the default coordinate, device span -4.0%, and
the lane-per-key arm is worth -6.1% again on top. Every arm is bit-identical.

## A generation step's attention

[The unit was written for eight rows and a generation step has one](docs/attn-row-groups.md).
`attn_chunk_unit` clamps its row index with `min(i, nrows - 1)`, so a one-row group computed row 0
eight times and discarded seven; every decode shape in this engine is one-row groups. It now
dispatches on the group's own row count, and it folds a KV group's six query heads into one unit,
which is the only key-chunk reuse a step has once its rows are different sequences. At 32 streams
and a 960-token prefix that is 37.4 to 12.6 ms of attention and **224.6 to 271.1 aggregate tokens
per second, +20.7%**, with one residual hash across every arm; prefill is a null by construction.

The same panel is the first measurement of a decode step's position axis, and it moves the floor
under the map below: attention goes 4.4 to 48.4 ms between a 64-token and a 960-token prefix while
five other phases stay flat to 0.4%, so it becomes the largest phase in a step at about a thousand
tokens of context. `--decode-prompt` is a list and `--decode-context` raises the KV allocation.

## Where a generation step goes

[The decode map](docs/decode-map.md) profiles the aggregate generation shape, 32
sequences of one token each, which no earlier profile had measured. **At its 64-token prefix** the
recurrent state, not the FFN, is the largest phase: 48.1 ms of a 125.9 ms step.
Holding the row count at 32 and moving from one sequence to 32 multiplies that
phase by 7.53 while nothing else moves by more than half, and the marginal
9.36 GB of state moves at 224 GB/s against the 242 GB/s `bench/bw` measures. No
schedule of the current FP32 recurrence can go faster there; the levers are more
tokens per step or fewer state bytes, both priced in the document.

[The same step by kernel instantiation](docs/decode-dispatch.md) separates the two FFN
stages the phase bracket merges, and finds the <=32-row arm every generation step takes
is issue-bound rather than byte-bound: its weights move at 109 GB/s of a 242 GB/s roof
while 321 of a block's 666 instructions lift five trits out of a byte. Only 2.1% of the
step is the gap between dispatches. Holding two blocks of that weight stream in flight,
and reading a group of residuals before storing any of it, takes the down stage from
13.7 to 9.8 ms and a 32-row prefill FFN down 14.4% against the untouched phases of its
own run, with every residual bit unchanged.

[The second lever is now built](docs/gdn-state-pack.md). The state's storage
coordinate is a choice, and holding it as int16 with one fp32 scale per state row
takes the recurrent phase of a 32-stream step from 44.4 to 24.2 ms and **aggregate
generation from 286 to 349 tok/s, +21.9%**, with prefill a measured null. The
recurrence itself is untouched: every coordinate decodes into the same registers
in the same lane order, so the summation tree does not move and a control that
stores fp32 through the packed kernels reproduces the fp32 residual exactly. A
shared exponent per row beats fp16 here because both consumers of a state row are
dot products against normalised vectors, where the error that matters is
absolute. fp32 stays the default; `HALO_GDN_STATE=i16` selects the packed route,
whose teacher-forced cost is +0.0024 nats against the +0.0615 of the A4 FFN map
this engine already ships.

[That packed state's *memory* coordinate is a second free parameter](docs/gdn-state-coord.md), and
it was worth **−9.4% of the recurrent phase of a 32-stream generation step** with every output bit
unchanged. Storing the four rows one wave owns in one contiguous 512-byte run turns four 128-byte
loads 1 kB apart into one `global_load_b128`, and taking the row scale's store out from under
`lane == 0` gives the store loop back its basic block. Both are inside
`kernels/gdn_state_codec.hpp`, `HALO_GDN_STATE=i8` gets them, and the measurement says what binds
that phase: not bytes at 53% of the roof, not issue slots (removing 19% of the unit's slots is an
exact null there), but the number of memory instructions a wave issues.

[int8 is the coordinate to use, and the precision question is
closed](docs/gdn-state-horizon.md). A packed state is re-rounded once per token per
layer in a generation step and once per *pass* in prefill, so the quality window that
held int8 back was measuring a few dozen roundings. Running 256 teacher-forced
generation steps per stream, paired per (stream, step) against a control that
reproduces its reference exactly, int8 costs `-0.0165 +- 0.0082` nats and int16
`-0.0042 +- 0.0077`, and neither drifts across the horizon. int8 is **+37.3%
aggregate generation** against fp32's 300 tok/s, 12.6 points ahead of int16. The
per-prediction spread is 0.282 nats for int16 and 0.279 for int8 for a 257-fold
difference in rounding, so a perturbed trajectory decorrelates on contact and then
stops caring how hard it was hit. fp32 remains the process default because
`r_gdn_fmt` moves every route at once and three consumers need the exact one.

[The write side is optional too](docs/gdn-state-defer.md). A step reads the state
because both consumers of a row contract all of it, and writes it back only because
the next step wants to find it somewhere — while what a token actually does to a head
is scale it and add one rank-1 term. Keeping those terms in the free upper half of the
same region and rebuilding the sum at load lets the commit happen once per four steps:
**the four steps of a cycle cost 74.1, 75.2, 77.0 and 91.9 ms against 88.3 committing
every one**, and full-model aggregate generation rises 7.4% on top of the coordinate's
own gain, with the packed state rounded four times less often rather than more. Prefill
is a null by construction and measures as one. `HALO_GDN_DEFER=4` selects it,
`--gdn-defer 0` runs the same kernels while committing every step.

[The exact coordinate defers too, and there it changes no bit](docs/gdn-defer-exact.md).
fp32 was excluded because its values fill its region, which is a reservation rather than a
numerical fact; given the room, and a rebuild that replays the pending triples in the order
`gdn_token` applied them, the state a step reads is the one the eager path would have written
element for element. **`gdn-resident-core` 45.0 → 26.4 ms, a 32-stream step 98.1 → 73.9, and
aggregate generation 323 → 428 tok/s (+32%) with the residual FNV-64 unmoved in every sample.**
A pending term has to keep *both* of `warp_sum`'s roundings to manage that, because the recurrence
applies each to the columns its own lanes own. The default is two steps per write-back at an exact
coordinate and one at a packed one, whose approximate map nobody asked to change.

Depth four is the answer and [contracting the pending terms does not change
that](docs/gdn-append-gram.md). An append step needs the rebuilt row only for two dot
products and both distribute over the terms, which takes a term from 1.14 ms to 0.69 —
but the *commit* step has to materialise the state to store it, so its rank-`P` update
grows linearly in the depth while the write it amortises shrinks as `1/C`, and those
cross at four to six terms however cheaply an append runs. That arm reassociates at
every depth including zero and is not in the tree. What is: the rebuild reads a term's
delta column from LDS instead of the state region, **bit-identically**, which is worth
2.0% of the commit step.

## Wide-batch FFN execution

[Row-tile ownership](docs/ffn-schedule.md) sets how many waves of a workgroup share one
weight row tile. The down projection owns 160 row tiles against gate/up's 1088, and
wants the wave slots more than the accumulator width: giving it four waves at two token
tiles instead of two at four takes the A4 FFN from 98.1 to 84.9 ms at 128 rows and
raises full-model prompt processing from 662.6 to 702.3 tokens/s. Scaled A8 gains 1.028x
on the same stage. Both reproduce their prior residual bits exactly.

[Wave slots are not what that kernel is short of](docs/ffn-occupancy.md) records the
experiment that rules occupancy out: freeing 19 VGPRs of operand addressing takes the gate/up
stage from five resident waves per SIMD32 to six, bit-identically, and the stage does not get
faster.

[The two projections want different weight images](docs/ffn-decode-shape.md). A4 holds a
dense five-trit image at 1.625 bits per weight and a pair-code image at 2.000, and mode 8
chose between them by row count for the whole FFN. What decides an arm is how many times a
stage walks its weight stream, and the two stages answer differently: at 64 rows the dense
map's two-token-tile cap makes gate/up read its stream twice where pair codes read it once,
while the pair map's down stage is under the wide-wave target and reads its stream four
times against the dense arm's two. Selecting per stage takes the 64-row FFN region from
62.6 to 51.9 ms and full-model prompt processing at 64 rows per pass from 502.5 to 537.4
tokens/s. 32 and 128 rows keep the arm they had and measure unchanged, 32-stream generation
is a null, and every arm produces the same residual bits.

[Load lookahead in the pair-code block](docs/ffn-pair-lookahead.md) keeps that kernel's
scheduling barrier at the four-token-tile shape it was fitted on. The down projection runs two
tiles, where a block holds sixteen fragment loads instead of thirty-two and cannot spill, and the
same barrier only forbids its lookahead: removing it there takes the A4 FFN region at 128 rows
from 85.2 to 80.3 ms and prompt processing from 719.6 to 737.2 tokens/s, bit-identically. The same
document records why the 32-row arm, which a generation step runs, keeps the dense five-trit
image: it loses 2.1x to pair codes there despite 1.74x fewer issue slots per block.

[The dense block's activation loads](docs/ffn-dense-loads.md) are what that arm actually waits
for. Its scheduling barriers were placed against the radix-3 peel, which is register pressure on
the VALU, but `sched_barrier(0)` fences every class and so pinned the block's sixteen
activation-fragment loads to the slice pair that consumes them. Letting VMEM reads cross takes the
`ffn` phase of a 32-row pass down 5.9% and of a 32-stream generation step down 3 to 4%,
bit-identically, for six registers and no wave slot. The same document prices the coordinate
question from both ends: pair codes at 37% fewer slots and 23% more bytes are 38.6% *slower* in a
generation step, and a third block in the weight cursor loses to the wave slot it costs.

[That coordinate question has since been closed by a rate](docs/ffn-decode-bytes.md), and the answer
re-prices every instruction cut made on this phase at a generation shape. Both images read their
bytes at the same speed - 107.2 GB/s for the dense 3.476 GB, 105.9 for the pair 4.278 GB, and
112.5 once the pair arm gets the block of operand lookahead it never had - so the pair arm's loss
**is** its 23% of extra bytes, and the 303-against-602 instruction gap between the two blocks buys
nothing. At 32 rows this phase runs at 46% of the memory roof and 56% of what `bench/wstream` gives
its own image geometry with no model, which is 15 ms of a 102 ms step and the largest single number
left in a batched generation step. The document names the three candidates for it and leaves the
lookahead arm out of the tree, with its patch, because no served route runs pair codes at two token
tiles.

**The rate is confirmed and one of its three candidates is now excluded.**
[The stream ceiling](docs/ffn-stream-ceiling.md) reconnected the ablation ladder to the body the
engine actually launches — it had been dispatching the activation stage first and baking
`DLS_LOADS` into it, so every `--ffn-dense` code above 1 was running the deployed schedule under
its own label — and the repaired ladder reads **weights pinned −21.9%, activations pinned +0.6%**.
The weight stream is a fifth of the phase and the operand path is closed. What the same turn
excludes is the stream's *shape*: `bench/wstream`'s occupancy ladder, its half-lane column and its
replica of the dense arm's own `lane & 15` read had never been run, and they say half-lane
addressing costs 0.97–1.01x, occupancy above six waves costs under 10%, and the deployed read at
the deployed occupancy delivers 194 GB/s where the kernel gets 107. The remaining fifth is cover
inside the block, and the block has one uncovered request left: the single-buffered activation
stage's own fetch, 55 slots of cover against 570 for everything else.

[One matrix per wave on the down projection](docs/ffn-down-waves.md) gives that stage the waves it
has always been short of. `k_proj_opt` hands every wave two weight matrices, which gate/up needs to
fuse SiLU but the down stage does not: its two matrices are the two row halves of one tensor, and
pairing them is what left the stage running 160 waves on 80 SIMD32 units. Splitting them gives 320
waves at identical weight bytes, and the stage costs 4.3 to 5.0% less at 32 rows and 3.7% at 64,
bit-identically at all three published residual hashes. It also drops the wave from 206 registers to
120, which lifts the two-token-tile cap on that stage. The document carries two negatives beside the
result: a deeper weight cursor loses 3% there even when its registers are free, and the split arm
with four token tiles does not take the 128-row down stage back from pair codes.

[Operand liveness](docs/ffn-order.md) found what does move it, from the other side of the
same block. Running two token tiles at a time against fragments expanded once, instead of one
slice at a time against every tile, takes the block loop from 508 issue slots to 489 and the
`ffn` region from 86.9 to 82.9 ms, worth about 1.6% of 128-row prompt processing, at unchanged
occupancy. It reproduces the same residual bits and all 39,731,200 compared logit bits. It also
carries the sharper half of the occupancy negative: the shape with seven resident waves and two
accumulator chains is 6.8% *slower* than five waves with eight, so what this kernel is short of
is independent matrix chains.

[Operand addressing](docs/ffn-operand-address.md) is the eighth of that block loop that was spent
reaching the operands rather than using them. Every load built a 64-bit VGPR address pair, because
the indices mixed the wave-uniform part with the lane inside one `size_t` expression; computing
wave-uniform cursors per block and indexing them with an unsigned lane offset gives the scalar-base
form instead. Gate/up goes 559 to 487 issue slots per 128-block, 256 to 229 VGPRs and five waves
per SIMD32 to six, and its 52-byte spill disappears; the FFN phase falls 5.6% against the phases
the change cannot touch, and 128-row prompt processing goes 772.6 to 801.4 tokens/s with a mode 20
control inside 0.15%. All 71,516,160 compared logit bits are unchanged. The document also records
the measured null that the same permutation is worth on its own, which retires cache-line
amplification as an explanation for this kernel's unexplained time.

## Wide sequence projections

[The IU8 operand build and row-tile ownership](docs/sequence-projection-operands.md) applies both
lessons to the second-largest phase of a prompt pass. Moving each stored two-bit code to the byte
lane its matrix operand reads turns twelve issue slots of bit spreading per K16 slice into one
mask and one shift, and a wave-uniform fragment pointer removes all 64 index multiplies of a
128-row block: 647 slots to 508, same 2.000 bits per weight, same K order. Sharing a row tile
across two waves then trades the accumulator width that costs 242 VGPRs and five waves per SIMD32
for four tiles at 153 and nine.

The rule is per stage, because the input matrix owns 1024 row tiles and the output matrix 320:
at 32 rows the input grid is already full and forcing the share onto it costs 20.6%, while the
output projection gains 4.2% from the same move. At 128 rows the input projection goes 39.6 to
38.4 ms and the output 16.1 to 16.0, with the untouched phases of the same traced pass moving
under 0.9%. All 71,516,160 compared full-vocabulary logit bits agree with the previous build.
`--seq-sched 0,1,2` orders the three ownerships inside one process.

The negative in that document is worth more than the gain: deleting 21.5% of the input
projection's block instructions moved it 0.4% at the shape it was measured on.

[Four-bit activations on the input projection](docs/sequence-input-a4.md) is what that price list
pointed at: stop deleting instructions around the matrix instruction and buy a narrower one.
`v_wmma_i32_16x16x16_iu4` retires the same 4096 MACs in half the cycles, and the stored two-bit
code word is a free coordinate — placing the code of K slot `k` at bit `4*(k&7) + 2*(k>>3)` makes
the nibble operand two masks and a three-instruction sign extension, **nine slots per K16 slice
against the eight-bit operand's eleven at the same 2.000 bits per weight**. The phase falls 40.494
to 22.010 ms at 128 rows and a 128-row pass 163.2 to 144.7, for +11.4% prompt at 128-row passes and
+11.0% at 256. Teacher-forced NLL does not move (−0.0086 ± 0.0380 nats against the eight-bit arm,
where the A4 FFN this engine already serves costs +0.0415 ± 0.0228). It is an explicit route,
`HALO_SEQ_QUANT=a4`, not a default; `a4e` runs the identical numerical map on the eight-bit
instruction, which is how the map was measured before the kernel existed and how the kernel was
then accepted — 79,462,400 logits, every bit equal. It became the default on `82dc7c7` once the
horizon panel could resolve it ([`docs/seq-a4-default.md`](docs/seq-a4-default.md)).

[The output projection now takes the same coordinate](docs/sequence-output-a4.md), and it is worth
more there than the arithmetic says. `sequence-output-projection` falls **17.769 to 10.073 ms at 128
rows, −39.9%** normalised by four phases the arm cannot reach, for **+4.2% full-model prompt
throughput at 128-row passes and +4.7% at 256**; with both stages four-bit the same document ingests
at **966 and 978 tok/s against 824 and 833**. Measured against the tile-major weight image one build
earlier the same arm was −54.4% and +7.0%, which is the ordinary arithmetic of two levers landing on
one phase. The document that read this
phase predicted a null, from a correct table: `iu8` and `iu4` both ask the operand path for 0.5
bytes per lane per cycle. **That ratio is the invariant at the matrix-pipe boundary, and a kernel
over its issue model is not at that boundary** — this one was at 158% of it, and below the boundary
halving the operand halves the wait as well as the issue. It is now at 115% of its own four-bit
model. Teacher-forced NLL against the *exact* engine goes +0.0415 ± 0.0228 (the eight-bit wide
route) to +0.0036 ± 0.0267, which is to say unresolved, with greedy agreement unchanged at 115/128.
`HALO_SEQ_QUANT` names both stages: `a4` is the input alone as published, `a4-out` the output alone,
`a4-both` the route, `a8` the exact operand on both.

**`a4-both` is the default.** [`docs/seq-out-a4-default.md`](docs/seq-out-a4-default.md) owns that
acceptance: on the horizon instrument the output stage costs +0.00458 ± 0.00901 nats against the
route's eight-bit operand and the pair costs +0.00397 ± 0.00940 — *less than either stage alone*, and
a fifth of the A4 FFN this engine already serves — so moving the default from `a4` to `a4-both` is
−0.00373 ± 0.00934. It is worth −40.05% of the phase, **+5.2% full-model prompt at 128-row passes
and +5.1% at 256**, and +0.4% aggregate generation. The four-bit kernel reproduces its own map over
95,354,880 logits (`a4e-both` hashes to the same FNV-64), and mode 0 hashes to the same value under
both arms, so the deployed sliced route that single-stream and drafted generation run is untouched.
A route with no wide writer — `mode 17`, `HALO_WIDE_PREP=0` — drops the affected stage back to eight
bits and says so once, while `HALO_SEQ_QUANT=a4-both` demands it and fails instead.

[Fragment reuse](docs/sequence-fragment-reuse.md) followed that up by giving a wave two or four
weight row tiles against one activation fragment, the way the batched FFN carries gate and up. It
halves or quarters the bytes a wave reads per output tile, costs no registers and reproduces every
output bit — and it loses, 2.3% and 15.5% of the input projection at 128 rows. [Staging the shared block in LDS](docs/seq-input-bstage.md) later cut the same traffic from the
other side — under per-wave tile ownership the four waves of a workgroup fetch byte-identical
activation bytes, and staging them once takes the block loop from 18 memory instructions to 4 at
two *fewer* issue slots — and it loses 21.6% of the input projection in a generation step, because
the barrier it needs costs seven of sixteen wave slots. Two arms from opposite directions now agree
that the fragment stream is not what this kernel waits on.
[The token map](docs/seq-operand-coord.md) closes that question from the third direction and leaves
a law behind it. Which token sits in column `col` of tile `t` is a labelling the kernel chooses, not
something the image fixes, so a wave's `TT` fragments can be fetched two at a time in half the
requests with no stored byte moving. Two such maps were built. They issue **identical
instructions** — same 23 `global_load`, same 32 `v_wmma`, same bytes — and differ by 16.6% of the
phase, because one of them deals its sixteen lanes across eight cache lines at half use where the
other covers four at full use. The one that keeps the deployed map's line floor is a dead null
(-0.35%) even though it also hands back sixteen requests and fourteen waits. **In this block loop a
fragment fetch costs one cache-line lookup per line its wave touches and nothing else**, the
deployed map is already on that floor, and `kernels/seq_tmap_check` prints the number for a
candidate map in a second with no GPU. What those panels
bought instead is a price list: an instruction added to that block loop costs one issue cycle, and
solving the serial model for the clock that fits each shape returns the measured shader clock. A
shader-clock sweep puts 83.5% of the input projection's time in clock-proportional work against
59.6% of the output projection's, which is why a scheduling change pays on one stage and not the
other.

[The weight address](docs/sequence-weight-address.md) is the third pass over the same block loop
and the first bounded negative on an axis this lane had been treating as settled. Reaching the two
code loads and the scale from a wave-uniform base instead of a 64-bit VGPR address pair is -8.3% on
the output projection at 128 rows and +1.1 to +3.0% at 32, 64 and 256 — with the same work-slot
count at four token tiles and seven fewer at one, and the same waves per SIMD32 everywhere. The two
shapes the engine actually runs, a 256-row prompt pass and a 32-row generation step, are both on
the losing side, so `--seq-sched 7` stays a named arm and the deployed path stays selected. **An
issue-slot census does not predict the sign of an addressing change on this device, only its
cost.** One 128-block of weight lookahead on top of it loses at every shape measured. That document
also carries the first phase map of the 256-row pass that prompt ingestion now defaults to: the two
sequence projections are 34% of it against the FFN's 44%.

`--batch` can now span up to 128 independent sequence slots. `--context N` controls
allocated KV positions per head, in multiples of 256 up to 32768. Use an explicit
context budget for larger batches; recurrent state also grows per sequence.

```sh
./bonsai-halo --batch 32 --context 512 --ffn auto --prompts prompts.txt -n 32
./bonsai-halo --batch 32 --context 512 --ffn a4 --prompts prompts.txt -n 32
./bonsai-halo --batch 32 --context 512 --ffn scaled-a8 --prompts prompts.txt -n 32
```

The `a8`, `a4` and `scaled-a8` modes adopt Kelana's compact ternary FFN kernels.
Weights are repacked once for all 64 layers. At each layer, the engine runs the
existing attention or recurrent computation in slices of eight rows, then runs
one FFN over the gathered batch. Prefill accepts up to 128 tokens per call while
preserving causal attention and recurrent replay. The original `deployed` mode
remains the default. `sliced` uses the new scheduling with the original FFN for
comparison. `auto` keeps eight-bit activations and selects the original pass at
1–4 rows, the split original FFN at 5–31 rows, and batched integer A8 at 32–128
rows. The tested 8-row wide kernels lose, so the automatic path avoids them.

The map modes port the measured schedules from Kelana's
[tile ownership](https://github.com/hara-seihun/kelana/blob/main/research/ffn/batched/tile-ownership/README.md) and
[dense consumer](https://github.com/hara-seihun/kelana/blob/main/research/ffn/batched/dense-consumer/README.md) experiments.
`map-a8` shares weight tiles across token waves with contiguous block loads.
`map-scaled-a8` selects the scaled-operand schedule by batch size.
`map-a4` uses dense five-trit bytes for small batches and pair-code wide loads
for larger batches. `auto-map-a8`, `auto-map-scaled-a8` and `auto-map-a4`
use those kernels at 32 or more rows and the same small-row routes as `auto`.
Direct map modes remain available for source-matched comparisons.

A8 preserves activation width but changes floating evaluation order. A4 changes
both FFN activation quantizers to four bits. Scaled A8 represents dequantized
activations in FP16 and accumulates across scale blocks. These are separately
labelled numerical modes, not claims of bit-identical model behavior. Drafters
and serving do not yet use the wide-batch path.

The [full-model comparison](tools/batch-compare/README.md) owns measured batched
generation TPS, prompt-processing TPS and fixed-input logit differences. The
runner `tools/run-batch-compare` holds the research GPU lock, pauses the resident
Bonsai server during measurement, runtime-masks it against recreation by a new
Pi session, and restores it on exit. It runs every payload mode in one systemd
scope with a 26 GiB memory throttle, a 28 GiB hard limit, no swap, and a 20-minute
runtime limit. The wrapper requires another 8 GiB of host memory before launch.
The scope owns the complete process tree, and cleanup stops that scope before it
releases the GPU lock. One-time loading and weight preparation are outside
timing; all input-dependent inference is inside.
Raw timing samples and logits live in
[`../../data/bonsai2/batch-comparison`](../../data/bonsai2/batch-comparison/README.md).

The [packed-map full-model comparison](tools/batch-compare/results/map-summary.md)
measured 176.6 generation tokens/s on optimized scaled A8 and 206.2 on optimized
A4 at 32 streams, against roughly 133 original and 160 previous automatic A8.
A4's larger numerical change is reported separately. Its five-round confirmation
runtime-masks the resident service after two earlier runs exposed a Pi-triggered
restart during measurement. The new kernel paths reproduce their matched
numerical controls bit-for-bit on the full-model logit checks.

The [ramped automatic-A8 comparison](tools/batch-compare/results/auto-a8-ramped.md)
measured 132.8 → 159.4 aggregate generation tokens/s at 32 streams and
133.1 → 142.5 at eight streams. A 256-token prompt processed in 128-row batches
reached 291.2 versus 154.9 tokens/s, including the LM head on all final-pass rows.
The [all-mode comparison](tools/batch-compare/results/full-model.md) includes A4
at 164.8 generation tokens/s and the split original FFN at 148.4, both at 32
streams. The reports retain all three rounds, clocks and numerical comparisons.
Preparation selects the images required by the chosen modes. The controls add
4.55 GB; optimized IU8 alone adds 4.55 GB in its lane-major order; optimized
scaled A8 needs both orders, 8.82 GB; optimized A4 holds pair codes and dense
five-trit bytes, 8.02 GB. All modes together need 16.58 GB. One 14.9 MB
activation workspace serves all layers. The 32-slot engine itself uses 15.43 GB
at context 512. Holding every representation beside it exceeds this host's
available GPU allocation budget, so the comparison runs the A8 and A4 panels
separately with original and automatic-A8 controls in both.

## Single-token experiments

Use `--batch 1 --ffn single-map`, `single-state`, `single-retile`, or `single-grid`
for the opt-in single-token comparisons. They prepare no additional weight image.
The [five-round comparison](tools/batch-compare/results/single-map.md) found
34.0 tokens/s original, 33.4 for the exact palette map and 34.5 for the
floating-point-reassociated K-split map. The default is unchanged.
`kernels/single_map_check.cpp` checks the integer operand identity on the CPU;
`kernels/single_map_isa.py` counts the emitted instructions.

[Where a single-token step goes](docs/single-stream.md) is the current phase map of that step,
29.694 ms with 5.90 GB of weights and state against a 24.4 ms roof. It prices every phase against
the bandwidth its own unit count allows, and records two exact changes measured down to nothing:
spending the phase-boundary idle on the next weight stream (0 to +0.3%, not in the tree) and moving
the workgroup count (+0.43% at 48, −0.81% at 40, default unchanged). What the deficit does track is
how many blocks each wave reads back to back per unit, which makes multi-tile units the next exact
attack. `HALO_BENCH_GRID=60,48 [HALO_PROFILE=1] ./bonsai-halo --bench` alternates arms token by
token in one process and prints paired medians, which is how anything this small gets measured
here; `HALO_GRID_ROWS` pins the grid for a process that cannot call `Engine::set_grid_rows`.

Both of that document's explanations for the deficit have since been tested and neither survives:
[the phase-length result](docs/matvec-phase-length.md) measures the barrier and the unit deal
directly, shows a deeper weight cursor is +44.7% in a long stream and a null in this engine, and
moves the narrow grid to 64 - the only value in that neighbourhood that divides every unit count
here - for +0.25% where the quantisation ceiling predicted +7.7%. What the deficit tracks is how
long a phase runs: 175 GB/s at one grid round, 214.8 at five, 235.6 in steady state, with this
engine's matvec phases at five to eighteen. That is 9.2 us per round plus 6.2 us fixed per phase,
and 257 matvec phases a token.

## Serving

```sh
./bonsai-halo serve --port 8471 --dflash ~/data/bonsai2/drafters/dflash2.safetensors   # OpenAI-compatible, 127.0.0.1
curl localhost:8471/v1/models
curl -N localhost:8471/v1/chat/completions -H 'content-type: application/json' -d '{"model":"bonsai-2-27b","stream":true,"reasoning_effort":"low","messages":[{"role":"user","content":"hi"}]}'
```

`/v1/chat/completions` renders the model's own chat template (system or developer message, tools as JSON in the system block, assistant turns with their `reasoning_content` preserved, tool results), streams `reasoning_content` and `content` deltas, converts the model's `<tool_call>` blocks into OpenAI `tool_calls`, and returns usage with `halo` timing fields. `reasoning_effort` `none`/`off`/`minimal` disables thinking, `low`/`medium`/`xhigh` select the template's effort text (Pi's thinking levels map onto these). Greedy requests use DFlash2 when loaded; `temperature > 0` samples without a drafter. [Concurrent requests decode together](docs/served-concurrency.md): `serve --slots N` gives the scheduler N sequence state slots and a step carries one row per active request, which is 1.64x aggregate throughput at four clients and 2.57x at eight (at `--slots 8`), with the completions identical to the serialized server's. A request that is alone still takes the drafted route; one that has shared a step stays on plain decode, because a batched pass does not feed the drafter. [Above eight rows a step shares its weight stream](docs/serve-decode-route.md): the persistent kernel still runs prep, the projections and attention eight rows to a slice, and the FFN and the vocabulary head run once for the whole step, which is 203.0 against 146.7 tok/s aggregate at sixteen rows and 220.4 against 146.6 at thirty-two, with identical completions. [At thirty-two rows a step takes the wide schedule](docs/serve-decode-default.md) - the one that already ingests every served prompt - for +52.1% engine-side and +19.9% aggregate over HTTP at 32 clients; below that floor mode 20 is bit-identical to the sliced route (0 differing bits in 7,946,240 logits) and +3.1%, and `HALO_SERVE_DECODE_MODE=4` restores the predecessor. At `--slots 4` a step never reaches nine rows, so those routes need more slots than the resident service runs. The resident service runs `--slots 4` at the full context, 23.1 GB on device against 13.8 GB for one slot; a slot costs 88 KiB per token of context plus 220 MB of recurrent state. Prompt-prefix snapshots (four, least recently used replaced) make a prompt that extends a previous one or shares its tools and system text resume instead of re-prefilling. A snapshot holds the recurrent state and the K/V of every attention, MTP and drafter cache slot for its positions, so restoring it is exact no matter what the requests in between prefilled; the K/V copies are sized to the snapshot (about 90 KB per token) within `HALO_SNAPSHOT_KV_GB` (default 6), and a prefix too long for the budget is recomputed rather than cached. Setting `HALO_SNAPSHOT_KV_GB=0` disables snapshots and also omits their recurrent state, convolution ring and replay-storage allocations, saving 880,410,624 bytes. Before K/V was part of the snapshot, a restored prefix could read stale cache contents from an unrelated request. The 32K context is `MAXCTX`; `max_tokens` is clamped to the room left.

The `serve` subcommand exposes a local OpenAI-compatible API.

## Tools

| Path | Purpose |
|---|---|
| `bench/bw` (`make bench/bw`) | GPU read bandwidth by allocation type. 242 GB/s on this box for every type. |
| `bench/coop_cost` (`make bench/coop_cost`) | What the persistent kernel's scaffolding costs, using its own `grid_sync`, `next_unit` and `mv_rows_t`: 747.6 ns per grid barrier, 265.9 ns per unit dealt, and streaming bandwidth against cursor depth and phase length. `--phase-units N` gives a phase the engine's own length, which is the difference between measuring this engine and measuring a steady state it never reaches. See [the phase-length result](docs/matvec-phase-length.md). No model, seconds per arm. |
| `tools/seq_a4_pack_check.py` | Host check that the four-bit and eight-bit sequence operand pairs agree on every dot product and that both storage orders round-trip. No GPU. |
| `tools/seq_quant_quality.py` | Reads a `batch_compare` run's quality dumps pivoted on `--seq-quant`: teacher-forced NLL, greedy agreement, and the `a4e`/`a4` bit-identity that accepts the four-bit kernel. |
| `tools/serve_load.py` | Drives the HTTP product with concurrent chat completions: per-arm servers or the resident service, aggregate and per-request rates, streaming against buffered, and completion-text comparison between arms. [Owner](docs/served-concurrency.md). |
| `bench/mv` | Matvec kernel GB/s on synthetic tiles, rotating over 96 MB so the MALL does not help. |
| `tools/refdump` | Dumps every f32 activation of the last prompt token from llama.cpp (CPU) into a directory. Build line is in the file header. |
| `tools/prefill_budget.py` | Conventional-work and streaming-traffic budgets for prompt processing; assumptions are explicit, not absolute speed limits. |
| `tools/gdn_state_isa.py` | Instruction, basic-block and memory-request census of the recurrent state codec's load and store loops, from `kernels/gdn_state_isa.hip`, which instantiates them alone with the format as a compile-time constant. Eight seconds, no GPU lock, no model, against 75 for `kernels/halo_rows.o`. It is what priced [the state's memory coordinate](docs/gdn-state-coord.md): the codec is 511 issue slots a unit where `gdn_token` is 166. |
| `tools/halo_scale_census.py` | Whether the HALO weight tiles' fp16 block scales are constant within a row, which would have let 7.143% of the 5.9 GB weight stream be hoisted to one scale per row. They are not: eight distinct values across the first eight blocks of every sampled row. Reads the tile cache with strided numpy views; no GPU, no model. |
| `tools/gdn_state_quality.py` | What a [packed recurrent-state coordinate](docs/gdn-state-pack.md) does to the distribution the model emits: paired teacher-forced NLL first, then the logit tables, which are saturated by the activation quantisers and cannot rank coordinates. It also reports [the horizon panel](docs/gdn-state-horizon.md), which is the generation shape rather than a prefill window, and merges arms measured in separate processes. |
| `tools/gdn_state_format.py` | An offline search over recurrent-state storage coordinates against a real dumped state, scored on the per-row relative readout error that a normalised dot product actually sees. One `--only state-dump` GPU run answers the whole family; see [the horizon document](docs/gdn-state-horizon.md) for what it settled. |
| `tools/kernel_resources.py` | Waves per SIMD32, VGPRs, SGPRs and spills per template instantiation, from the compiler, with no GPU. Finds an occupancy cliff before a panel is spent on it. |
| `tools/object_resources.py` | The same register, spill, scratch and LDS numbers read out of objects that are already built, in under a second, several objects side by side: `tools/object_resources.py BASE/kernels/halo_rows.o kernels/halo_rows.o 'k_forward_rows<8'`. `kernel_resources.py` recompiles the translation unit, which for `kernels/halo_rows.hip` is 36-75 s and often more than one bounded call. |
| `tools/isa_loop_count.py` | Per-basic-block mnemonic histogram of one kernel in a `hipcc -S` listing, with no GPU. Reports `N work + M scheduling`, because `s_delay_alu` and `s_waitcnt` are not work and [a census delta made of them predicts nothing](docs/activation-scale-axis.md). Also prints the memory address-form table, which is how you see that an addressing change reached the ISA rather than only the source. |
| `tools/phase_totals.py` | Per-phase totals of a `batch_profile` run, by launch kind, with each phase's launch count and row width. The phases a change cannot reach are its in-process control. |
| `tools/sequence_layout_compare.py` | Cross-run bitwise logit comparison after matching inputs and arithmetic settings; [direct-layout results](docs/wide-sequence.md#direct-layout-result). |
| `tools/sequence_state_check.cpp`, `tools/direct-commit/` | [Native mixed commit/replay acceptance and host state probe](tools/direct-commit/README.md), run under `tools/run-batch-compare --state-check`. |
| `tools/run-batch-compare --engine ARGS` | The engine binary itself under the research GPU lock, resident-service mask and bounded systemd scope, for `--bench`, drafter and serving-shaped checks that must not race the server. |
| `--prefill-sweep off,wide-deployed[,wide-commit-a4]`, `--sweep-rounds R` | Walks prompt-ingestion routes inside one process, resetting the sequence between arms, and prints each arm's prompt rate and an FNV-1a digest of its greedy token stream. Two routes in two processes cannot be compared on this box. |
| `tools/batch_profile`, `tools/batch_profile.py` | Real-model launch and device-phase tracing, paired with uninstrumented passes; see [the findings](docs/prefill-gap.md). `--decode-streams N` profiles a generation step across N slots instead of a prefill pass, and `--decode-tokens K` gives every slot K rows in that step, which is the verify shape of a speculative step; see [the decode map](docs/decode-map.md) and, for K up to 8, [the pass width](docs/pass-width.md). `--decode-prompt` is a list and `--decode-context` raises the KV allocation, which walks a generation step's position axis the way `--prefill-scan` walks prefill's; see [a generation step's attention](docs/attn-row-groups.md). |
| `tools/kernel_trace.py` | Reads a `rocprofv3 --kernel-trace` CSV, segments dispatches on host gaps and reports one pass or step by kernel instantiation, with grid, registers and the share of time that is dispatch gap; see [the step's dispatch map](docs/decode-dispatch.md). |
| `tools/profile_read_check`, `tools/profile_counters.py` | Known-byte read kernels and calibrated external-cache request summaries. |
| `kernels/single_map_check`, `kernels/single_map_isa.py` | Host operand exactness and emitted-ISA counts for the single-token maps. |
| `kernels/mv_nibble_check` (`make kernels/mv_nibble_check`) | Host exactness probe for the [IU4 ternary operand map](docs/mv-dot8-nibble.md) — element order, the nine-code alphabet and the accumulator identity — running `kernels/mv_nibble.hpp`'s own source against the stored HALO format. No GPU, under a second. |
| `tools/ffn_pack_check` (`make tools/ffn_pack_check`) | CPU check of the batched FFN weight decoder and scale placement. |
| `tools/ffn_a4_pack_check` (`make tools/ffn_a4_pack_check`) | CPU round-trip check of the lane-major, pair-code and dense five-trit representations. |
| `--dump DIR` | Same tensors from the v1 path of this engine; `tools/compare.py REF MINE [names]` prints cosine and max error per tensor. |
| `--logits FILE`, `--readback DIR`, `HALO_STOP=n`, `HALO_LAYERS=n` | Fused-kernel debugging: final logits, raw buffers, early exit after n barriers or n layers. |
| `HALO_PROFILE=1`, `HALO_PROFILE_ROWS=n` | Per-phase timing of the persistent kernel (device timestamps at every barrier), optionally only for passes of n rows. |
| `HALO_SPEC_DEBUG=1` | Print every speculative step (anchor, drafts, target argmax, accepted count). |
| `HALO_DF_DUMP=DIR`, `HALO_DF_STOP=n`, `HALO_DF_LAYERS=n`, `HALO_MTP_DUMP=DIR` | Drafter debugging: dump buffers after one draft step, early exit. |

## Layout

| Path | Contents |
|---|---|
| `kernels/halo_kernels.h` | Model constants, kernel argument structs, `LayerW`, `FwdParams`, launchers |
| `kernels/device.hpp` | Shared device code: trit peel, `mv_rows` (the ternary dot product), `prep_chunk` (v1 prep), reductions |
| `kernels/phases.hpp` | Phase machinery of the persistent kernels: device barrier, dynamic units, prep, dot4 and WMMA matvecs, attention (both head geometries), embedding, argmax, cooperative grid sizing |
| `kernels/halo_rows.hip` | The target's persistent kernel over rows: GDN with replay-based rollback, layer loop, capture of the DFlash2 feature layers |
| `kernels/halo_draft.hip` | The MTP head and the DFlash2 drafter (ingest, block, dynamic convs, top-k, selector) as persistent kernels |
| `src/q8.cpp` | Safetensors reader and the Q8 tile quantiser/cache for drafter weights |
| `kernels/halo_kernels.hip` | v1 per-op kernels (used by `--v1` and `--dump`) |
| `src/halo_format.h` | HALO tile format and the GGUF PTQ1_0/PQ2_0 decoders and encoder |
| `src/repack.cpp` | Multi-threaded repack and the `.halo` cache |
| `src/gguf.cpp` | Minimal GGUF v3 reader |
| `src/engine.cpp` | Weight upload, buffers, the v1 forward, row passes with sequence slots, drafter loading and steps, profile printing; `engine.h` holds the speculative generation loops |
| `src/batch.cpp` | Layer-wise gathering of up to 128 rows, causal slice scheduling and original-schedule comparison |
| `src/batch_profile.hpp` | Optional event pool and per-launch barrier timestamp collection |
| `kernels/sequence_batch.h`, `kernels/sequence_batch.hip` | Wide recurrent/attention projections, device-prepared lane-major ternary codes, direct producer/consumer layouts and an explicit staged control. [Owner](docs/wide-sequence.md). |
| `kernels/head_batch.h`, `kernels/head_batch.hip` | Vocabulary projection and argmax over every row of a batch, straight from the packed HALO tiles. [Owner](docs/wide-head.md); its row bound is [the pass width](docs/pass-width.md). |
| `kernels/ffn_batch.hip`, `kernels/ffn_operands.hpp`, `kernels/ffn_a4_operands.hpp` | Device-prepared weight representations and native A8, A4 and scaled-FP16 FFN execution, adopted from Kelana |
| `src/tokenizer.cpp` | libllama vocab-only tokenizer and the minimal single-turn chat prompt used by the CLI |
| `src/chat.cpp` | The model's full chat template from OpenAI-style messages and tools, and the parser for its tool-call output |
| `src/server.cpp` | The HTTP server (`serve`): chat completions, streaming, usage |
| `src/serve_batch.h`, `src/serve_batch.cpp` | The scheduler behind the server: request admission into sequence slots and decode steps carrying one row per active request, on [the batched decode route](docs/serve-decode-route.md) above eight rows and [the wide schedule](docs/serve-decode-default.md) above the wide floor. [Owner](docs/served-concurrency.md). |
| `vendor/` | cpp-httplib and nlohmann/json, single headers copied from the llama.cpp checkout |
| `src/main.cpp` | CLI, sampling |

## Owner

Source of truth is this repository. Model weights are not included; configure the model path with `-m`. The tokenizer and reference dumps require a HIP-enabled llama.cpp build (the original build directory was named `bonsai-hip/build-hip`).

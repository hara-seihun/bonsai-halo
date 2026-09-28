# Wide sequence projections and direct state commit

> **Read this before you use any bit-identity statement below as a route property.** Every
> "bit-identical", "unchanged numerical map" and "identical logits" claim in this document is a
> statement about *that change*, against the arm measured beside it, and they were all true when
> written. They are no longer true of the **wide route as it ships**: `c131fe2` made four-bit
> activations the default on the sequence input projection and
> [`f91a815`](seq-out-a4-default.md) on the output projection, so a pass that crosses into the wide
> route (`batch_mode >= 17` and 32 rows or more) applies two activation quantisers the eight-row
> route does not. **Crossing 32 rows is therefore a numerical change for every consumer of
> `forward_batch`, not only for a panel that asked for it** - `bc30b5f7` found it as 18 of 32
> concurrent greedy completions diverging from the eight-row route while 24 of 24 agreed at 16 rows,
> and spent a panel looking for a bug in their own scheduler first.
>
> `HALO_SEQ_QUANT=a8` restores the exact operand on both projections, and with it `mode 20` is the
> wide *schedule* over the deployed FFN arithmetic - the route accepted against the eight-row one on
> a single greedy digest ([`serve-prefill-route.md`](serve-prefill-route.md)). **It is still not
> bit-identical to the sliced route, and that is measured:** a 384-token document at `a8` hashes
> every logit to `7241142880296265908` at mode 0 and `15043080294956075508` at mode 20, at both 128
> and 256 rows per pass. The wide projections reassociate the FP32 drain, as the table below says
> they do. So token agreement between mode 4 and mode 20 is an empirical property of the prompts you
> measured, not a guarantee the route can offer - and at the default coordinate the two routes differ
> in their arithmetic as well as their reduction order.

The [measured prefill gap](prefill-gap.md) identified eight-row non-FFN projections and deferred GDN replay as work to remove. `kernels/sequence_batch.{h,hip}` now supplies wide recurrent and attention projections. `launch_sequence_part` in `kernels/halo_rows.hip` retains the original input quantizer and sequence arithmetic on either side of them.

## Resident-state result

The next [resident GDN change](resident-gdn.md) keeps each committed sequence's state in registers across the whole pass and runs its finite-history convolution across tokens. Full-model A4 prefill is now 589.7 tok/s and 32-stream generation 252.9 tok/s, versus source-matched controls of 541.4 and 245.8. A8 reaches 477.9 and 210.1. All checked logits remain bit-identical to the corresponding sliced-state maps. This is selected within the opt-in wide committed routes, not ordinary serving.

## Direct-layout result

The quantizer now writes the WMMA operand layout itself, and the sequence core reads the projection's output planes in place. At 128 rows this removes 3,072 capture/restore launches across 64 layers. No extra weight image is needed, and the numerical map is unchanged.

| workload | staged control | direct layout | increase |
|---|---:|---:|---:|
| A8 prefill, 128 rows/pass | 438.8 | 457.3 | 4.2% |
| A4 prefill, 128 rows/pass | 516.9 | 540.8 | 4.6% |
| A8 generation, 32 streams | 200.3 | 204.8 | 2.2% |
| A4 generation, 32 streams | 239.8 | 245.7 | 2.5% |

Values are median full-model tok/s over three rounds. Each layout has a separate run with mode 0 interleaved as the control; the layouts themselves were not interleaved. A8 prefill retains a 3% spread across the three direct samples. Loading, static repacking and prompt setup for generation remain outside timing. Prefill still processes 256 tokens and pays the output head on all rows of the last pass.

Sources: [A8 control](../tools/batch-compare/results/direct-layout-a8-control.md), [selected A8](../tools/batch-compare/results/direct-layout-selected-a8.md), [A4 control](../tools/batch-compare/results/direct-layout-a4-control.md), [selected A4](../tools/batch-compare/results/direct-layout-selected-a4.md). Against original mode 0 in those selected runs, prefill is 2.95x faster with A8 and 3.49x with A4; generation is 1.54x and 1.85x respectively. A4's earlier precision tradeoff is unchanged.

[Exact comparison](../tools/batch-compare/results/direct-layout-identity.json) covers 143,032,320 finite logit values at 32, 40, 88 and 128 rows in both modes 18 and 19. Every bit matches the staged control. Separate 32-stream decode comparisons also match, including the recorded continuation tokens. [Mixed-state acceptance](../tools/direct-commit/direct-layout-result.json) passes on the selected layout, with state equality checked at matching durable prefix positions.

The first direct implementation reached 448.4/534.2 prefill tok/s with A8/A4. Its projection epilogue selected the output plane for each value. Segment boundaries are multiples of 16, so a whole wave's row tile belongs to one plane. Selecting the base and row stride once per wave removed those repeated decisions. In the native 128-row phase workload, the staged median was 265.8 ms, the first direct implementation 255.9 ms and the wave-level addressing version 252.4 ms. Their residual hashes agree. [First direct profile](../tools/batch-compare/results/direct-layout-phase.md), [staged profile](../tools/batch-compare/results/direct-layout-phase-staged.md), [selected profile](../tools/batch-compare/results/direct-layout-plane.md).

Direct input prep adds one compiled VGPR without changing its occupancy; state-part register usage is unchanged. It also omits the activation-code sum because signed WMMA has no zero-point correction that consumes it. Norm, Hadamard, scale and rounding operations stay unchanged.

## Initial wide-projection result

Three randomized rounds, 64 layers, 512-token context capacity, real prompts. Generation is aggregate output tokens per second across 32 streams. Prefill processes 256 document tokens in 128-row passes, including logits on all 128 rows of the last pass. Preparation is outside timing; all input-dependent work is inside it.

| family | workload | original | previous optimized FFN | wide + commit |
|---|---|---:|---:|---:|
| A8 | prefill tok/s | 155.1 | 291.1 | 440.2 |
| A8 | generation tok/s | 133.2 | 176.1 | 200.6 |
| A4 FFN, A8 sequence projections | prefill tok/s | 156.4 | 324.4 | 518.6 |
| A4 FFN, A8 sequence projections | generation tok/s | 133.6 | 205.7 | 239.1 |

The A8 and A4 panels are separate runs with their own controls. Reports retain every sample, including a slow mode-16 prefill sample and the mode-0 prefill spread. These are median results, not worst-case latency promises.

Sources: [A8 timing and quality](../tools/batch-compare/results/wide-sequence-tps.md), [A4 timing and quality](../tools/batch-compare/results/wide-sequence-a4.md), [32/40/128-row preflight](../tools/batch-compare/results/wide-sequence-preflight.md).

Direct commit alone changes mode 10 to 16. It moved 128-row prefill from 291.1 to 308.5 tok/s. Widening without commit, mode 17, reached 417.0; combining both, mode 18, reached 440.2. The same separation at 32-stream generation was 176.1, 182.5, 198.9 and 200.6 tok/s.

Modes 16 versus 10 and 18 versus 17 produced identical finite full-vocabulary logits at 32, 40 and 128 prefill rows and at 32 decode rows. Their four-step continuations also matched each other. The wider projection changes FP32 reduction order, so it is not bit-identical to the previous projection. Mode 18 matched all 128 preflight top-token choices and 127/128 free-running continuation tokens against mode 0. A4 remains a separate approximation: mode 19 matched 25/32 prefill top-token choices, 30/32 decode choices and 118/128 continuation tokens. Its prefill mean KL was 0.0704 versus mode 11's 0.0651. No default or serving route changes here.

## What changed

The original quantizer still produces the activation codes and scales. Each layer now runs:

1. Original input prep over each slice, writing quantized columns directly to the WMMA layout and alpha/beta metadata to its per-batch destination.
2. One wide combined input projection across all rows: qkv plus gate for GDN, or q/k/v for attention. Each projection segment writes its own row-major plane.
3. Resident GDN core over each whole committed sequence, or the original causal core per slice for attention and uncommitted passes. Both write their output quantizer directly to the next WMMA operand.
4. One wide output projection adding to the residual.
5. The existing optimized FFN.

The two-bit weight words are lane-major, with each lane's 128-weight block in two contiguous 16-byte loads. All 16 WMMA token columns are useful at aligned sizes. A weight operand feeds up to eight token tiles before it is discarded. The default widths are 1/2/4/8 tiles at padded row counts 16/32/64/128. Final partial tile groups use allocated slack, and the row guard drops their outputs.

The input/output projection quantizers are the original kernel instantiations, not reimplementations intended to agree. The wide projection does change the floating sum: it drains each 128-weight integer block into one FP32 accumulator in block order rather than reducing the original K partitions. Signed IU8 dot products themselves stay exact.

The module prepares about 1.93 GB of additional code/scale images and workspace for all 64 layers. Resident GDN adds 8.44 MB of shared token/output workspace at capacity 128. It shares one workspace across layers. It does not replace the deployed weight image. A8 preflight with two state slots used 18.21 GB total, including 8.84 GB of FFN images/workspace. The allocator frees its partial allocations on creation failure. `prepare_sequence()` runs before inference and `sequence_project` allocates nothing.

## State contract

[Direct-commit contract and host probe](../tools/direct-commit/README.md) describe `FwdParams.commit_state`. An uncommitted pass leaves the durable state before its current rows and records a replay prefix. A committed pass leaves it after the current rows and sets the next replay count to zero. Incoming replay is still honored when switching into commit.

The engine records whether a sequence's final slice is rollbackable. Rejecting any row of a committed slice throws before changing sequence bookkeeping. Accepting it whole is legal. A later speculative, uncommitted pass can still reject its own rows. The option applies to every sequence in one launch, so committed and speculative requests must use separate launches. Batch-specific options are restored when `forward_batch` returns; they do not leak into a subsequent ordinary `forward` call.

Raw GDN state arrays represent different durable prefix lengths immediately after committed and uncommitted passes. Comparing those arrays as if they represented the same point is wrong. The native [state acceptance tool](../tools/sequence_state_check.cpp) compares outputs along the mixed trajectory and compares state only at matching durable positions, with a final common commit to align both sides. The [native result](../tools/direct-commit/native-result.json) passed both mixed trajectories, both synchronized state comparisons and all rollback checks. It records 18 raw-state differences at unequal durable prefixes, none treated as aligned state equality.

## Where the time went

[Paired tracing](../tools/batch-compare/results/wide-phase-auto.md) uses a 128-token prefix and one 128-row pass with logits on every row. This is a different workload from the 256-token throughput table.

The non-FFN projection phase stamps totaled about 180 ms in mode 10. The new wide input and output projection events total about 54 ms. GDN state phase stamps fell from 60.7 to 23.5 ms. The latter combines removal of replay with the state part's own occupancy grid; it is not a replay-only attribution. Native wall time changed from a 457.6 ms median to 310.7 ms in A8 and 264.7 ms with A4 FFNs.

At the new A4 point, FFNs still take about 96.6 ms, wide projections 54.0 ms, input-prep events 25.3 ms, sequence-core events 66.5 ms and the head 25.4 ms. Copy/gather kernels and unstamped launch work are included in the input/core event totals. They are not included in the phase stamps within those events. Do not add both tables together.

The new path has more launches. Traced runs cost 7–14 ms more than native runs, so profile values locate work rather than replacing the throughput measurement.

## Packed-map follow-up

`HALO_SEQUENCE_OPERAND=scaled-f16` tests removal of the block accumulator boundary. The ternary code directly selects zero, the stored FP16 scale, or the scale with its sign bit flipped. There is no intermediate floating weight multiplication. A8 activation codes times their scales are rounded to FP16 during capture. One FP32 WMMA accumulator then spans the entire K axis, eliminating per-block integer drains, row-scale shuffles and scale products.

This is a changed numerical map. It reuses the same weight image but doubles activation fragment width. [Its full-model run](../tools/batch-compare/results/wide-scaled-operands.md) reached 432.8 prefill and 184.0 generation tok/s, compared with the default wide map's separately measured 440.2 and 200.6. It did not improve the selected implementation. The result demonstrates why deleting an intermediate is a search direction, not a speed theorem: operand construction, traffic and scheduling still need pricing.

`HALO_SEQUENCE_TT=4` tests the register/width trade without changing arithmetic. The [three-round profile](../tools/batch-compare/results/wide-phase-tt4.md) gave 315.9 ms native median versus the default width-eight run's 310.7 ms. It did not earn a default change. These are separate runs; the reports keep their source hashes and raw timing lists.

The capture/restore boundary is now removed on the default integer sequence route, as measured above. `HALO_SEQUENCE_LAYOUT=staged` keeps the prior route as an explicit numerical and performance control and supplies the scaled-FP16 experiment. `tools/sequence_layout_compare.py` compares separate runs' complete logit dumps after checking their input and arithmetic fields match.

## Operation

```sh
make -j6 bonsai-halo tools/batch_compare tools/batch_profile
./bonsai-halo --batch 32 --context 512 --ffn wide-commit --prompts prompts.txt -n 32
./bonsai-halo --batch 32 --context 512 --ffn wide-commit-a4 --prompts prompts.txt -n 32

tools/run-batch-compare --tag wide-sequence-tps \
  --modes 0,10,16,17,18 --streams 8,32 --prefill-rows 32,128 \
  --prefill-tokens 256 --rounds 3 --gen-steps 8 --quality-rows 32 \
  --multistep-streams 32 --multistep-steps 4 --reference-pairs 16:10,18:17
make -C tools/direct-commit sequence_state_check
tools/run-batch-compare --state-check --out ../../data/bonsai2/sequence-state-check/run.json
```

Modes 16, 17 and 18 use optimized scaled-A8 FFNs above 31 rows; mode 19 uses optimized A4 FFNs. All retain deployed/sliced FFN routes at smaller sizes. Mode 16 only commits, 17 only widens, 18 and 19 do both. Widening starts at 32 total rows; committing applies at every row count.

`HALO_SEQUENCE_OPERAND` accepts `int8` or `scaled-f16` at module creation. `HALO_SEQUENCE_TT` accepts 1, 2, 4 or 8. The layout, operand and width are recorded in benchmark provenance. Default operation is `int8`, direct layout, automatic width, and resident GDN for committed wide passes. `HALO_GDN_RESIDENT=0` selects the sliced-state control; `HALO_GDN_SPLIT=4|8|16` defaults to 4. `HALO_SEQUENCE_LAYOUT=staged` selects the capture/restore control. Scaled-FP16 operands select staged layout by default; explicitly requesting direct plus scaled-FP16 is rejected. Raw data live under `../../data/bonsai2/batch-comparison/`, with the per-run names linked above. Source and executable hashes are in each run manifest; no downloaded weights or logit arrays are committed.

# What an ideal implementation could buy

The [joint research program](../orchestration/RESEARCH.md) keeps these budgets separate from semantic lower bounds and accepted inference results. `tools/prefill_budget.py` now distinguishes independent sequences from consecutive prompt rows and prices FFN-plus-sequence A4 separately from FFN-only A4. The tables below retain their stated one-sequence, F32-state hypotheses; they are not a model of every current route.

The question has two different meanings. Perfect scheduling of this GPU's existing instructions has a useful numerical budget. An unspecified "God's ISA" does not have an absolute tokens-per-second ceiling: its instructions have not been assigned a throughput, a state capacity or a communication cost. Making an entire fixed model one instruction does not say how long that instruction takes.

So the answer has two parts, and only the first is a number:

- *Under a declared service-capacity model* for this GPU's matrix instructions, the matrix work in the conventional computation occupies about **430 microseconds per token**: **2,320 prompt tokens/s** at IU4 rates, **1,160** at IU8, **1,743** for an FFN-IU4, other-matrices-IU8 mixed format. Adding the non-matrix sequence arithmetic serially on the vector lanes gives 2,142, 1,114 and 1,640 at 512 tokens of context. Those figures are consequences of the declared issue periods and the counted work, not a physically proved minimum: nothing here proves the hardware cannot issue faster, and everything omitted makes a real engine slower.
- Across representations, there is no such bound. These counts price one decomposition of the map. An exact whole-map rewrite that performs fewer products is not refuted by them.

These calculations price the conventional model computation. They are not impossibility proofs against a different exact map decomposition. The [Kelana resource-bound framework](../../kelana/research/discovery/resource-bounds/README.md) keeps those two kinds of claim separate.

## The amount of conventional work

Bonsai has 64 layers, hidden width 5120 and FFN width 17408. Shapes below are the engine's own constants (`kernels/halo_kernels.h`), not a paper's. Counting one multiplication and one addition as two operations:

| Component | Dense MACs per token |
|---|---:|
| All FFNs | 17,112,760,320 |
| GDN projections in 48 layers | 5,536,481,280 |
| Attention projections in 16 layers | 1,677,721,600 |
| Full 248320-entry output head | 1,271,398,400 |
| Total ternary-weight matrix work | 25,598,361,600 |

That is **51.20 billion conventional operations per token** with a full output head on every token.

The sequence-dependent and small-matrix work is separate, and the earlier draft left it uncounted:

| Component | MACs per token |
|---|---:|
| GDN delta-rule recurrence (three passes over 48x48 states of 128x128) | 113,246,208 |
| GDN state decay (multiplies, not MACs) | 37,748,736 |
| Attention QK and AV at 512 tokens of context | 113,344,512 |
| bf16 alpha/beta projections | 23,592,960 |
| GDN depthwise conv1d, 4 taps | 1,966,080 |

The recurrence total is fixed; the attention total is `16 layers x 24 query heads x 256 x 2 x (context + (B+1)/2)` MACs per token, so at 32k context it is 56x larger, 6.3 billion MACs per token. On the vector lanes that alone takes about 850 microseconds per token, more than all the matrix work put together. Softmax, normalization, Hadamards, scales and SiLU are on top of all of this.

Prompt ingestion only needs the final token's logits for generation. Omitting the other output-head rows lowers the dense work toward 24.33 billion MACs per token, about 5% less. Our existing 256-token prefill comparison computes the head on every row of the final pass, so its 32-row and 128-row pass configurations do different amounts of head work: the head is charged on 32 of 256 tokens in one and 128 of 256 in the other. `--logits last-pass --prompt-tokens 256` prices exactly that policy. Compare modes within a pass size, not those two configurations as if their work were equal.

## Perfect use of the current matrix instructions

The GPU has 40 CUs, so 80 physical SIMD32 units. A 16x16x16 wave matrix instruction performs 4096 MACs. Declare an ideal service period of 16 cycles for IU4 and 32 for IU8 or FP16, at 2.9 GHz:

- IU4: 118.8 trillion conventional operations/s.
- IU8 or FP16: 59.4 trillion conventional operations/s.

The periods are an explicit issue model, not instruction throughput guarantees supplied by the ISA manual. Two native probes back them: the [normalized issue-rate experiment](../../kelana/research/ffn/batched/triple-packing/README.md) reached 108.6 TOPS IU4, and the [native WMMA rate table](../../kelana/research/ffn/batched/arithmetic/README.md) reached 109.3. Both are rates those probes achieved, not capacity upper bounds.

Dividing by the work above gives:

| Hypothesis | Matrix only | Plus serial sequence arithmetic |
|---|---:|---:|
| Every matrix on IU8 | 1,160 | 1,114 |
| FFNs on IU4, other projections on IU8 | 1,743 | 1,640 |
| Every matrix on IU4 | 2,320 | 2,142 |
| Every matrix on IU4, 32k context | 2,320 | 769 |

The right column charges the recurrence and attention products at one f32 FMA per lane per cycle (7.42 trillion FMA/s), with no dual issue and no overlap with the matrix phases. The two columns bracket how the *declared arithmetic* could be scheduled; they do not bracket runtime. Operand construction, traffic, launches and conflicts are all omitted, and every one of them only adds time, so a real engine can land below both columns.

The rate table covers all three operand classes, so nothing needs extrapolating. Corrected to the physical SIMD32 count, its largest row is FP16 11.8495 ns, IU8 11.813 ns and IU4 5.99325 ns per WMMA per SIMD32: 55.3, 55.5 and 109.3 TOPS achieved. FP16 and IU8 are equal there and IU4 is half their time, which is the shape the 16/32-cycle service model assumes. Substituting the achieved figures (`--wmma-ns 5.99325 11.813`) moves the matrix-only rows to **1,084 IU8, 1,616 mixed and 2,136 IU4**, and the serial-sequence column to 1,043, 1,527 and 1,984.

The all-IU4 rows grant four-bit activations everywhere. That is not a semantics-preserving implementation of the original A8 model. The table's mixed row prices FFNs at A4 with other projections at A8. Current sequence projection kernels also have A4 input/output maps; the tool's `ffn_and_sequence_iu4_head_iu8` row prices that distinct combination while retaining an IU8 head. A serving mode must be matched to its actual quantizer settings before choosing a row. None of these rows includes the separately reported bf16 alpha/beta projection issue cost.

Operand construction, scales, memory traffic, synchronization and non-matrix phases still have to fit. Therefore these are generous budgets for that particular computation and precision, not forecasts. Conversely, a whole-map rewrite that needs fewer matrix products is not constrained by the original dense-MAC count. Our packed 2x2 examples are exactly why we must not turn that count into a theorem about every implementation.

## What five trits per byte actually buys

The premise is already implemented. HALO packs five trits into each of the 24 `qs` bytes of a 128-weight block and four into each of the two `qh` bytes:

- 26 code bytes per 128 weights, 4.92 trits/byte: 1.625 code bits/weight.
- Two bytes of block scale: 0.125 bits/weight.
- Total: **1.750 bits/weight**.

A code at a perfect five trits per byte would need 25.6 bytes instead of 26, with the same scales: **1.42% fewer bytes**. Five independent trits require `log2(243) = 7.925` bits, so a byte is within 0.95% of that code-density limit, and the whole-block entropy figure with the same scales is 2.27% below the current format. There is no 18.75% two-bit-to-dense saving left to collect; the engine already reads dense bytes.

The streamed ternary matrices occupy 5.600 GB, plus 47 MB for the bf16 alpha/beta matrices, so 5.647 GB crosses memory per streaming pass. The 5.947 GB PTQ1_0 file corroborates it: of the 300 MB difference, 278 MB is the same packing of the 248320-row embedding table, which is gathered rather than streamed, and the rest is f32 norms and metadata.

`log2(3)` per trit is the maximum-entropy length for independent uniform trits. It bounds that representation family with those scales. It says nothing about the shortest description of these particular trained weights, which are neither uniform nor independent; empirical trit statistics, weight-sharing structure or a different decomposition can all land below it. None of these numbers is a statement about the model's Kolmogorov complexity.

Reducing weight bits also does **not** multiply matrix throughput by the same ratio. The activation operand is not ternary, and it still needs its ports, registers and arithmetic. A native ternary matrix instruction could have a different rate, but that rate must be specified or built.

For scale only: if we *grant* a hypothetical ternary instruction 2.5 times IU4's MAC throughput, reflecting five weight trits where two four-bit weights fit, the matrix-only number becomes about 5,800 tokens/s. The grant is the missing hardware premise. It does not follow from five-trit packing, and at B=128 the memory system below would still cap the pass near 5,360 tokens/s.

## Memory and batching

Assume each matrix's weights are fetched once for B prompt tokens, with perfect reuse. At the physical 256 GB/s DRAM rate, weights alone allow approximately:

| Prompt tokens sharing each weight read | Weight-only tokens/s |
|---|---:|
| 1 | 45 |
| 32 | 1,451 |
| 128 | 5,803 |
| 256 | 11,606 |

At the measured 242 GB/s streaming rate those numbers are 5.5% lower. The batch dimension is what changes the problem: weights amortize over prompt tokens, but not over successive autoregressive single-token steps without speculation or another mechanism.

GDN has 151 MB of FP32 recurrent state. Reading and writing all of it for every token costs another 302 MB/token. That policy alone limits throughput to 848 tokens/s at 256 GB/s, even if weight traffic disappeared. It is **not** an invariant of GDN. The sliced `ph_gdn` path loads one head's state quarter into registers, carries the sequence's rows through it and writes back once. Resident direct-commit routes retain state across a wider sequence chunk. Independent sequences still require independent state. That puts the weight-plus-state budget at about 3,130 tokens/s for B=128. Carrying all 128 tokens through one state read/write lifts that budget again.

Two more real costs sit between those budgets and a running engine:

- K/V operands. Each position holds 64 KB across the 16 attention layers. The optimistic model reads the prefix once per pass; the current kernel reads it once per eight-row group, and its GQA sharing across the six query heads of a group only holds while those 256-position chunks (128 KB of K and 128 KB of V) stay in the 2 MB L2 or the 32 MB last-level cache.
- Logits. 128 full-vocabulary f32 rows are 127 MB written per pass.

The replayable sliced schedule also replays: each pass re-runs the previous pass's accepted rows through the recurrence before its own, so it performs the GDN recurrence about twice per token. Its 82 KB per token per GDN layer of replay records must survive from the pass that writes them, across all 64 layers and the whole weight stream, until the next pass reads them. Direct-commit routes avoid this term; it is not a universal engine cost. That is 7.90 MB per token of record traffic, and no cache-residency story is available for it.

Combining these gives two memory budgets at 512 tokens of context, 256 GB/s:

| B | Eight-row state, K/V, logits and replay, ideal weight reuse | Ideal single chunk, shared K/V |
|---:|---:|---:|
| 32 | 1,125 | 1,362 |
| 128 | 2,680 | 5,356 |
| 1024 | 4,226 | 37,096 |

Neither column is a ceiling on what the current engine can move, and the left one is not "the engine's bandwidth". Both assume every weight matrix is read once per B tokens. The sliced schedule re-reads projection weights per slice; current wide schedules batch the sequence projections and head too. Every route still has finite token-tile widths and can revisit weights, so none universally reads each matrix once per whole batch. A layer's non-FFN weights are 23-25 MB against a 32 MB last-level cache, so those re-reads may or may not reach DRAM, and nothing here measures which. Activation traffic is excluded too. "Omitted" means optimistic in every case.

At 32k context the left column falls to 724 tokens/s, so every number here is a short-context statement.

## Where the measured gaps are now

The [first native phase profile](prefill-gap.md) found the sliced projection and replay costs that motivated the wide schedule. Its 155.3 / 291.6 / 324.5 prompt tokens/s panels are historical evidence for that decision, not current engine throughput. [Wide sequence execution](wide-sequence.md), [resident GDN](resident-gdn.md), [decode widths](decode-streams.md) and the [combined full-model panel](full-tps-20260921.md) own the later configurations.

The [matrix-rate experiment](ffn-matrix-rate.md) corrected IU8 WMMA to about 34.3 cycles per physical SIMD32 and found independent VALU largely hidden beside it. The [sequence stream ablation](seq-block-roundtrip.md) found a different dependence structure: the exposed weight round trip is worth about 7.1% of its measured 32-stream generation step. Its pinned-stream diagnostic computes the wrong values, so that percentage is an optimization opportunity rather than a candidate speedup.

The current [research program](../orchestration/RESEARCH.md) uses these measured costs to select work while Kelana searches alternative complete maps. A new ratio to a budget must use the same workload, head policy, numerical map and state schedule. In particular, 32 independent generation streams are not a 32-token single-sequence prefill chunk. `--sequences 32 --batches 32` charges 32 independent state regions and prefixes, and one new causal position per sequence.

For single-token generation, the measured 34.5 tokens/s sits against a roughly 41 tokens/s budget at 242 GB/s — and that 41 is weights *plus* the 302 MB of per-token state traffic, 5.95 GB per token. Weights alone would allow 43. Either way the conventional window is small: dense-consumer work can remove overhead inside it, but a large multiplicative gain needs fewer bytes per accepted token, such as speculation, retained useful state, or a genuinely different computation. The 41 figure assumes this streamed representation and one target token per pass, not every possible map.

## Reproduce or change the assumptions

```sh
python3 tools/prefill_budget.py
python3 tools/prefill_budget.py --wmma-ns 5.99325 11.813 --bandwidth-gbs 242
python3 tools/prefill_budget.py --logits last-pass --prompt-tokens 256 --batches 32,128
python3 tools/prefill_budget.py --context 32000 --batches 128
python3 tools/prefill_budget.py --batches 32 --sequences 32 --context 512 --bandwidth-gbs 242 --json
python3 tools/prefill_budget.py --ternary-gain 2.5 --json
python3 tools/test_prefill_budget.py
```

`tools/prefill_budget.py` owns the counts and formulas. The last command explicitly grants hypothetical hardware. The output lists omitted costs and refuses to label these numbers absolute bounds. Neither CPU nor NPU throughput is added to the GPU: their data movement, supported arithmetic and executable schedules would have to be priced first.

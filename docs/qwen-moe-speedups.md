# Applying the Bonsai and Kelana speedups to Qwen MoE

the project owner requested application of the discovered speedups on September 23, after
the [Qwen baseline](qwen-moe-performance.md) was measured. The target weight
image remains the pinned 22.13 GB GGUF. A faster route must name its actual
full-model gain; Bonsai's measurements are not transferred percentages.

## Transfer inventory

| Discovered mechanism | Qwen implementation and disposition |
| --- | --- |
| Packed weight consumption without a full expanded matrix | Already present in the native Q4_K/Q5_K/Q6_K/Q8_0 MMVQ and MMQ consumers. Ternary radix-3 peel and nibble identities do not encode Qwen's general quantized weights. |
| Shape-specific matvec width and occupancy | Tested native gfx1151 Q8/Q6 wave splits. Eight/two waves lose about 1% full-model decode in two alternating pairs. Two/one waves did not establish a repeatable gain. Both changes were reverted. |
| Four independent integer-dot chains | Bonsai repaired a 32-dot chain. Qwen's Q8 partial has two dots, so there is no equivalent chain to remove. |
| Packed vector loads | An ISA probe found the native two-ushort Q8 read already becomes one 32-bit device load. Rewriting its spelling saves no instruction. |
| Parallel LDS drain across output rows | The relevant native single-token block owns one output row, not Bonsai's eight-row drain. Its native wave reduction is the applicable control. |
| Weight-image padding to avoid 4096-byte stride aliasing | The Q8 and Q6 block strides trigger that exact alignment only at K=65,536 and K=524,288. Qwen's corresponding widths do not trigger it. |
| Wide FFN, sequence projections and prompt scheduling | Native MMQ and prompt batching already provide these. The selected scheduling reference is batch 512 / ubatch 256, which outperformed ubatch 128 for the measured 256-token prompt. |
| Shared gate/up activation preparation and expert-major grouping | Already present in `mul_mat_id`. New work must improve the existing implementation rather than add another grouping pass. |
| Resident recurrent state across prompt tokens | Already present in native GDN's register state and token loop. |
| Direct indexed recurrent-state consumption | [Installed guarded in-place GDN](qwen-moe-gdn-fused.md) deletes 4.027 GB of logical intermediate state traffic per stable 32-stream generated step, preserves complete heads/state and raises paired full-model aggregate rate 217.44 → 258.02 tokens/s. Dynamic split rows and prompt chunks retain the gathered map. |
| Evaluate recurrent gates once per token/head | Enabled activated-gate exp precomputation on gfx1151 and added raw-gate precomputation at 32 or more prompt tokens. The measured Qwen path uses activated gates. `GGML_CUDA_GDN_PRECOMPUTE_GATES=0` is the same-binary control. Single-token decode is unchanged. |
| GDN row-reduction and lane-layout changes | Native recurrence uses a different transposed ownership/reduction layout. A literal port would change the accumulation map, not just remove redundant work. |
| One-row decode attention and grouped-query key reuse | Native FlashAttention already launches the one-row, eight-query-head Qwen shape against shared KV. It has no Bonsai eight-row clamping waste. |
| Separate attention bodies and wide prefill | Native tile, vector and matrix attention are already separate template dispatches. |
| Prompt output head only for requested rows | The graph gathers requested hidden rows before the vocabulary projection. Already present. |
| Window-limited drafter attention work | Specific to the DFlash sliding-window drafter. The Qwen MTP head has a different attention contract. |
| Target-verified speculative decoding | A separately acquired, source-matched MTP head works with the unchanged target image. Its Q4 form reaches 70–74 tokens/s on the measured list prompt, but longer greedy comparisons diverge at near ties. Explicit opt-in only. |
| Four-bit drafter weights | Applicable to the separate MTP head without changing target weights; compare acceptance and verified output, not draft-state equality. |
| Lossy sub-bit shared expert representations | The tested shared bases, gains and clipping studies have not earned runtime adoption. Their documented failures are not speedups to enable. |
| Seven-exp raw-top-eight router | [Forty-layer actual-producer replay](qwen-moe-router-seven-exp.md) matches 5,120 route sets with small local weighted-sum change, but it is a different FP32 map. The complete traced top-k dispatch budget is just 0.296719 ms/token of 20.397993 ms/token profiled device duration; no native port or whole-model gain. |

The native source is based on Prism llama.cpp `1a07bfa5f`, with task checkouts
kept separate from the reference runtime and Bonsai's tokenizer dependency.
`tools/qwen-moe/runtime.py` builds against the NixOS-owned HIP environment,
packages relocatable binaries with source custody, and selects an artifact only
with a matching acceptance receipt. It does not replace the resident Bonsai
service.

## Recorded native panels

At 1024 occupied tokens, 64 generated tokens and three repetitions, the original
runtime measured 52.1299 and 52.1505 tokens/s. The eight-wave Q8/two-wave Q6
candidate measured 51.6350 and 51.5708 in the alternating panels. The loss is
consistent; this policy is not an accepted speedup.

The narrower two-wave Q8/single-wave Q6 candidate produced warm samples around
52.7 to 53.1 tokens/s, but its complete panels include 47.7 and 44.4 samples while
the controls stayed near 52. These panels do not establish an overall gain.
The wrapper recorded different clocks and package load, which remain in the
raw receipts under `../../data/qwen-moe/benchmarks/`.

For the GDN gate port, the 465-token real-text prompt and 32 generated tokens
produced exactly the same 7,946,240 full-vocabulary float values with the gate
precomputation disabled and enabled. Raw `.f32`, token IDs and wrapper logs
are in `data/qwen-moe/acceptance/native-ports/`. The acceptance driver is
`tools/qwen-moe/accept.cpp`. Its decode-and-dump time includes output writing
and is not a throughput benchmark. Prompt timing panels are separate and do
not establish a useful full-model speedup from this small phase alone.

## Installed runtime

The current on-demand runtime is native revision `7861dc746ed49c6bec1aa2c4f4b8f25b1bf59674`, selected with [guarded GDN full-head/state acceptance](qwen-moe-gdn-fused.md) and retaining `3f552c2`'s [installed short-prompt J16 policy](qwen-moe-mmq-width16.md#installed-2432-row-selection-september-23). Its stable complete 32-stream generated GDN defaults to the no-copy indexed cache writer and gains **18.66% aggregate full-model rate** on the matched short-context panel; `GGML_CUDA_GDN_FUSED_ROWS=0` restores gathered execution. A reordered 31+1 split, ordinary single-stream decode and prompt ingestion keep the original gathered path. The selected image and output arithmetic contract are unchanged, with exact installed 465-token logits and all generated/swap heads and state as recorded in that report. It retains `f60e4fb`'s J=64 policy at 65–256 routed prompt rows: [the original installed panel](qwen-moe-mmq-width.md) raised matched 512-token ingestion from 872.92 to 1138.18 tokens/s, with 7,946,240 reference FP32 values and 32 greedy IDs matching. The preceding width revision selects J=16 for 24–32 routed rows and retains original J=32 for 9–23; its alternating complete-natural-prompt panel improves warm 24/32-row rates 4.51%/4.43% with all sixty requested vocabulary rows bit-identical. Eight steps of 32 independent teacher-forced streams preserve their requested logit bits. A subsequent [actual generated 32-stream panel](qwen-moe-generated-stream-j16.md) feeds back greedy tokens and computes every head row: all 1,271,398,400 compared bits per run match across width arms, while four paired full-model aggregate generation comparisons favor J16 by +0.62–7.69% under recorded host contention. Single-row plain decode remains on unchanged MMVQ and about 52 tokens/s from the previous acceptance; this revision does not claim a newly measured single-stream decode gain. `GGML_CUDA_MMQ_EXPERT_J=0` selects the global-width control. The new installed 465-token/32-greedy-step acceptance agrees bitwise with the original reference, and `bonsai-halo.service` is active.

The original gate-only selected runtime was native revision
`b4c67ced9f6bac6b1661b25714946d999374e06f`, retained under
`../../data/qwen-moe/runtime/b4c67ced9f6bac6b1661b25714946d999374e06f`. Its only source difference from
`1a07bfa5f` is GDN gate precomputation. The width candidates were reverted.
The installed artifact reproduced all 7,946,240 logit bits and 32 greedy token
IDs from the original reference on the 465-token real-text prompt.
`../../data/qwen-moe/acceptance/selected/receipt.json` owns acceptance,
file hashes and benchmark paths.

The packaged runtime measured 52.2933 tokens/s over three 96-token decode runs
at occupied depth 1024, and 879.9334 tokens/s over four 512-token prompt runs
with ubatch 256. These rates are near baseline, not a large full-model gain.
The GDN on/off device trace measured 33.641841 ms without precomputation and
31.235749 ms with it, including the added precompute launches. That is a 7.15%
phase reduction. The trace used the activated-gate path, not the new raw-gate
path. Raw traces and the summary live in `acceptance/gdn-phase/`.

Run the selected runtime from the Bonsai repository:

```sh
python3 tools/qwen-moe/runtime.py run -- --single-turn --no-conversation -p "Explain mixture of experts briefly." -n 128
```

The runner reserves the GPU and restores the resident Bonsai service afterward.
Plain decoding is the default. The target GGUF remains unchanged.

The [Q4 MTP candidate](qwen-moe-mtp.md) is available through `run --mtp`.
It measured 73.90 versus 49.40 tokens/s on the 96-token list prompt, but changed
a greedy choice at token 71. Original reference libraries also diverged at a
near tie, and disabling graph fusion did not remove the difference. The short
32-token compass response matched, but that does not establish longer output
agreement. MTP therefore remains outside the accepted default. A [top-k-only diagnostic](qwen-moe-graph-dispatch.md#top-k-only-intervention) makes 40 full-vocabulary width-replay rows bit-identical without disabling other CUDA fusions. Two complete three-repeat JSONL samples survived the attended timeout and were missed in the first receipt. On the same candidate binary at depth 1024 and 64 generated tokens, unfused top-k takes 5.81% more complete-model decode time averaged over the three samples, or 3.86% more over the last two. The arms were sequential and lack clock records, so the size of the loss needs an interleaved panel before adoption decisions. It is not a speedup or selected numerical map. The [recovery receipt](../../data/qwen-moe/acceptance/batch-layer-probe/speed-recovery.json) retains the raw hashes. Preserve fused top-k while reconciling its width behavior, and replay the MTP token-71 fork before changing the default. The [bounded graph-lifetime candidate](qwen-moe-router-lifetime.md) safely enables fused two-row top-k without a graph cut and passes the five-step width panel plus installed 465-token reference, but the full 96-token replay still forks at token 71 for every four-row offset; it is not selected.

`runtime.py build --source CHECKOUT` uses the declared NixOS HIP environment.
`package --source CHECKOUT` saves relocatable binaries, their hashes, a source
bundle and a source ref in `data/qwen-moe/runtime-source.git`. `select REV
--acceptance RECEIPT` verifies the artifacts and atomically selects them.
Build products do not depend on a retained task checkout. The source repository
and bundle preserve the rejected width experiments as ordinary Git history.

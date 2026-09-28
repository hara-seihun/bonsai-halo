# Comparing speculative proposals on Bonsai 2

the project owner requested a DSpark comparison on September 23, 2026, following Kelana's larger-target copy-tree experiment. This study uses the installed Ternary Bonsai 2 27B target and Radeon 8060S, rather than extrapolating small-model or datacenter numbers.

## Which baseline exists

The installed production drafter is `incoai/Qwen3.8-27B-DFlash2`, executed with the existing Q4 native image. It is not DSpark. The [DSpark paper](https://arxiv.org/html/2607.05147) and [DeepSpec releases](https://github.com/deepseek-ai/DeepSpec/blob/main/README.md) report other targets and hardware. Paper acceptance-length gains and throughput at per-user service-level constraints are not a Bonsai single-stream speed comparison.

[Prism's speculative guide](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/SPECULATIVE.md) says Bonsai 2 has no official paired DSpark checkpoint. The previous Ternary Bonsai generation's DSpark belongs to a different target generation. [RadixArk/Qwen3.8-27B-DSpark](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark) is a possible cross-target comparator, trained for the Qwen target rather than its ternary Bonsai finetune. Its configuration and safetensors header were inspected without downloading the full weights.

RadixArk shares the five target taps, width 5120, five draft layers, 32 query heads, eight KV heads and rank 256 with the installed DFlash2. The computations differ:

- DSpark uses full attention and YaRN rather than DFlash2's sliding window and dynamic convolutions.
- DSpark adds a full-vocabulary Markov bias; the current DFlash2 uses a hidden-conditioned pair term over a base-logit top-16 shortlist.
- DSpark includes a confidence head and a verification-capacity allocation policy. The latter addresses a multi-request resource allocation problem, not automatically a single-stream gain.
- DSpark's seven draft rows include the anchor-row prediction. Current DFlash2 consumes rows 1 through 7 of an eight-row block. The [checkpoint maintainer's alignment correction](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark/discussions/2) confirms this distinction for these weights.

Renaming tensors into `Engine::load_dflash` would not implement DSpark. A faithful comparator needs the different attention/position logic, full-vocabulary Markov head and exact row alignment, followed by reference-logit checks and target-state rejection tests. None of the measurements below is labeled a DSpark result.

## Native proposal experiment

[`tools/speculative_compare.cpp`](../tools/speculative_compare.cpp) runs a single loaded target and Q4 DFlash2 image. Every case starts from a full reset. The existing ten `bench/drafter-prompts.txt` prompts cover prose, code and reasoning. The selected prompts are known from earlier native studies, not a fresh held quality benchmark.

All arms consume every emitted token through the target, retain the next target decision and stop on the target's end token or the requested limit. Decode time includes proposal selection, target passes, rejection and state bookkeeping. Request time adds prefill and the first draft. These rates deliberately avoid the production CLI's first-emission timing boundary, so compare arms inside this tool rather than mixing its rates with README headline numbers.

- `serial` verifies one token at a time and disables drafter ingestion.
- `df8` uses the installed full eight-row verification block.
- `copy` searches only visible prompt/output history for a matching suffix plus retained target token. A match of at least two tokens supplies up to seven following tokens. The target verifies them normally.
- `hybrid` replaces DFlash with copying only after a four-token suffix match. It still ingests accepted target features into the drafter cache, so a later neural draft has the correct history. The first prefill draft remains paid.
- `recycle` keeps the last neural block's top-16 candidate IDs, base scores and projected hidden rows. After a rejection it recomputes the Markov selector for remaining absolute positions using the corrected predecessor. It ingests accepted target features but skips a new full draft when at least two cached proposal positions remain. These hidden rows describe the previous mask block, not the repaired prefix; the target must verify them.

[`speculative_recycle.hip`](../tools/speculative_recycle.hip) assigns one workgroup to the saved-table selector and keeps the original selector's arithmetic and tie rule. Before each neural arm, the driver compares its full saved-table selection to the original kernel's draft IDs. The arrays survive target verification and ingest-only drafter execution. A new full draft replaces them.

## Initial result and correctness finding

The initial six-case panel does not establish a win over DFlash2. Pure copy proposals are weak on these short instruction prompts. The four-case recycled-map panel is slower on every case despite avoiding many full drafts. It creates more target passes; a saved draft does not pay for an extra target weight traversal. The raw arrivals and all output mismatches remain in the data owner.

More importantly, the native serial baseline itself is not repeatable near a logit tie. On the multiplication prompt, six serial-only runs after a full reset produce four different output paths. At generated offset 35 all six have prefix hash `3464432181866694891`, but the winner alternates between IDs 1817 and 21261. Observed top-two margins range from 0.000586 to 0.058289. Disabling wide prefill does not remove the effect.

`Engine::reset()` also failed to clear host-side pending GDN metadata, unlike `reset_seq()`. This independent defect is repaired for all allocated sequence slots. The serial repeat variation persists after that fix, so it is not attributed to the reset bug.

The persistent target's split matvec adds two floating partials to a residual through cross-workgroup atomics. A comment claimed a grid below the output tile count forces part 0 to complete before part 1. Dynamic work allocation does not provide that ordering: a fast workgroup can claim part 1 while the workgroup computing part 0 is still running. [Earlier native diagnosis](gdn-sliced-gates.md) recorded the same issue. The repair assigns all K parts of an output tile to one workgroup and applies them in part order. Each part keeps its existing block and wave reductions. The target's persistent matvec and sliced FFN opt into this rule, including the WMMA path. The drafter retains its existing arithmetic. There is no parameter-structure ABI change.

This defines the target residual fold as `(residual + part0) + part1`. It matches one previously possible arrival order, not every historical race outcome. The change fixes execution nondeterminism; it does not change the mathematical weights or claim bit identity to an arbitrary earlier run.

## Ordered-target acceptance and result

With ordered parts, the six-run narrow-prefill reproducer has one output path and exactly identical top-two logits at the tested prefix. The margin is 0.0144634247 on every run. The complete ten-prompt comparison then produces 1,212 output tokens per arm, with the short prose answer stopping at its end token. Every speculative arm matches all ten serial token paths and all ten final next decisions. The standalone saved-table selector also agrees with the original full-block selector before every neural arm.

| Method | Decode tokens/s | Including prefill | Target calls |
| --- | ---: | ---: | ---: |
| Serial | 32.91 | 32.12 | 1,212 |
| DFlash2, eight rows | 80.92 | 76.11 | 322 |
| DFlash2 with copy arbitration | 81.68 | 76.90 | 321 |
| Repaired candidate-map suffix | 66.90 | 63.52 | 448 |

The copy arm selects only eight copied blocks in 321 target calls. Its initial 0.94% advantage does not repeat: running all ten cases again in opposite arm order gives 83.25 tokens/s for DFlash2 and 81.24 for the hybrid, a 2.42% loss. All twenty repeated paths and final decisions still match their primary serial references. No hybrid speedup is established.

Map recycling cuts mean drafting cost from 6.93 to 3.02 milliseconds per target call, but increases target calls from 322 to 448. Total decoding slows by 17.33%. Saving arithmetic in the proposal producer is insufficient when it reduces useful progress per expensive target pass.

The package was shared with other research. Wrapper logs retain clock, power and host activity; the primary stages reported 21–74% package-power limitation. Drafter floating-point split reductions also vary their candidate choices across runs. These are reasons to keep the matched repeat and its negative outcome, not to select a favorable headline.

**This round does not beat DSpark, and does not establish a win over the installed DFlash2 baseline.** It produces a native candidate-reuse experiment and repairs a target determinism defect that otherwise invalidated exact-output comparisons. The next useful competition requires a faithful RadixArk DSpark port or a paired Bonsai-2 DSpark, plus stronger proposals that retain full-block progress. Increasing copy frequency or recycling more stale rows is not justified by these results.

## Reproduction and custody

Build `make -j2 tools/speculative_compare`. Ordinary builds use the maintained make dependencies. This investigation reused unchanged native object files from the canonical checkout, retained their hashes, and compiled each changed translation unit. No additional model or compute was provisioned.

Use absolute command and prompt paths because the admission wrapper runs the child from its own working directory:

```sh
ROOT=.
"$ROOT/tools/run-batch-compare" --runtime-max 85s --memory-gib 18 --host-reserve-gib 4 \
  --exec "$ROOT/tools/speculative_compare" \
  --prompts "$ROOT/bench/drafter-prompts.txt" --offset 0 --count 2 \
  --tokens 128 --methods serial,df8,hybrid,copy
```

[`tools/speculative_panel.py`](../tools/speculative_panel.py) wraps these commands with content-addressed source snapshots, binary hashes, logs, exit status and immutable tags. Run it with `--tag NAME -- --offset ...`; its default child paths are absolute. [`tools/speculative_summary.py`](../tools/speculative_summary.py) regenerates the complete ten-case aggregate and repeat comparison in the data owner.

For the repeated same-prefix diagnostic, use case offset 2, count 1, six rounds, 64 tokens, `--methods serial --diagnostic-offset 35`. This readback changes timing and is not a speed benchmark. `--prefill-mode 0` isolates the narrow prefill route; the default is 20. Repeated rounds reverse method order.

Raw JSONL, wrapper logs, measured source snapshots, object hashes and the arrival-order benchmark binary live in [../../data/bonsai2/speculative-compare](../../data/bonsai2/speculative-compare/README.md). Preserve negative results and mismatches. Do not call a candidate faster and lossless by discarding divergent cases.

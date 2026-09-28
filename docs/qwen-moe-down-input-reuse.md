# Exact-code reuse of Qwen's routed down inputs on actual producers

The forty-layer post-SwiGLU capture settles one narrow MoE sharing question: **does a routed expert's down projection see the same prepared Q8_1 input block as another co-routed expert or as an earlier visit to that expert?** On each of two disjoint actual selected-GGUF 64-token prompt prefixes, **none** of the 327,680 32-coordinate down-input blocks repeats *within its layer* even as a code-only signed orbit with K position and expert ignored. The weaker same-token/co-routed and same-expert/same-K reuse proposals therefore also have zero hits before dictionary costs. The existing gate/up input already is shared across the eight experts; this result concerns their distinct *post-SwiGLU* down inputs.

| Captured split, 40 layers × 64 tokens × 8 experts × 16 blocks | Train | Held |
| --- | ---: | ---: |
| Prepared down-input blocks | 327,680 | 327,680 |
| Revisited (layer, expert) assignments after their first token | 16,493 | 16,704 |
| Repeated 32 signed Q8_1 codes even at **any** K, expert or token within a layer | 0 | 0 |
| Repeated codes up to global sign at any K, expert or token within a layer | 0 | 0 |
| Same-token, same-K cross-expert code matches | 0 | 0 |
| Same-expert, same-K across-token code matches | 0 | 0 |

The CPU encoder follows the installed **MMQ down** `quantize_mmq_q8_1` code rule: for each 32-float block, `d_inv=127/max(abs(x))`, then signed round-away-from-zero of `x*d_inv` into 32 int8 codes. The 37 Q5_K layers use DS4 (FP16 scale and FP16 original-input sum per 32 values); layers 34, 38 and 39 are Q6_K and use D4 (FP16 scale but no input sum). Both packed MMQ layouts cost 36 bytes per 32 input values. This reciprocal multiply matters: ordinary MMVQ instead divides by `max(abs(x))/127`, and that need not produce identical rounded codes. Code equality is only a *necessary*, not sufficient, condition for a reusable native operand under either layout. Granting free code lookup, free sign negation and even free permutation of K position cannot find a hit in either split. This count applies to the declared CPU code grammar, not a GPU bitwise quantizer replay; the real producer arrays come from the callback path, which cuts graph topology. It does not claim a bound for generated tokens, changed codes or learned common input coordinates.

The Q8_1 operand is 36 bytes per block. All down preparations combined are `40 × 8 × 16 × 36 = 184,320` bytes/token, merely **0.007019%** of the conditional 2,626,187,904-byte complete-model one-read weight stream. That is an optimistic *byte* ceiling even if every existing down activation operand could be made free, not a bound on quantizer instructions, routing, cached traffic or wall time. It cannot justify an equality-keyed down-input cache or a co-routed shared-input assumption on these observations. The selected native grouped-expert dispatch already shares the **gate/up** operand; the down branches do not have the same producer. A useful next engine question is unprofiled ordinary Q8/expert time with fixed-clock per-layer device markers and complete heads. A changed shared down-input coordinate instead requires new packed expert maps, a paid forty-layer image trained on broader routed producers and disjoint held complete-model loss. Local code-collision absence does not obstruct those learned representations.

## Custody and reproduction

[The receipt](../../data/qwen-moe/all-down-zero/input-reuse-receipt.json) binds every one of the 80 hidden/route files to the prior [capture receipt](../../data/qwen-moe/all-down-zero/README.md), and hashes its own source, installed quantizer, selected runtime, traffic inventory and model pin. Each split and layer retains the counters separately. From Bonsai:

```sh
python3 tools/qwen-moe/down_input_reuse.py \
  --output ../../data/qwen-moe/all-down-zero/input-reuse-receipt.json
```

CPU only. No model image, GPU reservation, runtime executable, numerical default or resident service changed. This is a finite exact-code reuse negative, not an approximate representation quality experiment or an inference-speed measurement.

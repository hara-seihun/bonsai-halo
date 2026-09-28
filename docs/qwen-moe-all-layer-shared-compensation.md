# Can Qwen's unchanged shared expert absorb one routed expert across the full stack?

**No useful low-error skip emerges on the two disjoint actual 64-token prompt captures.** The selected Qwen3.6 GGUF computes one shared expert per token at all forty layers. We decoded its actual Q8_0 gate, up and down matrices and F32 sigmoid gate at every layer, evaluated that branch on the captured post-attention input, and granted a free hindsight selector and an unrestricted, token-specific real scalar adjustment of the resulting shared output. The eight routed score-weighted down outputs come from the selected native callback. This is a strictly stronger *local* oracle than an executable gate chosen without the omitted output.

| 2,560 layer-token observations | Train | Held |
| --- | ---: | ---: |
| Best unchanged single routed omission, RMS relative to **complete FFN output** | .054194 | .057690 |
| Omission + free shared-output gain, same RMS | .053784 | .057208 |
| Decisions at most 1% local error, omission / shared gain | 9 / 9 | 9 / 9 |
| Decisions at most 5% local error, omission / shared gain | 269 / 273 | 310 / 312 |
| Held conditional complete one-read byte ceiling at 1%, omission / gain | — | .010343% / .010343% |
| Held conditional complete one-read byte ceiling at 5%, omission / gain | — | .358906% / .361167% |

The 5%-tolerance shared adjustment adds just **two** held layer-token decisions and **.002262 percentage points** of conditional complete one-read bytes over omitting the same easiest expert without a shared adjustment. Median held best-one relative error is .094741 after compensation against .094778 before it; the median per-layer absolute cosine of the oracle-selected routed output and the shared output is .01934. The available shared branch is not a common direction into which missing routed outputs collapse. These complete-output denominators differ from the earlier routed-sum-only omission studies; compare the two arms *within* this panel, not their counts against a different observation.

For a routed contribution `v_i = score_i * down_i`, shared output `s`, and complete local FFN output `y = s + sum_i v_i`, the squared error from removing `v_i` and changing the shared coefficient by `a` is `||v_i - a s||²`. Its unrestricted real minimum occurs at `a = (v_i·s)/(s·s)`, with residual `||v_i||² - (v_i·s)²/(s·s)`. We evaluate all eight choices and choose the least residual per token. This is an exact finite-vector FP64 projection identity; the script independently checks the chosen 2,048-vector residual. Restricting the final shared coefficient to nonnegative (`a >= -1`) does not change any displayed threshold or RMS. The reference shared computation uses the installed quantized weights but offline FP64 operations, not native FP32 reduction order.

The conditional byte ceiling charges one selected expert's actual per-layer bank image when an oracle decision meets tolerance, divided by `64 * 2,626,187,904` modeled complete one-read bytes for a 64-token sample. The 1% held oracle selects just 9/2,560 layer-tokens; an unconditional one-expert skip in every layer would at most save 2.91067% under the same optimistic model, but it incurs the displayed .057208 aggregate local error. No predictor, extra gain dot/reduction, native synchronization, physical traffic, FP32 equality, complete-model language quality or serving time is included. This is a bound **only** on omitting one of the unchanged captured routed vectors while scaling the unchanged shared output, not on trained changed directions or on a looser downstream observation. It does not assert that a 5% layer-local error is acceptable for language generation.

[Forty hashed per-layer receipts and the aggregate](../../data/qwen-moe/all-layer-shared-compensation/receipt.json) retain every token's best error, selected ID and gain; inputs are verified against the [native capture receipt](../../data/qwen-moe/all-producers/receipt.json). The GGUF's full SHA-256 is `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`, verified against the capture, and the receipt hashes the installed decoder and model provenance. Run from Bonsai with four CPU BLAS threads; each bounded range writes independent results and the final call binds them:

```sh
for start in 0 20; do
  OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python3 tools/qwen-moe/all_layer_shared_compensation.py --start "$start" --stop "$((start+20))"
done
python3 tools/qwen-moe/all_layer_shared_compensation.py --aggregate
```

No GPU, model, installed runtime or resident service was changed. **Next question:** do not implement a native shared-gate-only skip. A learned, changed input-dependent expert direction needs broader real routed producers, a paid complete forty-layer image and held whole-model language loss; independently diagnose ordinary unprofiled Q8/expert device time rather than trying to recover these vanishing oracle bytes.

# A free output-gain oracle does not turn short routed captures into an expert image

The selected Qwen3.6-35B-A3B GGUF's complete forty-layer callback captures contain two disjoint 64-token prompt prefixes. For each layer and expert, this experiment grants a dictionary of **all** its train down outputs, then asks whether a held routed down output is close to **any signed real multiple** of one stored vector. The oracle reads the *true held output* to select the best same-expert direction and coefficient. An independent selector chooses the nearest train **producer input** instead, but still receives the true-output optimal coefficient for that chosen direction. Both keep train-unseen expert outputs **exact**. Thus this generously favors an output-ray cache; it neither learns nor charges a deployable gain predictor.

We fold the eight score-weighted contributions in FP64 at each layer/token and measure aggregate local RMS relative to the captured eight-way routed sum. The [prior exact-output nearest-prototype oracle](qwen-moe-producer-coverage.md) used the same captures and also granted unseen outputs exact; it reached .85124 RMS. It could only choose stored output vectors, not scale them.

| Seen-slot replacement (unseen slots exact) | Pooled held routed-sum RMS / original |
| --- | ---: |
| Best stored same-expert output, no gain (prior oracle) | .85124 |
| Best signed-gain stored direction, per-slot hindsight | **.77197** |
| Nearest train producer direction, hindsight gain | **.79716** |
| Per-slot hindsight directions, then jointly optimal eight coefficients | **.75863** |
| Nearest-producer directions, then jointly optimal coefficients | **.78604** |

Of 20,480 held routed assignments, **14,648** have an example of that same layer/expert in train; 5,832 are held exact in these arms. The per-slot hindsight ray alone still misses **.92225 RMS of the energy in seen held down outputs**; nearest-input rays miss .94220. Its per-layer scored-sum error ranges .64398–.94284 (median .82279); none of the forty layers comes close to .1. Joint fitting gains against the complete sum improves .77197 only to .75863. Those eight gains must be known at each new token. Merely recognizing a revisited expert or permitting a scalar correction to its cached output is not the missing mechanism for these captures.

This is a **finite one-stored-direction-per-slot grammar**. Independent per-slot direction choices optimize individual squared errors, not the routed sum. The joint-gain arm improves the complete sum for those choices but does **not** optimize the combinatorial selection of directions jointly; consequently .75863 is *not* an impossibility bound for all combinations of cached vectors. Nor is it a lower bound on larger dictionaries, input-dependent trained maps or changed packed expert weights. The earlier [full train-slot span](qwen-moe-output-subspace.md) still loses .70492 pooled held RMS at rank 512, an even more generous *global* free coefficient arm. None of these local scores is complete-model loss or native FP32 equality.

As a raw FP16 direction bank, saving every train output would require **20,480 × 2,048 × 2 = 83,886,080 bytes** before IDs, indexing and the separately payable gains; this is not a complete Qwen image, and a CPU hindsight lookup does not substitute for an expert's native Q4/Q5 weight read. Rather than another scalar fit to this short capture, obtain diverse actual routed producers, train **input-dependent changed directions/codes** as a paid forty-layer image, select against disjoint complete-model language loss, and price a direct packed consumer. Independent unprofiled Q8/expert physical-time diagnosis remains a larger engine question.

## Reproduction and custody

`tools/qwen-moe/route_output_geometry.py` hashes all 320 train/held ID, score, down and producer arrays and checks them against the [native complete-capture receipt](../../data/qwen-moe/all-producers/receipt.json); its own source hash and original receipt hash are in each of the [four ten-layer CPU receipts](../../data/qwen-moe/all-producers/route-output-0-9.json) (`route-output-0-9.json`, `10-19.json`, `20-29.json`, `30-39.json`). The receipt retains each layer's numerator, denominator, seen count and gain quantiles. From the Bonsai root, with NumPy:

```sh
for n in 0 10 20 30; do
  OPENBLAS_NUM_THREADS=1 python3 tools/qwen-moe/route_output_geometry.py \
    ../../data/qwen-moe/all-producers --first "$n" --count 10 \
    --output "../../data/qwen-moe/all-producers/route-output-$n-$((n+9)).json"
done
```

This was CPU-only; no GPU reservation, selected runtime, model, executable, service or numerical map changed.

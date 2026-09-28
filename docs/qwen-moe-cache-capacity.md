# Arbitrary-granularity cache cannot recover the generated-route retention gap

The [full-image cache study](qwen-moe-generated-cache.md) capped a 32 MiB cache at 1.293% of the modeled complete Qwen serial-decode one-read stream, but left cache lines and subimages outside its bound. Removing image indivisibility **tightens**, rather than opens, the capacity result: a cold 48-step run can save at most **1.25107%** of that complete stream by retaining *any portions of the unchanged expert image* across token boundaries. This is a semantic/traffic bound in a declared family, not a measured speedup or a bound on changed packed representations.

## Domain, observation and proof

One causal token step visits 40 layers in order, selects eight **distinct** of 256 routed experts in each layer, and demands each selected expert's unchanged packed gate/up/down bytes once. Images of different `(layer, expert)` occupy disjoint addresses. The observation is the same decoded model computation; the hypothetical cache does not change arithmetic, routing or outputs. The comparison streams one read per demanded expert byte, 611,516,416 expert bytes and 2,626,187,904 complete modeled weight bytes per token. Nonexpert reads, activation traffic and all replacement/metadata work bypass the cache for free. No physical DRAM transaction count is assumed.

Grant a fully associative 33,554,432-byte cache, arbitrary byte-sized subdivisions (hence any cache-line or subimage partition), free foreknowledge, bypass and replacement. Let `D_t` be the set of expert-byte addresses demanded by token `t` and `C_t` the addresses in the cache **at the start** of that token. Each address in `D_t` occurs at most once in the step. A demand absent from `C_t` cannot become a free *reuse* later in the same step: that byte has no second demand. Loading or prefetching it still transfers that byte and is charged. Therefore the bytes avoided in step `t` are at most `|D_t ∩ C_t| ≤ |C_t| ≤ 33,554,432`. Starting cold, step zero avoids none. This argument does not require an image-sized slot, a route predictor, a particular replacement policy or a cache-line alignment. It also applies to any prior-token age, not just adjacent-token recurrence.

For 48 steps the maximum is `47 × 33,554,432 = 1,577,058,304` bytes out of `48 × 2,626,187,904 = 126,057,019,392`, or **1.251067%**. At steady state the ceiling is **1.277686%** of complete one-read bytes, or **5.487086%** of expert bytes alone. The older full-image 1.292912% cold ceiling used `17 × maximum-image-size` and intentionally credited **34,676,736 bytes per step to a cache holding 33,554,432**; it was valid but loose. Its realizable 16-image clairvoyant schedule already retains 1,429,209,088 bytes on held (1.1338% of complete bytes), leaving at most **147,849,216 bytes over 48 steps**, or **0.1173 percentage points of complete one-read traffic**, for an ideal subimage policy beyond that optimistic whole-image schedule. This is a comparison of two optimistic traffic models, not an incremental gain over installed runtime.

The source/inventory/capture/model-hashed [CPU receipt](../../data/qwen-moe/generated-routes/cache-capacity.json) checks all 40 layers on each of the two existing generated panels. Each adjacent-token image intersection contains at least 86,081,536 bytes on train and 70,459,392 on held: the ceiling is set by **capacity**, not absence of recurring expert bytes. These observed top-k arrays come from a callback that cuts graph fusion; the universal inequality does not depend on their identity. The receipt SHA-256 is `cb68bef513a710662b455ed5ca59f0e7c89651620f5b075cecc4d72a04321c0f`.

```sh
python3 tools/qwen-moe/cache_capacity_bound.py \
  ../../data/qwen-moe/generated-routes \
  ../../data/qwen-moe/traffic.json \
  --output ../../data/qwen-moe/generated-routes/cache-capacity.json
```

## What this changes

**Do not pursue a 32 MiB subimage retention policy solely to exploit the 35–40% same-layer generated-route recurrence.** In the unchanged one-read serial schedule, even perfectly chosen byte fragments cannot recover the ~9.26% free forty-layer simultaneous-retention scenario. The bound does **not** cover actual GPU cache transaction behavior or within-token rereads; these need physical counters or a native unprofiled intervention. It does not cover changing the representation so that expert weights share physical bytes, fusing multiple token steps/speculative work, batching independent streams (where one expert image may serve several tokens in a step), or a direct packed consumer that reads fewer bytes. Those are independent useful questions. The immediate native priority is the ordinary Q8/Q4-Q5 consumer's unprofiled device cost and actual memory traffic, not a route-based cross-token retention mechanism.

No GPU work, model image, selected runtime, serving default or service state changed.

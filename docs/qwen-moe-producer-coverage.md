# Short actual-route captures cannot support a frozen-output MoE replacement

The existing complete forty-layer Qwen3.6-35B-A3B callback captures contain 64 train and 64 disjoint held prompt tokens, each with eight selected routed experts, post-attention producers, normalized scores and all selected down outputs. This panel asks a different question from omitted unchanged experts or a low-rank basis: **how much of the held routed computation is even represented by training examples of the same (layer, expert), and could the observed output vectors serve as a cheap replacement bank?** The selected GGUF and native runtime are unchanged.

| Held support by train assignment count for the same layer/expert | Held assignments / 20,480 | Held normalized score mass / 2,560 |
| --- | ---: | ---: |
| Zero | **5,832 (28.48%)** | **680.169 (26.57%)** |
| At most one | 7,808 (38.13%) | 919.077 (35.90%) |
| At most two | **9,507 (46.42%)** | **1,122.562 (43.85%)** |
| At most four | 11,915 (58.18%) | 1,424.003 (55.63%) |
| At most eight | 15,538 (75.87%) | 1,897.606 (74.13%) |

Across the 10,240 possible layer/expert pairs, train sees only **3,987**. Held uses 3,776 pairs, including **1,045 never selected in train**. The train examples are not uniformly distributed: 1,214 observed pairs have one sample, 638 have two, while 6,253 have none. A local fit reporting error only on the trained experts silently excludes over a quarter of the held score mass. This does not prove a larger learned model cannot generalize: the original weights and shared cross-expert parameters may carry information about unseen pairs. It does tell us the 64-token captures are an inadequate *standalone* training set for separately learned expert directions.

For a concrete finite codebook grammar, store each train-selected expert's **unweighted 2,048-float down output** and approximate each held output by a train output from the *same layer/expert*. The most favorable selector is an offline oracle that sees the true held output and chooses its Euclidean-nearest stored vector. It costs no selection work in this comparison. Score the complete eight-way routed sum in FP64 with the actual held scores, using the actual held down vectors as reference:

| Replacement for held assignments | Relative aggregate local routed-sum RMS |
| --- | ---: |
| Original down vectors everywhere | 0 |
| Exact observed-expert outputs, zero for train-unseen experts | **0.37343** |
| Free true-output nearest stored vector for seen experts, exact original output for unseen experts | **0.85124** |
| Free true-output nearest stored vector for seen experts, zero for unseen experts | **0.95955** |
| Train mean per seen expert, zero for unseen experts | **0.96249** |

The nearest selector is deliberately unattainable without already computing the held output. Even with the unseen branches granted exact original computation, replacing the *seen* down outputs by their nearest stored responses incurs .85124 local RMS. A one-prototype-per-expert replacement loses for a reason beyond missing expert IDs: the down vector varies strongly with its input inside an expert. Saving 20,480 train outputs as FP16 would cost **83,886,080 bytes of prototype payload** before IDs, scores, indexing or a direct consumer; these examples are only a diagnostic, not a paid 40-layer runnable image. The FP64 offline fold does not reproduce the native FP32 multiplication/addition contract. Local RMS is not language loss and none of these rows is an inference-speed result.

This result rules out using these two short captures as an expert-output lookup image, even with a free hindsight nearest neighbor. It also identifies the next *different* representation experiment: collect producer-diverse training tokens until route coverage and per-expert input diversity are adequate, fit **input-dependent changed codes** shared across experts where possible, save the complete paid forty-layer image and compare disjoint complete-model language loss. The full-image ternary pilot shows why a lower local error is only a screening result. Ordinary unprofiled Q8/routed device diagnosis remains independent.

`tools/qwen-moe/producer_coverage.py` verifies all 240 ID/score/down files against the [original complete capture receipt](../../data/qwen-moe/all-producers/receipt.json), checks ID ranges and distinctness, then saves the per-layer support histograms, masses, denominators and five replacement controls in the [source/capture-hashed CPU receipt](../../data/qwen-moe/all-producers/coverage-receipt.json). Run with `../../data/fish-s2-pro/venv/bin/python tools/qwen-moe/producer_coverage.py ../../data/qwen-moe/all-producers --output ../../data/qwen-moe/all-producers/coverage-receipt.json`. No GPU reservation, runtime installation, model or service state changed.

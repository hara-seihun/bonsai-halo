# Forty-layer static expert-rate allocation: short captures leave a large oracle gap

**Question.** If an expert-aware image replaces selected Q4_K gate/up pairs with Q3_K while keeping every other tensor unchanged, how much of the *conditional one-image-read per routed assignment* stream can a frozen allocation chosen from the 64-token train capture remove on 64 disjoint held tokens? This is a **cost/coverage bound**, not a Q3 image, quality result or runtime speedup. The existing layer-0 Q3 recode [loses .084997 held local routed-sum RMS](../../kelana/research/moe/gateup-q3-recode/README.md); no such recode or language-loss measurement exists across all forty layers. That quality failure cannot be canceled by counting bytes.

There are 40 × 256 possible `(layer,expert)` pairs. Each Q4_K gate/up pair has 2,097,152 weights, or 1,179,648 installed bytes; a same-shape Q3_K pair at 110 rather than 144 bytes per 256 weights would cost 901,120 bytes, a conditional **278,528-byte** saving per chosen pair. All forty installed gate/up banks are Q4_K at this size in the pinned inventory. A fixed budget of `40k` pairs saves `40k × 278,528` **image bytes** whatever their IDs. For a one-row token step, each chosen `(layer,expert)` saves that number *only if selected*; no unchanged-map inference is implied by the different quantization.

| Q3 pairs per layer `k` | Conditional saved complete image | Train-choice held assignments / 20,480 | Held oracle assignments / 20,480 | Train-choice held one-read bytes/token | Held oracle bytes/token |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 89,128,960 | 3,056 | 8,110 | 13,299,712 (0.5064%) | 35,294,720 (1.3440%) |
| 16 | 178,257,920 | 4,875 | 11,405 | 21,216,000 (0.8079%) | 49,634,560 (1.8900%) |
| 32 | 356,515,840 | **7,946** | **15,237** | **34,580,992 (1.3168%)** | **66,311,424 (2.5250%)** |
| 64 | 713,031,680 | 12,103 | 18,944 | 52,672,256 (2.0057%) | 82,444,288 (3.1393%) |
| 128 | 1,426,063,360 | 15,528 | 20,417 | 67,577,856 (2.5732%) | 88,854,784 (3.3834%) |
| 256 | 2,852,126,720 | 20,480 | 20,480 | 89,128,960 (3.3939%) | 89,128,960 (3.3939%) |

The one-read denominator is the pinned **2,626,187,904 bytes/token** including nonexperts and output head; all numbers here grant zero Q3 decoding/dispatch/metadata cost. The images in the second column are *format-capacity calculations*, not built or evaluated complete model images. At `k=32`, choosing the 32 most frequent experts **independently in each layer** on train covers only 52.15% of the held clairvoyant assignment count. A free global allocation of the same 1,280 selected pairs across layers scarcely helps: 7,947 versus 7,946 held hits (global held oracle 15,326). Train sees 3,987 distinct pairs; held sees 3,776 and **5,832 of 20,480 held assignments use train-unseen pairs**. It is specifically the 64-token capture's support, not a missing cross-layer budget optimizer, that limits this frozen-frequency selector.

**Exact finite-family optimum.** For any fixed budget `k` in one layer, the expected/observed one-read saving is `s Σ_e c_e z_e`, with `s=278,528`, integer route counts `c_e`, and `z_e∈{0,1}`, `Σ z_e=k`. Swapping a selected lower-count ID with an unselected higher-count ID cannot decrease this sum, so top-`k` counts attain the maximum; the global variant applies the same exchange across all 10,240 IDs. Ties choose the lower expert ID. Applying this optimum to *held* gives a hindsight upper ceiling for **all static choices of these same-size pair images on these held routes**; applying it to train gives the deployable frozen-frequency choice whose held counts are reported. No claim about minimizing approximation error follows: low-count experts may be most expensive to perturb, and no all-layer Q3 response or held model loss is supplied.

The independently hashed [CPU receipt](../../data/qwen-moe/static-q3-oracle/receipt.json) contains all eighty input array hashes, pinned traffic/capture hashes, full train/held assignment counts, layer allocations and both budget rules. Recompute without GPU:

```sh
python3 tools/qwen-moe/static_q3_oracle.py --output ../../data/qwen-moe/static-q3-oracle/receipt.json
```

This changes the next representation question: **do not choose a forty-layer mixed-rate bank from only 64 train tokens/layer and call its nominal 356.5-MB image reduction a 2.525% inference benefit**. Collect broad real routed producers and scores; construct and pay for an actual changed complete image, select allocation on training/selection text by composed loss, then evaluate disjoint whole-model loss and a direct packed native reader. Grouped prompt MMQ already reuses images across rows, so the one-row byte fractions cannot be transferred to prompt TPS. The callback changes graph topology; neither these prompt routes nor a perfect byte optimizer proves selected no-callback decode behavior. No GPU, model, selected runtime or resident service changed.

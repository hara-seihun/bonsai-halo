# The first long Qwen MTP width fork does not depend on future batch tokens

The [serial-anchor panel](qwen-moe-mtp-anchor.md) matched every FP32 logit on eleven singleton continuation rows, then changed all 248,320 logits on the *first* four-token call's row 12. This intervention asks whether that first result is contaminated by the **values of future tokens in the same causal batch**, or by requesting their vocabulary heads. It does not identify a divergent layer or repair MTP.

The no-callback driver uses the same packaged target source `81326f7`, pinned GGUF and list prompt. It generates sixteen plain tokens, then for each of five arms resets the context, prefills the same prompt, and replays eleven singleton tokens at the same explicit positions and with the same full-head requests. Every one of those eleven rows agrees with its original plain logits bit-for-bit in every arm. Its first width-four call always has token **13** in its first row (the input to row-12 logits); the three later tokens are:

| Arm | Later batch tokens | Requested heads | Row 12 versus singleton | Row 12 versus arm 0 |
| --- | --- | --- | ---: | ---: |
| 0 | 220, 2972, 2014 | all four | 248,320 different bits | control |
| 1 | 13, 13, 13 | all four | 248,320 | **0** |
| 2 | 2014, 2972, 220 | all four | 248,320 | **0** |
| 3 | 16, 16, 16 | all four | 248,320 | **0** |
| 4 | 220, 2972, 2014 | first only | 248,320 | **0** |

Each arm's first-row head has the same 993,280 bytes and SHA-256 `7db28fc0dc14576a1d307e03f7195296aaed5cecfd3ef2f5a3e3db075a188158`. Its maximum absolute difference from the singleton row is **1.1809175** in all arms; the top ID remains 220. The changed future tokens are genuinely different (not just a same-ID replay). Discarding three future output heads does not change this first-row result. The direct conclusion is narrower than a general causal proof: in this fixed state, graph, four-row geometry and five tested suffix/head masks, the row-12 numerical fork is **independent of future token contents and unnecessary future heads**. A content-dependent later-row read or head-work explanation for *this* fork has failed. Width-dependent execution of the first token, its prior state, or shape-dependent graph storage are still possible. Equal heads before row 12 do not prove internal GDN/attention state equal, and this test did not sample it.

[The receipt and raw heads](../../data/qwen-moe/acceptance/mtp-causality/receipt.json) preserve source/executable/runtime/model hashes, every serial-control comparison, the five FP32 heads and the wrapper's service restoration. Run with `tools/run-batch-compare --runtime-max 42s --memory-gib 30 --host-reserve-gib 4 --exec` and the receipt's driver arguments. No callback was registered; the selected runtime, model image, service default and MTP opt-in status did not change. This is a numerical-map diagnosis, not a speed comparison.

**Next experiment:** observe the first differing activation/state on the first width-four call after eleven exact singleton rows and on width-eight row 1, without introducing a callback graph cut. Begin with a side-band disjoint capture at a GDN state boundary and a layer output on the *first row*; an ordinary callback read is not a valid control because it changes the graph. Compare before and after the four-row call, then replay the 96-step near-tie choice and whole-model rates for any repair. Varying more future token IDs at this same boundary is low value after their exact four-arm identity.

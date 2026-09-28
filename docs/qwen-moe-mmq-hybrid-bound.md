# Exact mixed-width bound for routed Qwen MMQ

The selected J=64 prompt route is much faster than J=128, but it still executes padded expert columns. I asked how much of that padding remains if each expert group can choose one of the native J=16,32,...,128 bodies. This is a source-level bound, not another claimed GPU speedup.

For a group with `n` routed assignments and an output matrix with `R` rows, the non-stream-K gfx1151 MMQ body issues

```
ceil(n/J) * ceil(R/I(J)) * I(J) * J
```

full-tile output positions and `ceil(n/J)` logical packed-image passes. Here `I=64` for J=16/32 and `I=128` for J=48..128. Qwen's gate/up and down row counts, 1024 and 2048, are divisible by both I values. Thus the position count reduces to `R * ceil(n/J) * J`. Since every available J is a multiple of 16, it is at least `R * 16 * ceil(n/16)`. Choosing `J=16 * ceil(n/16)` reaches that lower bound **and** reads the group image once. Every captured group contains at most 128 assignments, so this J exists for all train and held prompt groups. The construction attains both minima simultaneously in this single-body-per-group grammar; it does not reconstruct any inactive columns. It preserves the same groups, packed image, activation values and per-output accumulation order if lowered as the existing body with its usual masked store.

`tools/qwen-moe/mmq_hybrid_bound.py` checks this identity against every group-size histogram and separately replays all forty capture files to count the cost of launching different J bodies. The pinned route/model/native hashes and analysis-source hash are in [the raw receipt](../../data/qwen-moe/all-layer-routes/mmq-hybrid-bound.json), SHA-256 `fc675a2d5cb20a34c081b77275cb1382556bf007b4b6a1a2b8e2175328bb2e7b`.

| Held grouping across forty layers | Installed J | Installed gate/up tile positions | Mixed minimum | Logical image passes, installed to mixed | Separate-body launches, installed to mixed |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32-token tiles | 32 | 368,967,680 | 188,809,216, 51.17% | 11,260 to 11,260 | 160 to 279 |
| 64-token tiles | 64 | 507,052,032 | 136,232,960, 26.87% | 7,737 to 7,737 | 80 to 234 |
| Whole 126-token prompt | 64 | 328,138,752 | 97,271,808, 29.64% | 5,007 to 4,958 | 40 to 188 |

Down-projection positions double, with identical ratios. Disjoint train routes yield 50.93%, 26.43% and 28.55% of the respective installed positions. At whole width the minimum does *not* require the 5,937 image passes of a uniform J=16 launch. Four thousand three hundred seventy-nine of 4,958 held groups choose J=16, but large groups use wider bodies to achieve the same minimum position count in one pass. This is an exact joint optimum of the two static counts, rather than a trade of weight passes for empty arithmetic.

It is not a time lower bound. The mixed route asks for 77,664 output row/column workgroups versus 40,056 at installed J=64 on the whole held prompt. Eight distinct J bodies occur across its forty layers. With separate native launches per body, that is 188 rather than 40 gate/up launches, repeated for down, plus partitioning/group dispatch. A single mixed launch would need a new thread/row-geometry design, since the native J=16 and J=64 bodies use 128 and 256 threads and different I widths. Neither free dispatch nor equal physical DRAM bytes follows from the one-pass count. The existing J=16 forced 32-token full-model panel had only a noisy 408.10 to 420.65 tokens/s lead; the installed J=64 gain is measured on a different shape. No new GPU run, model image, executable or service change occurred here.

The useful next native question is whether a **two-width** gate/up/down dispatcher, J=16 for sparse groups and J=64 for the rest, earns more complete prompt and independent-sequence throughput than its additional launch/compaction cost. Measure actual group-size phase time before generalizing to eight bodies. Preserve installed logit bits and check decode, which uses MMVQ and cannot benefit from this tile construction.

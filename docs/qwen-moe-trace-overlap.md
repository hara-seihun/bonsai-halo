# A shared slow regime precedes both Qwen matvec families

The installed target's counter-free depth-1024 trace does not support a diagnosis in which another *traced* kernel overlaps and lengthens the slow wide Q8 or routed Q4 matvecs. Sort each of the eight measured 1,565-dispatch decode spans by device timestamp: all 12,520 dispatches use queue 2, with no overlapping adjacent intervals. The other queue appears only in the excluded setup region. This is a statement about this process's traced dispatches, not other GPU processes or the cause of slow device events.

The more useful signal is immediately before each slow matvec. Every one of the 280 warm width-8192 Q8 calls and 280 warm routed Q4 calls follows the same `quantize_q8_1` activation kernel at grid X=2048, workgroup X=256. Its own event duration and the gap to the matvec change alongside the matvec. Token 0 is cold; tokens 1–3 form the inspection split and tokens 4–7 are held.

| Warm calls | Slow matvec, µs | Ordinary matvec, µs | Preceding activation, slow / ordinary µs | Preceding gap, slow / ordinary µs |
| --- | ---: | ---: | ---: | ---: |
| Q8 width 8192, 50 / 230 | 116.893 | 92.887 | 6.199 / 1.445 | 7.152 / 2.406 |
| Routed Q4 gate/up, 63 / 217 | 113.898 | 50.608 | 6.364 / 1.439 | 7.788 / 2.679 |

Using the existing slow cutoffs, 105 µs for Q8 and 80 µs for Q4, a 3-µs activation-duration flag catches **every** held slow call: Q8 30/30 with two false positives among 130 ordinary calls; Q4 37/37 with two false positives among 123 ordinary calls. The inspection split gives Q8 20/20 with four false positives and Q4 26/26 with two. This flag predicts a regime, not an intervention that can make it disappear. In particular, speeding up the 6-µs activation alone cannot account for a 24-µs Q8 or 63-µs Q4 matvec difference.

The event timeline also illustrates why profiler timestamps must not be used as wall-phase time. Warm spans contain 20.85 ms/token of summed device work and 70.87 ms/token of gaps between dispatches; token 3 alone has a 112.90-ms gap sum. Those gaps include profiler/host effects and cannot be transferred to an unprofiled run. The prior [Q8 episode study](qwen-moe-q8-episodes.md) already found that removing only the slow wide-Q8 calls cannot close its ordinary one-read comparator gap. Together these results argue against porting Q4 or Q8 on the assumption that the striking slow-layer events are an isolated operand defect. Measure unprofiled per-layer intervals, with an activation marker beside each Q8 and Q4 call and matched whole-model wall time. If the long-activation flag and matvec latency still travel together without profiling, examine dispatch/clock/cache behavior before changing packed arithmetic. The ordinary wide-Q8 consumer's remaining gap is a separate, still open question.

[The receipt](../../data/qwen-moe/q8-unprofiled/trace-overlap.json) has every paired event, sorted-span accounting, source/installed-binary/model/trace hashes and train/held tables. Reproduce it without a GPU reservation:

```sh
python3 tools/qwen-moe/trace_overlap.py \
  ../../data/qwen-moe/q8-unprofiled/trace/gpu-host/1886571_kernel_trace.csv \
  --receipt ../../data/qwen-moe/q8-unprofiled/receipt.json \
  --output ../../data/qwen-moe/q8-unprofiled/trace-overlap.json
```

No runtime code, numerical map, installed executable or resident service changed.

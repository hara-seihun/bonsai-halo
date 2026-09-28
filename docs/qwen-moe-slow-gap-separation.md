# The traced expert slow mask follows profiler-sized dispatch gaps

The previously measured Qwen Q4 gate/up slow episodes should **not** be promoted to an unprofiled native optimization budget. In the selected runtime's counter-free depth-1024 HIP trace, every one of **37 held slow Q4 calls** occurs within 64 *dispatches after* a ≥0.5-ms device-timestamp gap; **zero of 112** held calls farther than 64 dispatches are slow. The corresponding inspection split gives 25/26 within 64 and 1/81 beyond. This 64-dispatch gate was chosen on inspection tokens 1–3 and scored unchanged on tokens 4–7. Its held confusion table is 37 slow and 11 ordinary near, versus 0 slow and 112 ordinary far. A tighter eight-dispatch gate explains only 8/37 held slow calls. Thus the slow period extends tens of launches after a profiler-sized pause, rather than merely making the immediately following kernel long.

The eight recurring long gaps are at dispatch ordinals 30, 62, 126, 254, 510, 766, 1022 and 1278, unrelated to a particular Q4 kernel. The prior [gap audit](qwen-moe-launch-gap.md) established that these pauses dominate the ~64-ms **profiled** idle interval and are inconsistent with the separate ~19.6-ms/token unprofiled wall. The [expert-layer analysis](qwen-moe-slow-regime.md) showed that a slow Q4 call often shares slow earlier router and later down projections. The present join supplies a much stronger common coordinate: the long profiler intervals occur at fixed dispatch ordinals each token, so the apparent persistent *layer IDs* are also approximately fixed. For example, held slow gate/up calls at ordinals 68, 144, 264, 536, 768 and 1044 are respectively 6, 18, 10, 26, 2 and 22 dispatches after one of those gaps. This does not establish that a profiler pause causes subsequent long device events; it demonstrates that treating those episodes as intrinsic Q4 weights or a reproducible unprofiled per-layer penalty is unjustified. One held run and one trace instrument remain the observation domain.

| Frozen distance from preceding ≥0.5-ms gap | Inspection slow/near; slow/far | Held slow/near; slow/far | Held near/far calls |
| --- | ---: | ---: | ---: |
| ≤8 dispatches | 6 / 20 | 8 / 29 | 8 / 152 |
| ≤16 dispatches | 9 / 17 | 12 / 25 | 12 / 148 |
| ≤32 dispatches | 18 / 8 | 24 / 13 | 24 / 136 |
| **≤64 dispatches** | **25 / 1** | **37 / 0** | **48 / 112** |

The Q4 threshold is the previously fixed >80 µs, between its observed ordinary and slow bands. The split excludes two synthetic-depth setup heads and cold token 0; each remaining token has 1,565 dispatches, forty routers, Q4 gate/up and routed down calls on queue 2. A gap is ≥500,000 ns from one dispatch's end to the next dispatch's start. Only *preceding* gaps are eligible. This is a profiled event association, not a same-binary unprofiled timing, a causal GPU-clock diagnosis, physical DRAM measurement or whole-model speedup. In particular, subtracting the profiled 0.898-ms/token hypothetical slow-Q4 equalization from wall time would compound two unlike instruments.

**Decision and next experiment:** do not prioritize per-layer slow-Q4 arithmetic based on this traced slow mask. The ordinary Q8/expert latency gap remains a real question, and low-overhead unprofiled device markers with complete heads and matched wall panels are still the needed native discriminator. Measure ordinary Q4/Q5 plus Q8 in the *same* unprofiled panel; a graph/clock regime remains possible, but trace-gap locality is now an explicit control to rule out before attributing the slow tail to an operand or image. No runtime, numerical map, GPU, model or resident service changed in this CPU analysis.

[`slow_gap_separation.py`](../tools/qwen-moe/slow_gap_separation.py) writes the full per-token ordinal/duration/distance table and both frozen panels. [The source/trace/model/binary/installed-receipt-hashed CPU receipt](../../data/qwen-moe/q8-unprofiled/slow-gap-separation.json) retains every observation. Reproduce without GPU reservation:

```sh
python3 tools/qwen-moe/slow_gap_separation.py \
  ../../data/qwen-moe/q8-unprofiled/trace/gpu-host/1886571_kernel_trace.csv \
  --receipt ../../data/qwen-moe/q8-unprofiled/receipt.json \
  --out ../../data/qwen-moe/q8-unprofiled/slow-gap-separation.json
```

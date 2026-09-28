# Qwen decode phase at occupied depth 1024

The selected `b4c67ced9` runtime spends 8.333 ms of summed HIP kernel duration per token in Q8 matvecs and 4.462 ms in routed Q4/Q5/Q6 expert matvecs on an eight-token synthetic decode at depth 1024. The entire measured device sum is 20.398 ms/token. This is a counter-free kernel trace, not a DRAM-byte measurement. A separate unprofiled, otherwise matched 32-token wall panel reports 50.929 tokens/s over three repetitions (51.171, 52.454, 49.161); the eight-token wall panel reports 47.653. The wrapper flagged competing host work in all three panels, so do not divide one panel's phase duration by another panel's wall time to claim an exact share.

| Kernel calls per token | Count | Device ms/token | Percent of device sum |
| --- | ---: | ---: | ---: |
| Q8 width 8192 | 40 | 3.866 | 19.0 |
| Q8 width 2048 | 80 | 2.222 | 10.9 |
| Q8 width 4096 | 30 | 1.443 | 7.1 |
| Q8 width 512, including fused shared gate/up | 60 | .802 | 3.9 |
| Routed Q4 gate/up | 40 | 2.524 | 12.4 |
| Routed Q5 down | 37 | 1.818 | 8.9 |
| Routed Q6 down | 3 | .119 | .6 |
| Q6 vocabulary head | 1 | 1.786 | 8.8 |
| All other kernels, including activation quantization | 1274 | 5.817 | 28.5 |

The image inventory charges 1,492.910 MB/token to Q8 and 611.516 MB/token to eight routed experts. Against the *conditional* one-read 242 GB/s comparator, Q8 has 8.333 − 6.169 = **2.164 ms/token** of room and routed experts have 4.462 − 2.527 = **1.935 ms/token**. Those are nearly equal opportunities even though Q8's current phase is almost twice as long. Improving Q8 width 8192 alone to that same comparator saves just .920 ms/token, or 4.7% of this device-time-equivalent step. The head's 417.178 MB would take 1.724 ms at 242 GB/s, already close to its measured 1.786 ms. These comparisons assume one DRAM read per selected weight and achievable uniform streaming; they are not upper bounds on all algorithms or measured traffic. The broken GL2C counter panel cannot decide whether the remaining gap is rereads, occupancy, operand preparation or launch scheduling.

[The device-timestamp gap audit](qwen-moe-launch-gap.md) finds the ~64-ms/token profiled idle interval concentrated at dispatch ordinals 30, 62, 126, 254 and then every 256; its 58.623-ms median long-gap subtotal is instrumentation-shaped and cannot be treated as native host-launch time against the separate 19.635-ms/token unprofiled wall panel. Family kernel durations are a within-trace ranking, not an additive decomposition of unprofiled wall time.

The trace needs careful slicing. `llama-bench -d 1024 --no-warmup -n 8` still evaluates the vocabulary head ten times: two synthetic-depth setup evaluations end at CSV rows 4679 and 9066, then eight measured decode evaluations end at rows 10631 through 21586, spaced exactly 1565 dispatches apart. Summing every dispatch would count depth construction as decode. [`decode_trace.py`](../tools/qwen-moe/decode_trace.py) isolates rows 9067–21586, checks every measured token's 210 Q8 calls by width and exactly one vocabulary head, and records each family. Profiler host launch gaps make the timestamp interval between heads about 88 ms/token while summed device kernel durations are 20.398 ms/token. Neither the 88 ms nor its subtraction from the unprofiled wall is a phase time. The older depth-zero trace had no such extra head evaluations.

The practical choice changes: do not port another Q8 wave width on the premise that it dominates the speed gap. First measure a counter-free shape-controlled native expert Q4/Q5 kernel against the installed grouped dispatch, including gather, activation quantization and sum, and investigate physical bytes only after the GL2C zero-run is explained. Expert down and gate/up offer almost as much conditional room as the entire Q8 family, but an isolated kernel gain does not certify a model speedup. For Q8, the 8192 group is the best single target only if a real program can recover its .920 ms without moving work to preparation or the other paths.

[Raw trace, wall samples, clocks, source/binary/model hashes and service restoration](../../data/qwen-moe/q8-unprofiled/receipt.json) are in durable data custody. The trace was `rocprofv3 --kernel-trace` without PMCs, selected `llama-bench`, depth 1024, batch 512, ubatch 256, eight CPU threads, flash attention on, full GPU offload and no benchmark warmup. The unprofiled wall panels used the same settings and 8 or 32 generated tokens. All GPU commands ran through `tools/run-batch-compare`; the resident service is active. No weights, installed executable or numerical map changed.

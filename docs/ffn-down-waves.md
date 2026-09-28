# The down projection paired two matrices in one wave, and that was its wave count

> **Two later measurements belong at the top of this document.**
> [The stream ceiling](ffn-stream-ceiling.md) prices the ownership directly from the tool now that
> `--ffn-dn-mats 2` is reachable at all — its validator refused the value its own error message
> documented — and halving this stage back to 160 waves is **+3.97% of the `ffn` phase**,
> bit-identical, which is the wave-count slope this kernel has. `bench/wstream`'s occupancy ladder
> agrees from outside the model: the same stream reads 126 GB/s at four waves per SIMD32 and 87 at
> two. **But the deep cursors this document fitted no longer pay.** They were dispatched away by
> the activation stage — `LAUNCH_BST` baked in `DLS_LOADS` — and with that repaired, depth 4 is
> −0.05% and depth 6 is +2.83% on the staged body. The stage removed the sixteen fragment loads a
> deeper cursor used to be covering.

`k_proj_opt` gives every wave two weight matrices. For gate/up that is structural: the wave holds
gate and up in its own accumulators and fuses `silu(gate) * up` before it writes anything, so both
matrices have to be in the same registers. The down projection inherited the same shape for no
reason at all — its two matrices are the two row halves of one tensor, and nothing in the epilogue
joins them.

That pairing is what set the stage's wave count. Down owns `DH/16 = 160` row tiles, and at 32 rows
the token axis has one group, so the stage launches **160 waves onto 80 SIMD32 units**. Two per
unit. [`docs/ffn-decode-shape.md`](ffn-decode-shape.md) named that shape as "latency with nothing to
hide it behind" and every repair anyone proposed for it bought waves by *sharing* a row tile, which
makes each sharing wave re-read the whole weight stream. The dense five-trit arm never takes that
trade, so it kept the two waves.

One matrix per wave buys the same waves for nothing. Wave `own` takes row tile `own >> 1` of matrix
`own & 1`, so 320 waves stream exactly the bytes that 160 waves streamed. Only the B activation
fragments are read by two waves instead of one, and those are a layer's 278 kB out of cache.

**It is bit-identical by construction and measured that way.** An output element is still one wave's
dot product over the same K order with the same per-block FP32 drain; which wave owns a row tile is
not in the arithmetic. The residual FNV-1a over every FP32 output is `1873717996769712938` at 32
rows, `14426824474163741251` at 64 and `7446865760224376151` at 128 — one value per row count across
both ownerships, and the same three values [`docs/ffn-decode-shape.md`](ffn-decode-shape.md)
published before this change existed.

## What it is worth

`rocprofv3 --kernel-trace`, both arms in one process, one build, `--pin-clock`, median over 64
layers per sample. The stage is selected by its grid and by the `MATS` template argument, so the
gate/up dispatches and the setup passes cannot contaminate it.

| rows | waves | paired, us/layer | split, us/layer | change |
|---:|---:|---:|---:|---:|
| 32 | 160 -> 320 | 199.2, 201.0 | **189.3, 192.4** | **-4.3 to -5.0%** |
| 64 | 320 -> 320 (4 token tiles) | 269.5, 272.9 | **259.5, 262.0** | **-3.7 to -4.0%** |

Full model, mode 19, both arms interleaved inside one process, two panels:

| workload | paired | split | change |
|---|---:|---:|---:|
| generation, 32 streams, aggregate | 297.7, 299.5, 299.8, 299.8, 300.0 | 297.4, 300.4, 301.2, 302.2, 302.2 | **+0.5%** |
| prompt processing, 64 rows/pass | 681.8, 683.8 | 689.6 | +1.0% |
| prompt processing, 32 rows/pass | 556.5, 559.7 | 561.3 | +0.6% |

**Say the size honestly: the full-model number is at the edge of this box's resolution, and the
arithmetic says it should be.** The down stage is 12.3 ms of a 106.7 ms 32-stream step, so 4.3% of
it is 0.5% of the step. The measurement and the prediction agree to within a tenth of a point, which
is the most a panel this size can claim. The change is selected because it is strictly cheaper on
every axis at identical output bits — fewer registers, the same weight bytes, more waves — not
because a full-model panel resolved it.

**Installed acceptance**, canonical build `dc730865809f`, executable
`90cd78b320a9396c24065ea0add2aa61786e0415b8d3784b4ca90c48a0d383c0`, service masked and restored:
32-stream aggregate generation 301.7 and 301.1 tok/s split against 298.9 and 298.8 paired, the
ordering reproduced in both rounds, **+0.87%**. Both residual hashes reproduced exactly on the
installed binary (`1873717996769712938` at 32 rows, `14426824474163741251` at 64, each equal across
both ownerships). Deployed single-stream `--bench` 33.16 tok/s, inside this box's 33.05-33.80 range
and untouched by construction: the default FFN mode is 0 and never enters this kernel.

## The register headroom is the part worth carrying

One matrix per wave halves everything the wave holds:

| shape | VGPR | waves per SIMD32 the registers allow | waves the grid supplies |
|---|---:|---:|---:|
| paired, two token tiles | 206 | 7 | 2 |
| **split, two token tiles** | **120** | **12** | **4** |
| split, four token tiles | 161 | 9 | 4 |
| paired, four token tiles | 249-252 | 5 | — |

The paired arm's accumulator budget is what
[`docs/ffn-decode-shape.md`](ffn-decode-shape.md) recorded as "the dense map's accumulator budget
caps it at two token tiles". Splitting removes that cap on this stage: `TileCap` stays at two for
the paired arm and the split down stage takes four, which is why a 64-row pass now walks its weight
stream **once** instead of twice.

Halving the weight traffic there bought 3.7%, not the factor the traffic suggests, and that is its
own small result: at 64 rows the down image is 18.1 MB per layer and the MALL is 32 MB, so the
second pass over it was already mostly cache hits. **Re-reading a weight stream that fits in the
last-level cache is nearly free; the published "reads its stream twice" cost model only applies
above that size.**

## Two negatives, and the register cliff between them

**A deeper weight cursor does not help this stage, even when its registers are free.** The obvious
thing to spend the headroom on is `ffn_dense_stream.hpp`'s block lookahead: the wave requests a
block of operand bytes some blocks before it hands them out, and depth was capped at two because a
third block cost gate/up a wave slot. On the split arm depth is genuinely free — depth 4 allocates
141 VGPR and depth 6 allocates 164, both far above the four waves the grid supplies. It still loses:

| arm, 32 rows | cursor 2 | cursor 4 | cursor 6 |
|---|---:|---:|---:|
| split, 320 waves | **189.3** | 195.5 (+3.3%) | 194.7 (+2.9%) |
| paired, 160 waves | 199.2 | 337.2 (+69%) | 339.8 (+70%) |

So the stage is not short of memory-level parallelism per wave. 320 waves two blocks deep already
hold 266 kB of weight requests in flight; adding more only lengthens the queue. **Read that beside
the wave result: doubling the waves helped 4-5% and doubling the depth hurt 3%, on the same stage,
in the same hour. Independent instruction streams and outstanding bytes are not interchangeable
ways to buy latency tolerance here.**

The paired arm's +69% is the register cliff, not a schedule: depth 4 takes it to 248 VGPR and depth 6
to 256, which is five waves per SIMD32 where the kernel needs headroom for its own liveness. It is
the same trap `docs/gdn-state-pack.md` names from the persistent kernel, reached from the other side.

**The image rule does not move.** With four token tiles the split dense arm reads the down stream
once per two token groups where the pair arm reads it four times, so it was fair to ask whether
dense now takes the 128-row down stage back. It does not: 425.8 us/layer against the pair arm's
386.3, with the FFN region 80.9 ms against 79.7 in the same process. `A4_DN_PAIR_ROWS = 64` stands.

## What is selected

The dense five-trit down stage always splits. It is the only arm this is built for — the pair arm
buys its waves by sharing a row tile and the two-bit and FP16 maps keep the shape they were measured
on — and the dense arm only ever runs the down stage at 64 rows or fewer, where the stage has never
had the waves it wants. Gate/up is untouched: its 1088 row tiles already fill the device, and the
`static_assert` in `k_proj_opt` refuses one matrix per wave on a fused stage.

`HALO_FFN_DN_MATS=2` pins the paired ownership for a process, `1` or `0` the split;
`--ffn-dn-mats 2,1` walks them as a case axis in both tools. `--ffn-dense 4` and `5` are the down
stage's cursor depths 4 and 6, which gate/up never sees — depth used to be one knob over two stages
that are limited by opposite things, and gate/up is the one whose registers pay for it.

## Reproduce

```sh
make -j8 tools/batch_profile tools/batch_compare
# the stage, isolated by kernel trace, both ownerships in one process
BONSAI_PROFILE_TRACE=1 BONSAI_PROFILE_OUTPUT=$DIR/kt \
tools/run-batch-compare --pin-clock --profile-tool --modes 19 --rows 32,64 --heads 0 --traces 0 \
  --ffn-dn-mats 2,1 --rounds 1 --out $DIR/probe.json
# the cursor depths, on both ownerships
tools/run-batch-compare --pin-clock --profile-tool --modes 19 --rows 32 --heads 0 --traces 0 \
  --ffn-dn-mats 2,1 --ffn-dense 1,4,5 --rounds 1 --out $DIR/depth.json
# full model, the shape this targets
tools/run-batch-compare --pin-clock --modes 19 --only prefill,decode --prefill-rows 32,64 \
  --streams 32 --ffn-dn-mats 2,1 --rounds 2 --tag ffn-down-waves-full
```

Raw samples: [`batch-comparison/ffn-down-waves/`](../../../data/bonsai2/batch-comparison/ffn-down-waves/README.md).

## The next question

The split stage is 189 us/layer for 18.1 MB, which is 96 GB/s against a 242 GB/s roof, and its own
compiled issue model is 113 us at 2.9 GHz. Both levers this document tried are now spent: it has all
320 waves its row tiles can supply and it does not want them deeper. What is left is **slots**, and
the split costs 14% more of them per unit of work than the pairing did — 321 issue slots per block
against 281, all of it in SALU address arithmetic (55 against 28) and in weight loads the paired arm
issued once for two matrices. Strength-reducing the three per-block pointer recomputations
(`bblk`, `sab`, `csb` are each rebuilt from `blk` every iteration rather than advanced) is the
cheapest 20-30 of those slots and needs no new shape.

Past that, the stage's wave count is capped by its 320 row tiles, and the only way past **that** is a
K split, which reassociates the per-block FP32 drain and therefore needs the teacher-forced NLL
instrument in [`docs/gdn-state-pack.md`](gdn-state-pack.md) rather than a residual hash.

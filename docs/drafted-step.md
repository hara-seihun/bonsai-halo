# Three weight coordinates in one kernel, and what the ternary one is actually short of

> **Corrected: the two-bit arm this document prices at "about 15 ms per drafted step" is a loss, and
> the `0.62` efficiency factor below is a property of the phase rather than of the coordinate.**
> The same instantiation that reads 137 GB/s here reads **208 GB/s** on the vocabulary head inside
> `k_forward_rows`, 96% of the issue ceiling this document computes, so a 1.214x image needs
> 252.8 GB/s against a 242 GB/s roof. [The ceiling panel](ternary-coordinate-ceiling.md) has the
> arithmetic, a fresh served-step map at 97% of nominal clock, and what the 15 ms actually belongs
> to. Everything else here stands, including the instrument, the census and the three-coordinate
> control.

The resident server generates with the DFlash2 drafter, and
[`docs/serving-decode.md`](serving-decode.md) is the only map of that step: two numbers, draft and
verify, taken five builds ago on a Q8 drafter. This is the inside of the drafter, measured for the
first time, and the result that came out of it is not about the drafter at all.

`k_dflash` streams **three different weight coordinates in one cooperative launch**, at eight rows,
on one grid, in one process, under one clock: the drafter's own Q8 or Q4 tiles, and the target
model's ternary HALO tiles for the vocabulary head. Nothing else in this engine can hold the
coordinate up against a control like that — every other kernel reads one coordinate, so a rate
measured in it carries the shape, the grid, the launch and the clock along with it.

**The ternary coordinate moves its bytes at 137 GB/s where Q4 and Q8 move theirs at 200-220, in the
same kernel, at the same row count, in the same process.** The gap is not memory and it is not
occupancy. It is instructions per byte, and the ternary coordinate is the only one in this engine
close to the line where that binds.

## The instrument

`HALO_PROFILE=1 HALO_PROFILE_DRAFT=1` accumulates `k_dflash`'s segment timeline over every drafted
step of a run. The kernel already stamped `wall_clock64()` after each grid sync when `P.prof` was
non-null; nothing ever set it. `Engine::dflash_segment_names` mirrors the kernel body, so the
readout names what produced the values each sync publishes, and one extra grid sync closes the
`selector`, which runs on a single workgroup after the last barrier. That sync exists only while
`P.prof` is set, so a deployed drafted step reaches exactly the code it always reached.

The drafted path already synchronises to read its drafts back, so the whole cost is one 8 kB copy
per step. `--draft-weights q8,q4` resets the accumulator per arm: before that fix the second arm
reported the average of both, which reads as a real number and is not one.

## Where a drafted step goes now

`--bench --dflash`, 23-token prose prompt, 60 generated tokens, `--pin-clock`, canonical `cd47cf6`.
Raw: [`drafted-step/coordinate-panel.txt`](../../../data/bonsai2/batch-comparison/drafted-step/coordinate-panel.txt).

| | Q8 drafter | Q4 drafter |
|---|---:|---:|
| drafted generation | 45.83 tok/s | 45.08 tok/s |
| tokens per step | 3.21 | 2.90 |
| accepted | 31.6% | 27.2% |
| draft, host | 10.990 ms | **7.224 ms** |
| verify, host | 57.910 ms | 56.148 ms |
| inside `k_dflash` | 11.495 ms | 7.489 ms |

The greedy digest is `16657175983209876683` in every arm of every run in this document, which is the
whole output contract of a lossless drafter. The two arms differ on this single prompt by acceptance
alone and in the opposite direction from [the ten-prompt panel](drafter-q4.md); one prose prompt at
19-21 steps does not resolve a 4-point acceptance difference, and that panel, not this one, is the
acceptance evidence.

Inside the Q4 drafter, 7.489 ms per step:

| segment | ms/step | share | what it streams |
|---|---:|---:|---|
| `mv-gate-up` | 2.131 | 28.5% | 459.6 MB of Q4 |
| `mv-lm-head` | **2.072** | **27.7%** | **278.1 MB of ternary HALO** |
| `mv-down` | 1.148 | 15.3% | 229.8 MB of Q4 |
| `mv-qkv` | 0.441 | 5.9% | 81.0 MB |
| `mv-o` | 0.380 | 5.1% | 54.0 MB |
| `ingest-mv-fc` | 0.322 | 4.3% | 67.6 MB |
| `mv-akp` + `mv-mkp` | 0.323 | 4.3% | 33.8 MB |
| `ingest-mv-kv` | 0.151 | 2.0% | 27.0 MB |
| `attn` | 0.145 | 1.9% | the drafter's own KV |
| everything else (25 names) | 0.375 | 5.0% | |

The byte column is the format arithmetic, and it closes: 858.0 MB for the five layers plus 67.6 for
`fc` plus 0.7 for `hproj` is 926.3 MB against the 926,580,736-byte Q4 cache file on disk.

**Two of this document's three candidate attacks die here.** All 25 remaining segments together —
every quantiser prep, every dynamic convolution, the whole four-phase top-k, the embedding and the
selector — are **5.0% of the drafter**, so the ~113 grid syncs a block draft pays for eight rows are
not what it spends. And the Q4 tile image is immune to the stream aliasing
[the weight image's run order](weight-stream-order.md) found costing the sequence and FFN images
113-143 GB/s: a Q4 block is **2112 bytes**, so a row tile's stream stride is `nb * 2112`, which
carries at most 2^9. The images that collapse are the ones whose 512-byte blocks put 2^12 or more
into the stride. Nothing to reorder.

## The coordinate control

Q8 and Q4 hold the same weights in the same tile and row mapping, and a Q8 block is 4160 bytes where
a Q4 block is 2112 — 1.970x, which is exactly the ratio of the two cache files. So the same segment
in the two arms is the same work over a known byte ratio, and `mv-lm-head` is the same code over the
same bytes in both arms and is the panel's own control.

| segment | coordinate | bytes | Q8 arm | Q4 arm | Q8 GB/s | Q4 GB/s |
|---|---|---:|---:|---:|---:|---:|
| `mv-gate-up` | Q8 / Q4 | 905.2 / 459.6 MB | 4.111 ms | 2.131 ms | **220.2** | **215.7** |
| `ingest-mv-fc` | Q8 / Q4 | 133.1 / 67.6 MB | 0.604 | 0.322 | **220.4** | **209.9** |
| `mv-down` | Q8 / Q4 | 452.6 / 229.8 MB | 2.105 | 1.148 | **215.0** | **200.2** |
| `mv-qkv` | Q8 / Q4 | 159.7 / 81.0 MB | 0.760 | 0.441 | 210.2 | 183.7 |
| `mv-o` | Q8 / Q4 | 106.5 / 54.0 MB | 0.597 | 0.380 | 178.4 | 142.1 |
| **`mv-lm-head`** | **ternary HALO** | **278.1 MB** | **1.974** | **2.072** | **140.9** | **134.2** |

The three largest streams reach **200-220 GB/s, 83-91% of the 242 GB/s [`bench/bw`](throughput-budgets.md)
measures on this box**, and halving the bytes from Q8 to Q4 buys 1.93x of the available 1.97x, so the
dense coordinates are at the memory wall and the cheap unpack costs them about 3%. `mv-o` and the
two `kp` projections fall off because they are small matrices, not because of their coordinate.

The ternary head, in the same launch, gets **137 GB/s, 57% of that roof**. The 4.8% between its two
arms is this phase's own repeatability across arms with different step counts.

## It is instructions per byte, and the number is on the device's line

`hipcc -S` on `kernels/halo_draft.hip`, `tools/isa_loop_count.py`
([`drafted-step/isa-census.txt`](../../../data/bonsai2/batch-comparison/drafted-step/isa-census.txt)):

| block loop | instructions | global loads | bytes it consumes | bytes per instruction |
|---|---:|---:|---:|---:|
| ternary HALO expansion | **686** | 3 | 896 | **1.31** |
| Q4 block loop, with its matrix slices | ~272 | 15 | 2112 | **7.8** |

The three loads are `uint4` + `uint2` + `unsigned` — 28 bytes, exactly one lane's share of a
896-byte HALO block — so that block is one 128-element block per lane and nothing else. Its 594 work
instructions are the radix-3 five-trit peel: 328 unclassified VALU, 104 bit operations and 72
multiplies, which is `m = v * 3u` and its siblings.

**The device's own ratio is 1.04 bytes per instruction**: 242 GB/s of memory against 80 SIMD32 at
2.9 GHz, which issue 232 G instructions per second no matter how many waves are resident. A
coordinate that delivers fewer bytes than that per issued instruction cannot reach the memory roof
by any amount of lookahead, occupancy or scheduling, because a SIMD32 retires one instruction per
cycle whatever is in flight behind it. **Q4 sits 7.5x above that line. The ternary expansion sits at
1.31 before a single matrix instruction, activation load, drain or wait is issued.**

The two shapes of the same coordinate confirm it and give the efficiency constant. At `TT = 1` the
matrix and drain work per block is small; at `TT = 8` it adds roughly 270 instructions to the same
896 bytes:

| shape | bytes per instruction | ceiling | measured |
|---|---:|---:|---:|
| ternary, `TT = 1` (`--bench` plain decode, 5.878 GB in 31.37 ms) | ~1.31 | 304 GB/s | **187 GB/s** (62%) |
| ternary, `TT = 8` (`mv-lm-head`, 278.1 MB in 2.02 ms) | ~0.94 | 218 GB/s | **137 GB/s** (63%) |

Both land at 62-63% of their own issue ceiling, and the ratio of the two measured rates, 1.37,
is the ratio of their bytes per instruction, 1.39. **The ternary stream's byte rate is
`0.62 x (bytes per instruction) x 232 GB/s` in both shapes this engine runs.** Q4 and Q8 are not on
that line at all; they are on the memory one.

### This retires three nulls and predicts the next one

Every instruction-cutting result on the eight-row deployed pass has measured near zero, and the lane
has filed them as evidence that the pass is latency-bound:

- [the nibble operand](mv-dot8-nibble.md) cut expansion 254 to 156 slots and the pass moved 1%
- the two-trit palette cut expansion 271 to 238 and measured null
- a peer cut 9.4% of the A4 FFN block's slots and the step did not move

The model says the first one *should* have paid about 10% on the matvec, and its own document says
where that went: the nibble activation pane it needs costs in prep about what the matvec gains. The
others are too small to clear the 62% efficiency factor. What the model rules out is any change that
does not move bytes per instruction — **deeper load lookahead, more waves and better scheduling
cannot lift a stream that is issue-limited**, and three of those have now been measured as nulls on
this pass.

**To reach the memory roof a coordinate needs `242 / (0.62 x 232) = 1.68` bytes per instruction.**
The deployed ternary coordinate is at 0.94 at eight rows. It needs 1.8x, and there are only two ways
to get it: halve the expansion, or store more bytes per trit.

## What that is worth, and what it costs

The target model is 5,877,784,576 bytes of HALO tiles at 1.75 bits per trit, and the eight-row
verify pass — 79-83% of the served drafted step — reads all of it.
[The nibble operand's panel](mv-dot8-nibble.md) measured that pass at 46.35 ms, 127 GB/s, which is
the same regime as the 137 GB/s this document measures on the same coordinate in another kernel.

- At the 215 GB/s the Q4 streams reach in the same kernel, **the same 5.878 GB takes 27.3 ms**.
- A two-bit trit coordinate is 6.72 GB. At 215 GB/s that is **31.3 ms**, and its unpack is a shift
  and a mask rather than a radix-3 peel, so it lands far above the 1.68 line.

Either way **there is about 15 ms per drafted step in the coordinate, which is 24% of a 63 ms
served step** — larger than everything inside the drafter put together.

**And it has a price this document will not hide.** Single-stream decode takes the `TT = 1` body,
where the same coordinate already reaches 187 GB/s and is much closer to the memory wall. At 6.72 GB
and 187 GB/s a token step goes 31.37 to 35.9 ms, **13% slower**, unless the wider coordinate also
cuts the `TT = 1` block's instructions enough to lift its rate — which it should, since that shape
is at 62% of a 304 GB/s ceiling, but it is not free by construction and must be measured. The
trade is real: the dense coordinate wins the shapes with rows to amortise over and loses the
single-row one.

## What is left in the drafter, priced

Nothing here is worth a turn, and that is the useful half of this document.

- **The vocabulary head is 27.7% of the drafter and the handoff's own "largest first" item** —
  "nobody has asked whether a drafter needs all 248,320 columns". Asked and priced. Its ceiling is
  the difference between 137 GB/s and the memory roof on 278 MB: **0.9 ms**, which is 12% of the
  drafter and **1.2% of the served step**. Sweeping fewer columns is the other direction and it is
  worse than it looks: the drafter's candidates are verified so it cannot move an output bit, but
  the top-16 comes from a full-vocabulary argmax whose cheap approximations (half-K estimate,
  frequency shortlist) all trade accepted tokens for a fraction of 1.2%.
  **[Built and measured since](drafter-head-columns.md)**, and the prediction holds on English and
  code with a wider margin than expected - acceptance is flat to a quarter of the vocabulary for
  +2% - while the same window is **-15.8% on five non-English prompts**, which is what keeps it out
  of the runtime. That panel also prices this document's own numbers on a current build:
  `mv-lm-head` is 1.525 ms, not 2.07, and it is column-proportional.
- **A Q4 or Q8 copy of the head is a measured loss, by arithmetic.** 636 MB at 205 GB/s is 3.10 ms
  and 1.27 GB at 215 GB/s is 5.9 ms, against 2.07 ms of ternary. The head is the one place where the
  ternary coordinate's byte count wins outright, because its stream is not shared with anything.
- **Three bits for the drafter's own weights**, the handover from [the Q4 drafter](drafter-q4.md),
  is worth what it removes from a memory-bound stream: 926 to about 700 MB takes the drafter from
  7.49 to about 6.3 ms, **1.9% of the served step**, against an acceptance cost that the ten-prompt
  instrument would have to clear.

## Reproducing

```sh
HALO_PROFILE=1 HALO_PROFILE_DRAFT=1 tools/run-batch-compare --pin-clock --engine \
  --bench --context 4096 --dflash ../../data/bonsai2/drafters/dflash2.safetensors \
  --draft-weights q8,q4 -n 96
```

`HALO_PROFILE=1` alone keeps the old behaviour and prints the verify pass's phase map through
`Engine::print_profile`; the two cannot be read in one run, because one `prof` buffer serves both
cooperative kernels and the second writer would be printed under the first one's names.

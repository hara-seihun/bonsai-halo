# The drafter reads 248,320 columns to keep sixteen, and narrowing that is a multilingual regression

A lossless drafter verifies every token against the target, so **what the drafter looks at cannot
change an emitted token** - only how often a proposal survives. That makes the vocabulary head the
one place in this engine where an approximation costs an accepted count and nothing else, and
[the drafter's own map](drafted-step.md) put that head at 20.8% of a draft: 278.1 MB of the target's
ternary HALO tiles at 134 GB/s, the only ternary stream in `k_dflash` and the only one under
180 GB/s while its Q4 neighbours run at 200-220.

This iteration builds the window, measures it, and **does not ship it**. On English and code it is
worth about +3% with acceptance intact down to a quarter of the vocabulary. On five non-English
prompts the same window is **-15.8%**, because a drafter proposal for those languages lives in the
high ids the window drops. A serving default has to be safe on the workload it serves, so the arm is
out of the runtime and its patch is beside the raw samples.

Raw, the patch and the pooling script:
[`batch-comparison/drafter-head-columns/`](../../../data/bonsai2/batch-comparison/drafter-head-columns/README.md).

## What was built

`DflashParams` gains `head_cols` and `head_tail`. The head's `MvR` takes two segments - the low
`head_cols` ids at stride `head_cols`, then the top `head_tail` ids written after them at
`logits + DF_BLOCK * head_cols` - and the four top-k phases walk that window instead of `VOCAB`,
carrying a tail column's true id into the candidate list so the selector and the host need no
remapping. The tail exists because this tokenizer keeps its special block at the very top
(`eos = 248046`, `<|im_end|>` and padding at 248044), and a chat turn that cannot end inside a draft
pays a whole rejected step per turn. `--draft-cols N[,N...]` walks it as an in-process axis and
`HALO_DRAFT_COLS` pins it.

One occupancy note for whoever picks this up: the window's second segment and its runtime slice
bounds cost one VGPR, and `k_dflash<4, true>` sits one register under the six-wave line at 240. That
one register takes it to five waves, and since `dflash_grid` is the minimum over the row
instantiations it would have cost **every** Q4 arm its grid, 60 workgroups down to 40.
`amdgpu_waves_per_eu(6)` holds all six instantiations at six waves and incidentally lifts the Q8
arms, which had been running at five, from 247 VGPR to 235.

## The output contract, measured rather than asserted

Nine prompts, up to five window widths, both arm orders - **every arm of every case digests its
greedy token stream to the same value**. That is the whole output contract of a lossless drafter and
it is what makes the rest of this document a throughput question instead of a quality one.

## English and code: the window is free down to a quarter of the vocabulary

Ten prompts from `bench/drafter-prompts.txt`, 40 tokens each, `--pin-clock`, Q4 drafter, canonical
`d8a66da`, arms alternating inside one process and the whole panel run twice with the arm order
reversed. `verify` is the target's own forward, which this change cannot reach, so it is the
in-panel control: `step@ctl` prices every arm at the pooled verify and removes the clock drift that
otherwise rewards whichever arm ran first.

| `--draft-cols` | share of vocab | draft ms | tokens/step | accepted | steps | step@ctl | tok/s@ctl | |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 248320 | 100.0% | 6.494 | 3.659 | 38.0% | 229 | 50.808 | 72.02 | — |
| **131072** | 52.8% | **5.813** | 3.726 | 38.9% | 226 | 50.128 | **74.32** | **+3.19%** |
| 65536 | 26.4% | 5.469 | 3.595 | 37.1% | 232 | 49.783 | 72.21 | +0.26% |
| 32768 | 13.2% | 5.331 | 3.123 | 30.3% | 260 | 49.645 | 62.91 | -12.66% |
| 16384 | 6.6% | 5.322 | 2.971 | 28.2% | 276 | 49.636 | 59.86 | -16.89% |

**The order control matters and it does not change the shape.** Run wide-first the pooled verify is
45.8 ms and run narrow-first it is 42.8, because a case's later arms run on a hotter device and the
narrow arms take more steps; normalised on the control, 131072 is +4.11% in one order and +2.29% in
the other, and the collapse below an eighth is -11% to -17% in both.

**The acceptance difference at 52.8% and 26.4% is noise and should be read as one.** Pooled, an arm
drafts about 1600 positions, so a binomial floor on 38% is 1.2 points - and per-position acceptance
is a run length rather than 1600 independent draws, so 1.2 points is a floor on the noise and not
the noise. 38.9% and 37.1% are inside it. 30.3% at an eighth is five floors below it and is real.

So the honest English/code result is the mechanical half: **the draft loses 0.68 ms at half the
vocabulary and 1.03 ms at a quarter**, which against a 50.8 ms step is +1.3% and +2.0%, and the rest
of the +3.19% is acceptance that did not move.

## Multilingual: the same window is a 15.8% regression

Five prompts, one each in Chinese, Japanese, Russian, Arabic and Portuguese, same build, same clock
policy, same instrument.

| `--draft-cols` | share of vocab | draft ms | tokens/step | accepted | steps | tok/s@ctl | |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 248320 | 100.0% | 6.709 | 1.712 | 10.2% | 118 | 34.90 | — |
| 131072 | 52.8% | 6.066 | 1.423 | **6.0%** | 142 | 29.38 | **-15.80%** |
| 65536 | 26.4% | 5.666 | 1.365 | 5.2% | 148 | 28.43 | -18.54% |
| 32768 | 13.2% | 5.472 | 1.320 | 4.6% | 153 | 27.61 | -20.88% |

Acceptance falls by 4.2 points at the first step of the ladder, four times the 1.05-point binomial
floor at this rate. Ids in a BPE vocabulary are merge-ordered, so a prefix is a frequency prefix -
for English and code. For a script whose merges were learned later the same prefix is a different
and much smaller fraction of the text, and the drafter's proposal is outside the window often
enough to lose more than the head is worth.

**A second fact worth more than this experiment.** On the same build and clock, DFlash2 accepts
**10.2%** of its proposals on these five prompts against **38.0%** on the ten English and code ones,
and drafted generation is **34.9 tok/s against 72.0**. The drafter this engine serves with is close
to useless outside English and code, and nothing in this repository said so. Anyone measuring the
served step should say which of those two machines they measured.

## Where the head's time actually is

`HALO_PROFILE_DRAFT=1`, three prompts, full window against 16384 columns, in one process:

| segment | 248320 cols | 16384 cols |
|---|---:|---:|
| `mv-lm-head` | 1.525 ms/step | **0.135** |
| `topk-slice-max` | 0.067 | 0.004 |
| `topk-collect` | 0.040 | 0.005 |
| `topk-select` | 0.016 | 0.016 |
| `topk-threshold` | 0.015 | 0.015 |
| `selector` | 0.014 | 0.013 |
| draft, host | 6.984 | 5.679 |

**The head is column-proportional to within its own phase-length ramp**: 6.6% of the columns for 8.9%
of the time, and the whole 1.49 ms of segment saving arrives at the host as 1.31 ms. There is no
fixed cost hiding in this phase to attack instead - the 1.525 ms is the 278 MB, and the only ways to
spend less are fewer columns, which this document prices, or fewer instructions per byte, which is
`mv_rows_t`'s radix-3 expansion and belongs to whoever holds `kernels/device.hpp`.

## Why the arm is not in the runtime

Two reasons, and the first alone is enough. It cannot be a serving default, and left in it would be
an unused alternative on the path the service runs. And it is not free where it is not used: with
the window compiled in, `mv-lm-head` measures 1.525 and 1.530 ms per step at the full window against
**1.436, 1.460 and 1.510 on canonical `27d4729`** - the deployed column set pays something for the
runtime slice bounds and the second segment. That comparison crosses two builds and is not clean
enough to quote as a number, which is exactly why the arm should not sit in the deployed path on the
strength of it.

`draft-head-window.patch` beside the raw samples applies to `d8a66da` and restores the whole
mechanism, the `--draft-cols` axis and the occupancy fix in one piece.

## The next useful question

**A static prefix fails for the reason a dynamic window would not.** The tokens a multilingual
proposal needs are the ones already in the prompt and in what has been generated - a context window
carries its own vocabulary. The candidate set that should be measured next is the union of a low-id
prefix and the distinct ids in the sequence's own context, which is a few hundred tiles more and is
rebuilt once per step, not once per position. The obstacle is not the idea but the coordinate: the
HALO image is tile-major over 32-row groups, so a scattered set of columns is a gather over tiles
rather than a range, and `ph_matvec`'s unit decomposition assumes segments are contiguous runs. A
per-step tile list is the smallest change that would let the head read a set instead of a range, and
it would serve the vocabulary head of every consumer in this engine, not only the drafter's.

## One bounded negative that came out of the same reading

The HALO tile block spends 2 of every 28 bytes on an fp16 scale - **7.1% of every weight stream this
engine reads** - and the obvious question is whether those bytes carry anything.
`tools/halo_scale_probe.py` reads the cache index and answers it with no GPU: a row's 40 blocks
carry **33 to 40 distinct scale values** and 1 to 3 distinct exponents, so the scale is not a per-row
constant and cannot be delivered from a per-row array without changing what the kernel computes; a
whole tensor's distinct values run into the thousands, so a byte palette does not close it either.
The codes themselves are 26 bytes per 128 trits, **1.625 bits a trit against the 1.585 entropy
bound**, so the image is within 2.5% of byte-granular optimal. There is nothing to win in the weight
image's size, and the next engineer should not spend a turn finding that out again.

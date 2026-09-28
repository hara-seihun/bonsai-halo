# The recurrence's lane layout: one lane per 32 columns of one row

The resident state kernel is an instruction-issue machine. [The census](resident-gdn.md#the-phase-is-an-instruction-issue-machine-and-the-count-says-so)
fixed that: 355 issue slots a token, and at the two arms that fit in the instruction cache the
counted slots explain 88.6% of the measured phase. It also named the largest remaining item - the
cross-lane reduction and the `s_delay_alu` chain that covers it - and
[the block map](gdn-map/README.md) had already located that cost precisely: "the cross-lane
reductions the block form was supposed to remove belong to the state's **lane layout**, not to the
recurrence", with `L = 4, R = 1` ranked at 1.69x on its lane-op count and then parked, because "any
trial belongs in the resident prompt kernel".

This is that trial. It is worth **-19.7% of the state phase and +2.4% full-model prompt throughput**
against today's default, it is **bit-identical**, and the interesting part is why bit-identity was
available at all: `warp_sum` does not leave one value in a wave, it leaves two, and the deployed
recurrence has been quietly applying both of them to different columns of the same state row for as
long as it has existed.

## The change

A wave used to own `R = 16/SPLIT` whole rows and spread each row's 128 columns over all 32 lanes,
four per lane. Both consumers of a row - the k-side contraction that builds the delta and the q-side
contraction that builds the output - were therefore cross-wave reductions, and the wave paid **two
five-stage `warp_sum` butterflies per row per token**: eight butterflies for 66 slots of arithmetic
at the deployed `SPLIT = 4`.

Turn the ownership 90 degrees. Lane `l` owns row `j = part*64 + wave + 8*(l>>2)` and, inside it, the
32 columns belonging to deployed lanes `8p .. 8p+7` for `p = l&3`. A row's contraction now lives in
four lanes, so **one pair of two-stage quad butterflies reduces all eight of the wave's rows at
once**. The arithmetic per state element does not change - a decay multiply, a k-side `fmaf`, an
update `fmaf`, a q-side `fmaf` - which puts a floor of 16 slots per row-token at any `L`; what moves
is everything else, because the per-token overhead is divided by the `32/L` rows a wave carries
instead of multiplied by the `R` rows a lane carries.

Two units a head instead of four, eight waves a unit, **768 waves a layer against 1536**.

`HALO_GDN_COLS=0|1` pins either layout for a whole process and `--gdn-cols 0,1` on either
measurement tool interleaves them inside one; unset, the shape rule below picks.

## warp_sum leaves two roundings in a wave, and the deployed kernel uses both

`warp_sum` is `quad_perm(1,0,3,2)`, `quad_perm(2,3,0,1)`, `row_ror 4`, `row_ror 8`, `permlanex16`,
each as `v += dpp(v)`. After the two quad stages every lane of a DPP quad holds the same quad sum
`Q_{H,q}`, because the two operand orders differ only by commutativity. The two rotations then give
lane `l`

    (Q_{H,j} + Q_{H,j-1}) + (Q_{H,j-2} + Q_{H,j-3}),    j = (l>>2)&3, indices mod 4

and `j = 2` reproduces `j = 0` while `j = 3` reproduces `j = 1` - commutativity again. `permlanex16`
preserves `j`. So a wave ends a reduction holding **exactly two bit patterns of the same
mathematical sum**, selected by bit 2 of the lane index, and `tools/gdn_warpsum_probe` measures that
signature directly: `00001111000011110000111100001111`, with the two classes genuinely different
bit patterns in **793 of 2000** random waves.

`gdn_token` then applies lane `l`'s value to the state columns lane `l` owns. That makes the per-lane
rounding part of the deployed numerical map:

- column `c` is updated with the delta of class `((c&31)>>2)&1`;
- `out_o` is written by lane 0, so **every recurrent output carries the even class**.

Reproducing that is what makes this change exact rather than approximate. Sub-lane `p` holds deployed
quads `(p>>1, 2(p&1))` and `(p>>1, 2(p&1)+1)`, so `gdn_cols_reduce` builds both classes from the
eight column-group partials one lane holds - 8 in-lane adds and 11 cross-lane slots - and applies
class 0 to its first four column groups and class 1 to the last four, which is exactly `(i>>2)&1`.

### The rotation goes the other way than it reads, and a self-consistent emulation will not notice

The first build of this arm measured 4-ULP differences on 52% of the state. The reduction was right;
the **direction of `row_ror`** was not. "Rotate right by n" reads as if lane `m` receives lane
`m + n`, which pairs quad `j` with quad `j+1` and makes class 0 equal `(Q0+Q1) + (Q2+Q3)`. The
hardware does the opposite:

    row_ror 4     lane <- 12,13,14,15,0,1,2,3,4,5,6,7,8,9,10,11,28,29,...
    row_ror 8     lane <- 8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7,24,25,...

Lane `m` receives lane `(m - n) mod 16`, so class 0 pairs quad `j` with quad `j-1` and equals
`(Q0+Q3) + (Q1+Q2)`. Getting it backwards swaps the two classes, which is a few ULP on about half
the state elements and nothing else at all.

**The trap worth carrying:** an offline emulation written from the same wrong reading confirms the
kernel and both disagree with the device. `tools/gdn_cols_check.py` did exactly that and printed
PASS. Measure the permutation; do not model it. `tools/gdn_warpsum_probe` (new, ~7 s, needs the
GPU lock and no model) runs each DPP stage on the lane index, composes `warp_sum` on the host from
the permutations it measured, and checks that composition against the device's own `warp_sum` over
2000 random float32 waves - 2000/2000, bit for bit. `tools/gdn_cols_check.py` now uses the measured
direction and reports 0 differing bits over 5000 waves with the two classes genuinely distinct in
40.5% of them.

## What it costs in issue slots

`tools/isa_loop_count.py` on the same build, committed token path, `GATE_HOIST`:

| | arithmetic block | rows it covers | per row-token |
|---|---:|---:|---:|
| deployed `resident_state<4,1,false>` | 166 slots | 4 | 41.5 |
| **column arm `resident_state_cols<1>`** | **164 slots** | **8** | **20.5** |

The whole committed loop is 302 slots for four rows against 308 for eight, so the census says a
little under 2x either way you cut it. The replay branch a committed pass never takes is 231 slots
in the deployed arm and 236 here, and is excluded from both.

Registers, from `tools/kernel_resources.py`, no spill and no LDS in any arm:

| instantiation | VGPR | waves/SIMD32 |
|---|---:|---:|
| `resident_state<4,0/1,false>` | 64 | 16 |
| `resident_state_cols<0>` | 121 | 10 |
| `resident_state_cols<1>` | 118 | 12 |

A 128-row single-sequence pass wants 96 blocks of eight waves, 9.6 waves per SIMD32, against the 12
this instantiation is granted. The layout costs occupancy it does not need: the deployed arm's 1536
waves a layer were 16 per SIMD32 and only ever needed the issue slots.

## Measured

### The phase

Mode 19, 128 rows, logits off, `--pin-clock`, all three arms interleaved and reshuffled inside one
process, four rounds. `gdn-resident-core` normalised by the five phases the change cannot reach,
which is how this lane reads a phase panel on a box that moves 15% inside one round. Raw in
[`batch-comparison/gdn-lane-layout/phase-128-rebased.json`](../../../data/bonsai2/batch-comparison/gdn-lane-layout/phase-128-rebased.json).

| arm | `gdn-resident-core`, four samples | normalised |
|---|---|---:|
| deployed rows-per-lane (`SPLIT` rule, 2 at one sequence) | 13.44, 13.42, 13.37, 13.45 | 0.10618 |
| **column arm** | 10.77, 10.82, 10.80, 10.75 | **0.08524 (-19.7%)** |
| the shape rule, unpinned | 10.75, 10.86, 10.83, 10.78 | 0.08516 (-19.8%) |

The five control phases sum to 126.3-127.6 ms in every one of the twelve samples, so the ratio is
not carrying a clock. The rule arm lands on the column arm, which is what it is supposed to do at
128 rows in one sequence. Every sample hashes the residual to **8607785813945241221**.

**Against the previous default the same change was -26.2%**, measured before
[the unit width](gdn-unit-width.md) landed: `gdn-resident-core` 13.975 -> 10.316 ms at `SPLIT = 4`,
device span 152.078 -> 148.464, residual `7446865760224376151`, in
[`phase-128.json`](../../../data/bonsai2/batch-comparison/gdn-lane-layout/phase-128.json). Widening
the unit from four row groups to eight amortises the same per-token operand and addressing overhead
that the column layout divides by its eight rows, so **the two changes are partly the same money**
and they compose to roughly the product of what is left. Anything else measured against the
`SPLIT = 4` phase needs re-fitting for the same reason.

### The model

Mode 19, 384-token document in three 128-row passes, tail-only head, `--pin-clock`, arms interleaved
in one process, eight rounds. Raw in
[`batch-comparison/gdn-lane-layout/prefill-rebased2/`](../../../data/bonsai2/batch-comparison/gdn-lane-layout/prefill-rebased2/).

| arm | rounds | median |
|---|---|---:|
| deployed rows-per-lane | 879.9, 886.7, 881.2, 880.7, 884.2, 882.9, 997.5, 981.4 | 883.6 tok/s |
| **column arm** | 907.1, 898.1, 895.1, 909.5, 900.2, 906.1, 1014.9, 1003.9 | **906.6 tok/s** |

**+2.60% full-model prompt throughput, ahead in all eight rounds** by 1.3% to 3.3%. The last two
rounds ran on a box that had found 13% more clock; they favour the column arm by the same margin as
the other six, which is the point of pairing inside one process.

### And it loses at a generation step, which is why the shape picks

32 streams of one token each, mode 19, fp32 state, context 256, arms interleaved in one process,
two rounds. Raw in [`batch-comparison/gdn-lane-layout/decode/`](../../../data/bonsai2/batch-comparison/gdn-lane-layout/decode/).

| arm | rounds | median |
|---|---|---:|
| deployed rows-per-lane | 321.9, 314.1 | **318.0 tok/s** |
| column arm, pinned on | 288.3, 280.4 | 284.4 tok/s |

**-10.6% aggregate generation**, both rounds, so the rule is not a precaution. Two mechanisms point
the same way and neither is a surprise. A unit that walks one token is not issuing, it is waiting for
its 64 kB state round trip, which [the decode map](decode-map.md) measures at 93% of this device's
memory roof - so removing issue slots buys nothing. And the column arm's state load is the thing that
gets worse there: a lane wants four runs of eight floats per `s` rather than four contiguous floats,
so one `global_load_b128` touches eight 128-byte lines using 64 bytes of each, twice the line
requests of the deployed load for the same bytes. Free when amortised over 128 tokens, not free when
the round trip *is* the phase.

The threshold is measured at 1 and 128 rows per sequence. Everything between is interpolation, both
arms are correct everywhere, and `--gdn-cols` pins either.

### Exactness, three ways

- `HALO_GDN_COMPARE=1` clones the entry state, runs the sliced eight-row reference beside the column
  arm on the first resident layer-0 pass and compares every bit: **0/1,310,720 token floats,
  0/786,432 recurrent outputs, 0/786,432 state floats, 0/30,720 ring floats**. Before the `row_ror`
  fix the same probe read 354,115/786,432 outputs and 407,072/786,432 state differing, all in the
  last bits, which is what a swapped rounding class looks like.
- The residual FNV-64 is identical across all eight timed samples of the phase panel.
- `tools/gdn_cols_check.py` proves the reduction offline against the measured permutations.

### Installed

Canonical `ba1b550`, `bonsai-halo` `a99c0acbc46fe2e8e697e6d1…`, `tools/batch_compare`
`3aadb3e70a24d37f94deb131…`. The installed build reads `gdn-resident-core` 13.37-13.51 ms on the
deployed arm against 10.75-10.92 on the column arm, **-19.6% normalised**, with the rule selecting
the column arm and the residual `8607785813945241221` in all nine samples; full-model prompt
**908.4 -> 930.2 tok/s, +2.39%, five of five rounds**. Raw in
[`installed-phase.json`](../../../data/bonsai2/batch-comparison/gdn-lane-layout/installed-phase.json)
and [`installed-prefill/`](../../../data/bonsai2/batch-comparison/gdn-lane-layout/installed-prefill/).
The resident server came back on it and answered a drafted completion at 50.38 tok/s. Serving never
enters this kernel - the eight-row route runs `ph_gdn` inside the persistent kernel - so that is a
smoke test of the install, not of the change.

## Where the rest of the predicted win went, and the next move

The census says the committed token path halves per row-token; the phase fell 26%. Put the two arms
against their own issue models - `waves x slots x tokens x layers / 80 SIMD32` at 2.9 GHz:

| arm | waves/layer | slots/token | predicted | measured | issue explains |
|---|---:|---:|---:|---:|---:|
| deployed `SPLIT = 4` | 1536 | 302 | 12.3 ms | 13.975 | 88% |
| column arm | 768 | 308 | 6.3 ms | 10.316 | **61%** |

The deployed arm sits where [the resident notes](resident-gdn.md) left it. The column arm does not,
and the reason is almost certainly the one
[the row batching](gdn-token-reduce.md) already proved on this kernel from the other side: **a wave
now has exactly one row, so its two reductions are a single serial dependency chain with nothing to
interleave.** The deployed arm's `R` rows gave the scheduler `R` independent chains, and batching
them was worth -15.1% of the phase; this layout spends that ILP to buy the slot count.

**The next experiment is to have both.** Give a column-owning lane `RR = 2` row groups: 64 state
VGPRs, 16 rows a wave, two independent k-side chains and two independent q-side chains per token -
and, for free, **the same 16 `float4` key and query loads serve two rows instead of one**, because
`k` and `q` are per (token, head) and not per row. Predicted 21.1 slots per row-token against 22.6,
which is not the point; the point is the 39% of this phase that its own issue count cannot explain.
It wants a four-wave block (`128` threads, `CONV_CH/128 = 80` ring-commit groups) to keep two units
a head and 96 blocks on 20 WGPs; one unit a head is 48 blocks and 2.4 per WGP, which throws away
more to imbalance than the ILP is worth. `--gdn-cols 2` is the natural arm name.

Two smaller things this leaves:

- **The operand cursor.** The committed loop spends about 100 SALU slots a token reaching its
  operands, and the token record advances by a constant `RESIDENT_TOKEN_FLOATS` per token with 16
  fixed offsets off it. This is the same disease [the vocabulary head](wide-head.md) removed 109
  slots of. The deployed arm pays it too. A running cursor is bit-identical by construction; the
  obstacle is that the replay test is inside the loop, and splitting the loop has already lost once
  on instruction-cache footprint.
- **fp32 only, deliberately.** A packed store derives a row's shared exponent with a whole-wave
  `warp_max`, which means "this row" only in the deployed layout, and the codec owns that
  expression. A deferred pass keeps its rank-1 term through machinery built on the deployed row
  decomposition. So the column arm runs on fp32 state and committed passes, which is the deployed
  prompt-ingestion route and every prefill panel this lane publishes. Extending it to `i8` is two
  quad-`fmax` instructions and a `p == 0` scale store inside
  [the codec](../kernels/gdn_state_codec.hpp), and it is worth doing only if the batched-generation
  shape turns out to want it - which the rule below says it does not.

## Operation

`HALO_GDN_COLS=0` pins the deployed rows-per-lane state units for a whole process, `=1` pins the
column arm, and unset lets the shape rule pick. `--gdn-cols -1,0,1` on `tools/batch_profile` or
`tools/batch_compare` walks rule, deployed and column inside one process, which is the only way to
resolve an effect this size on a shared host.

The rule is the same quantity [the gate placement](resident-gdn.md#in-a-generation-step-the-hoist-loses-so-the-shape-picks)
turns on, and it is measured, not argued: +2.2% full-model prompt at 128 rows per sequence, -10.6%
aggregate generation at one. `COLS_MIN_ROWS_PER_SEQ` is 8 rows per sequence, the same threshold the
gate placement uses. The arm additionally requires fp32 state and a committed (non-deferred)
pass; `gdn_resident_cols_available()` reports that, and both measurement tools record it.

```sh
make -j8 bonsai-halo tools/batch_compare tools/batch_profile
tools/gdn_cols_check.py --cases 20000                      # offline, no GPU
make tools/gdn_warpsum_probe && tools/run-batch-compare --exec tools/gdn_warpsum_probe
tools/run-batch-compare --pin-clock --profile-tool --out DIR/phase.json \
  --modes 19 --rows 128 --heads 0 --traces 1 --gdn-cols 0,1 --rounds 4
tools/run-batch-compare --pin-clock --tag NAME/prefill --modes 19 --only prefill \
  --prefill-rows 128 --prefill-tokens 384 --gdn-cols 0,1 --rounds 4
HALO_GDN_COMPARE=1 HALO_GDN_COLS=1 tools/run-batch-compare --profile-tool --out DIR/compare.json \
  --modes 19 --rows 128 --heads 0 --traces 0 --rounds 1
```

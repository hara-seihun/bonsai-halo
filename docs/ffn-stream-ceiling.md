# The A4 FFN's ablation ladder was not connected to the body the engine runs

Raw samples, the compiled loops, the register table and the `bench/wstream` panel are in
[`batch-comparison/ffn-stream-ceiling/`](../../../data/bonsai2/batch-comparison/ffn-stream-ceiling/README.md).
**No shipped arithmetic, schedule or default changed.** The tree gains a working ablation ladder, a
`bench/wstream` panel that had never been run, and two repairs to instruments that were reporting
the arm you asked for while running a different one.

**The headline is a retraction of a result this document produced two hours earlier in the same
turn, and the reason is worth more than the claim was.** The ladder said the A4 FFN's weight stream
costs nothing. It costs **21.9%** of the phase. The ladder was measuring the deployed arm under six
different labels.

## The defect

`launch_opt_sp` dispatches the activation stage before the load schedule:

```cpp
if constexpr (OP == OP_IU4 && !SHARE && (WM == WM_DENSE || WM == WM_PAIR) && TT <= 2) {
    if (bst == ffnb::BSTAGE_SHARED) { LAUNCH_BST(ffnb::BSTAGE_SHARED); return; }
    if (bst == ffnb::BSTAGE_WAVE)   { LAUNCH_BST(ffnb::BSTAGE_WAVE); return; }
}
```

and `LAUNCH_BST` instantiates `k_proj_opt` with `DLS_LOADS` baked in. Every code the `dls` argument
carries — the ablation ladder at 8..13, the down stage's deep cursors at 4 and 5, depth three at 2
and 3 — returns from that branch having run depth two. It has been silent since
[the activation stage](ffn-b-operand.md) became the real default at 21:30 on September 21, which is
[a defect two engineers found in the same twenty minutes](ffn-block-pipeline.md) and repaired an
hour before this measurement.

**It does not read like a broken instrument, which is why it cost a turn.** Six ablation arms in one
process produced *six different residual FNV-64 values* and one unmoved phase:

| arm, mode 19, 32 streams, ffn over the six phases it cannot reach | `ffn`/control | against deployed |
|---|---:|---:|
| deployed | 0.4603 | — |
| activation stream pinned to block 0 | 0.4605 | +0.05% |
| weight stream pinned to block 0 | 0.4610 | +0.16% |
| both pinned | 0.4600 | −0.05% |
| one activation fragment a block (15 fewer requests) | 0.4624 | +0.46% |
| weight-scale broadcast deleted (16 `ds_bpermute`) | 0.4601 | −0.03% |

The hashes moved because the probe *did* reach the launches that do not take the staged branch. The
phase did not move because the launches that dominate it ran the deployed schedule in all six arms.
A distinct hash per arm proves a code path changed; **it does not prove the path you are timing
changed.**

## What the ladder says once it is connected

Probe instantiations behind `#if HALO_FFN_PROBE` inside the staged branch, and `PROBE` generalised
from the dense map to every map so the pair image can be ablated for the first time. Mode 19,
32-stream generation step, both images, one process, one clock, `ffn` normalised by
`gdn-resident-core`, `sequence-core`, `sequence-input-prep`, `sequence-input-projection`,
`sequence-output-projection` and `head-projection`:

| image | deployed | activations pinned | **weights pinned** | both pinned |
|---|---:|---:|---:|---:|
| dense five-trit | 30.451 ms | +0.60% | **−21.90%** | −21.38% |
| pair codes | 38.185 ms | +0.02% | **−26.56%** | −26.76% |

- **The weight stream is a fifth of this phase**, and [the byte law](ffn-pair-stream.md) stands.
  `32.6 → 24.6 ms` from [the decode-bytes correction](ffn-decode-bytes.md) reproduces to a tenth of
  a point four builds and two shipped schedule changes later.
- **The activation stream is closed.** Pinning every activation fragment to one cached block is
  +0.60% and +0.02%. [The LDS stage](ffn-b-operand.md) took the whole of its side; there is nothing
  left on the operand path of this kernel at this width.
- **The pair image is 17.1% slower than the dense one with both streams pinned**, so most of its
  25.7% loss is not its bytes. Its down stage runs `MATS = 2` — 160 waves — and that shape is worth
  4% of the phase on its own (below); the rest is a 622-slot block loop losing to an 843-slot one,
  which no census in this repository predicts.

## The stream geometry was never the ceiling

[`bench/wstream`](../bench/wstream.hip) carries three axes its saved panel predates — an occupancy
ladder, a half-lane column, and `k_stream_dense`, the A4 dense arm's own read byte for byte: two
416-byte runs a block through `dense_load0/1/2` off `lane & 15`. Somebody built all three and
nobody ran them. 512 MB, 3 rounds, under the lock:

| resident waves per SIMD32, ffn gate/up geometry, padded order | work 192 | work 0 |
|---:|---:|---:|
| 16 | 195.5 | 207.8 |
| 12 | 184.8 | 210.4 |
| 10 | 177.4 | 216.2 |
| 8 | 170.6 | 217.5 |
| **7 — the deployed dense gate/up** | **177.5** | 220.5 |
| 6 | 157.4 | 214.5 |
| **4 — the deployed dense down** | **126.0** | 213.8 |
| 2 — the pair down | 87.2 | 198.9 |
| 1 | 58.5 | 147.7 |

| the A4 dense arm's own read, `lane & 15`, 416-byte runs | waves | tile-major | block-major |
|---|---:|---:|---:|
| gate/up, 1088 waves, nb 40, work 200 | 7 | **194.3** | 204.7 |
| down, 320 waves, nb 136, work 100 | 4 | **198.0** | 197.3 |

- **Half-lane addressing is free.** `padded_half_lane / padded_tile` is 0.97–1.01x on all ten
  geometries. A 32-lane `global_load_b128` covering 256 distinct bytes costs what one covering 512
  costs, so the three-instruction 416-byte read the A4 images use is not a bandwidth shape at all.
  That closes a mechanism the probe's own header raised and nothing could see.
- **Occupancy is nearly free above six waves** and expensive below four. Only a grid-starved stage
  pays, which is the down stage and nothing else.
- **The deployed read at the deployed occupancy delivers 194 GB/s** where the kernel gets about 107.
  The gap is a schedule inside the block, not a stream shape — so the remaining fifth of this phase
  is a cover problem, and [the order rule](ffn-image-order.md) has already taken what the image
  layout had to give.

## Three measured arms, all bit-identical, none of them shipped

**The down stage's wave count is worth 4%.** `--ffn-dn-mats` walks the two ownerships of the down
projection's two matrices: one per wave is 320 waves, the pair is 160. Two rounds an arm, one
process, residual FNV-64 `6678137683989106976` in all four samples:

| down stage | waves | waves/SIMD32 | `ffn`/control | |
|---|---:|---:|---:|---:|
| `MATS = 1`, deployed | 320 | 4 | 0.4667 | — |
| `MATS = 2` | 160 | 2 | 0.4853 | **+3.97%** |

That is the wave-count slope this kernel actually has, and it prices the pair image's structural
disadvantage: the pair map has no split arm, so its down stage runs at two waves per SIMD32.

**Deepening the down stage's cursor on the staged body buys nothing.** With the dispatch repaired so
`--ffn-dense 4,5` reaches the deployed arm, three rounds an arm interleaved, median of `ffn`/control,
residual FNV-64 identical in all nine samples:

| down-stage weight cursor | `ffn`/control | |
|---|---:|---:|
| depth 2 — deployed | 0.4701 | — |
| depth 4 | 0.4699 | **−0.05%, a null** |
| depth 6 | 0.4834 | **+2.83%** |

[The depths were fitted on the unstaged body](ffn-down-waves.md), where the block had sixteen
fragment loads of its own to cover. The stage removed those, and with them the room a deeper cursor
was buying. **The 21.9% the weight stream still costs is not reachable by lookahead depth.**

**The deployed gate/up spills 21 dwords to hold seven waves.** `tools/kernel_resources.py` on the
staged decode instantiations, which nobody has re-read since `ScalarStream` landed:

| `k_proj_opt<IU4, ·, LANE, false, TT, 4, ·, SLICE, LOADS, MATS, WAVE, 1>` | VGPR | waves | spill | scratch B/lane |
|---|---:|---:|---:|---:|
| dense gate/up, TT 2, MATS 2 | 216 | 7 | **21** | **84** |
| dense down, TT 2, MATS 1 | 163 | 9 | 0 | 0 |
| pair gate/up, TT 2, MATS 2 | 183 | 8 | 0 | 0 |

`HALO_FFN_BSTAGE_WPE = 7` asks for the seventh wave and the allocator pays for it in scratch. The
source says "nine spilled dwords"; it is 21 now. The spill is in the token-group loop, which runs
once at 32 rows, so it is a launch cost rather than a block cost — but the 216-register budget is
carried through the block loop either way, and `-DHALO_FFN_WPE=8` was already
[a 54.7% loss on this kernel](ffn-decode-bytes.md) for the same reason. **Nobody has re-priced the
seventh wave since the register demand grew.**

## What the block loop costs, on the listing rather than on the largest basic block

`tools/isa_loop_count.py` reports the largest *basic block*; the inner loop of these kernels spans
several. Whole-loop census at the deployed decode shape, converted with `bench/wmma_cost`'s measured
11.0 slots per `v_wmma_i32_16x16x16_iu4`:

| inner loop | instructions | work | `v_wmma` | VALU | LDS | `ds_bpermute` | `global_load` | issue slots |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense gate/up TT 2 MATS 2 | 629 | 523 | 32 | 392 | 12 | 16 | 14 | **843** |
| dense down TT 2 MATS 1 | 301 | 283 | 16 | 184 | 12 | 8 | 10 | **443** |
| pair gate/up TT 2 MATS 2 | 320 | 302 | 32 | 181 | 12 | 16 | 12 | **622** |

A step runs 34,816 wave-blocks of each stage per SIMD32, so the model is 29.35M + 15.42M = 44.8M
cycles against a measured 30.45 ms — **1.47 GHz of effective issue**, on a part
[whose own WMMA probe reads 1860 MHz under a saturating matrix load](ffn-slice-issue-order.md) and
whose socket budget [is shared with sixteen Zen5 cores](power-budget.md). The weight stream's 21.9%
sits inside that gap and does not close it.

**And the cover table is the thing to hand over.** The compiled block has exactly one exposed
request and it is not the weights: `s_waitcnt vmcnt(6)` at slot 76 retires the activation stage's
own fetch, issued at slot 21, so the commit to LDS waits 55 slots after the request — while the six
weight loads get 570 slots of cover and the two scalar words 71. The stage is single-buffered
(`Stage::BUFS = 1` for `BSTAGE_WAVE`), so the wave must drain all sixteen of this block's fragments
out of LDS before it may store the next block's into the same address. A second buffer is 8 KB more
LDS per workgroup, which this shape has (8192 B of a 128 KB WGP at four waves), and it is the only
uncovered round trip left in the loop.

## Reproducing

```sh
make -j6 tools/batch_profile bench/wstream                       # the shipped ladder and the probe
tools/run-batch-compare --pin-clock --exec ./bench/wstream --mb 512 --rounds 3
make -j6 DEFS='-DHALO_FFN_PROBE=1' tools/batch_profile           # the ablation ladder
tools/run-batch-compare --pin-clock --profile-tool --modes 19 --decode-streams 32 \
  --decode-prompt 64 --ffn-dense 1,8,9,10 --ffn-image 1,2 --rounds 1 --warmup-ms 300 \
  --rows 32 --heads 1 --traces 1 --out RESULT.json
tools/run-batch-compare --pin-clock --profile-tool --modes 19 --decode-streams 32 \
  --decode-prompt 64 --ffn-dn-mats 1,2 --rounds 2 --warmup-ms 300 --rows 32 --heads 1 \
  --traces 1 --out RESULT.json
```

`-DHALO_FFN_PROBE=1` costs `kernels/ffn_batch.o` about twenty seconds of extra compile; a default
build carries neither the probe instantiations nor their dispatch. Read `ffn` divided by the phases
your arm cannot reach — a raw millisecond column on this box is a clock reading.

## What shipped

Nothing in the numerical map, no serving default, and no arm in the selected runtime. Four repairs
to the instruments, all default-identical, verified by the deployed arm reproducing residual FNV-64
`6678137683989106976` and 98.25–98.83 ms on the 32-stream step:

1. **The ablation ladder reaches the staged body.** Probe instantiations inside the staged dispatch,
   `#if HALO_FFN_PROBE` only.
2. **`PROBE` applies to every weight map**, not only the dense one, through a masked run index
   `blkw` beside `blkr`. The pair image can be ablated; the two-bit and FP16 maps come with it.
3. **The down stage's deep cursors reach the staged body** — two instantiations, `dls` 4 and 5, for
   the split dense down shape only. Both are measured above and both lose, so this restores a
   documented axis rather than adding an arm: the default build launches exactly the kernel it
   launched before, and a future engineer can re-ask at another width without rediscovering the
   defect.
4. **`ffn_batch_set_a4_dense` refuses what it cannot run.** Depths 2 and 3 have no staged
   instantiation; asking for them with the stage on now throws instead of silently running depth
   two. `tools/batch_profile` also stops refusing `--ffn-dn-mats 2`, which its own error message has
   always documented and its bound has always rejected — the 160-wave arm of the down stage was
   unreachable from the tool until this turn.

## Installed and accepted

Canonical `18b72a9`. `bonsai-halo` `859931aeac3b17f974b8ce0a`, `tools/batch_profile`
`495eb370acc528abc249e29f`, both rebuilt from the merge and both the binaries the service and the
lane run. Four cells on the installed executable under `tools/run-batch-compare` — the deployed
cursor and the restored depth-4 one, crossed with both down-stage ownerships — give residual FNV-64
`6678137683989106976` in all four, which is the value canonical publishes for this shape
(`canonical-accept.json`). Main had moved to a deferred state commit by then, so the step reads
74.98–78.54 ms rather than the 98 ms the panels above ran at; the arms are unaffected because every
one of them is normalised inside its own process.

## The next useful question

**The single-buffered activation stage is the last uncovered request in the block, and it is now
the only one.** 55 slots of cover for a round trip, once per block, in a loop with 570 slots of
cover on everything else. `Stage::BUFS` is 1 for `BSTAGE_WAVE` and 2 for `BSTAGE_SHARED`; the wave
arm was chosen for its LDS footprint at a time when the footprint mattered and the fetch did not.
Double-buffering it is 8 KB more per workgroup, no extra byte from memory, no instruction moved in
the expansion or the drain, and bit-identical by construction.

Beyond that the phase is 843 issue slots a block delivering 1.47 GHz of effective issue with one
fifth of its weight stream exposed, and **the pair image's 17.1% loss at equal, pinned memory is the
open contradiction**: 622 slots losing to 843 is not a census result, not bytes, not occupancy and
not the down stage's wave count, which is priced here at 4% of the 25.7%. Whoever explains that
owns the conversion rate for every instruction cut anyone proposes in this kernel — which is the
question three engineers have now answered "about zero" without knowing why.

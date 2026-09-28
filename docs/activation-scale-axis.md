# The activation scale's axis, and two nulls that price the census

Every integer matrix kernel in this engine reads a per-(token, K block) activation scale once per
token tile per block, and all three store it with the **token** on the contiguous axis while reading
it with the **token on the lane axis**:

| kernel | read | blocks | bytes between lanes |
|---|---|---:|---:|
| `ffn_batch.hip` | `cscale[(tg + t*16 + col) * nb + blk]` | 40 gate/up, 136 down | 160, 544 |
| `sequence_batch.hip:124` | `scp[t*16*w.nb + b]`, `scp = sc + (first+col)*w.nb` | 48 | 192 |
| `head_batch.hip:112` | `a.scales[token*NB_D + kblk]` **and** `a.sums[...]` | 40 | 160 |

`col` is `lane & 15`. Every stride is over 128 bytes, so the sixteen lanes of one load land in
sixteen different cache lines: the load uses 64 bytes and asks L0 for sixteen lines. The *weight*
block scales in the same kernels are already tile-major with the sixteen a wave needs in 32
contiguous bytes, and `k_proj_int` and `sequence_batch.hip:84` both carry the comment saying why.
Only the activation side was missed, in all three.

Counted on a 128-row A4 pass: gate/up issues 2176 waves x 40 blocks x 4 token tiles and the down
stage 640 x 136 x 2, so **33.4 M drain scale loads ask for 534 M cache lines** to deliver 2.1 GB of
scales. Stored block-major the same loads ask for about 17 M, because the sixteen lanes fall in one
line and that line covers the next token tile too. Per token group a wave also pulls 4 KB of these
lines into an L0 that ten resident waves share 32 KB of, so the scales evict the B fragments and the
weight stream the same loop is reading.

That is a 32x amplification of cache-line requests, and removing it **changes nothing measurable**.

## The FFN transpose: bit-exact, and a null

`cscale[block][token]`, derived by `cs_stride(layout, nb, Npad)`, written by `k_prep` and read by
the three integer drains. Same values, same drain order, so it is a permutation of one buffer rather
than an arithmetic change. `--ffn-cscale 0,1` walks both layouts inside one process.

One process, mode 19, both layouts interleaved, `--rows 32,128 --heads 0 --traces 1 --rounds 2`:

| | token-major | block-major | delta |
|---|---:|---:|---:|
| **32 rows** `ffn` | 33.93, 31.88 | 30.85, 32.14 | -4.29% |
| `gdn-resident-core` *(control)* | 5.35, 4.89 | 4.64, 5.01 | -5.83% |
| `sequence-core` *(control)* | 4.30, 3.91 | 3.73, 3.97 | -6.26% |
| **128 rows** `ffn` | 80.00, 72.84 | 81.13, 76.33 | +3.02% |
| `gdn-resident-core` *(control)* | 16.87, 15.34 | 17.29, 16.12 | +3.71% |
| `sequence-core` *(control)* | 15.46, 15.21 | 16.87, 15.57 | +2.79% |

Normalised on the untouched phases: **+1.4% at 32 rows and -0.4% at 128**, opposite signs. The
control phases are the same launches in both arms by construction and disagree with each other by
2.8 points at 32 rows, so this panel resolves nothing below about 3%; what it establishes is that
the transpose is not a 5% change either way.

Bit-exact on device, not only by construction: residual FNV-1a `1873717996769712938` at 32 rows and
`7446865760224376151` at 128, identical in every arm.

Raw: `batch-comparison/ffn-cscale/phase-m19.json`. Source `52fff7d` on branch
`ffn-cscale-handoff` in `work/clones/bonsai-ffn-cscale`, kept only as provenance for this panel.
The transpose reached master inside `7ea7bf2`, the operand cursors in
[`docs/ffn-operand-address.md`](ffn-operand-address.md), because it is the **enabling** half of
that change and not a change on its own: a token-major `cscale` cannot offer the wave-uniform base
the cursor form needs, since the lane is the axis carrying the `nb` stride. The panel above is why
it is not credited with any of that bundle's +3.7%.

## The head cursor: 6.2% fewer instructions, a null, and a change that never happened

`head_tile` reads `a.sums[(size_t) token * NB_D + kblk]` and `a.scales[...]` per (token group, K
block). The index mixes the lane's own column with the wave-uniform block inside a 64-bit
expression, so each of the two loads rebuilt an address pair, exactly the shape the `boff` operand
cursor above it already fixed for the B fragment. The lane part is fixed for the whole K walk and
the block part advances by one, so a 32-bit offset on a uniform base *should* reach the same
elements in the same order from a scalar base.

Block loop of `head_tile<2>`, the width `run_head_batch` selects above 16 rows: **594 to 557
instructions, -6.2%**.

**It did not take.** The emitted memory address forms of that block loop are byte-identical between
the two builds — two `global_load_b128`, six `global_load_b32` and two `global_load_b64` on 64-bit
VGPR address pairs, sixteen `global_load_b128` on a scalar base, in both. The source index changed
and the addressing did not, which is exactly why only four work slots moved. The panel below is
therefore **not** evidence about the cursor form in the head: that is untested. The reason the
attempt failed is recorded so the second one does not repeat it — the divergent half has to be
32-bit *and* the base wave-uniform, and with the token on the lane axis carrying an `NB_D` stride
there is no uniform base to split out at all until the array is transposed. In the head that means
`a.scales` and `a.sums` want the block-major layout and the cursor together, which needs the shared
producer behind `FwdParams::sequence_scales` and so reaches `prep_batch.hip` and the sequence
projection's own reader.

One process, mode 19, both arms interleaved, `--rows 32,128 --heads 1 --traces 1 --rounds 3`:

| | deployed index | cursor | delta | normalised |
|---|---|---|---:|---:|
| 128 rows `head-projection` | 10.89, 10.75, 10.60 | 10.34, 10.72, 10.68 | -0.65% | **-0.59%** |
| 32 rows `head-projection` | 3.06, 2.69, 2.71 | 2.76, 2.73, 2.70 | +0.79% | **+1.12%** |

The controls in this panel are tight — `ffn`, `gdn-resident-core` and both sequence projections all
inside 1% — so the resolution is real, and the result is a null at both widths, which is what a
build issuing the same loads from the same addresses should measure. Residual FNV-1a unchanged in
every arm. **Not in the tree.** Raw: `batch-comparison/ffn-cscale/head-scale-m19.json`, with the
standalone FFN transpose kept beside it as a patch.

## Why both of them are nulls, which is the part worth keeping

Split the block loops by pipe instead of counting instructions:

| block loop | total | `s_delay_alu` | `s_waitcnt` | **work** | measured |
|---|---:|---:|---:|---:|---|
| head `TT` = 2, deployed | 594 | 80 | 22 | 492 | |
| head `TT` = 2, cursor | 557 | 47 | 22 | **488** | null |
| head `TT` = 1, deployed | 434 | 24 | 11 | 399 | |
| head `TT` = 1, cursor | 459 | 51 | 11 | **397** | |
| FFN gate/up, token-major | 559 | 7 | 36 | 516 | |
| FFN gate/up, block-major | 570 | 12 | 42 | **516** | null |

Both changes moved **real work by four slots and zero**. The raw totals said -6.2% and +2.0% and
were wrong in both directions, because `s_delay_alu` moved by 33 in one and the waitcnt split by six
in the other. The apparent 5.8% *regression* at `head_tile<1>` is a two-slot improvement in work.

`s_delay_alu` carries no operation and the scheduler places it freely; `s_waitcnt` marks where a
wave stops rather than what it does. **A census delta made of these predicts nothing.**
`tools/isa_loop_count.py` now splits them out and prints `N work + M scheduling` on every block, so
the next census reads the right number without anyone remembering this.

The head case adds the other half of the check, and it is the one that would have saved a lock
session. **An addressing change is not done when the source index changes; it is done when the
emitted load form changes.** The same tool now prints an address-form table per kernel:

```
== memory address forms ==
   global_load_b32      64-bit VGPR address pair      6     <- the wave built and holds this address
   global_load_b128     scalar base + lane offset    16     <- what a cursor is supposed to produce
```

Identical tables across two builds mean the addressing did not move, whatever the slot count says.
The operand-cursor work hit the same wall from the other side: `readfirstlane` on the wave index
alone took its block 559 to 558 slots, because a `size_t` index still forced the 64-bit add. Both
halves have to land together, and the table is how you tell in seconds and without a GPU.

One published model in this repository is priced off a total that includes them, and should be
re-derived before it is built on: [resident GDN](resident-gdn.md) puts the recurrent token loop at
355 slots and explicitly names **70 `s_delay_alu`** inside that figure. At 285 work slots the same
arithmetic predicts 11.6 ms against 16.3 measured, which is 71% rather than the 88.6% the document
claims — so the gap in that phase may not be as closed as it reads. The 545-slot arm has the same
question. Nothing in that measured 32% gain changes; only the issue-model claim on top of it does.

## What these two nulls rule out

- **The line-request path is not a bottleneck in either kernel at any width measured.** A 32x cut
  in cache lines requested per pass, and the L0 pressure that comes with it, is worth under 1%. Stop
  proposing uncoalesced scale gather as an explanation for the FFN's missing 30 ms or the head's
  distance from its issue model.
- **Address arithmetic pays only when it removes VALU.** The operand-cursor result that does
  convert removes 26 of the gate/up block's 38 `v_add`, takes it 256 to 229 VGPR, five waves per
  SIMD32 to six, and kills its 52-byte spill; it measures -5.56% on the FFN phase and +3.7% full
  model. These two removed none of that. The form is not the point, the freed work is.
- **And freed work converts only where the kernel is issue-bound.** The same bundle cuts 9.4% of
  the instructions from the dense arm a generation step runs and measures an exact null there,
  because below 32 rows that arm sits near a third of its issue model waiting on operands. Three
  changes, three different reasons to be a null: no work freed (both of mine) and work freed in a
  kernel that was not issuing (theirs). Establish which of the three you are in before spending a
  panel.

Still open, in the order their size suggests: the shader clock a WMMA-dense pass actually sustains,
which would retire several peak-rate claims at once; the operand cursors in
[`docs/ffn-operand-address.md`](ffn-operand-address.md); and the vectorised form of this same
buffer, which nothing here tests. Storing `[block/4][token][block % 4]` lets one `global_load_b128`
carry four blocks' scales with the sixteen lanes contiguous — one instruction and two lines per four
blocks against block-major's four and four. It costs four live floats per token tile across four
block iterations, sixteen registers at `TT` = 4 on a shape already at 256 with a twelve-byte spill,
and it needs the block loop split into an outer group loop. On the evidence above it should be a
null too, and it is only worth building if something first shows that the drain's load *instruction*
count matters.

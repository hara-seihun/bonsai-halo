# A row's dot products were one chain, and the compiler was fencing every pair of them

The dot4 matvec body `mv_rows_t` became the engine's most-served kernel an hour before this work
started. [`40499e3`](matvec-crossover.md) moved ternary weights at five to eight rows off the WMMA
body, so the eight-row verify pass that is [five sixths of a drafted generation
step](serving-decode.md) — what `bonsai-halo.service` runs — now spends its time here, alongside
single-token decode, which always did.

## Result

Both arms are one revision (`f982391`) and one source tree, differing only in `HALO_MV_ACC`.

| | before | after | |
|---|---:|---:|---:|
| drafted verify pass, per step | 52.6 ms | **48.65 ms** | −7.5% |
| drafted generation | 66.56 tok/s | **70.95 tok/s** | +6.6% |
| prompt ingestion, deployed 8-row passes | 160.5 tok/s | **171.6 tok/s** | +6.9% |
| single-stream `--bench` | 33.59 tok/s | 33.53 tok/s | null, untouched code |
| `k_forward_rows<8>` | 135 VGPR, 10 waves/SIMD32 | 135 VGPR, 10 waves/SIMD32 | — |

**Bit-identical.** 248,320 full-vocabulary logit floats of a 433-token prompt compare byte for byte
equal between the arms and across a repeat of each arm — one hash, `fe246bcf…`, in all four files.
Every drafted run digests its greedy token stream to the same value and reports the same 210
drafted and 104 accepted, in both orders. Raw samples, binaries and hashes:
[`batch-comparison/mv-acc-chains/`](../../../data/bonsai2/batch-comparison/mv-acc-chains/README.md).

## The body is over two budgets at once, so it is not short of either

Per 32-row, 128-K tile block, one wave, from the emitted gfx1151 assembly:

| | TT = 1 | TT = 8 |
|---|---:|---:|
| trit expansion (peel) | 232 | 232 |
| `v_dot4_i32_iu8` | 32 | 256 |
| activation operands, drain, addressing | 81 | 263 |
| **work slots** | **345** | **751** |
| `s_delay_alu` + `s_waitcnt` | 19 | **174** |

A whole pass over the 5.9 GB weight image is 7.07M of these blocks. At 80 SIMD32 and 2.4 GHz that
prices the eight-row pass at 34.1 ms of issue including its scheduling slots, and 5.9 GB at the
242 GB/s `bench/bw` measures is 26.5 ms. It took **49.8**.

One row is the opposite: 364 slots demand 2.46 B/cycle/SIMD, the machine supplies about one, and
single-token decode runs its matvecs at **240 GB/s of a 242 GB/s roof**. That is why
[three exact schedule changes](single-stream.md) measured null there and why this one does too.
Eight rows demand 0.97 B/cycle/SIMD. **The eight-row body is the only shape in this kernel that is
short of neither bytes nor slots**, and that is where the third resource shows up.

## The third resource, named by the compiler

The row loop accumulates a row's 32 dot products into one register:

```
v_dot4_i32_iu8 v51, v12, s8,  0    neg_lo:[0,1,0]
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_dot4_i32_iu8 v51, v24, s9,  v51  neg_lo:[0,1,0]
v_dot4_i32_iu8 v51, v14, s10, v51  neg_lo:[0,1,0]
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
```

Every dot4 reads the register the previous one wrote. `VALU_DEP_1` is the compiler telling the
hardware that the operand is one instruction old and the wave has to wait for it. At eight rows
there are **128 of those hints per block**, one per pair of the 256 matrix instructions, and the
block carries 174 scheduling slots against 751 of work. The rows cannot cover for each other: a
`sched_barrier` between them keeps each row's 32-dword activation load out of the next row's, so
the wave holds exactly one dependent chain at a time by construction.

Split the row's accumulator into four and the chains interleave:

```
v_dot4_i32_iu8 v51, v30, s39, 0  |  v_dot4_i32_iu8 v52, v24, s38, 0
v_dot4_i32_iu8 v53, v14, s36, 0  |  v_dot4_i32_iu8 v54, v29, s37, 0
v_dot4_i32_iu8 v51, v28, s43, v51 | ...
```

| chains per row | work | scheduling | the hints that remain |
|---:|---:|---:|---|
| 1 | 751 | 174 | 128 × `VALU_DEP_1` |
| 2 | 765 | 160 | 107 × `VALU_DEP_2` — **still stalling** |
| **4** | **758** | **83** | 27 × `VALU_DEP_4`, 7 × `VALU_DEP_3` |
| 8 | 807 | 73 | —, and 56 work slots more |

Two chains is not half a fix, it is no fix: the dependence just moves out one slot and the hardware
still waits. The chain needs three or more instructions of separation, and past four the extra
accumulators cost more slots than the hints they remove. Four is the knee.

## Why this is bit-identical and not a reassociation

A row's dot product is a sum of `NBLK * 32` terms, each a `v_dot4` of four unsigned trits against
four signed int8 activations, so each is bounded by `4 * 2 * 127` and the whole sum by 130,048.
**Integer addition in int32 is exact and associative over that range**, so the four partial chains
and the single chain produce the same integer, feed the same `acc - xsum`, and hand the same bits
to `fmaf(acc, wscale * xs, y)`. Nothing narrows and no float is reordered.

This is the distinction the lane keeps having to make: an algebraic identity over the reals is not
a proof about FP32, but this identity is over int32, where it is a proof. The measurement agrees
anyway — 248,320 logit floats, four files, one hash.

## Two bounded negatives from the same census, so nobody re-derives them

**The two-trit palette is subsumed.** `pal_block` lifts the nine-valued two-trit codes out of the
packed bytes and uses `v_perm_b32` as an eight-byte palette, and it really is cheaper: expansion
271 → 238 slots, which is −9.0% of a whole TT = 1 block and −3.6% of a TT = 8 one. But once the
accumulator is split it buys **nothing**: four chains alone and four chains plus the palette are
both 841 total slots at eight rows, and at one row the body is already at 99% of its bandwidth
roof, where slots are not what it is short of. That is the second time the palette has measured
null on this device and the first time with a reason attached — it was competing for the slack the
dependence had already taken.

**Relaxing the inter-row scheduling barrier does nothing.** `sched_barrier(0)` between rows fences
every class, and [the dense FFN block](ffn-dense-loads.md) taught the lane that a barrier should
name the pressure it contains. Here the pressure is SMEM — without it the compiler hoists all eight
rows' activation loads and blows the SGPR file — so mask `0x2`, which lets VALU cross, looks like
the same fix. It emits a **literally identical instruction census** at every row count and chain
count tried. The rows cannot interleave because each row's dot products depend on scalar loads the
barrier still pins, so the only way to give this wave a second chain is inside a row. The FFN
lesson does not transfer; the shape of the dependence decides, not the class of the fence.

## What this leaves

- **`k_forward_rows<8>`'s 135 VGPRs are not the matvec's.** `mv_rows_t<5, 8, false>` compiled alone
  needs 61 VGPR and 107 SGPR; the deployed single-token kernel is 95 VGPR at sixteen waves per
  SIMD32 and the eight-row one is 135 at ten. Forty registers of that kernel come from some other
  phase in the persistent body, and **128 VGPRs would buy twelve waves per SIMD32** — grid 120
  against today's 100, on a pass whose grid sweep improved at 60, 72, 84 and 100 without ever
  finding a ceiling it did not want more of. Seven registers anywhere in `k_forward_rows` are worth
  more than anything left in this body. Whoever owns `ph_gdn`, `ph_prep` or `ph_attn` is holding it.
- **The SM > 0 palette map has the same bare chain and nobody has re-measured it.**
  `mv_rows_pal` expands the whole block first and then issues 32 back-to-back dependent `v_dot4`
  with nothing to interleave — the exact shape this document repaired, and a candidate explanation
  for why that map lost. It is one line to give it four chains, and one panel to find out whether
  the single-token map was ever really a loss.
- **Q8 at two and four rows rides the same change and was not isolated.** The drafter's block pass
  was 11.5 ms before and 11.4 after, which is a null inside this box's spread, and its weights take
  a different body at eight rows anyway.
- **The activation stream is still per tile.** A wave re-issues the same 32 scalar dwords per row
  per block for every tile it visits, and the workgroup's working set is about 40 kB against a
  16 kB scalar cache. The unit decomposition is in `ph_matvec`, not here.

## The instrument

`kernels/mv_body_isa.hip` instantiates this body alone across row counts and chain counts.

```sh
hipcc --offload-arch=gfx1151 --cuda-device-only -S -O3 -Isrc -Ikernels kernels/mv_body_isa.hip -o /tmp/mv.s
tools/isa_loop_count.py /tmp/mv.s ILi5ELi8ELi4EE      # PW = 5, TT = 8, ACC = 4
tools/kernel_resources.py kernels/mv_body_isa.hip k_mv_census
```

Eight seconds a sweep against about forty for the persistent kernel and a GPU lock for a panel.
`tools/kernel_resources.py` now takes `-D NAME=VALUE`, which is how the wide route's shared
`k_ffn_slice` grid was cleared without building two engines: every TT ≤ 8 instantiation holds
sixteen waves per SIMD32 in both arms, and the minimum that sets the launch grid comes from
`k_ffn_slice<16>` and `<32>`, which this change cannot reach.

**One measurement lesson, paid for.** The wide-deployed prompt route measured −9.5% in two separate
lock holds and +9.3% in one hold that alternated the arms, on a route with no mechanism to move
either way. Cross-process pairs on this box are worth what the lane keeps saying they are worth.
Every number in the table above comes from a single lock hold that ran both arms, and the drafted
panel ran them in both orders.

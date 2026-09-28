# A generation step's FFN is priced in bytes, and both of its weight coordinates read them at the same rate

> **CORRECTION, and it is about the causal claim rather than the measurements.**
> [The activation operand's ablation](ffn-b-operand.md) pinned this kernel's weight stream to one
> block so that every request hit cache, and the phase fell only from 32.6 to 24.6 ms. **The bytes
> are worth 7.9 ms of 32.6, not the phase**; what remains is the block loop at its issue model. The
> table below is a real correlation between two arms and the wrong thing to choose work against.
> Two of its closing items are also now measured: the sixteen `ds_bpermute` of the weight-scale
> broadcast are **not** free money - removing them costs 2.4% - and the B fragments' cost is their
> request *count*, not their locality, which is what
> [the LDS stage](ffn-b-operand.md) ships against.

> **CONFIRMED four builds later, on a ladder that had to be repaired first.**
> [The stream ceiling](ffn-stream-ceiling.md) re-ran this pin on the body the engine now launches
> and reads **−21.9%** for the dense image and **−26.6%** for the pair one, so `32.6 → 24.6 ms`
> reproduces to a tenth of a point across the activation stage, the scalar stream and the image
> order rule. The activation side has since closed completely: pinning every fragment to one cached
> block is +0.6%. Two cautions from that turn. The ladder reaches the staged body only since
> September 21 — `launch_opt_sp` dispatched the activation stage first and baked `DLS_LOADS` into
> it, so a `--ffn-dense 8..13` panel taken between 21:30 and that repair measured the deployed arm
> under six labels and produced six different residual hashes doing it. And the "2.4%" for the
> `ds_bpermute` broadcast above is from a run on that path; re-measured on the connected ladder it
> is −0.03%.

The A4 FFN is the largest phase of a batched generation step and the lane has now spent four
iterations cutting its instructions for nothing: the two-trit palette (12% of the block, null), the
A4 block re-addressing (9.4%, null), the drain interchange (null), and the operand-order work whose
own document recorded a conversion rate of "about a third" for slot removal. This is why, stated as
a law with the two coordinates as its two data points.

**At 32 rows the phase is stream-rate bound at about 110 GB/s, and the ratio between its two weight
images is exactly their ratio of bytes.**

| arm, 32-stream step, mode 19 | GB read per step | `ffn` ms | GB/s |
|---|---:|---:|---:|
| dense five-trit, selected by the row rule | 3.476 | 32.425 | **107.2** |
| pair codes, deployed schedule | 4.278 | 40.407 | **105.9** |
| pair codes, one block of operand lookahead | 4.278 | 38.331 | **112.5** |

Byte ratio pair / dense is 1.2308; the measured time ratio is 1.2142 with the deployed schedules.
The two images decode to the same weights through the same nine-valued alphabet, so the residual
FNV-64 is `10815391166346028052` in all eight samples of both panels and the `ffn` row is the entire
difference between them.

## Why that is the useful sentence

The pair block and the dense block are not remotely the same amount of issue. Counted on the
instantiation each arm actually launches at this shape - `k_proj_opt<IU4, WM, LAY, SHARE=false,
TT=2, WV=4, FUSE>`, `tools/isa_loop_count.py`, no GPU:

| block loop, decode shape | pair codes | dense five-trit |
|---|---:|---:|
| instructions | **303** | **602** |
| `v_wmma_i32_16x16x16_iu4` | 32 | 32 |
| operand expansion | 86 (`expand_pair`, 0.34 VALU/weight) | 279 (radix-3 peel, 2.17) |
| `global_load` | 24 | 26 |
| `s_waitcnt` | 12 | 29 |
| VGPRs / waves per SIMD32 | 194 / 7 | 206 / 7 |

**The arm with half the issue and the same matrix work is 21% slower, and it is slower by its
bytes.** That is a much stronger statement than another null: it rules out the block census as the
binding constraint at this shape from the opposite direction, and it explains every earlier null at
once. Anything that removes non-matrix slots from this kernel at a generation shape is removing
slack.

(The 145 VGPR quoted in `ffn_batch.hip`'s pair comment is stale by a body or two; the shape is 194.)

## And the rate is 46% of the roof, with the geometry's own probe at 202

`bench/bw` measures 242 GB/s on this box. `bench/wstream` reads this exact image geometry with no
model in the picture - 2176 tiles of 40 blocks, 512-byte blocks, the padded stride the
[stride rule](ffn-image-order.md) now stores - and delivers **202.2 GB/s of distinct bytes**. The
kernel reads the same image at **112.5**, which is 56% of what its own stream geometry gives
standalone.

So the gap is not the image layout, which the stride rule already fixed, and not the instruction
count, which is half in the faster-per-byte arm. It is what the kernel does *between* its requests.
The compiled pair block says it exactly:

```
LLLLLLLLLLLLLLLLLLLLLLLL W DDDDDDDDDDDDDDDD WWWWWWWWWWW MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
```

Twenty-four `global_load`, one wait, sixteen `ds_bpermute` for the weight-scale broadcast, eleven
waits, then all thirty-two matrix instructions back to back. **A wave issues its whole block of
requests, waits, and then spends 512 cycles of matrix work with nothing outstanding.** The dense
arm, which has the block cursor, interleaves its B-fragment loads with its matrix instructions
instead, and the dense arm is the one hitting the higher fraction of its own byte budget.

## What one block of lookahead on the pair arm is worth (measured, and not shipped)

`ffn_dense_stream.hpp`'s `DenseStream` requests a block's weight bytes a block before handing them
out. [The 128-row version of this question](ffn-pair-lookahead.md) measured that cursor on the pair
arm at four token tiles and rejected it at +7.2%, on the grounds that gate/up at 128 rows already
keeps five or six waves resident per SIMD32. The two-token-tile cell it left open is a different
trade, and it is the one a generation step runs.

`PairStream`, one block deep, costs **seven registers and no wave slot** (194 -> 201 VGPR, both the
seven-wave granule, against 218 and six waves for the first version of the same cursor - that one
folded the lane index into the cursor's pointer and lost the scalar base, exactly as
[the head](wide-head.md) warns).

| panel | arms | `ffn` off -> on | normalised |
|---|---|---|---|
| 32 streams, `--ffn-dense 0,1`, one round | 39.369 -> 38.024 | control +0.28% | **-3.69%** |
| 32 streams, `--ffn-dense 1,0`, two rounds | 40.407 -> 38.331 | control -2.2% | **-3.86%** |

Two processes, reversed arm order, four pairs, residual FNV-64 identical in every cell. It is a real
movement of that 110 GB/s - the first one this lane has measured - and it is **not shipped, because
nothing served runs the arm it improves**:

- At 32 rows the row rule selects the dense image, and the dense image is still 15 to 18% ahead of
  the cursor's pair arm, for the same reason as everything else here: it reads 19% fewer bytes.
- Above 32 rows the pair arm runs `TT = 4`, where the same cursor is a measured +7.2% loss and where
  this implementation leaves it off by construction.
- The only shape that runs pair codes at `TT <= 2` is a 33-to-63-row pass, which no route generates
  today. If [the sub-64-row route](decode-width-route.md) lands, `pair-tt2-cursor.patch` beside the
  raw samples applies to that arm and is worth the 3.9% above.

The arm is therefore out of the tree rather than switched off in it. Raw panels, the block census,
the compiled block sequences and the patch are in
[`batch-comparison/ffn-decode-bytes/`](../../../data/bonsai2/batch-comparison/ffn-decode-bytes/).

## Two things the next engineer here should not have to rediscover

- **Prefill width and generation width are bound by different things, and this document is only
  about the second.** At 256 rows the same kernel is at 89% of its own block census and 31 GB/s;
  at 32 rows it is at 46% of the memory roof and its census is slack. A change fitted on one will
  read the wrong sign on the other, which is now the fifth time this kernel has said so.
- **An in-panel control does not resolve 4% at prefill width on this box.** Two launches of a
  *byte-identical* kernel (the `TT = 4` pair instantiation, verified identical instruction for
  instruction between builds) measured 80.913 and 85.108 ms at 128 rows - 3.9% apart - with their
  untouched phases 1.2% apart. The 32-stream decode cells in the same session held their controls to
  0.3% and reproduced the same effect twice. Take generation-shape measurements for anything under
  5%.

## The eighth wave is available, free of spills, and it costs 55% of the phase

The first candidate this document had for that gap was occupancy: a phase waiting for requests wants
more waves issuing them, and the dense arm's 206 VGPR is one granule above the 192 that buys eight
waves per SIMD32 instead of seven. **The registers are there for the asking and the wave is a
disaster.**

`-DHALO_FFN_WPE=N` puts `amdgpu_waves_per_eu(N)` on `k_proj_opt` (default 1, which is
codegen-identical to no attribute at all: 254,282 instruction lines in the device listing, the only
differing line being the HIP compilation-unit hash). What the allocator does when asked for eight:

| instantiation | default | `waves_per_eu(8)` | scratch |
|---|---:|---:|---:|
| dense gate/up, decode shape (`TT=2, WV=4, FUSE`) | 206 VGPR / **7 waves** | 191 / **8** | 0 |
| dense down, one matrix per wave | 120 / 12 | 118 / 12 | 0 |
| pair gate/up, prefill shape (`SHARE, TT=4, WV=2`) | 235 / **6** | 192 / **8** | 0 |

**Zero spilled registers and zero scratch in every one of them** - the allocator was spending slack,
not holding liveness, exactly as [the persistent kernel's attention unit](rows-grid-occupancy.md)
was. And the panel, same source, same build machinery, both images, two rounds, residual FNV-64
`3042000206086707537` in all eight cells:

| arm | default `ffn / control` | `waves_per_eu(8)` | change |
|---|---:|---:|---:|
| dense five-trit, the selected arm | 0.48818 | 0.75507 | **+54.7%** |
| pair codes, same build, same attribute | 0.59754 | 0.59533 | **-0.4%** |

The pair arm is the control that makes this unambiguous: it allocates 194 registers already, so the
constraint does not bite, and it does not move. The dense arm gives up fifteen registers with
nothing spilled and loses more than half of its phase.

**So those fifteen registers are instruction-level parallelism, not storage.** The radix-3 peel keeps
five expanded words per source dword live and the block interleaves four matrix instructions with
the next slice's expansion; squeezed into 191 registers the allocator reuses them, the false
dependencies serialize the peel against the matrix pipe, and no spill counter shows it. This is the
second distinct place in this engine where a register count and an occupancy step disagreed about
which way to trade - the first was the persistent kernel, where the *opposite* answer won - and the
difference is whether the constrained body has independent work to lose.

**Do not ask this kernel for occupancy.** The instrument is in the tree so the next engineer can
re-ask in one compile rather than re-deriving the table: `tools/kernel_resources.py` or a
`--cuda-device-only -S` listing with `-DHALO_FFN_WPE=8`.

## The next question, and it is worth more than this one

Closing the 112 -> 202 GB/s gap is **15 ms of a 102 ms generation step**, which is larger than every
kernel result this lane has published on this phase put together. Occupancy is now closed off; the
two remaining candidates both keep the register budget where it is:

1. **The B fragments, which are 16 of the 24 requests.** Every one of the four waves of a workgroup
   owns a different row tile and re-loads the *same* `(block, slice, token)` fragment; the whole
   block's distinct B bytes are 2 kB. Staged once into LDS with a block of lookahead, a workgroup
   issues two `global_load_b128` where it now issues 64 loads, and the block's reads become
   `ds_read_b64` with no `vmcnt`. [The pair-lookahead note](ffn-pair-lookahead.md) named this from
   the other side ("the side with room is the activations") and nobody has built it.
2. **The weight-scale broadcast, which every projection kernel in this engine pays.** Those sixteen
   `ds_bpermute_b32` per block are `__shfl(asl, 2*r + half)` placing one uint16 scale per
   accumulator row, and the block *waits* on them before its first matrix instruction. The sixteen
   scales of a `(tile, block)` are wave-uniform - 32 bytes - so a scalar load plus one `v_cndmask`
   per accumulator row replaces them with no LDS pipe and no `lgkmcnt`. The same sixteen are in
   `sequence_batch.hip`'s `project` and in `head_tile`.

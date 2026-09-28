# What a generation step actually dispatches, and what its FFN is waiting for

[The decode map](decode-map.md) measured a 32-stream generation step by phase bracket and found the
recurrent state at the memory roof. This is the same step measured one level down, by kernel
instantiation, which is where the two FFN stages separate: the engine brackets both of them as
`ffn`, and they are different shapes with different diseases.

## One step, by dispatch

Mode 19, 32 slots, 64-token prompt per stream prefilled outside timing, one real generation step
with logits and argmax on every row. `rocprofv3 --kernel-trace` over the whole process, then
`tools/kernel_trace.py` segments the dispatches on host gaps and reports the last step. Source
`16c5d16`, untraced arm of the step:

| kernel | n | ms | share | per call | VGPR | grid |
|---|---:|---:|---:|---:|---:|---|
| `resident_state<4>` | 48 | 46.437 | 38.1% | 967 us | 56 | 1572864 |
| `k_proj_opt<IU4, DENSE, LANE, false, 2, 4, true>` | 64 | 20.922 | 17.2% | 327 us | 200 | 34816 |
| `project<2, false, true>` | 64 | 16.208 | 13.3% | 253 us | 104 | 32768 |
| `k_proj_opt<IU4, DENSE, LANE, false, 2, 4, false>` | 64 | 13.795 | 11.3% | 216 us | 200 | 5120 |
| `project<2, true, false>` | 64 | 9.830 | 8.1% | 154 us | 104 | 10240 |
| `k_forward_rows<8, 0, 2, 1>` | 64 | 5.021 | 4.1% | 78 us | 104 | 30720 |
| `head_tile<2>` | 1 | 2.767 | 2.3% | | 144 | 256x7760 |
| `resident_conv`, `prep_meta`, `resident_output`, `k_prep` x2, `prep_chunk`, `head_argmax` | 274 | 3.9 | 3% | | | |

733 dispatches, span 121.942 ms, of which 119.426 is covered by a dispatch. **2.1% of a generation
step is the gap between kernels**, so nothing in this step is waiting on launches, and widening a
phase to save dispatches buys nothing here.

The two `k_proj_opt` rows are the FFN's gate/up and down stages. The grid is the useful column:
gate/up launches 34816 threads at 128 per workgroup, 1088 waves for 80 SIMD32 units; down launches
5120, which is **160 waves, two per SIMD32**. Down owns `DH/16 = 160` row tiles and at 32 rows one
wave covers the whole batch in two token tiles, so there is nothing else to spend the machine on.

## The arm is issue-bound, not byte-bound

At 32 rows this arm moves 3.42 GB of dense five-trit weights per step. Against the 242 GB/s that
`bench/bw` measures, the FFN's 34.7 ms is 109 GB/s on gate/up and 82 on down: a third to a half of
the roof, so the 19% the dense image saves over pair codes is buying nothing in this shape.

What it costs instead is issue. The compiled block loops, counted from
`hipcc --cuda-device-only -S` on `gfx1151` (innermost loop only, one 128-K block of one wave):

| instantiation | instructions | WMMA | VALU | five-trit decode | VMEM | LDS |
|---|---:|---:|---:|---:|---:|---:|
| dense gate/up TT=2 | 666 | 32 | 472 | 321 | 26 | 16 |
| dense down TT=2 | 658 | 32 | 471 | 322 | 26 | 16 |
| pair gate/up TT=4 | 508 | 64 | 252 | 80 | 42 | 16 |
| pair down TT=2 | 362 | 32 | 184 | 80 | 24 | 16 |

At 16 cycles per IU4 WMMA a dense block is about 1146 cycles of issue and a pair block about 842 at
the same token width. Lifting five trits out of a byte is 321 of the 666 slots; the matrix
instructions are 32 of them. Peer `2e6fc74a` reached the same census independently and owns the
operand build and the register footprint that follow from it.

Measured against that model, gate/up runs at 1.27x and down at 2.0x. Two waves per SIMD32 cannot
cover a memory round trip between them, which is what the down stage's extra factor is.

## More waves off the token axis is not the answer

The obvious response to 160 waves is to split the token axis: give the row tile to a workgroup and
one token group to each of two waves, which is what `HALO_FFN_W_DN=2 HALO_FFN_TT_DN=1` selects and
what `SCHED_ADAPT` would pick if the dense arm asked for it. Measured on the same step, with the
untouched kernels as controls:

| kernel | tile ownership | shared, W=2 TT=1 |
|---|---:|---:|
| FFN down | 13.795 | 13.086 |
| FFN gate/up (control) | 20.922 | 20.997 |
| `resident_state<4>` (control) | 46.437 | 46.159 |

Doubling the resident waves bought 5%, because each added wave re-reads and re-expands the same
weight bytes for half the tokens: 320 waves at 640 instructions a block is 55% more issue than 160
at 658. `2e6fc74a` measured the whole-arm version of the same trade at 33.8 to 42.7 ms. On the pair
image the sign flips, because a pair wave costs 80 expansion slots instead of 321 and the extra
waves are nearly free; that is why the same ownership is worth 21 ms there and a loss here.

## Keeping the stream in flight

Two places in this arm waited on memory nothing had asked for yet.

**The block loop had no lookahead.** A wave asked for a block's 26 operand bytes per matrix and
consumed them a few instructions later, so it paid a memory round trip per block that only other
resident waves could cover, and the down stage has two. [`ffn_dense_stream.hpp`](../kernels/ffn_dense_stream.hpp)
holds two blocks in flight in registers and hands out the oldest, so the wait a wave pays has
already been overlapped with a whole block of expansion and matrix work. One block of lookahead
covers about 1146 cycles of issue and measured 11.4 ms on the down stage; two covers 2292 and
measured 9.8. The cursor costs 54 VGPRs at depth two (200 to 254, no scratch), which is free on a
stage the grid holds to two waves and takes gate/up from seven resident waves per SIMD32 to six.

**The residual add serialized itself.** The down stage finishes with `out[o] = resid[o] + acc`, and
`run_opt` passes the same pointer as `resid` and `out`, so neither can be `__restrict__`. Written as
sixteen read-add-write triples, the compiler has to assume each store may land on the next load's
address, and it emitted sixteen `(global_load, s_waitcnt vmcnt(0), v_add_f32, global_store)` chains:
a full memory round trip per output element. Reading the whole group before storing any of it leaves
one wait for sixteen loads in flight. A lane reads exactly the sixteen elements it writes and no
other lane touches them, so the values, the additions and their order are unchanged. This half is
not specific to the dense image and reaches the pair-code arm above 32 rows.

## What it measures

One 32-stream step, rocprofv3 dispatch times, traced arm in every column, untouched kernels as
in-run controls:

| kernel | base `16c5d16` | one block ahead | two blocks ahead |
|---|---:|---:|---:|
| FFN down | 13.705 | 11.445 | **9.835** |
| FFN gate/up | 20.882 | 20.633 | 20.572 |
| `resident_state<4>` (control) | 46.427 | 46.358 | 46.200 |
| `project<2,false,true>` (control) | 16.567 | 16.614 | 16.752 |
| `project<2,true,false>` (control) | 12.439 | 12.559 | 12.585 |
| device busy | 122.121 | 119.729 | 118.108 |

Controls move by at most 1.2% across the three builds. The down stage falls 28.2%, to 1.44x its
block-loop issue model from 2.0x.

Prefill, where the same dense arm runs at 32 rows and the pair arm runs at 128, as the `ffn` phase
over the untouched phases of the same run:

| rows | base | two blocks ahead | |
|---:|---:|---:|---|
| 32 | 0.9401 | 0.8052 | -14.4%, both halves |
| 128 | 0.8807 | 0.8728 | -0.9%, the residual half alone |

Bit-identical in every shape: the residual FNV-1a over all rows is 1873717996769712938 at 32 rows
and 7446865760224376151 at 128, which are the values canonical main produces, and the 32-stream
step's residual hash is unchanged at 8087160186626784132.

## On current main

`cc5d259` landed a per-stage image rule and its own cuts to the dense operand build, including
hoisting a block's six weight requests above its peel. That captures part of what the cursor's first
block of lookahead was buying, so the change was re-measured on top of it. Same panel, one build per
arm, `ffn` over the untouched phases of the same run:

| rows | `cc5d259` | with both halves | |
|---:|---:|---:|---|
| 32 | 1.0739 | 1.0303 | -4.1% |
| 128 | 0.9044 | 0.8801 | -2.7% |

The 128-row row is the residual half alone and it is larger there than it was against `16c5d16`,
because the pair-code down stage it applies to is a bigger share of that pass now. Both residual
hashes are unchanged. The cursor costs 212 VGPR against the arm's 175 after the rebase, so it keeps
seven waves per SIMD32 where it used to cost one of them.

## Installed

Canonical `ca9d83a`, build SHA-256
`94170f99f5dcc4cd26d2cfdcbba06458a65a105c80eda45412e7eb2ca9de525b`. The installed executable
reproduced both documented residual hashes with and without the vocabulary head - 1873717996769712938
at 32 rows, 7446865760224376151 at 128 - and ran the aggregate generation shape at **274.3 and 275.4
tok/s across 32 streams**, 116.208 and 116.669 ms per step, against 268.7 documented before this
week's changes. The resident server came back on that binary and served. That number carries every change on main today, not this one alone; the controlled
attribution for this change is the ratio table above.

## Reproduce

```sh
make -j12 tools/batch_profile
# the dispatch table of one generation step
BONSAI_PROFILE_TRACE=1 BONSAI_PROFILE_OUTPUT=DIR tools/run-batch-compare --profile-tool \
  --modes 19 --decode-streams 32 --decode-prompt 64 --rounds 1 --warmup-ms 200 --rows 32 \
  --out DIR/step.json
tools/kernel_trace.py DIR --segments            # the step is the 733/734-dispatch segment
tools/kernel_trace.py DIR --segment N --kernels
# the cheap iteration loop: 32-row prefill runs the same FFN instantiations in 5.5 s
tools/run-batch-compare --profile-tool --modes 19 --rows 32,128 --heads 0 --traces 0,1 --rounds 2
```

Raw data, all builds and both shapes, under
`../../data/bonsai2/batch-comparison/decode-latency/`.

## Open after this

The down stage is now 154 us per layer against a 107 us block-loop issue model, and it is out of
register headroom at 254 VGPRs, so a third block in flight would spill. The remaining gap has to
come off the instruction count rather than the latency: 322 of its 658 slots are five-trit decode,
which is `2e6fc74a`'s half of this arm, and a mixed arm that keeps the dense image on gate/up and
puts the pair image with shared ownership on down would replace this stage's map entirely.

Gate/up did not want more waves and barely wanted lookahead: it is at 1.27x its issue model with
1088 waves, and 240-254 VGPRs cost it one of seven wave slots per SIMD32. Everything left there is
instruction count.

`project<2,false,true>` and `project<2,true,false>`, 26.0 ms of the step and 54 ms of a 128-row
pass, have never been read at this level. They do **not** have the residual serialization: they
accumulate through one pointer, `out[o] = out[o] + y[t][r]`, so the compiler knows the two addresses
are the same one and batches the group by itself. Exactly two pointers naming one buffer is what
defeats it.

The control kernels `k_proj_int` and `k_proj_scaled` still carry the serialized epilogue: 64 chains
in `k_proj_int<4, false, false>`, 121 in `k_proj_scaled<8, false>`. That is deliberate. Modes 1, 2
and 3 are the references every numerical comparison is taken against, and mode 5 is the in-process
speed control half this lane is using this week; a control that changes speed between builds is
worth less than the few milliseconds the fix would return. Fix them only together with a re-taken
reference panel.

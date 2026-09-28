# A store predicate was costing the recurrence its instruction-level parallelism

The gated-delta recurrence walks `R` state rows per token, and the rows are independent: each one
decays, dots the state against `k`, reduces across the wave, builds a rank-1 update and dots the
result against `q`. Written one row at a time, every row ended with

```c
if (out_o && lane == 0) out_o[h * 128 + pu * RPU + wave + 8 * r] = op * 0.08838834764831845f;
```

A divergent store is a branch, a branch terminates a basic block, and a scheduler does not move
instructions across one. So the emitted token body was `R` separate blocks, and **the machine spent
each of them covering a five-deep cross-lane butterfly while three fully independent chains sat in
other blocks where it could not reach them.**

Batching the rows fixes it with no arithmetic change: every row's k-dot, then every row's butterfly,
then every row's update and q-dot, then every row's butterfly, then the stores under one exec mask.

## What it measures

Mode 19, 128-row prefill pass, logits on the tail row, six traced samples per arm, arms interleaved
inside one GPU-lock acquisition on one warmed device.

| phase | serialised rows | batched rows | change |
|---|---:|---:|---:|
| **`gdn-resident-core`** | **17.467 ms** | **14.697 ms** | **-15.86%** |
| `ffn` | 78.161 | 77.742 | -0.54% |
| `sequence-input-projection` | 43.731 | 43.686 | -0.10% |
| `sequence-output-projection` | 16.709 | 16.620 | -0.53% |
| `sequence-core` | 4.331 | 4.294 | -0.85% |
| `sequence-input-prep` | 3.949 | 3.909 | -1.02% |
| `head-projection` | 1.682 | 1.661 | -1.24% |
| device span | 167.664 | 164.174 | -2.08% |

Normalised by the eight phases the change cannot reach, `gdn-resident-core` is **-15.10%** and the
span is **-1.58%**. The per-sample ranges do not overlap: 16.665 to 17.653 ms serialised against
14.599 to 14.991 batched.

Full model, 1536-token document in 128-row passes with the tail-row head, three rounds per process
and four processes alternating arms inside one lock: **prompt processing 796.0 to 809.0 tok/s,
+1.63%** on each process's final round, which is what a 2.77 ms cut out of a 167 ms pass predicts to
within a twentieth of a point. Pooling both warmed rounds of every process gives 796.6 against
808.2, +1.45%, and carries one 758.5 sample from the first process of a cold lock.

A 384-token document cannot measure this. The same panel at that length, while several engineers
were compiling, read 639.6 to 831.1 tok/s with the ordering uncorrelated with the arm. Prompt panels
on this box need a document long enough that the device is warm for most of it.

**Bit-identical.** Residual FNV-64 `7446865760224376151` in all twelve prefill samples, which is the
value this repository already publishes for that shape, and `8087160186626784132` in all twelve
decode samples. Nothing is reassociated: each sum keeps its own products in its own order, the
butterflies are the same `warp_sum` over the same values with full exec, and the stores are the same
values to the same addresses.

## Installed acceptance

Canonical `dc331a5`, installed `bonsai-halo`
`ff99af0b69d7953d586f5ed9244ee6767d24309eed6f309bfbfa68691e5be1bb`, `tools/batch_profile`
`8ede7dc576e3413f`. Both published residual hashes reproduced exactly: `7446865760224376151` in
both 128-row samples and `8087160186626784132` in both 32-stream samples. Single-stream `--bench`
reads 32.99 tok/s.

| | 128-row pass | 32-stream step |
|---|---:|---:|
| `gdn-resident-core` | 15.814 / 15.532 ms | 46.138 / 45.081 ms |
| device span | 177.613 / 174.376 | 103.625 / 101.634 |
| `gdn-resident-core` / eight untouched phases | **0.09844, 0.09847** | — |

That ratio is the acceptance. The panel above reads **0.11661 serialised and 0.09900 batched**, so
the installed build sits **-15.57% against the serialised arm and -0.55% against the batched one**.

Every absolute phase is 6 to 8% above the panel — `ffn` 82.5-84.0 against 77.7,
`sequence-input-projection` 46.1-46.9 against 43.7 — because the acceptance ran on a box several
engineers were compiling on, immediately after a serving window, and because main gained the IU4
nibble matvec operand and the projections' operand work between the panel and the install. Neither
is this change, which is why the attributable number is the ratio and not the span.

## The null is the other half of the result

| workload | serialised | batched | change |
|---|---:|---:|---:|
| 32-stream decode step, fp32 state | 101.28 ms | 101.55 ms | +0.27% |
| `gdn-resident-core` in that step | 44.848 ms | 44.874 ms | +0.06% |

[The decode map](decode-map.md) bounds the one-token-per-sequence shape at the machine's memory
roof: 302 MB of state per sequence per step, moving at 224 GB/s of the 242 GB/s `bench/bw` measures.
This is that bound being paid. **Removing 24.5% of the phase's issue slots bought exactly nothing
there**, on the same kernel, in the same build, with the same change.

So the recurrence is not one optimisation target. Prefill and the eight-row verify pass are issue-
shaped; the batched decode step is not. An arm that moves both is not moving them for the same
reason, and an arm measured only on decode cannot see this change at all.

## The two other shapes this engine serves

**The eight-row deployed route**, which is the drafted verify pass and 79-83% of a served drafted
generation step. Mode 4, eight rows, nine traced and nine untraced samples per arm across six
processes alternating arms in one lock:

| | serialised rows | batched rows | change |
|---|---:|---:|---:|
| pass wall, untraced | 57.873 ms | 57.341 ms | **-0.92%** |
| `sequence` (the persistent kernel) | 22.066 ms | 21.656 ms | **-1.85%** |
| `ffn`, `head` *(controls)* | 32.664 / 2.172 | 32.547 / 2.172 | -0.36% / -0.02% |

Normalised by the two controls the persistent kernel is -1.49%. That phase carries the matvecs, the
preps and the recurrence together, and the recurrence is about 8% of it, so a 15% cut on the
recurrence is most of what appears. Residual FNV-64 `9136695298984453119` in all eighteen samples.

**Single-stream generation is a null and the arithmetic says it must be.** Four `--bench -n 40`
processes alternating arms in one lock: 33.10 and 33.30 tok/s serialised, 33.60 and 33.37 batched.
The difference is inside this box's 33.05 to 33.80 spread and should not be read as a gain. A
single-token step spends 1257 µs of 29694 in `gdn`, and it spends it at 240 GB/s of a 242 GB/s roof
— the phase is already moving its state as fast as the machine moves bytes, and issue slots are not
what it is waiting for.

## The census, which is where the change came from

`hipcc -S` on `resident_state<4, GATE_HOIST, false>`, counted with `tools/isa_loop_count.py`:

| | serialised rows | batched rows |
|---|---|---|
| token body | **4 blocks, 54 instructions each** | **1 block, 166 instructions** |
| cross-lane (`v_add_f32_dpp`, `v_permlanex16_b32`) | 40 | 40 |
| `s_delay_alu` | 56 | 40 |
| `v_dual_*` VOPD pairs | 1 | 14 |
| total issue slots per token | 220 | **166**, -24.5% |

The cross-lane count is unchanged, which is the point: the butterflies were never the thing to
remove, only the thing to overlap. What the merge buys is 16 fewer scheduling slots and fourteen
dual-issue pairs where there was one, because four independent chains in one block give the VOPD
packer something to pair and give every DPP result three other chains to hide behind.

`k_forward_rows` inlines the same helper through `ph_gdn`, and the same four blocks collapse into
one there: 185 instructions at `TT = 8` and 167 at `TT = 1`, with `v_dual_*` 12 to 22 and 11 to 22.

**Registers.** `resident_state` 59-61 to 64 VGPRs with no occupancy change, because that kernel is
capped at eight 256-thread blocks per WGP long before registers bind. The `k_forward_rows`
instantiations that set the shared cooperative grid went **down**, 147 to 145 and 140 to 140, so the
grid `rows_grid_size` hands the deployed path is unchanged.

## What the reassociation was for, and why it is not here

The obvious algebraic move is to distribute the q-dot over the rank-1 update:

```
op_r = <g M_r + delta_r k, q>  =  <g M_r, q> + delta_r (k . q)
```

It takes the q-dot off both the update and `delta`, so a row's two reductions become independent and
`k . q` is uniform per (token, head group), computable once where `resident_conv` already holds
normalised `q` and `k`. It was the second arm of this claim.

**Batching the rows subsumes most of it.** The identity does not remove a single instruction — the
VALU count is 76 per token either way — it only shortens the dependency chain, from four independent
chains to eight. Having already given the scheduler four, the eighth is worth much less than the
first, and the identity is real-equal rather than fp32-equal, so it would need its own horizon panel
and would land as an opt-in alternative to an exact default. It is not built, and a peer is applying
the same distributive law across pending terms of a deferred commit, where the payoff is `2` vector
operations per term instead of `128` and the argument is much stronger.

## What still stands, and what it is not

After the cut the phase is **3.4x its own issue model** at 128 rows: 192 units x 8 waves x 128 tokens
x about 180 slots is 91 µs of issue per layer against 306 µs measured.

**One and two thirds of that 3.4x is makespan, and it exists only at one sequence.** A unit is
`(sequence, head, one of SPLIT row groups)`, so a 128-row prefill pass has `1 x 48 x 4 = 192` units
and the machine holds 160 blocks of 256 threads at once. 192 units on 160 slots is two rounds for
1.2 rounds of work: 60% utilisation, worth 1.67x by itself. A 32-stream decode step has 6144 units
and no quantisation at all, which is one more reason the two shapes do not share an answer.

Raising `SPLIT` is the obvious fix and it has already been measured as a loss —
[resident GDN](resident-gdn.md) has 235.6 ms at split 4 against 242.8 and 253.4 at 8 and 16 — because
every unit re-reads the whole token's `k` and `q` whatever fraction of the state rows it owns, so
splitting multiplies the operand traffic by the split. The decomposition that would pay is one that
adds units without adding operand reads, and it lives in `resident_state`, whose holder also has the
occupancy probe.

## Reproducing

```sh
tools/run-batch-compare --pin-clock --exec /bin/sh PANEL.sh      # both binaries, one lock
# PANEL.sh alternates two builds of tools/batch_profile:
#   batch_profile --modes 19 --rows 128 --heads 1 --traces 1 --rounds 2 --out OUT.json
#   batch_profile --modes 19 --decode-streams 32 --heads 1 --traces 1 --rounds 3 --out OUT.json
```

There is no in-process case axis for this arm. The change is a schedule with no parameter, and the
kernel it lives under is claimed by another engineer, so adding a template arm to `resident_state`
was not this turn's to make. The pairing is two builds of the same source under one lock acquisition,
normalised by the eight untouched phases, which for a 15% move on a phase whose arms do not overlap
is enough; it would not be enough for a 2% move.

Raw samples, both binaries' SHA-256 and the panel scripts are in
[`batch-comparison/gdn-token-reduce/`](../../../data/bonsai2/batch-comparison/gdn-token-reduce/README.md).

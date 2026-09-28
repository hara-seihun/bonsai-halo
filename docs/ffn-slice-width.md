# The deployed matvec was throwing away half of every matrix instruction

`mvw_rows` is the body every matrix phase of a short pass runs. It issues
`wmma_i32_16x16x16_iu8`, whose B operand carries **sixteen** activation columns, and it picks the
column's activation row with

```c
const int8_t * xrow = xq + (size_t) (col & 7) * NB * 128 + wb * 128;
```

Eight rows, duplicated into sixteen columns. The epilogue stores only `col < nrows`, so the second
copy is computed and discarded. Two consequences, and the second is the useful one:

- The matrix cost of a pass **does not depend on its row count** anywhere in 1..8 rows. The same
  tiles, the same blocks, the same sixteen WMMAs per 128-K block.
- Any caller that can present sixteen rows gets the second eight **for the price of the epilogue**.

## What that costs in the shape the machine actually runs

One deployed eight-row pass, measured with `HALO_PROFILE=1 HALO_PROFILE_ROWS=8` on a 1289-token
prompt ingested in eight-row passes (155.2 tok/s), 52.753 ms of internal phase time:

| phase | ms | share | scales with rows? |
|---|---:|---:|---|
| `mv_gate_up` | 17.823 | 33.8% | no |
| `mv_down` | 10.463 | 19.8% | no |
| `mv_qkv_z` | 6.586 | 12.5% | no |
| `gdn` | 4.226 | 8.0% | yes |
| `mv_ssm_out` | 3.785 | 7.2% | no |
| `attn` | 3.564 | 6.8% | yes |
| `mv_qkv` | 1.947 | 3.7% | no |
| `prep_norm` | 1.517 | 2.9% | yes |
| `mv_o` | 1.269 | 2.4% | no |
| `gdn_pre`, `prep_silu`, `prep_gdn`, `prep_attn`, `attn_pre`, `embed` | 1.573 | 3.0% | yes |

**79.4% of the pass is row-invariant matrix work and 20.6% scales with rows, about 1.36 ms per
row.** A sixteen-row pass of this shape would cost about 63.6 ms instead of 52.75 and serve twice
the rows.

## Where sixteen rows were already available

[Wide prompt ingestion](wide-prefill.md) runs the wide schedule over the deployed FFN (mode 20,
`--prefill-ffn wide-deployed`). Its sequence projections, recurrent state and head already take all
128 rows of a pass at once, but `forward_batch` handed the FFN to `launch_ffn_slice` **eight rows at
a time**, so a 128-row pass visited the entire FFN weight stream sixteen times and spent 79% of
itself there.

`k_ffn_slice` reads no row metadata, no block cache and no recurrent state — it is norm, gate/up,
SiLU-multiply, down, over `P.x` — so its width was never bound by `RMAX`, the pass row limit that
exists for replay, rollback and attention. `FMAX = 16` gives it sixteen rows; `RMAX` stays 8, which
keeps `blk_cache` (63 MB per slot pair) and every rollback contract exactly where they were.

## Measured

`tools/run-batch-compare --profile-tool --modes 20 --rows 128 --heads 1 --traces 1 --rounds 1`,
one 128-row pass with logits on every row, `HALO_FFN_SLICE=8` selecting the eight-row control.
Source `067a567`, canonical `27cafb1`.

| phase | 8-row slices | 16-row slices | change |
|---|---:|---:|---:|
| `ffn` | 412.577 ms (1024 launches) | **218.092 ms** (512) | **-47.1%** |
| `sequence-input-projection` | 40.069 | 40.358 | +0.7% |
| `gdn-resident-core` | 16.590 | 16.569 | -0.1% |
| `sequence-output-projection` | 16.373 | 16.554 | +1.1% |
| `sequence-core` | 16.073 | 16.195 | +0.8% |
| `head-projection` | 11.235 | 11.194 | -0.4% |
| `sequence-input-prep` | 4.234 | 4.242 | +0.2% |
| **device span** | **522.228** | **326.841** | **-37.4%** |

The six phases below the first line are the control: this change cannot reach them, and they move
by at most 1.1% across the two processes.

### Full model, and it is the served path

`--prefill-sweep` walks prompt routes inside one process with the DFlash2 drafter loaded, which is
how `bonsai-halo.service` runs. 1289-token prompt, greedy, eight generated tokens:

| route | prompt | tok/s | greedy token digest |
|---|---:|---:|---|
| 0, deployed eight-row passes | 8.487 s | 151.9 | 7796914616836249716 |
| 20, wide-deployed, eight-row FFN slices | 5.169 s | 249.4 | |
| **20, wide-deployed, sixteen-row FFN slices** | **3.121 s** | **413.0** | 7796914616836249716 |

**1.656x on wide prompt ingestion, 2.72x on what the server did this morning.** Drafted generation
is untouched by construction and measures so: 31.99 against 32.19 tok/s, 28 drafted and 4 accepted
in both arms, 52.8 against 52.5 ms of verify per step.

### Why it is bit-identical

Each output row's dot product reads the same weights in the same K order, accumulates into the same
per-block int32, and applies the same FP32 scale chain in the same sequence; the cross-wave
reduction in `ph_matvec_w` sums the same waves in the same order. The only thing that changes is
which lane holds a column, and lanes do not interact within a WMMA's column dimension. The evidence
matches the argument: residual FNV-64 `16921095978579329146` in both arms of the traced panel, and
the drafted greedy token stream identical to the deployed route's in the same process.

## What this does not do

**Generation does not move.** A drafted verify pass is eight rows through the persistent kernel
(`k_forward_rows`), not through `k_ffn_slice`, and a single generation step is one row. Those
shapes have no sixteenth row to offer, so they keep paying the duplicate column. The
[serving decode map](serving-decode.md) prices what that means.

**Sixteen is the instruction's width, not a tuning knob.** Above sixteen rows a slice would need a
second column group, which re-reads nothing but does issue a second set of WMMAs per weight block;
that is the token-tile structure `k_proj_opt` already has and `mvw_rows` does not. The peeled
weights and their scale table are live in registers at that point, so a second group would amortise
the ~112 slots of five-trit expansion per block as well as the weight load. That is the next
question in this kernel, and `y1[8]`/`y2[8]` per group is what it costs.

**And the ladder stops there.** [The third and fourth groups](ffn-slice-rows.md) are built behind
`-DHALO_SLICE_WIDE=1`, cost no wave slot and no grid, halve the weight stream of a 128-row pass -
and measure **+0.18% per row**, with three groups at **+29%**. What ends the ladder is not
registers and not bytes: each added group's operand set is requested inside the loop that consumes
it, and `tools/vmcnt_cover.py` puts the block's mean request cover at 167.7 slots with two groups
and 27.9 with four.

**That group is built and landed.** [The second column group](deployed-matvec-groups.md) takes
`k_ffn_slice` to 32 rows: 25% fewer non-matrix instructions per row, half the weight stream, the
`ffn` phase of a 128-row wide-deployed pass 216.3 to 173.2 ms, that route 407.2 to 458.5 prompt
tok/s, and 381,419,520 compared logit values with zero differing bits. It cost one more thing than
`y1[8]`/`y2[8]` predicted: the group loop has to be fenced against the scheduler, or every group's
activation fragments are hoisted above the first matrix instruction and the shape spills.

## Controls and reproduction

```sh
make -j8 bonsai-halo tools/batch_profile
# kernel level, one 128-row mode 20 pass, with the eight-row control
tools/run-batch-compare --profile-tool --modes 20 --rows 128 --heads 1 --traces 1 --rounds 1 \
  --warmup-ms 400 --context 128 --out .../m20-s16.json
HALO_FFN_SLICE=8 tools/run-batch-compare --profile-tool --modes 20 --rows 128 --heads 1 \
  --traces 1 --rounds 1 --warmup-ms 400 --context 128 --out .../m20-s8.json
tools/phase_totals.py .../m20-s8.json .../m20-s16.json
# product level, drafter loaded, both prompt routes in one process
tools/run-batch-compare --engine --bench --dflash ~/data/bonsai2/drafters/dflash2.safetensors \
  --prefill-sweep off,wide-deployed -p "$(sed -n '1,45p' PLAN.md)" -n 8
```

`HALO_FFN_SLICE=N` (1..16) forces the slice width; the default is `FMAX`. Raw samples, the
eight-row phase map and the drafted baseline are in
[`batch-comparison/ffn-slice-width/`](../../../data/bonsai2/batch-comparison/ffn-slice-width/).

## Installed

The canonical build `da4d16da501f2dd0a0a9921c71ea8633f1c8cc9d5e70f9642a0d974b7463031e` reproduced the
result on its own panel: 1289-token prompt with the DFlash2 drafter loaded, 152.0 tok/s on the
deployed eight-row route and **408.3 tok/s** on wide-deployed, both arms in one process with the
identical greedy token digest `7796914616836249716`, 28 drafted and 4 accepted in each, and
generation at 32.07 against 32.04 tok/s. The resident service was masked by the measurement wrapper
and left to its next holder, which is the correct state while the GPU lock is held.

The installed binary was then rebuilt at the canonical tip that carries this change plus the
sequence-projection work that landed beside it
(`d8864ad50610b2eff9fe8fc9250f21cedd5e30aa9fc3e733cdc4fbe859ef7815`) and re-measured on a busier
box: 136.7 tok/s deployed against **404.3 tok/s** wide-deployed, same digest in both arms,
generation 31.64 against 31.90 tok/s.

# The sixth block is reachable, it costs 18.6%, and registers are not what this kernel is short of

`bench/mvsched` measured the deployed eight-row matvec body across co-residencies and read **150.4
GB/s at four workgroups per WGP, 135.7 at five and 181.5 at six**, which made the sixth block the
largest priced number the lane had left: `k_forward_rows<8>` allocates 141-149 VGPRs, the ladder in
`kernels/halo_rows.hip` is four blocks at 136 and above, five from 97 to 135 and six at 96 or fewer,
and the one time anybody reached 96 the build spilled 31 registers and lost 4.9% of prompt
ingestion. So the question was whether a kernel that genuinely *needs* 96 registers gets the +21%.

**It is built, it spills seven registers instead of thirty-three, and the sixth block still loses
18.6% of the verify pass.** The whole of that loss is the grid, not the spills, and the grid the
engine already runs is the maximum of a unimodal curve. Nothing ships.

## Where the peak is

`tools/kernel_resources.py`, compile-only, `-DHALO_ROWS_PROBE=8`, source `f7b680d`:

| body | natural VGPR | waves | at `waves_per_eu(13)` |
|---|---:|---:|---:|
| deployed | **149** | 9 | 96 VGPR, **33 spilled** |
| attention deleted (`-DHALO_ROWS_DROP=4`) | 107 | 12 | 96 VGPR, **0 spilled** |
| GDN deleted (`-DHALO_ROWS_DROP=8`) | 142 | 10 | |
| score unit grouped at four rows (`-DHALO_PK_ATTN_RG=4`) | 128 | 10 | 96 VGPR, 15 spilled |
| the same, folded-head unit dropped from `TT = 8` | **117** | **12** | 96 VGPR, **7 spilled** |
| lane-per-key score body inlined here (`-DHALO_PK_ATTN_KM=1`) | **214** | 7 | |

**Every spilled register at the sixteen-wave budget is the attention phase's**, and the phase census
in `kernels/pk_pressure.hip` says the same thing from the other side: `k_probe_attn<8>` is 107 VGPR
on its own where the widest matvec is 90, prep 49 and the argmax 16.

Three things come out of that table.

- **`qr[TT][DPL]` is 64 registers at eight rows and a score unit does not have to carry eight rows.**
  A row's scores, its softmax and its value accumulation read only its own query and write only its
  own `(row, head, chunk)` partial, so splitting a unit's rows changes which workgroup computes which
  row and nothing that any row computes. `HALO_PK_ATTN_RG` groups them: four rows a unit is 21
  registers and 18 of the 33 spills, and a group takes its key count from its own last row, which is
  exactly the keys its rows would have masked. **Bit-identical, and measured so** — the greedy digest
  `18303137358360366343` is equal across the two executables on a real drafted run.
- **`RG = 2` is the same 128 registers as `RG = 4`**, so the residual is not the per-row arrays.
  Eleven of it is the folded-head unit, `qr[GQA][1][DPL]` at 48 VGPRs. **That unit is not dead code
  at this row capacity** and the gate that dropped it is not in the tree: `cc454371` established that
  a mode-4 slice of eight one-row sequences reaches it, which is what `ServeBatch` runs at 5 to 31
  concurrent requests. Its 11 registers are rent the eight-row kernel pays for the serving path.
- **The lane-per-key body is register-light standalone and register-heavy inlined.** It reads 94
  VGPRs as `k_attn_wide<8>` and 214 here, both with and without the `__restrict__` query. That closes
  the handoff's standing question — "if the scalar-query body lands inside `ph_attn` the whole ladder
  gets re-fitted" — with a number instead of a guess. It gets re-fitted the wrong way.

## What the sixth block is worth

One executable, one process, one pinned clock, the 96-register body throughout, `--grid-rows` as the
only axis. The deployed drafted step, `--bench --dflash` with the Q4 drafter the service runs:

| cooperative grid | blocks per WGP | verify ms/step | against the deployed grid |
|---:|---:|---:|---:|
| 80 | 4 | 54.665 | +25.8% |
| 90 | | 46.288 | +6.5% |
| **100 (deployed)** | **5** | **43.458** | |
| 110 | | 45.481 | +4.7% |
| 120 | 6 | 51.987 | **+19.6%** |

Second process, same body, `--grid-rows 100,-1`, where `-1` is the grid the device grants: 43.824
against 51.987 ms, **+18.6%**. The deployed grid is the maximum of a unimodal curve and both
neighbours lose, at ±10 workgroups as well as at the block boundaries.

**What this ladder does not cover is context.** Every cell above is a 26-token prompt, where the
attention phase deals 24 units a slice and the matvecs deal 1088 and 7760; a served request runs
1200-1400 tokens, where attention's unit count grows with position and the matvec's does not. The
curve's *shape* is a property of how the grid competes for a unit list, so a longer context can move
the peak without changing the conclusion that registers are not the lever. Anyone who wants the
deployed grid re-fitted for the served shape should walk the same axis at 1400 tokens; it is the same
panel with a longer `-p`.

**So `bench/mvsched`'s co-residency cell does not transfer.** That probe is the matvec body alone
with LDS ballast; this kernel is six phases, a cooperative grid and a unit list, and it does not
behave like its own inner loop. A body measured in isolation prices its instructions, not its grid.

## What a register is worth once the grid is fixed

Same process, same grid, the only difference the register budget the allocator was asked for:

| body | spilled | verify ms/step |
|---|---:|---:|
| 117 VGPR (`--pk-occ 0`) | 0 | 42.330 |
| 96 VGPR (`--pk-occ 1`) | 7 | 42.539 |
| 96 VGPR, deployed score unit, other executable | **33** | 41.990 |

**Nothing, to 0.5%.** Thirty-three spilled registers cost no more than seven and seven cost no more
than none, as long as the grid does not move. `docs/rows-grid-occupancy.md` reached the same
conclusion at 120 registers and seven spills — "the whole gain is grid" — and this extends it to the
tightest budget the ladder has.

**The rule that follows is worth more than the negative, because three engineers are inside this
kernel spending registers right now:** on `k_forward_rows`, spend registers freely. The only thing a
register buys is a block per WGP, the block above the deployed one loses, and the constraint plus
spills absorb the growth for free. A phase that wants sixteen more VGPRs for a value-block cursor or
a second accumulator chain should take them and measure the phase, not the allocator.

## What did not move, and what is in the tree

Canonical runs exactly the code it ran before. `HALO_PK_ATTN_RG` defaults to zero and
`HALO_PK_ATTN_KM` to `ATTN_KM_ROW`, so the deployed build reproduces the pre-change register report
exactly - 149 VGPR at nine waves, 96 with 33 spills at `waves_per_eu(13)` - and the drafted digest is
equal across both executables. On the tree this landed on, which carries the drafter's windowed unit
list from `2f260d20`, the same two cells read 34 spills with and without this change and 15 with
`HALO_PK_ATTN_RG=4`: the table above was taken at `f7b680d` and the spill count moves with the unit
list, which is the reason the arms are here rather than the numbers. They stay as compile arms because the register table above has to be
re-fitted every time a phase body inside this kernel moves, which happened three times on September
21, and because `HALO_PK_ATTN_RG` is the only lever anyone has on the number that sets the mark.

## The next useful question

Not the register budget of this kernel, and not its grid.

The curve has a maximum at five blocks per WGP and **falls off steeply on both sides**, which is not
what an occupancy curve normally looks like — more resident waves are supposed to help or saturate,
not cost 19%. Five blocks per WGP is the one rung that does not split evenly across the two CUs of a
WGP, and 7f4d6072 measured the same non-monotonicity in the opposite direction on the isolated body
(150.4 / 135.7 / 181.5 at four / five / six). Two instruments now disagree about the sign of the same
step, which means neither of them is measuring co-residency alone. What would settle it is the unit
distribution: this kernel hands units out through `next_unit`, so a bigger grid is more workgroups
competing for the same unit list, and `docs/mv-tile-fold.md` has just shown that this engine charges
by unit count more than by anything else. **Price the grid against the unit count, not against
occupancy** — the same panel with `--grid-rows` crossed against a phase whose unit count is known,
with the traced phase table rather than the whole verify pass, would say whether the 19% is
scheduling contention or the memory system.

## Raw

[`batch-comparison/pk-register-peak/`](../../../data/bonsai2/batch-comparison/pk-register-peak/README.md)
holds the three panels, the two executables' build lines and the census commands.

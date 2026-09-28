# The attention unit sets the whole model's cooperative grid

`launch_forward_rows` is one cooperative kernel per pass, and its workgroup count is
`rows_grid_size()`: the minimum, over the `TT = 1, 2, 4, 8` instantiations, of what the device will
hold co-resident. `k_forward_rows<8>` is always that minimum, so the eight-row instantiation's
register file decides how many waves every phase of every wide pass gets. The matvecs, the FFN, the
GDN recurrence and the head all live on that one number.

That number was 100 workgroups. This document is where it came from, what moved it to 120, and the
two things the next engineer in this kernel should not have to rediscover.

> **Two of the ladders below were read off the occupancy API and the API is wrong.** The hardware's
> own co-residency is measured in [what the device actually holds](#what-the-device-actually-holds-and-why-the-ladder-above-is-an-estimators)
> at the end of this document, and it retires three claims made above: 96 VGPRs is eight blocks per
> WGP and not six, a WGP's LDS budget is 128 kB and has never capped this kernel, and the step that
> matters sits at 120/121 registers rather than at 135/136. It also settles what the step is worth,
> which is the part that changes what anyone should build: **past 100 workgroups the eight-row pass
> does not get faster, and the register budget that buys the next block costs 13 to 15% of it.**

## The compiler's ladder is not the runtime's, and the runtime is the one that launches

Two ladders matter here and they disagree, which is why this document exists and why the first
version of this change measured a clean negative.

The compiler's, from `-Rpass-analysis=kernel-resource-usage`, fits an allocation granule of 24
registers against a 1536-register file per SIMD32:

The compiler will tell you the register file and the waves it leaves without a GPU, and
`tools/kernel_resources.py` reads it out of `-Rpass-analysis=kernel-resource-usage`. Fitting the
reported occupancy against the reported VGPR count over every `k_forward_rows` instantiation gives
an allocation granule of 24 registers against a 1536-register file per SIMD32:

| VGPRs | allocated | waves / SIMD32 |
|---:|---:|---:|
| ≤ 96 | 96 | 16 |
| 97 to 120 | 120 | 12 |
| 121 to 144 | 144 | 10 |
| 145 to 168 | 168 | 9 |

`351dc2ab` confirmed that granule from outside this kernel: `k_ffn_slice` at 241 VGPRs takes grid 40
and at 215 takes grid 60, which granule 24 reproduces and a granule-16 model does not. So
`docs/mv-dot4-body.md`'s "seven registers buy twelve waves" is one granule short; on the compiler's
ladder the ask from 135 is fifteen.

**And then the device does something else.** `tools/rows_occ_probe` asks the runtime what it will
actually co-schedule, which is what `coop_grid` launches on, and the answer is in whole blocks per
WGP:

| VGPRs the loader reports | blocks per WGP | cooperative grid |
|---:|---:|---:|
| 95, 96 | 6 | 120 |
| 120 | **5** | **100** |
| 135 | 5 | 100 |
| 140 | **4** | **80** |

The steps are in a different place and the blocks are the quantum: a workgroup is eight waves over
a WGP's four SIMD32s, so a block costs two waves per SIMD and **an odd wave buys nothing**. Fitting
the four points gives about 1408 usable registers per SIMD32 at a granule of 8, not 1536 at 24. Three
of us had planned a fifteen-register cut against the compiler's table; it lands on 120, which is the
same five blocks as 135, and the grid does not move at all.

**LDS is the ceiling above all of it.** At 9508 bytes a block, six blocks is 57 kB of the 64 kB the
API reports per block, and seven would be 66.5 kB. So **120 workgroups is the most this kernel can
ever have** at its current LDS, whatever its registers: the 16-wave step buys 6 blocks, not 8, and a
grid of 160 needs the LDS under 8192 bytes before it needs a single register.

## The phase that holds the high-water mark is the attention unit

`halo_rows.hip` carries two compile-time probes for exactly this question, because the resource loop
is otherwise too slow to search:

- `-DHALO_ROWS_PROBE=TT` compiles **one** instantiation and stops before the host launchers, since
  it is those launchers taking kernel addresses that instantiate the other forty. 9 seconds instead
  of 48, with no GPU and no engine build.
- `-DHALO_ROWS_DROP=MASK` deletes one phase from the body. The mask is a compile-time constant, so
  the branch folds and the rest of the kernel is compiled exactly as it is served.

At `TT = 8`, fp32 state:

| body | VGPRs | waves |
|---|---:|---:|
| deployed | 135 | 10 |
| without `ph_attn` + `ph_attn_pre` (`DROP=4`) | **90** | **16** |
| without `ph_gdn` + `ph_gdn_pre` (`DROP=8`) | 127 | 10 |
| without the FFN (`DROP=2`) | 135 | 10 |
| without the head and argmax tail (`DROP=1`) | 135 | 10 |

**45 registers, and the source says which ones.** `attn_chunk_unit` opens with

```c++
float qr[TT][DPL];      // DPL = HDh / 32 = 8
```

and holds it live across the whole key loop: at eight rows that is **64 VGPRs of query cache**, one
copy per wave, and every wave of the workgroup holds the same eight query vectors. Nothing else in
the body is close. The FFN and the head, which are most of the pass's time, cost the allocator
nothing at all.

`kernels/pk_pressure.hip` says the same thing from the other side, compiling one phase per kernel at
the same `__launch_bounds__` and the same LDS block, so each number is what that phase costs on its
own. No GPU, one compile:

| phase | VGPRs |
|---|---:|
| `attn_chunk_unit`, `TT = 8` | **101** |
| the matvec, every `TT` from 1 to 8 | 65 to 72 |
| `ph_prep`, across its five flag shapes | 32 to 49 |
| `embed_row` | 23 |
| `ph_attn_pre` | 20 |
| `ph_argmax_slices` | 14 to 16 |

Attention at eight rows is the only phase above 72, and it walks 28 / 38 / 57 / 101 as `TT` goes
1 / 2 / 4 / 8, which is the `qr` array growing. The matvec is flat across `TT`, so the four
accumulator chains `docs/mv-dot4-body.md` added are genuinely free. `b0d22d99` and `38d1092e` each
built this table while three of us held the same claim; the file in the tree is `b0d22d99`'s.

The peak is *at* the attention phase rather than *only* the attention phase: the same unit inside
`k_forward_rows<8, 0, 2>` (the state/attention half that `launch_sequence_part` runs on its own)
needs 102 registers, not 135. The difference is what the full body keeps live across the layer loop,
loop-invariant address arithmetic the compiler hoists out of the phases that follow. That is
consistent with what the constrained build spills, below.

## Asking for the step: twelve waves, five spills, none of them in a loop

The change is one attribute on the kernel, carried by a trailing template parameter so both budgets
are in the executable at once:

```c++
template <int TT, int SM, int PART, int CM, bool PK, int OCC = 1>
__global__ void __launch_bounds__(NT) __attribute__((amdgpu_waves_per_eu(OCC)))
k_forward_rows(FwdParams P, int stage);
```

`FwdParams::pk_occ` picks the arm per launch, `HALO_PK_OCC` and `Engine::set_pk_occ` set it, and
`--pk-occ 0,1` walks both inside one process. Only the shapes whose allocation is actually below
twelve waves get a second instantiation, so with fp32 state one, two and four rows are **the same
function in both arms**, byte for byte. That is a free in-panel control: any difference a panel
reads at one row is this box, not this change.

What the allocator does with the request, at `TT = 8`, fp32 state. The `waves / SIMD32` row is the
compiler's claim and the `workgroups` row is the runtime's answer, and this table is where they
disagree: twelve waves per SIMD32 would be six blocks per WGP if the compiler's granule were real,
and the runtime gives five. Read the workgroup row, and read it off `tools/rows_occ_probe`:

| | deployed | `waves_per_eu(12)` |
|---|---:|---:|
| VGPRs | 135 | **120** |
| waves / SIMD32, compiler | 10 | 12 |
| **workgroups, runtime** | **100** | **100** |
| VGPR spills | 0 | 5 |
| scratch | 0 | 24 B/lane |
| SGPR spills | 430 | 395 |
| whole-kernel work slots | 18666 | 18713 (+0.25%) |
| largest two blocks, work slots | 496 / 262 | 492 / 258 |

**Where the five spills are is the whole question, and they are nowhere expensive.** All five
`scratch_store_b32` are in the kernel prologue (`%bb.56`, outside every loop). All seven reloads sit
in blocks the listing annotates `Depth=2`, which is per-unit code, and **not one is inside a
`Depth=3` block**, where the weight-block loop and the attention key loop live. The values are
64-bit address pairs, which is what the hoisting story above predicts: constrained to 120 registers,
the compiler stops carrying a few hoisted addresses through the attention phase and re-reads them
once per unit.

Asking for more than twelve is measured and rejected on the census alone: `waves_per_eu(14)` and
`(16)` both reach 96 registers, but at 28 to 30 VGPR spills and 116 to 124 B/lane of scratch, 79
scratch instructions against 12, 57 of them in per-unit blocks. Sixteen waves per SIMD32 would also
put the grid at 160, which is exactly the bit-identity bound below.

### The packed state pays twenty registers for its codec

With int8 or int16 state every row count is fifteen to twenty registers above its fp32 twin, and the
eight-row shape lands a whole block lower:

| `TT` | packed, allocator's | packed, wide budget | scratch |
|---:|---:|---:|---:|
| 1 | 135 | 120 | 10 B/lane |
| 2 | 139 | 120 | 13 |
| 4 | 142 | 120 | 16 |
| 8 | **140 (four blocks)** | **120 (five)** | 12 |

Only `TT = 8` needs the second instantiation: rows 1 to 4 take `ROWS_GRID_NARROW = 64` workgroups,
which four blocks already covers. `351dc2ab` and `38d1092e` both reported this register table
independently while three of us were claiming the same step. What none of us saw until the runtime
was asked is that the packed eight-row kernel is a block short, and that is where the whole result
turned out to be.

## Why the grid could not guarantee K-part order

A `KS > 1` matvec used to send its K parts to independent workgroups, which `atomicAdd`ed into
an existing residual float. Claiming units in increasing order did not guarantee completion in
that order. At a grid of 140 and `m.total_tiles = 160`, a slow workgroup can still be computing
part 0 of tile 0 after faster groups have claimed the other 159 part-0 units and begun part 1.
The prior proof in this section confused claiming a unit with finishing it. The matching digests
in the panel below establish identity for those particular runs, not for every run at that grid.
`docs/gdn-sliced-gates.md` records different greedy digests from repeated runs of one binary.

The target's ordered `ph_matvec` now assigns both parts of a tile to one workgroup and adds them
to the residual in part order. `ROWS_GRID_BUDGET` retains the old cap while the new drain's cost
is measured; it does not enforce arithmetic order. The DFlash2 drafter retains its independent
K-part schedule and its run-to-run variation.

## What it is worth: nothing on fp32, a fifth of the packed path

Every cell below is one process, one loaded model, one clock: `tools/run-batch-compare --engine
--bench`, the DFlash2 q4 drafter, the **deployed eight-row prompt route** (`--prefill-sweep off`), a
937-token prompt and 64 greedy tokens. `--pk-occ 0,1` interleaves the budgets, which needs the
control build (`-DHALO_ROWS_OCC_CONTROL=1`). Raw stderr:
[`batch-comparison/rows-grid-occupancy/panels.txt`](../../../data/bonsai2/batch-comparison/rows-grid-occupancy/panels.txt).

**fp32 state, where the ask buys no grid.** 135 and 120 registers are both five blocks per WGP, so
the only thing that changes is the five spills:

| budget | grid | prompt tok/s | drafted tok/s | verify ms/step |
|---|---:|---:|---:|---:|
| allocator's, 135 VGPR | 100 | 170.5, 174.6 | 45.04, 45.26 | 46.4, 46.1 |
| wide, 120 VGPR | 100 | 171.2, 170.3 | 43.60, 43.99 | **48.1, 47.7** |

**Five spilled registers cost 3.5% of the verify pass**, which is a much larger price than twelve
scratch instructions per unit ought to be, and the census says why it cannot be read off a slot
count: the constrained build is 18713 work slots against 18666, and what moved is the scheduling
around the loads, not the count. So the fp32 arm is a measured loss and does not ship. It builds
only under `-DHALO_ROWS_OCC_CONTROL=1`, which is also where those numbers came from.

**Packed state, where it buys a quarter more grid.** At 140 registers the packed kernel is four
blocks per WGP, one step *below* fp32, and nobody had looked: the int8 and int16 coordinates have
been running the deployed route on 80 workgroups against fp32's 100 since they landed.

| budget | grid | prompt tok/s | drafted tok/s | verify ms/step |
|---|---:|---:|---:|---:|
| allocator's, 140 VGPR | 80 | 140.3, 141.0 | 37.34, 37.33 | 57.5, 57.5 |
| wide, 120 VGPR | 100 | 168.9, 168.6 | 40.29, 44.37 | 52.0, 47.2 |
| **shipped default** | 100 | **172.8, 172.9** | **44.83, 44.77** | **46.7, 46.8** |

Against its own control, the shipped packed build is **+22.8% deployed prompt ingestion** (140.7 to
172.9 median), **+20.0% drafted generation** (37.34 to 44.80) and **verify 57.5 to 46.75 ms,
-18.7%**. The fp32 route in the same build is untouched: 170.7 and 174.2 tok/s against 170.5 and
174.6 before, on the same grid of 100 and literally the same kernel.

The point worth carrying past this change: **the packed state coordinate was not slower than fp32
for any reason in its own arithmetic.** It was paying twenty registers for its codec, those twenty
registers crossed a block boundary the compiler's table does not show, and the deployed route lost a
fifth of its workgroups. With the grid back, packed ingestion matches fp32's (172.9 against 174.2)
while moving a quarter of the state traffic.

### Why the grid is worth that much

The verify pass moves 5.9 GB in 49.8 ms, 118 GB/s against a 242 GB/s roof, so it is latency-bound
and waves are the lever. `docs/matvec-crossover.md` measured the same axis from below (62.8, 55.6,
52.6 and 49.8 ms at 60, 72, 84 and 100 workgroups) and never found a ceiling. A grid also shortens
phases in whole rounds, because a phase of `U` units over `G` workgroups costs `ceil(U / G)`:

| phase | units | rounds at 80 | rounds at 100 |
|---|---:|---:|---:|
| gate/up | 1088 | 14 | **11** |
| qkv + z | 512 | 7 | **6** |
| q/k/v | 448 | 6 | **5** |
| `ssm_out`, `o`, `down`, head | 320 | 4 | 4 |
| GDN state | 192 | 3 | 2 |

### Identity

Greedy digest `17799813650833466988` and 182 drafted / 44 accepted tokens in **every arm of every
panel**: both state coordinates, grids 80 and 100, both budgets. The digests and accepted counts
agree for this panel. They do not establish that every run at either grid has the same bits.

### Instruments this leaves

- `tools/rows_occ_probe` reports, per instantiation, the registers the loader sees, the LDS, the
  blocks per WGP the runtime will give it and the grid that implies. One compile, no model, no
  kernel launch, and it is the only thing in the tree that can tell a register change that moved the
  grid from one that did not.
- `HALO_ROWS_GRID=1` prints the same for the budget a live process selected.
- `--pk-occ 0|1|0,1` and `--grid-rows N,N` walk the budget and the workgroup count as case axes of
  the drafted bench, which re-prefills per arm, so one process gives prompt tok/s, drafted tok/s,
  per-step verify milliseconds, acceptance and the greedy digest for every cell.
- `-DHALO_ROWS_PROBE=TT` compiles one instantiation in nine seconds instead of the file's fifty,
  and `-DHALO_ROWS_DROP=MASK` deletes a phase from it.
- `kernels/pk_pressure.hip` prices every phase on its own.

## How to re-fit this when the attention unit changes

The value 12 belongs to the body in the tree the day it was measured. The rule does not, and the
rule is what to keep:

1. `tools/kernel_resources.py -D HALO_ROWS_PROBE=8 kernels/halo_rows.hip k_forward_rows` for the
   register count, and `-D HALO_ROWS_DROP=4` beside it for what the attention unit is costing.
2. Read the target off the step table above. Ask for the next step up with `-D HALO_ROWS_WPE=N`,
   then look at `vgpr_spill` and `scratch`, and at where the spills land in a `-S` listing. Spills
   in `Depth=3` blocks are inside the weight-block and key loops and are a different trade from
   spills in per-unit code.
3. Check register usage and measure the grid. The ordered drain now controls K-part order;
   changing the grid cannot reorder those parts. Check bit identity on the model, including
   repeated runs near narrow argmax margins, instead of inferring it from workgroup count.
4. Measure with `--grid-rows`, which walks the arms inside one process.

`dd242d6f`'s lane-per-key score loop is the live case: it deletes `qr` outright by moving the query
to the scalar file, and it reads 144 VGPRs in `k_attn_wide` today with a live set of 24, which is
the allocator spending slack rather than liveness. If that lands under 96 inside this kernel the
answer becomes `waves_per_eu(16)`, grid 160, beyond the old budget. If it lands between
96 and 120 the attribute becomes a no-op that records the requirement and the five spills go away.
If it lands at 144, the attribute is what keeps the grid at 120 instead of letting it fall to 90.

**It landed on the fourth branch of that prediction and the rule was wrong for fifteen commits.**
The refit is the rest of this document.

## The refit: the body moved, and the rule named the wrong thing

`7eeb556` gave each attention kernel one score body, which is a clear win on its own terms and is
also the third change this year to move `k_forward_rows`'s register count. Its commit message states
the consequence plainly — `k_forward_rows fp32 TT=8  206, grid 60 -> 141, grid 80` — and reports it
as a recovery, which it is: `04898ea` had put the fp32 shape at 206 registers and three blocks per
WGP without anyone noticing, and 141 is a third more grid than that.

**141 is also one register step past the five-block edge.** The ladder at the top of this document
says 97 to 135 is five blocks and 136 and above is four, so the deployed fp32 route spent 100
minutes running the served path on 80 workgroups with the number sitting in a commit message. Nobody
acted on it because `rows_occ_rule` reads as a fact about the state coordinate:

```c++
static int rows_occ_rule(bool pk) { return pk ? 1 : 0; }
```

It is not. It is a fact about a **register count**, and the register count belongs to whatever body
is inside the persistent kernel today. When it was fitted, the fp32 shape allocated 135 registers
and the ask bought no grid at all, so `0` was right and its comment said why. One attention change
later the same `0` costs a fifth of the served path.

### What the probe reads now

| instantiation | VGPRs | spills | blocks/WGP | grid |
|---|---:|---:|---:|---:|
| fp32 `TT = 8`, allocator | 141 | 0 | 4 | **80** |
| fp32 `TT = 8`, `waves_per_eu(12)` | 120 | 7 | 5 | **100** |
| fp32 `TT = 8`, `waves_per_eu(13)` or `(14)` | 96 | 31 | 6 | **120** |
| fp32 `TT = 1`, allocator | 98 | 0 | 6 | 120 |
| packed `TT = 8`, allocator | 146 | 0 | 4 | 80 |
| packed `TT = 8`, `waves_per_eu(12)` | 120 | 12 | 5 | 100 |

The packed rows have moved too — 140 registers when this document was written, 146 now — and the
packed rule is still right for the same reason it was then.

### What the block is worth on the route the machine serves

One process, one loaded model, one clock, `--pin-clock`, the deployed eight-row route with the
DFlash2 q4 drafter, a 1604-token prompt and 96 greedy tokens. `--pk-occ 0,1,1,0` walks the budget as
a case axis in the palindrome order, so drift across the panel shows up as a disagreement between
the two cells of each arm rather than as a result. Raw:
[`batch-comparison/rows-occ-fp32/panels.txt`](../../../data/bonsai2/batch-comparison/rows-occ-fp32/panels.txt).

| budget | grid | prompt tok/s | drafted tok/s | verify ms/step | draft ms/step |
|---|---:|---:|---:|---:|---:|
| allocator's, 141 VGPR | 80 | 143.1, 143.8 | 42.36, 42.12 | 57.675, 57.950 | 8.978, 9.082 |
| **wide, 120 VGPR (selected)** | **100** | **172.1, 172.2** | **49.10, 49.04** | **48.533, 48.607** | 8.964, 8.967 |

**+20.0% deployed prompt ingestion, +16.2% drafted generation, -16.0% on the verify pass.** The
drafter's own phase is the in-process control and does not move: 9.030 ms against 8.966, -0.7%, on a
kernel this change cannot reach. The greedy digest `17642979010498673847` and the exact integers 238
drafted / 61 accepted are identical in all four cells. This panel's agreement does not prove
repeated-run identity for the old K-split drain.

Those numbers also reproduce this document's own earlier panel from the other side. The packed arm
measured 140.3/141.0 tok/s and 57.5 ms at grid 80 and 172.8/172.9 and 46.75 ms at grid 100; the fp32
arm now measures 143.5 and 57.8 at grid 80 and 172.2 and 48.6 at grid 100. **The grid is the
variable and the state coordinate is not**, which is the sentence the rule should have been written
as in the first place.

### The sixth block is reachable and it loses

[The next useful question](#the-next-useful-question) left one open: `96` registers is six blocks per
WGP, grid 120, the most this kernel's 9508 bytes of LDS can ever hold, and `waves_per_eu(13)` is how
to ask. It is now measured, in a control build with the same in-process axis:

| budget | grid | VGPR spills | prompt tok/s | drafted tok/s | verify ms/step |
|---|---:|---:|---:|---:|---:|
| allocator's, 141 VGPR | 80 | 0 | 143.9, 144.0 | 42.33, 42.25 | 57.730, 57.814 |
| wide 12, 120 VGPR | 100 | 7 | 172.1, 172.2 | 49.10, 49.04 | 48.533, 48.607 |
| ask 13, 96 VGPR | **120** | **31** | 163.8, 164.0 | 47.14, 47.16 | 50.862, 50.912 |

**The largest grid this kernel can hold is 4.9% of prompt ingestion and 3.9% of drafted generation
behind the middle rung.** The sixth block is real and it arrives, and the 24 extra spilled registers
cost more than it pays for. The cross-build comparison is legitimate here for one reason only: the
allocator cell is in both builds and reproduces to 0.6% (143.1/143.8 against 143.9/144.0, verify
57.675/57.950 against 57.730/57.814), so the two wide cells are anchored to a shared control.

That result has a consequence past this kernel. The standing advice in the handoff was that cutting
LDS below 8192 bytes a block — the only thing between this kernel and **eight** blocks per WGP — is
worth more than any register work. It still might be, but it is no longer free of the register
question: eight blocks needs 1408/16 = 88 registers at the same time, which is past the 96 that
already costs 31 spills. **An LDS cut for occupancy has to be priced together with the register
budget the extra blocks would force**, not as a separate lever.

### The instrument was answering, which is why this took 100 minutes

`--pk-occ 0,1` is listed in this document as the way to order two register budgets under one clock.
For the fp32 coordinate it was a **silent no-op**: `rows_fn_wide` fell through to `rows_fn_narrow3`
for `pk == false` unless the build defined `HALO_ROWS_OCC_CONTROL`, so both arms resolved to the
same function. Run on canonical `8d87e77` it reports

    pk-occ 0   prompt 143.8 tok/s   drafted 42.27   verify 57.817 ms   digest 17642979010498673847
    pk-occ 1   prompt 144.1          drafted 42.29   verify 57.784      digest 17642979010498673847

which is a correct measurement of one kernel twice, presented as a budget comparison, at the exact
moment the budget had started to matter. A missing instrument sends you to build one; an instrument
that answers ends the investigation.

Both fp32 eight-row arms are now in **every** build. That is two more instantiations of the largest
body in the repository and it was measured before being bought: `kernels/halo_rows.o` goes from about
35.5 to 37.5 seconds, because the two arms differ only in their register constraint and share almost
all their optimisation work. The packed coordinate keeps its control behind
`-DHALO_ROWS_OCC_CONTROL=1`, where asking for an arm the build does not carry silently gets the route
and `HALO_ROWS_GRID=1` says which one ran.

`-DHALO_ROWS_WPE=N` now re-asks the whole build, which step 2 of the refit procedure above has been
telling engineers to do since it was written without the define existing. The grid-120 panel is
`-DHALO_ROWS_WPE=13`, and `rm kernels/halo_rows.o` first: `make` does not see a changed `-D`.

### The seven spills cost nothing, and finding that out needed an instrument repair

The budget and the grid arrive together in the table above, so the panel does not say which one paid.
`--grid-rows` crossed with `--pk-occ` separates them: pin the constrained build to the grid the
unconstrained one gets and only the spills are left.

**That cross was silently broken.** `Engine::set_grid_rows(-1)` cleared the *override* and not the
*parameter*, and the launcher only ever writes `fwd.grid_rows`, so once a cell pinned a grid every
later cell in the arm list inherited it. `--grid-rows 80,-1` measured grid 80 twice and labelled one
cell the occupancy grid. The first isolation panel read 54.8 to 56.1 ms across all four cells and
looked like a null; it was one configuration measured four times. `set_grid_rows` now clears
`fwd.grid_rows` when the pin is released.

With that fixed, `--pk-occ 1 --grid-rows 80,-1,80,-1` alternates one budget across two grids, so the
register allocation, the seven spills and the scratch are identical in every cell and the only
variable is the workgroup count:

| cell | grid | verify ms/step | drafted tok/s |
|---|---:|---:|---:|
| 1 | 80 | 55.936 | 70.16 |
| 2 | **100** | **50.630** | **75.70** |
| 3 | 80 | 63.554 | 62.05 |
| 4 | **100** | **47.751** | **80.17** |

**-17.7% on the verify pass from the grid alone**, monotone in both adjacent pairs. The within-arm
spread is wide because this panel ran with ten engineers compiling and measuring on the host, which
is also why the headline table above uses a 1604-token prompt and the palindrome order; take this
cell pair as the sign and the size, not as a third decimal place.

So the whole 16% is the grid and the seven spilled address pairs are free at this occupancy. That
also settles what this document previously recorded from the other side: the -3.5% it measured for
five spills was at grid 100 in both arms, and it does not reproduce as a cost here. **A register cut
in this kernel is worth nothing until it crosses a block boundary.**

### Installed acceptance

Canonical `e1f8176`, installed `bonsai-halo` SHA-256
`d7a8b1e2195b5d85a49bf18e906a66af76766204f586a74a8eeda1d36f2f6369`, `tools/batch_compare` `0c99a80fd1f388b140072442`, `tools/batch_profile`
`0d59a6ae0dfe48d59be10605`, `tools/rows_occ_probe` `c8f273e5619b550c1c994e56`, all run from the
canonical tree through its own `tools/run-batch-compare`.

| build | arm | prompt tok/s | drafted tok/s | verify ms/step |
|---|---|---:|---:|---:|
| `b4e5365`, 1604-token prompt | wide | 171.7 | 65.80 | 48.555 |
| | allocator | 144.3 | 56.61 | 57.830 |
| `e1f8176`, 40-token code prompt | wide | 168.3, 172.3 | 82.93, 80.51 | 46.191, 47.569 |
| | allocator | 144.6 | 70.96 | 55.293 |

Both panels reproduce the development numbers, and `e1f8176` is **+15.2% drafted generation and
-15.2% on the verify pass** with one digest and the same 189 drafted / 92 accepted in all three
cells. Single-stream plain decode on the installed binary is 33.43 tok/s, inside this box's
33.05-33.80 range and untouched code. `bonsai-halo.service` was restored by the wrapper and serves
`bonsai-2-27b` on 127.0.0.1:8471 from the installed binary. Raw:
[`batch-comparison/rows-occ-fp32/panels.txt`](../../../data/bonsai2/batch-comparison/rows-occ-fp32/panels.txt).

### The ceiling this document left is measured, and 100 is the top of the curve

"LDS at 9508 bytes a block caps this kernel at six blocks whatever its registers" was read as a
ceiling waiting to be lifted. [`docs/rows-lds-occupancy.md`](rows-lds-occupancy.md) lifts it —
`rows_lds_floats` sizes the forward block to its own phases, 9508 -> 7204 bytes, and
`waves_per_eu(13)` then reaches 96 registers and base grid 160 — and measures every grid from 80 to
160 on one binary: **80 +24.0%, 100 the deployed point, 110 -3.1% inside a build that starts 2.3%
behind, 120 +6.1%, 140 +5.8%, 160 +14.9%.** The step this document won is real and it is the last
one. `rows_occ_rule` and `HALO_ROWS_WPE = 12` stay exactly as they are.

### What the rule says now

```c++
static int rows_occ_rule(bool pk) { (void) pk; return 1; }
```

Both coordinates want the wide budget, for the same reason and not by coincidence: both allocate
above 135 registers today. Keep the parameter. The next body change moves these numbers again, and
when it does, the thing to re-read is the probe, not this table.

<!-- MEASUREMENT -->

### The block above this one, measured

[`docs/pk-register-peak.md`](pk-register-peak.md) took the next rung. The 96-register budget is
reachable without the 31 spills - the whole mark is `ph_attn`'s `qr[TT][DPL]`, and grouping the score
unit's rows takes the body to 117 VGPRs and 7 - and the sixth block still **loses 18.6%** of the
verify pass. The grid ladder in one process is 54.665 / 46.288 / 43.458 / 45.481 / 51.987 ms at 80 /
90 / 100 / 110 / 120 workgroups, so the rule above sits on the maximum of a unimodal curve, and the
three budgets 117/0, 96/7 and 96/33 are one measurement at a fixed grid. **The spills this document
called free are free at every count that has been measured, and the grid is the only thing this
kernel's register count buys.**

> **Two engineers reached this rung the same hour from opposite sides and agree.**
> [`docs/pk-register-peak.md`](pk-register-peak.md) got to 96 registers with the spills removed and
> measured the sixth block losing anyway; the section below measures what the device will hold at
> all, why the launcher refuses some of it, and what the grid is worth at a fixed register budget.
> Their grid ladder is finer than mine and theirs is the one to quote: 54.665 / 46.288 / 43.458 /
> 45.481 / 51.987 ms at 80 / 90 / 100 / 110 / 120 workgroups, a unimodal curve with its maximum at
> the deployed 100. Mine walks the other side of it at a second budget - 49.210 / 49.334 / 50.918 at
> 100 / 120 / 140 - and reaches the same verdict, which is that **this pass has a knee at 100
> workgroups and nothing above it is worth a register.**

## What the device actually holds, and why the ladder above is an estimator's

Every grid in this engine is `coop_grid`'s answer, and `coop_grid` asks
`hipOccupancyMaxActiveBlocksPerMultiprocessor` and then clamps it with a granule-24 model. Both are
predictions, they disagree, and nothing in this repository had ever checked either one against the
machine. `bench/occ_resident` checks it: blocks announce themselves through one atomic, spin a fixed
1.5 ms of wall clock and leave, and the peak of the live count is how many the device held at once.
There is no grid sync in it, so it cannot deadlock the way a mistaken cooperative launch would.

**The hardware follows the granule-24 arithmetic in all 48 cells measured** — twelve register counts
across four LDS block sizes — and the occupancy API is wrong in both directions.

| VGPRs the loader reports | API says | **device holds** | grid |
|---:|---:|---:|---:|
| 69, 93 | 8 | **8** | 160 |
| 97, 98 | 7 | **6** | 120 |
| 108 | 6 | 6 | 120 |
| **116** | **5** | **6** | **120** |
| 121, 122 | 6 | **5** | 100 |
| 132 | 5 | 5 | 100 |
| 145, 150 | 4 | 4 | 80 |
| 180 | 3 | **4** | 80 |

The model that fits: registers round up to 24, `waves = min(1536 / that, 16)`, a 256-thread
workgroup is two waves per SIMD32, so the rungs are **8, 6, 5, 4, 3, 2 blocks and nothing between**.
There is no seven-block rung at all — 1536/144 is ten waves and 1536/168 is nine — which is why the
API's 7 at 97 and 104 registers is not merely optimistic but describes a state the machine has no
way to enter. In grid terms: **96 VGPRs and below is 160, 97 to 120 is 120, 121 to 144 is 100, 145
to 192 is 80.**

### The LDS cap in this document never existed

The ladder at the top says "LDS at 9508 bytes a block caps the whole thing at six however few
registers a kernel uses", from `65536 / 9508`. Measured: **eight blocks of a 9508-byte kernel are
co-resident**, which is 76 kB. `sharedMemPerBlock` is 65536 because that is the most one workgroup
may allocate; the WGP has 128 kB and hands it out to as many blocks as fit. At 7204 bytes the LDS
term allows eighteen blocks and the register rungs top out at eight, so **LDS does not bind any
kernel in this engine** and `rows_lds_floats`'s "9508 is six workgroups and 7204 is nine" is the
same 64 kB reading. `coop_grid` now carries the 128 kB term explicitly; it changes no grid today.

### The API's answer is still a ceiling, which is the part that decides the change

Taking the model alone looks like free money at 120 registers: the device holds six blocks and the
engine launches five. It is not available. `hipLaunchCooperativeKernel` validates the grid against
its own estimate and refuses:

```
wide prompt ingestion: mode 20 (default), 256 rows per pass
prompt: 23 tokens
slice launch failed: too many blocks in cooperative launch
```

So a block the occupancy API does not believe in cannot be launched however real it is, and
`min(api, model)` is restored — with both terms now justified by measurement rather than by caution.
The model term is not redundant: at 121 to 144 registers the API over-counts, and launching its six
blocks where the device holds five would wait forever at the first `grid_sync`.

**Where the two agree is therefore the only reachable ladder**: 96 registers and below (eight
blocks) and 105 to 112 (six). `waves_per_eu(12)` lands the eight-row body on 120, which is inside
the disagreement, so its grid is set by an estimator's arithmetic and the only way out is to move
the allocation into an agreeing window.

### What that block is worth, measured on the route the machine serves

`waves_per_eu(13)` is the reachable way down: 96 VGPRs, and the price is 33 VGPR spills and 128
bytes per lane of scratch against the deployed 8 spills and 36 bytes. Both budgets in one
executable (`-DHALO_ROWS_OCC2=1`), one process, one clock, `--pin-clock`, the deployed eight-row
drafted route with the DFlash2 q4 drafter:

| budget | grid | verify ms/step | drafted / accepted |
|---|---:|---:|---:|
| `waves_per_eu(12)`, 120 VGPR — deployed | 100 | **43.540** | 49 / 5 |
| `waves_per_eu(13)`, 96 VGPR | 140 | 47.673 (+9.5%) | 49 / 5 |

And the two halves of that separate cleanly, because `--grid-rows` walks the workgroup count at a
fixed register budget. **The grid past 100 is worth nothing:**

| 96 VGPR | grid 100 | grid 120 | grid 140 |
|---|---:|---:|---:|
| verify ms/step | 49.210 | 49.334 | 50.918 |

bit-identical across the three measured cells (greedy digest `14794818928082561934` in every
cell). It does not establish a general grid-order bound. **The grid below 100 is worth a
great deal:**

| 120 VGPR | grid 60 | grid 80 | grid 100 |
|---|---:|---:|---:|
| verify ms/step | 53.092 | 52.615 | **42.715** |

So the pass has a knee at 100 workgroups and the deployed budget is sitting exactly on it. The spill
cost alone, comparing the two budgets at the same grid of 100, is **+13.0% to +15.2%** (43.540 and
42.715 against 49.210).

### Three things this closes

1. **The register cut that three engineers queued is dead on this route.** `docs/mv-tile-fold.md`
   priced a forty-five-register cut at "about +21% of every matvec phase in the pass" from
   `bench/mvsched`'s co-residency curve (150.4 GB/s at four blocks, 135.7 at five, 181.5 at six).
   Measured in the engine, the cut that reaches the next rung costs 13 to 15% and the grid it buys
   returns nothing. A synthetic body at a fixed co-residency is not this pass: at 100 workgroups the
   eight-row route already has the memory parallelism it can use.
2. **`waves_per_eu` is the only budget worth asking for.** `__attribute__((amdgpu_num_vgpr(N)))`
   reaches exact ceilings the occupancy ask cannot name (128, 112, 104), and it spills harder at
   every one of them: 15 spills at 128, 23 at 112, 30 at 104, against 8 at the twelve-wave ask's
   120. `HALO_ROWS_VGPR` is in the tree as a census axis and is not a route.
3. **Prompt ingestion cannot be ranked on the drafted bench.** A 1697-token prompt reads 454.0 then
   486.0 tok/s at `--pk-occ 1,2` and 493.7 then 541.3 at `--pk-occ 2,1`: the second arm wins in both
   orders by more than the difference being measured. Digests are identical within each panel, so
   this is the panel warming and not the budget. Whoever wants that number needs an instrument that
   interleaves rounds rather than arms.

Raw samples, both probes and every panel with its clock and host-load line:
[`batch-comparison/rows-grid-step/`](../../../data/bonsai2/batch-comparison/rows-grid-step/README.md).

### The instruments

- `bench/occ_resident [spin_us]` — what the device co-schedules, by register count and LDS block.
  Launches no cooperative kernel and cannot hang. This is the ground truth the two predictions in
  `coop_grid` are estimates of.
- `tools/occ_surface [lds ...]` — the occupancy API's whole surface over (registers, LDS, workgroup
  size), 1.5 s to build, no GPU work at all. Use it to find where the API and the hardware disagree
  *before* spending registers, because only the agreeing windows are launchable.
- `-DHALO_ROWS_OCC2=1` puts both wide budgets in one executable so `--pk-occ 1,2` interleaves them;
  it is off by default because it is four more compiles of the largest body here.
- `-DHALO_ROWS_VGPR=N` asks the allocator for an exact ceiling, for the census only.

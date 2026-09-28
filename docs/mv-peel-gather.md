# The trit is already a byte of the product, so stop shifting it

The deployed FFN matvec spends about 230 of its 488 work instructions per 128-K block turning 26
stored bytes into 32 operand dwords. `2f17d1b2` found that a third of that work exists only to move
a value that is already where it needs to be, and wrote the replacement; this is that change in
`mvw_rows`, the body the served prompt route spends 60% of a pass in.

**Block loop 561 → 476 instructions at 238 VGPR either way, `ffn` phase -1.66% normalised in three
paired rounds, full-model prompt ingestion +1.40% median paired over seven, and the same residual
FNV-64 in every sample.**

## The observation

A HALO block stores 128 trits of one row in 26 bytes, radix-3, five trits to a byte pair. The peel
takes the next trit out with

```c
const unsigned m = r * 3;        // v_pk_mul_lo_u16 on two 16-bit lanes
r = m & 0x00ff00ff;              // the remainder
return (m >> 8) & 0x00ff00ff;    // the trit
```

and a later `v_lshl_or_b32` puts that trit in the byte lane the operand wants. But `b * 3 <= 765`,
so the high byte of each 16-bit product **is** the trit: byte 1 of `m` is `(b_first * 3) >> 8` and
byte 3 is `(b_second * 3) >> 8`, each a bare 0, 1 or 2. The shift only moves a byte to another byte,
and the or only moves it back.

So keep the products and gather. One `v_perm_b32` with the constant selector `0x07030501` takes
bytes 1 and 3 of two products straight into an operand dword:

```c
tr[4 * d] = hx_gather(m1, m0);   // __builtin_amdgcn_perm(m1, m0, 0x07030501)
```

Per source dword that is 26 operations for 20 trits instead of 36; per 128-trit block, **168 instead
of 232**. `kernels/halo_expand.hpp` holds both maps — `hx_expand_peel` is the deployed one written
out, `hx_expand_perm` is this one — and they are `__host__ __device__`, so the same source runs on
the CPU.

## Exactness

Not an argument: `kernels/head_op_check` runs both maps and the two-bit spread coordinate against
`halo::decode_block` with no GPU.

    placement, arm 1, arm 2 and the spread bit map agree over 3 uniform, 256 byte-sweep and
    4000 random blocks

`make kernels/head_op_check` builds it in a second. Because the gather produces the same `tr[32]`
byte for byte from the same bytes, every consumer keeps its accumulator seed, its matrix
instructions, its group fence, its reduction order and its output bits. The residual FNV-64 of a
traced 128-row pass is `8785572807677303713` in **all six samples of the panel below**, the value
this route produced before the change.

## Census

`hipcc --cuda-device-only -S`, gfx1151, the instantiation a 32-row slice launches
(`COLS = 16, GROUPS = 2`), gate/up block loop:

| | instructions | work | `s_delay_alu` | bitops | `other:v` | `v_perm_b32` | `v_wmma` | `global_load` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| peel | 561 | 488 | 50 | 103 | 66 | 32 | 32 | 23 |
| **gather** | **476** | **422** | 32 | 71 | **2** | 64 | 32 | 23 |

-15.2% of the loop and -13.5% of its work, with the matrix instructions and every memory instruction
unchanged. The down projection's loop moves the same way, 546 → 476.

**Registers do not move, at any budget**, which is what makes this free rather than a trade
(`tools/kernel_resources.py`, one compile):

| `waves_per_eu` | peel VGPR / spill | gather VGPR / spill |
|---:|---:|---:|
| 6 (deployed) | 238 / 0 | **238 / 0** |
| 7 | 205 / 0 | **205 / 0** |
| 8 | 192 / 18 | 192 / 24 |

Both arms take grid 60 workgroups, printed per arm by `HALO_FFN_GRID=1`. `2464a25e` measured that
asking this kernel for the fourth workgroup per WGP loses at both widths that can reach it, so the
six-wave row is the one that matters and it is a wash.

## Measured

Mode 20 (`wide-deployed`, the route the resident service ingests prompts on), both arms alternating
inside one process on one build and one pinned clock, the five untouched phases as the in-panel
control.

**The phase**, 128-row passes, three rounds (`peel-128.json`):

| arm | `ffn` ms | `ffn` / control, per round |
|---|---|---|
| peel | 169.993, 177.109, 178.220 | 3.0210, 3.0447, 3.0496 |
| **gather** | 165.756, 172.656, 175.413 | **2.9745, 2.9913, 2.9980** |

**-1.66% normalised, -2.19% raw, and the gather arm is ahead in all three paired rounds.**

**Full model**, a 384-token document in 128-row passes, seven paired rounds (`peel-full2/run.json`):

| round | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| peel tok/s | 559.3 | 556.4 | 549.2 | 573.5 | 570.9 | 572.8 | 571.9 |
| gather tok/s | 568.6 | 556.1 | 597.2 | 578.1 | 578.2 | 580.8 | 582.8 |
| paired | +1.66% | -0.05% | +8.75% | +0.80% | +1.27% | +1.40% | +1.91% |

**+1.40% median paired**, six of seven rounds positive. Round 2's +8.75% is one cell of a box
carrying six other engineers, not a result; the median and the phase panel are.

**Expect less than the census ratio.** -15% of the block loop bought -1.7% of the phase, because
this kernel is only partly issue-bound — the same arithmetic that made
[the operand coordinate](ffn-slice-operand.md) cost 4% for +10.5% of slots.

### Installed acceptance

Canonical `c074159`, installed `bonsai-halo`
`027acc6d8e1262e61cbdf9e105dc497ea9ee5eddcf6b15f1d1759abd561705bf`, `tools/batch_compare`
`506736d357d90fee5c0156b8a0545d374c5ca25c641b67caf03f378baa9d0d9c`, `tools/batch_profile`
`963ffbc2432c136529ef16e4eb2c81e9c9b51f3d248cebc2e8000d2d07fac884`.

- **Bits, on the installed binary**: `--prefill-identity` at 128 and 256 rows a pass, 95,354,880
  full-vocabulary logits each, FNV `9524621175407926160`, zero non-finite — **the value the binary
  before this change produced**, published in the negative above from four arms and two coordinates.
  The gather reproduces the peel's logits exactly on the shipped executable.
- **Prompt**, mode 20, 384-token document, three rounds at each width: 548.9 / 513.8 / 591.3 tok/s
  at 128 rows and 539.3 / 550.6 / 579.9 at 256, on a box with five other engineers measuring and a
  process that also holds the drafter images (15.46 GB on device). That spread is wider than this
  change, which is why the paired in-process panels above are the result and this run is the
  regression check.
- **Generation does not move by construction**: a one-row step takes the scalar-fed `dot4` body and
  the eight-row verify pass takes it too, so neither reaches `mvw_rows` on the ternary image.
- `bonsai-halo.service` came back on the new binary and answers `/health` with
  `{"active":0,"max_active":4,"status":"ok"}`.

## What this leaves

- **The head is the better consumer and it is being built.** `docs/wide-head.md` puts 220 of the
  vocabulary head's 600 block slots in this expansion at 24 GB/s of a 242 GB/s roof, against this
  kernel's 76 GB/s, so a loop that is more issue-bound should read a bigger number. `0dad0d38` has
  `head_tile`.
- ~~**`mv_rows_t` and `mv_rows_pal` still peel.**~~ **Done, and it came back with a rate.** `bdf1e3f`
  put `mv_expand` and `mv_rows_deployed` on this header and
  [the expansion axis](mv-peel-rows.md) then measured what the map is worth on the kernel itself:
  **-3.0% and -3.8% of the isolated eight-row body in two panels**, +1.8% and +2.2% on single-token
  decode, and a third arm that deletes the expansion entirely for -12.2% and -13.2%. That ladder
  prices **one issue slot per 128-weight block at 0.054% of the phase**, twice, which is the number
  to multiply a census by before building anything for this loop — and it says the eight-row wall is
  not the expansion: with every decode instruction gone the body still reads 150 GB/s where one row
  reads 226.
- **Arm 2 is priced and not built**: a two-bit spread image, `(w >> 2k) & 0x03030303` *is* operand
  dword k, 56 operations a block instead of 232 for 21.4% more weight bytes. `halo_expand.hpp`
  carries the coordinate and `head_op_check` verifies it, so it is one image build away for whoever
  wants to spend the bytes. On this kernel, at 76 GB/s, more bytes is the wrong direction; on the
  head at 24 GB/s it may not be.

## Credit and provenance

`kernels/halo_expand.hpp` and `kernels/head_op_check.cpp` are `2f17d1b2`'s, written for
`head-operand-map` and published on the lane board at 18:55 with the census that names arm 1 and
arm 2. They sat uncommitted in an idle checkout; they are preserved here verbatim, with their
Makefile target, so the map is in canonical for every consumer rather than in one engineer's tree.
`2464a25e` holds `mvw_rows` and handed the call site over in writing.

Raw samples are in
[`batch-comparison/ffn-slice-operand/`](../../../data/bonsai2/batch-comparison/ffn-slice-operand/README.md),
beside the negative that pointed at this loop.

## The second consumer took it, and there it is a null

`mv_expand` and `mv_rows_deployed` in `kernels/device.hpp` — the eight-row verify body
`bonsai-halo.service` generates on — now call `hx_expand_perm` instead of carrying their own copy of
the peel. Same 32 operand dwords, same 120 VGPRs, same 14 B of scratch in `k_forward_rows<8>`, and
**verify 42.628 ms against 42.617** with equal digests, alternating binaries under one lock. The
change ships for the single map, not for the time. That body is not issue-bound: a 27.6% cut of its
decode arithmetic at zero register cost is the fourth such null on it.
[`docs/rows-lds-occupancy.md`](rows-lds-occupancy.md) has the panel.

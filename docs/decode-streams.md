# Sequences per generation step: 64 is the operating point, not 32

[Generation at 64 and 128 streams](generation-128.md) measured this axis on `1c7d7f2` and concluded
that doubling from 64 to 128 streams bought 11.8%: 463.14 and 517.73 aggregate tok/s. Every
*continuing* number since — 264.3, 352.9, 383.32, and the phase maps the lane chooses work from —
is 32 streams, because `tools/batch-compare/prompts.txt` holds 40 usable prompts and
`Driver::prepare_inputs` refuses a decode workload it cannot give one distinct prompt per row.

**The curve has changed shape since then, and the lane has been choosing work at the wrong width.**
On `b4e5365`, 64 streams reaches **597 tok/s — above the 517.73 that panel measured at 128** — and
is 25% faster than the 32-stream operating point on the same executable, the same defaults and the
same output tokens. A generation step costs 1.67 ms per token at 64 streams against 2.10 at 32.

| streams | aggregate tok/s | per stream | step ms | ms per token |
|---:|---:|---:|---:|---:|
| 8 | 143.4 / 145.8 | 17.9 / 18.2 | 55.6 | 6.95 |
| 32 | 476.1 / 477.8 | 14.9 / 14.9 | 67.2 | 2.10 |
| 48 | 471.4 / 472.6 | 9.8 / 9.9 | 101.9 | 2.12 |
| 48 (pair-only images) | 470.4 / 472.6 | 9.8 / 9.9 | 101.7 | 2.12 |
| **64 (pair-only images)** | **597.7 / 596.4** | 9.3 / 9.3 | 107.0 | **1.67** |

Mode 19, INT8 state, defer 4, A4 sequence input, context 256, 16 greedy steps per timed region,
two rounds, `--pin-clock`. Raw in
[`batch-comparison/decode-streams/`](../../../data/bonsai2/batch-comparison/decode-streams/):
`curve-a/run.json` (8/32/48, `8d87e77`, both A4 images) and `curve-b/run.json` (48/64, `b4e5365`,
pair-only). **The 48-stream cell is the bridge between the two panels** and it reproduces to 0.5%
across two builds *and* across both image arms, which is what makes the 64-stream cell comparable
to the 32-stream one.

```sh
tools/run-batch-compare --pin-clock --memory-gib 34 --tag decode-streams/curve-b \
  --only decode --modes 19 --streams 48,64 --slots 64 --context 256 --a4-images wide \
  --gen-steps 16 --rounds 2 --gdn-state 3 --gdn-defer 4 --seq-quant a4 --warmup-steps 2
```

Accepted on the installed canonical build `5eb5fcb`, both widths in one process against the
published prompts file with no override (`decode-streams/installed/run.json`):

| streams | aggregate tok/s | per stream |
|---:|---:|---:|
| 32 | 440.3 / 440.2 | 13.8 |
| **64** | **597.5 / 600.0** | 9.3 |

Both cells take the pair-only image, which is what makes them one arm and also what costs the
32-row cell its dense gate/up: **+36% inside one process and one image arm, +25% against the 477
the 32-stream cell reaches with the image rule it would normally use.** The 64-stream cell
reproduces `curve-b` to 0.4% on a different executable.

## A row's tokens do not depend on how many sequences share its step

The wide route builds `wide_rows` per (sequence, token) and every row's arithmetic depends on its
own sequence's state and its own activation scale, so widening a step should be invisible in the
output. It is, measured rather than argued — `streams_out` in each `run.json` carries every
stream's 16 generated tokens:

- **48/48 streams identical** between the 48-row and 64-row steps of one process.
- **32/32 streams identical** between 32 rows on `8d87e77` with both images and 64 rows on
  `b4e5365` with pair-only.
- **32/32 streams identical** between 32 and 48 rows of one process.

Image choice is exact too: `residual_fnv64` is equal across all four `--ffn-image` arms at every
width (`196708199796562632` at 48 rows, `13239221889136524637` at 64), so the pair-only restriction
that buys the memory for 64 slots is a timing axis and nothing else.

## 48 rows is a trap, and the FFN phase is flat from 40 to 64

The 33..127 row band was named unfitted in [the decode map](decode-map.md) and it is worse than
unfitted: it is **non-monotone in throughput**. 48 streams is *slower in aggregate* than 32. The
reason is the FFN phase, priced across the band in one process, one clock, one warmed device
(384-token document, tail-only head, `--ffn-image` as a case axis, ms of `ffn`):

| rows | dense five-trit | pair codes | dn5 gate/up + pair down | pair gate/up + dn5 down |
|---:|---:|---:|---:|---:|
| 40 | 48.593 | 44.743 | 48.455 | 44.955 |
| 48 | 48.879 | 45.135 | 52.155 | 45.178 |
| 64 | 49.274 | **41.022** | 50.106 | **40.134** |
| 96 | 73.957 | 69.084 | 72.393 | 71.159 |
| 128 | 93.275 | **72.407** | 88.299 | 74.459 |

Read the best arm at each width: 44.7, 45.1, **40.1**, 69.1, 72.4. The phase is flat from 40 to 64
rows and then steps by 70%. **64 rows is the widest pass the FFN gives away for free**, which is
exactly why 64 streams wins: it buys twice the tokens of a 32-row step for 9 ms more FFN, while 48
streams pays 45 ms of FFN for 16 extra tokens.

Raw: `ffn-image-band.json`, `ffn-image-band2.json`.

### Two bounded negatives inside that table

**The A4 image rule is not the band's problem.** The rule takes pair gate/up above 32 rows and pair
down above 64. Against the best arm at each width it is within 0.1% at 48 rows and 2.2% at 64, and
it is already the best arm at 96 and 128. Nothing in the band is waiting for a better image
threshold. (`A4_GU_PAIR_ROWS = 32`, `A4_DN_PAIR_ROWS = 64` in `kernels/ffn_batch.hip`.)

**The step is not a weight-stream re-read.** The obvious story for a staircase — "a 128-row batch
needs two passes at four token tiles" — is refuted by the counter. `FETCH_SIZE` per FFN launch is
**constant across 32, 64 and 128 rows**: 27.31 MB per down launch on the 69,632-workgroup grid and
14.45 MB per gate/up launch on the 20,480 one, with per-64-launch group means holding inside 1% for
all 23 passes of the panel. The grid does not change with the row count either. So the step between
64 and 96 rows costs no extra bytes; it is issue or schedule inside one instantiation, and whoever
holds `k_proj_opt` should look for it there. Raw: `fetch-band.json`, `fetch/gpu-host/*.csv`.

## What the marginal sequence actually costs

Phase maps of a generation step on today's build, INT8 state, defer 4, A4 sequence input, logits on
every row (`phase-s32.json` on `8d87e77`, `phase-s48.json` on `b4e5365`):

| phase | 32 streams | 48 streams | ms per extra sequence |
|---|---:|---:|---:|
| `ffn` | 32.204 | 46.380 | 0.886 |
| `gdn-resident-core` | 15.582 | 21.610 | 0.377 |
| `sequence-input-projection` | 9.353 | 11.993 | 0.165 |
| `sequence-output-projection` | 7.387 | 8.761 | 0.086 |
| `head-projection` | 3.103 | 5.134 | 0.127 |
| `sequence-core` | 1.535 | 2.177 | 0.040 |
| `sequence-input-prep` | 1.827 | 1.782 | -0.003 |
| device span | 72.118 | 98.968 | 1.678 |

Two things worth carrying. The recurrent phase is **no longer the step's dominant cost** now that
the state is INT8 with a deferred commit: 15.6 ms and 21.6% where the decode map measured 48.5 ms
and 38% in FP32. And `sequence-output-projection` shows **no sign of the TT=4 cliff** that document
reported (12.924 ms at 32 rows against 28.467 at 64) — it is 7.387 and 8.761 here, so either
something between `ce4ba22` and today removed it or the decode shape never took that instantiation.

## What 128 streams costs to run, and the allocation that makes it expensive

[Generation at 64 and 128 streams](generation-128.md) got 128 slots to run and says exactly what it
took: `HALO_SNAPSHOT_KV_GB=0`, `HALO_MANAGED_ALLOC=1`, `--memory-gib 43 --host-reserve-gib 7`, and
**43.81 GB of tracked allocations**. That budget needs 50 GiB of `MemAvailable` at admission, which
this box reaches only just, and only immediately after the resident engine stops. It was not
re-taken here for that reason; with 597 tok/s at 64 streams and that panel's own +11.8% step it is
the first thing the next engineer should measure.

The arithmetic of why it is that expensive, which that document names in one line and is worth
spelling out:

- A slot costs **243 MB** at context 256. 151 MB of that is the recurrent state region, 22.5 MB is
  KV, and about 69 MB is everything else per slot.
- The state region is allocated `GDN_STATE_FLOATS * 4` bytes per (slot, layer) **whatever the
  format**, because the packed coordinates address their values, scales and pending triples at
  offsets derived from that constant. The INT8 coordinate uses 768 kB of values, 24 kB of scales
  and the deferred triples — about **1.17 MB of the 3 MB** it is given.
- GTT total on this box is **32.46 GB** and VRAM is a 2 GB carve-out, so that is the real ceiling.
  64 slots with both A4 images asks 30.6 GB and `hipMalloc` fails inside `sequence_batch`; 64 slots
  with pair-only is 29.2 GB and runs.

A format-aware region stride would put the per-slot cost near 151 MB and **128 slots near 31 GB**,
inside the ordinary allocation path and the default-class budget rather than needing managed
allocation, a 43 GiB process limit and a 50 GiB host. It is not a numerical change: the region's
contents and their order are unchanged, only the distance between regions and the offsets inside
them.

And note what the estimate says about width: **96 streams is worth less than 64**, two
weight-stream-priced passes for 1.5x the tokens. The widths worth measuring are 64 and 128, not the
points between them.

## The next questions

1. **Re-take 128 streams on today's build** with the command `generation-128.md` records, and watch
   host `MemAvailable` rather than the device: admission needs 50 GiB and this box has been sitting
   at 38 GiB with the resident engine up. If it clears, the pair (64, 128) on one executable settles
   whether 64 is the operating point or merely today's reachable one.
2. **Make the region stride follow the format** and the same measurement stops needing any of that
   machinery. `GDN_PACK_SCALE_OFF`,
   `GDN_DEFER_CTL` and `GDN_DEFER_OFF` in `kernels/gdn_state_codec.hpp` are compile-time constants
   derived from `GDN_STATE_FLOATS`; a runtime stride reaches `resident_state`, `gdn_defer_*`, the
   snapshot and rollback paths and `Engine::load`. Everything that consumes it is exact.
3. **Find the 64-to-96 step in `k_proj_opt`.** It costs no bytes and no grid, and it is 28 ms of a
   96-row pass. Prompt ingestion pays it on every document whose tail pass lands in the band, and a
   packed multi-prompt ingestion pass will land there constantly.
4. **`--decode-streams` in `tools/batch_profile` takes one integer**, so a streams curve costs one
   process per width while `--decode-tokens` walks its axis inside one. The engine has to be loaded
   at the widest case and each case given a prefix of the slots; that is the same shape
   `--decode-tokens` already has.
5. **The 9..31 band and the 33..63 band are different questions.** `docs/decode-width-route.md`
   owns the floor at 32; this document says the band above it is non-monotone, so a route that
   admits 40 rows to the wide path is admitting them to the worst width in the map.

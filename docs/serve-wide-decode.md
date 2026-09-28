# The wide route's floor was never measured, and it is a map change from nine rows up

`forward_batch` had one expression deciding whether a pass runs the wide schedule or the sliced one:

```c++
const bool wide_sequence = batch_mode >= 17 && total >= 32;
const int  mode = ... automatic ? (total <= 4 ? 0 : total < 32 ? 4 : wide_mode) : batch_mode;
```

Nothing in a wide kernel asks for 32 rows. `wide_head` and `wide_attn` admit at `total > RMAX`, the
projections take their token-tile count from the row count, the resident state is per sequence, and
`k_ffn_slice` takes its rows as an argument. The floor was a constant nobody had varied, and the
band under it — 9 to 31 rows — had never been run at all.

It is a runtime setting now (`src/batch_route.h`, `HALO_WIDE_MIN`, `batch_set_wide_min`), the band
is measured on both shapes that reach it, and the answer is in two parts that have to be read
together: **the wide route wins from the first row that needs a second sliced pass, and it wins by
changing what the model emits.**

The floor mechanism, `src/batch_route.h` and the `--wide-min` axis are `8fe2db0e`'s design, taken
from their checkout and carried forward rather than rebuilt.

## The band, on a generation step

`tools/batch_profile --decode-streams N --wide-min` walks the floor as an in-process case axis: one
process, one pinned clock, one warmed device, `decode_setup` re-prefilling every slot per case, arms
alternating. Mode 20 (the wide schedule over this engine's own ternary FFN), serving defaults
(`f32` state, defer 1), one row per stream, logits on every row, medians of three rounds. Raw:
[`serve-wide-decode/band-*.json`](../../../data/bonsai2/batch-comparison/serve-wide-decode/).

| rows | sliced (`--wide-min` above) | wide | step | aggregate tok/s |
|---:|---:|---:|---:|---:|
| 8 | 59.35 ms | 60.27 | **+1.5%** | 134.8 -> 132.7 |
| 9 | 72.14 | **63.01** | -12.7% | 124.8 -> 142.8 |
| 12 | 69.58 | **59.81** | -14.0% | 172.5 -> 200.6 |
| 16 | 82.07 | **67.78** | -17.4% | 195.0 -> 236.1 |
| 32 | 143.07 | **106.94** | -25.3% | 223.7 -> 299.2 |

**The crossover is nine and it is arithmetic rather than a fit.** A persistent-kernel pass carries
`RMAX` rows and reads the whole weight stream whatever it carries, so the sliced cost steps up at
every ninth row while the wide cost grows smoothly with rows. At eight rows both routes read the
stream once and the wide one pays its launch overhead, which is the 1.5% it loses by. From nine
rows the sliced route is paying for a second pass and cannot win again.

Note which control this is: at `batch_mode` 20 below the floor, `mode` is 4 but `fwd.commit_state`
stays true, so the sliced arm here is the layer-wise schedule committing state without residency.
That is what the engine runs today below the floor, and it is the right control for *what the floor
does*. It is not the right control for *which route a server should call*; `bc30b5f7` measured that
one from the other end (mode 0 146.7 tok/s flat at 16 and 32 streams, mode 4 203.0 and 220.4, mode
20 205.9 and 301.8) and owns the serving switch.

**The other automatic mode takes the band too, and that is the capability check.** Mode 19 adds the
A4 FFN module on top of the wide schedule, and it had never run below 32 rows either; at 16 decode
rows it goes 78.07 -> 67.68 ms, -13.3%, 204.9 -> 236.4 aggregate tok/s, with no guard hit
([`band-16-mode19.json`](../../../data/bonsai2/batch-comparison/serve-wide-decode/band-16-mode19.json)).
It lands on 67.68 ms against mode 20's 67.78 at the same width: **the A4 FFN module is worth
nothing at 16 rows**, which is its own small result and the reason to prefer mode 20 there if
anyone routes a served step across this floor.

## The band, on a prompt pass

The same axis on the prefill shape, mode 20, logits on the tail row, two rounds, medians. This is
the shape the product reaches today: `Engine::prefill` runs `forward_batch` at mode 20 by default,
so a prompt shorter than 32 tokens, and the last chunk of every longer prompt, lands in this band.

| rows | sliced | wide | pass | tok/s |
|---:|---:|---:|---:|---:|
| 16 | 59.43 ms | **46.72** | -21.4% | 269.2 -> 342.5 |
| 24 | 83.72 | **66.27** | -20.8% | 286.7 -> 362.2 |
| 31 | 103.88 | **68.50** | -34.1% | 298.4 -> 452.6 |

End to end through the product binary, one lock hold, one pinned clock, a 31-token prompt,
alternating arms: **274.6 / 293.3 / 297.7 tok/s at the shipped floor against 358.7 / 392.1 / 390.9
with the floor at nine** — about +31% — with generation untouched at 33.09 against 33.23 tok/s,
which is the in-panel null the change predicts (the floor cannot reach `Engine::forward`).

## What it costs, measured rather than argued

The wide route is **not** the persistent kernel's map. Its projections reduce FP32 in a different
order, and since [`c131fe2`](seq-a4-default.md) and `f91a815` they quantise their activations to
four bits, which the sliced route never does. The same 31-token prompt, greedy, 48 raw tokens, both
arms run twice in one lock hold:

- each arm is deterministic and reproduces itself exactly (`9ebbde79…` twice, `755f615e…` twice),
- and the two arms **diverge** after about forty identical tokens.

So the floor is a numerical decision at every width it newly admits. The useful way to see it: the
product already serves that map to every prompt of 32 tokens or more. One token longer and the same
question gets the wide answer today. The floor's default is left where the engine shipped it
because moving it changes emitted text, not because the sliced arm is a better map.

### Greedy continuation identity, wide against sliced

`tools/batch_compare --only multistep --modes 0,20 --multistep-streams 32 --multistep-steps 6`,
32 real prompts, six greedy steps, both activation widths in one process. Raw:
[`serve-wide-decode/identity/run.json`](../../../data/bonsai2/batch-comparison/serve-wide-decode/identity/run.json).

| arm | streams identical to mode 0 | tokens |
|---|---:|---:|
| mode 20, `a4` (today's default) | 27 / 32 | 179 / 192 |
| mode 20, `a8` | **31 / 32** | **191 / 192** |
| mode 0, `a4` against mode 0, `a8` | 32 / 32 | 192 / 192 |

The last row is the control that makes the other two readable: the sliced route does not use the
sequence projections at all, so the activation axis cannot reach it, and it does not.

**The four-bit operand is what most of the divergence is, and at a generation width it buys
nothing.** Mode 20 at 32 decode rows: `a4` 106.87 / 108.67 ms against `a8` 108.94 / 108.47 — inside
the round-to-round spread, call it 0.5% or less. On a 128-row prompt pass the same operand is worth
[12.2%](seq-a4-default.md). A prompt pass has 128 rows of activations to quantise and a generation
step has 32, against the same weight stream; the operand is a per-row cost on a phase whose time is
per-byte, so it disappears when the rows do.

That separates cleanly into a recommendation, because the activation width is a runtime setting and
the route is another: **a process that runs the wide route for generation should pin `a8` there and
leave ingestion on the default.** It costs nothing measurable at decode widths and it removes four
of the five diverging streams. The remaining one is the FP32 reduction order, which is what
[`wide-sequence.md`](wide-sequence.md) reported as "127 of 128 continuation tokens" for mode 18 a
day ago; it is not an approximation and it does not go away by pinning anything.

## What this leaves

1. **The serving decode route is `bc30b5f7`'s and the band is priced for it.** Their exact switch
   (mode 4 above eight rows, bit-identical, 24/24 identical served completions) is the right first
   step and does not use this floor: mode 4 is not automatic. The floor is what a *second* step
   needs, and the arithmetic says where: at 16 rows the exact route is already 98.6% of the wide
   one, and at 32 rows the wide one is 1.37x it. So the question is only worth asking at 32 rows
   and up, where it is worth about 37% of aggregate served generation, and the answer wants one
   horizon panel on mode 20 + `a8` against mode 0, not a token-identity check.
2. **The prompt floor is one line and one panel away.** `WIDE_MIN_DEFAULT = RMAX + 1` in
   `src/batch_route.h` is the whole change and it is +21 to +34% on every prompt pass in the band,
   about +31% end to end on a short served prompt. What it needs before it ships is the instrument
   [`seq-a4-default.md`](seq-a4-default.md) built and `deadbba9` widened: 1024 paired teacher-forced
   predictions with the *ingestion route* as the axis rather than the activation width. Nobody has
   pointed that instrument at a route yet, and this is the cheapest question it could answer.
3. **Eight rows is the one width where the wide route loses, and it is the width the server sits
   at.** `--slots 4` and `--slots 8` are the deployed configurations, so the serving switch's band
   is 9..31 rows until concurrency grows, which is exactly where the sliced route pays for a second
   weight stream and the wide one does not.
4. **The floor is now an axis in `tools/batch_profile` on both shapes** (`--wide-min`, recorded in
   every sample as `wide_min`), so any of these questions is one panel and no rebuild.

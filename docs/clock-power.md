# The shader clock, and what each phase does when you move it

Every arithmetic budget in this repository is written against 2.9 GHz. Nobody had
measured the clock a payload actually ran at, and two engineers left that as the next
useful question: if a WMMA-dense pass runs several hundred megahertz low, the missing
time in the FFN is not missing.

It does not. The clock is real, the SMU reports it honestly, and at steady state a
saturating load holds 2841 to 2886 MHz. But the answer that came out of measuring it is
more useful than the question: **the shader clock is a controllable input, so you can
ask each phase what happens to it when the clock moves, and the phases disagree
violently.** A 128-row prefill pass is 89% clock-elastic at 2.0 GHz and 14% elastic at
2.9 GHz. The recurrent phase of a 32-stream generation step does not respond to the
clock at all.

## The instrument

`tools/gpu_clock.py` samples `sclk`, package power, temperature, busy percent and three
throttle-residency counters, and reports the busy-weighted clock a payload got.

```sh
python3 tools/gpu_clock.py show
tools/run-batch-compare --clock-log out.json --profile-tool ...
```

`tools/run-batch-compare` moves the clock as well as measuring it.

- `--clock-fix MHZ` pins the shader clock flat at MHZ for the life of the GPU lock.
- `--clock-sweep LIST` runs the payload once per clock point inside one lock hold,
  substituting `@CLK` in the payload arguments so each point writes its own output.
- `--pin-clock` forces the performance level to `high` without fixing a value.
- `--exec CMD ...` runs any probe under the same lock, mask and scope as a benchmark.

The entry performance level and overdrive table are restored in the same cleanup that
restores the resident service, before the lock is released.

`pp_dpm_sclk` level masks are rejected by this APU's SMU, and forcing the level to `low`
drags fclk and mclk down with sclk, which would make a sweep a measurement of the whole
memory path. The overdrive table moves the shader clock alone. Across every sweep below,
fclk stayed at 2000 MHz and mclk at 1000 MHz.

## Two controls, because a clock sweep is easy to fool

A sweep is worthless if the reported clock is not the executing clock. AMD parts stretch
clocks under voltage droop and the SMU keeps reporting the nominal value.

**Control one, a pure issue probe.** `rate_probe.hip` from the Kelana arithmetic notes is
register-resident with no memory traffic, so its time must scale 1:1 with the clock.
Measured at pinned 2010 and 2886 MHz, a clock ratio of 1.4358:

| waves | f16 ns @2010 | f16 ns @2886 | speedup | fraction of ideal |
|---:|---:|---:|---:|---:|
| 1 | 17.248 | 12.277 | 1.4049 | 97.8% |
| 2 | 34.304 | 24.036 | 1.4272 | 99.4% |
| 4 | 67.892 | 47.445 | 1.4309 | 99.7% |
| 8 | 135.533 | 94.258 | 1.4379 | 100.1% |

No stretching. The reported clock is the clock that executes.

That table also anchors a constant the budgets have been quoting without one. IU4 at
2886 MHz and eight waves is 5.949 ns per WMMA per SIMD32, against the 5.99325 ns in
[throughput-budgets.md](throughput-budgets.md). The budget's rate and its 2.9 GHz
assumption agree, which nobody had checked.

**Control two, a phase already known to be memory-bound.** The decode map puts
`gdn-resident-core` in a 32-stream step at 224 GB/s of a 242 GB/s roof. Under a 2.9x
clock change it moves by 1.2 ms out of 50, in the wrong direction. A memory-bound phase
reads as zero elasticity, so the instrument is not simply reporting "everything scales".

Residual FNV-64 is identical at every clock point in both shapes, `7446865760224376151`
for the 128-row prefill and `7661163660210692155` for the 32-stream step. Moving the
clock does not move a bit, which is what you would expect and worth having on record.

## A 128-row prefill pass saturates before it reaches 2.9 GHz

Mode 19, 128 rows, logits on, two rounds per point, reproducible to 0.2 ms. Clocks are
the measured busy-weighted medians, not the requested values.

| phase | 2010 MHz | 2287 | 2586 | 2886 | 2.0→2.3 | 2.3→2.6 | 2.6→2.9 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ffn` | 93.80 | 82.86 | 75.79 | 73.30 | 96% | 73% | **30%** |
| `sequence-input-projection` | 47.38 | 42.09 | 38.91 | 37.79 | 92% | 64% | 27% |
| `sequence-output-projection` | 19.05 | 16.90 | 15.98 | 16.84 | 93% | 46% | -48% |
| `gdn-resident-core` | 19.24 | 17.58 | 16.06 | 16.14 | 70% | 74% | -4% |
| `sequence-core` | 18.20 | 16.78 | 15.56 | 15.75 | 63% | 61% | -11% |
| `head-projection` | 13.21 | 11.72 | 10.64 | 10.06 | 93% | 79% | 51% |
| `sequence-input-prep` | 4.40 | 3.97 | 3.63 | 3.81 | 78% | 73% | -42% |
| **device span** | **217.81** | **194.28** | **178.76** | **175.95** | **89%** | **68%** | **14%** |

The last three columns are elasticities: the percentage of a perfectly clock-bound phase.
100% means time scales exactly with 1/clock. 0% means the clock does not matter.

The pass is compute-limited at 2.0 GHz and almost entirely something else at 2.9 GHz. The
knee is around 2.6 GHz. Above it, only the vocabulary head is still meaningfully
clock-elastic, and the FFN returns 30 cents on the dollar.

**This is the fact that matters for the work the lane is doing.** The FFN issue census is
real, the ISA slots are real, and near the operating clock they buy about a third of what
the arithmetic says. That is a sufficient explanation for measured nulls that have been
blamed on occupancy, scheduling and panel noise: freeing 19 VGPRs of operand addressing
moved nothing, the occupancy trade moved nothing, and both were priced against an issue
model that assumes instructions are what the clock is spending time on.

It also partly closes the older puzzle. `ffn` was reported at 85 ms against a 55.5 ms
issue model. Pinned at 2886 MHz it is 73.3 ms. Roughly 10 ms of that 30 ms gap was the
DPM governor, because the panels that produced 85 ms ran while the clock was still
climbing. The rest is not issue and not DRAM, and it does not respond to the clock.

## A generation step is a different machine, and the clock says so

Mode 19, 32 streams, one token each, pinned at 994 and 2848 MHz.

| phase | 994 MHz | 2848 MHz | clock-elastic |
|---|---:|---:|---:|
| `gdn-resident-core` | 49.62 | 50.84 | **-1%** |
| `ffn` | 69.54 | 38.19 | 44% |
| `sequence-input-projection` | 27.05 | 14.12 | 49% |
| `sequence-output-projection` | 15.99 | 11.64 | 20% |
| `head-projection` | 6.56 | 2.54 | 85% |
| `sequence-core` | 5.96 | 2.54 | 72% |
| **device span** | **180.48** | **122.64** | **25%** |

A 2.9x clock change leaves the recurrent phase exactly where it was. The handoff's
bounded negative on that phase was right, and this is a stronger form of it: not "the
arithmetic cannot beat the traffic" but "the arithmetic is not what it is waiting for".

The same kernel at 128-row prefill is 70 to 74% elastic in its lower range. One kernel,
two shapes, opposite binding resources. Measure which shape you are optimizing.

The `<=32-row` FFN arm that several engineers are cutting instructions from is 44%
elastic, so 21.4 of its 38.2 ms does not move with the clock.

## The deployed serving paths do not care about the clock

Both were measured cold, the way the resident service meets a request, with the clock
sampled throughout.

| workload | auto | pinned | clock change | throughput change |
|---|---:|---:|---:|---:|
| single stream, `--bench -n 100` | 33.82, 33.74 tok/s | 33.89, 33.49 | +9% | 0% |
| 718-token prefill, 8-row passes | 157.0 tok/s | 158.1 | +9% mean | +0.7% |

Both are memory-bound, which the documents already said and which now has a causal test
behind it rather than a bandwidth estimate. Pinning the clock on the serving GPU is not
a throughput lever. It would cost idle power for nothing.

## What a panel measures, and why round one lies

The DPM governor ramps from 600 MHz over several seconds. A cold payload starts around
1500 MHz and takes about three seconds to reach 2.7 GHz.

A `--rounds 1` profile panel, the shape most of this lane runs, finishes at a
busy-weighted median of 1935 MHz. Its timed work happens on a clock that is still moving.
A three-round `batch_compare` prefill panel does better, because warmup gets it there,
but the clock then droops under sustained load: during the busy window of one such run it
held a median of 2579 MHz with a 2282 to 2725 spread and a downward drift, against 2893
flat with `--clock-fix 2900`.

That drift is a plausible common cause for two things the lane has already written down:
device spans wandering 177 to 201 ms between rounds inside one run, and two back-to-back
processes moving a mode 0 control by 8%. An arm measured first is measured on a different
clock than an arm measured third.

Interleaving arms inside one process, which this lane already does, is the right defence
and remains the right defence. `--clock-fix 2900` removes the gradient instead of
averaging over it, and costs nothing: the full-model prompt medians were 705.6 tok/s on
auto and 710.9 pinned, well inside the panel.

## Reproduce

```sh
tools/run-batch-compare --clock-sweep 2000,2300,2600,2900 \
    --clock-log DATA/top-@CLK-clk.json \
    --profile-tool --out DATA/top-@CLK.json \
    --modes 19 --rows 128 --heads 1 --traces 1 --rounds 2 --warmup-ms 600

tools/run-batch-compare --clock-sweep 1000,2900 --exec /tmp/rate_probe
```

Raw samples, clock series and the probe table are in
`../../data/bonsai2/batch-comparison/clock-power/`.

## What I could not settle, and what settled it

Package power sits at a 108 to 110 W median with 120 W peaks at every sweep point,
including 2010 MHz, and `power1_average` is package tracking that includes the CPU cores.
The issue probe rules out clock stretching in a register-resident loop, which is the case
that matters for the elasticity numbers, but I did not separate how much of the top-end
saturation is the memory side and how much is a power ceiling the GPU shares with sixteen
CPU cores and whatever else the host is running. A VM was using a third of a core's worth
of CPU throughout.

**[`power-budget.md`](power-budget.md) settled that half.** The ceiling is real and it is
the package: the socket sustains 120 W, the shader array gets 52-56 W of it, and the part
is never thermally limited. The CPU that was using a third of a core here is the same
resource - eight host spinners take the delivered clock from 2755 to 2393 MHz and cost a
prompt pass 7.6%, and pinning the clock low with headroom under it removes two thirds of
that. The throttle counters this document's instrument was reading as `thm_core`,
`thm_gfx` and `thm_soc` were fppt, sppt and `thm_core`, so the evidence for a package
ceiling was already in the samples taken here, under thermal names.

The obvious next cut is to hold the clock at 2886 and vary something else. Nothing here
identifies *which* non-scaling resource the FFN is waiting on, only that it is not
instruction issue and not the shader clock.

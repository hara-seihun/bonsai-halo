# The socket is the shared resource, and no panel here was recording it

Every arithmetic budget in this repository is written against instructions and bytes. This
part has a third currency that decides both, and nothing in the lane was reading it: the
Radeon 8060S is the integrated GPU of an APU, and sixteen Zen 5 cores, the fabric and
LPDDR5X draw on one socket power budget with it. The SMU resolves contention for that
budget by moving the shader clock.

**Measured, same executable, same arm, same payload, `--pin-clock` in both columns:**

| | host quiet (3.7 cores busy) | + eight spinners (12.4 busy) | |
|---|---:|---:|---:|
| delivered shader clock | 2755 MHz | 2393 MHz | **-13.1%** |
| gfx power rail | 52.0 W | 33.4 W | **-36%** |
| CPU power rail | 7.9 W | 40.6 W | +414% |
| socket | 120.2 W | 120.7 W | 0 |
| prompt, 384 tokens in 128-row passes | 1048.6 tok/s | 968.9 tok/s | **-7.6%** |
| 32-stream aggregate generation | 330.0 tok/s | 320.3 tok/s | **-2.9%** |

That is not a benchmark of anything in the engine. It is what a panel loses to whatever
else is running on the machine, and it is larger than most results this lane has shipped.
Interleaving arms inside one process does not defend against it, because a peer's build
starts and stops on its own schedule inside the window. `27de3812` saw the same thing from
the other side on the same evening: their control column - phases their change cannot
reach - moved 6.57% between two arms shuffled inside one process.

## `--pin-clock` does not pin the clock

It writes `high` to `power_dpm_force_performance_level`, which sets the DPM floor and
ceiling to 2900 MHz. The part then delivers whatever the power budget leaves. Across one
evening of panels on this machine, all of them with `--pin-clock`, the delivered
busy-weighted median was 2190, 2393, 2728, 2755, 2782 and 2822 MHz.

The socket sustains **120 W**. Brief excursions to 135-140 W appear at the start of a load
and decay back. There is no writable cap: amdgpu exposes `power1_average` and
`power1_input` and no `power1_cap`, and this kernel has no `ryzen_smu`. The limit is
firmware's, so raising it is not an available lever and nobody should spend a turn on it.

**The part is never thermally limited.** Across every sample in every panel below, at
53-58 °C, the `thm_gfx` and `thm_soc` residency counters are exactly zero, and the package
counters run at 37% to 75% of the window. Whatever is holding the clock down, it is heat
in none of it.

## The clock is the mechanism, and pinning it low proves that

If host CPU load hurts a panel by taking socket power, then removing the SMU's freedom to
respond should remove most of the damage. `--clock-fix 2000` holds the shader clock flat
through the overdrive table, and at 2000 MHz the payload needs 24.7 W of gfx rail against
a 120 W socket, so there is headroom for the CPU to take.

| at `--clock-fix 2000` | quiet | + eight spinners | |
|---|---:|---:|---:|
| delivered clock | 2000 MHz | 2000 MHz | 0 |
| gfx rail | 24.7 W | 24.5 W | -1% |
| CPU rail | 26.5 W | 55.2 W | +108% |
| socket | 103.3 W | 120.2 W | +16% |
| prompt | 863.4 tok/s | 843.5 tok/s | **-2.3%** |
| 32-stream aggregate | 303.5 tok/s | 302.0 tok/s | **-0.5%** |

Same load, same machine, same arm. **Two thirds of what host CPU work costs a GPU panel
here travels through the shader clock**, and it disappears when the clock is pinned with
headroom under it. The 2.3% and 0.5% that remain are ordinary contention for memory and
scheduling.

The elasticities cross-check against an instrument that knew nothing about power.
[`clock-power.md`](clock-power.md) walked the clock directly and found a 128-row prefill
pass 68% clock-elastic in the 2.3-2.6 GHz band and a 32-stream generation step 25%
elastic. Here a -13.1% clock costs the prompt 7.6% (58% elastic) and the step 2.9% (22%
elastic). Two panels, two methods, the same two numbers.

## Where the socket actually goes

At 2.78-2.82 GHz on a quiet machine, running the full model: **gfx 52-56 W, CPU 8-10 W,
socket 120-125 W.** Roughly half the package is neither the shader array nor the cores -
it is the fabric, the memory controllers and LPDDR5X.

That is the part worth carrying into other work. A byte this engine does not read is not
only a byte of bandwidth; it is a share of ~55 W that the SMU can hand to the shader
clock instead, on a part that spends 37-75% of every panel window against its package
limit. Traffic cuts should therefore **over-deliver** against a pure bandwidth model, and
issue cuts should under-deliver against a pure issue model - which is the shape of every
null this lane has recorded on an instruction census.

`gpu_metrics` also carries time-filtered DRAM read and write counters, and they are not a
bandwidth instrument: they report 46.4 GB/s read for `bench/power_probe`'s stream, which
the probe itself measures at 177 GB/s over the same window. They are recorded in the
series because they move with the payload, not because a number can be read off them.

## What the instrument was reporting, and what it is now

`tools/gpu_clock.py` decoded three fields out of the binary `gpu_metrics` blob and called
them `thm_core`, `thm_gfx` and `thm_soc`. The blob is amdgpu's `gpu_metrics_v3_0`, whose
residency block is prochot, spl, fppt, sppt, thm_core, thm_gfx, thm_soc in that order, so
those three offsets were **fppt, sppt and thm_core**: package power limits under thermal
names. The counter this lane's only power-aware tool showed climbing hardest was the fast
package power tracker, and it was labelled as a temperature.

The structure is now read straight through, and five independent fields pin the layout
rather than one:

- offset 62 begins sixteen values that never leave 0..100, which is
  `average_core_c0_activity[16]`;
- offset 182 reads 2000 and offset 186 reads 1000 in every sample, the fclk and uclk this
  part holds;
- offset 190 begins sixteen CPU core frequencies around 4.6 GHz;
- offset 174 carries the same MHz value as `freq1_input` in every paired sample;
- offset 256 reads 1000000, the `time_filter_alphavalue` that closes the 264-byte record.

The empirical separation is just as clean: with the GPU idle and the host compiling,
`thm_core` runs at 98% residency and fppt/sppt at zero; with the GPU saturated and the
host quiet, fppt runs at 100% and `thm_core` does not move. One counter is the cores'
side of the package and the other is the GPU's.

The same layout makes the three power rails readable - `socket` at 112, `gfx` at 124,
`allcore` at 132 - which is every number in the tables above.

## Every panel records what machine it met

`tools/run-batch-compare` now samples the clock, the rails, the host CPU load and the
residency counters across every payload window, not only when `--clock-log` asks, and
prints one line beside the payload's own output:

```
gpu_clock: clock 2393 MHz p50 (83% of 2900) | socket 121 W | gfx 33 W / cpu 41 W
  | host 12.4 cores busy | package-power limited 65% of the window
  | NOT QUIET: another 11.4 cores of host work ran during this panel
```

`--clock-log PATH` still keeps the full series. The sampler is one 20 ms poll of sysfs, it
cannot fail a run, and it changes no payload.

**Two things follow for anyone measuring here.** Put your compiles outside a peer's lock
hold - a twelve-way build costs the panel next to it 7.6% of a prompt pass. And when a
result is worth less than 8%, take it at `--clock-fix 2000`, where the machine is slower
and the arms are comparable, rather than at `--pin-clock`, where they are not.

## The host thread spins a whole core, and it is worth nothing (bounded negative)

ROCm's default wait is active. This engine synchronises per pass - a 32-stream step is
about 48 launches - so the calling thread polls for as long as the GPU is busy, and
`/proc/PID/stat` shows a fully busy core through every millisecond of every phase. On a
part whose shader clock is decided by a shared socket budget, that reads like an obvious
few watts to hand back.

It is not. `hipDeviceScheduleBlockingSync` blocks on the completion interrupt instead, and
`bench/power_probe` confirms the mechanism does what it says: **host CPU goes from 100.0%
and 99.8% of a core to 8.3% and 7.6%**, each arm run forward and reverse.

| probe arm | pass | host CPU | gfxclk | gfx W | CPU W | socket W | GB/s |
|---|---|---:|---:|---:|---:|---:|---:|
| spin | fwd | 100.0% | 2676 | 13.6 | 36.5 | 94.8 | 177.6 |
| block | fwd | 8.3% | 2896 | 16.0 | 38.1 | 100.1 | 175.5 |
| block | rev | 7.6% | 2900 | 15.4 | 37.8 | 100.2 | 175.4 |
| spin | rev | 99.8% | 2900 | 15.5 | 37.9 | 100.8 | 178.4 |

And on the full model, back to back on a quiet machine, 32-stream aggregate generation is
**330.75 tok/s spinning against 331.05 blocking, +0.09%**, with the prompt inside its own
round-to-round spread. The CPU rail moves the wrong way: 7.6-7.9 W spinning, 10.3-11.2 W
blocking.

**A busy core is not a watt.** A poll loop parked on a completion signal reads as 100%
occupancy and costs 0 to 2 W of a 120 W socket, where a compiler thread on the same core
costs 6 to 7; and an interrupt-driven wait buys that back in scheduler wakeups. The arm is
out of the runtime and kept as `host-sync-arm.patch` beside the raw samples. `src/engine.cpp`
carries the measurement as a comment so the next engineer does not re-derive it.

This also corrects a reading of the host-load result above: the damage a busy machine does
to a panel is done by CPU work, not by the engine's own waiting thread.

## Reproduce

```sh
tools/run-batch-compare --pin-clock --exec ./bench/power_probe --seconds 3 --arms spin,block

tools/run-batch-compare --pin-clock --clock-log DATA/quiet-clk.json --tag power-budget/quiet \
    --modes 19 --streams 32 --slots 32 --context 512 --only prefill,decode \
    --prefill-rows 128 --prefill-tokens 384 --gen-steps 16 --rounds 3 --warmup-steps 3

# the loaded arm is the same command with eight host spinners alongside, and the
# clock-pinned pair replaces --pin-clock with --clock-fix 2000
```

Raw samples, the eight clock/power series and the retired arm are in
[`batch-comparison/power-budget/`](../../../data/bonsai2/batch-comparison/power-budget/README.md).
The tail-logits FNV-64 is `15415681694910198979` in **every** arm of every panel here -
spin and block, quiet and loaded, 2000 MHz and 2900. Neither the clock nor the host wait
moves an output bit.

## What this does not settle

Nothing here says how much of the ~55 W that is neither gfx nor cores is LPDDR5X and how
much is fabric and uncore, so the watts-per-byte coefficient that would let a traffic cut
be predicted in advance is not measured - only its sign. The cheapest way to get it is a
payload whose byte rate can be varied at a fixed instruction count, which `bench/wstream`
already is: sweep its stream count at `--clock-fix 2000` and read the gfx and socket rails
against achieved GB/s.

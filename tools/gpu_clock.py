#!/usr/bin/env python3
"""Sample the gfx clock, package power and throttle residency of the iGPU.

Every arithmetic budget in this repository is written against a 2.9 GHz shader
clock. gfx1151 is the integrated GPU of an APU that shares one package power
budget with sixteen CPU cores, and its DPM governor ramps over seconds. This
tool reports what the clock actually was while a payload ran, so a budget or an
A/B panel can be read against the clock that produced it.

    tools/gpu_clock.py show
    tools/gpu_clock.py sample --duration 5
    tools/gpu_clock.py run --json out.json -- tools/batch_profile --modes 19 ...
    tools/gpu_clock.py force high      # needs root; returns the prior value
    tools/gpu_clock.py force auto

Field provenance. `sclk` and `power` come from the amdgpu hwmon attributes
`freq1_input` and `power1_average`, which the driver documents. Everything else
is decoded from the binary `gpu_metrics` blob, which this SMU publishes as
format 3, content 0, 264 bytes - amdgpu's `struct gpu_metrics_v3_0`. The layout
below is that structure read straight through, and five independent fields in
the record confirm it rather than one:

  * offset 62..93 is sixteen values that never leave 0..100, which is
    `average_core_c0_activity[16]` and nothing else in the record can be;
  * offset 182 reads 2000 and offset 186 reads 1000 in every sample, the fclk
    and uclk this part is documented to hold;
  * offset 190..221 is sixteen CPU core frequencies around 4.6 GHz;
  * offset 174 carries the same MHz value as `freq1_input` in every paired
    sample;
  * offset 256 reads 1000000, the `time_filter_alphavalue` that closes the
    structure.

The power rails matter as much as the clock here and were unread before
2026-09-22. This is an APU: sixteen Zen 5 cores, the Radeon 8060S, the fabric
and LPDDR5X draw on one socket budget, and the SMU resolves contention for it
by moving the shader clock. `socket` is the number `power1_average` reports,
`gfx` and `allcore` are the two rails that compete inside it.

The seven throttle-residency counters were previously read as three, labelled
`thm_core`/`thm_gfx`/`thm_soc`, which put package-power throttling under a
thermal name. In `gpu_metrics_v3_0` the residency block is prochot, spl, fppt,
sppt, thm_core, thm_gfx, thm_soc in that order, so those three offsets are
fppt, sppt and thm_core. On this machine under load fppt sits at 100% residency
and the two thermal counters do not move at all. Counters tick at 1 kHz, so a
delta divided by the window in milliseconds is a residency fraction.
"""

import argparse
import json
import os
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

DRM = Path('/sys/class/drm')
# gpu_metrics v3.0. See the module docstring for what pins each of these.
OFF_GFXCLK = 174
OFF_FCLK = 182
OFF_UCLK = 186
OFF_CUR_GFXCLK = 224
OFF_DRAM_READ = 94
OFF_DRAM_WRITE = 96
# Power rails, mW. `socket` is what hwmon reports; `gfx` and `allcore` are the
# two consumers that compete for it. `ipu` and `dgpu` are unpopulated on this
# part and are not read.
RAILS = {'socket_w': 112, 'apu_w': 120, 'gfx_w': 124, 'allcore_w': 132}
# Throttle residency, ticking at 1 kHz, in the order gpu_metrics_v3_0 declares.
THROTTLE = {'prochot': 228, 'spl': 232, 'fppt': 236, 'sppt': 240,
            'thm_core': 244, 'thm_gfx': 248, 'thm_soc': 252}
# Package-power limits, as distinct from thermal ones: these are the counters
# that decide the shader clock on this machine.
POWER_THROTTLE = ('prochot', 'spl', 'fppt', 'sppt')


def find_card():
    for card in sorted(DRM.glob('card*')):
        dev = card / 'device'
        if (dev / 'gpu_busy_percent').exists():
            return dev
    raise SystemExit('gpu_clock: no amdgpu card with gpu_busy_percent')


class Probe:
    def __init__(self):
        self.dev = find_card()
        hw = sorted((self.dev / 'hwmon').glob('hwmon*'))
        if not hw:
            raise SystemExit('gpu_clock: no hwmon under %s' % self.dev)
        self.hwmon = hw[0]
        self.files = {}
        self.metrics_path = self.dev / 'gpu_metrics'

    def _open(self, path):
        fh = self.files.get(path)
        if fh is None:
            fh = self.files[path] = open(path, 'rb')
        fh.seek(0)
        return fh

    def _int(self, path, default=None):
        try:
            return int(self._open(path).read().strip())
        except (OSError, ValueError):
            return default

    def metrics(self):
        try:
            with open(self.metrics_path, 'rb') as fh:
                return fh.read()
        except OSError:
            return b''

    def host_cpu(self):
        """Busy fraction of the whole host CPU since the previous call, 0..1 per core-sum.

        A panel that does not record this cannot tell a slow arm from a quiet
        machine: a peer's compile moves the shader clock by several hundred MHz.
        """
        try:
            f = self._open(Path('/proc/stat')).read().split(b'\n', 1)[0].split()
        except OSError:
            return None
        v = [int(x) for x in f[1:]]
        idle = v[3] + (v[4] if len(v) > 4 else 0)
        total = sum(v)
        prev = getattr(self, '_cpu_prev', None)
        self._cpu_prev = (total, idle)
        if prev is None or total <= prev[0]:
            return None
        return 1.0 - (idle - prev[1]) / (total - prev[0])

    def sample(self):
        s = {
            'sclk_mhz': (self._int(self.hwmon / 'freq1_input', 0) or 0) // 1_000_000,
            'power_w': (self._int(self.hwmon / 'power1_average', 0) or 0) / 1e6,
            'temp_c': (self._int(self.hwmon / 'temp1_input', 0) or 0) / 1e3,
            'busy_pct': self._int(self.dev / 'gpu_busy_percent', 0),
        }
        cpu = self.host_cpu()
        if cpu is not None:
            s['host_cpu_frac'] = round(cpu, 4)
        blob = self.metrics()
        if len(blob) >= 264:
            s['metrics_gfxclk_mhz'] = struct.unpack_from('<H', blob, OFF_GFXCLK)[0]
            s['cur_gfxclk_mhz'] = struct.unpack_from('<H', blob, OFF_CUR_GFXCLK)[0]
            s['fclk_mhz'] = struct.unpack_from('<H', blob, OFF_FCLK)[0]
            s['uclk_mhz'] = struct.unpack_from('<H', blob, OFF_UCLK)[0]
            s['dram_read_mbs'] = struct.unpack_from('<H', blob, OFF_DRAM_READ)[0]
            s['dram_write_mbs'] = struct.unpack_from('<H', blob, OFF_DRAM_WRITE)[0]
            for name, off in RAILS.items():
                w = struct.unpack_from('<I', blob, off)[0] / 1000.0
                # Unpopulated rails read as 0xffffffff-ish; drop them rather than
                # report four megawatts.
                if w < 1000.0:
                    s[name] = round(w, 2)
            for name, off in THROTTLE.items():
                s[name] = struct.unpack_from('<I', blob, off)[0]
        return s

    def force_level(self, level=None):
        """Read, or set and return the prior, power_dpm_force_performance_level."""
        path = self.dev / 'power_dpm_force_performance_level'
        prior = path.read_text().strip()
        if level is not None and level != prior:
            try:
                path.write_text(level)
            except PermissionError:
                subprocess.run(['sudo', '-n', 'tee', str(path)], input=level.encode(),
                               stdout=subprocess.DEVNULL, check=True)
        return prior

    def dpm_table(self):
        try:
            return (self.dev / 'pp_dpm_sclk').read_text().strip().splitlines()
        except OSError:
            return []


def summarize(samples, busy_floor):
    hot = [s for s in samples if s['busy_pct'] >= busy_floor] or samples
    if not samples:
        return {}
    clk = sorted(s['sclk_mhz'] for s in hot)
    pw = sorted(s['power_w'] for s in hot)

    def pct(xs, q):
        if not xs:
            return 0.0
        return xs[min(len(xs) - 1, int(q * (len(xs) - 1)))]

    out = {
        'samples': len(samples),
        'busy_samples': len(hot),
        'busy_floor_pct': busy_floor,
        'sclk_min': clk[0], 'sclk_p50': pct(clk, 0.5), 'sclk_p90': pct(clk, 0.9),
        'sclk_max': clk[-1], 'sclk_mean': round(sum(clk) / len(clk), 1),
        'power_p50_w': round(pct(pw, 0.5), 1), 'power_max_w': round(pw[-1], 1),
        'temp_max_c': round(max(s['temp_c'] for s in samples), 1),
        'busy_p50_pct': sorted(s['busy_pct'] for s in samples)[len(samples) // 2],
        # Every budget in docs/ is stated at 2.9 GHz; this is the correction factor.
        'clock_ratio_vs_2900': round(pct(clk, 0.5) / 2900.0, 4),
    }
    # The rails that compete for the socket, and the host load that took one side
    # of it. A panel without these cannot say whether its arms met the same machine.
    for name in ('gfx_w', 'allcore_w', 'socket_w'):
        vals = sorted(s[name] for s in hot if name in s)
        if vals:
            out[name + '_p50'] = round(pct(vals, 0.5), 1)
            out[name + '_max'] = round(vals[-1], 1)
    cpu = sorted(s['host_cpu_frac'] for s in hot if 'host_cpu_frac' in s)
    if cpu:
        out['host_cpu_p50'] = round(pct(cpu, 0.5), 3)
        out['host_cpu_max'] = round(cpu[-1], 3)
        ncpu = os.cpu_count() or 1
        out['host_cores_busy_p50'] = round(pct(cpu, 0.5) * ncpu, 2)
    first, last = samples[0], samples[-1]
    span_ms = (last['t'] - first['t']) * 1000.0 if len(samples) > 1 else 0.0
    for name in THROTTLE:
        if name in first and name in last:
            out['d_' + name] = last[name] - first[name]
            # Counters tick at 1 kHz, so a delta over the window in milliseconds
            # is the fraction of the window spent in that limit.
            if span_ms > 0:
                out['res_' + name] = round(min(1.0, (last[name] - first[name]) / span_ms), 3)
    out['power_limited_frac'] = round(max((out.get('res_' + n, 0.0) for n in POWER_THROTTLE), default=0.0), 3)
    # GPU-side thermal only. `thm_core` is a CPU-core counter: it runs at ~98%
    # residency with the GPU idle and the host compiling, and at zero with the
    # GPU saturated and the host quiet, which is the observation that separates
    # the two sides of this package and pins the order of the residency block.
    out['thermal_limited_frac'] = round(max((out.get('res_' + n, 0.0) for n in ('thm_gfx', 'thm_soc')), default=0.0), 3)
    out['host_core_limited_frac'] = out.get('res_thm_core', 0.0)
    # Time from window start until the clock first reaches 95% of its own max.
    target = 0.95 * clk[-1]
    t0 = samples[0]['t']
    out['ramp_to_95pct_s'] = None
    for s in samples:
        if s['sclk_mhz'] >= target:
            out['ramp_to_95pct_s'] = round(s['t'] - t0, 3)
            break
    return out


def collect(probe, interval, stop_fn, series):
    while not stop_fn():
        s = probe.sample()
        s['t'] = time.monotonic()
        series.append(s)
        time.sleep(interval)


def cmd_show(args):
    p = Probe()
    s = p.sample()
    s['force_level'] = p.force_level()
    s['dpm_sclk'] = p.dpm_table()
    print(json.dumps(s, indent=2))


def cmd_sample(args):
    """Sample until the duration elapses or a terminating signal arrives.

    The wrapper starts this beside a payload of unknown length and stops it with
    SIGTERM, so the summary has to survive the signal rather than depend on the
    duration being guessed correctly.
    """
    p = Probe()
    series = []
    stopping = []

    def on_signal(signum, frame):
        stopping.append(signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, on_signal)
    end = time.monotonic() + args.duration
    collect(p, args.interval_ms / 1000.0,
            lambda: stopping or time.monotonic() >= end, series)
    emit(args, series, {'stopped_by_signal': stopping[0] if stopping else None})


def cmd_run(args):
    p = Probe()
    series = []
    t0 = time.monotonic()
    proc = subprocess.Popen(args.command)
    interval = args.interval_ms / 1000.0
    while proc.poll() is None:
        s = p.sample()
        s['t'] = time.monotonic()
        series.append(s)
        time.sleep(interval)
    rc = proc.wait()
    meta = {'command': args.command, 'returncode': rc,
            'wall_s': round(time.monotonic() - t0, 3)}
    emit(args, series, meta)
    return rc


def verdict(summary):
    """One line saying whether this panel met a quiet machine, in plain terms.

    A panel measured beside a peer's twelve-way compile runs several hundred MHz
    low, and nothing in a run.json says so. This does.
    """
    if not summary or 'sclk_p50' not in summary:
        return 'clock: no samples'
    clk = summary['sclk_p50']
    bits = ['clock %d MHz p50 (%.0f%% of 2900)' % (clk, 100.0 * clk / 2900.0)]
    if 'socket_w_p50' in summary:
        bits.append('socket %.0f W' % summary['socket_w_p50'])
    if 'gfx_w_p50' in summary and 'allcore_w_p50' in summary:
        bits.append('gfx %.0f W / cpu %.0f W' % (summary['gfx_w_p50'], summary['allcore_w_p50']))
    if 'host_cores_busy_p50' in summary:
        bits.append('host %.1f cores busy' % summary['host_cores_busy_p50'])
    if summary.get('power_limited_frac', 0) > 0.05:
        bits.append('package-power limited %.0f%% of the window' % (100 * summary['power_limited_frac']))
    if summary.get('thermal_limited_frac', 0) > 0.05:
        bits.append('GPU THERMALLY limited %.0f%%' % (100 * summary['thermal_limited_frac']))
    if summary.get('host_core_limited_frac', 0) > 0.5:
        bits.append('host cores at their own limit %.0f%% of the window' % (100 * summary['host_core_limited_frac']))
    # One busy core beyond the payload's own is the level at which the shader
    # clock moves measurably; two is the level at which a panel is not comparable
    # with one taken on a quiet machine.
    cores = summary.get('host_cores_busy_p50')
    if cores is not None and cores >= 2.0:
        bits.append('NOT QUIET: another %0.1f cores of host work ran during this panel' % (cores - 1.0))
    return ' | '.join(bits)


def emit(args, series, meta):
    p = Probe()
    summary = summarize(series, args.busy_floor)
    summary['verdict'] = verdict(summary)
    doc = {'summary': summary, 'meta': meta or {},
           'force_level': p.force_level(), 'dpm_sclk': p.dpm_table(),
           'host': os.uname().nodename, 'when': time.strftime('%Y-%m-%dT%H:%M:%S')}
    print('gpu_clock: ' + summary['verdict'], file=sys.stderr)
    t0 = series[0]['t'] if series else 0
    for s in series:
        s['t'] = round(s['t'] - t0, 4)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(dict(doc, series=series), indent=1))
    # The series goes to the file; stdout stays readable beside a payload's own output.
    print(json.dumps(doc, indent=2))


def cmd_force(args):
    p = Probe()
    prior = p.force_level(args.level)
    print(json.dumps({'prior': prior, 'now': p.force_level(),
                      'dpm_sclk': p.dpm_table()}, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--interval-ms', type=float, default=20.0)
    ap.add_argument('--busy-floor', type=int, default=50,
                    help='summarize only samples at or above this gpu_busy_percent')
    ap.add_argument('--json', help='write the full series here')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('show').set_defaults(fn=cmd_show)
    s = sub.add_parser('sample')
    s.add_argument('--duration', type=float, default=5.0)
    s.set_defaults(fn=cmd_sample)
    r = sub.add_parser('run')
    r.add_argument('command', nargs=argparse.REMAINDER)
    r.set_defaults(fn=cmd_run)
    f = sub.add_parser('force')
    f.add_argument('level', choices=['auto', 'high', 'low'])
    f.set_defaults(fn=cmd_force)
    args = ap.parse_args()
    if args.cmd == 'run':
        if args.command and args.command[0] == '--':
            args.command = args.command[1:]
        if not args.command:
            ap.error('run needs a command after --')
    sys.exit(args.fn(args) or 0)


if __name__ == '__main__':
    main()

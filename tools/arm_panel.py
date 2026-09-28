#!/usr/bin/env python3
"""Compare the phases of one axis's arms in a tools/batch_profile run.

Reads the per-launch trace of every sample, sums it by phase, groups samples by the axis
value they ran at, and prints each phase's arm ratio against the phases the change cannot
reach. That control column is what separates a result from this box's clock drift: a panel
without it read a 4% regression as a win once already (docs/long-context-attention.md).

    tools/arm_panel.py RUN.json [--key mvw_order] [--phase ffn]
"""
import json
import sys
from collections import defaultdict

path = sys.argv[1]
key = sys.argv[sys.argv.index("--key") + 1] if "--key" in sys.argv else "mvw_order"
target = sys.argv[sys.argv.index("--phase") + 1] if "--phase" in sys.argv else "ffn"
doc = json.load(open(path))
arms = defaultdict(list)
for s in doc["samples"]:
    by = defaultdict(float)
    for e in s.get("trace") or []:
        by[e["kind"]] += e["end_ms"] - e["start_ms"]
    arms[(s.get(key), s.get("rows"), s.get("mode"))].append(
        (by, s.get("device_span_ms", 0.0), s.get("wall_ms", 0.0), s.get("residual_fnv64")))

kinds = sorted({k for v in arms.values() for by, *_ in v for k in by})
print(f"{'arm':>5} {'rows':>5} {'mode':>5} {'n':>3} " + " ".join(f"{k[:20]:>20}" for k in kinds)
      + f" {'control':>10} {'span':>9} {'wall':>9}")
summary = {}
for akey in sorted(arms, key=lambda x: (x[1], x[2], str(x[0]))):
    v = arms[akey]
    n = len(v)
    mean = {k: sum(by.get(k, 0.0) for by, *_ in v) / n for k in kinds}
    ctl = sum(mean[k] for k in kinds if k != target)
    span = sum(x[1] for x in v) / n
    wall = sum(x[2] for x in v) / n
    print(f"{str(akey[0]):>5} {akey[1]:>5} {akey[2]:>5} {n:>3} " + " ".join(f"{mean[k]:>20.3f}" for k in kinds)
          + f" {ctl:>10.3f} {span:>9.3f} {wall:>9.3f}")
    summary[akey] = (mean.get(target, 0.0), ctl, span, wall, {x[3] for x in v})

base = [k for k in summary if str(k[0]) in ("0", "-1")]
for akey, (ph, ctl, span, wall, hashes) in sorted(summary.items(), key=lambda kv: (kv[0][1], str(kv[0][0]))):
    ref = [b for b in base if b[1] == akey[1] and b[2] == akey[2]]
    if not ref or ref[0] == akey:
        continue
    p0, c0, s0, w0, _ = summary[ref[0]]
    print(f"\n{target} at {akey[1]} rows mode {akey[2]}: arm {akey[0]} {ph:.3f} against {p0:.3f} ms "
          f"= {100*(ph/p0-1):+.2f}%, control {100*(ctl/c0-1):+.2f}%, "
          f"normalised {100*((ph/ctl)/(p0/c0)-1):+.2f}%, span {100*(span/s0-1):+.2f}%, wall {100*(wall/w0-1):+.2f}%")
allh = {h for v in summary.values() for h in v[4]}
print(f"\nresidual FNV-64 over all arms: {' '.join(sorted(allh))}")

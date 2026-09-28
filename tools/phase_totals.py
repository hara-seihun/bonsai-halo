#!/usr/bin/env python3
"""Per-phase totals of a tools/batch_profile sample, by launch kind.

Every launch is one `trace` entry. A phase costs the sum of its launches, and the launches a
change cannot reach are its in-process control.

    tools/phase_totals.py RUN.json [RUN2.json ...]
"""
import json
import sys
from collections import defaultdict

for path in sys.argv[1:]:
    doc = json.load(open(path))
    print(f"== {path}  rev {doc.get('git_revision','')[:12]}")
    for s in doc["samples"]:
        span = s.get("device_span_ms", 0.0)
        print(f"   mode {s.get('mode')} rows {s.get('rows')} logits {s.get('logits')} round {s.get('round')} "
              f"span {span:.3f} wall {s.get('wall_ms',0):.3f} residual {s.get('residual_fnv64')}")
        by = defaultdict(lambda: [0.0, 0, 0])
        for e in s.get("trace") or []:
            t = by[e["kind"]]
            t[0] += e["end_ms"] - e["start_ms"]
            t[1] += 1
            t[2] = e.get("rows", 0)
        for k, (ms, n, rows) in sorted(by.items(), key=lambda kv: -kv[1][0]):
            print(f"      {k:<30} {ms:9.3f} ms {100*ms/span if span else 0:5.1f}%  x{n:<5d} rows {rows}")

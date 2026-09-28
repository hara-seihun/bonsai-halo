#!/usr/bin/env python3
"""Pair a tools/batch_profile panel by (rows, arm) and normalise by the phases an arm cannot reach.

A 128-row prefill pass is 89% clock-elastic (docs/clock-power.md) and arms minutes apart read that
elasticity as a result. Every phase here is divided by the geometric mean of the control phases in
the same arm, so what survives is what the change did.

    tools/seq_out_a4_panel.py PANEL.json --arms a8,a4e-out,a4-out
    tools/seq_out_a4_panel.py PANEL.json --key seq_sched --arms 1,5 --controls ffn,gdn-resident-core
"""
import argparse, json, math, statistics
from collections import defaultdict

DEFAULT_CONTROLS = "ffn,gdn-resident-core,sequence-core,head-projection"

def phases(s):
    by = defaultdict(float)
    for e in s.get("trace") or []:
        by[e["kind"]] += e["end_ms"] - e["start_ms"]
    return dict(by)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panel")
    ap.add_argument("--arms", default="")
    ap.add_argument("--key", default="seq_quant")
    ap.add_argument("--controls", default=DEFAULT_CONTROLS)
    ap.add_argument("--skip-rounds", type=int, default=1)
    a = ap.parse_args()
    doc = json.load(open(a.panel))
    ss = [s for s in doc["samples"] if s.get("round", 0) >= a.skip_rounds and s.get("trace")]
    controls = [c for c in a.controls.split(",") if c]
    arms = [x for x in a.arms.split(",") if x] or sorted({str(s.get(a.key)) for s in ss})
    groups = defaultdict(list)
    for s in ss:
        groups[(s["rows"], str(s.get(a.key)))].append(s)
    print(f"{a.panel}  binary {doc.get('binary_sha256','')[:16]}  rev {doc.get('git_revision','')[:12]}"
          f"  arms on {a.key}  controls {','.join(controls)}")
    for rows in sorted({r for r, _ in groups}):
        base = groups.get((rows, arms[0]))
        if not base: continue
        print(f"\n== rows {rows} ==  ({len(base)} samples per arm)")
        names = sorted({n for s in base for n in phases(s)},
                       key=lambda n: -statistics.mean([phases(s).get(n, 0.0) for s in base]))
        mean = lambda g, n: statistics.mean([phases(s).get(n, float('nan')) for s in g])
        factor = {}
        for arm in arms:
            g = groups.get((rows, arm))
            if not g: continue
            rs = [mean(g, c) / mean(base, c) for c in controls if mean(base, c) > 0]
            factor[arm] = math.exp(sum(map(math.log, rs)) / len(rs)) if rs else 1.0
        print(f"{'phase':<28}" + "".join(f"{x:>13}" for x in arms) + "".join(f"{'norm ' + x:>13}" for x in arms[1:]))
        for n in names:
            vals = [mean(groups[(rows, arm)], n) if (rows, arm) in groups else float('nan') for arm in arms]
            cor = [(vals[i] / factor.get(arms[i], 1.0) / vals[0] - 1) * 100 for i in range(1, len(arms))]
            print(f"{n:<28}" + "".join(f"{v:>13.3f}" for v in vals) + "".join(f"{c:>12.2f}%" for c in cor))
        span = [statistics.mean([s["device_span_ms"] for s in groups[(rows, arm)]]) if (rows, arm) in groups
                else float('nan') for arm in arms]
        print(f"{'device span':<28}" + "".join(f"{v:>13.3f}" for v in span)
              + "".join(f"{(span[i] / span[0] - 1) * 100:>12.2f}%" for i in range(1, len(arms))))
        print(f"{'control factor':<28}" + "".join(f"{factor.get(x, float('nan')):>13.4f}" for x in arms))
        for arm in arms:
            g = groups.get((rows, arm)) or []
            print(f"   {arm:<10} residual {sorted({str(s.get('residual_fnv64')) for s in g})}")

if __name__ == "__main__":
    main()

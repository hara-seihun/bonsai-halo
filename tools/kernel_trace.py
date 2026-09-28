#!/usr/bin/env python3
"""Read a rocprofv3 kernel trace and report where one pass or one step went.

`tools/run-batch-compare --profile-tool` with `BONSAI_PROFILE_TRACE=1` writes
`DIR/HOST/PID_kernel_trace.csv`: one row per dispatch, with the kernel's mangled name, its
begin and end timestamps in nanoseconds, its grid and its register counts. The engine's own
phase brackets group launches by the phase name the host asked for; this groups them by the
kernel instantiation the device actually ran, which is what tells a gate/up stage apart from
a down stage, a 32-row arm from a 128-row arm, and issue time apart from the gaps between
dispatches.

A run contains model load, warmup passes and several measured passes. Dispatches are
segmented on host gaps: a sync between two measured steps shows up as a gap far larger than
any gap inside a step. `--segments` lists them, `--segment N` reports one (negative counts
from the end, so -1 is the last measured step), `--kernels` breaks it down by kernel.

    tools/kernel_trace.py DIR --segments
    tools/kernel_trace.py DIR --segment -1 --kernels
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


def find_csv(path: Path) -> Path:
    if path.is_file():
        return path
    hits = sorted(path.rglob("*_kernel_trace.csv"))
    if not hits:
        raise SystemExit(f"no *_kernel_trace.csv under {path}")
    return hits[-1]


def short_name(name: str) -> str:
    """The instantiation, short enough to read: strip the ABI wrapper and the argument list."""
    n = re.sub(r"^void\s+", "", name)
    n = n.replace("(anonymous namespace)::", "").replace("halo::", "")
    depth = 0
    for i, ch in enumerate(n):          # cut the argument list, keeping template arguments
        if ch in "<":
            depth += 1
        elif ch in ">":
            depth -= 1
        elif ch == "(" and depth == 0:
            return n[:i]
    return n


def load(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for r in csv.DictReader(fh):
            if r["Kind"] != "KERNEL_DISPATCH":
                continue
            rows.append({
                "name": short_name(r["Kernel_Name"]),
                "start": int(r["Start_Timestamp"]),
                "end": int(r["End_Timestamp"]),
                "vgpr": int(r["VGPR_Count"]),
                "sgpr": int(r["SGPR_Count"]),
                "lds": int(r["LDS_Block_Size"]),
                "scratch": int(r["Scratch_Size"]),
                "wg": int(r["Workgroup_Size_X"]),
                "grid": (int(r["Grid_Size_X"]), int(r["Grid_Size_Y"]), int(r["Grid_Size_Z"])),
            })
    rows.sort(key=lambda d: d["start"])
    return rows


def segment(rows: list[dict], gap_ms: float) -> list[list[dict]]:
    out: list[list[dict]] = []
    cur: list[dict] = []
    gap = gap_ms * 1e6
    for r in rows:
        if cur and r["start"] - cur[-1]["end"] > gap:
            out.append(cur)
            cur = []
        cur.append(r)
    if cur:
        out.append(cur)
    return out


def span_ms(seg: list[dict]) -> float:
    return (seg[-1]["end"] - seg[0]["start"]) / 1e6


def busy_ms(seg: list[dict]) -> float:
    """Device time covered by at least one dispatch, so back-to-back kernels are not double counted."""
    total = 0
    cur_s, cur_e = seg[0]["start"], seg[0]["end"]
    for r in seg[1:]:
        if r["start"] > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = r["start"], r["end"]
        else:
            cur_e = max(cur_e, r["end"])
    total += cur_e - cur_s
    return total / 1e6


def by_kernel(seg: list[dict]) -> list[dict]:
    agg: dict[str, dict] = {}
    for r in seg:
        a = agg.setdefault(r["name"], {"name": r["name"], "n": 0, "ms": 0.0, "vgpr": r["vgpr"],
                                       "scratch": r["scratch"], "grid": r["grid"], "wg": r["wg"]})
        a["n"] += 1
        a["ms"] += (r["end"] - r["start"]) / 1e6
    return sorted(agg.values(), key=lambda a: -a["ms"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="trace directory or kernel_trace.csv")
    ap.add_argument("--gap-ms", type=float, default=2.0, help="host gap that separates two segments")
    ap.add_argument("--segments", action="store_true", help="list segments")
    ap.add_argument("--segment", type=int, help="report one segment (negative counts from the end)")
    ap.add_argument("--kernels", action="store_true", help="break the segment down by kernel")
    ap.add_argument("--min-ms", type=float, default=0.05, help="hide kernels below this total")
    ap.add_argument("--json", type=Path, help="write the segment breakdown here")
    args = ap.parse_args()

    csv_path = find_csv(args.path)
    rows = load(csv_path)
    segs = segment(rows, args.gap_ms)
    print(f"{csv_path}: {len(rows)} dispatches, {len(segs)} segments")

    if args.segments or args.segment is None:
        for i, s in enumerate(segs):
            print(f"  [{i:3d}] {len(s):5d} dispatches  span {span_ms(s):9.3f} ms  busy {busy_ms(s):9.3f} ms  "
                  f"first {s[0]['name'][:44]}")
        if args.segment is None:
            return 0

    seg = segs[args.segment]
    span, busy = span_ms(seg), busy_ms(seg)
    print(f"\nsegment {args.segment}: {len(seg)} dispatches, span {span:.3f} ms, busy {busy:.3f} ms, "
          f"idle {span - busy:.3f} ms ({100 * (span - busy) / span:.1f}%)")
    table = by_kernel(seg)
    if args.kernels:
        print(f"{'kernel':<72}{'n':>5}{'ms':>10}{'share':>8}{'us/call':>9}{'vgpr':>6}{'grid':>18}")
        for a in table:
            if a["ms"] < args.min_ms:
                continue
            g = "x".join(str(v) for v in a["grid"])
            print(f"{a['name'][:70]:<72}{a['n']:>5}{a['ms']:>10.3f}{100 * a['ms'] / span:>7.1f}%"
                  f"{1000 * a['ms'] / a['n']:>9.1f}{a['vgpr']:>6}{g:>18}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"csv": str(csv_path), "segment": args.segment,
                                         "span_ms": span, "busy_ms": busy, "kernels": table}, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

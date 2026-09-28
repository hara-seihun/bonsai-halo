#!/usr/bin/env python3
"""Scalar-wait census of the multi-row ternary matvec body, over the row-batch and depth axes.

A TT = 8 block streams the same 896 weight bytes a TT = 1 block does and pays sixteen
`s_waitcnt lgkmcnt(0)` against its one. That counter is all-or-nothing on gfx11, so every wait is a
full scalar round trip on the wave's critical path, and this is the number the eight-row pass's
bandwidth follows. This prices an arm in eight seconds with no GPU: waits, scalar loads, dot
products and issue slots in the block loop, plus registers, spills and resident waves.

    tools/mv_sched_census.py                     # the deployed schedule against the row batch
    tools/mv_sched_census.py --rb 1,2,4 --tt 8   # the batch width axis
"""
import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LLAMA = Path("../bonsai-hip")
SRC = ROOT / "kernels/mv_body_isa.hip"

SCHED = ("s_waitcnt", "s_delay_alu", "s_nop")


def build(defines, want_asm=True):
    out = "/tmp/mv-census.s" if want_asm else "/dev/null"
    cmd = ["hipcc", "--offload-arch=gfx1151", "-O3", "-std=c++17", "-Isrc", "-Ikernels",
           f"-I{LLAMA}/include", f"-I{LLAMA}/ggml/include", "--cuda-device-only",
           *[f"-D{k}={v}" for k, v in defines.items()]]
    cmd += (["-S", "-o", out, str(SRC)] if want_asm else
            ["-Rpass-analysis=kernel-resource-usage", "-c", "-o", out, str(SRC)])
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-4000:])
        raise SystemExit("compile failed")
    return Path(out).read_text() if want_asm else r.stderr


def kernel_body(asm, mangled_part):
    i = asm.index(mangled_part)
    i = asm.rindex("\n", 0, i)
    start = asm.index(":", i)
    return asm[start:asm.index(".Lfunc_end", start)]


def census(body):
    """Instruction totals for a kernel, and for the largest inner-loop region of its block loop.

    The block loop is everything between the inner-loop header label and the loop's backward
    branch, which in this body is one iteration over all TT rows.
    """
    lines = [l.strip() for l in body.splitlines()]
    mn, waits, in_loop, loop = [], [], False, []
    for l in lines:
        if "This Inner Loop" in l:
            in_loop = True
        m = re.match(r"([a-z][a-z0-9_]*)", l)
        if not m:
            continue
        op = m.group(1)
        mn.append(op)
        if in_loop:
            loop.append(op)
        if op == "s_waitcnt":
            waits.append(l.split(None, 1)[1].strip())
    return {
        "total": len(mn),
        "loop_slots": len(loop),
        "loop_work": sum(1 for o in loop if not o.startswith(SCHED)),
        "lgkm_waits": sum(1 for w in waits if "lgkmcnt" in w),
        "vm_waits": sum(1 for w in waits if "vmcnt" in w),
        "s_load": sum(1 for o in mn if o.startswith("s_load")),
        "global_load": sum(1 for o in mn if o.startswith("global_load")),
        "v_dot4": sum(1 for o in mn if o.startswith("v_dot4")),
        "scratch": sum(1 for o in mn if o.startswith("scratch_")),
        "waits": dict(Counter(waits)),
    }


FIELDS = {"TotalSGPRs": "sgpr", "VGPRs": "vgpr", "ScratchSize [bytes/lane]": "scratch",
          "Occupancy [waves/SIMD]": "waves", "SGPRs Spill": "sgpr_spill", "VGPRs Spill": "vgpr_spill"}


def resources(remarks, mangled_part):
    cur, hit = None, {}
    for line in remarks.splitlines():
        m = re.search(r"remark: Function Name: (\S+)", line)
        if m:
            cur = m.group(1)
            continue
        m = re.search(r"remark:\s+([A-Za-z ]+(?:\[[^\]]*\])?):\s*(\S+)", line)
        if m and cur and mangled_part in cur and m.group(1).strip() in FIELDS:
            hit[FIELDS[m.group(1).strip()]] = int(m.group(2))
    return hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mvs", default="0,1", help="0 deployed schedule, 1 row-batched scalar (the default arm)")
    ap.add_argument("--rb", default="2", help="rows per scalar batch in arm 1")
    ap.add_argument("--ar", default="1", help="kept for the bench's rejected vector arm")
    ap.add_argument("--depth", default="1", help="kept for the bench's rejected depth arm; the runtime has no depth axis")
    ap.add_argument("--tt", default="1,8", help="rows per pass")
    ap.add_argument("--acc", default="4")
    ap.add_argument("--json", help="write the table here")
    args = ap.parse_args()

    rows = []
    for mvs in [int(x) for x in args.mvs.split(",")]:
      for rb in [int(x) for x in args.rb.split(",")]:
        for ar in [int(x) for x in args.ar.split(",")]:
         for depth in [int(x) for x in args.depth.split(",")]:
            if mvs == 0 and depth != 1:
                continue  # the deployed body has no depth axis
            if mvs != 1 and rb != int(args.rb.split(",")[0]):
                continue  # RB only reaches arm 1
            if mvs != 2 and ar != int(args.ar.split(",")[0]):
                continue  # AR only reaches arm 2
            defs = {"HALO_MV_SCHED": mvs, "HALO_MV_RB": rb, "HALO_MV_DEPTH": depth, "HALO_MV_AR": ar}
            asm = build(defs)
            rem = build(defs, want_asm=False)
            for tt in [int(x) for x in args.tt.split(",")]:
                part = f"ILi5ELi{tt}ELi{args.acc}EE"
                c = census(kernel_body(asm, part))
                c.update(resources(rem, part))
                c.update(mvs=mvs, rb=rb, ar=ar, depth=depth, tt=tt)
                rows.append(c)
                print(f"arm={mvs} RB={rb} AR={ar} depth={depth} TT={tt}: "
                      f"lgkm {c['lgkm_waits']:3d} vm {c['vm_waits']:2d}  loop {c['loop_slots']:4d} slots "
                      f"({c['loop_work']} work)  s_load {c['s_load']:3d}  dot4 {c['v_dot4']:3d}  "
                      f"vgpr {c.get('vgpr')} sgpr {c.get('sgpr')} waves {c.get('waves')} "
                      f"spill {c.get('vgpr_spill')}/{c.get('sgpr_spill')} scratch {c['scratch']}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
        print("wrote", args.json)


if __name__ == "__main__":
    main()

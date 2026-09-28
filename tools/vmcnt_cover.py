#!/usr/bin/env python3
"""How long each memory request of a block loop survives before a wait drains it.

gfx11 gives a wave one in-order `vmcnt` counter. Loads return in issue order, so
`s_waitcnt vmcnt(N)` waits for every outstanding request except the last N - including
requests the waiting instruction does not want. A weight cursor issued at the top of a
block is therefore drained by the block's first wait for anything issued after it, and
no amount of lookahead depth changes that: docs/ffn-decode-schedule.md measured the
consequence on `k_proj_opt` and could not see it in a slot census, because a census
counts instructions and this is about their order.

This walks the largest basic block of a kernel as the loop it is - twice, so the second
pass reads a steady state with the previous iteration's requests still outstanding - and
reports, per request, how many issue slots and matrix instructions stand between the
request and the wait that drains it. That distance is the cover: what the wave has to
issue while the memory system works. Big cover, latency hidden; two slots of cover, the
wave stalls on every block whatever its occupancy.

    tools/vmcnt_cover.py /tmp/slice32.s k_ffn_slice
    tools/vmcnt_cover.py listing.s k_proj_opt --block 1

No GPU, no lock, no model: it reads a `hipcc --cuda-device-only -S` listing.
"""
import re
import sys

LOAD = re.compile(r"^(global_load|buffer_load|flat_load|scratch_load|global_atomic)")
WAIT = re.compile(r"^s_waitcnt\b")
VMCNT = re.compile(r"vmcnt\((\d+)\)")
# A wave issues one instruction per cycle; a 16x16x16 matrix instruction occupies its
# SIMD for many, so cover is reported in both units and the caller prices with the one
# its kernel is bound by.
MATRIX = re.compile(r"^v_(wmma|mfma|dot)")


def kernel_blocks(path, needle):
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    start = None
    for i, ln in enumerate(lines):
        s = ln.split(";")[0].strip()
        if s.endswith(":") and needle in s and not s.startswith("."):
            start = i
            name = s[:-1]
            break
    if start is None:
        raise SystemExit("kernel not found: " + needle)
    body = []
    for j in range(start + 1, len(lines)):
        if lines[j].strip().startswith(".Lfunc_end"):
            break
        body.append(lines[j])
    cur, out, label = [], [], "entry"
    for ln in body:
        s = ln.strip()
        if not s or (s.startswith((";", "//", ".")) and not s.startswith(".L")):
            continue
        if s.startswith(".L") and s.endswith(":"):
            if cur:
                out.append((label, cur))
            cur, label = [], s[:-1]
            continue
        m = s.split()[0]
        if not re.match(r"^[a-z][a-z0-9_]*$", m):
            continue
        cur.append(s)
        if m.startswith(("s_branch", "s_cbranch", "s_endpgm")):
            out.append((label, cur))
            cur, label = [], label + "+"
    if cur:
        out.append((label, cur))
    out.sort(key=lambda kv: -len(kv[1]))
    return name, out


def cover(block):
    """Simulate two passes of the block; report the second pass's drains."""
    n = len(block)
    outstanding = []          # (issue index in the doubled stream, text)
    drains = []               # (issue idx, wait idx, slots, matrix ops in between)
    matrix_at = []
    for rep in range(2):
        for i, ins in enumerate(block):
            k = rep * n + i
            mnem = ins.split()[0]
            if MATRIX.match(mnem):
                matrix_at.append(k)
            if LOAD.match(mnem):
                outstanding.append((k, ins))
                continue
            if WAIT.match(mnem):
                m = VMCNT.search(ins)
                if not m:
                    continue
                keep = int(m.group(1))
                while len(outstanding) > keep:
                    issued, text = outstanding.pop(0)
                    if rep == 1 or issued >= n:
                        mats = sum(1 for x in matrix_at if issued < x < k)
                        drains.append((issued, k, k - issued, mats, text))
    return drains


def main():
    path, needle = sys.argv[1], sys.argv[2]
    nth = 0
    if "--block" in sys.argv:
        nth = int(sys.argv[sys.argv.index("--block") + 1])
    name, blocks = kernel_blocks(path, needle)
    label, block = blocks[nth]
    n = len(block)
    print(name)
    print(f"block {label}: {n} instructions, "
          f"{sum(1 for i in block if LOAD.match(i.split()[0]))} requests, "
          f"{sum(1 for i in block if MATRIX.match(i.split()[0]))} matrix")
    drains = [d for d in cover(block) if d[0] >= n]
    if not drains:
        print("no request in the steady-state pass was drained inside two iterations")
        return
    print(f"{'request':<46} {'issued':>7} {'drained':>8} {'cover':>7} {'matrix in cover':>16}")
    for issued, waitidx, dist, mats, text in drains:
        short = " ".join(text.split()[:2])[:44]
        print(f"{short:<46} {issued - n:>7} {waitidx - n:>8} {dist:>7} {mats:>16}")
    worst = min(drains, key=lambda d: d[2])
    print(f"\nshortest cover {worst[2]} slots ({worst[3]} matrix) on: {worst[4][:80]}")
    tot = sum(d[2] for d in drains) / len(drains)
    print(f"mean cover {tot:.1f} slots over {len(drains)} requests")


if __name__ == "__main__":
    main()

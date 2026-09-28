#!/usr/bin/env python3
"""Count instructions per basic block of one kernel in a gfx assembly listing.

Reads a `hipcc -S` listing, isolates a kernel by mangled-name substring, splits it at
labels and branches, and prints the biggest blocks with a mnemonic histogram. The
inner block loop of a projection kernel is the largest block by a wide margin, so
this prices its issue directly instead of counting source statements.
"""
import re
import sys
from collections import Counter

CLASSES = [
    ("wmma", lambda m: m.startswith("v_wmma")),
    ("cvt", lambda m: m.startswith("v_cvt")),
    ("fma/mac", lambda m: m.startswith(("v_fma", "v_fmac", "v_mac", "v_dual_fma", "v_pk_fma"))),
    ("mul", lambda m: m.startswith(("v_mul", "v_dual_mul", "v_pk_mul"))),
    ("add", lambda m: m.startswith(("v_add", "v_dual_add", "v_pk_add"))),
    ("perm", lambda m: m.startswith("v_perm")),
    ("bitops", lambda m: m.startswith(("v_and", "v_or", "v_xor", "v_lshl", "v_lshr", "v_bfe", "v_bfi", "v_ashr"))),
    ("mov", lambda m: m.startswith(("v_mov", "v_dual_mov", "v_accvgpr", "v_readlane", "v_writelane"))),
    ("permlane", lambda m: m.startswith(("v_permlane", "ds_bpermute", "ds_permute"))),
    ("global_load", lambda m: m.startswith("global_load")),
    ("global_store", lambda m: m.startswith("global_store")),
    ("ds", lambda m: m.startswith("ds_")),
    ("scratch", lambda m: m.startswith("scratch_")),
    ("s_load", lambda m: m.startswith(("s_load", "s_buffer"))),
    ("salu", lambda m: m.startswith("s_") and not m.startswith(("s_load", "s_waitcnt", "s_branch", "s_cbranch", "s_delay", "s_nop", "s_buffer"))),
    ("s_delay_alu", lambda m: m.startswith("s_delay")),
    ("s_waitcnt", lambda m: m.startswith("s_waitcnt")),
    ("s_nop", lambda m: m.startswith("s_nop")),
    ("branch", lambda m: m.startswith(("s_branch", "s_cbranch"))),
]

# Classes the compiler emits to schedule the other ones. `s_delay_alu` is a hint that carries no
# operation, and the scheduler moves it around freely; `s_waitcnt` and `s_nop` mark where a wave
# stops rather than what it does. A census delta made entirely of these predicts nothing about
# time, which is not a guess: two bit-identical changes on this device moved the raw total of a
# selected block loop by -6.2% and +2.0% while moving the work total by -4 and 0 slots, and both
# measured null on the phase they changed. docs/activation-scale-axis.md has both panels.
SCHEDULING = ("s_delay_alu", "s_waitcnt", "s_nop")


def address_forms(body):
    """How each memory instruction reaches its address.

    A source-level addressing change can leave the emitted form untouched, and then the slot count
    moves by a handful of scheduling slots and the change does nothing. `v[n:n+1], off` is a 64-bit
    address pair the wave built and holds two registers for; `voff, s[n:n+1]` is one scalar base per
    wave with a 32-bit lane offset, which is what an operand cursor is supposed to produce. Compare
    this table across the two builds before believing that an addressing change took.
    """
    forms = Counter()
    for ln in body:
        m = re.match(r"^\s*((?:global|scratch|flat)_\w+)\s+(.*?)(?://.*)?$", ln)
        if not m:
            continue
        ops = m.group(2)
        if re.search(r",\s*s\[\d+:\d+\]", ops):
            kind = "scalar base + lane offset"
        elif re.search(r"v\[\d+:\d+\]\s*,\s*off", ops):
            kind = "64-bit VGPR address pair"
        else:
            kind = "other"
        forms[(m.group(1), kind)] += 1
    return forms


def classify(m):
    for name, pred in CLASSES:
        if pred(m):
            return name
    return "other:" + m.split("_")[0]


def kernel_body(path, needle):
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    start = None
    name = None
    for i, ln in enumerate(lines):
        s = ln.split(";")[0].strip()
        if s.endswith(":") and needle in s and not s.startswith("."):
            start = i
            name = s[:-1]
            break
    if start is None:
        raise SystemExit("kernel not found: " + needle)
    for j in range(start, len(lines)):
        if lines[j].strip().startswith(".Lfunc_end"):
            return name, lines[start + 1:j]
    return name, lines[start + 1:]


def blocks(body):
    cur, out, label = [], [], "entry"
    for ln in body:
        s = ln.strip()
        if not s or s.startswith((";", "//", ".")) and not s.startswith(".L"):
            continue
        if s.startswith(".L") and s.endswith(":"):
            if cur:
                out.append((label, cur))
            cur, label = [], s[:-1]
            continue
        m = s.split()[0]
        if not re.match(r"^[a-z][a-z0-9_]*$", m):
            continue
        cur.append(m)
        if m.startswith(("s_branch", "s_cbranch", "s_endpgm")):
            out.append((label, cur))
            cur, label = [], label + "+"
    if cur:
        out.append((label, cur))
    return out


def main():
    path, needle = sys.argv[1], sys.argv[2]
    top = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    name, body = kernel_body(path, needle)
    print(name)
    vg = [ln.strip() for ln in body if "vgpr_count" in ln or "occupancy" in ln or "spill" in ln]
    bs = sorted(blocks(body), key=lambda kv: -len(kv[1]))
    def report(title, counts):
        total = sum(counts.values())
        sched = sum(counts.get(k, 0) for k in SCHEDULING)
        print(f"\n== {title}: {total} instructions, {total - sched} work + {sched} scheduling ==")
        for k, v in counts.most_common():
            print(f"   {k:14s} {v:5d}{'   (scheduling)' if k in SCHEDULING else ''}")

    for label, insts in bs[:top]:
        report(f"block {label}", Counter(classify(m) for m in insts))
    report("whole kernel", Counter(classify(m) for _, insts in bs for m in insts))
    forms = address_forms(body)
    if forms:
        print("\n== memory address forms ==")
        for (op, kind), n in sorted(forms.items()):
            print(f"   {op:20s} {kind:26s} {n:4d}")
    for ln in vg:
        print("   " + ln)


main()

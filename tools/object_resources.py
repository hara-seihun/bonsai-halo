#!/usr/bin/env python3
"""Per-kernel registers, spills, scratch and LDS read out of objects that are already built.

`tools/kernel_resources.py` asks the compiler, which means compiling the translation unit again, and
`kernels/halo_rows.hip` takes 36 to 75 seconds of device codegen on this box - often more than one
bounded tool call. The numbers it prints are already in every object `make` or `tools/split-build`
wrote: the device code object carries AMDGPU metadata with each kernel's VGPR and SGPR count, spill
count, private segment (scratch per lane) and group segment (LDS per block). This reads them in
under a second and puts several objects side by side, which is the before/after register check a
persistent-kernel change owes its reviewers.

    tools/object_resources.py kernels/halo_rows.o k_forward_rows
    tools/object_resources.py /path/to/base/halo_rows.o kernels/halo_rows.o 'k_forward_rows<8'

The last argument is a substring filter on the demangled name when more than one path is given;
with one path and no filter every kernel is printed. A row is marked when the objects disagree.
Occupancy is not printed: on gfx1151 it follows the 24-register granule ladder that
`bench/occ_resident` measured (docs/rows-grid-occupancy.md), not the compiler's estimate.
"""
import os
import re
import subprocess
import sys
import tempfile

TARGET = "hipv4-amdgcn-amd-amdhsa--gfx1151"
FIELDS = ("name", "vgpr_count", "sgpr_count", "vgpr_spill_count", "private_segment_fixed_size",
          "group_segment_fixed_size")
FIELD_RE = re.compile(r"\s+\.(%s):\s+(\S+)" % "|".join(FIELDS))


def device_object(obj, workdir):
    """The gfx1151 code object inside a HIP host object's offload bundle."""
    fatbin = os.path.join(workdir, os.path.basename(obj) + ".fatbin")
    code = os.path.join(workdir, os.path.basename(obj) + ".co")
    subprocess.run(["objcopy", "-O", "binary", "--only-section=.hip_fatbin", obj, fatbin], check=True)
    subprocess.run(["clang-offload-bundler", "--unbundle", "--type=o", "--input=" + fatbin,
                    "--targets=" + TARGET, "--output=" + code], check=True)
    return code


def kernels(code):
    """name -> fields. The metadata notes list each kernel's keys in sorted order, so a repeated key
    starts the next kernel's record."""
    notes = subprocess.run(["llvm-readelf", "--notes", code], capture_output=True, text=True, check=True).stdout
    records, cur = [], {}
    for line in notes.splitlines():
        m = FIELD_RE.match(line)
        if not m:
            continue
        key, value = m.groups()
        if key in cur:
            records.append(cur)
            cur = {}
        cur[key] = value
    if cur:
        records.append(cur)
    return {r["name"]: r for r in records if "name" in r}


def demangle(names):
    out = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True, check=True).stdout
    return dict(zip(names, out.splitlines()))


def cell(r):
    if r is None:
        return "-"
    return "%3s VGPR %3s SGPR  spill %-3s scratch %-4s LDS %s" % (
        r.get("vgpr_count"), r.get("sgpr_count"), r.get("vgpr_spill_count"),
        r.get("private_segment_fixed_size"), r.get("group_segment_fixed_size"))


def main(argv):
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0 if len(argv) > 1 else 2
    args = argv[1:]
    objs = [a for a in args if os.path.isfile(a)]
    rest = [a for a in args if not os.path.isfile(a)]
    pattern = rest[-1] if rest else ""
    with tempfile.TemporaryDirectory() as workdir:
        tables = []
        for i, obj in enumerate(objs):
            sub = os.path.join(workdir, str(i))
            os.makedirs(sub)
            tables.append(kernels(device_object(obj, sub)))
    names = sorted(set().union(*[t.keys() for t in tables]))
    plain = demangle(names)
    for i, obj in enumerate(objs):
        print("[%d] %s" % (i, obj))
    for n in names:
        if pattern and pattern not in plain[n]:
            continue
        cells = [cell(t.get(n)) for t in tables]
        mark = "" if len(set(cells)) == 1 else "   <- differs"
        print(plain[n])
        for i, c in enumerate(cells):
            print("    [%d] %s%s" % (i, c, mark if i == len(cells) - 1 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

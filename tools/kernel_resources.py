#!/usr/bin/env python3
"""Per-instantiation register and occupancy report for a HIP translation unit.

The compiler already knows how many VGPRs a kernel shape needs and how many waves that leaves
resident per SIMD32, and it will tell you without a GPU. That is the cheapest way to find out
whether a change crossed an occupancy cliff, which on gfx1151 is what decides whether the machine
can hide a weight load behind other waves' work.

    tools/kernel_resources.py kernels/ffn_batch.hip k_proj_opt
    tools/kernel_resources.py -D HALO_MV_ACC=1 kernels/halo_rows.hip k_ffn_slice
    tools/kernel_resources.py --json kernels/ffn_batch.hip | jq .

Matching is on the demangled name, so 'k_proj_opt<1, 1, 1, true, 4, 2, true>' selects one shape.
"""
import argparse
import json
import re
import subprocess
import sys

LLAMA = "../bonsai-hip"
FIELDS = {
    "TotalSGPRs": "sgpr",
    "VGPRs": "vgpr",
    "ScratchSize [bytes/lane]": "scratch",
    "Occupancy [waves/SIMD]": "waves",
    "SGPRs Spill": "sgpr_spill",
    "VGPRs Spill": "vgpr_spill",
    "LDS Size [bytes/block]": "lds",
}


def compile_remarks(source, arch, defines=()):
    cmd = [
        "hipcc", f"--offload-arch={arch}", "-O3", "-std=c++17", "-Isrc", "-Ikernels",
        f"-I{LLAMA}/include", f"-I{LLAMA}/ggml/include", "--cuda-device-only",
        "-Rpass-analysis=kernel-resource-usage", *[f"-D{d}" for d in defines],
        "-c", "-o", "/dev/null", source,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.stderr.write(out.stderr)
        raise SystemExit(f"compile failed: {source}")
    return out.stderr


def parse(text):
    kernels, current = [], None
    name_re = re.compile(r"remark: Function Name: (\S+)")
    field_re = re.compile(r"remark:\s+([A-Za-z ]+(?:\[[^\]]*\])?):\s*(\S+)")
    for line in text.splitlines():
        m = name_re.search(line)
        if m:
            current = {"mangled": m.group(1)}
            kernels.append(current)
            continue
        if current is None:
            continue
        m = field_re.search(line)
        if m and m.group(1).strip() in FIELDS:
            value = m.group(2)
            current[FIELDS[m.group(1).strip()]] = int(value) if value.isdigit() else value
    names = subprocess.run(["c++filt"], input="\n".join(k["mangled"] for k in kernels),
                           capture_output=True, text=True).stdout.splitlines()
    for k, n in zip(kernels, names):
        n = n.removeprefix("void ").replace("halo::(anonymous namespace)::", "").replace("halo::", "")
        k["name"] = n[:n.rindex("(")] if "(" in n else n
    return kernels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("match", nargs="?", default="")
    ap.add_argument("--arch", default="gfx1151")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("-D", "--define", action="append", default=[],
                    help="preprocessor define, for pricing a compile-time axis without editing the source")
    a = ap.parse_args()
    kernels = [k for k in parse(compile_remarks(a.source, a.arch, a.define)) if a.match in k["name"]]
    if a.json:
        print(json.dumps(kernels, indent=2))
        return
    print(f"{'waves':>5} {'vgpr':>5} {'sgpr':>5} {'spill':>5} {'lds':>6}  kernel")
    for k in sorted(kernels, key=lambda k: (k["waves"], k["name"])):
        print(f"{k['waves']:5d} {k['vgpr']:5d} {k['sgpr']:5d} {k['vgpr_spill']:5d} "
              f"{k['lds']:6d}  {k['name']}")


if __name__ == "__main__":
    main()

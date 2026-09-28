#!/usr/bin/env python3
"""Instruction census of the recurrent state codec's load and store loops.

`resident_state` is a 75-second translation unit, and the only thing a storage-coordinate change
moves inside it is `gdn_state_load_row` and `gdn_state_store_row`. `kernels/gdn_state_isa.hip`
instantiates those two loops on their own with the format as a compile-time constant, so an arm can
be priced in eight seconds with no GPU lock and no model.

    tools/gdn_state_isa.py                       # compiles the harness and prints the table
    tools/gdn_state_isa.py --asm /tmp/gdn.s      # read an assembly file you already have

The columns that decide a coordinate: the memory op mix (four `global_load_b32` against one
`global_load_b128` is the whole point of the grouped row order) and `branch`, because a divergent
store ends a basic block and the R rows of the store loop stop being schedulable against each
other.
"""
import argparse
import collections
import pathlib
import re
import subprocess
import sys

FMT = {0: 'f32', 1: 'f16', 2: 'i16', 3: 'i8', 4: 'f32pk', 5: 'i8c', 6: 'i8s'}
KERNEL = re.compile(r'^_Z14k_state_censusILi(\d+)ELi(\d+)EEvPfif:')
COLUMNS = ['total', 'valu', 'salu', 'xlane', 'branch', 'delay', 'wait']


def compile_harness(root, out):
    src = root / 'kernels' / 'gdn_state_isa.hip'
    cmd = ['hipcc', '--offload-arch=gfx1151', '--cuda-device-only', '-S', '-O3',
           '-I' + str(root / 'src'), '-I' + str(root / 'kernels'), str(src), '-o', str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.stderr.write(r.stderr)
        raise SystemExit(f'compile failed: {" ".join(cmd)}')
    return out


def census(path):
    per, cur = collections.OrderedDict(), None
    for line in pathlib.Path(path).read_text().splitlines():
        m = KERNEL.match(line)
        if m:
            cur = f'{FMT.get(int(m.group(1)), m.group(1))}/S{m.group(2)}'
            per[cur] = collections.Counter()
            continue
        if cur is None:
            continue
        if '.size' in line:
            cur = None
            continue
        tok = line.strip().split()
        if not tok or tok[0].startswith((';', '.')) or tok[0].endswith(':'):
            continue
        op, c = tok[0], per[cur]
        c['total'] += 1
        if op.startswith(('global_', 'buffer_', 'flat_', 's_load', 'scratch_')):
            c[op] += 1
        elif op == 's_delay_alu':
            c['delay'] += 1
        elif op.startswith(('s_cbranch', 's_branch')):
            c['branch'] += 1
        elif 'dpp' in op:
            c['xlane'] += 1
        elif op.startswith('v_'):
            c['valu'] += 1
        elif op.startswith('s_wait'):
            c['wait'] += 1
        elif op.startswith('s_'):
            c['salu'] += 1
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--asm', help='assembly file; compiled from the harness when omitted')
    ap.add_argument('--out', default='/tmp/gdn-state-census.s')
    args = ap.parse_args()
    root = pathlib.Path(__file__).resolve().parent.parent
    path = args.asm or compile_harness(root, pathlib.Path(args.out))
    per = census(path)
    if not per:
        raise SystemExit(f'no k_state_census instantiations in {path}')
    mem = sorted({o for c in per.values() for o in c
                  if o.startswith(('global', 'flat', 'buffer', 's_load', 'scratch'))})
    print(f'{"":<18}' + ''.join(f'{k:>10}' for k in per))
    for row in COLUMNS + mem:
        print(f'{row:<18}' + ''.join(f'{per[k].get(row, 0):>10}' for k in per))
    return 0


if __name__ == '__main__':
    sys.exit(main())

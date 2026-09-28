#!/usr/bin/env python3
"""Instruction mix of the single-token matvec block loop, per FwdParams::single_map value.

The interesting number is the VALU count of the basic block that holds the 32 v_dot4 of one
128-weight block: that is the whole per-block cost of one lane's row, decode included.

    hipcc --offload-arch=gfx1151 -O3 -std=c++17 -w -Isrc -Ikernels --cuda-device-only -S \
          kernels/halo_rows.hip -o /tmp/rows.s
    python3 kernels/single_map_isa.py /tmp/rows.s
"""
import collections
import re
import sys

KIND = [
    ('dot4', ('v_dot4',)),
    ('perm', ('v_perm',)),
    ('pk_mul', ('v_pk_mul',)),
    ('shift', ('v_pk_lshr', 'v_lshr')),
    ('and', ('v_and',)),
    ('or/shl', ('v_or', 'v_lshl_or', 'v_bfi', 'v_lshlrev', 'v_alignbit')),
    ('vmem', ('global_load', 'global_store', 'buffer_')),
    ('s_load', ('s_load',)),
    ('spill_lane', ('v_readlane', 'v_writelane')),
    ('s_wait', ('s_wait',)),
]
VALU = {'dot4', 'perm', 'pk_mul', 'shift', 'and', 'or/shl', 'spill_lane', 'valu_other'}


def classify(op):
    for name, prefixes in KIND:
        if op.startswith(prefixes):
            return name
    if op.startswith('v_'):
        return 'valu_other'
    if op.startswith('s_'):
        return 'salu'
    return 'misc'


def functions(text):
    pattern = r'^([_A-Za-z0-9]+): *; *@[_A-Za-z0-9]+\n(.*?)^\.Lfunc_end'
    return {m.group(1): m.group(2) for m in re.finditer(pattern, text, re.S | re.M)}


def blocks(body):
    name, cur = 'entry', []
    for line in body.split('\n'):
        t = line.strip()
        if t.startswith('.LBB') and t.endswith(':'):
            yield name, cur
            name, cur = t[:-1], []
        else:
            cur.append(t)
    yield name, cur


def main(path):
    text = open(path).read()
    for fn, body in sorted(functions(text).items()):
        if 'k_forward_rows' not in fn and 'k_ffn_slice' not in fn:
            continue
        rows = []
        for name, seg in blocks(body):
            counts = collections.Counter()
            total = 0
            for t in seg:
                if not t or t.startswith(('.', ';', '//')) or t.endswith(':'):
                    continue
                counts[classify(t.split()[0])] += 1
                total += 1
            if counts['dot4'] >= 16:
                valu = sum(v for k, v in counts.items() if k in VALU)
                rows.append((counts['dot4'], total, valu, name, counts))
        if not rows:
            continue
        print(f'== {fn}')
        for dots, total, valu, name, counts in sorted(rows):
            per = dots // 32
            mix = '  '.join(f'{k}={v}' for k, v in sorted(counts.items()))
            # per counts groups of 32 dot4: one 128-weight block at TT = 1, one block per row above
            print(f'   {name:14s} x{per}: {total:4d} instr  {valu:4d} VALU  ({valu // per} VALU per 32 dot4)')
            print(f'   {"":14s} {mix}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '/tmp/rows.s')

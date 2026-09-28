#!/usr/bin/env python3
"""Summarize a rocprofv3 kernel CSV from a bounded Qwen decode panel."""

import argparse
import collections
import csv
import json
from pathlib import Path

QUANT = {'8': 'Q8_0', '12': 'Q4_K', '13': 'Q5_K', '14': 'Q6_K'}


def family(kernel):
    if 'mul_mat_vec_q<' in kernel:
        kind = kernel.split('mul_mat_vec_q<(ggml_type)', 1)[1].split(',', 1)[0]
        return 'matvec/' + QUANT.get(kind, kind)
    if 'mul_mat_vec_f<' in kernel:
        return 'matvec/float'
    if 'topk_moe_cuda<' in kernel:
        return 'router/topk'
    if 'quantize_q8_1(' in kernel:
        return 'activation/q8'
    if 'gated_delta_net_cuda<' in kernel:
        return 'state/gdn'
    if 'flash_attn_tile<' in kernel:
        return 'attention/flash'
    return 'other'


def summarize(path, tokens):
    buckets = collections.defaultdict(lambda: {'calls': 0, 'device_ms': 0.0})
    shapes = collections.defaultdict(lambda: {'calls': 0, 'device_ms': 0.0})
    total_ms = 0.0
    calls = 0
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            elapsed = (int(row['End_Timestamp']) - int(row['Start_Timestamp'])) / 1_000_000
            bucket = buckets[family(row['Kernel_Name'])]
            bucket['calls'] += 1
            bucket['device_ms'] += elapsed
            geometry = tuple(int(row[key]) for key in (
                'Workgroup_Size_X', 'Workgroup_Size_Y', 'Workgroup_Size_Z',
                'Grid_Size_X', 'Grid_Size_Y', 'Grid_Size_Z'))
            shape = shapes[(family(row['Kernel_Name']), geometry)]
            shape['calls'] += 1
            shape['device_ms'] += elapsed
            total_ms += elapsed
            calls += 1
    if calls == 0 or tokens < 1:
        raise ValueError('empty trace or invalid token count')
    return {'trace': str(path.resolve()), 'measured_tokens': tokens,
            'calls': calls, 'summed_device_ms': total_ms,
            'summed_device_ms_per_token': total_ms / tokens,
            'geometry': [{'family': name, 'workgroup': list(geometry[:3]),
                          'grid': list(geometry[3:]), 'calls': record['calls'],
                          'device_ms': round(record['device_ms'], 6)}
                         for (name, geometry), record in sorted(shapes.items(), key=lambda item: -item[1]['device_ms'])],
            'families': [{'name': name, 'calls': record['calls'],
                          'device_ms': round(record['device_ms'], 3),
                          'fraction': round(record['device_ms'] / total_ms, 4)}
                         for name, record in sorted(buckets.items(), key=lambda item: -item[1]['device_ms'])]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--tokens', type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.csv, args.tokens), indent=2))


if __name__ == '__main__':
    main()

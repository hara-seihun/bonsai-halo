#!/usr/bin/env python3
"""Join a one-token Qwen decode trace to the pinned GGUF Q8 tensor inventory."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

SHAPE_GRID = {512: 16384, 2048: 65536, 4096: 131072, 8192: 262144}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def analyze(trace, inventory_path, tokens, bandwidth):
    if tokens <= 0 or bandwidth <= 0:
        raise ValueError('tokens and bandwidth must be positive')
    records = defaultdict(lambda: {'calls': 0, 'nanoseconds': 0, 'variants': defaultdict(lambda: {'calls': 0, 'nanoseconds': 0})})
    with trace.open(newline='') as stream:
        for row in csv.DictReader(stream):
            name = row['Kernel_Name']
            if 'mul_mat_vec_q<(ggml_type)8,' not in name:
                continue
            grid = int(row['Grid_Size_X'])
            if grid not in SHAPE_GRID.values() or tuple(int(row[k]) for k in
                    ('Workgroup_Size_X', 'Workgroup_Size_Y', 'Workgroup_Size_Z')) != (32, 1, 1):
                raise ValueError(f'unexpected Q8 geometry: {grid} {name}')
            duration = int(row['End_Timestamp']) - int(row['Start_Timestamp'])
            if duration < 0:
                raise ValueError('negative duration')
            item = records[grid]
            item['calls'] += 1
            item['nanoseconds'] += duration
            variant = item['variants'][name.split('>', 1)[0].split('<', 1)[1]]
            variant['calls'] += 1
            variant['nanoseconds'] += duration
    inventory = json.loads(inventory_path.read_text())
    tensors = defaultdict(list)
    for item in inventory['tensors']:
        if item['type'] == 'Q8_0' and item['name'] != 'token_embd.weight':
            if item['shape'][-1] not in SHAPE_GRID:
                raise ValueError(f'unaccounted Q8 tensor: {item["name"]}')
            tensors[item['shape'][-1]].append(item)
    output = []
    for width, grid in SHAPE_GRID.items():
        calls = records[grid]['calls']
        tensor_count = len(tensors[width])
        if calls % tokens or not tensor_count:
            raise ValueError(f'invalid width {width} calls')
        per_token = calls // tokens
        # Shared gate/up is one fused call over two independent Q8 tensors.
        fused_calls = sum(r['calls'] for v, r in records[grid]['variants'].items()
                          if v.startswith('(ggml_type)8, 1, true, true,'))
        fused = fused_calls // tokens
        if fused_calls != fused * tokens:
            raise ValueError('incomplete fused-call panel')
        if per_token + fused != tensor_count:
            raise ValueError(f'Q8 call/tensor mismatch width {width}: {per_token}+{fused}!={tensor_count}')
        bytes_per_token = sum(t['bytes'] for t in tensors[width])
        ms = records[grid]['nanoseconds'] / (tokens * 1e6)
        floor = bytes_per_token / (bandwidth * 1e9) * 1000
        output.append({'output_width': width, 'grid_x': grid, 'calls_per_token': per_token,
                       'fused_gate_up_calls_per_token': fused, 'tensors_per_token': tensor_count,
                       'weight_bytes_per_token': bytes_per_token, 'device_ms_per_token': round(ms, 6),
                       'image_bytes_per_second_gb': round(bytes_per_token / (ms * 1e6), 3),
                       'one_read_floor_ms_at_bandwidth': round(floor, 6),
                       'excess_ms_over_floor': round(ms - floor, 6),
                       'variants': [{'template': name, 'calls_per_token': value['calls'] // tokens,
                                     'device_ms_per_token': round(value['nanoseconds'] / (tokens * 1e6), 6)}
                                    for name, value in sorted(records[grid]['variants'].items())],
                       'tensor_names': dict(sorted(Counter(t['name'].split('.')[-2] for t in tensors[width]).items()))})
    total_ms = sum(r['device_ms_per_token'] for r in output)
    total_bytes = sum(r['weight_bytes_per_token'] for r in output)
    return {'analysis_source_sha256': digest(Path(__file__)),
            'trace_sha256': digest(trace), 'inventory_sha256': digest(inventory_path),
            'header_sha256': inventory['header_sha256'], 'tokens': tokens,
            'assumed_one_read_bandwidth_gb_s': bandwidth, 'groups': output,
            'q8_device_ms_per_token': round(total_ms, 6), 'q8_image_bytes_per_token': total_bytes,
            'q8_one_read_floor_ms': round(total_bytes / (bandwidth * 1e9) * 1000, 6),
            'q8_excess_ms': round(sum(r['excess_ms_over_floor'] for r in output), 6)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('inventory', type=Path)
    parser.add_argument('--tokens', type=int, required=True)
    parser.add_argument('--bandwidth', type=float, default=242.0, help='one-read GB/s comparison, not DRAM counter')
    args = parser.parse_args()
    print(json.dumps(analyze(args.trace, args.inventory, args.tokens, args.bandwidth), indent=2))

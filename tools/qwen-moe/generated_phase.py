#!/usr/bin/env python3
"""Account for two complete generated 32-stream steps in paired Qwen kernel traces.

The preceding 256-row seed ends at the first full vocabulary Q6_K head.
Each subsequent full head closes one generated step. Never treat trace host gaps
as device work, and never count the seed as generated throughput.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def group(row):
    name = row['Kernel_Name']
    if '(ggml_type)' in name and name.startswith(('void mul_mat_q<', 'void mul_mat_vec_q<')):
        kind = name.split('(ggml_type)', 1)[1].split(',', 1)[0]
        if kind == '8':
            return 'nonexpert_q8'
        if kind in ('12', '13') or kind == '14' and row['Grid_Size_X'] in ('512', '1024'):
            return 'routed_experts'
        if kind == '14' and row['Grid_Size_X'] == '124160':
            return 'full_vocabulary_head'
        raise ValueError(f'unclassified quantized projection {name} grid={row["Grid_Size_X"]}')
    if name.startswith('void gated_delta_net_cuda<'):
        return 'gdn_core'
    if name.startswith('void k_get_rows_float_vec<'):
        return 'indexed_float_gather'
    if name.startswith('Cijk_'):
        return 'blas_gemm'
    return 'other'


def analyze(trace):
    with trace.open(newline='') as f:
        rows = list(csv.DictReader(f))
    heads = [i for i, row in enumerate(rows) if group(row) == 'full_vocabulary_head']
    if len(heads) != 3 or heads[-1] != len(rows) - 1:
        raise ValueError(f'expected seed + 2 generated full heads ending trace, found {heads}')
    steps = []
    for left, right in zip(heads, heads[1:]):
        buckets = defaultdict(lambda: {'calls': 0, 'ms': 0.})
        gather_shapes = defaultdict(lambda: {'calls': 0, 'ms': 0.})
        for row in rows[left + 1:right + 1]:
            category = group(row)
            duration = (int(row['End_Timestamp']) - int(row['Start_Timestamp'])) / 1e6
            if duration < 0:
                raise ValueError('negative kernel duration')
            buckets[category]['calls'] += 1
            buckets[category]['ms'] += duration
            if category == 'indexed_float_gather':
                shape = (row['Grid_Size_X'], row['Grid_Size_Y'], row['Workgroup_Size_X'])
                gather_shapes[shape]['calls'] += 1
                gather_shapes[shape]['ms'] += duration
        if {shape: value['calls'] for shape, value in gather_shapes.items()} != {
            ('8192', '24', '256'): 30, ('8192', '512', '256'): 30}:
            raise ValueError(f'unexpected indexed gather geometry: {gather_shapes}')
        expected =  {'nonexpert_q8': 250, 'routed_experts': 120,
                    'full_vocabulary_head': 1, 'gdn_core': 30,
                    'indexed_float_gather': 60, 'blas_gemm': 100}
        for name, count in expected.items():
            if buckets[name]['calls'] != count:
                raise ValueError(f'{name}: expected {count}, got {buckets[name]["calls"]}')
        steps.append({'first_dispatch_index': left + 1, 'last_dispatch_index': right,
                      'summed_device_ms': round(sum(b['ms'] for b in buckets.values()), 6),
                      'families': {name: {'calls': entry['calls'], 'device_ms': round(entry['ms'], 6)}
                                   for name, entry in sorted(buckets.items())},
                      'indexed_gathers_by_grid_y': {
                          y: round(gather_shapes[('8192', y, '256')]['ms'], 6) for y in ('24', '512')}})
    return {'csv_sha256': sha(trace), 'dispatches_including_seed': len(rows),
            'head_indices': heads, 'generated_steps': steps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path, help='directory containing control and selected traces')
    args = parser.parse_args()
    receipt = {'analyzer_sha256': sha(Path(__file__)),
               'model_sha256_from_pinned_acquisition': 'ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61',
               'acquisition_receipt_sha256': sha(Path('../../data/qwen-moe/acquisition.json')),
               'gpu_wrapper_log_sha256': sha(args.data / 'panel-wrapper.log'),
               'contract': 'installed selected 3f552c2 Qwen; 32 independent natural-text eight-token seeds, then two generated greedy steps, 32 full vocabulary heads/step; J32 forced control versus J16 selected; kernel-trace device durations, not wall time or DRAM transactions'}
    for arm in ('control', 'selected'):
        base = args.data / arm
        entry = analyze(base / 'profile_kernel_trace.csv')
        if (base / 'rows.f32').stat().st_size != 96 * 248320 * 4:
            raise ValueError('incomplete full-vocabulary logits')
        entry['logits_sha256'] = sha(base / 'rows.f32')
        entry['probe_sha256'] = sha(Path('../../data/qwen-moe/generated-stream-j16/probe'))
        entry['probe_source_sha256'] = sha(Path(__file__).with_name('generated_stream_j16.cpp'))
        entry['hip_library_sha256'] = sha(Path('../../data/qwen-moe/runtime/current/bin/libggml-hip.so.0.21.0'))
        entry['stdout_sha256'] = sha(base / 'stdout')
        entry['stderr_sha256'] = sha(base / 'stderr')
        receipt[arm] = entry
    if receipt['control']['logits_sha256'] != receipt['selected']['logits_sha256']:
        raise ValueError('full-vocabulary output bits differ')
    receipt['paired_logits_equal'] = True
    (args.data / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({arm: receipt[arm]['generated_steps'] for arm in ('control', 'selected')}, indent=2))


if __name__ == '__main__':
    main()

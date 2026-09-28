#!/usr/bin/env python3
"""Price the existing MoE score/sum dispatch and compare FP32 reduction orders."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

TRACE = Path('../../data/qwen-moe/q8-unprofiled/trace/gpu-host/1886571_kernel_trace.csv')
CAPTURE = Path('../../data/qwen-moe/route-capture')
SOURCE = Path('../../data/qwen-moe/runtime/current/source.bundle')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def family(name):
    if 'k_bin_bcast<&(op_add(float, float))' in name:
        return 'add_' + str(name.count('float const*'))
    if 'k_bin_bcast<&(op_mul(float, float))' in name:
        return 'mul_' + str(name.count('float const*'))
    return None


def trace_result(path):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    heads = [i for i, row in enumerate(rows) if
             'mul_mat_vec_q<(ggml_type)14,' in row['Kernel_Name'] and
             int(row['Grid_Size_X']) == 7946240]
    if len(heads) != 10:
        raise ValueError(f'expected two setup and eight decode heads, got {len(heads)}')
    spans = [rows[a + 1:b + 1] for a, b in zip(heads[-9:-1], heads[-8:])]
    if any(len(span) != 1565 for span in spans):
        raise ValueError('unexpected decode dispatch count')
    per_token = []
    for span in spans:
        down_count = sum(1 for row in span if
                         ('mul_mat_vec_q<(ggml_type)13,' in row['Kernel_Name'] or
                          'mul_mat_vec_q<(ggml_type)14,' in row['Kernel_Name']) and
                         int(row['Grid_Size_X']) == 65536)
        if down_count != 40:
            raise ValueError(f'expected forty routed down calls, got {down_count}')
        groups = {}
        for row in span:
            name = row['Kernel_Name']
            key = family(name)
            if key is None:
                continue
            group = groups.setdefault(key, {'calls': 0, 'duration_ns': 0})
            group['calls'] += 1
            group['duration_ns'] += int(row['End_Timestamp']) - int(row['Start_Timestamp'])
        if {key: groups[key]['calls'] for key in ('mul_4', 'add_16', 'add_6')} != {
                'mul_4': 80, 'add_16': 40, 'add_6': 40}:
            raise ValueError(f'unexpected sum kernel shape: {groups}')
        per_token.append({'sum_device_ns': sum(int(r['End_Timestamp']) - int(r['Start_Timestamp']) for r in span),
                          'groups': groups})
    warm = per_token[1:]
    return {'decode_tokens': per_token,
            'warm_mean_ms_per_token': {key: round(sum(t['groups'][key]['duration_ns'] for t in warm) / (7 * 1e6), 6)
                                       for key in ('mul_4', 'add_16', 'add_6')},
            'warm_mean_total_device_ms': round(sum(t['sum_device_ns'] for t in warm) / (7 * 1e6), 6),
            'warm_mean_fused_sum_fraction': sum(t['groups']['add_16']['duration_ns'] for t in warm) /
                                              sum(t['sum_device_ns'] for t in warm)}


def reduction_result(root, split):
    prefix = root / (split + '.0.')
    paths = {'down': Path(str(prefix) + 'ffn_moe_down-0.bin'),
             'scores': Path(str(prefix) + 'ffn_moe_weights_norm-0.bin')}
    down = np.fromfile(paths['down'], dtype='<f4').reshape(-1, 8, 2048)
    scores = np.fromfile(paths['scores'], dtype='<f4').reshape(-1, 8)
    if len(down) != len(scores):
        raise ValueError('capture shape mismatch')
    terms = np.multiply(down, scores[:, :, None], dtype=np.float32)
    left = terms[:, 0].copy()
    for slot in range(1, 8):
        left = np.add(left, terms[:, slot], dtype=np.float32)
    tree = terms.copy()
    while tree.shape[1] > 1:
        tree = np.add(tree[:, ::2, :], tree[:, 1::2, :], dtype=np.float32)
    tree = tree[:, 0, :]
    delta = left.astype(np.float64) - tree.astype(np.float64)
    differs = left.view('<u4') != tree.view('<u4')
    return {'tokens': len(down), 'compared_components': left.size,
            'different_fp32_bits': int(np.count_nonzero(differs)),
            'tokens_with_any_difference': int(np.count_nonzero(np.any(differs, axis=1))),
            'max_absolute_difference': float(np.max(np.abs(delta))),
            'relative_rms': float(np.linalg.norm(delta) / np.linalg.norm(left.astype(np.float64))),
            'capture_sha256': {key: digest(path) for key, path in paths.items()},
            'first_different_token_and_channel': list(map(int, np.argwhere(differs)[0]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, default=TRACE)
    parser.add_argument('--capture', type=Path, default=CAPTURE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {'analysis_source_sha256': digest(Path(__file__)), 'trace_sha256': digest(args.trace),
              'installed_source_bundle_sha256': digest(SOURCE), 'trace': trace_result(args.trace),
              'host_reduction': {split: reduction_result(args.capture, split) for split in ('train', 'held')},
              'contract': 'Trace device time, not unprofiled wall. Host FP32 reduction is not a native GGML bit witness.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'warm': result['trace']['warm_mean_ms_per_token'],
                      'held_different': result['host_reduction']['held']['different_fp32_bits']}))


if __name__ == '__main__':
    main()

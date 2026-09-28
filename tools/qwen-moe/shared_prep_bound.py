#!/usr/bin/env python3
"""Account for the duplicate Q8_1 preparation of routed and shared MoE inputs."""
import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

TRACE = Path('../../data/qwen-moe/q8-unprofiled/trace/gpu-host/1886571_kernel_trace.csv')
REVISION = 'b4c67ced9f6bac6b1661b25714946d999374e06f'
SELECTED = '7861dc746ed49c6bec1aa2c4f4b8f25b1bf59674'
SOURCE_GIT = Path('../../data/qwen-moe/runtime-source.git')
FILES = ('src/models/qwen35moe.cpp', 'src/llama-graph.cpp',
         'ggml/src/ggml-cuda/mmvq.cu', 'ggml/src/ggml-cuda/quantize.cu')


def source_hashes(revision):
    return {name: hashlib.sha256(subprocess.check_output([
        'git', '--git-dir=' + str(SOURCE_GIT), 'show', revision + ':' + name
    ])).hexdigest() for name in FILES}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def group(row):
    name = row['Kernel_Name']
    if name.startswith('quantize_q8_1('):
        return 'quant_' + row['Grid_Size_X']
    if 'mul_mat_vec_q<(ggml_type)' in name:
        quant_type = name.split('ggml_type)')[1].split(',')[0]
        return 'mmvq_' + quant_type + '_' + row['Grid_Size_X']
    if 'topk_moe_cuda<256,' in name:
        return 'topk'
    return 'other'


def duration(row):
    return int(row['End_Timestamp']) - int(row['Start_Timestamp'])


def measure(rows):
    heads = [i for i, row in enumerate(rows) if group(row) == 'mmvq_14_7946240']
    if len(heads) != 10:
        raise ValueError(f'expected two setup and eight decode full-head calls: {len(heads)}')
    spans = [rows[a + 1:b + 1] for a, b in zip(heads[-9:-1], heads[-8:])]
    records = []
    for span in spans:
        if len(span) != 1565:
            raise ValueError(f'wrong decode span: {len(span)}')
        sites = []
        for i, row in enumerate(span):
            if group(row) != 'topk':
                continue
            # The graph evaluates routed gate/up immediately after top-k; shared
            # gate/up follows the routed down product and routed sum.
            routed = next((j for j in range(i + 1, min(i + 6, len(span)))
                           if group(span[j]) == 'mmvq_12_16384'), None)
            if routed is None or group(span[routed - 1]) != 'quant_2048':
                raise ValueError(f'missing routed prepared input near {i}')
            shared = next((j for j in range(routed + 1, min(routed + 13, len(span)))
                           if group(span[j]) == 'mmvq_8_16384'), None)
            if shared is None or group(span[shared - 1]) != 'quant_2048':
                raise ValueError(f'missing shared prepared input near {i}')
            if shared - 1 == routed - 1:
                raise ValueError('expected distinct prep launches')
            sites.append({'topk_dispatch': int(row['Dispatch_Id']),
                          'routed_quant_dispatch': int(span[routed - 1]['Dispatch_Id']),
                          'shared_quant_dispatch': int(span[shared - 1]['Dispatch_Id']),
                          'routed_quant_ns': duration(span[routed - 1]),
                          'shared_quant_ns': duration(span[shared - 1])})
        if len(sites) != 40:
            raise ValueError(f'expected forty independent MoE layers: {len(sites)}')
        records.append({'sites': sites,
                        'all_device_ns': sum(duration(row) for row in span),
                        'duplicate_shared_prep_ns': sum(s['shared_quant_ns'] for s in sites),
                        'routed_prep_ns': sum(s['routed_quant_ns'] for s in sites)})
    warm = records[1:]
    return {'tokens': records,
            'warm_mean_duplicate_prep_ms': sum(t['duplicate_shared_prep_ns'] for t in warm) / 7e6,
            'warm_mean_routed_prep_ms': sum(t['routed_prep_ns'] for t in warm) / 7e6,
            'warm_mean_all_device_ms': sum(t['all_device_ns'] for t in warm) / 7e6,
            'warm_fraction_duplicate_prep': sum(t['duplicate_shared_prep_ns'] for t in warm) / sum(t['all_device_ns'] for t in warm),
            'logical_duplicate_q8_1_bytes_per_token': 40 * 2048 // 32 * 36}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', type=Path, default=TRACE)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    with a.trace.open(newline='') as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: int(row['Dispatch_Id']))
    result = {'trace_sha256': sha(a.trace), 'source_sha256': sha(Path(__file__)),
              'trace_runtime_revision': REVISION, 'selected_runtime_revision': SELECTED,
              'source_sha256_by_revision': {rev: source_hashes(rev) for rev in (REVISION, SELECTED)},
              'analysis': measure(rows),
              'contract': 'Perfect free reuse removes one of two Q8_1 prep launches per MoE layer on single-row MMVQ. This is a profiled summed-device-time ceiling, not whole-model wall TPS or exact installed-logit acceptance. MMQ prompt and group dispatch differ.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result['analysis'].items() if k != 'tokens'}, indent=2))


if __name__ == '__main__':
    main()

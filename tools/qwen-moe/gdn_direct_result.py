#!/usr/bin/env python3
"""Reproduce the Qwen 32-stream split indexed-state comparison from raw traces."""
import csv
import hashlib
import json
from pathlib import Path

DATA = Path('../../data/qwen-moe/gdn-direct')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(4 << 20), b''):
            h.update(part)
    return h.hexdigest()


def generated_steps(arm):
    path = next((DATA / f'phase-{arm}').rglob('*kernel_trace.csv'))
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    heads = [i for i, row in enumerate(rows)
             if '(ggml_type)14,' in row['Kernel_Name'] and row['Grid_Size_X'] == '124160']
    assert len(heads) == 3 and heads[-1] == len(rows) - 1, heads
    result = []
    for left, right in zip(heads, heads[1:]):
        buckets = {k: {'calls': 0, 'ms': 0.0} for k in ('gdn', 'state_gather', 'conv_gather', 'set_rows', 'other')}
        for row in rows[left + 1:right + 1]:
            name = row['Kernel_Name']
            if name.startswith('void gated_delta_net_cuda<'):
                kind = 'gdn'
            elif name.startswith('void k_get_rows_float_vec<'):
                kind = 'state_gather' if row['Grid_Size_Y'] == '512' else 'conv_gather'
            elif 'set_rows' in name:
                kind = 'set_rows'
            else:
                kind = 'other'
            entry = buckets[kind]
            entry['calls'] += 1
            entry['ms'] += (int(row['End_Timestamp']) - int(row['Start_Timestamp'])) / 1e6
        assert buckets['gdn']['calls'] == 30 and buckets['conv_gather']['calls'] == 30
        assert buckets['state_gather']['calls'] == (30 if arm == 'control' else 0)
        assert buckets['set_rows']['calls'] == (20 if arm == 'control' else 50)
        result.append({k: {'calls': v['calls'], 'device_ms': round(v['ms'], 6)} for k, v in buckets.items()})
    return {'csv_sha256': digest(path), 'head_indices': heads, 'generated_steps': result}


def wall(arm):
    out = []
    for i in (1, 2):
        path = DATA / f'warm-{arm}-{i}.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.startswith('{')]
        assert len(rows) == 128
        steps = [rows[j * 32]['milliseconds'] for j in range(4)]
        assert all(rows[j * 32 + k]['milliseconds'] == steps[j] for j in range(4) for k in range(32))
        logits = DATA / f'warm-{arm}-{i}.f32'
        assert logits.stat().st_size == 128 * 248320 * 4
        out.append({'jsonl_sha256': digest(path), 'logits_sha256': digest(logits),
                    'step_ms': steps, 'generated_aggregate_tps': 96 / (sum(steps[1:]) / 1000)})
    return out


def main():
    phase = {arm: generated_steps(arm) for arm in ('control', 'direct')}
    for arm in phase:
        phase[arm]['logits_sha256'] = digest(DATA / f'phase-{arm}/rows.f32')
    comparisons = {arm: wall(arm) for arm in ('control', 'direct')}
    assert len({entry['logits_sha256'] for entries in comparisons.values() for entry in entries}) == 1
    assert phase['control']['logits_sha256'] == phase['direct']['logits_sha256']
    result = {
        'contract': 'selected 3f552c2 source plus opt-in patch; actual 32 independent autoregressive streams, 8-token natural prefixes, 32 full vocabulary rows per step, n_rs_seq=0; split GDN indexed read followed by SET_ROWS vs eager gather and fused cache write',
        'model_sha256': 'ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61',
        'source_base': '3f552c2ef572707f05746bdc84c1788792523d62',
        'analyzer_sha256': digest(Path(__file__)),
        'probe_source_sha256': digest(Path(__file__).with_name('generated_stream_j16.cpp')),
        'wrapper_source_sha256': digest(Path(__file__).parents[2] / 'tools/run-batch-compare'),
        'patch_sha256': digest(DATA / 'split-rows.patch'),
        'probe_sha256': digest(DATA / 'probe'),
        'hip_library_sha256': digest(DATA / 'libggml-hip.so'),
        'llama_library_sha256': digest(DATA / 'libllama.so'),
        'log_sha256': {n: digest(DATA / f'{n}.log') for n in ('active-wrapper', 'phase-wrapper', 'warm-wrapper', 'wrapper', 'paired-wrapper', 'paired-wrapper-fixed', 'prof-wrapper', 'trace-wrapper', 'debug-wrapper')},
        'phase': phase, 'wall': comparisons,
        'selected_runtime_changed': False,
    }
    (DATA / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'phase': phase, 'wall_tps': {arm: [round(v['generated_aggregate_tps'], 3) for v in entries]
                                       for arm, entries in comparisons.items()}}, indent=2))


if __name__ == '__main__':
    main()

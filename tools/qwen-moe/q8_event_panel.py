#!/usr/bin/env python3
"""Summarize selected-runtime, no-graph quantized HIP launch event pairs.

An event wraps the original launch in its own stream; this is NOT an unmodified
wall benchmark, a physical DRAM counter, or a measurement of graph replay.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from statistics import median


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def analyze(path):
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert all(int(row['ordinal']) == i and float(row['device_ms']) > 0 for i, row in enumerate(rows))
    def kind(row):
        typ = int(row['name'].split('ggml_type')[1].split('E')[0])
        grid = int(row['grid'])
        if typ == 14 and grid == 248320:
            return 'head'
        return {8: 'Q8', 12: 'Q4', 13: 'Q5', 14: 'Q6'}[typ] + f'-{grid}'
    heads = [i for i, row in enumerate(rows) if kind(row) == 'head']
    # Two setup heads at p32, four at p1024; in both cases the final
    # nine heads are the prompt result and eight generation results.
    assert len(heads) in (11, 13), heads
    tokens = []
    for a, b in zip(heads[-9:-1], heads[-8:]):
        groups = defaultdict(list)
        for row in rows[a+1:b]:
            groups[kind(row)].append(float(row['device_ms']) * 1e3)
        expected = {'Q8-8192': 40, 'Q8-4096': 30, 'Q8-2048': 80,
                    'Q8-512': 60, 'Q4-4096': 40, 'Q5-16384': 37, 'Q6-16384': 3}
        assert {key: len(value) for key, value in groups.items()} == expected, (a,b,{key:len(v) for key,v in groups.items()})
        tokens.append({key: {'count': len(value), 'median_us': median(value),
                             'sum_ms': sum(value)/1000,
                             'over_105us': sum(t > 105 for t in value) if key == 'Q8-8192' else None,
                             'over_80us': sum(t > 80 for t in value) if key == 'Q4-4096' else None}
                       for key, value in sorted(groups.items())})
    warm = tokens[1:]
    def summarize(key):
        return {'calls': sum(t[key]['count'] for t in warm),
                'median_token_median_us': median(t[key]['median_us'] for t in warm),
                'median_token_sum_ms': median(t[key]['sum_ms'] for t in warm),
                'slow_count': sum(t[key]['over_105us'] if key == 'Q8-8192' else t[key]['over_80us'] for t in warm)
                    if key in ('Q8-8192','Q4-4096') else None}
    return {'contract': 'No-graph HIP event pairs bracketing every selected quantized kernel, same stream, no per-launch sync; exactly 8 decode heads, token 0 cold excluded from aggregates. Events perturb wall timing; graph-enabled path cannot be observed by this hook. Device event duration is neither physical DRAM bytes nor original unmodified kernel latency.',
            'csv_sha256': digest(path), 'selected_launches': len(rows), 'head_ordinals': heads,
            'warm': {key: summarize(key) for key in tokens[0]}, 'tokens': tokens}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv', type=Path)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    result = analyze(args.csv)
    root = args.data
    deep = analyze(root/'q8-event-depth1024.csv')
    result['depth1024'] = {'csv_sha256': deep['csv_sha256'],
                            'head_ordinals': deep['head_ordinals'],
                            'selected_launches': deep['selected_launches'],
                            'warm': deep['warm'], 'tokens': deep['tokens']}
    files = ['q8-event-probe.so', 'q8-event-panel.log', 'q8-event-depth1024.log', 'q8-event-control.log',
             'q8-event-graph.f32', 'q8-event-no-graph.f32', 'q8-event-with-markers.f32',
             'q8-event-graph.tokens', 'q8-event-no-graph.tokens', 'q8-event-with-markers.tokens',
             'q8-event-graph.log', 'q8-event-no-graph.log', 'q8-event-with-markers.log']
    result['evidence_sha256'] = {name: digest(root/name) for name in files}
    result['source_sha256'] = {name: digest(Path(__file__).with_name(name))
                               for name in ('q8_event_probe.cpp', 'q8_event_panel.py')}
    result['selected_runtime_sha256'] = digest(Path('../../data/qwen-moe/runtime/current/runtime.json'))
    result['model_acquisition_sha256'] = digest(Path('../../data/qwen-moe/acquisition.json'))
    heads = [result['evidence_sha256'][f'q8-event-{arm}.f32'] for arm in ('graph','no-graph','with-markers')]
    ids = [result['evidence_sha256'][f'q8-event-{arm}.tokens'] for arm in ('graph','no-graph','with-markers')]
    assert len(set(heads)) == len(set(ids)) == 1, 'No-graph and/or instrumented full head differs from selected graph reference'
    result['full_head_bits_equal'] = True
    result['full_head_count'] = 4*248320
    result['control'] = 'Same-binary no-graph uninstrumented and graph-enabled 22-token/4-head acceptance; same-binary eight-token no-graph benchmark with and without event pairs.'
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    for key, value in result['warm'].items(): print('p32', key, value)
    for key, value in result['depth1024']['warm'].items(): print('p1024', key, value)


if __name__ == '__main__':
    main()

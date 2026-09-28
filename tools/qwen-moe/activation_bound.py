#!/usr/bin/env python3
"""Price activation quantizer calls separately from correlated slow matvecs in a Qwen trace."""

import argparse
import csv
import json
import statistics
from pathlib import Path

from decode_trace import family, sha256


TARGETS = {'Q8_0/out8192': 105, 'Q4_K/out512': 80}


def analyze(trace, receipt):
    installed = json.loads(receipt.read_text())
    if installed['files'][str(trace.relative_to(trace.parents[2]))] != sha256(trace):
        raise ValueError('trace hash differs from installed receipt')
    with trace.open(newline='') as source:
        rows = list(csv.DictReader(source))
    heads = [i for i, row in enumerate(rows) if family(row) == 'Q6_K/out248320']
    if len(heads) != 10:
        raise ValueError('expected two setup heads and eight measured decode heads')
    tokens = []
    for token, (start, end) in enumerate(zip(heads[-9:-1], heads[-8:])):
        span = sorted(rows[start + 1:end + 1], key=lambda row: int(row['Start_Timestamp']))
        if len(span) != 1565 or {row['Queue_Id'] for row in span} != {'2'}:
            raise ValueError('unexpected decode dispatch inventory')
        durations = [(int(row['End_Timestamp']) - int(row['Start_Timestamp'])) / 1000 for row in span]
        quantizers = [duration for row, duration in zip(span, durations) if family(row) == 'activation/q8']
        pairs = {kind: [] for kind in TARGETS}
        for idx, (row, duration) in enumerate(zip(span, durations)):
            kind = family(row)
            if kind not in pairs:
                continue
            if idx == 0 or family(span[idx - 1]) != 'activation/q8':
                raise ValueError(f'{kind} has no immediately preceding activation quantizer')
            pairs[kind].append({'matvec_us': duration, 'quantizer_us': durations[idx - 1]})
        if len(quantizers) != 291 or any(len(pairs[kind]) != 40 for kind in TARGETS):
            raise ValueError('quantizer or matvec inventory changed')
        tokens.append({'token': token, 'device_us': sum(durations), 'all_quantizer_us': sum(quantizers),
                       'quantizer_calls': len(quantizers), 'pairs': pairs})
    warm = tokens[1:]
    summary = {}
    for kind, cutoff in TARGETS.items():
        events = [pair for token in warm for pair in token['pairs'][kind]]
        ordinary = [event for event in events if event['matvec_us'] <= cutoff]
        slow = [event for event in events if event['matvec_us'] > cutoff]
        ordinary_matvec = statistics.mean(event['matvec_us'] for event in ordinary)
        ordinary_quantizer = statistics.mean(event['quantizer_us'] for event in ordinary)
        summary[kind] = {
            'ordinary_calls': len(ordinary), 'slow_calls': len(slow),
            'ordinary_matvec_mean_us': ordinary_matvec,
            'ordinary_quantizer_mean_us': ordinary_quantizer,
            'slow_matvec_excess_ms_per_token': sum(event['matvec_us'] - ordinary_matvec for event in slow) / 7000,
            'slow_quantizer_excess_ms_per_token': sum(event['quantizer_us'] - ordinary_quantizer for event in slow) / 7000,
            'all_preceding_quantizer_ms_per_token': sum(event['quantizer_us'] for event in events) / 7000,
        }
    return {
        'contract': 'One counter-free profiled installed decode at depth 1024; tokens 1-7 warm. '
                    'Device durations are additive on the traced single queue, not unprofiled wall-time speedups. '
                    'Slow threshold is descriptive and the ordinary mean is not an attainable replacement.',
        'source_commit': installed['source_commit'], 'model_sha256': installed['model_sha256'],
        'binary_hashes': installed['binary_hashes'], 'trace_sha256': sha256(trace),
        'installed_receipt_sha256': sha256(receipt), 'script_sha256': sha256(Path(__file__)),
        'warm_all_quantizer_ms_per_token': sum(token['all_quantizer_us'] for token in warm) / 7000,
        'warm_all_device_ms_per_token': sum(token['device_us'] for token in warm) / 7000,
        'targets': summary, 'tokens': tokens,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.trace, args.receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'tokens'}, indent=2))

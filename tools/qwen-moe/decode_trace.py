#!/usr/bin/env python3
"""Isolate measured one-row Qwen decode passes from synthetic-depth setup in a HIP trace."""

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

from profile_trace import family as trace_family


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def family(row):
    name = row['Kernel_Name']
    if 'mul_mat_vec_q<(ggml_type)' not in name:
        return trace_family(name)
    quant = name.split('mul_mat_vec_q<(ggml_type)', 1)[1].split(',', 1)[0]
    label = {'8': 'Q8_0', '12': 'Q4_K', '13': 'Q5_K', '14': 'Q6_K'}.get(quant, quant)
    return f'{label}/out{int(row["Grid_Size_X"]) // 32}'


def analyze(path, tokens):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    heads = [i for i, row in enumerate(rows) if family(row) == 'Q6_K/out248320']
    if len(heads) < tokens + 1:
        raise ValueError('need at least one preceding head and all measured heads')
    starts = heads[-tokens - 1:-1]
    ends = heads[-tokens:]
    spans = [rows[a + 1:b + 1] for a, b in zip(starts, ends)]
    if len({len(span) for span in spans}) != 1:
        raise ValueError('measured decode spans differ in dispatch count')
    buckets = collections.defaultdict(lambda: [0, 0.0])
    for span in spans:
        per_token = collections.Counter(family(row) for row in span)
        if {key: per_token[key] for key in ('Q8_0/out512', 'Q8_0/out2048',
                                         'Q8_0/out4096', 'Q8_0/out8192')} != {
            'Q8_0/out512': 60, 'Q8_0/out2048': 80,
            'Q8_0/out4096': 30, 'Q8_0/out8192': 40}:
            raise ValueError(f'Q8 decode call inventory does not match: {per_token}')
        if per_token['Q6_K/out248320'] != 1:
            raise ValueError('a decode span must end at exactly one vocabulary head')
        for row in span:
            bucket = buckets[family(row)]
            bucket[0] += 1
            bucket[1] += (int(row['End_Timestamp']) - int(row['Start_Timestamp'])) / 1e6
    total = sum(bucket[1] for bucket in buckets.values()) / tokens
    return {
        'trace': str(path.resolve()), 'sha256': sha256(path),
        'dispatches_in_trace': len(rows), 'vocabulary_heads_in_trace': len(heads),
        'measured_tokens': tokens, 'measured_dispatches_per_token': len(spans[0]),
        'measured_row_start': starts[0] + 1, 'measured_row_end_inclusive': ends[-1],
        'summed_device_ms_per_token': total,
        'families': {name: {'calls_per_token': count // tokens,
                            'device_ms_per_token': milliseconds / tokens,
                            'fraction_of_device_time': milliseconds / tokens / total}
                     for name, (count, milliseconds) in sorted(buckets.items())},
        'note': 'HIP trace device durations exclude profiler host launch gaps; they are not counter-free wall phase times.'
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--tokens', type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.trace, args.tokens), indent=2))


if __name__ == '__main__':
    main()

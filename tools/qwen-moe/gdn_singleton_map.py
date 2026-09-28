#!/usr/bin/env python3
"""Check the state-row dependency in the captured 31+1 Qwen GDN split.

This checks only cache-row reads/writes, not floating arithmetic or graph execution.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess

LOG = Path('../../data/qwen-moe/gdn-fused/swap-debug-z.stderr')
SOURCE = Path('../../data/qwen-moe/runtime-source.git')
SELECTED = 'build-7861dc746ed49c6bec1aa2c4f4b8f25b1bf59674'
SOURCE_PATHS = ('src/llama-graph.cpp', 'src/models/qwen35moe.cpp',
                'src/models/delta-net-base.cpp', 'ggml/src/ggml-cuda/gated_delta_net.cu')
PATTERN = re.compile(
    r'GDN build: head=(\d+) zero=(-?\d+) n_rs=(\d+) n_seqs=(\d+) '
    r'n_tok=(\d+) selected=(\d+) src0=(-?\d+) src1=(-?\d+)'
)


def read_trace():
    raw = LOG.read_bytes()
    rows = [tuple(map(int, m.groups())) for m in PATTERN.finditer(raw.decode(errors='replace'))]
    # Locate the last split, not a seed or a prompt graph.
    assert rows[-2:] == [(1, -1, 31, 31, 1, 1, 1, 2), (0, -1, 1, 1, 1, 1, 0, -1)], rows[-2:]
    return rows[-2:], hashlib.sha256(raw).hexdigest()


def run_split(calls, width=32):
    assert sorted(r for head, n in calls for r in range(head, head + n)) == list(range(width))
    initial = tuple(range(1000, 1000 + width))
    gathered = list(initial)
    fused = list(initial)
    events = []
    for head, n in calls:
        sources = tuple(range(head, head + n))
        destinations = sources
        # Symbolic non-linear recurrence: old state is recoverable from result.
        expected = [(s * s + 7 * r + 29) for r, s in zip(destinations, [gathered[i] for i in sources])]
        for r, value in zip(destinations, expected):
            gathered[r] = value
        for r in destinations:
            old = fused[r]
            fused[r] = old * old + 7 * r + 29
        events.append({'head': head, 'rows': n, 'read': list(sources), 'write': list(destinations),
                       'read_write_same_row': sources == destinations})
    assert gathered == fused
    return {'initial': list(initial), 'final_sha256': hashlib.sha256(json.dumps(fused).encode()).hexdigest(),
            'events': events, 'equal': gathered == fused}


def full_head_diffs():
    import numpy as np
    root = LOG.parent
    a_path, b_path = root / 'swap-control.f32', root / 'swap-debug-z.f32'
    a = np.memmap(a_path, dtype='<u4')
    b = np.memmap(b_path, dtype='<u4')
    assert len(a) == len(b) == 96 * 248320
    diffs = [int(np.count_nonzero(a[i:i+248320] != b[i:i+248320]))
             for i in range(0, len(a), 248320)]
    assert diffs == [0] * 64 + [248320] + [0] * 31
    return {'control_sha256': hashlib.sha256(a_path.read_bytes()).hexdigest(),
            'candidate_sha256': hashlib.sha256(b_path.read_bytes()).hexdigest(),
            'different_full_head_rows': {64: diffs[64]}, 'requested_heads': 96,
            'bits_per_head': 248320 * 32}


def main():
    rows, trace_hash = read_trace()
    calls = [(head, n) for head, zero, n_rs, n, ntok, selected, src0, src1 in rows]
    assert all(zero == -1 and n_rs == n and ntok == 1 and selected == 1 and src0 == head
               for head, zero, n_rs, n, ntok, selected, src0, src1 in rows)
    result = run_split(calls)
    # A source-identity condition alone is NOT sufficient if a zero/extra-state
    # operation or another call overwrites an as-yet-unread source.
    disjoint = set(result['events'][0]['write']).isdisjoint(result['events'][1]['read'])
    assert disjoint and result['equal']
    source_hashes = {p: hashlib.sha256(subprocess.check_output(
        ['git', '-C', str(SOURCE), 'show', f'{SELECTED}:{p}'])).hexdigest() for p in SOURCE_PATHS}
    result.update({'failed_full_heads': full_head_diffs(),
                   'trace_sha256': trace_hash, 'selected_source': SELECTED,
                   'selected_source_sha256': source_hashes, 'captured_rows': [list(r) for r in rows],
                   'across_calls_disjoint': disjoint,
                   'state_shard_floats': 524288, 'layers': 30,
                   'split_shards_bytes': [31 * 524288 * 4 * 30, 524288 * 4 * 30],
                   'scope': 'stable zero-free main rows only; no floating-point, graph, convolution or model output claim'})
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()

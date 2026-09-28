#!/usr/bin/env python3
"""Count possible expert-image reuse on captured Qwen layer-0 routes.

This is a logical weight-read model, not a native MMQ or DRAM measurement.
"""
import argparse
import hashlib
import json
import math
import struct
from collections import Counter
from pathlib import Path

EXPERTS = 256
ROUTED = 8
TILES = (1, 2, 4, 8, 16, 32)
WINDOWS = (1, 8, 16, 32, 64, 128)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as src:
        for block in iter(lambda: src.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_routes(path):
    blob = path.read_bytes()
    if len(blob) % (4 * ROUTED):
        raise ValueError(f'Incomplete top-8 records in {path}')
    rows = list(struct.iter_unpack('<8i', blob))
    if not rows or any(len(set(row)) != ROUTED or min(row) < 0 or max(row) >= EXPERTS for row in rows):
        raise ValueError(f'Invalid top-8 expert ids in {path}')
    return rows


def window(rows, tile, expert_bytes):
    counts = Counter(e for row in rows for e in row)
    assignments = len(rows) * ROUTED
    loads = sum(math.ceil(n / tile) for n in counts.values())
    # Each routed assignment also needs an output, hence fewer matrix products
    # do not follow from fewer logical image reads.
    return dict(tokens=len(rows), assignments=assignments, distinct=len(counts),
                singleton_experts=sum(n == 1 for n in counts.values()),
                max_occupancy=max(counts.values()), image_reads=loads,
                reuse=round(assignments / loads, 6),
                image_bytes=loads * expert_bytes,
                saved_vs_one_read_per_assignment=(assignments - loads) * expert_bytes)


def profile(rows, expert_bytes):
    result = {}
    for size in WINDOWS:
        groups = [rows[i:i + size] for i in range(0, len(rows), size)]
        result[str(size)] = {}
        for tile in TILES:
            parts = [window(group, tile, expert_bytes) for group in groups]
            loads = sum(p['image_reads'] for p in parts)
            assignments = len(rows) * ROUTED
            result[str(size)][str(tile)] = dict(windows=len(groups), assignments=assignments,
                image_reads=loads, reuse=round(assignments / loads, 6),
                image_bytes=loads * expert_bytes,
                saved_bytes=(assignments - loads) * expert_bytes,
                distinct_per_window=[p['distinct'] for p in parts],
                singleton_experts_per_window=[p['singleton_experts'] for p in parts],
                max_occupancy_per_window=[p['max_occupancy'] for p in parts])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('inventory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    layer = inventory['layers']['0']
    bank = layer['bank_bytes']
    if inventory['expert_count'] != EXPERTS or inventory['experts_per_token'] != ROUTED or bank % EXPERTS:
        raise ValueError('GGUF inventory does not match the route shape')
    expert_bytes = bank // EXPERTS
    streams = {}
    capture_receipt = args.capture / 'receipt.json'
    provenance = json.loads(capture_receipt.read_text())
    for name in ('train', 'held'):
        path = args.capture / f'{name}.0.ffn_moe_topk-0.bin'
        if sha(path) != provenance[name]['tensor_sha256']['ids']:
            raise ValueError(f'Capture hash disagrees with receipt: {path}')
        rows = load_routes(path)
        adjacent = [len(set(a) & set(b)) for a, b in zip(rows, rows[1:])]
        streams[name] = dict(input=str(path), input_sha256=sha(path),
                             token_sha256=provenance[name]['token_sha256'],
                             text_sha256=provenance[name]['text_sha256'], tokens=len(rows),
                             adjacent_shared_experts=Counter(adjacent),
                             whole_window_uniform_distinct_expectation=round(EXPERTS * (1 - (1 - ROUTED / EXPERTS) ** len(rows)), 6),
                             tiles=profile(rows, expert_bytes))
    result = dict(contract='Layer-0 selected GGUF expert bank, one logical full-image read per occupied expert tile; no claim about cache or actual MMQ transactions',
                  script_sha256=sha(Path(__file__)), capture_receipt_sha256=sha(capture_receipt),
                  inventory=str(args.inventory), inventory_sha256=sha(args.inventory),
                  gguf_header_sha256=inventory['header_sha256'], expert_bytes=expert_bytes,
                  expert_tensor_bytes={t['name']: t['bytes'] // EXPERTS for t in layer['tensors']},
                  splits=streams)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    for name, split in streams.items():
        print(name, 'tokens', split['tokens'], 'uniform distinct expected', split['whole_window_uniform_distinct_expectation'])
        for size in (1, 8, 16, 32, 64, 128):
            print('window', size, 'tile 1/4/8/32 loads',
                  [split['tiles'][str(size)][str(t)]['image_reads'] for t in (1, 4, 8, 32)])
    print(args.output)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Finite-capacity full-expert image cache bounds on captured layer-0 routes."""
import argparse
import hashlib
import json
from collections import Counter, OrderedDict, defaultdict, deque
from pathlib import Path

from route_traffic import EXPERTS, ROUTED, load_routes, sha

CAPACITIES = (0, 8, 16, 32, 64)


def flattened(rows):
    return [expert for row in rows for expert in row]


def static_misses(stream, resident):
    return sum(expert not in resident for expert in stream)


def lru_misses(stream, capacity):
    cache = OrderedDict()
    misses = 0
    for expert in stream:
        if expert in cache:
            cache.move_to_end(expert)
        else:
            misses += 1
            if capacity:
                if len(cache) == capacity:
                    cache.popitem(last=False)
                cache[expert] = None
    return misses


def optimal_misses(stream, capacity):
    """Belady's offline optimum, cold initial cache and unit-size images."""
    future = defaultdict(deque)
    for pos, expert in enumerate(stream):
        future[expert].append(pos)
    cache = set()
    misses = 0
    for pos, expert in enumerate(stream):
        assert future[expert].popleft() == pos
        if expert in cache:
            continue
        misses += 1
        if capacity:
            if len(cache) == capacity:
                victim = max(cache, key=lambda e: future[e][0] if future[e] else len(stream))
                cache.remove(victim)
            cache.add(expert)
    return misses


def panel(stream, train_stream, expert_bytes):
    train_counts = Counter(train_stream)
    local_counts = Counter(stream)
    rows = {}
    for capacity in CAPACITIES:
        trained = set(sorted(range(EXPERTS), key=lambda e: (-train_counts[e], e))[:capacity])
        hindsight = set(sorted(range(EXPERTS), key=lambda e: (-local_counts[e], e))[:capacity])
        cases = dict(lru_cold=lru_misses(stream, capacity),
                     belady_cold=optimal_misses(stream, capacity),
                     trained_pinned=static_misses(stream, trained),
                     hindsight_pinned=static_misses(stream, hindsight))
        rows[str(capacity)] = dict(capacity=capacity, footprint_bytes=capacity * expert_bytes,
                                  misses=cases, hits={name: len(stream) - n for name, n in cases.items()},
                                  image_bytes={name: n * expert_bytes for name, n in cases.items()},
                                  trained_experts=sorted(trained))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('inventory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    receipt_path = args.capture / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    bank = inventory['layers']['0']['bank_bytes']
    assert inventory['expert_count'] == EXPERTS and inventory['experts_per_token'] == ROUTED
    assert bank % EXPERTS == 0
    expert_bytes = bank // EXPERTS
    inputs = {}
    for name in ('train', 'held'):
        path = args.capture / f'{name}.0.ffn_moe_topk-0.bin'
        digest = sha(path)
        if digest != receipt[name]['tensor_sha256']['ids']:
            raise ValueError(f'Capture hash mismatch: {path}')
        rows = load_routes(path)
        inputs[name] = dict(path=str(path), sha256=digest, text_sha256=receipt[name]['text_sha256'],
                            token_sha256=receipt[name]['token_sha256'], rows=rows)
    train_stream = flattened(inputs['train']['rows'])
    splits = {}
    for name, record in inputs.items():
        stream = flattened(record.pop('rows'))
        splits[name] = dict(**record, tokens=len(stream) // ROUTED, assignments=len(stream),
                            capacities=panel(stream, train_stream, expert_bytes))
    result = dict(contract='Cold unit-size full-image cache, top-eight rank order, no bypass, no other weights or state in cache; pinned static images preloaded without charging residency',
                  script_sha256=sha(Path(__file__)), loader_sha256=sha(Path(__file__).with_name('route_traffic.py')),
                  capture_receipt_sha256=sha(receipt_path), inventory_sha256=sha(args.inventory),
                  gguf_header_sha256=inventory['header_sha256'], expert_bytes=expert_bytes, splits=splits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    for name, split in splits.items():
        for capacity, row in split['capacities'].items():
            print(name, capacity, row['misses'])
    print(args.output)


if __name__ == '__main__':
    main()

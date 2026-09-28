#!/usr/bin/env python3
"""Exact co-route pair opportunity on the pinned all-layer prompt captures."""
import hashlib
import itertools
import json
import random
import struct
from collections import Counter
from pathlib import Path

import networkx as nx

DATA = Path('../../data/qwen-moe/all-layer-routes')
OUT = DATA / 'route-set-affinity.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def routes(split, layer):
    path = DATA / f'{split}.layer-{layer}.i32'
    raw = path.read_bytes()
    ids = struct.unpack(f'<{len(raw)//4}i', raw)
    assert len(ids) % 8 == 0
    groups = [tuple(sorted(ids[i:i+8])) for i in range(0, len(ids), 8)]
    assert all(len(set(g)) == 8 and all(0 <= x < 256 for x in g) for g in groups)
    return groups


def matching(groups):
    counts = Counter(pair for group in groups for pair in itertools.combinations(group, 2))
    graph = nx.Graph()
    graph.add_nodes_from(range(256))
    graph.add_weighted_edges_from((a, b, n) for (a, b), n in sorted(counts.items()))
    selected = {tuple(sorted(p)) for p in nx.algorithms.matching.max_weight_matching(graph)}
    unmatched = sorted(set(range(256)) - set().union(*(set(p) for p in selected)))
    selected.update(tuple(unmatched[i:i+2]) for i in range(0, len(unmatched), 2))
    assert len(selected) == 128
    return selected


def matched(groups, pairs):
    return sum(sum(a in group and b in group for a, b in pairs) for group in map(set, groups))


def report(train, held, rng):
    pairs = matching(train)
    hindsight = matching(held)
    shuffled = list(range(256))
    rng.shuffle(shuffled)
    random_pairs = {tuple(sorted(shuffled[i:i+2])) for i in range(0, 256, 2)}
    frequent = Counter(x for g in train for x in g)
    popular = sorted(range(256), key=lambda x: (-frequent[x], x))
    popular_pairs = {tuple(sorted(popular[i:i+2])) for i in range(0, 256, 2)}
    first_train = matching(train[:len(train)//2])
    first_held = matching(held[:len(held)//2])
    return {
        'train_pairs': len(pairs), 'train_pair_hits': matched(train, pairs),
        'train_half_to_half_hits': matched(train[len(train)//2:], first_train),
        'held_half_to_half_hits': matched(held[len(held)//2:], first_held),
        'held_half_oracle_hits': matched(held[len(held)//2:], matching(held[len(held)//2:])),
        'train_half_oracle_hits': matched(train[len(train)//2:], matching(train[len(train)//2:])), 
        'held_pair_hits': matched(held, pairs),
        'held_oracle_pair_hits': matched(held, hindsight),
        'held_random_pair_hits': matched(held, random_pairs),
        'held_frequency_pair_hits': matched(held, popular_pairs),
        'held_route_sets': len(set(held)),
        'held_routes_seen_in_train': sum(g in set(train) for g in held),
        'pair_ids': sorted(map(list, pairs)),
    }


def main():
    rng = random.Random(20260923)
    layers = []
    hashes = {}
    for layer in range(40):
        train, held = routes('train', layer), routes('held', layer)
        assert len(train) == 113 and len(held) == 126
        layers.append(report(train, held, rng))
        for split in ('train', 'held'):
            name = f'{split}.layer-{layer}.i32'
            hashes[name] = sha(DATA / name)
    totals = {k: sum(row[k] for row in layers) for k in (
        'train_pair_hits', 'held_pair_hits', 'held_oracle_pair_hits',
        'held_random_pair_hits', 'held_frequency_pair_hits',
        'train_half_to_half_hits', 'held_half_to_half_hits',
        'train_half_oracle_hits', 'held_half_oracle_hits',
        'held_route_sets', 'held_routes_seen_in_train')}
    totals['held_assignments'] = 40 * 126 * 8
    totals['held_pair_slots'] = 40 * 126 * 4
    receipt = {
        'contract': 'Exact static disjoint pair bank per layer; one hit iff both distinct expert IDs occur in a routed eight. No weight, arithmetic or runtime savings assumed.',
        'source_sha256': sha(Path(__file__)),
        'capture_receipt_sha256': sha(DATA / 'receipt.json'),
        'input_sha256': hashes,
        'rng_seed': 20260923,
        'totals': totals,
        'layers': layers,
    }
    OUT.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(totals, indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""CPU-only causal expert-image anticipation on pinned all-layer prompt routes."""
import argparse
import json
from pathlib import Path

from route_locality import EXPERTS, LAYERS, ROUTED, read_split, sha


def rank(scores, n):
    return set(sorted(range(EXPERTS), key=lambda x: (-scores[x], x))[:n])


def panel(train, held, capacity, inventory):
    # Each layer has its own bank. Fit only on the train prompt, independently
    # for each layer; no held frequencies or same-token held routes are visible.
    totals = {key: 0 for key in ('requests', 'static_hits', 'previous_hits',
                                 'transition_hits', 'hindsight_hits', 'full_static',
                                 'full_previous', 'full_transition', 'full_hindsight',
                                 'adjacent_overlap', 'train_adjacent_overlap', 'cold_rows',
                                 'previous_hit_image_bytes', 'routed_request_bytes')} 
    per_layer = []
    for layer, (training, test) in enumerate(zip(train, held)):
        image_bytes = inventory['layers'][str(layer)]['bank_bytes'] // EXPERTS
        assert image_bytes * EXPERTS == inventory['layers'][str(layer)]['bank_bytes']
        freq = [0] * EXPERTS
        trans = [[0] * EXPERTS for _ in range(EXPERTS)]
        for row in training:
            for expert in row:
                freq[int(expert)] += 1
        for prev, curr in zip(training[:-1], training[1:]):
            for source in prev:
                for target in curr:
                    trans[int(source)][int(target)] += 1
        static = rank(freq, capacity)
        hindsight_freq = [0] * EXPERTS
        for row in test[1:]:
            for expert in row:
                hindsight_freq[int(expert)] += 1
        hindsight = rank(hindsight_freq, capacity)
        row_result = {key: 0 for key in totals}
        row_result['train_adjacent_overlap'] = sum(len(set(a) & set(b)) for a, b in zip(training[:-1], training[1:]))
        for prev, curr in zip(test[:-1], test[1:]):
            prev = set(map(int, prev))
            curr = set(map(int, curr))
            keep_prev = prev | rank([freq[x] if x not in prev else 1 << 30 for x in range(EXPERTS)], capacity)
            assert len(keep_prev) == capacity
            # Pooled pair counts; each train transition contributes eight votes
            # per output expert. This is intentionally a simple causal control.
            predicted = rank([sum(trans[source][target] for source in prev) for target in range(EXPERTS)], capacity)
            groups = (('static', static), ('previous', keep_prev),
                      ('transition', predicted), ('hindsight', hindsight))
            row_result['requests'] += ROUTED
            row_result['cold_rows'] += 1
            row_result['adjacent_overlap'] += len(prev & curr)
            row_result['routed_request_bytes'] += ROUTED * image_bytes
            row_result['previous_hit_image_bytes'] += len(prev & curr) * image_bytes
            for name, selected in groups:
                hits = len(selected & curr)
                row_result[name + '_hits'] += hits
                row_result['full_' + name] += hits == ROUTED
        per_layer.append(row_result)
        for key in totals:
            totals[key] += row_result[key]
    return {'totals': totals, 'per_layer': per_layer}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('inventory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    assert inventory['expert_count'] == EXPERTS and inventory['experts_per_token'] == ROUTED
    train, train_hash = read_split(args.capture, 'train')
    held, held_hash = read_split(args.capture, 'held')
    assert len(train) == len(held) == LAYERS
    assert all(len(a) == 113 and len(b) == 126 for a, b in zip(train, held))
    results = {str(n): panel(train, held, n, inventory) for n in (8, 16, 32)}
    receipt = {
        'contract': 'One full packed image per selected expert; eight distinct assignments per row. Train-only static/previous/pooled-transition predictor of next same-layer row; held row 0 excluded. Hindsight pinned per-layer frequencies are a noncausal control. Prediction cannot alter selected arithmetic. Route captures are batched prompts, not serial generated decode.',
        'source_sha256': sha(Path(__file__)), 'capture_sha256': {'train': train_hash, 'held': held_hash},
        'capture_receipt_sha256': sha(args.capture / 'receipt.json'), 'inventory_sha256': sha(args.inventory),
        'model_sha256': json.loads((args.capture.parent / 'acquisition.json').read_text())['sha256'],
        'capacities': results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + '\n')
    for width, result in results.items():
        print(width, result['totals'])
    print(args.output)


if __name__ == '__main__':
    main()

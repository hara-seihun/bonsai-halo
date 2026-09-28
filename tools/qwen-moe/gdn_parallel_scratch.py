#!/usr/bin/env python3
"""Finite parallel-round/snapshot frontier for fixed-destination old-row maps."""
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path('../../data/qwen-moe')


def rounds(destinations, source, saved):
    """Earliest rounds; None means that the remaining dependency graph cycles."""
    pending = set(destinations)
    groups = []
    while pending:
        ready = tuple(sorted(d for d in pending if all(
            q == d or source[q] != d or d in saved or q not in pending
            for q in pending)))
        if not ready:
            return None
        groups.append(ready)
        pending.difference_update(ready)
    return groups


def best(destinations, source):
    options = []
    for size in range(len(destinations) + 1):
        feasible = []
        for saved in itertools.combinations(destinations, size):
            groups = rounds(destinations, source, set(saved))
            if groups is not None:
                feasible.append((len(groups), saved, groups))
        if feasible:
            options.append(min(feasible))
        else:
            options.append(None)
    return options


def transition(value, d, step):
    return (value * 17 + d * 31 + step * 7 + 3) % 1000003


def check(state, source, saved, groups, step):
    old = state[:]
    expected = old[:]
    for d, s in source.items():
        expected[d] = transition(old[s], d, step)
    snapshots = {d: old[d] for d in saved}
    actual = state[:]
    for group in groups:
        simultaneous = actual[:]
        for d in group:
            s = source[d]
            actual[d] = transition(snapshots[s] if s in snapshots else simultaneous[s], d, step)
    assert actual == expected, (source, saved, groups, actual, expected)
    return actual


def main():
    maps = 0
    cyclic = 0
    hist = {}
    witnesses = {}
    for n in range(1, 5):
        for size in range(1, n + 1):
            for destinations in itertools.combinations(range(n), size):
                for assignment in itertools.product(range(n), repeat=size):
                    maps += 1
                    source = dict(zip(destinations, assignment))
                    options = best(destinations, source)
                    cyclic += options[0] is None
                    frontier = tuple(None if o is None else o[0] for o in options)
                    hist[str(frontier)] = hist.get(str(frontier), 0) + 1
                    for option in options:
                        if option is None:
                            continue
                        _, saved, groups = option
                        state = check(list(range(100, 100 + n)), source, saved, groups, 0)
                        reversed_destinations = tuple(reversed(destinations))
                        second = dict(zip(reversed_destinations, reversed(assignment)))
                        second_option = best(reversed_destinations, second)[len(saved)]
                        assert second_option is not None
                        check(state, second, second_option[1], second_option[2], 1)
                    if destinations == tuple(range(n)) and assignment == tuple((i+1) % n for i in range(n)):
                        witnesses[str(n)] = frontier
    assert maps == 696 and cyclic == 208 and witnesses['4'] == (None, 4, 2, 2, 1)
    row_bytes = 524288 * 4
    receipt = {
        'domain': 'fixed-destination atomic whole-row operations; snapshots chosen before all transition rounds, held until completion',
        'maps_checked_two_steps': maps,
        'maps_cyclic_without_snapshot': cyclic,
        'frontier_histogram': hist,
        'cycle_witness_min_rounds_at_snapshot_count': witnesses,
        'cycle_32_min_rounds_at_snapshot_counts': {str(k): (32+k-1)//k for k in (1, 2, 4, 8, 16, 32)},
        'state_row_bytes_per_layer': row_bytes,
        'cycle_32_intermediate_logical_bytes_across_30_layers': {str(k): 2*k*row_bytes*30 for k in (1, 2, 4, 8, 16, 32)},
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'model_sha256': json.loads((ROOT/'acquisition.json').read_text())['sha256'],
        'installed_source_bundle_sha256': hashlib.sha256((ROOT/'runtime/current/source.bundle').read_bytes()).hexdigest(),
        'failed_split_trace_sha256': hashlib.sha256((ROOT/'gdn-fused/trace-swap.stdout').read_bytes()).hexdigest(),
        'failed_split_identity_frontier': [1],
    }
    output = ROOT/'gdn-parallel-scratch/receipt.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'maps': maps, 'cyclic': cyclic, 'cycle4': witnesses['4'], 'frontiers': len(hist)}))


if __name__ == '__main__':
    main()

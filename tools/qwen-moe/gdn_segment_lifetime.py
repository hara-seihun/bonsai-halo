#!/usr/bin/env python3
"""Exhaust two-segment old-snapshot maps with source-owned, relabeled GDN rows."""
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path


def transition(value, destination, segment):
    return (17 * value + 31 * destination + 7 * segment + 3) % 1000003


def run(n, first, source, order):
    second = tuple(i for i in range(n) if i not in first)
    old = {i: 1000 + 101 * i for i in range(n)}
    physical = old.copy()
    labels = dict.fromkeys(range(n))
    exposed = set(first) & {source[d] for d in second}
    ghost = {x: physical[x] for x in exposed}
    for segment, destinations in enumerate((first, second)):
        groups = defaultdict(list)
        for d in destinations:
            groups[source[d]].append(d)
        safe_owners = set(destinations) & set(groups) - (exposed if segment == 0 else set())
        free = sorted(set(destinations) - safe_owners)
        assignment = {}
        for x in sorted(groups):
            for d in groups[x]:
                if x in safe_owners and not any(assignment.get(q) == x for q in groups[x]):
                    assignment[d] = x
                else:
                    assignment[d] = free.pop(0)
        assert not free and len(set(assignment.values())) == len(destinations)
        for x in (sorted(groups) if order == 0 else sorted(groups, reverse=True)):
            ds = [d for d in groups[x] if assignment[d] != x] + [d for d in groups[x] if assignment[d] == x]
            for d in ds:
                value = ghost[x] if x in exposed else physical[x]
                assert value == old[x], (n, first, source, d, x, segment)
                physical[assignment[d]] = transition(value, d, segment)
                labels[d] = assignment[d]
    expected = {d: transition(old[source[d]], d, int(d in second)) for d in range(n)}
    actual = {d: physical[labels[d]] for d in range(n)}
    assert actual == expected and len(set(labels.values())) == n
    return len(exposed)


def main():
    cases = 0
    histogram = defaultdict(int)
    for n in range(2, 5):
        for width in range(1, n):
            for first in itertools.combinations(range(n), width):
                for sources in itertools.product(range(n), repeat=n):
                    source = dict(enumerate(sources))
                    k = run(n, first, source, 0)
                    assert run(n, first, source, 1) == k
                    cases += 1
                    histogram[k] += 1
    path = Path('../../data/qwen-moe/gdn-segment-lifetime/receipt.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    evidence = {
        'cases_two_group_orders': cases, 'exposed_old_row_histogram': dict(sorted(histogram.items())),
        'contract': 'two disjoint segment destination sets, one original old-source snapshot, independently variable full rows, source-owned row-local transition',
        'transition': '(17*old+31*destination+7*segment+3) mod 1000003',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'selected_native_source_sha256': hashlib.sha256(Path('../../data/qwen-moe/runtime/current/source.bundle').read_bytes()).hexdigest(),
        'actual_failed_split': {'first': list(range(1, 32)), 'second': [0],
                                'sources': 'identity', 'exposed_rows': 0,
                                'trace_sha256': json.loads(Path('../../data/qwen-moe/gdn-fused/singleton-map-receipt.json').read_text())['trace_sha256']},
    }
    path.write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps({'receipt': str(path), 'cases': cases, 'histogram': dict(histogram)}))


if __name__ == '__main__':
    main()

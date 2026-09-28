#!/usr/bin/env python3
"""Check a zero-copy GDN state permutation with physical-source destinations.

This is an integer state/metadata proof, not a simulation of GDN floating arithmetic.
The update function depends on the sequence and the entire input row, and is
applied identically by the gathered and source-address schedules.
"""
import hashlib
import itertools
import json


def update(i, row):
    return tuple(((x * 1103515245 + i * 12345 + k) & 0xffffffff) for k, x in enumerate(row))


def check(n, routes, order):
    original = {i: (i + 1, i * 17 + 3, i ^ 0x55aa) for i in range(n)}
    reference = dict(original)
    physical = dict(original)
    address = list(range(n))  # logical cache row -> physical cache row
    for route in routes:
        before = dict(reference)
        reference = {i: update(i, before[route[i]]) for i in range(n)}
        source = [address[route[i]] for i in range(n)]
        assert len(set(source)) == n, 'source-address writes require injective sources'
        # No CUDA block ordering assumption: disjoint source/destination pairs.
        for i in order:
            physical[source[i]] = update(i, physical[source[i]])
        address = source
        assert {i: physical[address[i]] for i in range(n)} == reference
    return reference, address


def main():
    checks = 0
    witnesses = []
    h = hashlib.sha256()
    for n in range(1, 6):
        permutations = list(itertools.permutations(range(n)))
        # All two-step source permutations, and every second-step CTA order
        # through four rows. Five rows checks every two-step route at reversed
        # order. First-step order-independence follows from disjoint rows.
        orders = permutations if n <= 4 else [tuple(reversed(range(n)))]
        for first in permutations:
            for second in permutations:
                for order in orders:
                    result = check(n, (first, second), order)
                    h.update(str(result).encode())
                    checks += 1
        swap = (1, 0, *range(2, n)) if n >= 2 else (0,)
        _, mapping = check(n, (swap, swap), tuple(reversed(range(n))))
        witnesses.append({'rows': n, 'route': swap, 'mapping_after_two_swaps': mapping})
    original = {0: (1, 3, 5), 1: (2, 4, 6)}
    fixed = dict(original)
    # CTA 0 reads source 1 and writes destination 0 before CTA 1 reads 0.
    fixed[0] = update(0, fixed[1])
    fixed[1] = update(1, fixed[0])
    expected = {0: update(0, original[1]), 1: update(1, original[0])}
    assert fixed[1] != expected[1]
    print(json.dumps({'checked_two_step_schedules': checks,
                      'digest': h.hexdigest(), 'witnesses': witnesses,
                      'fixed_destination_swap': {'actual': fixed, 'expected': expected},
                      'physical_writes_per_step': 'n', 'state_row_copies_per_step': 0}, indent=2))


if __name__ == '__main__':
    main()

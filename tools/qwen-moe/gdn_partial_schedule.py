#!/usr/bin/env python3
"""Check fixed-destination, serial, partial GDN state schedules against a gathered map."""
import hashlib
import itertools
import json
from pathlib import Path


def cycles(destinations, source):
    active = set(destinations)
    visited = set()
    result = []
    for start in destinations:
        if start in visited:
            continue
        chain = []
        position = {}
        node = start
        while node in active and node not in visited and node not in position:
            position[node] = len(chain)
            chain.append(node)
            node = source[node]
        if node in position:
            ring = chain[position[node]:]
            if len(ring) > 1:
                result.append(tuple(ring))
        visited.update(chain)
    return result


def schedule(destinations, source):
    """Return transactions (save source, then ordered destinations), one reusable row buffer."""
    rings = cycles(destinations, source)
    selected = [min(ring) for ring in rings]
    active = set(destinations)
    pending = set(destinations)
    output = []
    for saved in selected + [None]:
        if saved is not None:
            output.append(('save', saved))
        # A reader q must write before its old source row d is overwritten.
        # Saving d removes all such dependencies from d's pending readers.
        while True:
            ready = sorted(d for d in pending if d == saved or not any(
                q != d and source[q] == d for q in pending))
            if not ready:
                break
            d = ready[0]
            pending.remove(d)
            output.append(('write', d, saved if source[d] == saved else None))
        if saved is not None:
            output.append(('release', saved))
    assert not pending, (destinations, source, pending)
    return output, rings


def transition(value, dest, step):
    return (value * 17 + 31 * dest + 7 * step + 3) % 1000003


def check(state, source, operations, step):
    old = state[:]
    expected = state[:]
    for d, s in source.items():
        expected[d] = transition(old[s], d, step)
    saved = None
    for op, index, *from_save in operations:
        if op == 'save':
            assert saved is None
            saved = (index, state[index])
        elif op == 'write':
            s = source[index]
            if from_save[0] is not None:
                assert saved is not None and from_save[0] == saved[0] == s
                value = saved[1]
            else:
                value = state[s]
            state[index] = transition(value, index, step)
        else:
            assert op == 'release' and saved is not None and saved[0] == index
            saved = None
    assert saved is None and state == expected, (state, expected, source, operations)
    return state


def main():
    histogram = {}
    cases = []
    for n in range(1, 5):
        for size in range(1, n + 1):
            for destinations in itertools.combinations(range(n), size):
                for assignment in itertools.product(range(n), repeat=size):
                    source = dict(zip(destinations, assignment))
                    operations, rings = schedule(destinations, source)
                    before = list(range(100, 100 + n))
                    state = check(before[:], source, operations, 0)
                    histogram[len(rings)] = histogram.get(len(rings), 0) + 1
                    cases.append((n, destinations, assignment, state))
    # Independent second call with an unrelated source map over the carried
    # physical rows, including inactive rows and fanout, without reset.
    for n, destinations, assignment, state in cases:
        next_destinations = tuple(reversed(destinations))
        next_source = dict(zip(next_destinations, reversed(assignment)))
        operations, _ = schedule(next_destinations, next_source)
        check(state, next_source, operations, 1)
    evidence = {
        'checked_one_step_and_two_step_cases': len(cases),
        'nontrivial_cycle_histogram': histogram,
        'row_transition': '(17*old + 31*dest + 7*step + 3) mod 1000003',
        'scope': 'serial atomic row transactions, fixed physical destinations, fully materialized live old rows',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'model_sha256': json.loads(Path('../../data/qwen-moe/acquisition.json').read_text())['sha256'],
        'selected_native_source_sha256': hashlib.sha256(Path('../../data/qwen-moe/runtime/current/source.bundle').read_bytes()).hexdigest(),
        'example_partial_swap': {'destinations': [0, 1], 'source': [1, 0],
                                 'schedule': schedule((0, 1), {0: 1, 1: 0})[0]},
    }
    path = Path('../../data/qwen-moe/gdn-partial-schedule/receipt.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps({'receipt': str(path), 'cases': len(cases), 'histogram': histogram}))


if __name__ == '__main__':
    main()

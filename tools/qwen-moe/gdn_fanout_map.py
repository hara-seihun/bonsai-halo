#!/usr/bin/env python3
"""Check a source-owned in-place recurrent-state schedule with repeated sources."""

import hashlib
import itertools
import json
from pathlib import Path


def transition(row, destination, step):
    # A row-local nonlinear state update; arithmetic is exactly defined on integers.
    return tuple((x * x + 17 * x + (destination + 1) * (j + 3) + step * 19) % 65521
                 for j, x in enumerate(row))


def plan(active, sources, physical):
    """Assign active output slots using only old active slots; group by old source."""
    active = tuple(sorted(active))
    if set(sources) != set(active) or len(sources) != len(active):
        raise ValueError('one source per active destination is required')
    if len(set(physical.values())) != len(physical) or not set(sources.values()) <= set(physical):
        raise ValueError('invalid live physical map or source')
    groups = {s: sorted(d for d in active if sources[d] == s) for s in sorted(set(sources.values()))}
    # No group can write a source of another group. Untouched logical rows stay live.
    unused = iter(sorted(set(active) - set(groups)))
    assigned = {}
    schedule = {}
    for source, destinations in groups.items():
        owner = destinations[0] if source in active else None
        for destination in destinations:
            assigned[destination] = physical[source] if destination == owner else physical[next(unused)]
        # The branch writing its own source row must execute last in a native shard loop.
        schedule[source] = [d for d in destinations if d != owner] + ([owner] if owner is not None else [])
    assert len(assigned) == len(active) and len(set(assigned.values())) == len(active)
    assert set(assigned.values()) == {physical[d] for d in active}
    for source in groups:
        other_writes = {assigned[d] for s in groups if s != source for d in groups[s]}
        assert physical[source] not in other_writes
    return assigned, schedule


def gathered(state, physical, active, sources, step):
    old = {logical: state[slot] for logical, slot in physical.items()}
    return {**{d: old[d] for d in physical if d not in active},
            **{d: transition(old[sources[d]], d, step) for d in active}}


def execute(state, physical, active, sources, step, order):
    next_map, schedule = plan(active, sources, physical)
    next_state = state.copy()
    for source in order:
        # Native lowering loads a shard fragment once, evaluates each dependent
        # transition in original FP32 order, and stores each output fragment.
        fragment = state[physical[source]]
        for destination in schedule[source]:
            next_state[next_map[destination]] = transition(fragment, destination, step)
    next_map = {**{d: physical[d] for d in physical if d not in active}, **next_map}
    return next_state, next_map


def run():
    cases = schedules = 0
    fanout_cases = external_cases = 0
    digest = hashlib.sha256()
    # Exhaustive arbitrary old rows, active sets and source maps through four live rows;
    # all group orders. Run two steps so carried physical labels are exercised.
    for live in range(1, 5):
        logical = tuple(range(live))
        for active_size in range(1, live + 1):
            for active in itertools.combinations(logical, active_size):
                for choices in itertools.product(logical, repeat=active_size):
                    sources = dict(zip(active, choices))
                    groups = tuple(sorted(set(choices)))
                    cases += 1
                    fanout_cases += len(groups) < active_size
                    external_cases += bool(set(groups) - set(active))
                    for order in itertools.permutations(groups):
                        physical = {i: (i * 3 + 1) % live for i in logical} if live != 3 else {i: (i + 1) % live for i in logical}
                        # Above affine map is a permutation for live=1,2,4.
                        state = {physical[i]: (i + 2, 2 * i + 5, 3 * i + 7) for i in logical}
                        reference = gathered(state, physical, active, sources, 1)
                        state, physical = execute(state, physical, active, sources, 1, order)
                        actual = {d: state[physical[d]] for d in logical}
                        assert actual == reference
                        # A second call exercises a different active set and repeated
                        # sources under labels carried from the first call.
                        second_active = tuple(logical[:active_size])
                        second_sources = {d: logical[(d + active_size) % live] for d in second_active}
                        next_reference = gathered(state, physical, second_active, second_sources, 2)
                        second_order = tuple(reversed(sorted(set(second_sources.values()))))
                        state, physical = execute(state, physical, second_active, second_sources, 2, second_order)
                        assert {d: state[physical[d]] for d in logical} == next_reference
                        schedules += 1
                        digest.update(bytes([live, active_size, *active, *choices, *order]))
    # Fixed-destination parallel writers are unsafe even for a bijective swap.
    old = {0: (3,), 1: (7,)}
    fixed = old.copy()
    fixed[0] = transition(fixed[1], 0, 1)
    fixed[1] = transition(fixed[0], 1, 1)
    assert fixed[1] != transition(old[0], 1, 1)
    # Independently scheduled fanout writers are unsafe if a source is overwritten.
    old = {0: (3,), 1: (7,)}
    naive = old.copy()
    naive[0] = transition(naive[0], 0, 1)
    naive[1] = transition(naive[0], 1, 1)
    assert naive[1] != transition(old[0], 1, 1)
    return {'cases': cases, 'two_step_schedules': schedules, 'fanout_cases': fanout_cases,
            'external_source_cases': external_cases, 'digest_sha256': digest.hexdigest(),
            'fixed_destination_swap_hazard': True, 'independent_fanout_hazard': True}


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))

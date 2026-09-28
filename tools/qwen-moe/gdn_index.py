#!/usr/bin/env python3
"""Exact row-level dependency certificate for a direct indexed Qwen GDN consumer.

The payload is an opaque row label: the GDN arithmetic is deliberately unchanged.
A CTA reads its complete input row before writing its output row, but CTAs can
execute in any order. The replay checks all CTA orders on bounded cache maps.
"""

import argparse
import itertools
import json


def hazard_rows(main, head):
    """Rows requiring a pre-GDN snapshot under arbitrary inter-CTA ordering."""
    writer = {head + i: i for i in range(len(main))}
    return sorted({row for i, row in enumerate(main)
                   if row in writer and writer[row] != i})


def replay(main, extra, head, order, staged=()):
    n = len(main)
    size = max(main + extra + [head + n + len(extra)]) + 1
    state = list(range(size))
    # The production zero kernel is upstream of both gathers; start after it.
    old_main = [state[r] for r in main]
    old_extra = [state[r] for r in extra]
    # Extra output slots are disjoint from the main output slots.
    reference = state.copy()
    for i, value in enumerate(old_extra):
        reference[head + n + i] = value
    for i, value in enumerate(old_main):
        reference[head + i] = (i, value)
    stage = {row: state[row] for row in staged}
    candidate = state.copy()
    for i in order:
        src = main[i]
        value = stage[src] if src in stage else candidate[src]
        candidate[head + i] = (i, value)
    # Extra rows must have been captured before GDN; only their writes move.
    for i, value in enumerate(old_extra):
        candidate[head + n + i] = value
    return candidate == reference


def certify(max_rows):
    cases = 0
    unsafe = 0
    max_hazards = 0
    counterexample = None
    # Every input can read any of the active source/destination rows, including
    # main and extra source overlap; no uniqueness assumption on the reads.
    for n in range(1, max_rows + 1):
        head = 1
        for n_extra in range(2):
            rows = range(head + n + n_extra + 1)
            for main in itertools.product(rows, repeat=n):
                for extra in itertools.product(rows, repeat=n_extra):
                    main, extra = list(main), list(extra)
                    hazards = hazard_rows(main, head)
                    max_hazards = max(max_hazards, len(hazards))
                    for order in itertools.permutations(range(n)):
                        cases += 1
                        assert replay(main, extra, head, order, hazards)
                        without = replay(main, extra, head, order)
                        if not without:
                            unsafe += 1
                            if counterexample is None:
                                counterexample = dict(n=n, main=main, extra=extra,
                                                      head=head, order=order, staged=hazards)
                    # Each *distinct* hazardous row is individually necessary:
                    # the writer can run before one of its other readers.
                    for row in hazards:
                        assert any(not replay(main, extra, head, order, set(hazards) - {row})
                                   for order in itertools.permutations(range(n)))
    return dict(max_rows=max_rows, schedules=cases, unsafe_without_stage=unsafe,
                maximum_staged_rows=max_hazards, first_counterexample=counterexample)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--max-rows', type=int, default=3)
    p.add_argument('--main', type=int, nargs='+', help='Actual source cache indices')
    p.add_argument('--extra', type=int, nargs='*', default=[])
    p.add_argument('--head', type=int, default=0)
    args = p.parse_args()
    if args.main is None:
        print(json.dumps(certify(args.max_rows), indent=2))
    else:
        hazards = hazard_rows(args.main, args.head)
        print(json.dumps({'head': args.head, 'main': args.main, 'extra': args.extra,
                          'safe_without_main_staging': not hazards,
                          'stage_main_rows': hazards,
                          'stage_extra_rows_before_gdn': sorted(set(args.extra)),
                          'main_stage_bytes_per_layer': len(hazards) * 524288 * 4,
                          'state_row_bytes': 524288 * 4}, indent=2))

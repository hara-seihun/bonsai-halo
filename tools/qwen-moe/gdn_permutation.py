#!/usr/bin/env python3
"""Replay the selected recurrent allocator's row metadata for stable/changed batches.

This is deliberately a metadata model, not a GDN numerical implementation. The
transition mirrors find_slot's exclusive-tail reorder, src0 assignment, and
src=i reset; the observation is the source index passed to s_copy_main.
"""

import argparse
import hashlib
import json
from pathlib import Path


def step(cells, order):
    assert len(set(order)) == len(order)
    assert all(seq in cells for seq in order)
    positions = {seq: cells.index(seq) for seq in order}
    head = min(positions.values())
    assert head + len(order) <= len(cells)
    for i, seq in enumerate(order):
        dst = head + i
        src = positions[seq]
        cells[dst], cells[src] = cells[src], cells[dst]
        positions = {s: cells.index(s) for s in order}
    assert cells[head:head + len(order)] == order
    return head


def observe(source, cells, order, head):
    main = [source[cells[head + i]] for i in range(len(order))]
    writer = {head + i: i for i in range(len(order))}
    hazards = sorted({row for i, row in enumerate(main)
                      if row in writer and writer[row] != i})
    return {'head': head, 'main': main, 'hazards': hazards}


def run(n):
    cells = list(range(n))
    # A fresh multi-sequence seed sets src0 to the zero row and src=i on
    # completion; only the latter participates in subsequent generated steps.
    sources = {i: i for i in cells}
    order = list(range(n))
    seed_head = step(cells, order)
    seed = observe(sources, cells, order, seed_head)
    generated = []
    for _ in range(3):
        head = step(cells, order)
        generated.append(observe(sources, cells, order, head))
        sources = {seq: cells.index(seq) for seq in cells}
    swapped = order.copy()
    swapped[0], swapped[1] = swapped[1], swapped[0]
    head = step(cells, swapped)
    changed = observe(sources, cells, swapped, head)
    assert all(not record['hazards'] and record['main'] == list(range(n))
               for record in generated)
    assert changed['hazards'] == [0, 1]
    return {'rows': n, 'seed_source_after_reset': seed,
            'unchanged_generated': generated, 'reordered_generated': changed}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows', type=int, default=32)
    args = p.parse_args()
    if args.rows < 2:
        p.error('at least two rows required')
    result = run(args.rows)
    result['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(json.dumps(result, indent=2))

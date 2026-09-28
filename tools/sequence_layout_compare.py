#!/usr/bin/env python3
"""Compare complete logits and continuations from matched sequence schedules."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(left, right, allow_identical=False):
    a = json.loads((left / 'run.json').read_text())
    b = json.loads((right / 'run.json').read_text())
    for field in ('model', 'context', 'input_sha256', 'quality_inputs', 'quality_rows', 'quality_shapes', 'modes', 'sequence_operand'):
        if a.get(field) != b.get(field):
            raise ValueError(f'unmatched input or arithmetic field: {field}')
    # Route selection is environment driven and read once per process, so the two arms of an on/off
    # comparison are different processes with different HALO_* sets. Two arms that share their
    # flags, their executable and their revision cannot be an implementation comparison at all;
    # they compare a build against itself, and every bit agreeing means nothing. Pass
    # --allow-identical when that is deliberate, as in a run-to-run determinism check.
    # Runs recorded before halo_env existed cannot be checked this way; say so rather than
    # rejecting evidence that is still perfectly reproducible.
    left_env, right_env = a.get('halo_env'), b.get('halo_env')
    recorded = left_env is not None and right_env is not None
    identical = recorded and (left_env == right_env
                              and a.get('executable_sha256') == b.get('executable_sha256')
                              and a.get('git_revision') == b.get('git_revision'))
    if identical and not allow_identical:
        raise ValueError('both runs used the same executable, revision and HALO_* route flags '
                         f'({left_env}); this compares a build against itself. Pass '
                         '--allow-identical if that is intended.')
    result = {'left': str(left.resolve()), 'right': str(right.resolve()),
              'left_layout': a.get('sequence_layout'), 'right_layout': b.get('sequence_layout'),
              'left_resident': a.get('gdn_resident', False), 'right_resident': b.get('gdn_resident', False),
              'left_halo_env': left_env, 'right_halo_env': right_env,
              'route_env_recorded': recorded,
              'route_env_differs': (left_env != right_env) if recorded else None,
              'identical_build_and_route': identical if recorded else None,
              'same_executable': a.get('executable_sha256') == b.get('executable_sha256'),
              'cases': [], 'passed': True}
    files = sorted(left.glob('logits-*.f32'))
    if not files or {p.name for p in files} != {p.name for p in right.glob('logits-*.f32')}:
        raise ValueError('empty or different logit dump sets')
    for p in files:
        x = np.fromfile(p, dtype=np.float32)
        y = np.fromfile(right / p.name, dtype=np.float32)
        if x.size != y.size or not x.size or x.size % 248320:
            raise ValueError(f'invalid logit shapes: {p.name}')
        finite = np.isfinite(x) & np.isfinite(y)
        different = x.view(np.uint32) != y.view(np.uint32)
        nonfinite = int(np.count_nonzero(~finite))
        mismatch = int(np.count_nonzero(different))
        result['cases'].append({'file': p.name, 'elements': int(x.size), 'bit_mismatches': mismatch,
                                'rows_differing': int(np.count_nonzero(different.reshape(-1, 248320).any(axis=1))),
                                'nonfinite_pairs': nonfinite,
                                'max_abs_diff': float(np.max(np.abs(x[finite] - y[finite]))) if finite.any() else None})
        result['passed'] &= mismatch == 0 and nonfinite == 0
    if a.get('multistep') != b.get('multistep'):
        raise ValueError('unmatched multistep inputs')
    ma = {r['mode']: r for r in a.get('multistep_runs', [])}
    mb = {r['mode']: r for r in b.get('multistep_runs', [])}
    if ma.keys() != mb.keys():
        raise ValueError('different multistep mode sets')
    result['continuations'] = []
    for mode, x in ma.items():
        equal = x['tokens'] == mb[mode]['tokens']
        result['continuations'].append({'mode': mode, 'tokens': sum(map(len, x['tokens'])), 'equal': equal})
        result['passed'] &= equal
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--allow-identical', action='store_true',
                        help='permit two runs with the same build and the same HALO_* route flags')
    args = parser.parse_args()
    result = compare(args.left, args.right, args.allow_identical)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)

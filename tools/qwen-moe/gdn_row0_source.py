#!/usr/bin/env python3
"""Compare no-callback layer-0 input extraction with the unobserved row permutation."""
import hashlib
import json
import mmap
from pathlib import Path

import numpy as np
from gdn_permutation_equiv import states
from gdn_row0_origin import report as r0_report

ROOT = Path('../../data/qwen-moe/gdn-permutation-equiv')
REPO = Path(__file__).resolve().parents[2]
VOCAB = 248320


def digest(path):
    with path.open('rb') as stream:
        return {'bytes': path.stat().st_size, 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}


def head_differences(a, b):
    x = np.memmap(a, dtype='<u4', shape=(3, 32, VOCAB))
    y = np.memmap(b, dtype='<u4', shape=(3, 32, VOCAB))
    return [[int(np.count_nonzero(x[t, i] != y[t, i])) for i in range(32)] for t in range(3)]


def input_differences(step):
    n = 256 if step == 0 else 32
    x = np.memmap(ROOT/f'row0-source-normal.step{step}.layer0.f32', dtype='<u4', shape=(n, 2048))
    y = np.memmap(ROOT/f'row0-source-swap.step{step}.layer0.f32', dtype='<u4', shape=(n, 2048))
    # The exported tensor is in physical row order, before logical-ID head gathering.
    order = [1, 0, *range(2, 32)] if step == 2 else list(range(n))
    return {'physical_row_bits_different': [int(np.count_nonzero(x[i] != y[i])) for i in range(n)],
            'logical_row_bits_different': [int(np.count_nonzero(x[i] != y[order[i]])) for i in range(n)],
            'normal_distinct_row_0_1': int(np.count_nonzero(x[0] != x[1]))}


def state_summary(a, b):
    comparison = states(ROOT/f'{a}.state', ROOT/f'{b}.state')
    return {'metadata_equal': comparison['recurrent_meta_a'] == comparison['recurrent_meta_b'],
            'different_regions': [{'name': part['region'], 'rows': part.get('rows', [])}
                                  for part in comparison['different_regions']]}


def main():
    source = REPO/'tools/qwen-moe/gdn_row0_source.cpp'
    files = {f'{arm}.{suffix}': digest(ROOT/f'{arm}.{suffix}')
             for arm in ('normal-gather', 'swap-gather', 'row0-source-normal', 'row0-source-swap')
             for suffix in ('f32', 'state', 'wrapper.log')}
    files.update({f'row0-source-{arm}.step{step}.layer0.f32': digest(ROOT/f'row0-source-{arm}.step{step}.layer0.f32')
                  for arm in ('normal', 'swap') for step in range(3)})
    receipt = {
        'contract': 'Selected runtime, identical 32 logical IDs and tokens; no eval callback. Only added built-in layer-0-input graph output. All 248320 FP32 logits per logical ID and whole serialized state compared to no-output controls. Extra graph output can perturb the map and is not accepted as transparent until checked.',
        'prior_full_head_state_receipt': digest(ROOT/'receipt.json'),
        'model_upstream_sha256': 'ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61',
        'selected_libllama': digest(Path('../../data/qwen-moe/runtime/current/bin/libllama.so')),
        'source': digest(source), 'analyzer': digest(Path(__file__)),
        'probe': digest(ROOT/'row0-source-probe'), 'files': files,
        'inputs': {str(step): input_differences(step) for step in range(3)},
        'head_against_original': {arm: head_differences(ROOT/f'{arm}-gather.f32', ROOT/f'row0-source-{arm}.f32')
                                  for arm in ('normal', 'swap')},
        'state_against_original': {arm: state_summary(arm+'-gather', 'row0-source-'+arm)
                                   for arm in ('normal', 'swap')},
    }
    with (ROOT/'row0-source-normal.state').open('rb') as fa, (ROOT/'row0-source-swap.state').open('rb') as fb:
        with mmap.mmap(fa.fileno(), 0, access=mmap.ACCESS_READ) as x, mmap.mmap(fb.fileno(), 0, access=mmap.ACCESS_READ) as y:
            receipt['r0_matched_instrumented'] = r0_report(x,y)['recurrent/r/0']
    out = ROOT/'row0-source-receipt.json'
    out.write_text(json.dumps(receipt, indent=2)+'\n')
    print(out)
    print('input step2 logical mismatches', [x for x in receipt['inputs']['2']['logical_row_bits_different'] if x])
    for arm in ('normal','swap'):
        print(arm,'head step2 differences', [(i,n) for i,n in enumerate(receipt['head_against_original'][arm][2]) if n],
              'state regions',len(receipt['state_against_original'][arm]['different_regions']))
    print('row0 new QKV',receipt['r0_matched_instrumented']['row0_current_difference'])


if __name__ == '__main__':
    main()

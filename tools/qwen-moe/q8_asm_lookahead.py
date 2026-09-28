#!/usr/bin/env python3
"""Certify an early-issued Q8 pair in the compiled one-row gfx1151 MMVQ loop."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def kernel(path):
    data = json.loads(path.read_text())
    return next(x for x in data['kernels'] if
                '<(ggml_type)8, 1, false, false, false, false>' in x['demangled'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('candidate', type=Path)
    p.add_argument('baseline', type=Path)
    p.add_argument('patch', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    new, old = kernel(a.candidate), kernel(a.baseline)
    lines = new['selected_loop'].splitlines()
    pairs = [i for i in range(len(lines)-2) if
             'global_load_b64' in lines[i] and 'global_load_b64' in lines[i+1]
             and 'v_dot4_i32_iu8' in lines[i+2]]
    if len(pairs) != 1:
        raise ValueError(f'expected one adjacent two-load/first-dot trio, found {pairs}')
    i = pairs[0]
    if not any('s_waitcnt vmcnt(2)' in line for line in lines[max(0, i-5):i]):
        raise ValueError('current operands not visibly retired before the inline-ISA pair')
    # This compiler places the fold at the top of the cyclic body: after the
    # paired loads and first dot, the back edge reaches the second dot/fold.
    if not any('v_fmac_f32' in line for line in lines[:i]) or not any(
            's_branch' in line for line in lines[i+3:]):
        raise ValueError('missing cyclic second-dot/fold or back edge')
    result = {
        'question': 'Can a concrete gfx1151 Q8 one-row schedule issue the next K packed pair before its current integer dot?',
        'answer_in_compiled_isa': True,
        'scope': 'Q8_0 x Q8_1, ncols_dst=1, one wave/row, non-fused, gfx1151; CPU ISA construction only',
        'candidate_isa_sha256': sha(a.candidate), 'baseline_isa_sha256': sha(a.baseline),
        'patch_sha256': sha(a.patch), 'candidate_object_sha256': new['object_sha256'],
        'baseline_object_sha256': old['object_sha256'],
        'baseline_resources': old['resources'], 'candidate_resources': new['resources'],
        'steady_pair_then_first_dot': [line.strip() for line in lines[i:i+3]],
        'preceding_wait': next(line.strip() for line in reversed(lines[:i]) if 's_waitcnt vmcnt(2)' in line),
        'cyclic_fold_before_pair': next(line.strip() for line in reversed(lines[:i]) if 'v_fmac_f32' in line),
        'back_edge_after_pair': next(line.strip() for line in lines[i+3:] if 's_branch' in line),
        'decision': 'Early issuance is realizable, unlike C++ preload. The candidate is not a selected optimization: it adds branch/exec work and 3 VGPR; measure equal-map full-head decode, prompt and generated batch before adoption.',
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'early_pair': result['steady_pair_then_first_dot'],
                      'resources': [result['baseline_resources'], result['candidate_resources']]}, indent=2))


if __name__ == '__main__':
    main()

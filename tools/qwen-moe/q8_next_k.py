#!/usr/bin/env python3
"""Preserve the compiled one-iteration Q8 lookahead and its source/ISA costs."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def selected(receipt):
    return next(k for k in receipt['kernels'] if '<(ggml_type)8, 1, false, false, false, false>' in k['demangled'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('candidate_isa', type=Path)
    p.add_argument('baseline_isa', type=Path)
    p.add_argument('source', type=Path)
    p.add_argument('patch', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    new = selected(json.loads(a.candidate_isa.read_text()))
    old = selected(json.loads(a.baseline_isa.read_text()))
    lines = new['selected_loop'].splitlines()
    fold = next(i for i, line in enumerate(lines) if 'v_dual_fmac_f32' in line or 'v_fmac_f32' in line)
    # Find the reload belonging to the steady-state next iteration, not the
    # prologue loads. The compiler may freely sink ordinary nonvolatile C++ reads.
    next_load = next(i for i in range(fold + 1, len(lines)) if 'global_load_b64' in lines[i])
    branch = next(i for i in range(next_load + 1, len(lines)) if 's_branch' in lines[i])
    summary = {
        'question': 'Does a C++ one-iteration register lookahead issue next-K weight loads before current-K dot/fold in the one-row Q8 MMVQ?',
        'answer': False,
        'scope': 'gfx1151 selected Q8_0 one-output one-wave no-fusion kernel; compiled C++ candidate, not a runtime timing result',
        'source_sha256': digest(a.source), 'patch_sha256': digest(a.patch),
        'candidate_isa_sha256': digest(a.candidate_isa), 'baseline_isa_sha256': digest(a.baseline_isa),
        'baseline_object_sha256': old['object_sha256'], 'candidate_object_sha256': new['object_sha256'],
        'baseline_resources': old['resources'], 'candidate_resources': new['resources'],
        'baseline_operations': old['operations'], 'candidate_operations': new['operations'],
        'steady_state_order': {
            'current_fold': lines[fold].strip(),
            'following_weight_load': lines[next_load].strip(),
            'following_branch': lines[branch].strip(),
            'current_dot_before_fold': any('v_dot4_i32_iu8' in line for line in lines[:fold]),
        },
        'decision': 'The emitted loop reloads after the preceding fold, not during its dot dependency; do not install this candidate. Test a truly early load schedule and physical traffic before full-model acceptance.',
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary['steady_state_order'], indent=2))


if __name__ == '__main__':
    main()

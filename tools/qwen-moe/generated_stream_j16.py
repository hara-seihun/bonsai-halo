#!/usr/bin/env python3
"""Compare real generated independent-stream decode under native expert J policies.

Run under tools/run-batch-compare --exec. Each arm loads the same installed
model, seeds 32 natural-text prefixes, then feeds back its own greedy tokens.
All 32 complete vocabulary rows per step are compared bitwise.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('probe', type=Path)
    parser.add_argument('model', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--arms', default='control,candidate')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    arms = []
    reference = None
    for idx, name in enumerate(args.arms.split(',')):
        if name not in ('control', 'candidate'):
            raise ValueError(name)
        prefix = args.out / f'{idx}-{name}'
        env = dict(os.environ)
        if name == 'control':
            env['GGML_CUDA_MMQ_EXPERT_J'] = '0'
        else:
            env.pop('GGML_CUDA_MMQ_EXPERT_J', None)
        start = time.monotonic()
        run = subprocess.run([str(args.probe), str(args.model), str(prefix), str(args.steps)],
                             env=env, capture_output=True, text=True)
        prefix.with_suffix('.jsonl').write_text(run.stdout)
        prefix.with_suffix('.stderr').write_text(run.stderr)
        rows = [json.loads(line) for line in run.stdout.splitlines() if line.startswith('{')]
        entry = {'arm': name, 'exit': run.returncode, 'elapsed_s': time.monotonic() - start,
                 'rows': rows, 'stdout_sha256': sha(prefix.with_suffix('.jsonl')),
                 'stderr_sha256': sha(prefix.with_suffix('.stderr'))}
        if run.returncode == 0 and len(rows) == args.steps * 32:
            logit_file = prefix.with_suffix('.f32')
            if logit_file.stat().st_size != len(rows) * rows[0]['vocabulary'] * 4:
                raise ValueError('incomplete logits')
            entry['logits_sha256'] = sha(logit_file)
            entry['logits_bytes'] = logit_file.stat().st_size
            entry['step_milliseconds'] = [rows[s * 32]['milliseconds'] for s in range(args.steps)]
            entry['generated_tokens'] = [row['top'] for row in rows]
            if reference is not None:
                entry['logits_equal_reference'] = entry['logits_sha256'] == reference['logits_sha256']
                entry['tokens_equal_reference'] = entry['generated_tokens'] == reference['generated_tokens']
            else:
                reference = entry
        arms.append(entry)
        print(f'{name} {idx}: {run.returncode}, {len(rows)} rows, {entry.get("step_milliseconds")}', flush=True)
        if run.returncode or len(rows) != args.steps * 32:
            break
    receipt = {'contract': '32 independent eight-token natural-text prefixes; subsequent tokens are each stream\'s previous greedy decision. Every stream\'s complete vocabulary head computed and compared. Step zero is prompt setup, steps 1+ are generated single-token steps. Same installed executable and weights; control forces original global J, candidate uses selected J16 at 32 rows.',
               'source_sha256': sha(__file__),
               'probe_source_sha256': sha(Path(__file__).with_suffix('.cpp')),
               'probe_sha256': sha(args.probe),
               'runtime_hip_sha256': sha('../../data/qwen-moe/runtime/current/bin/libggml-hip.so.0.21.0'),
               'model_path': str(args.model), 'model_bytes': args.model.stat().st_size,
               'steps': args.steps, 'arms': arms}
    (args.out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if len(arms) != len(args.arms.split(',')) or not all(a['exit'] == 0 for a in arms):
        raise ValueError('incomplete arm; inspect raw logs')
    if not all(a.get('logits_equal_reference', True) and a.get('tokens_equal_reference', True) for a in arms):
        raise ValueError('exact output mismatch; inspect saved receipt')


if __name__ == '__main__':
    main()

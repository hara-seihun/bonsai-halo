#!/usr/bin/env python3
"""Run interleaved complete-model short-prompt J-width arms with one GPU reservation.

Call under tools/run-batch-compare --exec. The probe loads one complete model per
arm; its first two same-width evaluations are setup, not steady requests.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('probe', type=Path)
    parser.add_argument('model', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--arms', default='control,candidate,candidate,control')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    arms = []
    n_vocab = None
    ref = {}
    for index, arm in enumerate(args.arms.split(',')):
        if arm not in ('control', 'candidate'):
            raise ValueError(arm)
        prefix = args.out / f'{index}-{arm}'
        env = dict(os.environ)
        if arm == 'control':
            env['GGML_CUDA_MMQ_EXPERT_J'] = '0'
        else:
            env.pop('GGML_CUDA_MMQ_EXPERT_J', None)
        run = subprocess.run([str(args.probe), str(args.model), str(prefix), str(args.repeats), '3'],
                             text=True, capture_output=True, env=env, check=True)
        (prefix.with_suffix('.jsonl')).write_text(run.stdout)
        (prefix.with_suffix('.stderr')).write_text(run.stderr)
        rows = [json.loads(line) for line in run.stdout.splitlines() if line.startswith('{')]
        if len(rows) != args.repeats * 3:
            raise ValueError(f'{arm}: incomplete rows {len(rows)}')
        output = prefix.with_suffix('.f32')
        if n_vocab is None:
            n_vocab = rows[0]['vocabulary']
        if output.stat().st_size != len(rows) * n_vocab * 4:
            raise ValueError(f'{arm}: incomplete logit file')
        digests = []
        with output.open('rb') as f:
            for row in rows:
                if row['vocabulary'] != n_vocab:
                    raise ValueError('vocabulary changed')
                bits = f.read(n_vocab * 4)
                digest = hashlib.sha256(bits).hexdigest()
                width = row['width']
                if width in ref and digest != ref[width]:
                    raise ValueError(f'finite logit bits differ at {arm} width {width}, repeat {row["repeat"]}')
                ref[width] = digest
                digests.append(digest)
        arms.append({'arm': arm, 'rows': rows, 'logits_sha256': sha(output),
                     'jsonl_sha256': sha(prefix.with_suffix('.jsonl')), 'row_digests': digests})
        print(f'{arm} {index} complete', flush=True)
    summary = {}
    for width in (16, 24, 32):
        summary[width] = {}
        for arm in ('control', 'candidate'):
            samples = [row['milliseconds'] for entry in arms if entry['arm'] == arm
                       for row in entry['rows'] if row['width'] == width and row['repeat'] >= 2]
            summary[width][arm] = {'milliseconds': samples, 'median': statistics.median(samples)}
        summary[width]['rate_gain_percent'] = round(100 * (summary[width]['control']['median'] /
                                                           summary[width]['candidate']['median'] - 1), 3)
    receipt = {'source_sha256': sha(Path(__file__)), 'probe_source_sha256': sha(Path(__file__).with_name('short_width_probe.cpp')),
               'probe_sha256': sha(args.probe), 'model_sha256': sha(args.model), 'vocabulary': n_vocab,
               'arms': arms, 'summary': summary, 'complete_logit_bits_equal': True}
    (args.out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

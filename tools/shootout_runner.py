#!/usr/bin/env python3
"""Contract M1 headline sweep. Every GPU payload is admitted by run-batch-compare.

Run one scenario or ALL; each point keeps argv, raw streams and a receipt even on failure.
A quiet-host headline sweep is sequential. Existing peer work is recorded, not hidden.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

DATA = Path('../../data/engine-shootout')
PROMPTS = DATA / 'prompts/qwen36'
CORPUS = DATA / 'corpus/wiki.test.raw'
RECEIPTS = DATA / 'receipts/bonsai-halo/M1'


def cmd(*args):
    return subprocess.run(args, capture_output=True, text=True).stdout.strip()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def run(root, scenario, label, payload, route, env=None, memory=32, seconds=600):
    when = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    directory = RECEIPTS / scenario / when
    directory.mkdir(parents=True)
    executable = Path(payload[0]) if Path(payload[0]).is_absolute() else None
    argv = [str(root / 'tools/run-batch-compare'), '--runtime-max', f'{seconds}s',
            '--memory-gib', str(memory), '--host-reserve-gib', '4', '--exec', *map(str, payload)]
    environment = dict(os.environ)
    environment.update(env or {})
    metadata = {'scenario': scenario, 'label': label, 'argv': argv, 'env_overrides': env or {},
                'numerical_route': route, 'git_revision': cmd('git', '-C', str(root), 'rev-parse', 'HEAD'),
                'git_dirty': cmd('git', '-C', str(root), 'status', '--short'),
                'binary_sha256': sha(executable) if executable and executable.is_file() else None,
                'libllama_sha256': sha(Path('../bonsai-hip/build-hip/bin/libllama.so')),
                'uptime_before': Path('/proc/uptime').read_text().strip(),
                'loadavg_before': Path('/proc/loadavg').read_text().strip(),
                'gpu_jobs_before': [line for line in cmd('ps', '-eo', 'pid,comm').splitlines()
                                    if any(name in line.lower() for name in ('bonsai', 'llama', 'vllm', 'sglang', 'kelana'))],
                'timing_boundary': 'wall clock including host synchronization; decode excludes prefill; S5 includes HTTP, prefill and streaming'}
    start = time.monotonic()
    try:
        finished = subprocess.run(argv, cwd=root, env=environment, capture_output=True, text=True,
                                  timeout=seconds + 180)
        rc, stdout, stderr = finished.returncode, finished.stdout, finished.stderr
    except subprocess.TimeoutExpired as error:
        rc = 124
        stdout = (error.stdout or b'').decode(errors='replace') if isinstance(error.stdout, bytes) else (error.stdout or '')
        stderr = (error.stderr or b'').decode(errors='replace') if isinstance(error.stderr, bytes) else (error.stderr or '')
        stderr += '\nshootout runner timed out\n'
    (directory / 'stdout').write_text(stdout)
    (directory / 'stderr').write_text(stderr)
    metadata.update(returncode=rc, wall_s=time.monotonic() - start,
                    loadavg_after=Path('/proc/loadavg').read_text().strip(),
                    shader_clock_line=[line for line in stderr.splitlines() if 'gpu_clock:' in line],
                    observed=extract(stdout, stderr),
                    stdout='stdout', stderr='stderr')
    (directory / 'receipt.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'{scenario}/{label}: {json.dumps(metadata["observed"])} rc={rc} {directory}', flush=True)
    return directory, metadata


def extract(stdout, stderr):
    result = {}
    for pattern, key in [(r'prefill: (\d+) tokens in ([\d.]+) s \(([\d.]+) tok/s\)', 'prefill'),
                         (r'generated (\d+) tokens in ([\d.]+) s: ([\d.]+) tok/s .*digest (\d+)', 'decode'),
                         (r'batch (\d+): (\d+) tokens in (\d+) steps, ([\d.]+) s: ([\d.]+) tok/s', 'batch'),
                         (r'dflash2: .*prompt \d+ tokens in [\d.]+ s: [\d.]+ tok/s; (\d+) tokens in ([\d.]+) s: ([\d.]+) tok/s; .*digest (\d+)', 'dflash')]:
        match = re.search(pattern, stderr)
        if match:
            result[key] = match.groups()
    for line in stdout.splitlines():
        if line.startswith('{'):
            try:
                result['json'] = json.loads(line)
            except json.JSONDecodeError:
                pass
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('scenario', choices=['S1', 'S2', 'S3', 'S4', 'S5', 'Q1', 'Q2', 'ALL'])
    ap.add_argument('--engine-root', type=Path, default=Path('.'))
    ap.add_argument('--chunks', type=int, default=2, help='fixed Q1/Q2 subset; 0 = all remaining windows')
    ap.add_argument('--start-chunk', type=int, default=0, help='resume a window partition; scored reference output must match the preceding partition')
    ap.add_argument('--only', help='comma-separated point labels for an exploratory subset')
    args = ap.parse_args()
    root = args.engine_root.resolve()
    engine, ppl = root / 'bonsai-halo', root / 'tools/perplexity'
    targets = ['tools/perplexity', 'bonsai-halo'] if args.scenario in ('ALL', 'Q1', 'Q2') else ['bonsai-halo']
    subprocess.run(['make', '-j8', *targets], cwd=root, check=True)
    chosen = set(args.only.split(',')) if args.only else None
    def selected(label):
        return chosen is None or label in chosen
    def do(s, label, payload, route, **kw):
        return run(root, s, label, payload, route, **kw) if selected(label) else (None, None)
    scenarios = ['S1', 'S2', 'S3', 'S4', 'S5', 'Q1', 'Q2'] if args.scenario == 'ALL' else [args.scenario]
    for s in scenarios:
        if s == 'S1':
            for depth in (0, 4096):
                ids = PROMPTS / f'{128 if depth == 0 else depth}.ids.json'
                do(s, f'depth-{depth}', [engine, '--profile', 'max', '--plain', '--prompt-ids', ids,
                     '--context', str(512 if depth == 0 else 4608), '--bench', '-n', '256'], 'exact f32 deployed')
        elif s == 'S2':
            ids = PROMPTS / '128.ids.json'
            for name, extra in [('plain', ['--plain']), ('dflash-q4', [])]:
                do(s, name, [engine, '--profile', 'max', *extra, '--prompt-ids', ids,
                   '--context', '512', '--bench', '-n', '256'], 'exact f32 target + Q4 draft' if extra == [] else 'exact f32')
        elif s == 'S3':
            for size in (512, 4096):
                do(s, str(size), [engine, '--profile', 'max', '--plain', '--prompt-ids', PROMPTS / f'{size}.ids.json',
                     '--context', str(size + 256), '--bench', '-n', '0'], 'exact f32 wide-deployed')
        elif s == 'S4':
            for streams in (1, 8, 32, 64, 128):
                experimental = streams == 128
                env = {'HALO_SNAPSHOT_KV_GB': '0'}
                if streams >= 64:
                    env['HALO_MANAGED_ALLOC'] = '1'
                if experimental:
                    env.update(HALO_GDN_REGION='packed', HALO_MANAGED_ALLOC='1')
                do(s, str(streams), [engine, '--profile', 'max', '--batch', str(streams),
                   '--batch-ids-dir', PROMPTS / 'batch', '--context', '256', '--bench', '-n', '128'],
                   'approx mode19 A4 FFN + A4-both sequence + i8 GDN defer4' +
                   ('; experimental packed state-region stride, mixed acceptance red' if experimental else ''),
                   env=env, memory=50 if experimental else (46 if streams >= 64 else 32), seconds=1000)
        elif s == 'S5':
            for clients in (1, 8, 32):
                do(s, str(clients), ['python3', root / 'tools/served_shootout.py', '--engine', engine,
                   '--prompts', PROMPTS / 'batch', '--clients', str(clients)],
                   'exact f32 wide-deployed; singleton DFlash2 Q4; shared-step plain',
                   memory=36, seconds=950)
        elif s in ('Q1', 'Q2'):
            # A reference is a raw row-major f32 distribution over precisely the scored positions.
            # Store it once for Q2, and preserve the file with its source receipt.
            reference_file = RECEIPTS / f'reference-{args.chunks or "full"}-chunks.f32'
            base = [ppl, '--ppl', CORPUS, '--chunks', str(args.chunks), '--start-chunk', str(args.start_chunk)]
            if s == 'Q1' or not reference_file.exists():
                if selected('exact'):
                    # The reference is written by the GPU process inside admission; the pathname
                    # carries its token count so a different subset cannot silently reuse it.
                    reference_file = RECEIPTS / f'reference-{args.chunks or "full"}-chunks.f32'
                    do(s, 'exact', base + ['--route', 'deployed', '--logits-out', reference_file],
                       'exact f32 deployed', seconds=1450)
            reference_file = RECEIPTS / f'reference-{args.chunks or "full"}-chunks.f32'
            for label, route, env in [('wide-deployed', 'wide-deployed', {}),
                                      ('a4-f32', 'wide-a4', {}),
                                      ('a4-i8', 'wide-a4', {'HALO_GDN_STATE':'i8', 'HALO_GDN_DEFER':'4'})]:
                if selected(label):
                    if not reference_file.is_file():
                        raise RuntimeError('Q2 needs exact reference for the same chunk count; run Q1 exact first')
                    do(s, label, base + ['--route', route, '--reference', reference_file],
                       f'{route} {env.get("HALO_GDN_STATE", "f32")} {env.get("HALO_GDN_DEFER", "0")}',
                       env=env, seconds=1450)


if __name__ == '__main__':
    main()

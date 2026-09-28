#!/usr/bin/env python3
"""Bounded, reproducible llama.cpp HIP measurements on the shared Radeon."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
MODEL = Path('../../data/qwen-moe/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf')
BINARY = Path('../bonsai-hip/build-hip/bin/llama-bench')
SOURCE = Path('../bonsai-hip')
RECEIPTS = Path('../../data/qwen-moe/benchmarks')
EXPECTED_SIZE = 22_134_528_992
EXPECTED_SHA256 = 'ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_state(source):
    manifest = source / 'runtime.json'
    if manifest.is_file():
        installed = json.loads(manifest.read_text())
        return {'revision': installed['source_revision'],
                'source_bundle_sha256': installed['source_bundle_sha256'],
                'runtime_manifest_sha256': sha256(manifest), 'dirty_paths': []}
    def git(*args):
        return subprocess.check_output(['git', '-C', str(source), *args]).strip()
    diff = subprocess.check_output(['git', '-C', str(source), 'diff', 'HEAD', '--binary'])
    return {'revision': git('rev-parse', 'HEAD').decode(),
            'tracked_diff_sha256': hashlib.sha256(diff).hexdigest(),
            'dirty_paths': git('status', '--porcelain=v1').decode().splitlines()}


def runtime_libraries(binary):
    listing = subprocess.check_output(['ldd', str(binary)], text=True)
    libraries = {}
    for line in listing.splitlines():
        if ' => ' not in line:
            continue
        name, rest = line.strip().split(' => ', 1)
        path = Path(rest.split(' ', 1)[0])
        if path.is_file() and path.resolve().parent == binary.resolve().parent:
            libraries[name] = {'path': str(path.resolve()), 'sha256': sha256(path)}
    if not {'libggml-hip.so.0', 'libllama.so.0'}.issubset(libraries):
        raise ValueError('ldd did not resolve the local HIP and llama libraries')
    return libraries


def prepare(args):
    if not args.model.is_file():
        raise ValueError(f'model is not a complete file: {args.model}')
    before = args.model.stat()
    if before.st_size != EXPECTED_SIZE:
        raise ValueError(f'incomplete or unexpected GGUF: {before.st_size} bytes, expected {EXPECTED_SIZE}')
    digest = sha256(args.model)
    after = args.model.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError('model changed during hashing')
    if digest != EXPECTED_SHA256:
        raise ValueError(f'model digest mismatch: {digest}')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {'model': str(args.model.resolve()), 'model_size': before.st_size,
                'model_mtime_ns': before.st_mtime_ns, 'model_inode': before.st_ino,
                'model_sha256': digest, 'source_repository':
                'unsloth/Qwen3.6-35B-A3B-GGUF',
                'source_revision': 'a483e9e6cbd595906af30beda3187c2663a1118c'}
    (args.output / 'model.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


def run(args):
    manifest = json.loads((args.output / 'model.json').read_text())
    stat = args.model.stat()
    identity = (str(args.model.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ino)
    recorded = (manifest['model'], manifest['model_size'], manifest['model_mtime_ns'], manifest['model_inode'])
    if identity != recorded or manifest['model_sha256'] != EXPECTED_SHA256:
        raise ValueError('model identity changed: run prepare again after completing acquisition')
    if args.tokens < 1 or args.tokens > 512 or args.depth < 0 or args.depth > 16384 or args.repetitions < 1 or args.repetitions > 10:
        raise ValueError('tokens 1..512, depth 0..16384, repetitions 1..10 required')
    if args.depth + args.tokens > 16384:
        raise ValueError('occupied context plus measured tokens must not exceed 16384')
    if args.action == 'sample' and args.depth:
        raise ValueError('sample uses a real prompt, not synthetic occupied depth; omit --depth')
    if args.runtime < 10 or args.runtime > 48:
        raise ValueError('runtime must be 10..48 seconds; split longer experiments into calls')
    binary = args.binary if args.action == 'run' else args.binary.with_name('llama-cli')
    if not binary.is_file():
        raise ValueError(f'missing HIP binary: {binary}')
    args.output.mkdir(parents=True, exist_ok=True)
    phase = args.phase if args.action == 'run' else 'sample'
    name = f'{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}-{phase}-d{args.depth}-{uuid.uuid4().hex[:8]}'
    directory = args.output / name
    directory.mkdir()
    cmd = [str(ROOT / 'tools/run-batch-compare'), '--runtime-max', f'{args.runtime}s',
           '--memory-gib', str(args.memory_gib), '--host-reserve-gib', str(args.host_reserve_gib),
           '--exec', str(binary), '-m', str(args.model)]
    if args.action == 'run':
        cmd += ['-o', 'jsonl', '-r', str(args.repetitions), '-d', str(args.depth),
                '-b', str(args.batch), '-ub', str(args.ubatch), '-t', str(args.threads),
                '-ngl', '99', '-fa', args.flash,
                '-p', str(args.tokens if args.phase == 'prompt' else 0),
                '-n', str(args.tokens if args.phase == 'decode' else 0)]
    else:
        cmd += ['-p', args.prompt, '-n', str(args.tokens), '-c', '2048',
                '-t', str(args.threads), '-ngl', '99', '-fa', args.flash,
                '--temp', '0', '--seed', '1', '-st', '--simple-io', '--reasoning', 'off',
                '--no-display-prompt', '--no-warmup']
    receipt = {'phase': phase, 'measured_tokens': args.tokens,
               'occupied_context_tokens': args.depth, 'repetitions': args.repetitions,
               'batch': args.batch, 'ubatch': args.ubatch, 'threads': args.threads,
               'flash_attention': args.flash,
               'runtime_environment': {key: value for key, value in os.environ.items()
                                       if key.startswith(('GGML_', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))},
               'model': manifest, 'binary': str(binary.resolve()),
               'binary_sha256': sha256(binary), 'runtime_libraries': runtime_libraries(binary),
               'llama_cpp_source': source_state(args.source),
               'wrapper': str(ROOT / 'tools/run-batch-compare'),
               'wrapper_sha256': sha256(ROOT / 'tools/run-batch-compare'),
               'wrapper_source_revision': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
               'memory_gib': args.memory_gib, 'host_reserve_gib': args.host_reserve_gib,
               'runtime_max_seconds': args.runtime, 'argv': cmd,
               'note': 'llama-bench uses synthetic tokens. Occupied depth is constructed before each timed region. The prompt and decode rates are separate, not a mixed pp+tg mean.'}
    if args.action == 'sample':
        receipt['prompt'] = args.prompt
        receipt['note'] = 'Greedy real-prompt output sample; not a throughput measurement.'
    started = time.monotonic()
    stdout_path = directory / ('stdout.jsonl' if args.action == 'run' else 'output.txt')
    with stdout_path.open('wb') as out, (directory / 'stderr.log').open('wb') as err:
        result = subprocess.run(cmd, stdout=out, stderr=err, check=False)
    receipt['elapsed_wall_s'] = time.monotonic() - started
    receipt['exit_code'] = result.returncode
    if args.action == 'run':
        try:
            samples = [json.loads(line) for line in stdout_path.read_text().splitlines() if line.strip()]
            receipt['samples'] = [{'avg_ts': row.get('avg_ts'), 'stddev_ts': row.get('stddev_ts'),
                                   'avg_ns': row.get('avg_ns'), 'samples_ns': row.get('samples_ns'),
                                   'samples_ts': row.get('samples_ts'), 'n_prompt': row.get('n_prompt'),
                                   'n_gen': row.get('n_gen'), 'n_depth': row.get('n_depth')}
                                  for row in samples]
        except (ValueError, UnicodeError) as exc:
            receipt['parse_error'] = str(exc)
    (directory / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'directory': str(directory), 'exit_code': result.returncode,
                      'samples': receipt.get('samples', []), 'elapsed_wall_s': receipt['elapsed_wall_s']}))
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'sample'])
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--binary', type=Path, default=BINARY)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=RECEIPTS)
    parser.add_argument('--phase', choices=['prompt', 'decode'], default='decode')
    parser.add_argument('--prompt', default='Answer in one sentence: Why does a compass point north?')
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--depth', type=int, default=0)
    parser.add_argument('--repetitions', type=int, default=2)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--ubatch', type=int, default=256)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--flash', choices=['on', 'off', 'auto'], default='on')
    parser.add_argument('--memory-gib', type=int, default=27)
    parser.add_argument('--host-reserve-gib', type=int, default=4)
    parser.add_argument('--runtime', type=int, default=42)
    args = parser.parse_args()
    try:
        if args.action == 'prepare':
            prepare(args)
            return 0
        return run(args)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f'benchmark: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())

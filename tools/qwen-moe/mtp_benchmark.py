#!/usr/bin/env python3
"""Measure plain and MTP Qwen3.6 completions under Bonsai's GPU reservation."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

from mtp import DATA, NAME, SHA256, TRUNK, TRUNK_SHA256, verify
from benchmark import source_state

Q4_NAME = 'mtp-Qwen3.6-35B-A3B-Q4_K_M.gguf'
Q4_SHA256 = '6ec218ee63c6c2ad94981cbf7dbcec8f019a7744952602507c5861ad1362c79f'
Q4_SIZE = 1_260_898_400

ROOT = Path(__file__).resolve().parents[2]
BINARY = DATA / 'runtime/current/bin/llama-server'
OUTPUT = DATA / 'mtp' / 'benchmarks'
PROMPT = 'Answer in one sentence: Why does a compass point north?'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def libraries(binary):
    listing = subprocess.check_output(['ldd', str(binary)], text=True)
    result = {}
    for line in listing.splitlines():
        if ' => ' not in line:
            continue
        name, rest = line.strip().split(' => ', 1)
        path = Path(rest.split(' ', 1)[0])
        if path.is_file() and (str(path).startswith('../work/clones/') or
                               str(path).startswith(str(DATA / 'runtime'))):
            result[name] = {'path': str(path.resolve()), 'sha256': sha256(path)}
    if not {'libllama.so.0', 'libggml-hip.so.0'} <= result.keys():
        raise ValueError('Missing native HIP runtime libraries in ldd')
    return result


def request(url, payload=None, timeout=2):
    body = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=body,
                                 headers={'Content-Type': 'application/json'} if body is not None else {})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def serve(args):
    binary = Path(args.binary)
    cmd = [str(binary), '-m', str(TRUNK), '-ngl', '99', '-fa', 'on',
           '-c', '2048', '-b', '512', '-ub', '256', '-t', '8',
           '--parallel', '1', '--host', '127.0.0.1', '--port', str(args.port),
           '--no-webui']
    if args.mode == 'mtp':
        cmd += ['--model-draft', str(args.draft_model),
                '--spec-type', 'draft-mtp', '--spec-draft-ngl', '99',
                '--spec-draft-n-max', str(args.draft_n),
                '--spec-draft-p-min', str(args.draft_p)]
    server_log = Path(args.log)
    with server_log.open('wb') as out:
        server = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
        try:
            base = f'http://127.0.0.1:{args.port}'
            deadline = time.monotonic() + args.startup_seconds
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f'llama-server exited {server.returncode}, see {server_log}')
                try:
                    state = request(base + '/health')
                    if state.get('status') == 'ok':
                        break
                except (urllib.error.URLError, TimeoutError, ValueError):
                    pass
                time.sleep(.25)
            else:
                raise TimeoutError(f'llama-server did not become healthy, see {server_log}')
            result = request(base + '/completion', {'prompt': args.prompt,
                             'n_predict': args.tokens, 'temperature': 0,
                             'seed': 1, 'cache_prompt': False, 'stream': False,
                             'return_tokens': True, 'n_probs': 3}, timeout=args.completion_seconds)
            if 'content' not in result or 'timings' not in result:
                raise ValueError(f'Incomplete completion: {result}')
            Path(args.response).write_text(json.dumps(result, indent=2) + '\n')
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


def panel(args):
    if args.mode == 'mtp':
        verify(DATA / 'mtp' / NAME)
        draft = Path(args.draft_model)
        if draft.name == Q4_NAME:
            if draft.stat().st_size != Q4_SIZE or sha256(draft) != Q4_SHA256:
                raise ValueError('Q4 MTP draft does not match its quantization receipt')
        elif draft.name != NAME or draft.resolve() != (DATA / 'mtp' / NAME).resolve():
            raise ValueError(f'Unregistered MTP draft: {draft}')
    if not (DATA / 'benchmarks' / 'model.json').is_file():
        raise ValueError('Run benchmark.py prepare for the pinned trunk first')
    trunk_receipt = json.loads((DATA / 'benchmarks' / 'model.json').read_text())
    stat = TRUNK.stat()
    if trunk_receipt['model_sha256'] != TRUNK_SHA256 or (trunk_receipt['model_size'], trunk_receipt['model_mtime_ns'], trunk_receipt['model_inode']) != (stat.st_size, stat.st_mtime_ns, stat.st_ino):
        raise ValueError('Pinned target identity changed; rerun benchmark.py prepare')
    if args.tokens < 1 or args.tokens > 128:
        raise ValueError('tokens must be 1..128')
    binary = Path(args.binary)
    if not binary.is_file():
        raise ValueError(f'Missing llama-server: {binary}')
    args.output.mkdir(parents=True, exist_ok=True)
    directory = args.output / f'{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}-{args.mode}-{uuid.uuid4().hex[:8]}'
    directory.mkdir()
    response = directory / 'response.json'
    server_log = directory / 'server.log'
    cmd = [str(ROOT / 'tools/run-batch-compare'), '--runtime-max', '48s',
           '--memory-gib', '30', '--host-reserve-gib', '4', '--exec',
           sys.executable, str(Path(__file__).resolve()), 'serve',
           '--mode', args.mode, '--binary', str(binary), '--draft-model', str(args.draft_model), '--prompt', args.prompt,
           '--tokens', str(args.tokens), '--draft-n', str(args.draft_n),
           '--draft-p', str(args.draft_p), '--port', str(args.port),
           '--response', str(response), '--log', str(server_log)]
    runtimes = libraries(binary)
    library = Path(runtimes['libllama.so.0']['path'])
    source = library.parents[1] if (library.parents[1] / 'runtime.json').is_file() else library.parents[2]
    identity = source_state(source)
    meta = {'argv': cmd, 'mode': args.mode, 'prompt': args.prompt, 'requested_tokens': args.tokens,
            'draft_n': args.draft_n if args.mode == 'mtp' else 0,
            'draft_p': args.draft_p if args.mode == 'mtp' else None,
            'trunk': trunk_receipt, 'draft_path': str(Path(args.draft_model).resolve()) if args.mode == 'mtp' else None,
            'draft_sha256': sha256(Path(args.draft_model)) if args.mode == 'mtp' else None,
            'binary': str(binary.resolve()), 'binary_sha256': sha256(binary),
            'runtime_libraries': runtimes,
            'native_revision': identity['revision'],
            'native_source': identity,
            'runtime_environment': {k: v for k, v in os.environ.items()
                                    if k.startswith(('GGML_', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))},
            'wrapper_sha256': sha256(ROOT / 'tools/run-batch-compare')}
    start = time.monotonic()
    with (directory / 'wrapper.log').open('wb') as out:
        proc = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, check=False)
    meta['exit_code'] = proc.returncode
    meta['wall_seconds'] = time.monotonic() - start
    if response.exists():
        result = json.loads(response.read_text())
        meta['content'] = result['content']
        meta['tokens'] = result.get('tokens')
        meta['completion_probabilities'] = result.get('completion_probabilities')
        meta['timings'] = result['timings']
        meta['tokens_predicted'] = result.get('tokens_predicted')
        meta['stop_type'] = result.get('stop_type')
    (directory / 'receipt.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps({'directory': str(directory), 'exit_code': proc.returncode,
                      'timings': meta.get('timings'), 'content': meta.get('content')}, indent=2))
    return proc.returncode


def compare(paths):
    records = [json.loads((p / 'receipt.json').read_text()) for p in paths]
    for row in records:
        if row['exit_code'] != 0 or 'content' not in row:
            raise ValueError('Cannot compare an incomplete panel')
    baseline = next(row for row in records if row['mode'] == 'plain')
    draft = next(row for row in records if row['mode'] == 'mtp')
    if (baseline['prompt'], baseline['requested_tokens'], baseline['trunk']['model_sha256']) != (draft['prompt'], draft['requested_tokens'], draft['trunk']['model_sha256']):
        raise ValueError('The panels do not share their prompt, length, and target')
    for library in ('libllama.so.0', 'libggml-hip.so.0'):
        if baseline['runtime_libraries'][library]['sha256'] != draft['runtime_libraries'][library]['sha256']:
            raise ValueError(f'The panels use different {library} builds')
    a, b = baseline['timings'], draft['timings']
    bt, dt = baseline.get('tokens') or [], draft.get('tokens') or []
    first_divergence = next((i for i, (plain, proposed) in enumerate(zip(bt, dt)) if plain != proposed), None)
    if first_divergence is None and len(bt) != len(dt):
        first_divergence = min(len(bt), len(dt))
    result = {'greedy_output_agrees': baseline['content'] == draft['content'],
              'first_divergent_token_index': first_divergence,
              'first_divergent_tokens': [bt[first_divergence] if first_divergence < len(bt) else None,
                                         dt[first_divergence] if first_divergence < len(dt) else None] if first_divergence is not None else None,
              'baseline_text': baseline['content'], 'draft_text': draft['content'],
              'baseline_tokens_per_second': a['predicted_per_second'],
              'draft_tokens_per_second': b['predicted_per_second'],
              'speed_ratio': b['predicted_per_second'] / a['predicted_per_second'],
              'draft_accepted': b.get('draft_n_accepted'), 'draft_proposed': b.get('draft_n'),
              'baseline': str(paths[records.index(baseline)]), 'draft': str(paths[records.index(draft)])}
    print(json.dumps(result, indent=2))
    return 0 if result['greedy_output_agrees'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'serve', 'compare'])
    parser.add_argument('panels', nargs='*', type=Path)
    parser.add_argument('--mode', choices=['plain', 'mtp'], default='plain')
    parser.add_argument('--binary', default=str(BINARY))
    parser.add_argument('--draft-model', type=Path, default=DATA / 'mtp' / NAME)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--prompt', default=PROMPT)
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--draft-n', type=int, default=2)
    parser.add_argument('--draft-p', type=float, default=0.0)
    parser.add_argument('--port', type=int, default=18095)
    parser.add_argument('--response', default='/tmp/qwen-mtp-response.json')
    parser.add_argument('--log', default='/tmp/qwen-mtp-server.log')
    parser.add_argument('--startup-seconds', type=int, default=35)
    parser.add_argument('--completion-seconds', type=int, default=25)
    args = parser.parse_args()
    try:
        if args.action == 'serve':
            serve(args)
            return 0
        if args.action == 'compare':
            if len(args.panels) != 2:
                raise ValueError('compare needs plain and MTP receipt directories')
            return compare(args.panels)
        return panel(args)
    except (OSError, ValueError, RuntimeError, TimeoutError, subprocess.CalledProcessError) as error:
        print(f'mtp benchmark: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())

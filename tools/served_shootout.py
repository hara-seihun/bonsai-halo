#!/usr/bin/env python3
"""One bounded HTTP load wave, inside run-batch-compare's GPU reservation."""
import argparse
import json
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--engine', required=True)
    p.add_argument('--prompts', required=True)
    p.add_argument('--clients', type=int, required=True)
    p.add_argument('--tokens', type=int, default=256)
    p.add_argument('--port', type=int, default=18571)
    a = p.parse_args()
    base = f'http://127.0.0.1:{a.port}/v1'
    prompts = [Path(a.prompts, f'{i:03}.txt').read_text() for i in range(a.clients)]
    proc = subprocess.Popen([a.engine, 'serve', '--profile', 'max', '--slots', str(a.clients),
                             '--context', '512', '--port', str(a.port)],
                            stdout=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 90
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f'server exited at startup: {proc.returncode}')
            try:
                with urllib.request.urlopen(base + '/models', timeout=2):
                    break
            except (urllib.error.URLError, OSError):
                if time.monotonic() > deadline:
                    raise RuntimeError('server readiness timeout')
                time.sleep(.2)

        def request(prompt):
            body = json.dumps({'messages': [{'role': 'user', 'content': prompt}],
                               'max_tokens': a.tokens, 'temperature': 0, 'stream': True,
                               'chat_template_kwargs': {'enable_thinking': False}}).encode()
            req = urllib.request.Request(base + '/chat/completions', data=body,
                                         headers={'Content-Type': 'application/json'})
            start = time.monotonic()
            first = None
            usage = {}
            with urllib.request.urlopen(req, timeout=600) as resp:
                for raw in resp:
                    line = raw.decode().strip()
                    if not line.startswith('data: '):
                        continue
                    payload = line[6:]
                    if payload == '[DONE]':
                        break
                    chunk = json.loads(payload)
                    if chunk.get('usage'):
                        usage = chunk['usage']
                    for choice in chunk.get('choices', []):
                        delta = choice.get('delta', {})
                        if first is None and (delta.get('content') or delta.get('reasoning_content') or delta.get('tool_calls')):
                            first = time.monotonic() - start
            return {'ttft_s': first, 'completion_tokens': usage.get('completion_tokens', 0),
                    'request_s': time.monotonic() - start, 'halo': usage.get('halo', {})}

        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=a.clients) as pool:
            results = list(pool.map(request, prompts))
        wall = time.monotonic() - start
        tokens = sum(r['completion_tokens'] for r in results)
        print(json.dumps({'clients': a.clients, 'tokens': tokens, 'wall_s': wall,
                          'aggregate_tps': tokens / wall, 'median_ttft_s': statistics.median(
                              r['ttft_s'] for r in results if r['ttft_s'] is not None),
                          'requests': results}))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Summarize full-logit first-request and warm Qwen short-prompt panels."""
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT = Path('../../data/qwen-moe/mmq-short-real')
RUNTIME = Path('../../data/qwen-moe/runtime/current')
N = 248320
ARMS = ('first24-j32', 'first24-j16', 'first32-j16', 'first32-j32',
        'first24-nograph-j32', 'first24-nograph-j16')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(name):
    sample_path = ROOT / (name + '.jsonl')
    logits_path = ROOT / (name + '.f32')
    log_path = ROOT / (name + '.log')
    samples = [json.loads(line) for line in sample_path.read_text().splitlines()]
    raw = logits_path.read_bytes()
    assert len(samples) == 5 and len(raw) == 5 * N * 4
    width = 32 if '32-' in name else 24
    assert [(s['width'], s['repeat'], s['vocabulary']) for s in samples] == [
        (width, i, N) for i in range(5)]
    digests = [hashlib.sha256(raw[i*N*4:(i+1)*N*4]).hexdigest() for i in range(5)]
    assert len(set(digests)) == len({s['top'] for s in samples}) == 1
    log = log_path.read_text()
    assert 'Running as unit: bonsai-halo.service' in log
    clock = re.search(r'gpu_clock: clock (\d+) MHz p50', log)
    assert clock
    times = [s['milliseconds'] for s in samples]
    return {'width': width, 'milliseconds': times, 'first_over_steady_ms': times[0] - statistics.median(times[2:]),
            'warm_median_ms': statistics.median(times[2:]), 'full_logit_sha256': digests[0],
            'top': samples[0]['top'], 'shader_clock_p50_mhz': int(clock.group(1)),
            'host_warning': next((line for line in log.splitlines() if 'NOT QUIET:' in line), None),
            'samples_sha256': sha(sample_path), 'logits_sha256': sha(logits_path), 'wrapper_log_sha256': sha(log_path)}


def main():
    runs = {name: run(name) for name in ARMS}
    for width in (24, 32):
        assert len({v['full_logit_sha256'] for v in runs.values() if v['width'] == width}) == 1
    installed = json.loads((RUNTIME / 'runtime.json').read_text())
    assert sha(RUNTIME / 'bin/libllama.so.0.2.0') == installed['artifacts']['libllama.so.0.2.0']
    receipt = {'question': 'Does the first complete natural Qwen prompt select J16, and is graph capture its dominant first-use cost?',
               'contract': 'Unchanged installed runtime and packed image, one context/model per process, first ever 24/32-token natural prompt after model load, four KV-cleared repeated prompts, one full vocabulary row each. Cold latency is wall inside llama_decode/synchronize, not model load. J16 and graph disable are environment-controlled diagnostics. Distinct processes; do not subtract cross-process cold times as a causal speedup.',
               'model_sha256': 'ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61',
               'runtime_revision': installed['source_revision'],
               'runtime_manifest_sha256': sha(RUNTIME / 'runtime.json'),
               'runtime_library_sha256': sha(RUNTIME / 'bin/libllama.so.0.2.0'),
               'source_sha256': sha(Path(__file__).with_name('first_width_probe.cpp')),
               'panel_source_sha256': sha(Path(__file__)),
               'probe_binary_sha256': sha(ROOT / 'first-probe'),
               'gpu_wrapper': 'tools/run-batch-compare --runtime-max 42s --memory-gib 30 --host-reserve-gib 4 --exec; restores bonsai-halo.service per arm',
               'runs': runs, 'pairs': {f'{width}-{graph}': {
                   'warm_gain_percent': 100 * (runs[f'first{width}{graph}-j32']['warm_median_ms'] /
                                               runs[f'first{width}{graph}-j16']['warm_median_ms'] - 1),
                   'cold_j32_ms': runs[f'first{width}{graph}-j32']['milliseconds'][0],
                   'cold_j16_ms': runs[f'first{width}{graph}-j16']['milliseconds'][0]}
                   for width, graph in ((24, ''), (32, ''), (24, '-nograph'))},
               'decision': 'J16 retains an exact-logit warm complete-model gain. First-use penalty persists without graph capture and flips arm order across 24/32/no-graph processes. Do not install J16 for cold requests from these samples; next instrument first-use HIP/kernel setup and repeat paired served requests before changing dispatch.'}
    output = ROOT / 'first-width-receipt.json'
    output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'receipt': str(output), 'sha256': sha(output), 'pairs': receipt['pairs']}, indent=2))


if __name__ == '__main__':
    main()

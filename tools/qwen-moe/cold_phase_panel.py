#!/usr/bin/env python3
"""Summarize bounded, full-logit first-request host/device timing on installed Qwen."""
import hashlib
import json
from pathlib import Path
import re
import statistics

DATA = Path('../../data/qwen-moe/mmq-cold-phase')
ROOT = Path(__file__).resolve().parent
RUNTIME = Path('../../data/qwen-moe/runtime/current')
ARMS = ('graph-j32', 'nograph-j32', 'graph-j16', 'nograph-j16',
        'nograph-j32-b', 'graph-j32-b')
FIELDS = ('clear_ms', 'decode_ms', 'sync_ms', 'logits_ms', 'complete_ms')
VOCAB = 248320


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(name):
    raw = (DATA / (name + '.f32')).read_bytes()
    samples = [json.loads(line) for line in (DATA / (name + '.jsonl')).read_text().splitlines()]
    assert len(samples) == 5 and len(raw) == 5 * VOCAB * 4
    assert all(s['width'] == 24 and s['vocabulary'] == VOCAB and s['repeat'] == i
               for i, s in enumerate(samples))
    digests = [hashlib.sha256(raw[i*VOCAB*4:(i+1)*VOCAB*4]).hexdigest()
               for i in range(5)]
    assert len(set(digests)) == 1
    assert len({s['top'] for s in samples}) == 1
    for s in samples:
        assert abs(s['complete_ms'] - sum(s[k] for k in ('decode_ms','sync_ms','logits_ms'))) < .001
    log = (DATA / (name + '.log')).read_text()
    assert 'Running as unit: bonsai-halo.service' in log
    clock = re.search(r'gpu_clock: clock (\d+) MHz p50', log)
    assert clock
    first = {k: samples[0][k] for k in FIELDS}
    warm = {k: statistics.median(s[k] for s in samples[2:]) for k in FIELDS}
    return {'samples': samples, 'first_ms': first, 'warm_median_ms': warm,
            'first_minus_warm_ms': {k: first[k] - warm[k] for k in FIELDS},
            'full_logit_sha256': digests[0], 'top': samples[0]['top'],
            'shader_clock_p50_mhz': int(clock.group(1)),
            'host_warning': next((line for line in log.splitlines() if 'NOT QUIET:' in line), None),
            'samples_sha256': sha(DATA / (name + '.jsonl')),
            'logits_sha256': sha(DATA / (name + '.f32')),
            'wrapper_log_sha256': sha(DATA / (name + '.log'))}


def main():
    runs = {name: run(name) for name in ARMS}
    assert len({r['full_logit_sha256'] for r in runs.values()}) == 1
    installed = json.loads((RUNTIME / 'runtime.json').read_text())
    assert sha(RUNTIME / 'bin/libllama.so.0.2.0') == installed['artifacts']['libllama.so.0.2.0']
    receipt = {
        'question': 'Which measured region owns first-use excess on a natural Qwen MoE prompt?',
        'contract': 'Installed runtime and full model, 24 natural text tokens and one full vocabulary logit row per evaluation, first ever evaluation followed by four KV-cleared repeats in each of six processes. Monotonic host wall partitions clear, llama_decode, llama_synchronize and logits lookup/argmax. Decode can itself block on GPU work: these partitions are host API boundaries, not a GPU event trace. Cold excess compares first with median of repeats 2-4 within each process; independent graph/J arms are not paired causal effects.',
        'model_sha256': 'ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61',
        'runtime_revision': installed['source_revision'],
        'runtime_manifest_sha256': sha(RUNTIME / 'runtime.json'),
        'runtime_library_sha256': sha(RUNTIME / 'bin/libllama.so.0.2.0'),
        'source_sha256': sha(ROOT / 'cold_phase_probe.cpp'),
        'panel_source_sha256': sha(Path(__file__)),
        'probe_binary_sha256': sha(DATA / 'probe'),
        'gpu_wrapper': 'tools/run-batch-compare --runtime-max 40s --memory-gib 30 --host-reserve-gib 4 --exec; six separate reservations, service restored in each log',
        'environment': {'graph-j32': {}, 'nograph-j32': {'GGML_CUDA_DISABLE_GRAPHS': '1'},
                        'graph-j16': {'GGML_CUDA_MMQ_EXPERT_J': '16'},
                        'nograph-j16': {'GGML_CUDA_DISABLE_GRAPHS': '1', 'GGML_CUDA_MMQ_EXPERT_J': '16'},
                        'nograph-j32-b': {'GGML_CUDA_DISABLE_GRAPHS': '1'}, 'graph-j32-b': {}},
        'runs': runs,
        'decision': 'The first-request excess is in llama_decode host API wall, not after its return at llama_synchronize. Graph-disabled runs also pay it, and their second evaluation does not. Investigate lazy HIP compilation/initialization, GGML graph allocation and inside-decode synchronization with native regions before changing J width or graph policy for cold serving. No selected runtime change.'}
    output = DATA / 'receipt.json'
    output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'receipt': str(output), 'sha256': sha(output),
                      'regions': {k: v['first_minus_warm_ms'] for k, v in runs.items()}}, indent=2))


if __name__ == '__main__':
    main()

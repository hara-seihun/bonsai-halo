#!/usr/bin/env python3
"""Summarize the pinned installed-binary J=16 prompt panel without rerunning the GPU."""
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT = Path('../../data/qwen-moe/mmq-width16')
BENCH = {
    32: [('20260924T004437Z-prompt-d0-698b9494', '20260924T004452Z-prompt-d0-025c59e5'),
         ('20260924T004508Z-prompt-d0-fa807d6e', '20260924T004518Z-prompt-d0-89f6e597')],
    24: [('20260924T004746Z-prompt-d0-96f60dcf', '20260924T004756Z-prompt-d0-6c1aa34d')],
    16: [('20260924T004813Z-prompt-d0-7b704da8', '20260924T004823Z-prompt-d0-ae9a5f27')],
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bench_run(name, width, forced):
    directory = ROOT / name
    receipt = json.loads((directory / 'receipt.json').read_text())
    assert receipt['exit_code'] == 0 and receipt['phase'] == 'prompt'
    assert receipt['measured_tokens'] == width and receipt['repetitions'] == 3
    assert receipt['runtime_environment'].get('GGML_CUDA_MMQ_EXPERT_J') == ('16' if forced else None)
    assert receipt['samples'][0]['n_prompt'] == width
    ns = receipt['samples'][0]['samples_ns']
    assert len(ns) == 3
    stderr = (directory / 'stderr.log').read_text()
    clock = re.search(r'gpu_clock: clock (\d+) MHz p50', stderr)
    assert clock and 'Running as unit: bonsai-halo.service' in stderr
    return {'receipt_sha256': sha(directory / 'receipt.json'),
            'stdout_sha256': sha(directory / 'stdout.jsonl'),
            'stderr_sha256': sha(directory / 'stderr.log'),
            'samples_ns': ns, 'warm_median_ns': statistics.median(ns[1:]),
            'shader_clock_p50_mhz': int(clock.group(1)),
            'binary_sha256': receipt['binary_sha256'],
            'source': receipt['llama_cpp_source'],
            'model_sha256': receipt['model']['model_sha256']}


def main():
    sizes = {}
    for width, pairs in BENCH.items():
        results = []
        for base, trial in pairs:
            a, b = bench_run(base, width, False), bench_run(trial, width, True)
            assert (a['binary_sha256'], a['source'], a['model_sha256']) == (b['binary_sha256'], b['source'], b['model_sha256'])
            results.append({'selected_J': a, 'forced_J16': b,
                            'warm_median_speedup_percent': 100 * (a['warm_median_ns'] / b['warm_median_ns'] - 1)})
        sizes[str(width)] = results
    acceptance = ROOT / 'acceptance'
    arms = {}
    for name in ('short-j32', 'short-j16', 'short-j16-b', 'short-j32-b'):
        prefix = acceptance / name
        log = (prefix.with_suffix('.log')).read_text()
        match = re.search(r'\{"prompt_tokens":\d+,"steps":\d+,"vocabulary":\d+,"prompt_seconds":\d+\.\d+,"decode_and_dump_seconds":\d+\.\d+\}', log)
        assert match and 'Running as unit: bonsai-halo.service' in log
        arms[name] = {'log_sha256': sha(prefix.with_suffix('.log')),
                      'logit_sha256': sha(prefix.with_suffix('.f32')),
                      'tokens_sha256': sha(prefix.with_suffix('.tokens')),
                      'response': json.loads(match.group())}
    assert len({a['logit_sha256'] for a in arms.values()}) == 1
    assert len({a['tokens_sha256'] for a in arms.values()}) == 1
    binary = ROOT / 'qwen-accept-short'
    result = {'question': 'Does forcing installed J16 improve short whole-model prompt work while preserving its arithmetic map?',
              'analysis_source_sha256': sha(Path(__file__)),
              'accept_source_sha256': sha(Path(__file__).with_name('accept.cpp')),
              'accept_binary_sha256': sha(binary),
              'installed_binary_sha256': sizes['32'][0]['selected_J']['binary_sha256'],
              'sizes': sizes, 'short_real_prompt_acceptance': arms,
              'compared_logit_bits': 2 * 248320 * 32,
              'note': 'Each bench run uses its own GPU wrapper and fresh process. Warm medians omit first process sample but retain all raw samples. The 22-token natural prompt is acceptance, not a stable prefill-speed benchmark. No J16 deployment or J16 full-model 512-token speed claim.'}
    (ROOT / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'warm_gain_percent': {w: [round(p['warm_median_speedup_percent'], 3) for p in pairs] for w, pairs in sizes.items()},
                      'matching_logit_bits': result['compared_logit_bits'],
                      'receipt_sha256': sha(ROOT / 'receipt.json')}, indent=2))


if __name__ == '__main__':
    main()

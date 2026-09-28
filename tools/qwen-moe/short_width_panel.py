#!/usr/bin/env python3
"""Validate the real-token, repeated-prefill J=16/J=32 GPU receipt."""
import hashlib
import json
from pathlib import Path
import re
import statistics

DATA = Path('../../data/qwen-moe/mmq-short-real')
RUNTIME = Path('../../data/qwen-moe/runtime/current')
PAIRS = [('j32-a', 'j16-a'), ('j32-b', 'j16-b'), ('j32-c', 'j16-c')]
WIDTHS = (16, 24, 32)
VOCAB = 248320


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(name):
    rows = [json.loads(line) for line in (DATA / (name + '.jsonl')).read_text().splitlines()]
    floats = (DATA / (name + '.f32')).read_bytes()
    assert len(floats) == len(rows) * VOCAB * 4
    assert [(row['width'], row['repeat']) for row in rows] == [
        (width, rep) for width in WIDTHS for rep in range(len(rows) // len(WIDTHS))]
    assert all(row['vocabulary'] == VOCAB for row in rows)
    digests = [hashlib.sha256(floats[i * VOCAB * 4:(i + 1) * VOCAB * 4]).hexdigest()
               for i in range(len(rows))]
    by_width = {}
    for width in WIDTHS:
        entries = [row for row in rows if row['width'] == width]
        hashes = [digests[i] for i, row in enumerate(rows) if row['width'] == width]
        assert len(set(hashes)) == 1
        by_width[str(width)] = {
            'milliseconds': [row['milliseconds'] for row in entries],
            'steady_median_ms': statistics.median(row['milliseconds'] for row in entries[2:]),
            'logit_sha256': hashes[0], 'top': entries[0]['top'],
        }
        assert len({row['top'] for row in entries}) == 1
    log = (DATA / (name + '.log')).read_text()
    clock = re.search(r'gpu_clock: clock (\d+) MHz p50', log)
    assert clock and 'Running as unit: bonsai-halo.service' in log
    return {
        'samples_sha256': sha(DATA / (name + '.jsonl')),
        'logits_sha256': sha(DATA / (name + '.f32')),
        'wrapper_log_sha256': sha(DATA / (name + '.log')),
        'shader_clock_p50_mhz': int(clock.group(1)),
        'host_warning': next((line for line in log.splitlines() if 'NOT QUIET:' in line), None),
        'widths': by_width,
    }


def main():
    selected = json.loads((RUNTIME / 'runtime.json').read_text())
    assert sha(RUNTIME / 'bin/libllama.so.0.2.0') == selected['artifacts']['libllama.so.0.2.0']
    arms = {}
    pairs = []
    for control, forced in PAIRS:
        arms[control], arms[forced] = run(control), run(forced)
        gain = {}
        for width in WIDTHS:
            left = arms[control]['widths'][str(width)]
            right = arms[forced]['widths'][str(width)]
            assert left['logit_sha256'] == right['logit_sha256']
            assert left['top'] == right['top']
            gain[str(width)] = 100 * (left['steady_median_ms'] / right['steady_median_ms'] - 1)
        pairs.append({'control': control, 'forced_J16': forced, 'gain_percent': gain})
    for width in WIDTHS:
        assert len({arm['widths'][str(width)]['logit_sha256'] for arm in arms.values()}) == 1
    source = Path(__file__).with_name('short_width_probe.cpp')
    receipt = {
        'question': 'Does J16 improve repeated natural-token complete-model prefill at 16/24/32 rows?',
        'domain': 'Pinned Qwen3.6-35B-A3B UD-Q4_K_M, full 40-layer prompt plus 248320-row head; natural 16/24/32 token prefixes; five or six in-process repetitions after KV reset, discard first two per shape for steady comparison.',
        'model_sha256': 'ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61',
        'runtime_revision': selected['source_revision'],
        'runtime_manifest_sha256': sha(RUNTIME / 'runtime.json'),
        'runtime_library_sha256': sha(RUNTIME / 'bin/libllama.so.0.2.0'),
        'probe_source_sha256': sha(source),
        'panel_source_sha256': sha(Path(__file__)),
        'probe_binary_sha256': sha(DATA / 'probe'),
        'gpu_wrapper': 'tools/run-batch-compare --memory-gib 30 --host-reserve-gib 4 --runtime-max 42s; pairs a/b pin performance level, pair c fixes 2400 MHz; model and executable unchanged.',
        'compared_logit_bits_per_shape': VOCAB * 32,
        'arms': arms, 'pairs': pairs,
        'selection': 'No installed default changed: first-evaluation latency is noisy and dominated by graph/setup work; steady short-prompt J16 wins at 24 and 32 rows, 16 is a null.',
    }
    (DATA / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'gain_percent': [p['gain_percent'] for p in pairs],
                      'receipt_sha256': sha(DATA / 'receipt.json')}, indent=2))


if __name__ == '__main__':
    main()

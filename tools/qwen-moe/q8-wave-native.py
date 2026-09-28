#!/usr/bin/env python3
"""Summarize alternating same-byte phase-aware Q8 reader measurements."""
import hashlib
import json
import re
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('../../data/qwen-moe/q8-wave-native')
BINARY = DATA.parent / 'q8-wave-probe'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def panel(name):
    path = DATA / (name + '.log')
    raw = path.read_text()
    shape = re.search(r'rows=(\d+) groups=(\d+) image_bytes=(\d+) differing_f32=(\d+)', raw)
    assert shape and int(shape[4]) == 0
    rows, groups, image, _ = map(int, shape.groups())
    assert groups == rows * 32 and image == groups * 272
    samples = {'inter': [], 'wave': []}
    for line in raw.splitlines():
        if line.startswith('round='):
            arms = re.findall(r'(inter|wave)_ms=(\d+\.\d+)', line)
            assert len(arms) == 2
            for arm, elapsed in arms:
                samples[arm].append(float(elapsed))
    assert len(samples['inter']) == len(samples['wave']) == 7
    med = {arm: median(values) for arm, values in samples.items()}
    assert 'Running as unit: bonsai-halo.service' in raw
    return {'rows': rows, 'groups': groups, 'image_bytes_per_arm': image,
            'different_output_words': 0, 'samples_ms': samples,
            'median_ms': med, 'wave_over_inter': med['wave'] / med['inter'],
            'wrapper_log_sha256': digest(path),
            'clock_and_host': next(s for s in raw.splitlines() if s.startswith('gpu_clock:')),
            'service_restored': True}


def main():
    result = {'contract': 'Standalone gfx1151 wave reader, K8192, synthetic deterministic codes and scales, exactly 272 bytes per wave group and ordered fmaf/fold. Alternating device-event elapsed, no GGUF or selected MMVQ, graph, model quality or whole-model throughput.',
              'source_sha256': digest(ROOT / 'bench/qwen_q8_wave.hip'),
              'summarizer_sha256': digest(Path(__file__)),
              'executable_sha256': digest(BINARY),
              'panels': {name: panel(name) for name in ('cache', 'full', 'full-2400')}}
    DATA.joinpath('receipt.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: {'ratio': v['wave_over_inter'], 'medians': v['median_ms']}
                      for k, v in result['panels'].items()}, indent=2))


if __name__ == '__main__':
    main()

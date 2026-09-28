#!/usr/bin/env python3
"""Summarize bounded gfx1151 Q8 layout panels with source/binary/raw custody."""
import hashlib
import json
import re
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('../../data/qwen-moe/q8-layout-native')
BINARY = DATA.parent / 'q8-layout-native-probe'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def panel(path):
    raw = path.read_text()
    shape = re.search(r'rows=(\d+) groups=(\d+) image_bytes=(\d+) differing_f32=(\d+)', raw)
    assert shape and int(shape[4]) == 0
    samples = {'inter': [], 'split': []}
    for line in raw.splitlines():
        if not line.startswith('round='):
            continue
        for arm, ms in re.findall(r'(inter|split)_ms=(\d+\.\d+)', line):
            samples[arm].append(float(ms))
    assert len(samples['inter']) == len(samples['split']) == 5
    medians = {arm: median(samples[arm]) for arm in samples}
    return {'log_sha256': sha(path), 'rows': int(shape[1]), 'groups': int(shape[2]),
            'image_bytes_per_arm': int(shape[3]), 'differing_f32': int(shape[4]),
            'samples_ms': samples, 'median_ms': medians,
            'split_over_inter': medians['split']/medians['inter'],
            'wrapper_clock_and_host': next(s for s in raw.splitlines() if s.startswith('gpu_clock:')),
            'service_restored': "Running as unit: bonsai-halo.service" in raw}


def main():
    p = {label: panel(DATA / f'{label}-panel.log') for label in ('small', 'cache', 'full')}
    result = {'contract': 'gfx1151 standalone model-less Q8 wave reader, one K8192 row per wave; five alternating measurements per image size. Same 272 source bytes per eight-block group, same int8 codes/FP16 scales/input and per-lane FP32 FMA and warp fold; synthetic weights and activation, not native llama.cpp or complete-model throughput.',
              'source_sha256': sha(ROOT/'bench/qwen_q8_layout.hip'),
              'summarizer_sha256': sha(Path(__file__)),
              'executable_sha256': sha(BINARY),
              'assembly_sha256': sha(DATA/'probe.s'),
              'panels': p}
    assert all(x['service_restored'] for x in p.values())
    (DATA/'receipt.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({k: {'image_bytes': v['image_bytes_per_arm'],
                          'inter_ms': v['median_ms']['inter'],
                          'split_ms': v['median_ms']['split'],
                          'split_over_inter': v['split_over_inter']} for k,v in p.items()}, indent=2))


if __name__ == '__main__':
    main()

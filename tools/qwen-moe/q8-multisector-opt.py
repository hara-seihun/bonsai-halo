#!/usr/bin/env python3
"""Certify the phase-aware Q8 layout against per-wave 32/64/128-B sector floors."""
import argparse
import importlib.util
import json
from collections import Counter
from hashlib import sha256
from math import gcd
from pathlib import Path

SECTOR = Path(__file__).with_name('q8-sector-opt.py')
spec = importlib.util.spec_from_file_location('sector', SECTOR)
sector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sector)


def minimum_requests(phase, width):
    # A code request set covering exactly 256 distinct bytes needs 256/width
    # sectors only if that many complete aligned sectors lie inside this
    # fixed, unpadded 272-byte wave group. Otherwise it needs at least one more.
    full = (phase + 272) // width - (phase + width - 1) // width
    codes = 256 // width + (full < 256 // width)
    return codes + 1  # 16 distinct FP16-scale bytes need a separate request.


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--traffic', type=Path, default=Path('../../data/qwen-moe/traffic.json'))
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    raw = args.traffic.read_bytes()
    traffic = json.loads(raw)
    align = traffic['metadata'].get('general.alignment', 32)
    base = (traffic['header_bytes'] + align - 1) // align * align
    tensors = [t for t in traffic['tensors'] if t['type'] == 'Q8_0'
               and t['name'] != 'token_embd.weight' and '_exps.' not in t['name']]
    widths = (32, 64, 128)
    counts = {str(w): Counter() for w in widths}
    phases = {str(w): Counter() for w in widths}
    for t in tensors:
        k, rows = t['shape']
        assert k % 256 == 0 and t['bytes'] == k * rows * 34 // 32
        stride = k * 34 // 32
        for width in widths:
            period = width // gcd(width, stride)
            for row in range(min(rows, period)):
                repeats = (rows - 1 - row) // period + 1
                for group in range(k // 256):
                    phase = (base + t['offset'] + row * stride + group * 272) % width
                    assert phase % 16 == 0
                    # One parity decision for the *actual image*, not a distinct
                    # layout optimized for each hypothetical sector width.
                    scale_first = bool(phase & 16)
                    code, scale, union = sector.requests(phase, width, scale_first)
                    floor = minimum_requests(phase, width)
                    assert code + scale == floor, (t['name'], row, group, width, phase)
                    assert union * width >= 272
                    counts[str(width)]['wave_groups'] += repeats
                    counts[str(width)]['floor_sectors'] += floor * repeats
                    counts[str(width)]['code_sectors'] += code * repeats
                    counts[str(width)]['scale_sectors'] += scale * repeats
                    phases[str(width)][str(phase)] += repeats
    groups = sum(t['bytes'] for t in tensors) // 272
    assert all(counts[str(w)]['wave_groups'] == groups for w in widths)
    result = {
        'contract': 'unpadded fixed 272-byte wave boundaries, distinct 256 code and 16 scale bytes, separately counted instruction-sector unions; not DRAM traffic or native time',
        'layout': 'code-first at group offset 0 mod 32, scale-first at offset 16 mod 32, independent of sector width',
        'source_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
        'sector_source_sha256': sha256(SECTOR.read_bytes()).hexdigest(),
        'geometry_source_sha256': sha256(sector.GEOMETRY.read_bytes()).hexdigest(),
        'traffic_sha256': sha256(raw).hexdigest(),
        'traffic_header_sha256': traffic['header_sha256'],
        'tensors': len(tensors), 'image_bytes': sum(t['bytes'] for t in tensors),
        'wave_groups': groups,
        'widths': {str(w): {
            'phases': dict(sorted(phases[str(w)].items(), key=lambda kv: int(kv[0]))),
            'separate_request_bytes': counts[str(w)]['floor_sectors'] * w,
            'code_request_bytes': counts[str(w)]['code_sectors'] * w,
            'scale_request_bytes': counts[str(w)]['scale_sectors'] * w,
            'sectors_per_phase': {str(p): minimum_requests(p, w) for p in range(0, w, 16)},
        } for w in widths},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['widths'], indent=2))


if __name__ == '__main__':
    main()

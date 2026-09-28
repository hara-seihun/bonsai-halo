#!/usr/bin/env python3
"""Find phase-aware, zero-padding Q8_0 wave layouts under a sector-request model."""
import argparse
import importlib.util
import json
from collections import defaultdict
from functools import lru_cache
from hashlib import sha256
from math import gcd
from pathlib import Path

GEOMETRY = Path(__file__).with_name('q8-transaction-geometry.py')
spec = importlib.util.spec_from_file_location('q8_geometry', GEOMETRY)
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)


@lru_cache(None)
def requests(phase, width, scale_first):
    code_offset = 16 if scale_first else 0
    scale_offset = 0 if scale_first else 256
    codes = [(code_offset + 32*b + 8*lane, 8) for b in range(8) for lane in range(4)]
    scales = [(scale_offset + 2*b, 2) for b in range(8)]
    return (geometry.sectors(codes, phase, width),
            geometry.sectors(scales, phase, width),
            geometry.sectors(codes + scales, phase, width))


def all_counts(tensors, base, width):
    old = [0, 0, 0]
    transposed = [0, 0, 0]
    chosen = [0, 0, 0]
    phases = defaultdict(int)
    groups = 0
    for t in tensors:
        k, rows = t['shape']
        assert k % 256 == 0 and t['bytes'] == k * rows * 34 // 32
        stride = k * 34 // 32
        row_period = width // gcd(width, stride)
        for row in range(min(rows, row_period)):
            repeats = (rows - 1 - row) // row_period + 1
            for block in range(k // 256):
                phase = (base + t['offset'] + row * stride + 272*block) % width
                choices = [requests(phase, width, first) for first in (False, True)]
                winner = min(choices, key=lambda x: x[0]+x[1])
                for count, dst in ((geometry.wave(phase, width, 'installed'), old),
                                   (geometry.wave(phase, width, 'wave_transposed'), transposed),
                                   (winner, chosen)):
                    for i in range(3):
                        dst[i] += count[i] * repeats
                groups += repeats
                phases[phase] += repeats
    assert groups * 272 == sum(t['bytes'] for t in tensors)
    assert chosen[2] == old[2] == transposed[2]
    if width == 32:
        assert chosen[0] + chosen[1] == groups * 9
    return dict(groups=groups, phases=dict(sorted(phases.items())),
                installed=dict(code=old[0]*width, scale=old[1]*width, union=old[2]*width),
                uniform_scale_first=dict(code=transposed[0]*width, scale=transposed[1]*width,
                                         union=transposed[2]*width),
                phase_aware=dict(code=chosen[0]*width, scale=chosen[1]*width, union=chosen[2]*width),
                lower_bound_separate_bytes=groups * (256 // width + 1)*width if width in (32,64,128) else None)


def check_byte_bijection():
    source = bytes((i * 113 + 47) % 256 for i in range(272))
    for first in (False, True):
        scale_at = 0 if first else 256
        code_at = 16 if first else 0
        packed = bytearray(272)
        for b in range(8):
            packed[scale_at+2*b:scale_at+2*b+2] = source[34*b:34*b+2]
            packed[code_at+32*b:code_at+32*b+32] = source[34*b+2:34*b+34]
        rebuilt = bytearray(272)
        for b in range(8):
            rebuilt[34*b:34*b+2] = packed[scale_at+2*b:scale_at+2*b+2]
            rebuilt[34*b+2:34*b+34] = packed[code_at+32*b:code_at+32*b+32]
        assert rebuilt == source


def main():
    check_byte_bijection()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--traffic', type=Path, default=Path('../../data/qwen-moe/traffic.json'))
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    raw = args.traffic.read_bytes()
    traffic = json.loads(raw)
    align = traffic['metadata'].get('general.alignment', 32)
    base = (traffic['header_bytes'] + align - 1) // align * align
    tensors = [t for t in traffic['tensors'] if t['type'] == 'Q8_0' and t['name'] != 'token_embd.weight' and '_exps.' not in t['name']]
    table = {str(w): all_counts(tensors, base, w) for w in (32,64,128)}
    result = dict(contract='one wave / one row / eight Q8_0 blocks; separate code and scale load sector unions, not DRAM traffic',
                  source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  geometry_source_sha256=sha256(GEOMETRY.read_bytes()).hexdigest(),
                  traffic_sha256=sha256(raw).hexdigest(), traffic_header_sha256=traffic['header_sha256'],
                  tensor_count=len(tensors), image_bytes=sum(t['bytes'] for t in tensors),
                  results=table)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(table, indent=2))


if __name__ == '__main__':
    main()

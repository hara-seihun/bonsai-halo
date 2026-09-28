#!/usr/bin/env python3
"""Count requested sectors for selected one-wave Q8_0 MMVQ loads (not DRAM bytes)."""
import argparse
from collections import defaultdict
from functools import lru_cache
from hashlib import sha256
from math import gcd
from pathlib import Path
import json


def sectors(ranges, base, width):
    return len({s for start, size in ranges for s in range((base + start) // width,
                                                             (base + start + size - 1) // width + 1)})


@lru_cache(None)
def wave(base, width, layout):
    # qi/vdr=4: four lanes consume consecutive 8-byte code slices of each
    # Q8_0 block; eight blocks per wave. One 16-bit scale is requested by all
    # four lanes of a block (duplicate addresses count once per wave).
    if layout == 'installed':
        codes = [(34 * b + 2 + 8 * lane, 8) for b in range(8) for lane in range(4)]
        scales = [(34 * b, 2) for b in range(8)]
    else:
        # Same 272 bytes and same 8 scales/256 codes, transposed within each
        # wave's K group; no padding or extra index image.
        codes = [(16 + 32 * b + 8 * lane, 8) for b in range(8) for lane in range(4)]
        scales = [(2 * b, 2) for b in range(8)]
    return sectors(codes, base, width), sectors(scales, base, width), sectors(codes + scales, base, width)


def tensor(t, data_base, width, layout):
    k, rows = t['shape']
    assert k % 256 == 0 and t['bytes'] == k * rows * 34 // 32
    stride = k * 34 // 32
    group_count = k // 256
    period = width // gcd(width, stride)
    offset = data_base + t['offset']
    counts = [0, 0, 0]
    for r in range(period):
        repeats = (rows - 1 - r) // period + 1 if r < rows else 0
        for group in range(group_count):
            result = wave((offset + r * stride + group * 272) % width, width, layout)
            for i, n in enumerate(result):
                counts[i] += n * repeats
    unique = (offset + t['bytes'] - 1) // width - offset // width + 1
    return [v * width for v in counts], unique * width


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--traffic', type=Path, default=Path('../../data/qwen-moe/traffic.json'))
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    traffic = json.loads(args.traffic.read_text())
    align = traffic['metadata'].get('general.alignment', 32)
    data_base = (traffic['header_bytes'] + align - 1) // align * align
    tensors = [t for t in traffic['tensors'] if t['type'] == 'Q8_0' and t['name'] != 'token_embd.weight' and '_exps.' not in t['name']]
    table = {}
    for layout in ('installed', 'wave_transposed'):
        table[layout] = {}
        for width in (32, 64, 128):
            totals = defaultdict(lambda: [0, 0, 0, 0, 0, 0])
            for t in tensors:
                parts, unique = tensor(t, data_base, width, layout)
                k, _ = t['shape']
                group = totals[str(k)]
                group[0] += t['bytes']
                group[1] += parts[0]
                group[2] += parts[1]
                group[3] += parts[2]
                group[4] += unique
                group[5] += 1
            table[layout][str(width)] = {k: dict(image_bytes=v[0], code_request_bytes=v[1], scale_request_bytes=v[2],
                                         separate_request_bytes=v[1]+v[2], merged_per_wave_bytes=v[3],
                                         unique_image_sector_bytes=v[4], tensor_count=v[5]) for k, v in totals.items()}
            total = [sum(v[i] for v in totals.values()) for i in range(6)]
            table[layout][str(width)]['all'] = dict(image_bytes=total[0], code_request_bytes=total[1], scale_request_bytes=total[2],
                                                    separate_request_bytes=total[1]+total[2], merged_per_wave_bytes=total[3],
                                                    unique_image_sector_bytes=total[4], tensor_count=total[5])
    out = dict(contract='One-wave, one-row, 8 Q8_0 blocks/wave iteration, aligned sector model; separate code/scale ISA loads; requests are not DRAM transactions',
               source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
               traffic_sha256=sha256(args.traffic.read_bytes()).hexdigest(),
               traffic_header_sha256=traffic['header_sha256'], traffic_file=str(args.traffic),
               data_base=data_base, q8_tensor_count=len(tensors), geometry=table)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(table, indent=2))


if __name__ == '__main__':
    main()

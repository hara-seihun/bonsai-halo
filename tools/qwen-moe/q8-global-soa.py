#!/usr/bin/env python3
"""Count same-byte global Q8 code/scale bank requests, beyond wave-local layouts."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def check_bijection():
    # Five consecutive waves, including an original row/tensor boundary: labels
    # remain tied to (wave, block) under one global coordinate transformation.
    src = bytes((i * 113 + 47) % 256 for i in range(272 * 5))
    code, scale = bytearray(256 * 5), bytearray(16 * 5)
    for wave in range(5):
        for block in range(8):
            start = wave * 272 + block * 34
            code[wave*256+block*32:wave*256+(block+1)*32] = src[start+2:start+34]
            scale[wave*16+block*2:wave*16+(block+1)*2] = src[start:start+2]
    assert len(code)+len(scale) == len(src)
    dst = bytearray(len(src))
    for wave in range(5):
        for block in range(8):
            start = wave * 272 + block * 34
            dst[start:start+2] = scale[wave*16+block*2:wave*16+(block+1)*2]
            dst[start+2:start+34] = code[wave*256+block*32:wave*256+(block+1)*32]
    assert dst == src


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--traffic', type=Path, default=Path('../../data/qwen-moe/traffic.json'))
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    check_bijection()
    raw = args.traffic.read_bytes()
    traffic = json.loads(raw)
    tensors = [t for t in traffic['tensors'] if t['type'] == 'Q8_0'
               and t['name'] != 'token_embd.weight' and '_exps.' not in t['name']]
    # A contiguous Q8-only device allocation is stipulated 128B aligned.
    # Tensor code offset is sum of preceding code sizes; scale offset is sum
    # of preceding scale sizes. Both banks have zero logical padding.
    starts = []
    code_total = scale_total = groups = 0
    for t in tensors:
        k, rows = t['shape']
        assert k % 256 == 0 and t['bytes'] == k * rows * 34 // 32
        n = (k // 256) * rows
        starts.append((t['name'], code_total, scale_total, n))
        code_total += n * 256
        scale_total += n * 16
        groups += n
    assert all(code % 128 == 0 and scale % 128 == 0 for _,code,scale,_ in starts)
    assert code_total % 128 == 0 and scale_total % 128 == 0
    assert code_total + scale_total == sum(t['bytes'] for t in tensors)
    table = {}
    for width in (32, 64, 128):
        assert code_total % width == 0
        phases = Counter()
        for _, code, scale, n in starts:
            # All code waves start sector aligned. A wave's 16B scales
            # sit entirely inside one sector at each of these widths.
            for g in range(n):
                phases[(scale + 16*g) % width] += 1
        assert sum(phases.values()) == groups
        assert all(p + 16 <= width for p in phases)
        code_bytes = groups * (256 // width) * width
        scale_bytes = groups * width
        table[str(width)] = dict(code_request_bytes=code_bytes,
                                 scale_request_bytes=scale_bytes,
                                 separate_request_bytes=code_bytes+scale_bytes,
                                 scale_phase_counts=dict(sorted(phases.items())))
    result = dict(contract='128B-aligned contiguous device Q8 codes followed by scales, no logical padding, separate per-wave instruction sector-union requests; not DRAM',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  traffic_sha256=hashlib.sha256(raw).hexdigest(),
                  traffic_header_sha256=traffic['header_sha256'],
                  tensors=len(tensors), groups=groups, image_bytes=code_total+scale_total,
                  code_bank_bytes=code_total, scale_bank_bytes=scale_total,
                  all_tensor_bank_offsets_128_aligned=True, results=table)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({w: r['separate_request_bytes'] for w,r in table.items()}, sort_keys=True))


if __name__ == '__main__':
    main()

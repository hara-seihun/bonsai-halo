#!/usr/bin/env python3
"""Locate serialized llama.cpp hybrid KV/recurrent differences without loading 2 GiB."""
import argparse
import json
import mmap
import struct
from pathlib import Path


class Cursor:
    def __init__(self, blob):
        self.blob = blob
        self.pos = 0

    def uint(self, fmt):
        size = struct.calcsize(fmt)
        result = struct.unpack_from('<' + fmt, self.blob, self.pos)[0]
        self.pos += size
        return result

    def skip(self, n):
        if n < 0 or self.pos + n > len(self.blob):
            raise ValueError(f'out-of-bounds state field at {self.pos}: {n}')
        start = self.pos
        self.pos += n
        return start


def regions(blob):
    c = Cursor(blob)
    arch_size = c.uint('I')
    arch = bytes(blob[c.skip(arch_size):c.pos]).decode()
    yield ('architecture', 0, c.pos, 0)
    n_stream = c.uint('I')
    if n_stream > 256:
        raise ValueError(f'unexpected stream count {n_stream}')
    for stream in range(n_stream):
        start = c.pos
        count = c.uint('I')
        for _ in range(count):
            c.uint('i')  # position
            n_seq = c.uint('I')
            if n_seq > 256:
                raise ValueError(f'unexpected sequence count {n_seq}')
            c.skip(8)  # llama_kv_cell_ext; this model has multiple positions/embedding
            c.skip(4*n_seq)  # IDs
        yield (f'kv/{stream}/meta', start, c.pos, count)
        if not count:
            continue
        trans = c.uint('I')
        layers = c.uint('I')
        if layers > 40:
            raise ValueError(f'unexpected KV layer count {layers}')
        for kind in ('k', 'v'):
            for layer in range(layers):
                start = c.pos
                dtype = c.uint('i')
                if kind == 'v' and trans:
                    elsize, embd = c.uint('I'), c.uint('I')
                    size = count * elsize * embd
                else:
                    size = count * c.uint('Q')
                c.skip(size)
                yield (f'kv/{stream}/{kind}/{layer}', start, c.pos, count)
    start = c.pos
    count = c.uint('I')
    if count > 256:
        raise ValueError(f'unexpected recurrent count {count}')
    for _ in range(count):
        c.uint('i')
        n_seq = c.uint('I')
        if n_seq > 256:
            raise ValueError(f'unexpected recurrent sequence count {n_seq}')
        c.skip(4*n_seq)
    trans = c.uint('I')
    layers = c.uint('I')
    yield ('recurrent/meta', start, c.pos, count)
    if layers != 40:
        raise ValueError(f'unexpected model layers {layers}')
    for kind in ('r', 's'):
        for layer in range(layers):
            # Pinned Qwen3.6 model: every fourth layer is full attention.
            if layer % 4 == 3:
                continue
            start = c.pos
            dtype = c.uint('i')
            if kind == 's' and trans:
                elsize, embd = c.uint('I'), c.uint('I')
                size = count * elsize * embd
            else:
                row_size = c.uint('Q')
                size = count * row_size
            payload = c.pos
            c.skip(size)
            yield (f'recurrent/{kind}/{layer}', start, c.pos, {'count': count, 'type': dtype, 'payload': payload, 'row_bytes': size // count})
    if c.pos != len(blob):
        raise ValueError(f'unparsed tail at {c.pos} of {len(blob)} bytes')
    if arch != 'qwen35moe':
        raise ValueError(f'unexpected architecture {arch}')


def compare(left, right):
    if len(left) != len(right):
        raise ValueError('state file lengths differ')
    left_regions = list(regions(left))
    right_regions = list(regions(right))
    if [(name, start, end) for name, start, end, _ in left_regions] != [
        (name, start, end) for name, start, end, _ in right_regions
    ]:
        raise ValueError('serialized tensor layout differs between arms')
    results = []
    for name, start, end, info in left_regions:
        if left[start:end] == right[start:end]:
            continue
        payload = info.get('payload', start) if isinstance(info, dict) else start
        stride = info.get('row_bytes', 0) if isinstance(info, dict) else 0
        different = 0
        first = None
        by_row = {}
        for off in range(start, end, 1 << 20):
            a = left[off:min(off+(1 << 20), end)]
            b = right[off:min(off+(1 << 20), end)]
            if a == b:
                continue
            for i, (x,y) in enumerate(zip(a,b)):
                if x != y:
                    absolute = off+i
                    if first is None:
                        first = absolute
                    different += 1
                    row = (absolute-payload)//stride if stride and absolute >= payload else -1
                    by_row[row] = by_row.get(row, 0) + 1
        results.append({'region': name, 'start': start, 'bytes': end-start,
                        'first_byte': first, 'first_payload_byte': first-payload,
                        'different_bytes': different, 'by_row': by_row})
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('control', type=Path)
    ap.add_argument('candidate', type=Path)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    with args.control.open('rb') as a, args.candidate.open('rb') as b:
        with mmap.mmap(a.fileno(), 0, access=mmap.ACCESS_READ) as left, mmap.mmap(b.fileno(), 0, access=mmap.ACCESS_READ) as right:
            report = {'control': str(args.control), 'candidate': str(args.candidate),
                      'size': len(left), 'differences': compare(left,right)}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    else:
        print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Census the fp16 block scales stored in a HALO tile cache.

The HALO tile format stores one fp16 scale per (row, 128-block) in the 4-byte
tail of every 896-byte block. This asks a structural question about the model
rather than about the engine: for a fixed row, do all of that row's block
scales carry the same fp16 bits? If they do, the per-block scale is a per-row
constant and the stream is carrying 2 of every 28 bytes for nothing.

Reads the cache with numpy strided views; no GPU, no model load.

    tools/halo_scale_census.py [--cache PATH] [--tensors N] [--blocks N]
"""
import argparse
import mmap
import os
import struct
import sys

import numpy as np

TILE_ROWS = 32
BLOCK = 128
TILE_QS_A = TILE_ROWS * 16
TILE_QS_B = TILE_ROWS * 8
TILE_TAIL_OFF = TILE_QS_A + TILE_QS_B       # 768
TILE_BLOCK_BYTES = TILE_QS_A + TILE_QS_B + TILE_ROWS * 4  # 896


def read_index(buf):
    magic, src_size, src_mtime, n_entries, data_offset = struct.unpack_from('<8sQQQQ', buf, 0)
    if magic != b'HALOCAC2':
        raise SystemExit(f'not a HALOCAC2 cache (magic {magic!r})')
    p = struct.calcsize('<8sQQQQ')
    entries = []
    for _ in range(n_entries):
        (nl,) = struct.unpack_from('<I', buf, p)
        p += 4
        name = buf[p:p + nl].decode()
        p += nl
        N, K, off, nbytes = struct.unpack_from('<qqQQ', buf, p)
        p += 32
        entries.append((name, N, K, off, nbytes))
    return entries


def tensor_scales(buf, N, K, off):
    """-> uint16 array [ntiles, nb, 32] of the stored fp16 scale bits."""
    ntiles, nb = N // TILE_ROWS, K // BLOCK
    base = off + TILE_TAIL_OFF + 2
    raw = np.frombuffer(buf, dtype=np.uint8)
    view = np.lib.stride_tricks.as_strided(
        raw[base:],
        shape=(ntiles, nb, TILE_ROWS, 2),
        strides=(nb * TILE_BLOCK_BYTES, TILE_BLOCK_BYTES, 4, 1),
        writeable=False,
    )
    return view[..., 0].astype(np.uint16) | (view[..., 1].astype(np.uint16) << 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='../../data/bonsai2/PTQ1_0.gguf.halo')
    ap.add_argument('--tensors', type=int, default=0, help='census only the first N tensors (0 = all)')
    ap.add_argument('--blocks', type=int, default=0, help='census only the first N K-blocks of each tensor (0 = all)')
    ap.add_argument('--trits', action='store_true', help='also census the trit value distribution')
    args = ap.parse_args()

    fd = os.open(args.cache, os.O_RDONLY)
    buf = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
    entries = read_index(buf)
    if args.tensors:
        entries = entries[:args.tensors]

    print(f'{len(entries)} ternary tensors in {args.cache}')
    print(f'{"tensor":<34}{"N":>8}{"K":>7}{"nb":>5}  rows  uniform  distinct/row')
    tot_rows = tot_uniform = 0
    tot_bytes = tot_scale_bytes = 0
    worst = []
    for name, N, K, off, nbytes in entries:
        s = tensor_scales(buf, N, K, off)          # [ntiles, nb, 32]
        if args.blocks:
            s = s[:, :args.blocks, :]
        first = s[:, :1, :]
        same = np.all(s == first, axis=1)          # [ntiles, 32]
        rows = same.size
        uniform = int(same.sum())
        distinct = np.array([len(np.unique(s[t, :, r])) for t in range(min(4, s.shape[0])) for r in range(4)])
        tot_rows += rows
        tot_uniform += uniform
        tot_bytes += nbytes
        tot_scale_bytes += (N // TILE_ROWS) * (K // BLOCK) * TILE_ROWS * 2
        flag = '' if uniform == rows else f'  <-- {rows - uniform} rows vary'
        print(f'{name:<34}{N:>8}{K:>7}{s.shape[1]:>5}{rows:>6}{uniform:>9}{distinct.max():>8}{flag}')
        if uniform != rows:
            worst.append((name, rows - uniform, rows))
    print()
    print(f'rows: {tot_rows}  uniform-scale rows: {tot_uniform}  ({100.0 * tot_uniform / tot_rows:.4f}%)')
    print(f'tile bytes {tot_bytes / 1e9:.3f} GB, of which stored scales {tot_scale_bytes / 1e9:.3f} GB '
          f'({100.0 * tot_scale_bytes / tot_bytes:.3f}%)')
    if worst:
        print('tensors with any non-uniform row:')
        for name, bad, rows in worst:
            print(f'  {name}: {bad}/{rows}')
    buf.close()
    os.close(fd)


if __name__ == '__main__':
    sys.exit(main())

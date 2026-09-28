#!/usr/bin/env python3
"""Price independently addressable, lossless pages of a saved Qwen GDN S state."""
import argparse
import ctypes
import hashlib
import json
import mmap
import numpy as np
import re
import shutil
import struct
import subprocess
from pathlib import Path

LAYERS = 30
ROWS = 32
ROW_BYTES = 524288 * 4
HEADER = struct.Struct('<iQ')


def zstd_library():
    binary = shutil.which('zstd')
    assert binary is not None
    listing = subprocess.check_output(['ldd', binary], text=True)
    path = re.search(r'libzstd\.so\.1 => (\S+)', listing)
    assert path is not None, listing
    lib = ctypes.CDLL(path.group(1))
    lib.ZSTD_compressBound.argtypes = [ctypes.c_size_t]
    lib.ZSTD_compressBound.restype = ctypes.c_size_t
    lib.ZSTD_compress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    lib.ZSTD_compress.restype = ctypes.c_size_t
    lib.ZSTD_decompress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
    lib.ZSTD_decompress.restype = ctypes.c_size_t
    lib.ZSTD_isError.argtypes = [ctypes.c_size_t]
    lib.ZSTD_isError.restype = ctypes.c_uint
    return lib


def inspect(path, page_bytes, level, transform):
    lib = zstd_library()
    count = compressed = zero_pages = 0
    minimum = page_bytes
    maximum = 0
    source = hashlib.sha256()
    decoded = hashlib.sha256()
    layer_sizes = []
    with path.open('rb') as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as image:
        # S is the last field of the complete hybrid cache serialization. Its
        # thirty non-null layers each have (F32 type, 2-MiB row-size) header.
        start = len(image) - LAYERS * (HEADER.size + ROWS * ROW_BYTES)
        assert start >= 0
        cursor = start
        for layer in range(LAYERS):
            type_id, row_size = HEADER.unpack_from(image, cursor)
            assert (type_id, row_size) == (0, ROW_BYTES), (layer, cursor, type_id, row_size)
            cursor += HEADER.size
            size = 0
            for row in range(ROWS):
                row_start = cursor + row * ROW_BYTES
                for off in range(0, ROW_BYTES, page_bytes):
                    page = image[row_start + off:row_start + off + page_bytes]
                    source.update(page)
                    zero_pages += not any(page)
                    if transform == 'byte-shuffle':
                        coded = np.frombuffer(page, dtype=np.uint8).reshape(-1, 4).T.tobytes()
                    elif transform == 'bit-shuffle':
                        bits = np.unpackbits(np.frombuffer(page, dtype=np.uint8)).reshape(-1, 8, 4, 8)
                        coded = np.packbits(bits.transpose(0, 2, 3, 1), axis=-1).tobytes()
                    else:
                        coded = page
                    input_buffer = ctypes.create_string_buffer(coded)
                    buffer = ctypes.create_string_buffer(lib.ZSTD_compressBound(len(coded)))
                    n = lib.ZSTD_compress(buffer, len(buffer), input_buffer, len(coded), level)
                    assert not lib.ZSTD_isError(n)
                    out = ctypes.create_string_buffer(len(page))
                    got = lib.ZSTD_decompress(out, len(out), buffer, n)
                    if transform == 'byte-shuffle':
                        restored = np.frombuffer(out.raw, dtype=np.uint8).reshape(4, -1).T.tobytes()
                    elif transform == 'bit-shuffle':
                        bits = np.unpackbits(np.frombuffer(out.raw, dtype=np.uint8)).reshape(-1, 4, 8, 8)
                        restored = np.packbits(bits.transpose(0, 3, 1, 2).reshape(-1, 8), axis=-1).tobytes()
                    else:
                        restored = out.raw
                    assert got == len(page) and restored == page, (layer, row, off)
                    decoded.update(restored)
                    compressed += n
                    size += n
                    count += 1
                    minimum = min(minimum, n)
                    maximum = max(maximum, n)
            layer_sizes.append(size)
            cursor += ROWS * ROW_BYTES
        assert cursor == len(image)
    with path.open('rb') as handle:
        file_sha = hashlib.file_digest(handle, 'sha256').hexdigest()
    return dict(source=str(path), source_sha256=file_sha,
                s_payload_sha256=source.hexdigest(), decoded_sha256=decoded.hexdigest(),
                file_bytes=path.stat().st_size, s_start=start, s_bytes=LAYERS * ROWS * ROW_BYTES,
                layers=LAYERS, rows_per_layer=ROWS, row_bytes=ROW_BYTES, page_bytes=page_bytes,
                codec='zstd', level=level, transform=transform, page_count=count, zero_pages=zero_pages,
                compressed_bytes=compressed, directory_bytes=count * 8,
                physical_bytes=compressed + count * 8, min_page_bytes=minimum,
                max_page_bytes=maximum, layer_compressed_bytes=layer_sizes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('state', type=Path)
    parser.add_argument('--page-bytes', type=int, required=True)
    parser.add_argument('--level', type=int, default=1)
    parser.add_argument('--transform', choices=('identity', 'byte-shuffle', 'bit-shuffle'), default='identity')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert ROW_BYTES % args.page_bytes == 0 and args.page_bytes >= 4096
    result = inspect(args.state, args.page_bytes, args.level, args.transform)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('page_bytes', 's_bytes', 'compressed_bytes', 'physical_bytes', 'zero_pages', 'page_count', 's_payload_sha256')}))

if __name__ == '__main__':
    main()

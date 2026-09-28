#!/usr/bin/env python3
"""Acquire and check the separate Qwen3.6 MoE MTP head without rewriting the target."""

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import mmap
import os
from pathlib import Path
import struct
import sys
import time
import urllib.request

DATA = Path('../../data/qwen-moe')
TRUNK = DATA / 'Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
NAME = 'mtp-Qwen3.6-35B-A3B-NVFP4.gguf'
REPO = 'LibertAIDAI/Qwen3.6-35B-A3B-NVFP4-MTP-GGUF'
REVISION = '5d65e0414ad42cbf5aededa15f509d7c9e90a193'
SIZE = 3_735_545_952
SHA256 = 'c976a2de32fadc7a5632f2dcbb01563b9ae660bcb1ea42b55188e0ddc1057a17'
TRUNK_SIZE = 22_134_528_992
TRUNK_SHA256 = 'ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def value(buf, offset, kind):
    width = struct.calcsize('<' + kind)
    return struct.unpack_from('<' + kind, buf, offset)[0], offset + width


def string(buf, offset):
    size, offset = value(buf, offset, 'Q')
    if size > 1 << 30 or offset + size > len(buf):
        raise ValueError('Invalid GGUF string size')
    return bytes(buf[offset:offset + size]), offset + size


def field(buf, offset, kind):
    scalars = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
               6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
    if kind == 8:
        return string(buf, offset)
    if kind == 9:
        element, offset = value(buf, offset, 'I')
        count, offset = value(buf, offset, 'Q')
        if count > 1 << 26:
            raise ValueError('Invalid GGUF array length')
        if element in scalars:
            size = struct.calcsize('<' + scalars[element]) * count
            if offset + size > len(buf):
                raise ValueError('Truncated GGUF array')
            return bytes(buf[offset:offset + size]), offset + size
        if element != 8:
            raise ValueError(f'Unsupported GGUF array type {element}')
        h = hashlib.sha256()
        for _ in range(count):
            part, offset = string(buf, offset)
            h.update(struct.pack('<Q', len(part)))
            h.update(part)
        return (count, h.hexdigest()), offset
    if kind not in scalars:
        raise ValueError(f'Unsupported GGUF value type {kind}')
    return value(buf, offset, scalars[kind])


def header(path):
    with path.open('rb') as source, mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as buf:
        if buf[:4] != b'GGUF':
            raise ValueError(f'Not a GGUF: {path}')
        version, offset = value(buf, 4, 'I')
        if version != 3:
            raise ValueError(f'Unsupported GGUF version {version}')
        count, offset = value(buf, offset, 'Q')
        nfields, offset = value(buf, offset, 'Q')
        if count > 100_000 or nfields > 100_000:
            raise ValueError('Implausible GGUF header counts')
        metadata = {}
        for _ in range(nfields):
            key, offset = string(buf, offset)
            kind, offset = value(buf, offset, 'I')
            val, offset = field(buf, offset, kind)
            name = key.decode()
            if name in metadata:
                raise ValueError(f'Duplicate GGUF field {name}')
            metadata[name] = val
        tensors = {}
        for _ in range(count):
            name, offset = string(buf, offset)
            rank, offset = value(buf, offset, 'I')
            if rank > 4:
                raise ValueError('Invalid GGUF tensor rank')
            shape = []
            for _ in range(rank):
                dim, offset = value(buf, offset, 'Q')
                shape.append(dim)
            kind, offset = value(buf, offset, 'I')
            position, offset = value(buf, offset, 'Q')
            name = name.decode()
            if name in tensors:
                raise ValueError(f'Duplicate GGUF tensor {name}')
            tensors[name] = [shape, kind, position]
        alignment = metadata.get('general.alignment', 32)
        start = (offset + alignment - 1) // alignment * alignment
        return {'metadata': metadata, 'tensors': tensors, 'data_offset': start}


def inspect(trunk, draft):
    if trunk.stat().st_size != TRUNK_SIZE:
        raise ValueError('Target GGUF does not have its pinned size')
    if draft.stat().st_size != SIZE:
        raise ValueError('MTP GGUF is incomplete')
    target, head = header(trunk), header(draft)
    tm, hm = target['metadata'], head['metadata']
    if tm.get('general.architecture') != b'qwen35moe' or hm.get('general.architecture') != b'qwen35moe':
        raise ValueError('Target and MTP must both have qwen35moe architecture')
    if tm.get('qwen35moe.block_count') != 40 or hm.get('qwen35moe.block_count') != 41 or hm.get('qwen35moe.nextn_predict_layers') != 1:
        raise ValueError('Expected 40 target layers and one extra MTP layer')
    independent = {'qwen35moe.block_count', 'qwen35moe.nextn_predict_layers',
                   'tokenizer.ggml.padding_token_id', 'tokenizer.chat_template'}
    keys = [key for key in hm if key.startswith(('qwen35moe.', 'tokenizer.')) and key not in independent]
    differences = [key for key in keys if key not in tm or tm[key] != hm[key]]
    if differences:
        raise ValueError(f'Head metadata differs from target: {differences}')
    names = set(head['tensors'])
    shared = {'output.weight', 'output_norm.weight', 'token_embd.weight'}
    if len(names) != 23 or not shared <= names or any(not name.startswith('blk.40.') for name in names - shared):
        raise ValueError('MTP tensor set is not a separate 23-tensor layer-40 head')
    mandatory = {'blk.40.nextn.eh_proj.weight', 'blk.40.nextn.enorm.weight',
                 'blk.40.nextn.hnorm.weight', 'blk.40.nextn.shared_head_norm.weight',
                 'blk.40.ffn_down_exps.weight', 'blk.40.ffn_gate_exps.weight',
                 'blk.40.ffn_up_exps.weight', 'blk.40.attn_q.weight',
                 'blk.40.attn_k.weight', 'blk.40.attn_v.weight'}
    if not mandatory <= names:
        raise ValueError(f'MTP tensors missing: {sorted(mandatory - names)}')
    return {'architecture': 'qwen35moe', 'trunk_layers': 40, 'mtp_layers': 1,
            'draft_tensor_count': len(names), 'shared_metadata_keys': len(keys),
            'trunk_size': trunk.stat().st_size, 'draft_size': draft.stat().st_size}


def download(path):
    url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{NAME}'
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.partial')
    with (path.parent / '.mtp.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists():
            return verify(path)
        start = partial.stat().st_size if partial.exists() else 0
        if start > SIZE:
            raise ValueError(f'Oversized partial MTP file: {start}')
        if start < SIZE:
            chunk_size = 64 * 1024 * 1024
            parts = path.parent / '.mtp-parts'
            parts.mkdir(exist_ok=True)
            ranges = [(offset, min(offset + chunk_size, SIZE) - 1)
                      for offset in range(start, SIZE, chunk_size)]

            def fetch(item):
                offset, end = item
                target = parts / str(offset)
                length = end - offset + 1
                if target.exists() and target.stat().st_size == length:
                    return length
                request = urllib.request.Request(
                    url + f'?download=true&part={offset}',
                    headers={'Range': f'bytes={offset}-{end}', 'Accept-Encoding': 'identity'})
                for attempt in range(3):
                    try:
                        with urllib.request.urlopen(request, timeout=90) as response:
                            expected = f'bytes {offset}-{end}/{SIZE}'
                            if response.status != 206 or response.headers.get('Content-Range') != expected:
                                raise ValueError(f'Bad range response: {response.status}, {response.headers.get("Content-Range")}')
                            temp = target.with_suffix('.partial')
                            with temp.open('wb') as out:
                                while block := response.read(1024 * 1024):
                                    out.write(block)
                            if temp.stat().st_size != length:
                                raise ValueError(f'Short range {offset}: {temp.stat().st_size} of {length}')
                            os.replace(temp, target)
                            return length
                    except (OSError, ValueError) as error:
                        print(f'MTP range {offset} attempt {attempt + 1}: {error}', file=sys.stderr, flush=True)
                        if attempt == 2:
                            raise
                        time.sleep(attempt + 1)

            completed = start
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                for future in concurrent.futures.as_completed([pool.submit(fetch, item) for item in ranges]):
                    completed += future.result()
                    print(f'MTP {completed}/{SIZE} bytes ({completed / SIZE:.1%})', file=sys.stderr, flush=True)
            with partial.open('ab') as out:
                for offset, _ in ranges:
                    with (parts / str(offset)).open('rb') as source:
                        while block := source.read(8 * 1024 * 1024):
                            out.write(block)
                out.flush()
                os.fsync(out.fileno())
        inspect(TRUNK, partial)
        actual = digest(partial)
        if actual != SHA256:
            raise ValueError(f'MTP hash mismatch: {actual}; partial retained for repair')
        os.replace(partial, path)
        parts = path.parent / '.mtp-parts'
        if parts.exists():
            for item in parts.iterdir():
                item.unlink()
            parts.rmdir()
        return verify(path)


def verify(path):
    detail = inspect(TRUNK, path)
    actual = digest(path)
    if actual != SHA256:
        raise ValueError(f'MTP hash mismatch: {actual}')
    receipt = {'repository': REPO, 'revision': REVISION, 'filename': NAME,
               'sha256': SHA256, 'size': SIZE, 'path': str(path.resolve()),
               'target': str(TRUNK), 'target_sha256': TRUNK_SHA256,
               'target_size': TRUNK_SIZE, **detail}
    dest = path.parent / 'receipt.json'
    temp = dest.with_suffix('.tmp')
    temp.write_text(json.dumps(receipt, indent=2) + '\n')
    os.replace(temp, dest)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['inspect', 'verify', 'acquire'])
    parser.add_argument('--draft', type=Path, default=DATA / 'mtp' / NAME)
    args = parser.parse_args()
    try:
        result = (inspect(TRUNK, args.draft) if args.action == 'inspect' else
                  verify(args.draft) if args.action == 'verify' else download(args.draft))
        print(json.dumps(result, indent=2))
    except (OSError, ValueError) as error:
        print(f'mtp: {error}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())

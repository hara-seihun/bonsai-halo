#!/usr/bin/env python3
"""Inventory the GGUF's actual expert bytes and a one-token weight-traffic budget."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct

DEFAULT_CONSTANTS = Path('../bonsai-hip/gguf-py/gguf/constants.py')
FORMATS = {0:'B', 1:'b', 2:'H', 3:'h', 4:'I', 5:'i', 6:'f', 7:'?', 10:'Q', 11:'q', 12:'d'}


def scan(path, constants):
    with path.open('rb') as f:
        def scalar(fmt):
            n = struct.calcsize('<' + fmt)
            b = f.read(n)
            if len(b) != n:
                raise ValueError('Truncated GGUF metadata')
            return struct.unpack('<' + fmt, b)[0]
        def string(keep=True):
            n = scalar('Q')
            if keep:
                b = f.read(n)
                if len(b) != n:
                    raise ValueError('Truncated GGUF string')
                return b.decode('utf-8')
            f.seek(n, 1)
        def value(kind, keep=True):
            if kind in FORMATS:
                v = scalar(FORMATS[kind])
                return v if keep else None
            if kind == 8:
                return string(keep)
            if kind == 9:
                child, n = scalar('I'), scalar('Q')
                if not keep and child in FORMATS:
                    f.seek(n * struct.calcsize('<' + FORMATS[child]), 1)
                    return None
                if keep:
                    return [value(child) for _ in range(n)]
                for _ in range(n):
                    value(child, False)
                return None
            raise ValueError(f'Unknown GGUF value type {kind}')
        if f.read(4) != b'GGUF' or scalar('I') != 3:
            raise ValueError('Expected GGUF v3')
        nt, nkv = scalar('Q'), scalar('Q')
        metadata = {}
        for _ in range(nkv):
            name, kind = string(), scalar('I')
            keep = name.startswith(('general.', 'qwen35moe.'))
            v = value(kind, keep)
            if keep:
                metadata[name] = v
        tensors = []
        for _ in range(nt):
            name, nd = string(), scalar('I')
            shape = [scalar('Q') for _ in range(nd)]
            kind, offset = scalar('I'), scalar('Q')
            qt = constants.GGMLQuantizationType(kind)
            block, size = constants.GGML_QUANT_SIZES[qt]
            count = math.prod(shape)
            if count % block:
                raise ValueError(f'Invalid quantized tensor dimensions: {name}')
            tensors.append(dict(name=name, shape=shape, type=qt.name, elements=count,
                                bytes=count // block * size, offset=offset))
        header_end = f.tell()
    return metadata, tensors, header_end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--constants', type=Path, default=DEFAULT_CONSTANTS)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bandwidth-gbs', type=float, default=242.)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('gguf_constants', args.constants)
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)
    meta, tensors, header_end = scan(args.model, constants)
    if meta.get('general.architecture') != 'qwen35moe':
        raise ValueError('Expected qwen35moe architecture')
    experts = meta['qwen35moe.expert_count']
    selected = meta['qwen35moe.expert_used_count']
    banks = [t for t in tensors if '_exps.' in t['name']]
    if not banks or any(len(t['shape']) != 3 or t['shape'][2] != experts for t in banks):
        raise ValueError('Unknown expert-bank layout')
    routed = sum(t['bytes'] for t in banks)
    # Per-token embedding lookup does not read the full vocabulary embedding image.
    embeddings = [t for t in tensors if t['name'] == 'token_embd.weight']
    if len(embeddings) != 1:
        raise ValueError('Expected one embedding tensor')
    embedding = embeddings[0]
    others = [t for t in tensors if t not in banks and t != embedding]
    active = routed * selected // experts
    fixed = sum(t['bytes'] for t in others)
    lookup = embedding['bytes'] // embedding['shape'][1]
    all_bytes = active + fixed + lookup
    per_layer = {}
    for t in banks:
        layer = t['name'].split('.')[1]
        item = per_layer.setdefault(layer, {'bank_bytes': 0, 'selected_bytes': 0, 'tensors': []})
        item['bank_bytes'] += t['bytes']
        item['selected_bytes'] += t['bytes'] * selected // experts
        item['tensors'].append({'name': t['name'], 'type': t['type'], 'bytes': t['bytes']})
    h = hashlib.sha256()
    with args.model.open('rb') as f:
        remaining = header_end
        while remaining:
            data = f.read(min(remaining, 1024 * 1024))
            h.update(data)
            remaining -= len(data)
    result = dict(metadata=meta, header_bytes=header_end, header_sha256=h.hexdigest(),
                  constants_path=str(args.constants),
                  constants_sha256=hashlib.sha256(args.constants.read_bytes()).hexdigest(),
                  expert_count=experts, experts_per_token=selected,
                  tensor_payload_bytes=sum(t['bytes'] for t in tensors),
                  routed_bank_bytes=routed, active_routed_bytes_per_token=active,
                  nonexpert_nonembedding_bytes=fixed, embedding_lookup_bytes=lookup,
                  one_token_weight_stream_bytes=all_bytes,
                  bandwidth_gbs=args.bandwidth_gbs,
                  weight_stream_ms=all_bytes / (args.bandwidth_gbs * 1e6),
                  weight_only_tokens_per_second=args.bandwidth_gbs * 1e9 / all_bytes,
                  assumptions=['One complete read of each selected routed expert and nonexpert tensor per token.',
                               'Full vocabulary output head; embedding is one row.',
                               'No cross-token weight residency; all listed nonexpert tensors charged.',
                               'Excludes KV/state traffic, arithmetic, routing, launches and synchronization.',
                               'Conditional traffic budget, not a universal lower bound or measured rate.'],
                  layers=per_layer, tensors=tensors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['tensor_payload_bytes','routed_bank_bytes',
                     'active_routed_bytes_per_token','nonexpert_nonembedding_bytes',
                     'one_token_weight_stream_bytes','weight_stream_ms','weight_only_tokens_per_second']}, indent=2))


if __name__ == '__main__':
    main()

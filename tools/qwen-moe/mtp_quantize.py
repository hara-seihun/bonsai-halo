#!/usr/bin/env python3
"""Rebuild the standalone Qwen3.6 MTP Q4_K_M draft, never the target trunk."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from mtp import DATA, NAME, SHA256, header, verify

SOURCE = DATA / 'mtp' / NAME
DEST = DATA / 'mtp' / 'mtp-Qwen3.6-35B-A3B-Q4_K_M.gguf'
EXPECTED_SIZE = 1_260_898_400
EXPECTED_SHA256 = '6ec218ee63c6c2ad94981cbf7dbcec8f019a7744952602507c5861ad1362c79f'
BINARY = DATA / 'runtime/current/bin/llama-quantize'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check(path):
    if path.stat().st_size != EXPECTED_SIZE or digest(path) != EXPECTED_SHA256:
        raise ValueError('Quantized MTP image fails pinned size or SHA-256')
    source, result = header(SOURCE), header(path)
    if source['metadata'].get('general.architecture') != result['metadata'].get('general.architecture') or source['metadata'].get('qwen35moe.nextn_predict_layers') != result['metadata'].get('qwen35moe.nextn_predict_layers'):
        raise ValueError('Quantized head lost MTP model metadata')
    if source['tensors'].keys() != result['tensors'].keys():
        raise ValueError('Quantized head lost or added tensors')
    return {'path': str(path.resolve()), 'size': EXPECTED_SIZE, 'sha256': EXPECTED_SHA256,
            'source_sha256': SHA256, 'tensor_count': len(result['tensors'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'verify'])
    parser.add_argument('--binary', type=Path, default=BINARY)
    args = parser.parse_args()
    try:
        verify(SOURCE)
        if args.action == 'build' and not DEST.exists():
            temp = DEST.with_suffix(DEST.suffix + '.partial')
            if temp.exists():
                raise ValueError(f'Incomplete quantization output needs inspection: {temp}')
            cmd = [str(args.binary), str(SOURCE), str(temp), 'Q4_K_M', '8']
            with (DATA / 'mtp' / 'quantize.log').open('wb') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            check(temp)
            os.replace(temp, DEST)
        result = check(DEST)
        result['binary'] = str(args.binary.resolve())
        result['binary_sha256'] = digest(args.binary)
        receipt = DEST.parent / 'quantization.json'
        temp = receipt.with_suffix('.tmp')
        temp.write_text(json.dumps(result, indent=2) + '\n')
        os.replace(temp, receipt)
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'mtp quantize: {error}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())

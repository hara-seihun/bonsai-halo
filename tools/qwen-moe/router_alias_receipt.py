#!/usr/bin/env python3
"""Rebase no-cut router allocation logs and certify cross-row write/read overlap."""
import argparse
import hashlib
import json
import re
from pathlib import Path

PATTERN = re.compile(r'router_alloc stage=(\d) name=(ffn_moe_\S+-0) op=(\d+) shape=(\d+),(\d+) bytes=(\d+) data=(0x[0-9a-f]+) buffer=(0x[0-9a-f]+)')
WANTED = ('logits', 'probs', 'argsort', 'weights', 'weights_norm')


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def analyze(log):
    rows = {}
    for match in PATTERN.finditer(log):
        stage, name, op, dim0, dim1, size, pointer, buffer = match.groups()
        if name not in ('ffn_moe_' + key + '-0' for key in WANTED):
            continue
        rows.setdefault(int(stage), {})[name] = dict(op=int(op), shape=[int(dim0), int(dim1)], bytes=int(size),
                                                   address=int(pointer, 16), buffer=buffer)
    assert set(rows) == {1, 2}
    result = {}
    for stage, tensors in rows.items():
        assert len(tensors) == len(WANTED)
        origin = tensors['ffn_moe_logits-0']['address']
        buffers = {value['buffer'] for value in tensors.values()}
        assert len(buffers) == 1
        result[str(stage)] = {name: {key: value for key, value in tensor.items() if key not in ('address', 'buffer')} |
                              {'offset': tensor['address'] - origin} for name, tensor in tensors.items()}
    one = result['1']
    two = result['2']
    assert one['ffn_moe_logits-0']['bytes'] == 1024 and two['ffn_moe_logits-0']['bytes'] == 2048
    assert one['ffn_moe_weights_norm-0']['offset'] == 1024
    assert two['ffn_moe_weights_norm-0']['offset'] == 1024
    assert two['ffn_moe_weights_norm-0']['bytes'] == 64
    assert two['ffn_moe_weights-0']['offset'] >= 2048
    assert two['ffn_moe_argsort-0']['offset'] >= 2048
    assert '"width":2,"offset":0,"row":1,"bit_differences":248319' in log
    return dict(tensors=result, cross_row_collision=dict(writer='ffn_moe_weights_norm-0 row 0',
        reader='ffn_moe_logits-0 row 1', overlapping_offsets=[1024, 1056],
        contract='The fused kernel maps one 1024-byte logit row per warp and writes 32 bytes of normalized weights per row. Four warps occupy one CTA; there is no cross-warp read-before-write barrier. A row-0 write could overwrite row-1 logits before its warp loads them. The existing fusion memory guard correctly rejects this in-place multirow graph. Single-row logits end at offset 1024 and do not overlap the output.'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('log', type=Path)
    p.add_argument('source', type=Path)
    p.add_argument('binary', type=Path)
    p.add_argument('model', type=Path)
    p.add_argument('selected_runtime', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    receipt = dict(contract='No-cut callback; 4-token selected-target list prefix and 2-token causal replay at context 2048, batch 512, ubatch 256, eight CPU threads and flash attention. Pointer offsets are rebased independently per stage; this is an allocator and data-race certificate, not a native performance claim.',
        prompt='Write 100 consecutive integers starting at 1, each on its own line, without commentary.',
        native_source_revision='3f552c2ef572707f05746bdc84c1788792523d62',
        log_sha256=sha(args.log), source_sha256=sha(args.source), binary_sha256=sha(args.binary),
        model_sha256=sha(args.model), selected_runtime_sha256=sha(args.selected_runtime),
        analyzer_sha256=sha(Path(__file__)), result=analyze(args.log.read_text()))
    args.output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt['result']['cross_row_collision'], indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Record the serial-prefix intervention on the Qwen target width fork."""
import hashlib
import json
from pathlib import Path

ROOT = Path('../../data/qwen-moe')
PANEL = ROOT / 'acceptance/mtp-anchor'
SOURCE = Path(__file__).resolve().parent
RUNTIME = ROOT / 'runtime/81326f729efd7b5bfa2ea5461e9027e43dcc9480/bin'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.startswith('{')]


current = rows(PANEL / 'anchored16.stdout')
previous = rows(ROOT / 'acceptance/mtp-fork/fused16.stdout')
serial = [r for r in current if r['width'] == 4 and r['row'] <= 11]
first = next(r for r in current if r['width'] == 4 and r['row'] == 12)
control = [r for r in current if r['width'] == 1]
assert len(serial) == 12 and all(r['bit_differences'] == 0 for r in serial)
assert len(control) == 17 and all(r['bit_differences'] == 0 for r in control)
assert first['bit_differences'] == 248320
assert next(r for r in previous if r['width'] == 4 and r['offset'] == 0 and r['row'] == 12)['bit_differences'] == 248320
assert (PANEL / 'anchored16.tokens').read_bytes() == (ROOT / 'acceptance/mtp-fork/fused16.tokens').read_bytes()
wrapper = (PANEL / 'anchored16.wrapper').read_text()
assert 'Removed ' in wrapper and 'Running as unit: bonsai-halo.service' in wrapper
names = ('anchored16.stdout', 'anchored16.wrapper', 'anchored16.tokens',
         'failed-auto-position.stdout', 'failed-auto-position.wrapper', 'failed-auto-position.tokens',
         'failed-explicit-position.stdout', 'failed-explicit-position.wrapper', 'failed-explicit-position.tokens',
         'diagnostic-sync.stdout', 'diagnostic-sync.wrapper', 'diagnostic-sync.tokens', 'batch-logits-anchor')
receipt = {
    'result': 'serial prefix rows 1-11 match all logit bits; first width-four call at row 12 changes all 248320',
    'selected': False,
    'source_revision': '81326f729efd7b5bfa2ea5461e9027e43dcc9480',
    'model_sha256': 'ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61',
    'serial_rows': len(serial) - 1,
    'first_changed': first,
    'command': "tools/run-batch-compare --runtime-max 43s --memory-gib 30 --host-reserve-gib 4 --exec env QWEN_SERIAL_ANCHOR=1 LD_LIBRARY_PATH=<packaged runtime bin> batch-logits-anchor <pinned GGUF> 'Write 100 consecutive integers starting at 1, each on its own line, without commentary.' 16 anchored16",
    'source': {p.name: digest(p) for p in (SOURCE / 'batch_logits.cpp', SOURCE / 'mtp_anchor.py')},
    'runtime': {p.name: digest(p) for p in (RUNTIME / 'libllama.so.0.2.0', RUNTIME / 'libggml-hip.so.0.21.0')},
    'previous_fused16_sha256': digest(ROOT / 'acceptance/mtp-fork/fused16.stdout'),
    'files': {name: digest(PANEL / name) for name in names},
}
(PANEL / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'serial_equal_rows': len(serial) - 1, 'first_changed': first['row'], 'changed_bits': first['bit_differences']}))

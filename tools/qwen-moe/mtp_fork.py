#!/usr/bin/env python3
"""Certify the matched, no-cut 16-step Qwen target width/fusion panel."""
import hashlib
import json
from pathlib import Path

DATA = Path('../../data/qwen-moe')
PANEL = DATA / 'acceptance/mtp-fork'
RUNTIME = DATA / 'runtime/81326f729efd7b5bfa2ea5461e9027e43dcc9480/bin'
DRIVER = DATA / 'acceptance/router-alias/batch-logits-alloc'
PRIOR = DATA / 'acceptance/router-alias/view-96.stdout'
FIELDS = ('width', 'offset', 'row')

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def rows(path):
    output = [json.loads(line) for line in path.read_text().splitlines() if line.startswith('{')]
    indexed = {tuple(r[k] for k in FIELDS): r for r in output}
    assert len(output) == len(indexed)
    return indexed

fused, unfused, prior = (rows(PANEL / name) for name in
    ('fused16.stdout', 'unfused16.stdout', '../router-alias/view-96.stdout'))
assert fused.keys() == unfused.keys()
assert (PANEL / 'fused16.tokens').read_bytes() == (PANEL / 'unfused16.tokens').read_bytes()
assert (PANEL / 'fused16.tokens').read_bytes() == b''.join(
    (line + b'\n') for line in (DATA / 'acceptance/router-alias/view-96.tokens').read_bytes().splitlines()[:16])
summary = {}
for label, panel in (('fused', fused), ('unfused', unfused)):
    out = {}
    for w, o in ((1, 0), (2, 0), (2, 1), (4, 0), (4, 1), (4, 2), (4, 3), (8, 0)):
        seq = [panel[w, o, r] for r in range(17)]
        changed = [r for r in seq if r['bit_differences']]
        out[f'{w}:{o}'] = {'changed_rows': len(changed),
                          'first_changed': changed[0]['row'] if changed else None,
                          'first_changed_bits': changed[0]['bit_differences'] if changed else 0}
    summary[label] = out
    assert [out[f'4:{o}']['first_changed'] for o in range(4)] == [12] * 4
    assert [out[f'4:{o}']['first_changed_bits'] for o in range(4)] == [248320] * 4
    assert out['8:0']['first_changed'] == 1 and out['8:0']['first_changed_bits'] == 248320
    assert all(out[f'2:{o}']['changed_rows'] == 0 for o in (0, 1))
    assert out['1:0']['changed_rows'] == 0
    assert all((panel[key]['bit_differences'] != 0) == (prior[key]['bit_differences'] != 0) for key in panel)

files = ['fused16.stdout', 'fused16.wrapper', 'fused16.tokens',
         'unfused16.stdout', 'unfused16.wrapper', 'unfused16.tokens']
for name in ('fused16.wrapper', 'unfused16.wrapper'):
    log = (PANEL / name).read_text()
    assert 'Running as unit: bonsai-halo.service' in log and 'Removed ' in log
receipt = {
    'selected': False, 'steps': 16, 'generated_ids_identical': True,
    'model_sha256': 'ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61',
    'runtime_source_revision': '81326f729efd7b5bfa2ea5461e9027e43dcc9480',
    'driver': {'path': str(DRIVER), 'sha256': digest(DRIVER)},
    'receipt_source_sha256': digest(Path(__file__)),
    'runtime_libraries': {p.name: digest(p) for p in (RUNTIME / 'libllama.so.0.2.0', RUNTIME / 'libggml-hip.so.0.21.0')},
    'prior_96_sha256': digest(PRIOR), 'summary': summary,
    'files': {name: digest(PANEL / name) for name in files},
    'command': "tools/run-batch-compare --runtime-max 40s --memory-gib 30 --host-reserve-gib 4 --exec env LD_LIBRARY_PATH=<packaged 81326f7 bin> [GGML_CUDA_DISABLE_FUSION=1] <batch-logits-alloc> <pinned GGUF> 'Write 100 consecutive integers starting at 1, each on its own line, without commentary.' 16 <output prefix>",
}
(PANEL / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(summary, indent=2))

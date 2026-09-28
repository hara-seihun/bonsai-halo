#!/usr/bin/env python3
"""Certify the bounded Qwen multirow router lifetime experiment from retained evidence."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess

DATA = Path('../../data/qwen-moe')
BASE = DATA / 'acceptance/router-alias'
SOURCE = '81326f729efd7b5bfa2ea5461e9027e43dcc9480'
PACKAGE = DATA / 'runtime' / SOURCE
REFERENCE = DATA / 'short-j16/installed-465'
MODEL = DATA / 'Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
TRACE = next((BASE / 'width8-trace').glob('*/*kernel_trace.csv'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


rows = [json.loads(line) for line in (BASE / 'view-width8.stdout').read_text().splitlines()
        if line.startswith('{"width"')]
assert len(rows) == 40 and {r['width'] for r in rows} == {1, 2, 4, 8}
assert sum(r['bit_differences'] for r in rows) == 0
assert [sum(r['width'] == w for r in rows) for w in (1, 2, 4, 8)] == [5, 10, 20, 5]
long_rows = [json.loads(line) for line in (BASE / 'view-96.stdout').read_text().splitlines()
             if line.startswith('{"width"')]
assert len(long_rows) == 776
changed = {(width, offset): [r['row'] for r in long_rows if r['width'] == width and
            r['offset'] == offset and r['bit_differences']]
           for width, offset in ((1, 0), (2, 0), (2, 1), (4, 0), (4, 1), (4, 2), (4, 3), (8, 0))}
assert not changed[(1, 0)] and not changed[(2, 0)] and not changed[(2, 1)]
assert all(changed[(4, offset)][0] == 12 and len(changed[(4, offset)]) == 85
           for offset in range(4))
assert changed[(8, 0)][0] == 1 and len(changed[(8, 0)]) == 96
assert all(any(r['row'] == 71 and r['plain_top'] == 4980 and r['batch_top'] == 33565
               for r in long_rows if r['width'] == 4 and r['offset'] == offset)
           for offset in range(4))
assert digest(BASE / 'packaged-accept.f32') == digest(REFERENCE.with_suffix('.f32'))
assert digest(BASE / 'packaged-accept.tokens') == digest(REFERENCE.with_suffix('.tokens'))
assert (BASE / 'packaged-accept.f32').stat().st_size == 32 * 248320 * 4
assert json.loads((PACKAGE / 'runtime.json').read_text())['source_revision'] == SOURCE
assert digest(MODEL) == 'ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61'
with TRACE.open() as stream:
    names = [row['Kernel_Name'] for row in csv.DictReader(stream)]
counts = {key: sum(key in name for name in names)
          for key in ('topk_moe_cuda<256', 'soft_max_f32<true, 256', 'k_argsort_f32_i32')}
assert counts == {'topk_moe_cuda<256': 120, 'soft_max_f32<true, 256': 80, 'k_argsort_f32_i32': 80}
files = [BASE / p for p in ('view-width8.stdout', 'view-width8.stderr', 'view-width8.wrapper',
         'view-alias.log', 'width8-trace.wrapper', 'packaged-accept.f32', 'packaged-accept.tokens',
         'packaged-accept.wrapper', 'view-accept.f32', 'view-accept.tokens', 'view-accept.wrapper',
         'view-accept8.f32', 'view-accept8.tokens', 'view-accept8.wrapper',
         'control-bench.jsonl', 'candidate-bench.jsonl', 'control-p8.jsonl', 'candidate-p8.jsonl',
         'control-p8-reverse.jsonl', 'candidate-p8-reverse.jsonl',
         'view-96.stdout', 'view-96.stderr', 'view-96.wrapper',
         'view-96-alloc.stdout', 'view-96-alloc.stderr', 'view-96-alloc.wrapper',
         'batch-logits-alloc')]
files.extend((TRACE, PACKAGE / 'source.bundle', PACKAGE / 'runtime.json',
              PACKAGE / 'bin/libllama.so.0.2.0', PACKAGE / 'bin/libggml-hip.so.0.21.0',
              MODEL, Path(__file__)))
for path in files:
    assert path.is_file(), path
receipt = {
    'source_revision': SOURCE, 'accepted': False, 'installed': False,
    'model_sha256': digest(MODEL), 'support_source_revision': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'contract': '465-token installed reference logits; no-cut widths 1/2/4/8; only 2..8 route graph changed',
    'width_rows': len(rows), 'different_logit_values': sum(r['bit_differences'] for r in rows),
    'reference_logit_values': 32 * 248320, 'long_panel_rows': len(long_rows),
    'long_panel_changed_rows': {f'{w}:{o}': len(value) for (w, o), value in changed.items()},
    'long_panel_first_changed_row': {f'{w}:{o}': value[0] if value else None
                                     for (w, o), value in changed.items()},
    'long_panel_bit_differences': sum(r['bit_differences'] for r in long_rows),
    'long_panel_argmax_forks': [(r['width'], r['offset'], r['row'], r['plain_top'], r['batch_top'])
                               for r in long_rows if r['plain_top'] != r['batch_top']],
    'profile_dispatch_counts': counts,
    'profile_note': 'Prompt prefill invokes 80 separate softmax/argsort pairs; the three one/two-row evaluations invoke 120 fused router kernels.',
    'failed_full_prompt_image_sha256': digest(BASE / 'view-accept.f32'),
    'files': {str(path.relative_to(DATA) if path.is_relative_to(DATA) else Path('source') / path.name): digest(path)
              for path in files},
}
(BASE / 'lifetime-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(BASE / 'lifetime-receipt.json')

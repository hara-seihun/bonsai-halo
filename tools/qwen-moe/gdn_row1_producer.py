#!/usr/bin/env python3
"""Join host KV scatter indices with complete saved heads/state of the split-row panel."""
import argparse
import hashlib
import json
import mmap
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gdn_state_diff import regions

ROOT = Path('../../data/qwen-moe/gdn-permutation-equiv')
MODEL = Path('../../data/qwen-moe/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf')
PATTERN = re.compile(r'ROW1_K_INDEX call=(\d+) rows=(\d+) row=(\d+) stream=(\d+) cell=(\d+) global=(\d+) kv_size=(\d+)')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def parse_indices(path):
    records = [dict(zip(('call', 'rows', 'row', 'stream', 'cell', 'global', 'kv_size'),
                        map(int, m.groups()))) for m in PATTERN.finditer(path.read_text(errors='replace'))]
    assert len(records) == 32 * 2, (path, len(records))
    calls = {}
    for record in records:
        calls.setdefault(record['call'], []).append(record)
        assert record['global'] == record['stream'] * record['kv_size'] + record['cell']
    assert [len(calls[k]) for k in sorted(calls)] in ([32, 32], [32, 31, 1])
    for call in calls.values():
        assert all(row['rows'] == len(call) and row['row'] == i for i, row in enumerate(call))
    return calls


def cells(path):
    with path.open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as blob:
        lookup = {name: (start, end) for name, start, end, _ in regions(blob)}
        return {str(slot): {kind: [int(np.count_nonzero(np.frombuffer(
            blob, dtype='<u2', count=512, offset=lookup[f'kv/{slot}/{kind}/{layer}'][0] + 12 + 9*1024)))
            for layer in range(10)] for kind in ('k', 'v')} for slot in (0, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-root', required=True, type=Path,
                        help='Research native source checkout at branch research-row1-index, with built diagnostic binaries')
    native = parser.parse_args().native_root
    prefix = 'row1-index-'
    arms = ('normal', 'swap-gather')
    parsed = {}
    for arm in arms:
        stem = prefix + arm
        log = ROOT / (stem + '.log')
        state = ROOT / (stem + '.state')
        heads = ROOT / (stem + '.f32')
        calls = parse_indices(log)
        last = calls[max(calls)]
        earlier = calls[max(calls)-1]
        parsed[arm] = dict(
            artifacts={suffix: {'bytes': (ROOT / (stem + suffix)).stat().st_size,
                                'sha256': sha(ROOT / (stem + suffix))} for suffix in ('.log', '.state', '.f32')},
            second_step_indices=calls[0],
            third_step_indices=earlier + last if len(last) == 1 else last,
            third_step_call_sizes=[len(earlier), len(last)] if len(last) == 1 else [len(last)],
            newest_cell_nonzero_halves=cells(state),
        )
        assert all(r['cell'] == 9 and r['kv_size'] == 256 for r in parsed[arm]['third_step_indices'])
    normal = parsed['normal']['third_step_indices']
    split = parsed['swap-gather']['third_step_indices']
    assert [r['stream'] for r in normal] == list(range(32))
    assert [r['stream'] for r in split] == list(range(1, 32)) + [0]
    assert [r['global'] for r in split] == [256*i+9 for i in range(1, 32)] + [9]
    for arm in ('swap-gather',):
        for kind in ('k', 'v'):
            assert parsed[arm]['newest_cell_nonzero_halves']['1'][kind] == [0] * 10
            assert all(n > 0 for n in parsed[arm]['newest_cell_nonzero_halves']['0'][kind])
    baselines = {}
    for arm in arms:
        original = ROOT / f'{arm}-gather.f32' if arm == 'normal' else ROOT / 'swap-gather.f32'
        candidate = np.memmap(ROOT / f'row1-index-{arm}.f32', dtype='<u4').reshape(3, 32, 248320)
        reference = np.memmap(original, dtype='<u4').reshape(3, 32, 248320)
        diffs = [{'step': step, 'logical_sequence': seq, 'different_words': int(np.count_nonzero(candidate[step, seq] != reference[step, seq]))}
                 for step in range(3) for seq in range(32)
                 if np.any(candidate[step, seq] != reference[step, seq])]
        baselines[arm] = {'reference_head_sha256': sha(original),
                          'reference_state_sha256': sha(original.with_suffix('.state')),
                          'head_differences': diffs,
                          'state_identical': parsed[arm]['artifacts']['.state']['sha256'] == sha(original.with_suffix('.state'))}
    assert baselines['normal']['head_differences'] == [] and baselines['normal']['state_identical']
    assert baselines['swap-gather']['head_differences'] == [{'step': 2, 'logical_sequence': 1, 'different_words': 248320}]
    micro = ROOT / 'row1-setrows.log'
    micro_rows = [json.loads(line) for line in micro.read_text().splitlines() if line.startswith('{"stream":')]
    assert len(micro_rows) == 32 and [r['stream'] for r in micro_rows] == list(range(32))
    assert all(r['matched_words'] == r['expected_word_count'] == 512 for r in micro_rows)
    receipt = dict(
        contract='Host graph-input K indices and post-state K/V cells on matched 32-logical-sequence three-step panel. This localizes index calculation, not unsaved K/V producers, GPU SET_ROWS execution or export. Diagnostic library changes no graph node or selected runtime.',
        source_sha256=sha(Path(__file__)),
        native_source_sha256=sha(native / 'src/llama-kv-cache.cpp'),
        native_library_sha256=sha(native / 'build/bin/libllama.so.0.2.0'),
        probe_sha256=sha(native / 'build/bin/qwen-row1-index'),
        model={'bytes':MODEL.stat().st_size, 'verified_sha256':json.loads((ROOT / 'receipt.json').read_text())['model']['sha256']},
        prior_panel_sha256=sha(ROOT / 'receipt.json'),
        original_panel_comparison=baselines,
        untraced_swapped_hashes={suffix:sha(ROOT / ('row1-index-untraced' + suffix)) for suffix in ('.f32', '.state', '.log')},
        isolated_set_rows={'source_sha256':sha(Path(__file__).with_suffix('.cpp')),
                           'binary_sha256':sha(native / 'build/bin/qwen-row1-setrows'),
                           'log_sha256':sha(micro), 'matching_halfwords':sum(r['matched_words'] for r in micro_rows),
                           'total_halfwords':sum(r['expected_word_count'] for r in micro_rows),
                           'rows':micro_rows},
        arms=parsed,
    )
    output = ROOT / 'row1-index-receipt.json'
    output.write_text(json.dumps(receipt, indent=2) + '\n')
    print('split first row target=265 (stream1, cell9), singleton target=9 (stream0, cell9); '
          'slot1 all twenty newest K/V cells zero despite correct host target; '+str(output))


if __name__ == '__main__':
    main()

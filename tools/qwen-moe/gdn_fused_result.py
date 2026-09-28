#!/usr/bin/env python3
"""Reconcile Qwen indexed-GDN full-head/state/rate panels and retained failed arms."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

D = Path(sys.argv[1] if len(sys.argv) > 1 else '../../data/qwen-moe/gdn-fused')
NATIVE = Path(sys.argv[2] if len(sys.argv) > 2 else '../work/clones/qwen-gdn-fused-native')
BONSAI = Path(__file__).resolve().parents[2]
REF = Path('../../data/qwen-moe/acceptance/selected/reference.f32')
PACKAGE = Path('../../data/qwen-moe/runtime/current')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while b := stream.read(8 * 1024 * 1024):
            h.update(b)
    return h.hexdigest()


def lines(name):
    return [json.loads(line) for line in (D / (name + '.jsonl')).read_text().splitlines()]


arms = ('domain-control-1', 'domain-default-1', 'domain-default-2', 'domain-control-2')
heads = {a: sha(D / (a + '.f32')) for a in arms}
assert len(set(heads.values())) == 1
steps = {}
for arm in arms:
    rows = [r for r in lines(arm) if 'stream' in r]
    assert len(rows) == 6 * 32 and all(len([r for r in rows if r['step'] == i]) == 32 for i in range(6))
    ms = [rows[i * 32]['milliseconds'] for i in range(6)]
    steps[arm] = dict(milliseconds=ms, warm_2_to_5_ms=statistics.mean(ms[2:]),
                      aggregate_tps=32000 / statistics.mean(ms[2:]))
control_ms = statistics.mean(steps[a]['warm_2_to_5_ms'] for a in arms if 'control' in a)
default_ms = statistics.mean(steps[a]['warm_2_to_5_ms'] for a in arms if 'default' in a)
for pair in (('selected-state-bytes-control', 'selected-state-bytes-default'),
             ('swap-guard-control', 'swap-guard-default')):
    assert sha(D / (pair[0] + '.f32')) == sha(D / (pair[1] + '.f32'))
    assert sha(D / (pair[0] + '.state')) == sha(D / (pair[1] + '.state'))
for name in ('domain-packaged-465', 'domain-installed-465'):
    assert json.loads((D / (name + '.json')).read_text())['prompt_tokens'] == 465
    assert sha(D / (name + '.f32')) == sha(REF)
    assert sha(D / (name + '.tokens')) == sha(REF.with_suffix('.tokens'))
assert sha(D / 'domain-installed-generated.f32') == heads[arms[0]]
assert sha(D / 'accept465-fused.f32') != sha(REF)
assert sha(D / 'swap-control.f32') != sha(D / 'swap-default.f32')
assert sha(D / 'swap-control.state') != sha(D / 'swap-default.state')


def geometry(name):
    f = next((D / name).glob('*/*kernel_trace.csv'))
    with f.open() as stream:
        rows = list(csv.DictReader(stream))
    return f, {f'{x}:{y}': sum('get_rows_float_vec' in r['Kernel_Name'] and
                            r['Grid_Size_X'] == x and r['Grid_Size_Y'] == y for r in rows)
               for x, y in [('8192', '512'), ('7936', '512'), ('256', '512')]}


trace_fused, fused_geometry = geometry('trace-fused')
trace_swap, swap_geometry = geometry('trace-swap-guard')
assert fused_geometry['8192:512'] == 30  # seed only, earlier same indexed GDN body
assert swap_geometry == {'8192:512': 30, '7936:512': 30, '256:512': 30}
bench = {}
for arm in ('solo-control', 'solo-default', 'domain-bench'):
    bench[arm] = [{k: row[k] for k in ('n_prompt', 'n_gen', 'n_depth', 'avg_ts', 'stddev_ts')}
                  for row in lines(arm)]
paths = [D / (arm + suffix) for arm in (*arms, 'domain-installed-generated')
         for suffix in ('.f32', '.jsonl')]
paths += [D / (name + suffix) for name in ('swap-guard-control', 'swap-guard-default',
                                          'swap-control', 'swap-default',
                                          'selected-state-bytes-control', 'selected-state-bytes-default')
          for suffix in ('.f32', '.state', '.jsonl')]
paths += [D / (name + suffix) for name in ('domain-packaged-465', 'domain-installed-465')
          for suffix in ('.f32', '.tokens', '.json')]
paths += [D / (name + '.log') for name in ('domain-control2-panel', 'domain-panel',
    'swap-guard-panel', 'swap-panel', 'domain-packaged-panel', 'domain-installed-panel',
    'solo-panel', 'state-bytes-panel', 'trace-swap-wrapper', 'accept465-panel')]
paths += [trace_fused, trace_swap, BONSAI / 'tools/qwen-moe/generated_stream_j16.cpp',
          BONSAI / 'tools/qwen-moe/generated_state_fused.cpp',
          BONSAI / 'tools/qwen-moe/gdn_swap_probe.cpp', Path(__file__),
          PACKAGE / 'runtime.json', PACKAGE / 'bin/libggml-hip.so', PACKAGE / 'bin/libllama.so',
          PACKAGE / 'bin/qwen-accept', D / 'qwen-generated-installed',
          NATIVE / 'src/models/qwen35moe.cpp', NATIVE / 'src/llama-graph.cpp',
          NATIVE / 'ggml/src/ggml-cuda/gated_delta_net.cu']
receipt = dict(native_revision=subprocess.check_output(['git', '-C', str(NATIVE), 'rev-parse', 'HEAD'], text=True).strip(),
    source_bonsai_revision=subprocess.check_output(['git', '-C', str(BONSAI), 'rev-parse', 'HEAD'], text=True).strip(),
    model='../../data/qwen-moe/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf',
    model_sha256='ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61',
    workload=dict(streams=32, seeded_tokens_per_stream=8, generated_steps=5,
                  requested_full_vocabulary_heads=192, vocabulary=248320,
                  context_capacity=8192, batch=512, ubatch=256, threads=8,
                  expert_J='selected J16 at 32 rows', flash_attention=True),
    full_heads_sha256=heads, steps=steps,
    control_warm_ms=control_ms, default_warm_ms=default_ms,
    control_aggregate_tps=32000 / control_ms, default_aggregate_tps=32000 / default_ms,
    wall_speedup_ratio=control_ms / default_ms,
    exact_serialized_state_sha256=sha(D / 'selected-state-bytes-control.state'),
    swapped_serialized_state_sha256=sha(D / 'swap-guard-control.state'),
    installed_465_head_sha256=sha(REF), installed_generated_head_sha256=sha(D / 'domain-installed-generated.f32'),
    rejected_multitoken_head_sha256=sha(D / 'accept465-fused.f32'),
    rejected_split_head_sha256=sha(D / 'swap-default.f32'),
    indexed_trace_seed_gathers=30, indexed_trace_generated_gathers=0,
    swapped_trace_gather_geometry=swap_geometry, benchmarks=bench,
    artifacts_sha256={str(p): sha(p) for p in paths})
(D / 'domain-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({k: receipt[k] for k in ('native_revision', 'control_aggregate_tps',
    'default_aggregate_tps', 'wall_speedup_ratio', 'swapped_trace_gather_geometry')}, indent=2))

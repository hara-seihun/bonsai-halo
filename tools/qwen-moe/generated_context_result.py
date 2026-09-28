#!/usr/bin/env python3
"""Reconcile the installed 32-stream GDN context panel without hiding seed drift."""
import hashlib
import json
from pathlib import Path
import statistics

DATA = Path('../../data/qwen-moe/gdn-context')
BINARY = DATA / 'generated-context'
SOURCE = Path(__file__).with_name('generated_context.cpp')
RUNTIME = Path('../../data/qwen-moe/runtime/current')
MODEL = Path('../../data/qwen-moe/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sample(name):
    records = [json.loads(line) for line in (DATA / (name + '.jsonl')).read_text().splitlines()]
    assert all(x['heads'] == 32 and x['vocabulary'] == 248320 for x in records)
    assert [x['step'] for x in records] == list(range(len(records)))
    assert (DATA / (name + '.f32')).stat().st_size == len(records) * 32 * 248320 * 4
    return {'steps_ms': [x['milliseconds'] for x in records],
            'warm_ms': statistics.mean(x['milliseconds'] for x in records[1:]),
            'seed_head_sha256': digest(DATA / (name + '.seed.f32')) if (DATA / (name + '.seed.f32')).exists() else None,
            'generated_heads_sha256': digest(DATA / (name + '.f32'))}


names = ['128-control', '128-fused', '512-control', '512-fused',
         '128-control-seed', '128-control-repeat', '128-fused-seed',
         '512-control-seed', '512-control-repeat', '512-fused-seed', '512-fused-repeat',
         '512-saved-seed', '512-replay-control', '512-replay-fused',
         '512-final-control', '512-final-fused']
samples = {name: sample(name) for name in names}
assert samples['128-control-repeat']['seed_head_sha256'] == samples['128-fused-seed']['seed_head_sha256']
assert samples['128-control-repeat']['generated_heads_sha256'] == samples['128-fused-seed']['generated_heads_sha256']
assert samples['128-control-seed']['seed_head_sha256'] != samples['128-control-repeat']['seed_head_sha256']
assert samples['512-control-seed']['seed_head_sha256'] == samples['512-control-repeat']['seed_head_sha256']
assert len({samples[n]['seed_head_sha256'] for n in ('512-control-seed', '512-fused-seed', '512-fused-repeat')}) == 3
assert samples['512-replay-control']['generated_heads_sha256'] == samples['512-replay-fused']['generated_heads_sha256']
assert samples['512-final-control']['generated_heads_sha256'] == samples['512-final-fused']['generated_heads_sha256']
assert digest(DATA / '512-final-control.endstate') == digest(DATA / '512-final-fused.endstate')
receipt = {
    'contract': 'Installed Qwen GGUF; 32 independent sequences; each seeded in 8-token chunks to depth 128 or 512, 32 complete vocabulary heads per generated step; 5-step speed panel and 3-step seed-provenance panel. For the deep matched panel, save one complete 512-token-per-sequence context and 32 next-token IDs and load identical state into both arms before five generated steps. Same packaged runtime, GGML_CUDA_GDN_FUSED_ROWS=0 vs default. Timings exclude seed and serialization but include every requested head and greedy feedback.',
    'runtime_revision': RUNTIME.resolve().name,
    'model': str(MODEL),
    'model_sha256': 'ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61',
    'source_sha256': digest(SOURCE), 'binary_sha256': digest(BINARY),
    'runtime_libllama_sha256': digest(RUNTIME / 'bin/libllama.so'),
    'runtime_hip_sha256': digest(RUNTIME / 'bin/libggml-hip.so'),
    'samples': samples,
    'unmatched_five_step_speed_ratio': {str(depth): samples[f'{depth}-control']['warm_ms'] / samples[f'{depth}-fused']['warm_ms'] for depth in (128, 512)},
    'matched_512_replay_speed_ratio': samples['512-replay-control']['warm_ms'] / samples['512-replay-fused']['warm_ms'],
    'matched_512_final_speed_ratio': samples['512-final-control']['warm_ms'] / samples['512-final-fused']['warm_ms'],
    'matched_512_final_endstate_sha256': digest(DATA / '512-final-control.endstate'),
    'matched_128_seed_and_generated_heads': True,
    'seed_variation': '128-control-seed vs 128-control-repeat differs; 512-fused-seed vs 512-fused-repeat differs, even before generated computation. Five-step raw control/fused heads at either depth cannot prove identity because their seed heads were not captured. The single saved 512-prefix state removes that confound: both five-step replays agree in every head bit, and the second replay pair also agrees in every serialized final state byte.',
}
paths = [SOURCE, Path(__file__), BINARY, RUNTIME / 'runtime.json', DATA / '512-fixed.state', DATA / '512-fixed.next',
         DATA / '512-final-control.endstate', DATA / '512-final-fused.endstate']
for name in names:
    paths.extend(DATA / (name + ext) for ext in ('.jsonl', '.wrapper', '.f32'))
    if (DATA / (name + '.seed.f32')).exists():
        paths.append(DATA / (name + '.seed.f32'))
receipt['artifact_sha256'] = {str(p): digest(p) for p in sorted(set(paths))}
(DATA / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'matched_512_final_speed_ratio': receipt['matched_512_final_speed_ratio'],
    'matched_512_replay_speed_ratio': receipt['matched_512_replay_speed_ratio'],
    'matched_512_head_sha256': samples['512-final-fused']['generated_heads_sha256'],
    'matched_512_endstate_sha256': receipt['matched_512_final_endstate_sha256']}, indent=2))

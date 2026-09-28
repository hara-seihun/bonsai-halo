#!/usr/bin/env python3
"""Verify the J16 24..32 native bundle before runtime selection."""
import hashlib
import json
from pathlib import Path

ROOT = Path('../../data/qwen-moe')
DATA = ROOT / 'short-j16'
REV = '3f552c2ef572707f05746bdc84c1788792523d62'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    candidate = DATA / 'accept-465-matched.f32'
    control = ROOT / 'acceptance/selected/installed.f32'
    candidate_tokens = DATA / 'accept-465-matched.tokens'
    control_tokens = ROOT / 'acceptance/selected/installed.tokens'
    if sha(candidate) != sha(control) or sha(candidate_tokens) != sha(control_tokens):
        raise ValueError('465-token prompt, 32 greedy decisions or complete logits differ')
    panel = json.loads((DATA / 'panel-a/receipt.json').read_text())
    if not panel['complete_logit_bits_equal'] or len(panel['arms']) != 4:
        raise ValueError('short natural text panel is incomplete')
    streams_control = DATA / 'streams-control.f32'
    streams_candidate = DATA / 'streams-candidate.f32'
    if sha(streams_control) != sha(streams_candidate) or streams_control.stat().st_size != 8*248320*4:
        raise ValueError('eight independent-stream full-vocabulary rows differ')
    manifest = json.loads((ROOT / 'runtime' / REV / 'runtime.json').read_text())
    if manifest['source_revision'] != REV:
        raise ValueError('packaged revision differs')
    receipt = {
        'accepted': True, 'source_revision': REV, 'default_arguments': [],
        'model_sha256': panel['model_sha256'],
        'source_sha256': sha(Path(__file__)),
        'runtime_manifest_sha256': sha(ROOT / 'runtime' / REV / 'runtime.json'),
        'runtime_hip_sha256': sha(ROOT / 'runtime' / REV / 'bin/libggml-hip.so.0.21.0'),
        'acceptance': {'prompt_tokens': 465, 'greedy_steps': 32,
            'logit_float_bits_equal': candidate.stat().st_size * 8,
            'candidate_logits': str(candidate), 'reference_logits': str(control),
            'logits_sha256': sha(candidate), 'greedy_tokens_sha256': sha(candidate_tokens),
            'natural_prompt_16_24_32_panel': str(DATA / 'panel-a/receipt.json'),
            'panel_sha256': sha(DATA / 'panel-a/receipt.json'),
            'independent_streams': 32, 'independent_steps': 8,
            'stream_logit_float_bits_equal': streams_control.stat().st_size * 8,
            'streams_sha256': sha(streams_control),
            'stream_probe_sha256': sha(DATA / 'streams'),
            'stream_probe_source_sha256': sha(Path(__file__).with_name('short_j16_streams.cpp')),
            'stream_control_jsonl_sha256': sha(DATA / 'streams-control.jsonl'),
            'stream_candidate_jsonl_sha256': sha(DATA / 'streams-candidate.jsonl')},
    }
    (DATA / 'acceptance.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()

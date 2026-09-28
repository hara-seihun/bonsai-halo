#!/usr/bin/env python3
"""Summarize the instrumented engine's measured phases, keeping tracing overhead visible."""
import argparse
import collections
import json
import statistics
from pathlib import Path

PHASES = {
    'gdn': ['input prep + alpha/beta', 'qkv + gate projection', 'conv + GDN prep', 'GDN state + replay', 'output prep', 'output projection'],
    'attention': ['input prep', 'qkv projection', 'rotary + KV prep', 'QK/softmax/AV', 'output prep', 'output projection'],
    'head': ['input prep', 'vocabulary projection', 'argmax slices'],
    'embed': ['embedding gather', 'embedding transform'],
}

# Every case axis the driver walks belongs in the group key. A panel that varies one of them and
# groups only on (mode, rows, logits) takes a median across the arms it was measuring, which reads
# as noise rather than as an effect. Absent keys fall back to the value a run without that axis had,
# so an older run groups exactly as it did before.
AXES = (('decode_tokens', 1), ('ffn_a4_sched', -1), ('ffn_a4_image', -1), ('gdn_pregate', -1),
        ('ffn_a4_order', -1), ('seq_sched', -1), ('gdn_state', 'f32'), ('head_rows', 'all'),
        ('attn_sched', 1), ('attn_narrow', 1), ('attn_hg', 1), ('seq_quant', 'a8'),
        ('ffn_slice_rows', -1), ('decode_prompt', 64), ('seq_image', -1), ('ffn_run', 0),
        ('gdn_split', -1), ('wide_min', 32))

def case_key(s):
    return (s['mode'], s['rows'], s['logits']) + tuple(s.get(k, d) for k, d in AXES)

def summarize(run):
    groups = collections.defaultdict(list)
    for s in run['samples']:
        groups[case_key(s)].append(s)
    result = []
    for key, samples in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
        mode, rows, logits = key[:3]
        axes = dict(zip([a for a, _ in AXES], key[3:]))
        detail = collections.defaultdict(list)
        for s in samples:
            if not s['profile']:
                continue
            totals = collections.Counter()
            phases = collections.Counter()
            for t in s['trace']:
                elapsed = t['end_ms'] - t['start_ms']
                if elapsed < 0:
                    raise ValueError('negative event interval')
                totals[t['kind']] += elapsed
                sequence = t['kind'] in ('sequence', 'sequence-input-prep', 'sequence-core')
                typ = ('gdn' if t['layer'] % 4 != 3 else 'attention') if sequence else t['kind']
                names = PHASES.get(typ, [])
                if t['kind'] == 'sequence-input-prep': names = names[:1]
                if t['kind'] == 'sequence-core': names = names[2:5]
                for name, us in zip(names, t['phase_us']):
                    phases[typ + ': ' + name] += us / 1000
            totals['between launches'] = s['device_span_ms'] - sum(totals.values())
            for k, v in totals.items(): detail[k].append(v)
            for k, v in phases.items(): detail[k].append(v)
        unprofiled = [s['wall_ms'] for s in samples if not s['profile']]
        profiled = [s['wall_ms'] for s in samples if s['profile']]
        spans = [s['device_span_ms'] for s in samples if s['profile']]
        result.append(dict(mode=mode, rows=rows, logits=logits, **axes,
            unprofiled_wall_ms=unprofiled, profiled_wall_ms=profiled, device_spans_ms=spans,
            residual_hashes=sorted({s['residual_fnv64'] for s in samples}),
            median_ms={k: statistics.median(v) for k,v in detail.items()},
            samples_ms=dict(detail)))
    return result

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('run'); p.add_argument('--out',required=True)
    args=p.parse_args(); source=Path(args.run); run=json.loads(source.read_text()); summary=summarize(run)
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.with_suffix('.json').write_text(json.dumps({'source':str(source.resolve()),'groups':summary},indent=2)+'\n')
    lines=['# Full-model phase profile','',f'Source: `{source.resolve()}`.', '',
        f"Prefix {run['context']} tokens, {run['rounds']} rounds. Every sample is retained; phase values are per-group medians. No cross-mode timing ratio is inferred from counter-instrumented runs.", '',
        f"Sequence layout `{run.get('sequence_layout', 'staged')}`, operand `{run.get('sequence_operand', 'int8')}`, tile width `{run.get('sequence_tile_width', 'auto')}`.", '',
        '## Tracing perturbation','', '| mode | rows | logits | arm | native wall ms | traced wall ms | traced device span ms | residual hashes |',
        '|---|---:|---|---|---|---|---|---:|']
    fmt=lambda xs: ', '.join(f'{x:.2f}' for x in xs)
    arm=lambda g: ' '.join(f'{a}={g[a]}' for a,d in AXES if g.get(a,d)!=d) or 'as configured'
    for g in summary: lines.append(f"| {g['mode']} | {g['rows']} | {g['logits']} | {arm(g)} | {fmt(g['unprofiled_wall_ms'])} | {fmt(g['profiled_wall_ms'])} | {fmt(g['device_spans_ms'])} | {len(g['residual_hashes'])} |")
    lines += ['', 'A single residual hash per group means the 64-bit residual checksums agree across these samples. Trace wall time includes collection; device spans exclude readback. Outliers remain in the list and in the median calculation.', '']
    for g in summary:
        if not g['logits']: continue
        lines += [f"## Mode {g['mode']}, {g['rows']} rows, logits on every row, {arm(g)}", '', '| phase | median ms |', '|---|---:|']
        for k,v in g['median_ms'].items(): lines.append(f'| {k} | {v:.3f} |')
        lines += ['', 'The sequence total contains the GDN and attention subphases. Do not add them twice. Barrier stamps include grid waits; event totals also include unstamped prologue and epilogue work.', '']
    out.write_text('\n'.join(lines).rstrip()+'\n')

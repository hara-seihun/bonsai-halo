#!/usr/bin/env python3
"""Audit device idle intervals in the retained counter-free Qwen decode trace."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import median

START = 9067
STRIDE = 1565
TOKENS = 8


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def short(name):
    return name.split('<', 1)[0].removeprefix('void ').split('(', 1)[0]


def panel(path):
    with path.open(newline='') as f:
        records = list(csv.DictReader(f))
    assert len(records) == START + TOKENS * STRIDE, len(records)
    rows = []
    for token in range(TOKENS):
        chunk = records[START + token * STRIDE:START + (token + 1) * STRIDE]
        assert len(chunk) == STRIDE
        assert 'mul_mat_vec_q' in chunk[-1]['Kernel_Name']
        ranked = sorted(enumerate(chunk), key=lambda pair: int(pair[1]['Start_Timestamp']))
        intervals = []
        for ordinal, ((_, before), (_, after)) in enumerate(zip(ranked, ranked[1:])):
            gap = max(0, int(after['Start_Timestamp']) - int(before['End_Timestamp']))
            if gap >= 500_000:
                intervals.append({'dispatch_ordinal': ordinal, 'idle_ms': round(gap / 1e6, 6),
                                  'before': short(before['Kernel_Name']), 'after': short(after['Kernel_Name'])})
        busy = sum(int(r['End_Timestamp']) - int(r['Start_Timestamp']) for r in chunk)
        lo = min(int(r['Start_Timestamp']) for r in chunk)
        hi = max(int(r['End_Timestamp']) for r in chunk)
        idle = sum(max(0, int(b['Start_Timestamp']) - int(a['End_Timestamp']))
                   for (_, a), (_, b) in zip(ranked, ranked[1:]))
        assert busy + idle == hi - lo, (token, busy, idle, hi-lo)
        rows.append({'token': token, 'device_busy_ms': round(busy / 1e6, 6),
                     'trace_span_ms': round((hi-lo) / 1e6, 6), 'idle_ms': round(idle / 1e6, 6),
                     'long_idle_ms': round(sum(x['idle_ms'] for x in intervals), 6),
                     'long_intervals': intervals})
    warm = rows[1:]
    sites = {}
    for row in warm:
        for item in row['long_intervals']:
            sites.setdefault(item['dispatch_ordinal'], []).append(item['idle_ms'])
    return {'trace_sha256':sha(path), 'source_sha256':sha(Path(__file__)),
            'contract':'Counter-free profiler GPU timestamps, dispatch rows 9067..21586: two depth setup rows excluded. Sorted within each 1565-dispatch decode token; intervals >=0.5 ms. These are not unprofiled wall or launch times.',
            'tokens':rows,
            'warm_medians_ms':{name:round(median(r[name] for r in warm),6) for name in
                               ('device_busy_ms','trace_span_ms','idle_ms','long_idle_ms')},
            'long_sites':[{'dispatch_ordinal':k, 'count_of_seven_warm_tokens':len(v),
                           'median_idle_ms':round(median(v),6)}
                          for k,v in sorted(sites.items())]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace',type=Path)
    p.add_argument('--out',required=True,type=Path)
    args=p.parse_args()
    result=panel(args.trace)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'warm_medians_ms':result['warm_medians_ms'],'long_sites':result['long_sites']},indent=2))


if __name__ == '__main__':
    main()

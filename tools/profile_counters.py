#!/usr/bin/env python3
"""Calibrate FETCH_SIZE against known reads before summarizing the final full-model pass.

These are requests leaving the counted cache, not proof of DRAM traffic after
other cache levels. Timing under counter collection is not native throughput.
"""
import argparse
import csv
import json
import statistics
from pathlib import Path


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def calibration(path):
    rows = [r for r in read(path) if r['Kernel_Name'].startswith('profile_read(') and r['Counter_Name']=='FETCH_SIZE']
    expected = [n for n in (1<<20, 256<<20, 1<<30) for _ in range(3)]
    if len(rows)!=len(expected):
        raise ValueError('expected nine known-byte calibration dispatches')
    ratios=[n/(float(r['Counter_Value'])*1024) for n,r in zip(expected,rows)]
    scale=statistics.median(ratios)
    if any(abs(x/scale-1)>.002 for x in ratios):
        raise ValueError('FETCH_SIZE calibration is not consistent across sizes/repetitions')
    return scale, ratios


def summarize(path, n, scale):
    if n<8 or n%8: raise ValueError('this trace join requires complete eight-row slices')
    rows=[r for r in read(path) if r['Counter_Name']=='FETCH_SIZE']
    slices=n//8
    sequence=[r for r in rows if 'k_forward_rows<8, 0>' in r['Kernel_Name']]
    count=66*slices
    if len(sequence)<count: raise ValueError('incomplete full-model trace')
    sequence=sequence[-count:]
    groups={'embed':sequence[:slices], 'head':sequence[-slices:]}
    for kind in ('gdn','attention'):
        groups[kind]=[r for l in range(64) if (l%4!=3)==(kind=='gdn') for r in sequence[(l+1)*slices:(l+2)*slices]]
    for kind,pattern in [('ffn producers','::k_prep<'),('ffn projections','::k_proj_opt<')]:
        selected=[r for r in rows if pattern in r['Kernel_Name']]
        if len(selected)<128: raise ValueError('incomplete FFN trace')
        groups[kind]=selected[-128:]
    result={}
    for kind,items in groups.items():
        raw=sum(float(r['Counter_Value'])*1024 for r in items)
        ns=sum(int(r['End_Timestamp'])-int(r['Start_Timestamp']) for r in items)
        result[kind]={'dispatches':len(items),'raw_fetch_size_bytes':raw,
            'calibrated_external_request_bytes':raw*scale,'counter_instrumented_ms':ns/1e6,
            'grid_workgroups':sorted({int(r['Grid_Size'])//int(r['Workgroup_Size']) for r in items}),
            'allocated_vgprs':sorted({int(r['VGPR_Count']) for r in items}),
            'scratch_bytes':sorted({int(r['Scratch_Size']) for r in items})}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('counters'); p.add_argument('--calibration',required=True)
    p.add_argument('--rows',type=int,default=128); p.add_argument('--out',required=True)
    args=p.parse_args(); scale,ratios=calibration(args.calibration)
    result={'counter_source':str(Path(args.counters).resolve()),'calibration_source':str(Path(args.calibration).resolve()),
        'bytes_per_reported_byte':scale,'calibration_ratios':ratios,
        'contract':'Final pass only, all 64 layers, all rows pay head. The uniform-read calibration factor is transferred to these regions, not independently established per region. External-cache requests are not a physical DRAM counter. Counter instrumentation changes scheduling.',
        'regions':summarize(args.counters,args.rows,scale)}
    Path(args.out).write_text(json.dumps(result,indent=2)+'\n')

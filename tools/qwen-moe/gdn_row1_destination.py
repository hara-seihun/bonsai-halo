#!/usr/bin/env python3
"""Test whether the missing swapped sequence-1 KV cell was merely stored elsewhere."""
import argparse
import hashlib
import json
import mmap
import struct
from pathlib import Path

import numpy as np
from gdn_state_diff import regions


ROOT = Path('../../data/qwen-moe/gdn-permutation-equiv')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def metadata(blob, start, end):
    count, = struct.unpack_from('<I', blob, start)
    pos = start + 4
    cells = []
    for _ in range(count):
        position, nseq = struct.unpack_from('<iI', blob, pos)
        x, y = struct.unpack_from('<ii', blob, pos+8)
        pos += 16
        ids = struct.unpack_from('<' + 'i'*nseq, blob, pos)
        pos += 4*nseq
        cells.append({'position':position, 'xy':[x,y], 'ids':list(ids)})
    if pos != end:
        raise ValueError('KV metadata length mismatch')
    return cells


def census(control, swapped):
    tables = [{name:(start,end,count) for name,start,end,count in regions(blob)} for blob in (control,swapped)]
    if list(tables[0]) != list(tables[1]):
        raise ValueError('KV layouts differ')
    result = {'metadata': {}, 'layers': []}
    for slot in range(32):
        name = f'kv/{slot}/meta'
        a,b,_ = tables[0][name]
        c,d,_ = tables[1][name]
        if slot in (0,1,4) or control[a:b] != swapped[c:d]:
            result['metadata'][str(slot)] = {'normal':metadata(control,a,b), 'swap':metadata(swapped,c,d)}
    for layer in range(10):
        entry = {'layer':layer}
        for kind in ('k','v'):
            name = f'kv/1/{kind}/{layer}'
            start,end,count = tables[0][name]
            if count != 10 or end-start != 12+1024*count:
                raise ValueError(f'unexpected cell size {name}')
            target = np.frombuffer(control, dtype='<u2', count=512, offset=start+12+9*1024)
            if not np.all(target):
                raise ValueError(f'normal destination not fully nonzero {name}')
            candidates = []
            for slot in range(32):
                a,b,n = tables[1][f'kv/{slot}/{kind}/{layer}']
                if n != 10 or b-a != 12+1024*n:
                    raise ValueError('unexpected candidate size')
                for token in range(10):
                    candidate = np.frombuffer(swapped, dtype='<u2', count=512, offset=a+12+token*1024)
                    equal = int(np.count_nonzero(target == candidate))
                    candidates.append((equal,slot,token))
            candidates.sort(reverse=True)
            entry[kind] = {'normal_slot1_newest_nonzero':int(np.count_nonzero(target)),
                           'swap_slot1_newest_nonzero':int(np.count_nonzero(np.frombuffer(swapped, dtype='<u2', count=512, offset=tables[1][name][0]+12+9*1024))),
                           'exact_relocations':[[slot,token] for eq,slot,token in candidates if eq == 512],
                           'closest': [{'equal_halfwords':eq,'slot':slot,'token':token} for eq,slot,token in candidates[:3]]}
        result['layers'].append(entry)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    pairs = (('normal-gather','swap-gather'),('normal','swap'))
    receipt = {'contract':'Finite whole-cell relocation of normal logical-slot-1 newest K/V among every saved cell of each matching swapped arm; no claim about unsaved device data, arbitrary transforms or graph producer.',
               'source_sha256':sha(Path(__file__)), 'parser_sha256':sha(Path(__file__).with_name('gdn_state_diff.py')),
               'panel_sha256':sha(args.root/'receipt.json'), 'states':{}, 'pairs':{}}
    for left,right in pairs:
        paths = [args.root/(arm+'.state') for arm in (left,right)]
        receipt['states'].update({arm:sha(path) for arm,path in zip((left,right),paths)})
        with paths[0].open('rb') as f, paths[1].open('rb') as g:
            with mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as a, mmap.mmap(g.fileno(),0,access=mmap.ACCESS_READ) as b:
                receipt['pairs'][f'{left} vs {right}'] = census(a,b)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(receipt,indent=2)+'\n')
    for pair, report in receipt['pairs'].items():
        print(pair, 'metadata_changed_slots',sum(m['normal'] != m['swap'] for m in report['metadata'].values()),
              'exact_relocations',sum(len(layer[kind]['exact_relocations']) for layer in report['layers'] for kind in ('k','v')),
              'closest_last',report['layers'][0]['v']['closest'][:1])


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Keep every native proposal arm and its serial equality outcome."""
import hashlib
import json
from pathlib import Path
DATA=Path('../../data/bonsai2/speculative-compare')


def rows(name):
    return [json.loads(line) for line in (DATA/(name+'.jsonl')).read_text().splitlines() if line.startswith('{')]


def aggregate(records):
    result={}
    reference={(r['case'],r['round']):r for r in records if r['method']=='serial'}
    for method in sorted({r['method'] for r in records}):
        selected=[r for r in records if r['method']==method]
        total=sum(len(r['tokens']) for r in selected)
        comparisons=[(r,reference.get((r['case'],r['round']))) for r in selected]
        result[method]={'cases':len(selected),'tokens':total,'decode_tps':total/sum(r['decode_s'] for r in selected),'request_tps':total/sum(r['request_s'] for r in selected),
                        'steps':sum(r['steps'] for r in selected),'draft_s':sum(r['draft_s'] for r in selected),'verify_s':sum(r['verify_s'] for r in selected),'copy_steps':sum(r['copy_steps'] for r in selected),'recycle_steps':sum(r.get('recycle_steps',0) for r in selected),
                        'serial_comparisons':sum(b is not None for a,b in comparisons),'matched_paths':sum(b is not None and a['tokens']==b['tokens'] for a,b in comparisons),'matched_next':sum(b is not None and a['next']==b['next'] for a,b in comparisons)}
    return result


def main():
    names=['ordered-00','ordered-04','ordered-08']
    records=sum((rows(n) for n in names),[])
    assert len({r['case'] for r in records})==10
    result={'ordered':aggregate(records),'cases':records,'receipts':{n:hashlib.sha256((DATA/(n+'.jsonl')).read_bytes()).hexdigest() for n in names},
            'serial_repeat_distinct_paths':{name:len({tuple(r['tokens']) for r in rows(name)}) for name in ['reset-identity','narrow-identity','ordered-identity']}}
    repeat=DATA/'ordered-repeat.jsonl'
    if repeat.exists():
        repeated=rows('ordered-repeat')
        result['opposite_order_repeat']=aggregate(repeated)
        refs={r['case']:r for r in records if r['method']=='serial'}
        result['repeat_matches_primary_serial']={m:sum(r['tokens']==refs[r['case']]['tokens'] and r['next']==refs[r['case']]['next'] for r in repeated if r['method']==m) for m in {r['method'] for r in repeated}}
    (DATA/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))


if __name__=='__main__':main()

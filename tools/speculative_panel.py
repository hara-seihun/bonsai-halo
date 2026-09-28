#!/usr/bin/env python3
"""Run one bounded native proposal panel with source and binary custody."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=Path('../../data/bonsai2/speculative-compare')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',required=True)
    parser.add_argument('--seconds',type=int,default=85)
    parser.add_argument('arguments',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if not args.tag.replace('-','').replace('_','').isalnum() or not 1<=args.seconds<=300:
        parser.error('use a simple tag and a bounded duration of at most 300 seconds')
    tail=args.arguments[1:] if args.arguments[:1]==['--'] else args.arguments
    DATA.mkdir(exist_ok=True)
    receipt=DATA/(args.tag+'.receipt.json')
    paths=[DATA/(args.tag+'.jsonl'),DATA/(args.tag+'.log'),receipt]
    if any(p.exists() for p in paths):
        parser.error('tag already has evidence; choose another tag')
    binary=ROOT/'tools/speculative_compare'
    command=[str(ROOT/'tools/run-batch-compare'),'--runtime-max',str(args.seconds)+'s','--memory-gib','18','--host-reserve-gib','4','--exec',str(binary),'--prompts',str(ROOT/'bench/drafter-prompts.txt'),*tail]
    sources=['tools/speculative_compare.cpp','tools/speculative_recycle.hip','tools/speculative_recycle.h','src/engine.cpp','kernels/halo_rows.hip','kernels/phases.hpp','bench/drafter-prompts.txt']
    snapshots=DATA/'sources';snapshots.mkdir(exist_ok=True)
    hashes={}
    for name in sources:
        path=ROOT/name;digest=sha(path);hashes[name]=digest
        (snapshots/(digest+path.suffix)).write_bytes(path.read_bytes())
    record={'command':command,'binary_sha256':sha(binary),'source_sha256':hashes,'git_head':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),'state':'running'}
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    with paths[0].open('w') as out,paths[1].open('w') as err:
        process=subprocess.run(command,stdout=out,stderr=err,check=False)
    rows=[json.loads(line) for line in paths[0].read_text().splitlines() if line.startswith('{')]
    record.update(state='complete' if process.returncode==0 else 'failed',exit_code=process.returncode,rows=len(rows),stdout_sha256=sha(paths[0]),stderr_sha256=sha(paths[1]))
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
    raise SystemExit(process.returncode)


if __name__=='__main__':
    main()

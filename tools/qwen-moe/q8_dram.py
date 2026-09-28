#!/usr/bin/env python3
"""Join the saved gfx1151 GL2C counter panels; never infer DRAM bytes from request counts."""
import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def calls(path):
    by_id = defaultdict(dict)
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            call = by_id[int(row['Dispatch_Id'])]
            call['grid'] = int(row['Grid_Size'])
            call['kernel'] = row['Kernel_Name']
            call['duration_ns'] = int(row['End_Timestamp']) - int(row['Start_Timestamp'])
            call[row['Counter_Name']] = int(float(row['Counter_Value']))
    return [by_id[k] for k in sorted(by_id)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--qwen', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--artifact', type=Path, action='append', default=[])
    args = parser.parse_args()
    files = {'calibration': args.calibration, 'replay': args.replay, 'qwen': args.qwen}
    data = {name: calls(path) for name, path in files.items()}
    sizes = [16, 64, 128, 256]
    repeated = [16, 16, 16, 256, 256, 16]
    assert len(data['calibration']) == 4 and len(data['replay']) == 6
    assert len(data['qwen']) == 582
    output = {
        'files_sha256': {str(p): digest(p) for p in [*files.values(), *args.artifact]},
        'source_sha256': {str(p): digest(p) for p in [Path(__file__), Path(__file__).parents[2] / 'bench/qwen_dram.hip']},
        'calibration': [], 'replay': [], 'qwen_by_kernel_grid': [],
    }
    for name, widths in [('calibration', sizes), ('replay', repeated)]:
        for mib, call in zip(widths, data[name]):
            ea = call['GL2C_EA_RDREQ_128B_sum']
            mc = call['GL2C_MC_RDREQ_sum']
            output[name].append({'read_mib': mib, 'ea_requests': ea, 'mc_requests': mc,
                                 'mc_over_ea': mc / ea, 'mc_times_128_over_read_bytes': mc * 128 / (mib << 20),
                                 'l2_hit_counter': call['GL2C_HIT_sum'],
                                 'l2_miss_counter': call['GL2C_MISS_sum'],
                                 'duration_ns': call['duration_ns']})
    grouped = defaultdict(list)
    for call in data['qwen']:
        grouped[(call['kernel'], call['grid'])].append(call)
    for (kernel, grid), group in sorted(grouped.items()):
        ea = [x['GL2C_EA_RDREQ_128B_sum'] for x in group]
        mc = [x['GL2C_MC_RDREQ_sum'] for x in group]
        output['qwen_by_kernel_grid'].append({
            'kernel': kernel, 'grid': grid, 'count': len(group),
            'ea_median': statistics.median(ea), 'mc_median': statistics.median(mc),
            'mc_over_ea': sum(mc) / sum(ea),
            'all_positive': all(a > 0 and b > 0 for a, b in zip(ea, mc)),
        })
    assert all(x['all_positive'] for x in output['qwen_by_kernel_grid'])
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()

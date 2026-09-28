#!/usr/bin/env python3
"""Join selected Qwen's GL2C/EA read-request counters to its active GGUF image."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

SIZES = (32, 64, 96, 128)
Q8_GRIDS = {16384: 512, 65536: 2048, 131072: 4096, 262144: 8192}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_dispatches(path):
    dispatches = {}
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            ident = row['Dispatch_Id']
            if ident not in dispatches:
                dispatches[ident] = {'name': row['Kernel_Name'], 'grid': int(row['Grid_Size']),
                    'start': int(row['Start_Timestamp']), 'end': int(row['End_Timestamp']), 'counters': {}}
            entry = dispatches[ident]
            if any(entry[k] != v for k, v in [('name', row['Kernel_Name']), ('grid', int(row['Grid_Size'])),
                   ('start', int(row['Start_Timestamp'])), ('end', int(row['End_Timestamp']))]):
                raise ValueError(f'inconsistent dispatch {ident}')
            counter = row['Counter_Name']
            if counter in entry['counters']:
                raise ValueError(f'duplicate counter {counter} in dispatch {ident}')
            entry['counters'][counter] = float(row['Counter_Value'])
    for entry in dispatches.values():
        names = {f'GL2C_EA_RDREQ_{size}B_sum' for size in SIZES}
        if set(entry['counters']) != names:
            raise ValueError(f'incomplete counter set {set(entry["counters"])}')
        entry['bytes'] = int(sum(size * entry['counters'][f'GL2C_EA_RDREQ_{size}B_sum'] for size in SIZES))
        entry['duration_ns'] = entry['end'] - entry['start']
    return dispatches


def counter_runs(dispatches):
    runs = []
    for ident, row in sorted(dispatches.items(), key=lambda item: item[1]['start']):
        present = row['bytes'] != 0
        if not runs or runs[-1]['nonzero'] != present:
            runs.append({'nonzero': present, 'first_dispatch': int(ident),
                'last_dispatch': int(ident), 'count': 0})
        runs[-1]['last_dispatch'] = int(ident)
        runs[-1]['count'] += 1
    return runs


def analyze(path, inventory_path, tokens):
    if tokens < 1:
        raise ValueError('tokens must be positive')
    inventory = json.loads(inventory_path.read_text())
    dispatches = read_dispatches(path)
    groups = defaultdict(list)
    for row in dispatches.values():
        found = re.search(r'mul_mat_vec_q<\(ggml_type\)(\d+),', row['name'])
        if not found:
            continue
        family = {'8': 'Q8_0', '12': 'Q4_K', '13': 'Q5_K', '14': 'Q6_K'}.get(found.group(1))
        if family:
            width = Q8_GRIDS.get(row['grid']) if family == 'Q8_0' else row['grid']
            groups[(family, width)].append(row)
    expected = defaultdict(int)
    for tensor in inventory['tensors']:
        family, name = tensor['type'], tensor['name']
        if family == 'Q8_0' and name != 'token_embd.weight':
            expected[(family, tensor['shape'][-1])] += tensor['bytes']
        elif family == 'Q6_K' and name == 'output.weight':
            expected[(family, 7946240)] += tensor['bytes']
    # The three expert quant families have banks of 256. One token selects eight.
    for family in ('Q4_K', 'Q5_K', 'Q6_K'):
        expert_bytes = sum(t['bytes'] for t in inventory['tensors'] if t['type'] == family
                           and '_exps.weight' in t['name'])
        if expert_bytes:
            expected[(family, 131072 if family == 'Q4_K' else 524288)] = expert_bytes * 8 // 256
    output = []
    for key, rows in sorted(groups.items()):
        observed = sum(row['bytes'] for row in rows) / tokens
        image = expected.get(key)
        positive = [row for row in rows if row['bytes']]
        mean_positive = sum(row['bytes'] for row in positive) / len(positive) if positive else None
        image_per_call = image / (len(rows) / tokens) if image else None
        output.append({'family': key[0], 'width_or_grid': key[1], 'calls_per_token': len(rows) / tokens,
            'nonzero_calls': len(positive), 'calls': len(rows),
            'mean_nonzero_counter_bytes_per_call': mean_positive,
            'mean_nonzero_to_image_per_call': mean_positive / image_per_call if image_per_call else None,
            'counter_bytes_per_token': observed, 'counter_device_ms_per_token': sum(row['duration_ns'] for row in rows) / (tokens * 1e6),
            'image_bytes_per_token': image, 'counter_to_image_ratio': observed / image if image else None})
    missing = sum(group['calls'] - group['nonzero_calls'] for group in output)
    return {'source_sha256': sha256(Path(__file__)), 'counter_csv_sha256': sha256(path),
            'inventory_sha256': sha256(inventory_path), 'model_header_sha256': inventory['header_sha256'],
            'tokens': tokens, 'dispatches': len(dispatches), 'counter_runs': counter_runs(dispatches),
            'quant_groups': output, 'zero_counter_quantized_calls': missing,
            'warning': (
                f'{missing} executed quantized calls have zero read-request counters. '
                'Do not sum this panel into model bytes; profile quantized kernels alone and inspect every dispatch.'
                if missing else
                'All selected quantized calls have nonzero read-request counters. The calibrated GL2C '
                'metric counts about half of a known once-read stream; it is not independently measured DRAM bytes.'
            )}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('counter_csv', type=Path)
    parser.add_argument('inventory', type=Path)
    parser.add_argument('--tokens', type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.counter_csv, args.inventory, args.tokens), indent=2))

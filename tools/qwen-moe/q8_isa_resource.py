#!/usr/bin/env python3
"""Extract installed HIP Q8 MMVQ code objects and report kernel ISA/resources."""
import argparse
import csv
import hashlib
import json
import re
import struct
import subprocess
import tempfile
from pathlib import Path


def run(*cmd, input=None):
    return subprocess.run(cmd, input=input, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout


def objects(library):
    sections = run('llvm-readobj', '--sections', str(library)).decode()
    section = re.search(r'Name: \.hip_fatbin .*?Offset: (0x[0-9a-fA-F]+|\d+).*?Size: (0x[0-9a-fA-F]+|\d+)', sections, re.S)
    if not section:
        raise ValueError('no installed HIP fatbin')
    with library.open('rb') as f:
        f.seek(int(section[1], 0))
        data = f.read(int(section[2], 0))
    off = 0
    while off + 32 <= len(data) and data[off:off + 4] == b'CCOB':
        compressed, size = struct.unpack_from('<QQ', data, off + 8)
        decoded = subprocess.run(('zstd', '-dc'), input=data[off + 32:off + 32 + compressed], capture_output=True)
        blob = decoded.stdout
        # CCOB's compressed length includes trailing alignment bytes. The zstd
        # CLI reports those as an unsupported next frame after emitting the payload.
        if len(blob) != size:
            raise ValueError((off, size, len(blob)))
        # Each compressed bundle carries a host marker and one gfx1151 ELF.
        pos = blob.find(b'\x7fELF')
        if pos >= 0:
            yield off, blob[pos:]
        off = (off + 32 + compressed + 4095) & ~4095


def inspect(library):
    matches = []
    with tempfile.TemporaryDirectory() as d:
        for offset, elf in objects(library):
            path = Path(d) / 'kernel.co'
            path.write_bytes(elf)
            if b'mul_mat_vec_qIL9ggml_type8E' not in elf:
                continue
            symbols = run('llvm-objdump', '-t', str(path)).decode(errors='replace')
            names = re.findall(r'^([0-9a-f]+)\s+\w+\s+F\s+\.text\s+[0-9a-f]+\s+(?:\.protected\s+)?(\S*mul_mat_vec_q\S*)$', symbols, re.M)
            q8 = [(address, name) for address, name in names if re.search(r'(?:IL9ggml_type8E|IL9ggml_type8EL)', name)]
            if not q8:
                continue
            metadata = run('llvm-readobj', '--notes', str(path)).decode(errors='replace')
            disassembly = run('llvm-objdump', '-d', str(path)).decode(errors='replace')
            for address, name in q8:
                # The symbol is demangled separately for readability, but the raw name is the stable key.
                demangled = run('c++filt', name).decode().strip()
                block = re.search(r'^0*' + address.lstrip('0') + r'\s+<' + re.escape(name) + r'>:\n(.*?)(?=^[0-9a-f]+\s+<|\Z)', disassembly, re.M | re.S)
                isa = block[1] if block else ''
                operations = re.findall(r'\b(?:global_load_\w+|buffer_load_\w+|v_dot\w+|v_wmma\w+|v_fma\w+|v_mul\w+|v_add\w+|s_waitcnt\w*|s_barrier)\b', isa)
                from collections import Counter
                notes = re.search(r'    \.name:\s+' + re.escape(name) + r'\n(.*?)(?=  - \.args:|\Z)', metadata, re.S)
                if not notes or not isa:
                    raise ValueError(f'missing compiled metadata or ISA for {name}')
                resources = {key: int(value) for key, value in re.findall(r'\.(vgpr_count|sgpr_count|vgpr_spill_count|sgpr_spill_count|private_segment_fixed_size|group_segment_fixed_size|wavefront_size|max_flat_workgroup_size):\s+(\d+)', notes[1])}
                loop = ''
                if '<(ggml_type)8, 1, false, false, false, false>' in demangled:
                    lines = isa.splitlines()
                    first = next(i for i, line in enumerate(lines) if 'global_load_b64' in line)
                    # A changed loop may branch on scc rather than exec. In that
                    # case retain the whole post-first-load body for inspection.
                    last = next((i for i in range(first, len(lines)) if 's_cbranch_execnz' in lines[i]), len(lines) - 1)
                    loop = '\n'.join(lines[first:last + 1])
                matches.append({'bundle_offset': offset, 'object_sha256': hashlib.sha256(elf).hexdigest(), 'symbol': name,
                                'demangled': demangled, 'isa_bytes': len(isa), 'operations': dict(Counter(operations)),
                                'resources': resources, 'selected_loop': loop})
    return {'library_sha256': hashlib.file_digest(library.open('rb'), 'sha256').hexdigest(), 'kernels': matches}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('library', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--trace', type=Path)
    args = p.parse_args()
    result = inspect(args.library.resolve())
    if args.trace:
        with args.trace.open(newline='') as f:
            rows = list(csv.DictReader(f))
        selected = [r for r in rows if 'mul_mat_vec_q<(ggml_type)8, 1, false, false, false, false>' in r['Kernel_Name'] and r['Grid_Size_X'] == '262144']
        result['trace'] = {'sha256': hashlib.file_digest(args.trace.open('rb'), 'sha256').hexdigest(),
                           'selected_dispatches': len(selected),
                           'observed_resources': sorted({(r['VGPR_Count'], r['SGPR_Count'], r['Scratch_Size'], r['LDS_Block_Size'], r['Workgroup_Size_X'], r['Grid_Size_X']) for r in selected})}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f"{len(result['kernels'])} installed Q8 MMVQ variants -> {args.output}")

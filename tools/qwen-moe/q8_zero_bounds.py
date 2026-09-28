#!/usr/bin/env python3
"""Census exact-zero installed Q8_0 weight blocks; no GPU or model evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

DEFAULT_MODEL = Path('../../data/qwen-moe/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf')
DEFAULT_INVENTORY = Path('../../data/qwen-moe/traffic.json')
DEFAULT_OUTPUT = Path('../../data/qwen-moe/q8-zero-bounds/receipt.json')
BLOCK = np.dtype([('scale', '<u2'), ('codes', 'u1', (32,))])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    p.add_argument('--inventory', type=Path, default=DEFAULT_INVENTORY)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    a = p.parse_args()
    inventory = json.loads(a.inventory.read_text())
    assert inventory['metadata']['general.architecture'] == 'qwen35moe'
    assert inventory['header_sha256'] == hashlib.sha256(a.model.open('rb').read(inventory['header_bytes'])).hexdigest()
    align = inventory['metadata'].get('general.alignment', 32)
    base = (inventory['header_bytes'] + align - 1) // align * align
    items = []
    total_blocks = total_zero = total_rows = total_zero_rows = 0
    for t in inventory['tensors']:
        if t['type'] != 'Q8_0' or t['name'] == 'token_embd.weight' or '_exps.' in t['name']:
            continue
        k = t['shape'][0]
        assert k % 32 == 0 and t['bytes'] % BLOCK.itemsize == 0
        blocks_per_row = k // 32
        assert t['bytes'] // BLOCK.itemsize % blocks_per_row == 0
        nrows = t['bytes'] // BLOCK.itemsize // blocks_per_row
        zblocks = zrows = 0
        hasher = hashlib.sha256()
        with a.model.open('rb') as f:
            f.seek(base + t['offset'])
            for row_start in range(0, nrows, 256):
                take = min(256, nrows - row_start)
                payload = f.read(take * blocks_per_row * BLOCK.itemsize)
                assert len(payload) == take * blocks_per_row * BLOCK.itemsize
                hasher.update(payload)
                b = np.frombuffer(payload, dtype=BLOCK).reshape(take, blocks_per_row)
                # The GGML Q8_0 map is d * int8(q). Either d==0 or all code bytes
                # zero is sufficient; exclude nonfinite scales from this shortcut.
                scale = b['scale']
                finite = (scale & 0x7c00) != 0x7c00
                zero = finite & ((scale & 0x7fff == 0) | np.all(b['codes'] == 0, axis=-1))
                zblocks += int(zero.sum())
                zrows += int(np.all(zero, axis=1).sum())
        blocks = nrows * blocks_per_row
        items.append(dict(name=t['name'], shape=t['shape'], type=t['type'], offset=t['offset'], bytes=t['bytes'],
                          sha256=hasher.hexdigest(), blocks=blocks, zero_blocks=zblocks,
                          rows=nrows, zero_rows=zrows))
        total_blocks += blocks
        total_zero += zblocks
        total_rows += nrows
        total_zero_rows += zrows
    q8_bytes = total_blocks * BLOCK.itemsize
    one_read = inventory['one_token_weight_stream_bytes']
    zero_row_bytes = sum(t['zero_rows'] * t['shape'][0] // 32 * BLOCK.itemsize for t in items)
    row_bitmap_bytes = (total_rows + 7) // 8
    summary = dict(q8_nonembedding_bytes=q8_bytes, blocks=total_blocks, zero_blocks=total_zero,
                   zero_fraction=total_zero / total_blocks, rows=total_rows, zero_rows=total_zero_rows,
                   free_zero_block_skip_bytes=total_zero * BLOCK.itemsize,
                   free_zero_block_skip_fraction_of_complete_one_read=total_zero * BLOCK.itemsize / one_read,
                   bitmap_bytes=(total_blocks + 7) // 8,
                   bitmap_paid_net_byte_upper_bound=total_zero * BLOCK.itemsize - (total_blocks + 7) // 8,
                   zero_row_bytes=zero_row_bytes, row_bitmap_bytes=row_bitmap_bytes,
                   row_bitmap_paid_net_byte_upper_bound=zero_row_bytes - row_bitmap_bytes,
                   row_bitmap_net_fraction_of_complete_one_read=(zero_row_bytes - row_bitmap_bytes) / one_read)
    result = dict(source_sha256=digest(Path(__file__)), inventory_sha256=digest(a.inventory),
                  header_sha256=inventory['header_sha256'], model_sha256=inventory.get('model_sha256',
                  'ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61'),
                  model_size=a.model.stat().st_size, block_bytes=BLOCK.itemsize,
                  observation='Q8_0 scale times 32 signed int8 weights; exact real decoded zero, finite scale',
                  assumptions=['One uncached read per Q8_0 weight per token; bitmap paid in model image, one read per block.',
                               'Free sparse index/dispatch and no performance cost for conditional skipping; not a native timing bound.',
                               'FP32 bit identity requires signed-zero and reduction-order acceptance; this counts decoded real-zero blocks only.'],
                  summary=summary, tensors=items)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()

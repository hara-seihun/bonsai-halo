#!/usr/bin/env python3
"""Selected-route exact packed down-superblock equality, CPU only.

Fingerprint is only a candidate filter: every proposed equal block is compared
bytewise, including scale/minimum metadata and code bytes.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path('../../data/qwen-moe')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def layer(base, traffic, image, offset, number):
    tensor = next(t for t in traffic['tensors'] if t['name'] == f'blk.{number}.ffn_down_exps.weight')
    assert tensor['shape'] == [512, 2048, 256] and tensor['type'] in ('Q5_K', 'Q6_K')
    block_bytes = {'Q5_K': 176, 'Q6_K': 210}[tensor['type']]
    blocks = 2048 * 512 // 256
    bank = image[offset + tensor['offset']:offset + tensor['offset'] + tensor['bytes']].reshape(256, blocks, block_bytes)
    assert tensor['bytes'] == 256 * blocks * block_bytes
    # Sum over word positions with two distinct odd multipliers, plus length.
    # This is not an equality oracle: proposed equal fingerprints must match
    # the complete packed payload below. The final partial word is zero-padded.
    padded = np.zeros((256, blocks, (block_bytes + 7) // 8 * 8), dtype='u1')
    padded[:, :, :block_bytes] = bank
    words = padded.view('<u8').reshape(256, blocks, -1)
    n = words.shape[-1]
    coeff_a = np.random.default_rng(0x51ab).integers(0, 2**64, n, dtype='u8') | np.uint64(1)
    coeff_b = np.random.default_rng(0xb105).integers(0, 2**64, n, dtype='u8') | np.uint64(1)
    fingerprints = ((words * coeff_a).sum(axis=-1), (words * coeff_b).sum(axis=-1))
    order = np.lexsort((fingerprints[1], fingerprints[0]), axis=0)
    sorted_a = np.take_along_axis(fingerprints[0], order, axis=0)
    sorted_b = np.take_along_axis(fingerprints[1], order, axis=0)
    equal = (sorted_a[1:] == sorted_a[:-1]) & (sorted_b[1:] == sorted_b[:-1])
    ranks, positions = np.nonzero(equal)
    for rank, position in zip(ranks, positions):
        assert np.array_equal(bank[order[rank, position], position],
                              bank[order[rank + 1, position], position]), (number, rank, position)
    result = {'layer': number, 'tensor': tensor['name'], 'type': tensor['type'], 'block_bytes': block_bytes,
              'blocks_per_expert': blocks, 'tensor_payload_sha256': hashlib.sha256(bank).hexdigest(),
              'all_bank_same_position_redundant_blocks': len(ranks),
              'all_bank_same_position_redundant_bytes': len(ranks) * block_bytes, 'splits': {}}
    for split in ('train', 'held'):
        route = base / 'all-producers' / f'{split}.layer-{number}.ffn_moe_topk.i32'
        ids = np.fromfile(route, dtype='<i4').reshape(64, 8)
        assert ids.min() >= 0 and ids.max() < 256 and np.all(np.diff(np.sort(ids, axis=1), axis=1))
        matched_pairs = 0
        redundant_blocks = 0
        matched_coordinates = 0
        # Process one token at a time so a candidate's complete payload is
        # available without replicating the 180-MB tensor 64 times.
        for route_ids in ids:
            a, b = (fingerprints[k][route_ids] for k in (0, 1))
            equal = (a[:, None, :] == a[None, :, :]) & (b[:, None, :] == b[None, :, :])
            equal &= np.triu(np.ones((8, 8), dtype=bool), 1)[:, :, None]
            first, second, coord = np.nonzero(equal)
            for i, j, p in zip(first, second, coord):
                assert np.array_equal(bank[route_ids[i], p], bank[route_ids[j], p]), (number, split, i, j, p)
            matched_pairs += len(coord)
            matched_coordinates += len(np.unique(coord))
            # The graph on eight equal-payload blocks consists of disjoint
            # cliques. Its duplicate count is 8 minus connected components.
            if len(coord):
                for position in np.unique(coord):
                    edges = [(int(i), int(j)) for i, j, p in zip(first, second, coord) if p == position]
                    parents = list(range(8))
                    def root(x):
                        while parents[x] != x:
                            x = parents[x]
                        return x
                    for i, j in edges:
                        parents[root(i)] = root(j)
                    redundant_blocks += 8 - len({root(i) for i in range(8)})
        result['splits'][split] = {'route_sha256': sha(route), 'tokens': len(ids),
                                   'pair_equal_payloads': matched_pairs, 'positions_with_equal_pair': matched_coordinates,
                                   'redundant_blocks': redundant_blocks,
                                   'free_repeated_payload_bytes_per_token': redundant_blocks * block_bytes / len(ids)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--output', type=Path, default=BASE / 'down-packed-reuse')
    parser.add_argument('--first', type=int, default=0)
    parser.add_argument('--last', type=int, default=40)
    parser.add_argument('--aggregate', action='store_true')
    args = parser.parse_args()
    assert 0 <= args.first < args.last <= 40
    args.output.mkdir(parents=True, exist_ok=True)
    inventory = args.base / 'traffic.json'
    traffic = json.loads(inventory.read_text())
    image_path = args.base / 'Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
    image = np.memmap(image_path, mode='r', dtype='u1')
    offset = (traffic['header_bytes'] + 31) // 32 * 32
    if not args.aggregate:
        for i in range(args.first, args.last):
            result = layer(args.base, traffic, image, offset, i)
            result['identities'] = {'source_sha256': sha(__file__), 'inventory_sha256': sha(inventory),
                                    'model_sha256': json.loads((args.base / 'acquisition.json').read_text())['sha256']}
            (args.output / f'layer-{i}.json').write_text(json.dumps(result, indent=2) + '\n')
            print(i, result['all_bank_same_position_redundant_blocks'], result['splits']['held'], flush=True)
    if args.aggregate or (args.first == 0 and args.last == 40):
        shards = [json.loads((args.output / f'layer-{i}.json').read_text()) for i in range(40)]
        assert all(shard['layer'] == i and shard['identities'] == shards[0]['identities']
                   for i, shard in enumerate(shards))
        assert shards[0]['identities']['source_sha256'] == sha(__file__)
        receipt = {'contract': 'same output-row and K256 position, full packed Q5_K/Q6_K bytes; entire 256-expert bank and actual eight-expert routes',
                   'identities': shards[0]['identities'], 'shard_sha256': [sha(args.output / f'layer-{i}.json') for i in range(40)],
                   'all_bank_same_position_redundant_blocks': sum(r['all_bank_same_position_redundant_blocks'] for r in shards),
                   'all_bank_same_position_redundant_bytes': sum(r['all_bank_same_position_redundant_bytes'] for r in shards),
                   'splits': {s: {k: sum(r['splits'][s][k] for r in shards) for k in ('pair_equal_payloads', 'positions_with_equal_pair', 'redundant_blocks', 'free_repeated_payload_bytes_per_token')} for s in ('train', 'held')}}
        (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps({'all_bank_same_position_redundant_blocks': receipt['all_bank_same_position_redundant_blocks'], 'splits': receipt['splits']}, indent=2))


if __name__ == '__main__':
    main()

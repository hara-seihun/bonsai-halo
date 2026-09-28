#!/usr/bin/env python3
"""Price lossless expert-image sharing between two actual generated Qwen streams.

This is a CPU analysis of captured routes, not a native throughput measurement.
"""
import argparse
import json
from pathlib import Path

from generated_route_reuse import EXPERTS, LAYERS, ROUTED, capture, sha


def analyze(train, held, inventory):
    sizes = [inventory['layers'][str(layer)]['bank_bytes'] // EXPERTS for layer in range(LAYERS)]
    assert all(size * EXPERTS == inventory['layers'][str(layer)]['bank_bytes'] for layer, size in enumerate(sizes))
    steps = len(train[0])
    assert len(held[0]) == steps and all(len(rows) == steps for rows in train + held)
    per_layer = [0] * LAYERS
    aligned = []
    cross_product = []
    for t in range(steps):
        for u in range(steps):
            counts = [len(set(train[layer][t]) & set(held[layer][u])) for layer in range(LAYERS)]
            shared_bytes = sum(count * size for count, size in zip(counts, sizes))
            cross_product.append(shared_bytes)
            if t == u:
                aligned.append(shared_bytes)
                per_layer = [a + b for a, b in zip(per_layer, counts)]
    routed_two = 2 * inventory['active_routed_bytes_per_token']
    nonexpert = inventory['nonexpert_nonembedding_bytes']
    embedding = inventory['embedding_lookup_bytes']
    one_read_grouped_two = nonexpert + 2 * embedding + routed_two
    one_read_separate_two = 2 * inventory['one_token_weight_stream_bytes']
    assert one_read_separate_two - one_read_grouped_two == nonexpert
    return {
        'steps_per_stream': steps,
        'aligned_pair_count': len(aligned),
        'cross_product_state_pairs': steps * steps,
        'expected_uniform_independent_overlap_per_layer': ROUTED * ROUTED / EXPERTS,
        'aligned_overlap_by_layer': per_layer,
        'aligned_overlap_assignments': sum(per_layer),
        'aligned_requested_assignments': steps * LAYERS * 2 * ROUTED,
        'aligned_interstream_shared_image_bytes': sum(aligned),
        'aligned_shared_bytes_per_step': aligned,
        'all_cross_product_shared_bytes': {
            'mean': sum(cross_product) / len(cross_product),
            'min': min(cross_product),
            'median_upper': sorted(cross_product)[len(cross_product) // 2],
            'max': max(cross_product),
        },
        'one_read_separate_two_bytes_per_step': one_read_separate_two,
        'one_read_nonexpert_shared_two_bytes_per_step': one_read_grouped_two,
        'nonexpert_sharing_bytes_per_step': nonexpert,
        'aligned_best_two_stream_union_bytes_per_step': one_read_grouped_two - sum(aligned) / steps,
        'aligned_incremental_expert_share_fraction_of_nonexpert_grouped': sum(aligned) / (steps * one_read_grouped_two),
        'aligned_incremental_expert_share_fraction_of_separate': sum(aligned) / (steps * one_read_separate_two),
        'aligned_total_nonexpert_plus_expert_share_fraction_of_separate': (steps * nonexpert + sum(aligned)) / (steps * one_read_separate_two),
        'cross_product_incremental_expert_share_fraction_of_nonexpert_grouped': sum(cross_product) / (len(cross_product) * one_read_grouped_two),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('inventory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    assert inventory['expert_count'] == EXPERTS and inventory['experts_per_token'] == ROUTED
    arrays = {}
    hashes = {}
    for split in ('train', 'held'):
        arrays[split], files, mismatches, steps = capture(args.capture, split)
        assert mismatches == 0 and steps == 48
        hashes.update(files)
    result = analyze(arrays['train'], arrays['held'], inventory)
    original = json.loads((args.capture / 'receipt.json').read_text())
    receipt = {
        'contract': 'Two independent 48-step greedy continuations from different 64-token prefixes, callback-observed top-k routes (fusion cut). Aligned t/t is a real pair of independently reachable states, not an actually measured concurrent decode. Every output is required. Nonexpert projection image read once per paired step; each of 16 selected full expert images read once per distinct (layer,expert) pair. Same-layer duplicates alone may share their packed weight read for free; no image compression, other traffic, native timing, or FP32 scheduling assumed. Cross product enumerates 2304 possible independent state pairs, not one causal 2304-step run. Native grouping may already realize some or all of this sharing.',
        'source_sha256': sha(Path(__file__)),
        'capture_source_sha256': original['observer_source_sha256'],
        'capture_receipt_sha256': sha(args.capture / 'receipt.json'),
        'capture_files_sha256': hashes,
        'inventory_sha256': sha(args.inventory),
        'model_sha256': original['model_sha256'],
        'result': result,
    }
    args.output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('aligned_shared_bytes_per_step', 'aligned_overlap_by_layer')}, indent=2))


if __name__ == '__main__':
    main()

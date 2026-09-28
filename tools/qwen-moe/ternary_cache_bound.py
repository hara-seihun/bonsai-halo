#!/usr/bin/env python3
"""Conditional cache frontier for a *hypothetical* uniform-rate routed MoE image.

This does not construct a quantizer or claim native execution/quality. Cache
semantics are one demand per distinct (token, layer, expert) image, serial tokens.
"""
import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

from generated_cache_bound import CAPACITY_BYTES, EXPERTS, LAYERS, ROUTED, accesses, load, simulate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ceil_fraction(x):
    return -(-x.numerator // x.denominator)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture', type=Path)
    p.add_argument('inventory', type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    inv = json.loads(args.inventory.read_text())
    assert inv['expert_count'] == EXPERTS and inv['experts_per_token'] == ROUTED
    # Three dense 2048x512 projections per expert. This assumes the *complete*
    # routed image, scales and metadata included, pays this rate; it is not a
    # measurement of an existing 35B ternary image.
    weights_per_image = 3 * 2048 * 512
    nonexpert = inv['nonexpert_nonembedding_bytes'] + inv['embedding_lookup_bytes']
    result = dict(contract='Conditional image size and ideal offline serial cache, not a saved Qwen ternary model, measured DRAM, quality, or GPU time. Distinct expert labels occupy disjoint physical images. Shared metadata and any decoder bytes must be paid inside the stated rate.',
                  analysis_sha256=sha(Path(__file__)), model_sha256=json.loads((args.capture.parent / 'acquisition.json').read_text())['sha256'],
                  inventory_sha256=sha(args.inventory), capture_observer_sha256=sha(Path(__file__).with_name('capture_generated_routes.cpp')),
                  capacity_bytes=CAPACITY_BYTES, weights_per_image=weights_per_image,
                  baseline_one_read_bytes_per_token=inv['one_token_weight_stream_bytes'], rates={})
    assert nonexpert + inv['active_routed_bytes_per_token'] == inv['one_token_weight_stream_bytes']
    for rate_text in ('1.585', '1.65', '1.727'):
        # 1.585 is near the worst-case ternary information floor log2(3),
        # not an achieved paid complete-model image.
        rate = Fraction(rate_text)
        image = ceil_fraction(rate * weights_per_image / 8)
        slots = CAPACITY_BYTES // image
        assert slots < LAYERS * ROUTED
        expert_bytes = image * LAYERS * ROUTED
        complete = nonexpert + expert_bytes
        entry = dict(paid_bpw=str(rate), expert_image_bytes=image, cache_slots=slots,
                     eight_per_layer_footprint_bytes=expert_bytes,
                     complete_one_read_bytes_per_token=complete,
                     rate_only_fraction_saved_from_current=(inv['one_token_weight_stream_bytes'] - complete) / inv['one_token_weight_stream_bytes'],
                     cold_48_token_arbitrary_byte_bound=47 * CAPACITY_BYTES / (48 * complete),
                     steady_arbitrary_byte_bound=CAPACITY_BYTES / complete,
                     complete_forty_layer_residency_capacity_ratio=expert_bytes / CAPACITY_BYTES,
                     splits={})
        for split in ('train', 'held'):
            arrays, hashes = load(args.capture, split)
            requests = accesses(arrays)
            steps = len(arrays[0])
            oracle = simulate(requests, slots, 'clairvoyant')
            lru = simulate(requests, slots, 'lru')
            assert lru['hits'] <= oracle['hits'] <= slots * (steps - 1)
            # Offline next-use with bypass is optimal for equal-sized whole
            # images: this is a realizable traffic schedule with *free* oracle.
            entry['splits'][split] = dict(steps=steps, route_sha256=hashes,
                oracle_hits=oracle['hits'], oracle_hits_per_token=oracle['hits_per_token'],
                lru_hits=lru['hits'], oracle_saved_bytes=oracle['hits'] * image,
                oracle_fraction_of_complete_one_read=oracle['hits'] * image / (steps * complete),
                unrestricted_byte_cache_bound_bytes=(steps - 1) * CAPACITY_BYTES,
                oracle_gap_to_byte_bound_bytes=(steps - 1) * CAPACITY_BYTES - oracle['hits'] * image)
        result['rates'][rate_text] = entry
        print(rate_text, 'image', image, 'slots', slots, 'complete', complete,
              'held oracle', entry['splits']['held']['oracle_hits'], 'held LRU', entry['splits']['held']['lru_hits'],
              'fraction', entry['splits']['held']['oracle_fraction_of_complete_one_read'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()

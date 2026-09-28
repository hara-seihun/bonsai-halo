#!/usr/bin/env bash
# The two paths this change is NOT aimed at, on each arm's executable, in one lock hold.
#
# The sliced state phase belongs to mode 0. Prompt ingestion (mode 20, 128-row passes) and batched
# generation (mode 19) run the resident kernels instead, and the only thing they share with this
# change is `k_forward_rows`'s one register budget and the block cache's gate form on a replay
# prefix they do not have. This is the check that says so with a measurement.
set -euo pipefail
out=${HALO_PHASE_OUT:-../../data/bonsai2/batch-comparison/gdn-sliced-gates}
mkdir -p "$out"
: "${HALO_REG_STREAMS:=8}"
for arm in "$@"; do
    name=$(basename "$arm")
    printf '\n=== REGRESS %s (%s) ===\n' "$name" "$(sha1sum "$arm/batch_profile" | cut -c1-12)"
    "$arm/batch_profile" --modes 20 --rows 128 --rounds 2 \
        --decode-streams "$HALO_REG_STREAMS" --decode-tokens 1 --decode-prompt 128 --decode-context 512 \
        --out "$out/regress-$name.json"
done

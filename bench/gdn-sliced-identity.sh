#!/usr/bin/env bash
# The device's own sliced-against-resident diagnostic, run on each arm's executable.
#
# `HALO_GDN_COMPARE=1` makes a wide pass rebuild its first layer through the SLICED route into
# private buffers and compare token records, recurrent output, state and conv ring against what the
# resident kernel produced from the same inputs, inside one process. That is the identity witness
# this change needs and a cross-process residual hash cannot be: the deployed route's
# `mv_down`/`mv_ssm_out`/`mv_o` drains are `atomicAdd` over KS = 2 parts onto a residual, so two
# processes of ONE executable do not agree on the last bits (docs/gdn-sliced-gates.md).
#
#   tools/run-batch-compare --pin-clock --runtime-max 44s --exec bench/gdn-sliced-identity.sh \
#       /tmp/arm-base /tmp/arm-both
set -euo pipefail
out=${HALO_PHASE_OUT:-../../data/bonsai2/batch-comparison/gdn-sliced-gates}
mkdir -p "$out"
for arm in "$@"; do
    name=$(basename "$arm")
    printf '\n=== IDENTITY %s (%s) ===\n' "$name" "$(sha1sum "$arm/batch_profile" | cut -c1-12)"
    HALO_GDN_COMPARE=1 "$arm/batch_profile" --modes 20 --rows 128 --rounds 1 --gdn-defer 1 \
        --out "$out/identity-$name.json"
done

#!/usr/bin/env bash
# The same controlled step through each arm's own executable, in one lock hold and one pinned clock.
#
# `--modes 0 --decode-streams 1 --decode-tokens 8` is the drafted verify shape: the persistent
# eight-row kernel, one sequence, eight rows, logits on every row. Unlike `--bench --dflash` it is a
# FIXED workload - a drafted step's replay prefix is whatever the last step happened to accept, and
# that count moves the state phase's token walk by tens of percent between two runs of one binary -
# and every sample carries the residual FNV-64 that says whether the arms produce the same bits.
#
#   tools/run-batch-compare --pin-clock --runtime-max 44s --exec bench/gdn-sliced-phase.sh \
#       /tmp/arm-base /tmp/arm-both /tmp/arm-base
set -euo pipefail
: "${HALO_PHASE_ROUNDS:=2}"
: "${HALO_PHASE_PROMPT:=1024}"
: "${HALO_PHASE_CTX:=2048}"
out=${HALO_PHASE_OUT:-../../data/bonsai2/batch-comparison/gdn-sliced-gates}
mkdir -p "$out"
i=0
for arm in "$@"; do
    i=$((i + 1))
    name=$(basename "$arm")
    printf '\n=== ARM %s (%s) cell %d ===\n' "$name" "$(sha1sum "$arm/batch_profile" | cut -c1-12)" "$i"
    # The prefix is ingested on the wide exact route (mode 20) rather than eight rows at a time: the
    # step measured is still mode 0, and a 1024-token prefix drops from about 10 s to 2, which is
    # what lets four arms share one bounded call.
    "$arm/batch_profile" --modes 0 --decode-streams 1 --decode-tokens 8 --decode-setup-mode 20 --warmup-ms 1000 \
        --decode-prompt "$HALO_PHASE_PROMPT" --decode-context "$HALO_PHASE_CTX" --rounds "$HALO_PHASE_ROUNDS" \
        --out "$out/phase-$i-$name.json"
done

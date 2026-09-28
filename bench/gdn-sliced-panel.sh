#!/usr/bin/env bash
# The sliced state phase's arms, alternating inside one lock hold and one pinned clock.
#
# The arms are compile-time (HALO_GDN_BLK_GATE, HALO_GDN_PK_SPLIT) because a runtime switch inside
# the persistent kernel costs the whole kernel its register budget, so each arm is its own
# executable and this walks them in one `run-batch-compare` hold instead of one process. Give it
# the executables in the order you want them run; repeat one to carry a drift control.
#
#   tools/run-batch-compare --pin-clock --runtime-max 44s --exec bench/gdn-sliced-panel.sh \
#       /tmp/halo-base /tmp/halo-both /tmp/halo-gate /tmp/halo-base
#
# The workload is the published served-step cell (docs/ternary-coordinate-ceiling.md): the Q4
# DFlash2 drafter, a 23-token prompt, 48 greedy tokens, the deployed grid pinned at 100, and
# HALO_PROFILE=1 so every cell prints its own per-phase device timing. The greedy digest is the
# identity control: every arm here is meant to be bit-identical to every other.
set -euo pipefail
: "${HALO_PANEL_TOKENS:=48}"
: "${HALO_PANEL_GRID:=100}"
drafter=${HALO_PANEL_DRAFTER:-../../data/bonsai2/drafters/dflash2.safetensors}
args=(--bench --context 4096 --dflash "$drafter" --draft-weights q4 ${HALO_PANEL_PROMPTS:+--prompts "$HALO_PANEL_PROMPTS"} ${HALO_PANEL_SWEEP:+--sweep-rounds "$HALO_PANEL_SWEEP"}
      --grid-rows "$HALO_PANEL_GRID" -n "$HALO_PANEL_TOKENS")
for arm in "$@"; do
    printf '\n=== ARM %s (%s) ===\n' "${arm##*/}" "$(sha1sum "$arm" | cut -c1-12)"
    HALO_PROFILE=1 "$arm" "${args[@]}"
done

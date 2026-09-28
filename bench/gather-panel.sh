#!/bin/sh
# Two binaries from one checkout, alternating inside one GPU lock and one pinned clock, so the
# arms share a thermal state and the order cannot favour either. A is canonical (the deployed
# peel), B is the same tree with the perm gather in mv_expand/mv_rows_deployed and the forward
# kernel's shared block sized to its own phases.
P=../../data/bonsai2/drafters/dflash2.safetensors
CAN=./bonsai-halo
NEW=../work/clones/rows-lds-occupancy/bonsai-halo.gather
N=${N:-48}
cd . || exit 1
for pair in "A $CAN" "B $NEW" "B $NEW" "A $CAN"; do
    arm=${pair%% *}; bin=${pair#* }
    printf '@@ arm %s %s\n' "$arm" "$bin"
    "$bin" --bench --dflash "$P" -n "$N" 2>&1 | grep -E '^dflash2:'
done

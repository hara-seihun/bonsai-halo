#!/usr/bin/env bash
# Where a phase's weight bytes live, at the three shapes the verify pass runs.
#
# `bench/coop_cost` has always walked ONE hipMalloc in ascending windows, so phase p+1 starts where
# phase p stopped and a launch is one sweep. `Engine::upload_halo` gives every ternary tensor its
# own hipMalloc, and consecutive phases of the engine read unrelated allocations. That is the last
# structural difference between the probe and the engine at the shape where they disagree by 26%
# (docs/drafted-serve-step.md, open question 1). Three arms separate it:
#
#   seq       one allocation, ascending windows          (what the probe has always measured)
#   scatter   one allocation, a permutation of windows   (continuity broken, allocation unchanged)
#   regions   one allocation PER PHASE                   (what the engine does)
#
# Arms alternate inside one lock hold and one pinned clock, because two processes on this box differ
# by more than the effect. Run under tools/run-batch-compare.
set -euo pipefail
cd "$(dirname "$0")/.."
probe=bench/coop_cost
grid=${GRID:-120}
phases=${PHASES:-64}
rounds=${ROUNDS:-4}

cell() { # name pw units regions extra...
    local name=$1 pw=$2 units=$3 nreg=$4; shift 4
    printf '\n== %s ==\n' "$name"
    $probe --mode 2 --tt 8 --grid "$grid" --phases "$phases" --rounds "$rounds" \
        --pw "$pw" --phase-units "$units" --exact-units "$@"
}

for pass in 1 2; do
    printf '\n######## pass %d ########\n' "$pass"
    # gate/up: 1088 units of 40 blocks, the phase that is 37%% of the verify pass
    cell "gate_up seq"      5 1088 0 --bytes 2048
    cell "gate_up scatter"  5 1088 0 --bytes 2048 --scatter
    cell "gate_up regions"  5 1088 0 --bytes 64 --regions 32
    # o / ssm_out: 320 units of 24 blocks, the 113-115 GB/s phases
    cell "o_ssm seq"        3 320 0 --bytes 2048
    cell "o_ssm scatter"    3 320 0 --bytes 2048 --scatter
    cell "o_ssm regions"    3 320 0 --bytes 64 --regions 64
    # the vocabulary head: one launch, 7760 units, the phase already on the probe's curve
    cell "head seq"         5 7760 0 --bytes 2048
    cell "head regions"     5 7760 0 --bytes 320 --regions 6
done

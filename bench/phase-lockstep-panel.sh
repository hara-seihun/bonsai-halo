#!/usr/bin/env bash
# What a short phase is actually losing: 100 workgroups leave the grid barrier in lockstep, at the
# same offset inside their own units, and their weight streams collide until dealing lets them
# drift. Two ways to break it - a delay (costs time) and a block rotation (costs nothing) - against
# the deployed shapes, palindrome ordered so drift across the panel shows as a disagreement between
# the two cells of an arm rather than as a result.
#
#   tools/run-batch-compare --pin-clock --exec bash bench/phase-lockstep-panel.sh
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
probe=bench/coop_cost
rounds=${ROUNDS:-3}
want=" ${*:-1 2 3} "
run() { [[ "$want" == *" $1 "* ]]; }

if run 1; then
echo "=== 1. the stagger ladder, palindrome, 320 units on grid 100 (mv_down's shape)"
for s in 0 16 64 16 0; do
    $probe --mode 2 --tt 8 --pw 5 --grid 100 --phase-units 320 --exact-units --drain 3 \
           --arm 0 --stagger "$s" --rounds "$rounds" --phases 64 | tail -n +3 | sed "s/^/ stagger $s /"
done
fi

if run 2; then
echo "=== 2. the rotation arm against the control, every phase length, one process"
for u in 100 320 512 1088 7760; do
    $probe --mode 2 --tt 8 --pw 5 --grid 100 --phase-units "$u" --exact-units --drain 3 \
           --arm 0,4,0,4 --rounds "$rounds" --phases 64 | tail -n +3
done
fi

if run 3; then
echo "=== 3. the rotation on the other deployed shapes: K-split pair (pw 3), down (pw 17), one row"
$probe --mode 2 --tt 8 --pw 3 --grid 100 --phase-units 320 --exact-units --drain 4 --split 2 \
       --arm 0,4,0,4 --rounds "$rounds" --phases 64 | tail -n +3
$probe --mode 2 --tt 8 --pw 17 --grid 100 --phase-units 320 --exact-units --drain 4 \
       --arm 0,4,0,4 --rounds "$rounds" --phases 64 | tail -n +3
$probe --mode 2 --tt 1 --pw 5 --grid 100 --phase-units 320 --exact-units --drain 1 \
       --arm 0,4,0,4 --rounds "$rounds" --phases 64 | tail -n +3
fi

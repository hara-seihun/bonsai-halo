#!/usr/bin/env bash
# The phase-ramp panel: what a short phase loses, and whether a request that survives the barrier
# gets it back. Every cell runs in one process against one warmed clock, arms interleaved per shape.
#
#   tools/run-batch-compare --pin-clock --exec bash bench/phase-ramp-panel.sh
#
# Shapes are the deployed drafted verify pass's own (docs/drafted-serve-step.md): grid 100, eight
# rows, the spread drain, and the unit counts its phases actually deal - 320 for the three 5120-row
# projections, 512 for qkv_z, 1088 for gate_up, 7760 for the vocabulary head.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
probe=bench/coop_cost
arms=${ARMS:-0,1,2,8,10,11}
rounds=${ROUNDS:-3}
want=" ${*:-1 2 3 4 5 6} "
run() { [[ "$want" == *" $1 "* ]]; }

if run 1; then echo "=== 1. the phase-length curve, arm 0 against the arms, pw 5 (the 40-block projections)"
for u in 100 320 512 1088 7760; do
    $probe --mode 2 --tt 8 --pw 5 --grid 100 --phase-units "$u" --exact-units --drain 3 \
           --arm "$arms" --rounds "$rounds" --phases 64 | tail -n +3
done

fi
if run 2; then echo "=== 2. the K-split pair's shape: pw 3, 320 units, part-major"
$probe --mode 2 --tt 8 --pw 3 --grid 100 --phase-units 320 --exact-units --drain 4 --split 2 \
       --arm "$arms" --rounds "$rounds" --phases 64 | tail -n +3

fi
if run 3; then echo "=== 3. the down stage: pw 17, 320 units"
$probe --mode 2 --tt 8 --pw 17 --grid 100 --phase-units 320 --exact-units --drain 4 \
       --arm "$arms" --rounds "$rounds" --phases 64 | tail -n +3

fi
if run 4; then echo "=== 4. one row, the single-token step's width, same shapes"
for u in 320 1088; do
    $probe --mode 2 --tt 1 --pw 5 --grid 100 --phase-units "$u" --exact-units --drain 1 \
           --arm "$arms" --rounds "$rounds" --phases 64 | tail -n +3
done

fi
if run 5; then echo "=== 5. the burst hypothesis: a per-workgroup delay at the phase start, arm 0"
for s in 0 1 4 16; do
    $probe --mode 2 --tt 8 --pw 5 --grid 100 --phase-units 320 --exact-units --drain 3 \
           --arm 0 --stagger "$s" --rounds "$rounds" --phases 64 | tail -n +3 | sed "s/^/ stagger $s /"
done

fi
if run 6; then echo "=== 6. the inside of a phase: workgroup 0's unit timeline, 320 units on grid 100"
$probe --mode 2 --tt 8 --pw 5 --grid 100 --phase-units 320 --exact-units --drain 3 \
       --arm 0,10 --rounds 2 --phases 8 --trace-units 40 | tail -n +3

fi
#!/usr/bin/env bash
# Which variable the eight-row matvec phase is actually short of.
#
# Panel 1 found that at tt 8 a phase's byte rate climbs from 142 GB/s at one grid round to
# 204 at 78, with blocks per wave nearly flat. Three candidate variables produce that:
#   rounds      - a phase needs many deals before its workgroups drift out of lockstep
#   workgroups  - the grid is too wide for the units the phase has
#   unit size   - the per-unit cold start, which is what tt 1 is sensitive to
#
#   A  grid sweep at a fixed 320-unit phase: same work, 3.2 to 6.4 rounds.
#   B  equal bytes per phase, few large units against many small ones.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
out=${1:?usage: matvec-shape-panel2.sh OUTDIR}
mkdir -p "$out"
cd "$root"
log=$out/shape-cause.txt
: > "$log"
echo "revision $(git rev-parse --short HEAD)" | tee -a "$log"

echo "== A: grid sweep, 320 units, tt 8 ==" | tee -a "$log"
for g in 40 50 60 80 100; do
  echo "-- grid $g --" | tee -a "$log"
  bench/coop_cost --mode 2 --tt 8 --grid "$g" --pw 3,5 --phase-units 320 --exact-units \
      --phases 32 --rounds 4 2>&1 | tee -a "$log"
done

echo "== B: equal bytes per phase, unit size against unit count, tt 8, grid 100 ==" | tee -a "$log"
# 14.3 MB a phase: pw 5 x 400 units, pw 8 x 250, pw 20 x 100, pw 40 x 50
for spec in "5 400" "8 250" "20 100" "40 50" "3 667"; do
  set -- $spec
  echo "-- pw $1 units $2 --" | tee -a "$log"
  bench/coop_cost --mode 2 --tt 8 --grid 100 --pw "$1" --phase-units "$2" --exact-units \
      --phases 32 --rounds 4 2>&1 | tee -a "$log"
done
# and the same pair at four times the bytes, where the long arm has four rounds
for spec in "5 1600" "20 400" "40 200"; do
  set -- $spec
  echo "-- pw $1 units $2 --" | tee -a "$log"
  bench/coop_cost --mode 2 --tt 8 --grid 100 --pw "$1" --phase-units "$2" --exact-units \
      --phases 32 --rounds 4 2>&1 | tee -a "$log"
done
echo "cause panel written to $log"

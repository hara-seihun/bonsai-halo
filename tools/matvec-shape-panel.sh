#!/usr/bin/env bash
# The deployed matvec's unit shape, priced on the row count that runs it.
#
#   tools/run-batch-compare --pin-clock --exec tools/matvec-shape-panel.sh OUTDIR
#
# The drafted verify pass runs seven matvec phases through one body and reads them at 113 to
# 203 GB/s. The two things that differ between those phases are blocks per wave per unit (pw)
# and how many units the phase deals onto a 100-workgroup grid. This walks both, at the row
# count the pass runs (tt 8) and at the single-token control (tt 1), with no model in the way.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
out=${1:?usage: matvec-shape-panel.sh OUTDIR}
mkdir -p "$out"
cd "$root"
grid=${GRID:-100}
log=$out/shape-surface.txt
: > "$log"
{
  printf 'revision %s\nprobe %s\ngrid %s\n' "$(git rev-parse --short HEAD)" \
      "$(sha256sum bench/coop_cost | cut -c1-16)" "$grid"
} | tee -a "$log"
for tt in 8 1; do
  for units in 100 200 300 320 400 512 1000 1088 2000 7760; do
    echo "== tt $tt units $units ==" | tee -a "$log"
    bench/coop_cost --mode 2 --tt "$tt" --grid "$grid" --pw 3,5,8 \
        --phase-units "$units" --exact-units --phases 32 --rounds 4 2>&1 | tee -a "$log"
  done
done
echo "shape surface written to $log"

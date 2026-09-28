#!/usr/bin/env bash
# The shapes a deeper K split would give the eight-row pass, priced against the deployed ones.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd); out=${1:?outdir}; mkdir -p "$out"; cd "$root"
log=$out/shape-ksplit.txt; : > "$log"
echo "revision $(git rev-parse --short HEAD)" | tee -a "$log"
# deployed -> split, per phase:  (pw, units)
#   gate/up   (5, 1088) -> (1, 5440)      qkv+z (5, 512) -> (1, 2560)     qkv (5, 416) -> (1, 2080)
#   ssm_out/o (3, 320)  -> (2, 480) or (1, 960)          down (8, 320) -> (1, 2720)
for spec in "1 960" "2 480" "1 2720" "1 2560" "1 5440" "2 1360" "1 320" "2 320" "4 320"; do
  set -- $spec
  echo "-- pw $1 units $2 --" | tee -a "$log"
  bench/coop_cost --mode 2 --tt 8 --grid 100 --pw "$1" --phase-units "$2" --exact-units \
      --phases 32 --rounds 4 2>&1 | tail -1 | tee -a "$log"
done

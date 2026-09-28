#!/usr/bin/env bash
# The eight-row matvec's matrix body against the dot4 body it replaces: two binaries from one tree,
# one define apart, arms alternating inside one lock hold. The drafter's own phase, gdn, attn and
# the preps are the in-panel control; the greedy digest and the exact drafted/accepted integers are
# the numerical control.
#
#   tools/run-batch-compare --pin-clock -- tools/mv8-ab.sh 'ctl wmma8 wmma8 ctl' OUTDIR
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd); out=${2:?usage: mv8-ab.sh ARMS OUTDIR}; mkdir -p "$out"; cd "$root"
drafter=../../data/bonsai2/drafters/dflash2.safetensors
for b in bonsai-halo-ctl bonsai-halo-wmma8; do printf '%s %s\n' "$b" "$(sha256sum $b | cut -c1-24)"; done | tee "$out/build.txt"
git rev-parse --short HEAD >> "$out/build.txt"
i=0
for arm in $1; do
  i=$((i+1))
  echo "== round $i arm $arm =="
  HALO_PROFILE=1 ./bonsai-halo-$arm --bench --dflash "$drafter" -n "${NGEN:-24}" \
      --prompts "${PROMPTS:-bench/mv8-prompts.txt}" >> "$out/ab-$arm.out" 2>> "$out/ab-$arm.txt"
  grep -E "drafted|verify|tok/s|digest" "$out/ab-$arm.txt" | tail -4
done

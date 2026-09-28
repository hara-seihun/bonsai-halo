#!/usr/bin/env bash
# The retiled eight-row matvec against the split it ships with, same tree, same flags,
# one constant apart. Arms alternate inside one lock hold; the vocabulary head (KS = 1 in
# both arms), gdn, attn and the prep phases are the in-panel control.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd); out=${2:?usage: retile-ab.sh ARMS OUTDIR}; mkdir -p "$out"; cd "$root"
drafter=../../data/bonsai2/drafters/dflash2.safetensors
for arm in $1; do
  echo "== arm $arm =="
  HALO_PROFILE=1 ./bonsai-halo-$arm --bench --dflash "$drafter" -n "${NGEN:-32}" \
      --prompts "${PROMPTS:-bench/drafter-prompts.txt}" >> "$out/ab-$arm.out" 2>> "$out/ab-$arm.txt"
  tail -3 "$out/ab-$arm.txt" | head -1
done

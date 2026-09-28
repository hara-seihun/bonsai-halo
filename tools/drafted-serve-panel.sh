#!/usr/bin/env bash
# One bounded panel for the served drafted step: the verify pass's phase table, the
# drafter's segment timeline and the step's acceptance, on one build and one clock.
#
#   tools/run-batch-compare --pin-clock -- tools/drafted-serve-panel.sh OUTDIR
#
# Two processes, because one `prof` buffer serves both cooperative kernels: the verify
# pass's phases and the drafter's segments cannot be read in the same run.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
arm=${1:?usage: drafted-serve-panel.sh ARM OUTDIR}
out=${2:?usage: drafted-serve-panel.sh ARM OUTDIR}
mkdir -p "$out"
cd "$root"
drafter=../../data/bonsai2/drafters/dflash2.safetensors
ngen=${NGEN:-60}

rev=$(git rev-parse --short HEAD)
sha=$(sha256sum bonsai-halo | cut -c1-24)
printf 'revision %s\nbinary %s\nngen %s\n' "$rev" "$sha" "$ngen" | tee "$out/build.txt"

case "$arm" in
# the verify pass, phase by phase, during a drafted run
verify) HALO_PROFILE=1 ./bonsai-halo --bench --dflash "$drafter" -n "$ngen" \
            --prompts bench/drafter-prompts.txt > "$out/verify-phases.out" 2> "$out/verify-phases.txt" ;;
# the drafter's own segments over the same workload
draft)  HALO_PROFILE=1 HALO_PROFILE_DRAFT=1 ./bonsai-halo --bench --dflash "$drafter" -n "$ngen" \
            --prompts bench/drafter-prompts.txt > "$out/draft-segments.out" 2> "$out/draft-segments.txt" ;;
# the single-row control: the same kernels with one row per pass, no drafter
plain)  HALO_PROFILE=1 ./bonsai-halo --bench -n "$ngen" > "$out/plain-phases.out" 2> "$out/plain-phases.txt" ;;
*) echo "unknown arm $arm" >&2; exit 2 ;;
esac

echo "panel arm complete: $arm -> $out"

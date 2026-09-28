#!/usr/bin/env bash
# The phase stagger on the route the machine serves: the drafted step, control and arm alternating
# in palindrome order so drift across the panel shows as a disagreement between an arm's two cells.
# Two binaries one -D apart (the arm's constant is compile-time, so a build is the only axis), both
# trimmed the same way, run back to back inside one lock hold.
#
#   tools/run-batch-compare --pin-clock --exec bash bench/stagger-engine-panel.sh CTL ARM [NGEN]
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
ctl=${1:?usage: stagger-engine-panel.sh CTL_BINARY ARM_BINARY [NGEN]}
arm=${2:?usage: stagger-engine-panel.sh CTL_BINARY ARM_BINARY [NGEN]}
ngen=${3:-40}
drafter=../../data/bonsai2/drafters/dflash2.safetensors

for cell in ctl arm ctl arm; do
    bin=$ctl; [[ $cell == arm ]] && bin=$arm
    echo "--- cell $cell ($(sha256sum "$bin" | cut -c1-16))"
    "$bin" --bench --dflash "$drafter" -n "$ngen" --prompts bench/drafter-prompts.txt 2>&1 |
        grep -Ei "tok/s|digest|drafted|accepted|prompt|verify|step" | tail -8
done

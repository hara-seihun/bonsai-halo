#!/usr/bin/env bash
# What a tiny phase between two streamed phases costs the stream.
#
# The engine never runs two matvec phases back to back: a prep phase normalises and quantises the
# row every matvec reads, so the deployed verify pass alternates 60-234 us of stream with 5-10 us
# of prep, 580 phases deep. `bench/coop_cost` has always run one long stream against another, and
# it reads 15-28% higher than the engine at the engine's own shapes (docs/weight-placement.md).
# `--gap-units` inserts the engine's kind of neighbour; `--no-stream` prices that neighbour alone.
#
#   base        matvec phases only, what the probe has always measured
#   gap0        one extra empty barrier per matvec phase
#   gap40       one 40-unit prep-sized phase per matvec phase (prep_norm deals 40)
#   *-only      the same gap phases with no stream at all: the subtraction control
set -euo pipefail
cd "$(dirname "$0")/.."
probe=bench/coop_cost
grid=${GRID:-120}
phases=${PHASES:-64}
rounds=${ROUNDS:-4}
run() { printf '\n== %s ==\n' "$1"; shift; $probe --mode 2 --tt 8 --grid "$grid" --phases "$phases" --rounds "$rounds" --exact-units "$@"; }

for pass in 1 2; do
    printf '\n######## pass %d ########\n' "$pass"
    for cell in "5 1088" "3 320"; do
        set -- $cell
        pw=$1 units=$2
        run "pw$pw u$units base"      --pw "$pw" --phase-units "$units" --bytes 2048
        run "pw$pw u$units gap0"      --pw "$pw" --phase-units "$units" --bytes 2048 --gap-units 0
        run "pw$pw u$units gap40"     --pw "$pw" --phase-units "$units" --bytes 2048 --gap-units 40
        run "pw$pw u$units gap0only"  --pw "$pw" --phase-units "$units" --bytes 2048 --gap-units 0 --no-stream
        run "pw$pw u$units gap40only" --pw "$pw" --phase-units "$units" --bytes 2048 --gap-units 40 --no-stream
    done
done

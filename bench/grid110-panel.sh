#!/bin/sh
P=../../data/bonsai2/drafters/dflash2.safetensors
CAN=./bonsai-halo
NEW=../work/clones/rows-lds-occupancy/bonsai-halo.wpe13
cd . || exit 1
for pair in "A100 $CAN -1" "B110 $NEW 110" "B100 $NEW 100" "B110 $NEW 110" "A100 $CAN -1"; do
    set -- $pair
    printf '@@ %s\n' "$1"
    if [ "$3" = "-1" ]; then "$2" --bench --dflash "$P" -n 48 2>&1 | grep -E '^dflash2:'
    else "$2" --bench --dflash "$P" -n 48 --grid-rows "$3" 2>&1 | grep -E '^dflash2:'; fi
done

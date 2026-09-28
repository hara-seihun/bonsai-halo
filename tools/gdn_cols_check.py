#!/usr/bin/env python3
"""Prove the column-ownership arm reproduces `warp_sum`'s two roundings, in float32, without a GPU.

`warp_sum` is not a broadcast of one value. It is five DPP stages -
`quad_perm(1,0,3,2)`, `quad_perm(2,3,0,1)`, `row_ror 4`, `row_ror 8`, `permlanex16` - and after them
lane `l` holds `(Q_j + Q_{j+1}) + (Q_{j+2} + Q_{j+3})` with `j = (l>>2)&3`, the quad indices taken
mod 4 and `Q_q` the sum of quad `q`'s four partials. Because IEEE addition is commutative, `j = 2`
reproduces `j = 0` and `j = 3` reproduces `j = 1`, so the wave holds exactly TWO bit patterns of the
same mathematical sum. `gdn_token` then applies lane `l`'s value to the columns lane `l` owns, so
the deployed recurrence already updates column `c` with the delta of class `((c&31)>>2)&1` and
writes an output from lane 0, which is always the even class.

The column arm gives one lane 32 columns of one row and has to rebuild both classes from the eight
column groups that lane holds. This file checks that reconstruction the only way that settles it:
emulate both expressions on the same float32 partials and compare the bits.

    tools/gdn_cols_check.py                 # 20000 random waves, mixed magnitudes
    tools/gdn_cols_check.py --cases 200000

It proves the reduction, which is the part of the change an argument could get wrong. It does not
prove the products that feed it: those are the same `fmaf` on the same operands in both arms, which
is why the partials are this check's input rather than its subject. The residual FNV-64 over a
128-row pass is the end-to-end control.
"""
import argparse
import sys

import numpy as np

F32 = np.float32


def dpp_quad_perm(v, sel):
    """One `v_mov_b32 dpp quad_perm:[...]` over 32 lanes: lane l reads lane (l & ~3) | sel[l & 3]."""
    out = np.empty_like(v)
    for l in range(32):
        out[l] = v[(l & ~3) | sel[l & 3]]
    return out


def dpp_row_ror(v, n):
    """`row_ror:n` over rows of 16 lanes: lane l reads lane (l - n) mod 16 inside its row.

    The direction is MEASURED, not read: `tools/gdn_warpsum_probe` runs each DPP stage on the lane
    index on device and prints the permutation. `row_ror 4` reports lane 0 <- 12 and lane 4 <- 0.
    The natural reading of "rotate right" gives the other direction, and an emulation written from
    that reading will happily agree with a kernel written from the same reading while both disagree
    with the hardware by a few ULP. That is what happened here; the probe is the control.
    """
    out = np.empty_like(v)
    for l in range(32):
        base, m = l & ~15, l & 15
        out[l] = v[base | ((m - n) & 15)]
    return out


def permlanex16(v):
    """`v_permlanex16_b32` with the identity selectors: lane l reads lane l ^ 16."""
    return v[np.arange(32) ^ 16]


def warp_sum(part):
    """kernels/device.hpp:warp_sum, lane for lane, in float32."""
    v = part.copy()
    v = v + dpp_quad_perm(v, [1, 0, 3, 2])
    v = v + dpp_quad_perm(v, [2, 3, 0, 1])
    v = v + dpp_row_ror(v, 4)
    v = v + dpp_row_ror(v, 8)
    return v + permlanex16(v)


def cols_reduce(part):
    """kernels/halo_rows.hip:gdn_cols_reduce over the four sub-lanes of one row.

    Sub-lane `p` holds the partials of deployed lanes `8p .. 8p+7`, which are deployed quads
    `(p>>1, 2*(p&1))` and `(p>>1, 2*(p&1)+1)`. Returns (class 0, class 1) per sub-lane.
    """
    A = np.empty(4, dtype=F32)
    B = np.empty(4, dtype=F32)
    for p in range(4):
        q = part[8 * p:8 * p + 8]
        A[p] = (q[0] + q[1]) + (q[2] + q[3])
        B[p] = (q[4] + q[5]) + (q[6] + q[7])
    swap1 = [1, 0, 3, 2]
    swap2 = [2, 3, 0, 1]
    e = np.array([A[p] + B[swap1[p]] for p in range(4)], dtype=F32)
    e = np.array([e[p] + e[swap1[p]] for p in range(4)], dtype=F32)
    e = np.array([e[p] + e[swap2[p]] for p in range(4)], dtype=F32)
    o = np.array([A[p] + B[p] for p in range(4)], dtype=F32)
    o = np.array([o[p] + o[swap1[p]] for p in range(4)], dtype=F32)
    o = np.array([o[p] + o[swap2[p]] for p in range(4)], dtype=F32)
    return e, o


def column_classes():
    """Which delta class each of a row's 128 columns is updated with, from both sides.

    Deployed: column `c` lives in lane `c & 31`, whose class is `((c&31)>>2)&1`.
    Column arm: column `c = 8p + i + 32s` is in this lane's group `i`, and the arm uses the even
    class for `i < 4`. The two have to agree column for column.
    """
    bad = []
    for c in range(128):
        deployed = ((c & 31) >> 2) & 1
        p, i = (c & 31) >> 3, (c & 31) & 7
        arm = 0 if i < 4 else 1
        if deployed != arm or 8 * p + i != (c & 31):
            bad.append((c, deployed, arm))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260921)
    args = ap.parse_args()

    bad_cols = column_classes()
    print(f"column -> delta class agreement over all 128 columns: "
          f"{'OK' if not bad_cols else f'{len(bad_cols)} MISMATCHES {bad_cols[:8]}'}")

    rng = np.random.default_rng(args.seed)
    # Mixed magnitudes on purpose: a reassociation that only shows up when the terms differ by
    # more than the mantissa is exactly what this has to be able to see.
    scales = [1.0, 1e-3, 1e3, 1e-8, 1e8]
    worst_rel = 0.0
    mismatch = 0
    classes_distinct = 0
    for n in range(args.cases):
        s = scales[n % len(scales)]
        part = (rng.standard_normal(32) * s).astype(F32)
        if n % 7 == 0:                       # one big term among small ones
            part[rng.integers(32)] = F32(s * 1e6)
        ws = warp_sum(part)
        e, o = cols_reduce(part)
        if e[0].view(np.uint32) != e[3].view(np.uint32) or o[0].view(np.uint32) != o[2].view(np.uint32):
            print(f"case {n}: the arm's own sub-lanes disagree", file=sys.stderr)
            return 2
        if e[0].view(np.uint32) != o[0].view(np.uint32):
            classes_distinct += 1
        for l in range(32):
            want = e[0] if ((l >> 2) & 1) == 0 else o[0]
            if ws[l].view(np.uint32) != want.view(np.uint32):
                mismatch += 1
                if mismatch < 4:
                    print(f"case {n} lane {l}: warp_sum {ws[l]!r} arm {want!r}", file=sys.stderr)
        ref = np.float64(part.astype(np.float64).sum())
        if ref != 0.0:
            worst_rel = max(worst_rel, abs(float(e[0]) - ref) / abs(ref))

    print(f"cases {args.cases}, differing bits across all 32 lanes: {mismatch}")
    print(f"cases where the two classes are genuinely different bit patterns: {classes_distinct}"
          f" ({100.0 * classes_distinct / args.cases:.1f}%)")
    print(f"worst relative gap of the even class against a float64 sum: {worst_rel:.3e}")
    ok = mismatch == 0 and not bad_cols
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


sys.exit(main())

#!/usr/bin/env python3
"""The Gram form of a deferred append step, against the rebuild it replaces.

`kernels/gdn_state_codec.hpp` replaces the materialised state row of an append step with two
contractions. The identity is easy to state and easy to get wrong in the indices, and the device
has no cheap way to tell you which: a sign or a stride error produces plausible text and only the
teacher-forced panel notices, which is a quality run and a GPU lock away.

So this mirrors both paths in double precision, from the source expressions and with the region
offsets the device uses, on random data:

    tools/gdn_gram_check.py

The rebuild path here is transcribed from `gdn_defer_rebuild` and `gdn_token`; the Gram path from
`gdn_defer_stage_gram` and `gdn_defer_append_gram`. Agreement is a statement about the algebra and
the layout, not about fp32 rounding - the two arms are deliberately different maps in float, which
is why the deferred residual hash cannot be the acceptance instrument for this change.
"""
import sys

import numpy as np

SS = 128
C_MAX = 8                       # GDN_DEFER_MAX
DEFER_HEAD = C_MAX * (1 + 2 * SS)
GDN_OUT_SCALE = 0.08838834764831845


def region_offsets(h):
    """Where one head's pending triples live, mirroring gdn_defer_head/_deltas/_append_head."""
    base = h * DEFER_HEAD
    return {
        "g": base,                                  # H[i]
        "a": base + C_MAX,                          # A[i*SS + row]
        "k": base + C_MAX * (1 + SS),               # K[i*SS + dim]
    }


def stage_weights(H, pend):
    """gdn_defer_stage: w[i] = prod_{t>i} g_t, and w[C_MAX] = what the committed base is worth."""
    w = np.zeros(C_MAX + 1)
    acc = 1.0
    for i in range(pend - 1, -1, -1):
        w[i] = acc
        acc *= H[i]
    w[C_MAX] = acc
    return w


def rebuild_path(B, region, off, w, pend, g, beta, k, q, v):
    """gdn_defer_rebuild followed by gdn_token: materialise the row, then contract it."""
    base = w[C_MAX] if pend > 0 else 1.0
    m = base * B.copy()
    for i in range(pend - 1, -1, -1):
        a = region[off["a"] + i * SS: off["a"] + i * SS + SS]
        ki = region[off["k"] + i * SS: off["k"] + i * SS + SS]
        m += w[i] * np.outer(a, ki)
    m *= g
    part = m @ k
    delta = (v - part) * beta
    m += np.outer(delta, k)
    return delta, (m @ q) * GDN_OUT_SCALE


def gram_path(B, region, off, w, pend, g, beta, k, q, v):
    """gdn_defer_stage_gram followed by gdn_defer_append_gram: contract, never materialise."""
    wbase = w[C_MAX] if pend > 0 else 1.0
    gk = np.zeros(C_MAX)
    gq = np.zeros(C_MAX)
    for i in range(pend):
        ki = region[off["k"] + i * SS: off["k"] + i * SS + SS]
        gk[i] = ki @ k
        gq[i] = ki @ q
    rho = k @ q
    dk = wbase * (B @ k)
    dq = wbase * (B @ q)
    for i in range(pend - 1, -1, -1):
        a = region[off["a"] + i * SS: off["a"] + i * SS + SS]
        c = w[i] * a
        dk = dk + c * gk[i]
        dq = dq + c * gq[i]
    delta = (v - g * dk) * beta
    return delta, (g * dq + delta * rho) * GDN_OUT_SCALE


def staged_deltas(region, off, pend, part, rpu):
    """The LDS delta staging: dels[i*rpu + p] must be A[i*SS + part*rpu + p]."""
    dels = np.zeros(pend * rpu)
    for idx in range(pend * rpu):
        dels[idx] = region[off["a"] + (idx // rpu) * SS + part * rpu + idx % rpu]
    return dels


def main():
    rng = np.random.default_rng(20260921)
    worst_delta = worst_out = 0.0
    for trial in range(200):
        h = int(rng.integers(0, 48))
        pend = int(rng.integers(0, C_MAX))
        off = region_offsets(h)
        region = np.zeros(48 * DEFER_HEAD)
        region[off["g"]: off["g"] + pend] = rng.uniform(0.85, 0.999, pend)
        region[off["a"]: off["a"] + pend * SS] = rng.normal(0, 0.3, pend * SS)
        for i in range(pend):                      # pending keys are L2 normalised
            ki = rng.normal(0, 1, SS)
            region[off["k"] + i * SS: off["k"] + i * SS + SS] = ki / np.linalg.norm(ki)

        B = rng.normal(0, 0.05, (SS, SS))
        k = rng.normal(0, 1, SS); k /= np.linalg.norm(k)
        q = rng.normal(0, 1, SS); q /= np.linalg.norm(q)
        v = rng.normal(0, 1, SS)
        g = float(rng.uniform(0.85, 0.999))
        beta = float(rng.uniform(0.05, 0.95))
        w = stage_weights(region[off["g"]: off["g"] + C_MAX], pend)

        d0, o0 = rebuild_path(B, region, off, w, pend, g, beta, k, q, v)
        d1, o1 = gram_path(B, region, off, w, pend, g, beta, k, q, v)
        scale = max(1e-12, float(np.abs(o0).max()))
        worst_delta = max(worst_delta, float(np.abs(d0 - d1).max() / max(1e-12, np.abs(d0).max())))
        worst_out = max(worst_out, float(np.abs(o0 - o1).max() / scale))

        for split in (4, 8, 16):
            rpu = SS // split
            for part in range(split):
                dels = staged_deltas(region, off, pend, part, rpu)
                for i in range(pend):
                    for r in range(16 // split):
                        for wave in range(8):
                            p = wave + 8 * r
                            if p >= rpu:
                                continue
                            got = dels[i * rpu + p]
                            want = region[off["a"] + i * SS + part * rpu + p]
                            if got != want:
                                print(f"delta staging mismatch split {split} part {part} term {i} row {p}")
                                return 1

    print(f"200 trials, depth 0..{C_MAX - 1}: "
          f"worst relative delta difference {worst_delta:.3e}, worst output difference {worst_out:.3e}")
    print("delta staging index agrees for SPLIT 4, 8 and 16 at every part")
    ok = worst_delta < 1e-9 and worst_out < 1e-9
    print("IDENTITY HOLDS" if ok else "IDENTITY FAILS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

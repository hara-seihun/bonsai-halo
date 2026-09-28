#!/usr/bin/env python3
"""Block forms of one GDN head's observed map, checked against the source recurrence.

The reference is a direct transcription of `gdn_token` in `kernels/halo_rows.hip`
(state rows = value dim, columns = key dim):

    M *= g                              # alpha decay, scalar per token and head
    p  = M @ k                          # = g * (S_{t-1} k_t)
    d  = (v - p) * beta                 # rank-one delta correction
    M += outer(d, k)
    o  = (M @ q) * 1/sqrt(128)          # output uses the POST-update state

Every alternative here computes the same real-arithmetic function of
(S_0, g, beta, k, q, v) -> (o_1..o_m, S_m) with a different association order.
Float64 agreement to a few ulps is evidence the algebra is right; it is not a
claim that any of them is bit-identical to the kernel's float32 output.

    python3 docs/gdn-map/gdn_block.py [--m 128] [--seed 7] [--dump FILE]
"""
import argparse
import sys

import numpy as np

SCALE = 0.08838834764831845  # 1/sqrt(128), as spelled in halo_rows.hip


# ---------------------------------------------------------------------------
# reference: the source recurrence, one token at a time
# ---------------------------------------------------------------------------
def serial_source(S0, g, beta, K, Q, V):
    """Transcription of gdn_token. Returns (O, S_m)."""
    m, dv = V.shape
    S = S0.copy()
    O = np.empty((m, dv), dtype=S0.dtype)
    for t in range(m):
        S *= g[t]
        p = S @ K[t]
        d = (V[t] - p) * beta[t]
        S += np.outer(d, K[t])
        O[t] = (S @ Q[t]) * SCALE
    return O, S


def serial_preupdate_output(S0, g, beta, K, Q, V):
    """Identity A: o_t = g_t (S_{t-1} q_t) + (k_t . q_t) delta_t.

    Both state contractions now read the same pre-update state, so they are
    independent; nothing between them depends on the other.
    """
    m, dv = V.shape
    S = S0.copy()
    O = np.empty((m, dv), dtype=S0.dtype)
    for t in range(m):
        S *= g[t]
        a = S @ K[t]                     # independent ...
        b = S @ Q[t]                     # ... of this one
        d = (V[t] - a) * beta[t]
        O[t] = (b + d * float(K[t] @ Q[t])) * SCALE
        S += np.outer(d, K[t])
    return O, S


def serial_deferred_decay(S0, g, beta, K, Q, V, limit=1e30):
    """Identity B: carry S_hat = S_t / gamma_t and fold the decay into a scalar.

    The per-element decay multiply disappears; 1/gamma_t grows, so the state is
    rescaled whenever the running reciprocal would leave a safe range. Returns
    (O, S_m, rescales).
    """
    m, dv = V.shape
    S = S0.copy()
    O = np.empty((m, dv), dtype=S0.dtype)
    gam = 1.0
    rescales = 0
    for t in range(m):
        gam *= g[t]
        if gam < 1.0 / limit:            # scalar test, uniform across the head
            S *= gam                     # one multiply per element, then restart
            gam = 1.0
            rescales += 1
        a = S @ K[t]
        b = S @ Q[t]
        d = (V[t] - a * gam) * beta[t]
        O[t] = (b * gam + d * float(K[t] @ Q[t])) * SCALE
        S += np.outer(d / gam, K[t])
    return O, S * gam, rescales


# ---------------------------------------------------------------------------
# block forms
# ---------------------------------------------------------------------------
def decay_ratios(g):
    """R[t, s] = prod_{r=s+1..t} g_r for s <= t, 0 above the diagonal.

    Accumulated as segment products with no division anywhere. Forming
    gamma_t / gamma_s from two cumulative products is the thing to avoid: for a
    fast-decaying head both underflow and the quotient is 0/0. Every entry here
    is at most 1 and underflows to zero, which is the contribution vanishing.
    A log-domain sum of the per-token log g is the other stable route.
    """
    m = g.shape[0]
    R = np.zeros((m, m), dtype=g.dtype)
    R[0, 0] = 1.0
    for t in range(1, m):
        R[t, :t] = R[t - 1, :t] * g[t]
        R[t, t] = 1.0
    return R


def block_chunk(S0, g, beta, K, Q, V):
    """One chunk, WY/triangular form. No intermediate 128x128 state exists.

    gamma_t      = prod_{s<=t} g_s
    Ntil[t,s]    = (gamma_t/gamma_s) (k_s . k_t),  s < t      strictly lower
    Ctil[t,s]    = (gamma_t/gamma_s) (q_t . k_s),  s <= t     lower, inclusive
    (I + diag(b) Ntil) D = diag(b) (V - diag(gamma) K S0^T)   forward substitution
    O            = (diag(gamma) Q S0^T + Ctil D) / sqrt(128)
    S_m          = gamma_m S0 + sum_t (gamma_m/gamma_t) d_t k_t^T
    """
    m = K.shape[0]
    R = decay_ratios(g)
    gam = R[:, 0] * g[0]                 # gamma_t = prod_{s<=t} g_s
    Ntil = np.tril(R * (K @ K.T), -1)
    Ctil = np.tril(R * (Q @ K.T))
    B = V - gam[:, None] * (K @ S0.T)    # m x dv
    D = np.empty_like(B)
    for t in range(m):                   # unit lower triangular solve
        acc = B[t] if t == 0 else B[t] - Ntil[t, :t] @ D[:t]
        D[t] = beta[t] * acc
    O = (gam[:, None] * (Q @ S0.T) + Ctil @ D) * SCALE
    tail = R[m - 1]                      # gamma_m / gamma_t as a segment product
    Sm = gam[m - 1] * S0 + (tail[:, None] * D).T @ K
    return O, Sm


def block_subchunked(S0, g, beta, K, Q, V, c):
    """Same map, sub-chunks of c tokens. Keeps the m x m objects c x c."""
    m = K.shape[0]
    S = S0
    out = []
    for i in range(0, m, c):
        j = min(i + c, m)
        O, S = block_chunk(S, g[i:j], beta[i:j], K[i:j], Q[i:j], V[i:j])
        out.append(O)
    return np.concatenate(out), S


def factored_state(S0, g, beta, K, Q, V, r):
    """Never materialise the full state between tokens: carry base plus r pairs.

    S_t = gamma S_base + sum_i c_i u_i k_i^T, every coefficient <= 1. Each token
    reads the base, appends its own (delta_t, k_t), and the base absorbs the
    pending pairs with one rank-r update every r tokens. This is the m-token
    block map applied to the write side of a one-token pass.
    """
    m, dv = V.shape
    base = S0.copy()
    gam = 1.0
    pend = []
    O = np.empty((m, dv), dtype=S0.dtype)
    for t in range(m):
        gam *= g[t]                                  # decayed base coefficient
        for p in pend:
            p[0] *= g[t]                             # r scalar multiplies, not d*dv
        a = gam * (base @ K[t]) + sum(c * u * float(k @ K[t]) for c, u, k in pend)
        b = gam * (base @ Q[t]) + sum(c * u * float(k @ Q[t]) for c, u, k in pend)
        d = (V[t] - a) * beta[t]
        O[t] = (b + d * float(K[t] @ Q[t])) * SCALE
        pend.append([1.0, d, K[t]])
        if len(pend) == r:                           # rank-r materialisation
            base = gam * base + sum(c * np.outer(u, k) for c, u, k in pend)
            gam, pend = 1.0, []
    S = gam * base + sum(c * np.outer(u, k) for c, u, k in pend)
    return O, S


def block_attention_form(S0, g, beta, K, Q, V):
    """Fully materialised chunk map: an m x m causal operator plus a rank-m state map.

    T    = (I + diag(b) Ntil)^-1 diag(b)           lower triangular
    A    = Ctil T                                  causal token mixing
    Qeff = diag(gamma) Q - A diag(gamma) K         modified queries
    O    = (Qeff S0^T + A V) / sqrt(128)
    Psi  = T^T diag(gamma_m/gamma)
    S_m  = S0 (gamma_m I - K^T diag(gamma) Psi K) + V^T Psi K
    """
    m = K.shape[0]
    R = decay_ratios(g)
    gam = R[:, 0] * g[0]
    Ntil = np.tril(R * (K @ K.T), -1)
    Ctil = np.tril(R * (Q @ K.T))
    T = np.linalg.solve(np.eye(m) + beta[:, None] * Ntil, np.diag(beta))
    A = Ctil @ T
    Qeff = gam[:, None] * Q - A @ (gam[:, None] * K)
    O = (Qeff @ S0.T + A @ V) * SCALE
    Psi = T.T * R[m - 1][None, :]        # gamma_m/gamma_t, segment product, no division
    Sm = S0 @ (gam[m - 1] * np.eye(K.shape[1]) - K.T @ (gam[:, None] * (Psi @ K))) + V.T @ Psi @ K
    return O, Sm, T, A


def wy_operator(beta, K):
    """prod_t (I - beta_t k_t k_t^T) as I - K^T T0^T K, with T0 the undecayed T."""
    m, d = K.shape
    T0 = np.linalg.solve(np.eye(m) + beta[:, None] * np.tril(K @ K.T, -1), np.diag(beta))
    P = np.eye(d)
    for t in range(m):
        P = P @ (np.eye(d) - beta[t] * np.outer(K[t], K[t]))
    return P, np.eye(d) - K.T @ T0.T @ K


# ---------------------------------------------------------------------------
# operands with the shapes and ranges the kernel actually sees
# ---------------------------------------------------------------------------
def make_case(rng, m, d=128, dv=128, fast_decay=False, normalise=True):
    K = rng.standard_normal((m, d))
    Q = rng.standard_normal((m, d))
    if normalise:                                     # ph_gdn_pre L2-normalises q and k
        K /= np.linalg.norm(K, axis=1, keepdims=True)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    V = rng.standard_normal((m, dv))
    # ph_gdn_pre divides by max(sqrt(sum of squares), NORM_EPS), so over the reals
    # ||k|| <= 1, equal to 1 whenever the pre-norm magnitude clears NORM_EPS.
    S0 = rng.standard_normal((dv, d)) * 0.5
    beta = 1.0 / (1.0 + np.exp(-rng.standard_normal(m)))              # sigmoid
    a = -9.0 if fast_decay else -0.35                                 # ssm_a is negative
    x = rng.standard_normal(m) * 0.7 - 1.5                            # alpha + dt.bias
    sp = np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))
    g = np.exp(a * sp)
    return S0, g, beta, K, Q, V


def rel(a, b):
    den = max(np.abs(b).max(), 1e-300)
    return float(np.abs(a - b).max() / den)


# ---------------------------------------------------------------------------
# float32 reordering study
# ---------------------------------------------------------------------------
def lane_dot_f32(row, vec):
    """The kernel's dot order: 4 elements per lane, then a 5-stage DPP tree.

    Lane l owns columns l + 32*s2. numpy has no fused multiply-add, so this
    reproduces the summation tree, not the kernel's rounding.
    """
    f32 = np.float32
    part = np.zeros(32, dtype=f32)
    for s2 in range(4):                                   # per-lane serial chain
        part = (part + (row[np.arange(32) + 32 * s2] * vec[np.arange(32) + 32 * s2]).astype(f32)).astype(f32)
    for step in (1, 2, 4, 8, 16):                         # quad_perm, row_ror, permlanex16
        part = (part + part[np.arange(32) ^ step]).astype(f32)
    return part[0]


def serial_f32_kernel_order(S0, g, beta, K, Q, V):
    f32 = np.float32
    S = S0.astype(f32).copy()
    m, dv = V.shape
    O = np.empty((m, dv), dtype=f32)
    g32, b32 = g.astype(f32), beta.astype(f32)
    K32, Q32, V32 = K.astype(f32), Q.astype(f32), V.astype(f32)
    for t in range(m):
        S = (S * g32[t]).astype(f32)
        p = np.array([lane_dot_f32(S[j], K32[t]) for j in range(dv)], dtype=f32)
        d = ((V32[t] - p) * b32[t]).astype(f32)
        S = (S + np.outer(d, K32[t]).astype(f32)).astype(f32)
        O[t] = np.array([lane_dot_f32(S[j], Q32[t]) for j in range(dv)], dtype=f32) * f32(SCALE)
    return O, S


def block_f32(S0, g, beta, K, Q, V, c):
    f32 = np.float32
    return block_subchunked(S0.astype(f32), g.astype(f32), beta.astype(f32),
                            K.astype(f32), Q.astype(f32), V.astype(f32), c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--f32-m", type=int, default=32, help="chunk length for the float32 study")
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()

    out = []

    def say(s=""):
        out.append(s)
        print(s)

    rng = np.random.default_rng(args.seed)
    say(f"GDN block-map equivalence, float64, d=dv=128, seed={args.seed}")
    say()
    say("Reference: serial_source, a transcription of gdn_token (post-update output).")
    say("Reported numbers are max|difference| relative to the largest reference entry.")
    say()
    header = f"{'case':<34}{'m':>5}{'outputs O':>13}{'final state':>14}"
    for fast in (False, True):
        for norm in (True, False):
            for m in sorted({8, 32, args.m}):
                S0, g, beta, K, Q, V = make_case(rng, m, fast_decay=fast, normalise=norm)
                Oref, Sref = serial_source(S0, g, beta, K, Q, V)
                tag = ("fast decay" if fast else "slow decay") + (", unit k,q" if norm else ", raw k,q")
                say(f"--- {tag}, gamma_m = {np.prod(g):.3e}")
                say(header)
                for name, fn in (
                    ("identity A pre-update output", serial_preupdate_output),
                    ("identity B deferred decay", lambda *a: serial_deferred_decay(*a)[:2]),
                    ("block chunk (WY/triangular)", block_chunk),
                    ("block sub-chunked c=8", lambda *a: block_subchunked(*a, c=8)),
                    ("block sub-chunked c=16", lambda *a: block_subchunked(*a, c=16)),
                    ("attention form (Qeff, A, Psi)", lambda *a: block_attention_form(*a)[:2]),
                    ("factored state, materialise r=8", lambda *a: factored_state(*a, r=8)),
                ):
                    O, S = fn(S0, g, beta, K, Q, V)
                    say(f"{name:<34}{m:>5}{rel(O, Oref):>13.2e}{rel(S, Sref):>14.2e}")
                _, _, rescales = serial_deferred_decay(S0, g, beta, K, Q, V)
                say(f"{'  (identity B rescales)':<34}{m:>5}{rescales:>13d}")
                say()

    # WY operator identity, independent of the value path
    S0, g, beta, K, Q, V = make_case(rng, 64)
    P, WY = wy_operator(beta, K)
    say(f"prod_t (I - beta_t k_t k_t^T) vs I - K^T T0^T K, m=64: {rel(WY, P):.2e}")

    # the homogeneous part of the chunk map is that product scaled by gamma_m
    Z = np.zeros_like(S0)
    _, S_state_only = block_chunk(S0, g, beta, K, Q, np.zeros_like(V))
    _, S_value_only = block_chunk(Z, g, beta, K, Q, V)
    _, S_both = block_chunk(S0, g, beta, K, Q, V)
    say(f"homogeneous + particular == full chunk map (linearity in S0,V): "
        f"{rel(S_state_only + S_value_only, S_both):.2e}")
    say(f"homogeneous part == gamma_m S0 prod(I - beta k k^T):            "
        f"{rel(S_state_only, np.prod(g) * (S0 @ P)):.2e}")
    say()

    # float32: same algebra, different association order
    m32 = args.f32_m
    S0, g, beta, K, Q, V = make_case(rng, m32)
    O64, S64 = serial_source(S0, g, beta, K, Q, V)
    Ok, Sk = serial_f32_kernel_order(S0, g, beta, K, Q, V)
    say(f"float32 association study, m={m32} (numpy has no FMA: this measures")
    say("reordering sensitivity, not the kernel's bits)")
    say(f"{'variant':<34}{'O vs f64':>13}{'state vs f64':>14}")
    say(f"{'serial, kernel summation order':<34}{rel(Ok.astype(float), O64):>13.2e}"
        f"{rel(Sk.astype(float), S64):>14.2e}")
    for c in (8, 16, 32):
        Ob, Sb = block_f32(S0, g, beta, K, Q, V, c)
        say(f"{'block sub-chunked c=' + str(c):<34}{rel(Ob.astype(float), O64):>13.2e}"
            f"{rel(Sb.astype(float), S64):>14.2e}")
    Ob, Sb = block_f32(S0, g, beta, K, Q, V, 8)
    say(f"{'block c=8 vs serial f32 (each other)':<34}{rel(Ob.astype(float), Ok.astype(float)):>13.2e}"
        f"{rel(Sb.astype(float), Sk.astype(float)):>14.2e}")
    say()
    say("Neither float32 column is the exact real map, and neither reproduces the")
    say("other's bits. For ||k|| <= 1 and beta in (0,1) the per-token operator")
    say("g_t (I - beta_t k_t k_t^T) has norm at most 1, so an error injected at one")
    say("token is not amplified by later tokens; errors are still injected at every")
    say("token and accumulate. These are observations on these random operands, not")
    say("a bound, and they say nothing about the triangular system's condition")
    say("number, which is not established here for unit k either.")

    if args.dump:
        with open(args.dump, "w") as f:
            f.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

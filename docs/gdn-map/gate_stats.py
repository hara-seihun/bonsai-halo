#!/usr/bin/env python3
"""Per-head decay range of the real model's GDN gates.

Reads `ssm_a` and `ssm_dt.bias` straight out of the GGUF (both are F32) and
reports what `g = exp(ssm_a * softplus(alpha + dt))` can do over a chunk of m
tokens. The `alpha` term is a per-token projection this script cannot know, so
the numbers below bracket it: `alpha = 0` is the bias-only gate, and the sweep
shows how fast gamma_t = prod g falls for the fastest-forgetting heads.

    python3 docs/gdn-map/gate_stats.py [--model ../../data/bonsai2/PTQ1_0.gguf]
"""
import argparse
import struct
import sys

import numpy as np

GGUF_MAGIC = 0x46554747
# metadata value type ids
T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 = range(13)
FIXED = {T_U8: "B", T_I8: "b", T_U16: "H", T_I16: "h", T_U32: "I", T_I32: "i",
         T_F32: "f", T_BOOL: "?", T_U64: "Q", T_I64: "q", T_F64: "d"}
GGML_F32 = 0


class Reader:
    def __init__(self, f):
        self.f = f

    def u32(self):
        return struct.unpack("<I", self.f.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.f.read(8))[0]

    def string(self):
        return self.f.read(self.u64()).decode("utf-8", "replace")

    def value(self, t):
        if t == T_STR:
            return self.string()
        if t == T_ARR:
            et = self.u32()
            n = self.u64()
            return [self.value(et) for _ in range(n)]
        fmt = FIXED[t]
        return struct.unpack("<" + fmt, self.f.read(struct.calcsize(fmt)))[0]


def load_f32_tensors(path, want):
    """Return {name: np.float32 array} for the named F32 tensors."""
    out = {}
    with open(path, "rb") as f:
        r = Reader(f)
        if r.u32() != GGUF_MAGIC:
            raise SystemExit("not a GGUF file: " + path)
        version = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()
        alignment = 32
        for _ in range(n_kv):
            key = r.string()
            val = r.value(r.u32())
            if key == "general.alignment":
                alignment = int(val)
        infos = {}
        for _ in range(n_tensors):
            name = r.string()
            dims = [r.u64() for _ in range(r.u32())]
            ttype = r.u32()
            offset = r.u64()
            infos[name] = (dims, ttype, offset)
        base = f.tell()
        if base % alignment:
            base += alignment - base % alignment
        for name in want:
            if name not in infos:
                continue
            dims, ttype, offset = infos[name]
            if ttype != GGML_F32:
                raise SystemExit(f"{name} is type {ttype}, expected F32")
            n = int(np.prod(dims))
            f.seek(base + offset)
            out[name] = np.frombuffer(f.read(4 * n), dtype="<f4").astype(np.float64)
    return out, version


def softplus(x):
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../../data/bonsai2/PTQ1_0.gguf")
    ap.add_argument("--layers", type=int, default=64)
    ap.add_argument("--alpha", type=float, nargs="*", default=[-2.0, 0.0, 2.0],
                    help="stand-in values for the per-token alpha projection")
    args = ap.parse_args()

    names = []
    for l in range(args.layers):
        if (l + 1) % 4 == 0:
            continue  # attention layer, no GDN tensors
        names += [f"blk.{l}.ssm_a", f"blk.{l}.ssm_dt.bias"]
    tensors, version = load_f32_tensors(args.model, names)
    if not tensors:
        raise SystemExit("no ssm_a tensors found; wrong model file?")

    a = np.stack([tensors[n] for n in names if n.endswith("ssm_a")])
    dt = np.stack([tensors[n] for n in names if n.endswith("ssm_dt.bias")])
    print(f"model      {args.model} (GGUF v{version})")
    print(f"recurrent layers {a.shape[0]}, heads per layer {a.shape[1]}")
    print(f"ssm_a      min {a.min():+.4f}  max {a.max():+.4f}  mean {a.mean():+.4f}"
          f"   (sign: {'all negative' if (a < 0).all() else 'MIXED'})")
    print(f"dt.bias    min {dt.min():+.4f}  max {dt.max():+.4f}  mean {dt.mean():+.4f}")
    print()
    print("g = exp(ssm_a * softplus(alpha + dt.bias)), per head, over all recurrent layers.")
    print("alpha is a per-token projection of the hidden state, which this script cannot")
    print("know, so each row is a hypothetical gate at a stand-in alpha. These are ranges")
    print("the gate can take, not measured token gates; only the ssm_a line above is data.")
    print(f"{'alpha':>7} {'min g':>12} {'median g':>12} {'max g':>12} "
          f"{'gamma_16':>11} {'gamma_128':>11} {'1/gamma_128':>12}")
    for al in args.alpha:
        g = np.exp(a * softplus(al + dt))
        gmin = g.min()
        print(f"{al:>7.1f} {gmin:12.3e} {np.median(g):12.6f} {g.max():12.9f} "
              f"{gmin ** 16:11.3e} {gmin ** 128:11.3e} {1.0 / max(gmin ** 128, 1e-300):12.3e}")
    print()
    f32_min_normal = np.float32(np.finfo(np.float32).tiny)
    print(f"fp32 smallest normal {float(f32_min_normal):.3e}, largest finite "
          f"{float(np.finfo(np.float32).max):.3e}")
    print("A deferred-decay chunk carries 1/gamma_t, so it needs a rescale before")
    print("gamma_t leaves that range; the fastest heads reach it within a few tokens.")


if __name__ == "__main__":
    sys.exit(main())

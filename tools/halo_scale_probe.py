#!/usr/bin/env python3
"""Read the HALO tile cache index and report how a row's per-block fp16 scales vary.

The tile block stores 26 code bytes and a 2-byte fp16 scale per row (896 bytes per
32-row, 128-K block). The scale is 7.1% of the weight stream every matvec in this
engine reads. If a row's scale were constant across its K blocks, the same values
could be delivered from a per-row array and the stream would shrink by that much
with no numerical change at all. This tells you whether that is true, and what the
distribution looks like if it is not.
"""
import mmap, struct, sys
import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else "../../data/bonsai2/PTQ1_0.gguf.halo"
TILE_ROWS, BLOCK, TB = 32, 128, 896


def entries(mm):
    magic, src_size, src_mtime, n, data_off = struct.unpack_from("<8sQQQQ", mm, 0)
    assert magic == b"HALOCAC2", magic
    p = 40
    out = []
    for _ in range(n):
        (nl,) = struct.unpack_from("<I", mm, p); p += 4
        name = mm[p:p + nl].decode(); p += nl
        N, K, off, nbytes = struct.unpack_from("<qqqq", mm, p); p += 32
        out.append((name, N, K, off, nbytes))
    return out


def tensor_scales(mm, N, K, off, tiles=None):
    """scales[tile, block, lane] as uint16 bits."""
    nb, nt = K // BLOCK, N // TILE_ROWS
    sel = range(nt) if tiles is None else tiles
    out = np.empty((len(list(sel)), nb, TILE_ROWS), dtype=np.uint16)
    sel = range(nt) if tiles is None else tiles
    for i, T in enumerate(sel):
        base = off + T * nb * TB
        buf = np.frombuffer(mm, dtype=np.uint8, count=nb * TB, offset=base).reshape(nb, TB)
        tail = buf[:, 768:896].reshape(nb, TILE_ROWS, 4)
        out[i] = tail[:, :, 2].astype(np.uint16) | (tail[:, :, 3].astype(np.uint16) << 8)
    return out


def main():
    with open(PATH, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        ents = entries(mm)
        print(f"{len(ents)} tensors in {PATH}")
        total = sum(e[4] for e in ents)
        print(f"total {total/1e9:.3f} GB; scale bytes {total*2/896/1e9:.3f} GB ({2/28*100:.1f}% of codes+scale)")
        pick = []
        seen = set()
        for name, N, K, off, nbytes in ents:
            kind = name.split(".")[-2] if "." in name else name
            if kind in seen and len(pick) > 8:
                continue
            seen.add(kind)
            pick.append((name, N, K, off, nbytes))
        for name, N, K, off, nbytes in pick[:10]:
            nb, nt = K // BLOCK, N // TILE_ROWS
            tiles = list(range(0, nt, max(1, nt // 8)))[:8]
            s = tensor_scales(mm, N, K, off, tiles)          # [tile, block, lane]
            per_row_unique = np.array([len(np.unique(s[t, :, l])) for t in range(s.shape[0]) for l in range(TILE_ROWS)])
            f16 = s.view(np.float16).astype(np.float32)
            exp = (s >> 10) & 0x1F
            man = s & 0x3FF
            per_row_uexp = np.array([len(np.unique(exp[t, :, l])) for t in range(s.shape[0]) for l in range(TILE_ROWS)])
            print(f"\n{name}  N={N} K={K} nb={nb} ntiles={nt}")
            print(f"  distinct scales per row over {nb} blocks: min {per_row_unique.min()} med {int(np.median(per_row_unique))} max {per_row_unique.max()}")
            print(f"  distinct exponents per row: min {per_row_uexp.min()} med {int(np.median(per_row_uexp))} max {per_row_uexp.max()}")
            print(f"  scale range {f16.min():.5g} .. {f16.max():.5g}; mantissa low bits zero: {(man & 3 == 0).mean()*100:.1f}%")
            print(f"  distinct scale values in sample: {len(np.unique(s))} of {s.size}")
            u, c = np.unique(s, return_counts=True)
            top = np.argsort(-c)[:5]
            print(f"  top values: " + ", ".join(f"{f16.flat[0]*0+np.float32(np.frombuffer(np.uint16(u[i]).tobytes(),dtype=np.float16)[0]):.5g}x{c[i]}" for i in top))


if __name__ == "__main__":
    main()

import numpy as np, sys, os
ref, mine = sys.argv[1], sys.argv[2]
names = sys.argv[3:] if len(sys.argv) > 3 else None
files = sorted(os.listdir(mine), key=lambda f: (int(f.split('-')[-1].split('.')[0]) if '-' in f and f.split('-')[-1].split('.')[0].isdigit() else -1, f))
for f in files:
    if names and not any(f.startswith(n) for n in names): continue
    rp = os.path.join(ref, f)
    if not os.path.exists(rp): print(f"{f:40s} (no ref)"); continue
    a = np.fromfile(rp, dtype=np.float32); b = np.fromfile(os.path.join(mine, f), dtype=np.float32)
    if a.size != b.size: print(f"{f:40s} size mismatch {a.size} {b.size}"); continue
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    err = np.abs(a - b).max(); rel = err / (np.abs(a).max() + 1e-30)
    print(f"{f:40s} cos={cos:.6f} maxabs={err:.4g} rel={rel:.4g} ref|max|={np.abs(a).max():.4g} argmax {a.argmax()} {b.argmax()}")

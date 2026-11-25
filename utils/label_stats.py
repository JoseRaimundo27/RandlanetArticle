"""
Mostra, por arquivo PLY: dtype e min/max de X,Y,Z (precisao total) e a
contagem de valores dos campos de rotulo (label, class, orig_label).

Uso (PowerShell, em D:\ArtigoRandLANET):
    python label_stats.py bridge-pointclouds getec > label_stats.txt
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from plyfile import PlyData

LABEL_FIELDS = ("label", "class", "orig_label")
STEP = 5_000_000
MAX_SHOW = 40


def axis_stats(v):
    out = {}
    for ax in ("x", "y", "z"):
        mn, mx = np.inf, -np.inf
        for i in range(0, v.count, STEP):
            a = np.asarray(v[ax][i:i + STEP])
            mn = min(mn, float(a.min()))
            mx = max(mx, float(a.max()))
        out[ax] = (str(v[ax].dtype), mn, mx)
    return out


def count_field(v, name):
    c = Counter()
    kind = v[name].dtype.kind
    for i in range(0, v.count, STEP):
        a = np.asarray(v[name][i:i + STEP])
        vals, cnt = np.unique(a, return_counts=True)
        if kind == "f" and len(vals) > 100:
            return None  # campo continuo, nao e rotulo
        for x, n in zip(vals.tolist(), cnt.tolist()):
            c[x] += n
        if len(c) > 1000:
            return None
    return c


def main(folders):
    files = []
    for f in folders:
        files += sorted(Path(f).rglob("*.ply"))
    for p in files:
        print(f"\n=== {p}")
        try:
            ply = PlyData.read(str(p), mmap="r")
            v = ply["vertex"]
            names = [q.name for q in v.properties]
            print(f"pontos: {v.count:,}")
            for ax, (dt, mn, mx) in axis_stats(v).items():
                print(f"  {ax} ({dt}): min={mn!r}  max={mx!r}  ext={mx - mn!r}")
            for name in LABEL_FIELDS:
                if name not in names:
                    continue
                c = count_field(v, name)
                if c is None:
                    print(f"  {name}: campo continuo / muitos valores distintos")
                    continue
                tot = sum(c.values())
                items = sorted(c.items())[:MAX_SHOW]
                txt = ", ".join(f"{k}: {n:,} ({100 * n / tot:.1f}%)" for k, n in items)
                print(f"  {name} [{v[name].dtype}]: {txt}")
            if "d19" in p.name:
                print("  primeiras 3 linhas (x,y,z):")
                for i in range(3):
                    print("   ", float(v['x'][i]), float(v['y'][i]), float(v['z'][i]))
        except Exception as e:
            print(f"  ERRO: {e}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["."])
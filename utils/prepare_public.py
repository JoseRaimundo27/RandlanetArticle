"""
Prepara o dataset publico de pontes para o RandLA-Net (ponte vs ambiente).

- Le o campo `label` (-1 = sem rotulo/ignorar, 0 = ambiente, 1 = ponte).
- Aplica voxel grid (media de XYZ/RGB, voto majoritario no rotulo).
- Salva um .npz por cena em <saida>/ e um resumo em <saida>/resumo.csv.
- Gera <saida>/splits.json dividindo POR PONTE (blk e rtc da mesma ponte
  ficam sempre no mesmo split, evitando vazamento), estratificado por sensor.

Uso (PowerShell, em D:\\ArtigoRandLANET):
    pip install plyfile numpy
    python prepare_public.py bridge-pointclouds processed\\public --voxel 0.10
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from plyfile import PlyData

NAME_RE = re.compile(r"bridge_(\d+)_(cb|fr)_(\w+)\.ply$", re.IGNORECASE)


def voxelize(xyz, rgb, label, voxel):
    """Voxel grid: media de xyz/rgb, voto majoritario do rotulo (ignora -1)."""
    mn = xyz.min(axis=0)
    idx = np.floor((xyz - mn) / voxel).astype(np.int64)
    dims = idx.max(axis=0) + 1
    key = idx[:, 0] + dims[0] * (idx[:, 1] + dims[1] * idx[:, 2])
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    nv = len(cnt)

    out_xyz = np.stack([np.bincount(inv, weights=xyz[:, k], minlength=nv) for k in range(3)], 1) / cnt[:, None]
    out_rgb = np.stack([np.bincount(inv, weights=rgb[:, k].astype(np.float64), minlength=nv) for k in range(3)], 1) / cnt[:, None]

    c0 = np.bincount(inv[label == 0], minlength=nv)
    c1 = np.bincount(inv[label == 1], minlength=nv)
    out_lab = np.where(c1 > c0, 1, np.where(c0 > 0, 0, -1)).astype(np.int8)
    return (out_xyz.astype(np.float32), np.rint(out_rgb).astype(np.uint8), out_lab)


def make_splits(meta, n_test, n_val, seed):
    """Divide por numero de ponte, estratificado por sensor (cb / fr)."""
    rng = np.random.default_rng(seed)
    split_of = {}
    for sensor in sorted({m["sensor"] for m in meta}):
        ids = sorted({m["bridge_id"] for m in meta if m["sensor"] == sensor})
        rng.shuffle(ids)
        for i, b in enumerate(ids):
            split_of[b] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    splits = {"train": [], "val": [], "test": []}
    for m in meta:
        splits[split_of[m["bridge_id"]]].append(m["npz"])
    return splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entrada")
    ap.add_argument("saida")
    ap.add_argument("--voxel", type=float, default=0.10, help="tamanho do voxel em metros")
    ap.add_argument("--n-test", type=int, default=2, help="pontes de teste por sensor")
    ap.add_argument("--n-val", type=int, default=2, help="pontes de validacao por sensor")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.saida)
    out.mkdir(parents=True, exist_ok=True)
    meta = []
    for p in sorted(Path(a.entrada).glob("*.ply")):
        m = NAME_RE.search(p.name)
        if not m:
            print(f"[pulando] nome fora do padrao: {p.name}")
            continue
        v = PlyData.read(str(p))["vertex"].data
        xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
        rgb = np.column_stack([v["red"], v["green"], v["blue"]])
        label = np.asarray(v["label"]).astype(np.int64)
        bad = set(np.unique(label).tolist()) - {-1, 0, 1}
        assert not bad, f"{p.name}: valores inesperados em label: {bad}"

        vx, vr, vl = voxelize(xyz, rgb, label, a.voxel)
        name = p.stem + ".npz"
        np.savez_compressed(out / name, xyz=vx, rgb=vr, label=vl)
        valid = vl >= 0
        meta.append({"arquivo": p.name, "npz": name, "bridge_id": int(m.group(1)),
                     "sensor": m.group(2).lower(), "variante": m.group(3),
                     "n_original": len(xyz), "n_voxel": len(vx),
                     "pct_ponte": round(100 * float((vl[valid] == 1).mean()), 2),
                     "pct_ignorado": round(100 * float((~valid).mean()), 2)})
        r = meta[-1]
        print(f"{p.name}: {r['n_original']:,} -> {r['n_voxel']:,} pts | ponte {r['pct_ponte']}% | ignorado {r['pct_ignorado']}%")

    with open(out / "resumo.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(meta[0]))
        w.writeheader()
        w.writerows(meta)

    splits = make_splits(meta, a.n_test, a.n_val, a.seed)
    with open(out / "splits.json", "w", encoding="utf-8") as fh:
        json.dump(splits, fh, indent=2)
    print("\nSplits:", {k: len(v) for k, v in splits.items()}, "cenas")


if __name__ == "__main__":
    main()
"""
Inspeciona nuvens de pontos PLY (inclusive arquivos grandes, via memory-map).

Uso (PowerShell, na pasta D:\ArtigoRandLANET):
    pip install plyfile numpy scipy
    python inspect_ply.py bridge-pointclouds getec

Gera inspecao.csv e imprime um resumo por arquivo.
"""
import sys
import csv
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

SAMPLE = 200_000  # pontos usados para estimar densidade


def inspect(path: Path) -> dict:
    ply = PlyData.read(str(path), mmap="r")  # nao carrega tudo na RAM
    v = ply["vertex"]
    n = v.count
    props = [p.name for p in v.properties]
    row = {"arquivo": path.name, "pasta": path.parent.name, "n_pontos": n,
           "formato": "ascii" if ply.text else ("binary_" + ply.byte_order),
           "propriedades": ";".join(props),
           "tem_rgb": all(c in props for c in ("red", "green", "blue")),
           "tem_normais": all(c in props for c in ("nx", "ny", "nz")),
           "tem_intensidade": any(c in props for c in ("intensity", "scalar_intensity", "scalar_Intensity")),
           "tem_label": any(("label" in p.lower() or "class" in p.lower()) for p in props)}

    # bounding box em passes (evita copiar 3.6 GB para a RAM)
    mn = np.full(3, np.inf)
    mx = np.full(3, -np.inf)
    step = 5_000_000
    for i in range(0, n, step):
        xyz = np.column_stack([v["x"][i:i + step], v["y"][i:i + step], v["z"][i:i + step]]).astype(np.float64)
        mn = np.minimum(mn, xyz.min(0))
        mx = np.maximum(mx, xyz.max(0))
    ext = mx - mn
    row.update({"x_min": mn[0], "y_min": mn[1], "z_min": mn[2],
                "ext_x": ext[0], "ext_y": ext[1], "ext_z": ext[2]})

    # densidade: distancia ao vizinho mais proximo numa amostra
    # (amostra aleatoria de uma nuvem esparsa subestima densidade real, entao
    # usamos um bloco contiguo de pontos como proxy quando o arquivo e grande)
    k = min(SAMPLE, n)
    start = 0 if n <= k else (n // 2) - k // 2
    blk = np.column_stack([v["x"][start:start + k], v["y"][start:start + k], v["z"][start:start + k]]).astype(np.float64)
    d, _ = cKDTree(blk).query(blk, k=2)
    row["dist_nn_mediana_m"] = float(np.median(d[:, 1]))

    if row["tem_rgb"]:
        c = np.column_stack([v["red"][start:start + k], v["green"][start:start + k], v["blue"][start:start + k]])
        row["rgb_max"] = int(c.max())
        row["rgb_media"] = float(c.mean())
    return row


def main(folders):
    files = []
    for f in folders:
        files += sorted(Path(f).rglob("*.ply"))
    if not files:
        print("Nenhum .ply encontrado.")
        return
    rows = []
    for p in files:
        print(f"-> {p} ...", flush=True)
        try:
            r = inspect(p)
            rows.append(r)
            print(f"   {r['n_pontos']:,} pts | extensao (m): "
                  f"{r['ext_x']:.1f} x {r['ext_y']:.1f} x {r['ext_z']:.1f} | "
                  f"NN mediana: {r['dist_nn_mediana_m']*100:.2f} cm | "
                  f"rgb={r['tem_rgb']} int={r['tem_intensidade']} normais={r['tem_normais']}")
            print(f"   campos: {r['propriedades']}")
        except Exception as e:
            print(f"   ERRO: {e}")
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 999)
    with open("inspecao.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("\nSalvo em inspecao.csv")


if __name__ == "__main__":
    main(sys.argv[1:] or ["."])
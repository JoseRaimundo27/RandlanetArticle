"""Converte uma nuvem PLY externa em .npz (voxel + rotulo opcional) para o RandLA-Net.

Sem rotulo (so predizer):
    python prepare_external.py getec/d19.ply processed/getec/d19.npz --voxel 0.10

Com rotulo (para medir IoU), usando o PLY so com a ponte recortada do CloudCompare:
    python prepare_external.py getec/d19.ply processed/getec/d19.npz --voxel 0.10 --bridge getec/segmented_d19.ply

Funciona com arquivos grandes: le o PLY em blocos (memory-map). O voxel deve ser o
MESMO usado no treino (0.10). O rotulo so e valido se o segmented_*.ply estiver nas
MESMAS coordenadas da nuvem completa (o script confere isso e avisa).
"""
import argparse
import time
from pathlib import Path

import numpy as np
from plyfile import PlyData

STEP = 4_000_000


def chunks(path):
    data = PlyData.read(str(path), mmap="r")["vertex"].data
    for i in range(0, len(data), STEP):
        yield data[i:i + STEP]


def xyz_of(c):
    return np.column_stack([c["x"], c["y"], c["z"]]).astype(np.float64)


def rgb_of(c, scale16):
    if not all(k in c.dtype.names for k in ("red", "green", "blue")):
        return np.zeros((len(c), 3))
    rgb = np.column_stack([c["red"], c["green"], c["blue"]]).astype(np.float64)
    return rgb / 256.0 if scale16 else rgb


def keys_of(xyz, mn, voxel, dims):
    idx = np.floor((xyz - mn) / voxel).astype(np.int64)
    ok = np.all((idx >= 0) & (idx < dims), axis=1)
    key = idx[:, 0] + dims[0] * (idx[:, 1] + dims[1] * idx[:, 2])
    return key, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entrada")
    ap.add_argument("saida")
    ap.add_argument("--voxel", type=float, default=0.10)
    ap.add_argument("--bridge", default=None, help="PLY so com a ponte (mesmas coordenadas)")
    a = ap.parse_args()
    t0 = time.time()

    # passo 1: limites e escala de cor
    mn, mx, rmax, n = np.full(3, np.inf), np.full(3, -np.inf), 0.0, 0
    for c in chunks(a.entrada):
        xyz = xyz_of(c)
        mn, mx, n = np.minimum(mn, xyz.min(0)), np.maximum(mx, xyz.max(0)), n + len(c)
        if "red" in c.dtype.names:
            rmax = max(rmax, float(np.max(c["red"])))
    scale16 = rmax > 255
    dims = np.floor((mx - mn) / a.voxel).astype(np.int64) + 1
    print(f"{n:,} pontos | extensao {np.round(mx - mn, 1)} m | grade {dims.tolist()}", flush=True)

    # passo 2: voxel grid em blocos (soma xyz, rgb e contagem por voxel)
    parts = []
    for c in chunks(a.entrada):
        xyz, rgb = xyz_of(c), rgb_of(c, scale16)
        key, _ = keys_of(xyz, mn, a.voxel, dims)
        uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        s = np.column_stack([np.bincount(inv, weights=xyz[:, k], minlength=len(uk)) for k in range(3)] +
                            [np.bincount(inv, weights=rgb[:, k], minlength=len(uk)) for k in range(3)])
        parts.append((uk, cnt, s))
        print(f"  bloco processado ({time.time() - t0:.0f}s)", flush=True)
    keys = np.concatenate([p[0] for p in parts])
    cnts = np.concatenate([p[1] for p in parts])
    sums = np.concatenate([p[2] for p in parts])
    del parts
    vk, inv = np.unique(keys, return_inverse=True)
    count = np.bincount(inv, weights=cnts, minlength=len(vk))
    tot = np.column_stack([np.bincount(inv, weights=sums[:, k], minlength=len(vk)) for k in range(6)])
    xyz_v = (tot[:, :3] / count[:, None]).astype(np.float32)
    rgb_v = np.clip(np.rint(tot[:, 3:] / count[:, None]), 0, 255).astype(np.uint8)
    del keys, cnts, sums, inv

    label = -np.ones(len(vk), np.int8)
    if a.bridge:
        bk, bc, total_b, inside = [], [], 0, 0
        for c in chunks(a.bridge):
            key, ok = keys_of(xyz_of(c), mn, a.voxel, dims)
            total_b += len(key)
            inside += int(ok.sum())
            u, cn = np.unique(key[ok], return_counts=True)
            bk.append(u); bc.append(cn)
        bk, bc = np.concatenate(bk), np.concatenate(bc)
        ub, inv_b = np.unique(bk, return_inverse=True)
        bcount = np.bincount(inv_b, weights=bc)
        pos = np.searchsorted(vk, ub)
        pos[pos >= len(vk)] = len(vk) - 1
        exists = vk[pos] == ub
        frac = float(bcount[exists].sum() / max(total_b, 1))
        print(f"\nponte: {total_b:,} pontos | {100 * inside / max(total_b, 1):.1f}% dentro da caixa da nuvem completa | "
              f"{100 * frac:.1f}% caem em voxels que existem na nuvem completa")
        if frac < 0.9:
            print("ATENCAO: menos de 90% dos pontos da ponte coincidem com a nuvem completa. "
                  "As coordenadas provavelmente NAO sao as mesmas (shift/referencial diferente); "
                  "o rotulo gerado nao e confiavel.")
        bridge_in_voxel = np.zeros(len(vk))
        bridge_in_voxel[pos[exists]] = bcount[exists]
        label = (bridge_in_voxel * 2 > count).astype(np.int8)
        print(f"voxels de ponte: {100 * label.mean():.2f}% do total")

    Path(a.saida).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.saida, xyz=xyz_v, rgb=rgb_v, label=label)
    print(f"\n{n:,} pontos -> {len(vk):,} voxels | salvo em {a.saida} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
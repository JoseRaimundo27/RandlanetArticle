"""Avalia um checkpoint em um split (ou em .npz avulsos) e gera metricas e PLYs.

    python evaluate.py --ckpt runs/exp1/best.pt --data processed/public --split test --out results/exp1 --save-ply
    python evaluate.py --ckpt runs/exp1/best.pt --scenes processed/getec/d19.npz --out results/getec
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from common import load_scene, predict_scene, confusion, metrics_from_conf
from randlanet import RandLANet


def write_ply(path, sc, pred):
    d = np.zeros(len(pred), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"),
                                   ("green", "u1"), ("blue", "u1"), ("label", "i1"), ("pred", "i1")])
    d["x"], d["y"], d["z"] = sc["xyz"].T
    d["red"], d["green"], d["blue"] = sc["rgb"].T
    d["label"], d["pred"] = sc["label"], pred
    PlyData([PlyElement.describe(d, "vertex")]).write(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="processed/public")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--scenes", nargs="*", help="npz avulsos (ignora --split)")
    ap.add_argument("--out", default="results/exp1")
    ap.add_argument("--save-ply", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=dev)
    cfg = ck["cfg"]
    model = RandLANet(d_in=cfg["d_in"]).to(dev)
    model.load_state_dict(ck["model"])
    paths = [Path(p) for p in a.scenes] if a.scenes else \
        [Path(a.data) / f for f in json.load(open(Path(a.data) / "splits.json"))[a.split]]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, total = [], np.zeros((2, 2), np.int64)
    for p in paths:
        sc = load_scene(p)
        pred = predict_scene(model, sc, cfg, dev, num_workers=a.workers).argmax(1).astype(np.int8)
        cm = confusion(pred, sc["label"])
        total += cm
        row = {"cena": p.name, "n_pontos": len(pred)}
        if cm.sum() > 0:
            row.update({k: round(float(v), 4) for k, v in metrics_from_conf(cm).items()})
        rows.append(row)
        print(row, flush=True)
        if a.save_ply:
            write_ply(out / (p.stem + "_pred.ply"), sc, pred)

    summary = {"checkpoint": a.ckpt, "epoca": ck["epoch"] + 1, "n_cenas": len(paths)}
    if total.sum() > 0:
        summary["agregado"] = {k: round(float(v), 4) for k, v in metrics_from_conf(total).items()}
        summary["media_por_cena_iou_ponte"] = round(float(np.mean([r["iou_ponte"] for r in rows])), 4)
        summary["desvio_por_cena_iou_ponte"] = round(float(np.std([r["iou_ponte"] for r in rows])), 4)
    json.dump({"resumo": summary, "cenas": rows}, open(out / "metricas.json", "w"), indent=2, ensure_ascii=False)
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(out / "metricas_por_cena.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print("\nRESUMO", json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
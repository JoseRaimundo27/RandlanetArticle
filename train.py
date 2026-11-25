"""Treino do RandLA-Net (ponte vs ambiente).

    python train.py --data processed/public --out runs/exp1
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (CropDataset, load_scene, predict_scene, confusion,
                    metrics_from_conf, to_device)
from randlanet import RandLANet


def validate(model, val_scenes, cfg, device, workers):
    cm = np.zeros((2, 2), np.int64)
    for _, sc in val_scenes:
        pred = predict_scene(model, sc, cfg, device, num_workers=workers).argmax(1)
        cm += confusion(pred, sc["label"])
    return metrics_from_conf(cm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="processed/public")
    ap.add_argument("--out", default="runs/exp1")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--steps-per-epoch", type=int, default=250, help="iteracoes por epoca")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-points", type=int, default=40960)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--lr-decay", type=float, default=0.95, help="fator por epoca")
    ap.add_argument("--no-rgb", action="store_true", help="treina so com XYZ")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out, data = Path(a.out), Path(a.data)
    out.mkdir(parents=True, exist_ok=True)
    splits = json.load(open(data / "splits.json"))
    cfg = {"num_points": a.num_points, "use_rgb": not a.no_rgb}
    cfg["d_in"] = 6 if cfg["use_rgb"] else 3

    print(f"device={dev} | cena treino/val: {len(splits['train'])}/{len(splits['val'])} | cfg={cfg}")
    ds = CropDataset([data / f for f in splits["train"]], cfg, a.steps_per_epoch * a.batch_size)
    val_scenes = [(f, load_scene(data / f)) for f in splits["val"]]
    freq = ds.class_counts / ds.class_counts.sum()
    w = 1.0 / np.sqrt(freq)
    w = torch.tensor(w / w.sum() * 2, dtype=torch.float32, device=dev)
    print(f"freq ambiente/ponte: {freq.round(4).tolist()} | pesos da perda: {w.cpu().numpy().round(3).tolist()}")

    model = RandLANet(d_in=cfg["d_in"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, a.lr_decay)
    try:                                    # torch >= 2.3
        scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    except AttributeError:                  # torch mais antigo
        scaler = torch.cuda.amp.GradScaler(enabled=dev.type == "cuda")
    start, best = 0, -1.0
    if a.resume:
        ck = torch.load(a.resume, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); start, best = ck["epoch"] + 1, ck["best"]
        print(f"retomando da epoca {start}")

    dl = DataLoader(ds, batch_size=a.batch_size, num_workers=a.workers, drop_last=True,
                    persistent_workers=a.workers > 0, pin_memory=dev.type == "cuda")
    log = open(out / "log.jsonl", "a")
    for ep in range(start, a.epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for batch in dl:
            batch = to_device(batch, dev)
            label = batch.pop("label")
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                logits = model(batch)
            loss = F.cross_entropy(logits.reshape(-1, 2).float(), label.reshape(-1),
                                   weight=w, ignore_index=-1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt); scaler.update()
            tot += loss.item(); nb += 1
        sched.step()
        rec = {"epoca": ep + 1, "loss": tot / nb, "lr": opt.param_groups[0]["lr"],
               "tempo_s": round(time.time() - t0, 1)}
        if (ep + 1) % a.val_every == 0 or ep + 1 == a.epochs:
            m = validate(model, val_scenes, cfg, dev, min(a.workers, 4))
            rec.update({f"val_{k}": round(float(v), 4) for k, v in m.items()})
            if m["iou_ponte"] > best:
                best = float(m["iou_ponte"])
                torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": ep, "best": best},
                           out / "best.pt")
                rec["melhor"] = True
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "cfg": cfg, "epoch": ep, "best": best}, out / "last.pt")
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        log.write(json.dumps(rec, ensure_ascii=False) + "\n"); log.flush()


if __name__ == "__main__":
    main()
"""Dados, inferencia por cena e metricas (ponte = 1, ambiente = 0, -1 = ignorar)."""
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset, DataLoader

NUM_LAYERS, K_NEIGHBORS, SUB_RATIO = 4, 16, 4


def load_scene(path):
    z = np.load(path)
    n = len(z["xyz"])
    rgb = z["rgb"] if "rgb" in z.files else np.zeros((n, 3), np.uint8)
    label = z["label"].astype(np.int64) if "label" in z.files else -np.ones(n, np.int64)
    return {"xyz": z["xyz"].astype(np.float32), "rgb": rgb, "label": label}


def build_inputs(xyz, feat):
    """xyz (N,3) ja embaralhado. Calcula vizinhanca, sub-amostragem e interpolacao por camada."""
    xyz_l, neigh, sub, interp = [], [], [], []
    pts = xyz
    tree = cKDTree(pts)
    for _ in range(NUM_LAYERS):
        n_sub = len(pts) // SUB_RATIO
        nb = tree.query(pts, k=K_NEIGHBORS, workers=1)[1]
        sub_pts = pts[:n_sub]
        sub_tree = cKDTree(sub_pts)
        it = sub_tree.query(pts, k=1, workers=1)[1]
        xyz_l.append(torch.from_numpy(np.ascontiguousarray(pts)))
        neigh.append(torch.from_numpy(nb.astype(np.int64)))
        sub.append(torch.from_numpy(nb[:n_sub].astype(np.int64)))
        interp.append(torch.from_numpy(it[:, None].astype(np.int64)))
        pts, tree = sub_pts, sub_tree
    return {"feat": torch.from_numpy(feat.astype(np.float32)),
            "xyz": xyz_l, "neigh": neigh, "sub": sub, "interp": interp}


def crop_indices(tree, center, num_points, n, rng):
    if n >= num_points:
        idx = tree.query(center, k=num_points, workers=1)[1]
    else:
        idx = np.concatenate([np.arange(n), rng.choice(n, num_points - n)])
    idx = np.asarray(idx)
    rng.shuffle(idx)
    return idx


def make_sample(xyz, rgb, center, cfg, rng=None, augment=False):
    x = xyz.astype(np.float32).copy()
    x[:, :2] -= center[:2]
    c = rgb.astype(np.float32) / 255.0
    if augment:
        th = rng.uniform(0, 2 * np.pi)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], np.float32)
        x[:, :2] = x[:, :2] @ R.T
        x *= rng.uniform(0.9, 1.1)
        x += rng.normal(0, 0.01, x.shape).astype(np.float32)
        c = np.clip(c * rng.uniform(0.8, 1.2) + rng.normal(0, 0.02, (1, 3)), 0, 1).astype(np.float32)
    x[:, 2] -= x[:, 2].min()
    feat = np.concatenate([x, c], 1) if cfg["use_rgb"] else x
    return build_inputs(x, feat)


class CropDataset(Dataset):
    """Recortes aleatorios de num_points pontos; metade centrada em pontos de ponte."""

    def __init__(self, paths, cfg, length, p_bridge=0.5):
        self.cfg, self.length, self.p = cfg, length, p_bridge
        self.scenes = [load_scene(p) for p in paths]
        self.trees = [cKDTree(s["xyz"]) for s in self.scenes]
        self.bridge = [np.flatnonzero(s["label"] == 1) for s in self.scenes]
        lab = np.concatenate([s["label"] for s in self.scenes])
        self.class_counts = np.array([(lab == 0).sum(), (lab == 1).sum()], dtype=np.float64)

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        rng = np.random.default_rng()
        s = int(rng.integers(len(self.scenes)))
        sc, n = self.scenes[s], len(self.scenes[s]["xyz"])
        if rng.random() < self.p and len(self.bridge[s]) > 0:
            c = int(rng.choice(self.bridge[s]))
        else:
            c = int(rng.integers(n))
        idx = crop_indices(self.trees[s], sc["xyz"][c], self.cfg["num_points"], n, rng)
        out = make_sample(sc["xyz"][idx], sc["rgb"][idx], sc["xyz"][c], self.cfg, rng, augment=True)
        out["label"] = torch.from_numpy(sc["label"][idx])
        return out


class _CropList(Dataset):
    def __init__(self, scene, crops, cfg):
        self.sc, self.crops, self.cfg = scene, crops, cfg

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, i):
        idx, center, w = self.crops[i]
        out = make_sample(self.sc["xyz"][idx], self.sc["rgb"][idx], center, self.cfg)
        out["idx"] = torch.from_numpy(idx.astype(np.int64))
        out["w"] = torch.from_numpy(w.astype(np.float32))
        return out


def to_device(b, dev):
    if isinstance(b, dict):
        return {k: to_device(v, dev) for k, v in b.items()}
    if isinstance(b, list):
        return [to_device(v, dev) for v in b]
    return b.to(dev, non_blocking=True)


@torch.no_grad()
def predict_scene(model, scene, cfg, device, batch_size=4, num_workers=4, seed=0):
    """Cobre a cena com recortes sobrepostos; devolve probabilidades (N,2)."""
    model.eval()
    xyz, n, npts = scene["xyz"], len(scene["xyz"]), cfg["num_points"]
    tree, rng = cKDTree(xyz), np.random.default_rng(seed)
    covered = np.zeros(n, bool)
    crops = []
    while not covered.all():
        c = int(rng.choice(np.flatnonzero(~covered)))
        idx = crop_indices(tree, xyz[c], npts, n, rng)
        d = np.linalg.norm(xyz[idx] - xyz[c], axis=1)
        dmax = max(float(d.max()), 1e-6)
        crops.append((idx, xyz[c], np.maximum(1e-3, 1 - d / (dmax * 1.0001))))
        covered[idx[d <= 0.7 * dmax]] = True
        if n <= npts:
            break
    dl = DataLoader(_CropList(scene, crops, cfg), batch_size=batch_size, num_workers=num_workers)
    probs = np.zeros((n, 2), np.float32)
    for batch in dl:
        batch = to_device(batch, device)
        idx, w = batch.pop("idx"), batch.pop("w")
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            p = torch.softmax(model(batch).float(), -1)
        p, idx, w = p.cpu().numpy(), idx.cpu().numpy(), w.cpu().numpy()
        for b in range(len(p)):
            np.add.at(probs, idx[b], w[b][:, None] * p[b])
    return probs / np.maximum(probs.sum(1, keepdims=True), 1e-9)


def confusion(pred, label, nc=2):
    m = label >= 0
    return np.bincount(label[m] * nc + pred[m], minlength=nc * nc).reshape(nc, nc)


def metrics_from_conf(cm):
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / np.maximum(tp + fn, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return {"iou_ambiente": iou[0], "iou_ponte": iou[1], "miou": iou.mean(),
            "precisao_ponte": prec[1], "recall_ponte": rec[1], "f1_ponte": f1[1],
            "acc_global": tp.sum() / max(cm.sum(), 1)}
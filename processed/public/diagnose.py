python - <<'PY'
import numpy as np
from pathlib import Path

for f in sorted(Path(".").glob("*.npz")):
    d = np.load(f)

    xyz = d["xyz"]
    rgb = d["rgb"]
    labels = d["label"]

    unique, counts = np.unique(labels, return_counts=True)

    print(f"\n{f.name}")
    print("XYZ:", xyz.shape)
    print("RGB:", rgb.shape)
    print("Labels:", dict(zip(unique.tolist(), counts.tolist())))
    print("Proporção ponte:",
          round(100 * np.mean(labels == 1), 2), "%")
PY
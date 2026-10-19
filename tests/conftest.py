import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signa.config import Config  # noqa: E402


def make_hand(rng: np.random.Generator, center=(320.0, 240.0), palm=80.0) -> np.ndarray:
    """Mano sintética (21, 3) en píxeles con palma de tamaño `palm`."""
    pts = rng.normal(0, 0.5, (21, 3)).astype(np.float32)
    pts[0] = 0
    pts[9] = [0, -1, 0]  # nudillo medio una palma arriba de la muñeca
    pts *= palm
    pts[:, :2] += center
    return pts


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def cfg():
    return Config()


def synth_dataset(path: str, labels, signers=("ana", "beto", "caro"), per_label=60, seed=1,
                  with_pose=True) -> None:
    """Escribe un CSV con el esquema real del dataset (lo que produciría collect)."""
    from signa.dataset import ANCHOR_COLS, COLUMNS, SCHEMA_VERSION
    import csv
    rng = np.random.default_rng(seed)
    protos = {l: rng.normal(0, 0.08, (21, 3)) for l in labels}
    for p in protos.values():
        p[0] = 0
    os.makedirs(path, exist_ok=True)
    for k, s in enumerate(signers):
        rows = []
        size = 0.9 + 0.1 * k
        for l in labels:
            for i in range(per_label):
                pts = (protos[l] + rng.normal(0, 0.006, (21, 3))) * size
                pts[:, :2] += [0.5 + rng.uniform(-.1, .1), 0.45 + rng.uniform(-.1, .1)]
                hand = "Right"
                if s == "caro" and i % 2:
                    pts[:, 0] = 1 - pts[:, 0]
                    hand = "Left"
                anchors = [0.5, 0.25, 0.62, 0.5, 0.38, 0.5] if with_pose else [""] * len(ANCHOR_COLS)
                rows.append([SCHEMA_VERSION, s, f"{s}_s1", l, hand, 0.95, 640, 640, i / 6]
                            + [f"{v:.5f}" for v in pts.ravel()] + anchors)
        with open(os.path.join(path, f"{s}_s1.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLUMNS)
            w.writerows(rows)

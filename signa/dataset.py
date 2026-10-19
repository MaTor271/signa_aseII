"""
Formato del dataset (RF-11, RNF-02). Un CSV por sesión: data/<señante>_<fecha-hora>.csv.
Solo se guardan coordenadas normalizadas, nunca imágenes. Se guarda el dato CRUDO y no las
características: si mañana cambia features.py, se reentrena sin volver a grabar a nadie.
"""
import csv
import glob
import os
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .features import N_LANDMARKS, POSE_ANCHORS, to_pixel_space

SCHEMA_VERSION = 2
META_COLS = ["schema", "signer", "session", "label", "handedness", "hand_score", "width", "height", "t"]
HAND_COLS = [f"{c}{i}" for i in range(N_LANDMARKS) for c in "xyz"]
ANCHOR_COLS = [f"{name}_{c}" for name in POSE_ANCHORS for c in "xy"]
COLUMNS = META_COLS + HAND_COLS + ANCHOR_COLS


class SessionWriter:
    """Escribe una sesión de captura y permite deshacer las últimas N muestras."""

    def __init__(self, data_dir: str, signer: str):
        os.makedirs(data_dir, exist_ok=True)
        self.signer = signer
        self.session = f"{signer}_{time.strftime('%Y%m%d-%H%M%S')}"
        self.path = os.path.join(data_dir, f"{self.session}.csv")
        self._f = open(self.path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow(COLUMNS)
        self._f.flush()
        self._rows: list = []  # (etiqueta, posición del archivo antes de la fila)

    def write(self, label: str, hand_norm: np.ndarray, handedness: str, hand_score: float,
              width: int, height: int, t: float, anchors_norm: Optional[np.ndarray] = None) -> None:
        anchors = np.full(len(ANCHOR_COLS), np.nan) if anchors_norm is None else np.ravel(anchors_norm)
        row = [SCHEMA_VERSION, self.signer, self.session, label, handedness, f"{hand_score:.3f}",
               width, height, f"{t:.3f}"]
        row += [f"{v:.5f}" for v in np.ravel(hand_norm)]
        row += ["" if np.isnan(v) else f"{v:.5f}" for v in anchors]
        self._rows.append((label, self._f.tell()))
        self._w.writerow(row)
        self._f.flush()

    def undo(self, n: int) -> Counter:
        k = min(n, len(self._rows))
        if k == 0:
            return Counter()
        removed = self._rows[-k:]
        del self._rows[-k:]
        self._f.seek(removed[0][1])
        self._f.truncate()
        self._f.flush()
        return Counter(label for label, _ in removed)

    def close(self) -> None:
        self._f.close()
        if not self._rows:
            os.remove(self.path)  # no dejar archivos vacíos


@dataclass
class Dataset:
    hand_px: np.ndarray      # (N, 21, 3)
    anchors_px: np.ndarray   # (N, 3, 2), NaN si no hubo Pose
    handedness: np.ndarray   # (N,)
    labels: np.ndarray       # (N,)
    signers: np.ndarray      # (N,)
    sessions: np.ndarray     # (N,)

    def __len__(self) -> int:
        return len(self.labels)

    def subset(self, mask: np.ndarray) -> "Dataset":
        return Dataset(*(getattr(self, f)[mask] for f in self.__dataclass_fields__))

    def counts(self) -> Dict[str, int]:
        return dict(sorted(Counter(self.labels.tolist()).items()))


def load_dataset(data_dir: str) -> Dataset:
    import pandas as pd

    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No hay CSV en {data_dir}/. Graben datos con: python -m signa collect")
    frames = []
    for path in files:
        df = pd.read_csv(path, keep_default_na=True)
        if "schema" not in df.columns:  # compatibilidad con archivos de la v1
            df["schema"] = 1
            df["session"] = os.path.splitext(os.path.basename(path))[0]
            df["hand_score"] = 1.0
            df["t"] = np.nan
        frames.append(df.reindex(columns=COLUMNS))
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=HAND_COLS + ["label", "width", "height"])
    w, h = df["width"].to_numpy(np.float32), df["height"].to_numpy(np.float32)
    hand = df[HAND_COLS].to_numpy(np.float32).reshape(-1, N_LANDMARKS, 3)
    anchors = df[ANCHOR_COLS].to_numpy(np.float32).reshape(-1, len(POSE_ANCHORS), 2)
    return Dataset(
        hand_px=to_pixel_space(hand, w, h),
        anchors_px=to_pixel_space(anchors, w, h),
        handedness=df["handedness"].astype(str).to_numpy(),
        labels=df["label"].astype(str).to_numpy(),
        signers=df["signer"].astype(str).to_numpy(),
        sessions=df["session"].astype(str).to_numpy(),
    )

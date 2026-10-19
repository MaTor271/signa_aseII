"""Prueba de integración de la app sin hardware: cámara, detector y ventana simulados."""
import json
import os

import numpy as np

from conftest import synth_dataset
from test_pipeline import LABELS, _cfg


def test_app_loop_without_hardware(tmp_path, monkeypatch):
    import cv2
    from signa import app, train, vision
    from signa.dataset import load_dataset

    cfg = _cfg(tmp_path)
    synth_dataset(cfg.data_dir, LABELS, per_label=40)
    train.run(cfg, epochs=40, n_aug=0, skip_eval=True)
    ds = load_dataset(cfg.data_dir)
    sample = {l: int(np.flatnonzero(ds.labels == l)[0]) for l in ("L", "O")}

    # Guion: L detenida, transición, O detenida, sin mano -> palabra "LO"
    script = [("L", 12), (None, 0), ("move", 4), ("O", 12), ("absent", 25)]
    frames = []
    for kind, n in script:
        frames += [kind] * n

    class FakeCamera:
        def __init__(self, *a, **k): self.i = 0
        def read(self, last_seq=0, timeout=2.0):
            self.i += 1
            return np.zeros((480, 640, 3), np.uint8), self.i
        def release(self): pass

    class FakeTracker:
        def __init__(self, cfg): self.i = 0
        def detect(self, rgb):
            kind = frames[min(self.i, len(frames) - 1)]; self.i += 1
            if kind == "absent":
                return None
            idx = sample["L"] if kind in ("L", "move") else sample["O"]
            hand = ds.hand_px[idx].copy()
            if kind == "move":  # la mano va y regresa (transición real, sin teletransporte)
                k = frames[:self.i].count("move")
                hand[:, 0] += (30, 60, 30, 0)[k - 1]
            return vision.Observation(None, hand, ds.handedness[idx], 0.95, None, None)
        def close(self): pass

    t = {"now": 0.0}
    monkeypatch.setattr(app.time, "monotonic", lambda: t.__setitem__("now", t["now"] + 0.05) or t["now"])
    monkeypatch.setattr(vision, "Camera", FakeCamera)
    monkeypatch.setattr(vision, "Tracker", FakeTracker)
    shown = []
    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "imshow", lambda w, img: shown.append(img))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cv2, "waitKey", lambda d: 27 if len(shown) >= len(frames) else 255)
    said = []
    monkeypatch.setattr(app.Speaker, "say", lambda self, text: said.append(text))

    app.run(cfg)
    assert said == ["LO"]
    rec = json.loads(open(os.path.join(cfg.reports_dir, "runtime.jsonl")).read().splitlines()[-1])
    assert rec["letters"] == 2 and rec["latency_ms_p95"] > 0

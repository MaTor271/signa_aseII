import json
import os

import numpy as np
import pytest

from conftest import make_hand
from signa.config import Config
from signa.dataset import COLUMNS, HAND_COLS, SessionWriter, load_dataset
from signa.ui import Renderer, ViewState
from signa.vision import pick_hand


# ---------- RF-02: mano dominante ----------
def test_pick_dominant_when_two_hands():
    assert pick_hand(["Left", "Right"], [0.99, 0.90], "Right") == 1
    assert pick_hand(["Left", "Right"], [0.90, 0.99], "Left") == 0


def test_single_hand_is_used_even_if_label_differs():
    assert pick_hand(["Left"], [0.9], "Right") == 0


# ---------- Config ----------
def test_config_overrides_and_validation(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"dominant_hand": "Left", "min_hold_s": 0.3}))
    cfg = Config.load(str(p))
    assert cfg.dominant_hand == "Left" and cfg.min_hold_s == 0.3
    p.write_text(json.dumps({"dominant": "Left"}))
    with pytest.raises(ValueError):
        Config.load(str(p))
    p.write_text(json.dumps({"move_speed_lo": 3.0, "move_speed_hi": 2.0}))
    with pytest.raises(ValueError):
        Config.load(str(p))


# ---------- RF-11 / RNF-02: dataset ----------
def _write(writer, label, rng, n=1):
    for _ in range(n):
        writer.write(label, rng.uniform(0, 1, (21, 3)), "Right", 0.95, 640, 480, 0.0)


def test_session_undo_and_reload(tmp_path, rng):
    w = SessionWriter(str(tmp_path), "ana")
    _write(w, "A", rng, 5); _write(w, "B", rng, 5)
    assert w.undo(3) == {"B": 3}
    _write(w, "C", rng, 2)
    w.close()
    ds = load_dataset(str(tmp_path))
    assert ds.counts() == {"A": 5, "B": 2, "C": 2}
    assert ds.hand_px.shape == (9, 21, 3) and np.isnan(ds.anchors_px).all()


def test_empty_session_leaves_no_file(tmp_path):
    w = SessionWriter(str(tmp_path), "ana"); w.close()
    assert os.listdir(tmp_path) == []


def test_dataset_stores_only_coordinates(tmp_path, rng):
    w = SessionWriter(str(tmp_path), "ana"); _write(w, "A", rng); w.close()
    header = open(w.path, encoding="utf-8").readline().strip().split(",")
    assert header == COLUMNS
    assert not any(k in c for c in header for k in ("img", "image", "pixel", "frame"))


def test_loads_v1_files(tmp_path, rng):
    cols = ["signer", "label", "handedness", "width", "height"] + HAND_COLS
    row = ["ana", "A", "Right", 640, 480] + list(rng.uniform(0, 1, 63))
    with open(tmp_path / "landmarks.csv", "w") as f:
        f.write(",".join(cols) + "\n" + ",".join(map(str, row)) + "\n")
    ds = load_dataset(str(tmp_path))
    assert len(ds) == 1 and ds.sessions[0] == "landmarks"


# ---------- RF-08/09, RNF-09: interfaz ----------
@pytest.mark.parametrize("round_display", [False, True])
def test_ui_renders_accents_and_feedback(rng, round_display):
    frame = np.zeros((480, 640, 3), np.uint8)
    st = ViewState(hand_px=make_hand(rng), phase="detencion", candidate="A", progress=0.6,
                   text="HOLA", heard="¿Cómo estás? Mañana nos vemos", speaking=True)
    out = Renderer(round_display=round_display).render(frame, st)
    assert out.shape[:2] == ((480, 480) if round_display else (480, 640))
    assert out.any()
    if round_display:
        assert not out[0, 0].any()  # esquinas fuera del círculo quedan negras

"""Pipeline completo con datos sintéticos: carga -> evaluación -> exportación -> golden -> inferencia.
Los datos sintéticos son separables a propósito: esto prueba que el sistema funciona, no la precisión."""
import json

import numpy as np
import pytest

from conftest import synth_dataset
from signa.config import REST, Config

LABELS = ["A", "B", "L", "O", REST]


def _cfg(tmp_path) -> Config:
    m = tmp_path / "models"
    return Config(data_dir=str(tmp_path / "data"), reports_dir=str(tmp_path / "reports"),
                  classifier_model=str(m / "signa_mlp.tflite"), meta_file=str(m / "meta.json"),
                  golden_file=str(m / "golden.json"))


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    from signa import train
    tmp = tmp_path_factory.mktemp("pipe")
    cfg = _cfg(tmp)
    synth_dataset(cfg.data_dir, LABELS)
    meta = train.run(cfg, use_location=False, epochs=60, n_aug=1)
    return cfg, meta


def test_meta_and_reports(trained):
    cfg, meta = trained
    assert meta["labels"] == sorted(LABELS)
    assert meta["metrics"]["scheme"] == "señante no visto"
    assert set(meta["metrics"]["per_fold"]) == {"ana", "beto", "caro"}
    assert meta["metrics"]["f1_macro"] > 0.9
    assert meta["metrics"]["classifier_ms"] < 5


def test_golden_reproduces(trained):
    """RNF-06: lo mismo que verificará `python -m signa check` en la Pi."""
    from signa.check import verify_golden
    ef, ep = verify_golden(trained[0])
    assert ef < 1e-4 and ep < 1e-3


def test_classifier_predicts_with_margin(trained):
    from signa.classifier import SignClassifier
    from signa.dataset import load_dataset
    from signa.features import extract_features
    cfg, _ = trained
    clf = SignClassifier(cfg.classifier_model, cfg.meta_file)
    ds = load_dataset(cfg.data_dir)
    i = int(np.flatnonzero(ds.labels == "L")[0])
    pred = clf.predict(extract_features(ds.hand_px[i], ds.handedness[i]))
    assert pred.label == "L" and 0 <= pred.margin <= pred.conf <= 1


def test_version_mismatch_is_rejected(trained, tmp_path):
    """RNF-07: un modelo de otra versión de características no se carga en silencio."""
    from signa.classifier import ModelMismatchError, SignClassifier
    cfg, meta = trained
    bad = tmp_path / "meta.json"
    bad.write_text(json.dumps({**meta, "feature_version": 1}))
    with pytest.raises(ModelMismatchError):
        SignClassifier(cfg.classifier_model, str(bad))


def test_location_model(tmp_path):
    from signa import train
    from signa.check import verify_golden
    cfg = _cfg(tmp_path)
    synth_dataset(cfg.data_dir, LABELS, per_label=30)
    meta = train.run(cfg, use_location=True, epochs=20, n_aug=0, skip_eval=True)
    assert meta["n_features"] == 77 and meta["use_location"] is True
    ef, ep = verify_golden(cfg)
    assert ef < 1e-4 and ep < 1e-3


def test_location_requires_pose_data(tmp_path):
    from signa import train
    cfg = _cfg(tmp_path)
    synth_dataset(cfg.data_dir, LABELS, per_label=10, with_pose=False)
    with pytest.raises(ValueError):
        train.run(cfg, use_location=True, epochs=1, skip_eval=True)

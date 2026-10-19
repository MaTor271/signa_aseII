"""
Diagnóstico del equipo (RF-01, RNF-06): `python -m signa check`.
Corre en la laptop y en la Pi. Si los vectores golden coinciden, el dispositivo calcula
exactamente lo que se entrenó; si se porta el código a otro lenguaje, debe pasar la misma prueba.
"""
import json
import logging
import os
import platform
from typing import Tuple

import numpy as np

from .config import Config
from .features import FEATURE_VERSION, extract_features

log = logging.getLogger("signa.check")


def verify_golden(cfg: Config) -> Tuple[float, float]:
    """Devuelve (error máximo en características, error máximo en probabilidades)."""
    from .classifier import SignClassifier
    with open(cfg.golden_file, encoding="utf-8") as f:
        golden = json.load(f)
    if golden["feature_version"] != FEATURE_VERSION:
        raise ValueError("golden.json es de otra versión de características")
    clf = SignClassifier(cfg.classifier_model, cfg.meta_file)
    err_f = err_p = 0.0
    for s in golden["samples"]:
        anchors = None if s["anchors_px"] is None else np.array(s["anchors_px"], np.float32)
        feats = extract_features(np.array(s["hand_px"], np.float32), s["handedness"], anchors,
                                 golden["use_location"])
        err_f = max(err_f, float(np.abs(feats - np.array(s["features"])).max()))
        err_p = max(err_p, float(np.abs(clf.probs(np.array(s["features"], np.float32))
                                        - np.array(s["probs"])).max()))
    return err_f, err_p


def run(cfg: Config, camera: bool = True) -> bool:
    ok = True
    log.info("Python %s | %s %s", platform.python_version(), platform.system(), platform.machine())
    for mod in ("mediapipe", "cv2", "numpy", "ai_edge_litert", "piper", "vosk", "sounddevice", "PIL"):
        try:
            m = __import__(mod)
            log.info("  [ok] %-15s %s", mod, getattr(m, "__version__", ""))
        except Exception as e:  # noqa: BLE001 - diagnóstico: reportar cualquier fallo
            log.info("  [--] %-15s %s", mod, type(e).__name__)
    files = {"detector de manos": cfg.hand_model, "clasificador": cfg.classifier_model,
             "metadata": cfg.meta_file, "golden": cfg.golden_file, "voz Piper": cfg.piper_voice,
             "Vosk": cfg.vosk_model}
    if cfg.use_pose:
        files["Pose"] = cfg.pose_model
    for name, path in files.items():
        exists = os.path.exists(path)
        log.info("  [%s] %-17s %s", "ok" if exists else "--", name, path)
        ok &= exists or name in ("voz Piper", "Vosk")
    if os.path.exists(cfg.golden_file) and os.path.exists(cfg.classifier_model):
        ef, ep = verify_golden(cfg)
        passed = ef < 1e-4 and ep < 1e-3
        ok &= passed
        log.info("  [%s] golden: error features %.2e, error probabilidades %.2e",
                 "ok" if passed else "XX", ef, ep)
    if camera:
        from .vision import Camera
        try:
            cam = Camera(cfg.camera_source, cfg.frame_width, cfg.frame_height)
            frame, _ = cam.read()
            cam.release()
            log.info("  [%s] cámara %s %s", "ok" if frame is not None else "XX", cfg.camera_source,
                     None if frame is None else frame.shape[1::-1])
            ok &= frame is not None
        except Exception as e:  # noqa: BLE001
            log.info("  [XX] cámara: %s", e)
            ok = False
    log.info("Resultado: %s", "LISTO" if ok else "REVISAR")
    return ok

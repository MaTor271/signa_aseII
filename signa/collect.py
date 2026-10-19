"""
Captura del dataset (RF-11). Solo guarda muestras con la mano DETENIDA y bien detectada:
los cuadros de transición entre letras son etiquetas falsas que confunden al modelo.

Controles: ESPACIO grabar/pausar · n/p siguiente/anterior seña · u deshacer 10 · ESC salir
"""
import logging
import time
from typing import Sequence

import cv2

from .config import REST, Config
from .dataset import SessionWriter
from .segmenter import MotionEstimator
from .ui import GRAY, GREEN, TextPainter, WHITE, AMBER, HAND_CONNECTIONS

log = logging.getLogger("signa.collect")


def run(cfg: Config, signer: str, labels: Sequence[str]) -> None:
    from .vision import Camera, Tracker
    cam = Camera(cfg.camera_source, cfg.frame_width, cfg.frame_height)
    tracker = Tracker(cfg)
    writer = SessionWriter(cfg.data_dir, signer)
    motion, painter = MotionEstimator(cfg.speed_tau_s), TextPainter(cfg.font_path)
    counts = {l: 0 for l in labels}
    idx, recording, last_save, seq = 0, False, 0.0, 0
    log.info("Sesión %s | mano dominante %s | Pose %s", writer.session, cfg.dominant_hand,
             "sí" if cfg.use_pose else "no")
    try:
        while True:
            rgb, seq = cam.read(seq)
            if rgb is None:
                log.error("La cámara dejó de entregar cuadros"); break
            t = time.monotonic()
            obs = tracker.detect(rgb)
            label = labels[idx]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            h, w = bgr.shape[:2]

            if obs is None:
                motion.reset()
                reason = "sin mano"
            else:
                speed = motion.update(t, obs.hand_px)
                if speed >= cfg.move_speed_lo:
                    reason = "mano en movimiento"
                elif obs.score < cfg.collect_min_hand_score:
                    reason = "detección débil: mejoren luz o fondo"
                else:
                    reason = ""
                color = GREEN if not reason else GRAY
                pts = obs.hand_px[:, :2].astype(int)
                for a, b in HAND_CONNECTIONS:
                    cv2.line(bgr, tuple(pts[a]), tuple(pts[b]), color, 2, cv2.LINE_AA)
                if (recording and not reason and t - last_save >= 1.0 / cfg.collect_hz
                        and counts[label] < cfg.collect_per_label):
                    writer.write(label, obs.hand_norm, obs.handedness, obs.score, w, h, t, obs.anchors_norm)
                    counts[label] += 1
                    last_save = t
            if counts[label] >= cfg.collect_per_label:
                recording = False

            name = "REPOSO (mano relajada)" if label == REST else label
            painter.add(f"Seña: {name}", (12, 10), 40, AMBER if recording else WHITE)
            painter.add(f"{counts[label]}/{cfg.collect_per_label}   "
                        f"{'GRABANDO' if recording else 'pausa: ESPACIO'}", (12, 58), 24, WHITE)
            if reason:
                painter.add(reason, (12, h - 40), 22, GRAY)
            cv2.imshow("SIGNA - captura", painter.flush(bgr))

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord(" "):
                recording = not recording
            elif key in (ord("n"), ord("p")):
                idx = (idx + (1 if key == ord("n") else -1)) % len(labels)
                recording = False
            elif key == ord("u"):
                for lab, n in writer.undo(10).items():
                    counts[lab] -= n
    finally:
        writer.close()
        tracker.close()
        cam.release()
        cv2.destroyAllWindows()
        log.info("Guardado en %s: %s", writer.path, {k: v for k, v in counts.items() if v})

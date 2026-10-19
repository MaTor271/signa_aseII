"""
Aplicación del dispositivo: seña -> texto + voz, y voz del oyente -> texto.
Controles: BACKSPACE borrar · c limpiar · h repetir frase en voz · d diagnóstico · ESC salir
(En la pantalla táctil, mapeen botones a las mismas acciones: Speller.backspace/clear, Speaker.say.)
"""
import json
import logging
import os
import time
from typing import Optional

import cv2
import numpy as np

from .classifier import SignClassifier
from .config import Config
from .features import extract_features
from .segmenter import Speller
from .tts import Speaker
from .ui import Renderer, ViewState

log = logging.getLogger("signa.app")


def run(cfg: Config, listen: bool = False, mic: Optional[int] = None, round_display: bool = False,
        fullscreen: bool = False) -> None:
    from .vision import Camera, Tracker
    clf = SignClassifier(cfg.classifier_model, cfg.meta_file)
    if clf.use_location and not cfg.use_pose:
        log.info("El modelo usa ubicación: se activa Pose")
        cfg.use_pose = True
    cam = Camera(cfg.camera_source, cfg.frame_width, cfg.frame_height)
    tracker = Tracker(cfg)
    speller, speaker = Speller(cfg), Speaker(cfg.piper_voice)
    renderer = Renderer(cfg.font_path, round_display)
    listener = None
    if listen:
        from .stt import VoiceListener
        listener = VoiceListener(cfg.vosk_model, mic, speaker.is_speaking).start()

    win = "SIGNA"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    proc_ms, stamps, n_letters, debug, seq = [], [], 0, False, 0
    try:
        while True:
            rgb, seq = cam.read(seq)
            if rgb is None:
                log.error("La cámara dejó de entregar cuadros"); break
            now = time.monotonic()
            t0 = time.perf_counter()
            obs = tracker.detect(rgb)
            pred = None
            if obs is not None:
                feats = extract_features(obs.hand_px, obs.handedness, obs.anchors_px, clf.use_location)
                pred = clf.predict(feats)
            proc_ms.append((time.perf_counter() - t0) * 1000)
            stamps.append(now)

            events = speller.update(now, None if obs is None else obs.hand_px,
                                    *(() if pred is None else (pred.label, pred.conf, pred.margin)))
            for ev in events:
                if ev.kind == "letter":
                    n_letters += 1
                else:
                    speaker.say(ev.value)

            recent = stamps[-30:]
            fps = (len(recent) - 1) / max(recent[-1] - recent[0], 1e-6) if len(recent) > 1 else 0.0
            view = ViewState(
                hand_px=None if obs is None else obs.hand_px, phase=speller.phase,
                candidate=speller.candidate, progress=speller.progress, text=speller.text,
                heard=listener.text if listener else "", speaking=speaker.is_speaking.is_set(),
                debug=[f"{fps:4.1f} FPS  {proc_ms[-1]:4.0f} ms",
                       f"vel {speller.motion.speed:4.2f} palmas/s  {speller.phase}",
                       "" if pred is None else f"{pred.label} p={pred.conf:.2f} m={pred.margin:.2f}"]
                if debug else [])
            cv2.imshow(win, renderer.render(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), view))

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == 8:
                speller.backspace()
            elif key == ord("c"):
                speller.clear()
            elif key == ord("h"):
                speaker.say(speller.text)
            elif key == ord("d"):
                debug = not debug
    finally:
        tracker.close()
        cam.release()
        cv2.destroyAllWindows()
        _save_runtime(cfg, proc_ms, stamps, n_letters, clf)


def _save_runtime(cfg: Config, proc_ms, stamps, n_letters: int, clf: SignClassifier) -> None:
    """RNF-04: evidencia de latencia y FPS para el Capítulo 4."""
    if len(proc_ms) < 20:
        return
    a = np.array(proc_ms[10:])  # descarta el arranque
    dur = stamps[-1] - stamps[10]
    rec = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "camera": cfg.camera_source,
           "pose": cfg.use_pose, "frames": len(a), "fps": round((len(a) - 1) / max(dur, 1e-6), 2),
           "latency_ms_mean": round(float(a.mean()), 2),
           "latency_ms_p95": round(float(np.percentile(a, 95)), 2),
           "letters": n_letters, "model_created": clf.meta.get("created")}
    os.makedirs(cfg.reports_dir, exist_ok=True)
    with open(os.path.join(cfg.reports_dir, "runtime.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("Sesión: %.1f FPS | latencia media %.1f ms, p95 %.1f ms | %d letras",
             rec["fps"], rec["latency_ms_mean"], rec["latency_ms_p95"], n_letters)

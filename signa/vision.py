"""
Cámara y percepción (RF-01, RF-02, O3).

Notas:
  - mediapipe >= 1.0 eliminó `mp.solutions.*`; se usa la Tasks API (HandLandmarker, PoseLandmarker).
  - El cuadro se refleja (modo espejo) ANTES de detectar: así la etiqueta Left/Right de MediaPipe
    corresponde a la mano real del señante, y la pantalla funciona como un espejo para él.
  - La cámara se lee en un hilo aparte: mientras se procesa un cuadro ya se está capturando el
    siguiente. En la Pi esto sube los FPS sin cambiar el modelo.
"""
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import Config
from .features import POSE_ANCHORS, to_pixel_space


def _require(path: str, hint: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Falta {path}. {hint}")
    return path


class Camera:
    """Webcam (OpenCV) o cámara CSI (Picamera2). Entrega RGB espejado del cuadro más reciente."""

    def __init__(self, source: str, width: int, height: int):
        self.source = str(source)
        if self.source == "picam":
            from picamera2 import Picamera2  # apt: python3-picamera2
            self._cam = Picamera2()
            self._cam.configure(self._cam.create_video_configuration(
                main={"size": (width, height), "format": "RGB888"}))
            self._cam.start()
        else:
            self._cam = cv2.VideoCapture(int(self.source))
            self._cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self._cam.isOpened():
                raise RuntimeError(f"No se pudo abrir la cámara {self.source}")
        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._cond = threading.Condition()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _grab(self) -> Optional[np.ndarray]:
        if self.source == "picam":
            # "RGB888" de Picamera2 entrega bytes en orden BGR (convención de libcamera)
            return cv2.cvtColor(self._cam.capture_array(), cv2.COLOR_BGR2RGB)
        ok, frame = self._cam.read()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None

    def _loop(self) -> None:
        while self._running:
            frame = self._grab()
            if frame is None:
                time.sleep(0.01)
                continue
            with self._cond:
                self._frame, self._seq = cv2.flip(frame, 1), self._seq + 1
                self._cond.notify_all()

    def read(self, last_seq: int = 0, timeout: float = 2.0) -> Tuple[Optional[np.ndarray], int]:
        """Espera un cuadro más nuevo que `last_seq`. Nunca procesa dos veces el mismo cuadro."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > last_seq, timeout):
                return None, last_seq
            return self._frame, self._seq

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        if self.source == "picam":
            self._cam.stop()
        else:
            self._cam.release()


@dataclass
class Observation:
    hand_norm: np.ndarray                 # (21, 3) normalizado (lo que se guarda en el dataset)
    hand_px: np.ndarray                   # (21, 3) píxeles (lo que usan features y segmentador)
    handedness: str                       # "Left" | "Right"
    score: float
    anchors_norm: Optional[np.ndarray]    # (3, 2) nariz y hombros, o None
    anchors_px: Optional[np.ndarray]


def pick_hand(handedness: list, scores: list, dominant: str) -> int:
    """RF-02. Con una sola mano visible se usa esa (MediaPipe a veces confunde la lateralidad de
    una mano aislada). Con dos, se usa la dominante: en LSM la mano no dominante suele estar
    visible como apoyo o en reposo y no debe confundirse con la que deletrea."""
    if len(handedness) == 1:
        return 0
    matches = [i for i, h in enumerate(handedness) if h == dominant]
    pool = matches or range(len(handedness))
    return max(pool, key=lambda i: scores[i])


class Tracker:
    def __init__(self, cfg: Config):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp, self.cfg = mp, cfg
        hint = "Ver README, sección 'Modelos'."
        self._hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_require(cfg.hand_model, hint)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=cfg.min_hand_conf,
            min_hand_presence_confidence=cfg.min_hand_conf,
            min_tracking_confidence=cfg.min_hand_conf,
        ))
        self._pose = None
        if cfg.use_pose:
            self._pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=_require(cfg.pose_model, hint)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
            ))
        self._anchors_norm: Optional[np.ndarray] = None
        self._frame_i = 0
        self._t0 = time.monotonic()
        self._last_ts = -1

    def _timestamp(self) -> int:
        ts = max(int((time.monotonic() - self._t0) * 1000), self._last_ts + 1)
        self._last_ts = ts
        return ts

    def detect(self, rgb: np.ndarray) -> Optional[Observation]:
        h, w = rgb.shape[:2]
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        ts = self._timestamp()
        if self._pose is not None and self._frame_i % self.cfg.pose_every_n == 0:
            res = self._pose.detect_for_video(image, ts)
            if res.pose_landmarks:
                lm = res.pose_landmarks[0]
                self._anchors_norm = np.array([[lm[i].x, lm[i].y] for i in POSE_ANCHORS.values()],
                                              dtype=np.float32)
            else:
                self._anchors_norm = None
        self._frame_i += 1

        res = self._hands.detect_for_video(image, ts)
        if not res.hand_landmarks:
            return None
        labels = [c[0].category_name for c in res.handedness]
        scores = [c[0].score for c in res.handedness]
        i = pick_hand(labels, scores, self.cfg.dominant_hand)
        hand_norm = np.array([[p.x, p.y, p.z] for p in res.hand_landmarks[i]], dtype=np.float32)
        anchors_px = None if self._anchors_norm is None else to_pixel_space(self._anchors_norm, w, h)
        return Observation(hand_norm, to_pixel_space(hand_norm, w, h), labels[i], float(scores[i]),
                           self._anchors_norm, anchors_px)

    def close(self) -> None:
        self._hands.close()
        if self._pose is not None:
            self._pose.close()

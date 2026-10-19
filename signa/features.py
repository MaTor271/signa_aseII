"""
De landmarks de MediaPipe a vector de características. Única fuente de verdad para captura,
entrenamiento, dispositivo y cualquier port a otro lenguaje (los vectores golden lo verifican).

Mapeo a los parámetros de la LSM:
  Configuración -> 21 puntos relativos a la muñeca + 10 distancias entre yemas (contacto del pulgar)
  Orientación   -> se conserva: NO se normaliza la rotación
  Ubicación     -> opcional: muñeca relativa a hombros y nariz (requiere Pose)
Invariancias deseadas: posición en la imagen, distancia a la cámara, lateralidad (zurdos).

Cambiar cualquier cálculo aquí exige subir FEATURE_VERSION y reentrenar (RNF-07).
"""
from typing import Optional, Sequence

import numpy as np

FEATURE_VERSION = 2
N_LANDMARKS = 21
WRIST, MIDDLE_MCP = 0, 9
FINGERTIPS = np.array([4, 8, 12, 16, 20])  # pulgar, índice, medio, anular, meñique
_ia, _ib = np.triu_indices(len(FINGERTIPS), k=1)
_TIP_A, _TIP_B = FINGERTIPS[_ia], FINGERTIPS[_ib]
N_HAND_FEATURES = N_LANDMARKS * 3 + len(_TIP_A)  # 73
POSE_ANCHORS = {"nose": 0, "left_shoulder": 11, "right_shoulder": 12}  # índices de MediaPipe Pose
N_LOCATION_FEATURES = 4  # rel_x, rel_y, distancia a la cara, validez
_EPS = 1e-6


def n_features(use_location: bool) -> int:
    return N_HAND_FEATURES + (N_LOCATION_FEATURES if use_location else 0)


def to_pixel_space(norm: np.ndarray, width, height) -> np.ndarray:
    """(…, K, 2|3) normalizado [0,1] -> píxeles. z usa la escala de x, como define MediaPipe.
    width/height pueden ser escalares o arreglos (N,) para lotes."""
    out = np.array(norm, dtype=np.float32, copy=True)
    w, h = np.asarray(width, np.float32), np.asarray(height, np.float32)
    if w.ndim:  # lote: (N,) -> (N, 1) para multiplicar cada fila por su propio ancho
        w = w.reshape(-1, *([1] * (out.ndim - 2)))
        h = h.reshape(-1, *([1] * (out.ndim - 2)))
    out[..., 0] *= w
    out[..., 1] *= h
    if out.shape[-1] == 3:
        out[..., 2] *= w
    return out


def palm_size(hand_px: np.ndarray) -> np.ndarray:
    """Distancia muñeca -> nudillo medio en el plano de la imagen. Unidad de movimiento ("palma")."""
    return np.linalg.norm(hand_px[..., MIDDLE_MCP, :2] - hand_px[..., WRIST, :2], axis=-1)


def _hand_features(H: np.ndarray, mirror: np.ndarray) -> np.ndarray:
    P = H - H[:, WRIST:WRIST + 1, :]            # origen en la muñeca (crea copia nueva)
    P[:, :, 0] *= np.where(mirror, -1.0, 1.0).astype(np.float32)[:, None]  # zurdos -> derecha
    scale = np.sqrt((P[:, MIDDLE_MCP, :] ** 2).sum(axis=1))
    bad = scale <= _EPS
    if bad.any():                               # mano degenerada: usar el punto más lejano
        far = np.sqrt((P[bad] ** 2).sum(axis=2)).max(axis=1)
        scale[bad] = np.where(far > _EPS, far, 1.0)
    P /= scale[:, None, None]                   # invariante a distancia
    tips = np.sqrt(((P[:, _TIP_A] - P[:, _TIP_B]) ** 2).sum(axis=2))
    return np.concatenate([P.reshape(len(P), -1), tips], axis=1)


def _location_features(H: np.ndarray, mirror: np.ndarray, A: np.ndarray) -> np.ndarray:
    """A: (N, 3, 2) nariz, hombro izq., hombro der. en píxeles; NaN si no hubo Pose."""
    wrist = H[:, WRIST, :2]
    nose, ls, rs = A[:, 0], A[:, 1], A[:, 2]
    mid = (ls + rs) / 2.0
    width = np.linalg.norm(ls - rs, axis=1)
    valid = np.isfinite(width) & (width > _EPS) & np.isfinite(nose).all(axis=1)
    safe_w = np.where(valid, width, 1.0)
    rel = (wrist - np.nan_to_num(mid)) / safe_w[:, None]
    rel[mirror, 0] *= -1.0                       # lado dominante siempre del mismo signo
    d_face = np.linalg.norm(wrist - np.nan_to_num(nose), axis=1) / safe_w
    out = np.column_stack([rel, d_face, valid.astype(np.float32)]).astype(np.float32)
    out[~valid, :3] = 0.0
    return out


def extract_features_batch(H: np.ndarray, handedness: Sequence[str], A: Optional[np.ndarray] = None,
                           use_location: bool = False) -> np.ndarray:
    """H: (N, 21, 3) píxeles. handedness: N etiquetas 'Left'/'Right'. A: (N, 3, 2) o None."""
    H = np.asarray(H, dtype=np.float32).reshape(-1, N_LANDMARKS, 3)
    mirror = np.asarray(handedness) == "Left"
    feats = _hand_features(H, mirror)
    if use_location:
        if A is None:
            A = np.full((len(H), 3, 2), np.nan, dtype=np.float32)
        feats = np.concatenate([feats, _location_features(H, mirror, np.asarray(A, np.float32))], axis=1)
    return feats.astype(np.float32)


def extract_features(hand_px: np.ndarray, handedness: str, anchors_px: Optional[np.ndarray] = None,
                     use_location: bool = False) -> np.ndarray:
    """Versión de una muestra. Usa el mismo código que el lote: imposible que diverjan."""
    A = None if anchors_px is None else np.asarray(anchors_px, np.float32)[None]
    return extract_features_batch(np.asarray(hand_px)[None], [handedness], A, use_location)[0]


def augment_batch(H: np.ndarray, rng: np.random.Generator, max_rot_deg: float = 8.0,
                  jitter: float = 0.01) -> np.ndarray:
    """Rotación pequeña en el plano alrededor de la muñeca + ruido proporcional a la palma.
    La rotación es pequeña a propósito: la orientación distingue letras en LSM."""
    H = np.asarray(H, dtype=np.float32)
    theta = np.deg2rad(rng.uniform(-max_rot_deg, max_rot_deg, size=len(H))).astype(np.float32)
    c, s = np.cos(theta), np.sin(theta)
    wrist = H[:, WRIST:WRIST + 1, :2]
    xy = H[..., :2] - wrist
    rot = np.stack([c[:, None] * xy[..., 0] - s[:, None] * xy[..., 1],
                    s[:, None] * xy[..., 0] + c[:, None] * xy[..., 1]], axis=-1)
    out = H.copy()
    out[..., :2] = rot + wrist
    scale = np.maximum(palm_size(out), _EPS)[:, None, None]
    out += rng.normal(0.0, jitter, size=out.shape).astype(np.float32) * scale
    return out

"""
Configuración central (RNF-10). Todo umbral vive aquí, con unidades físicas (segundos, palmas/segundo)
para que el comportamiento no dependa de los FPS del equipo (RNF-03).

Para ajustar sin tocar código: crear `config.json` con solo los campos a cambiar, p. ej.
    {"dominant_hand": "Left", "camera_source": "picam", "min_hold_s": 0.3}
"""
import json
import os
from dataclasses import asdict, dataclass, fields

# Alfabeto manual LSM. Las letras dinámicas llevan trazo o movimiento: fuera del alcance estático.
LSM_STATIC_LETTERS = tuple("ABCDEFGHILMNOPRSTUVWY")  # 21
LSM_DYNAMIC_LETTERS = ("J", "K", "Ñ", "Q", "X", "Z")
REST = "_"  # mano visible sin seña: frontera natural entre palabras y letras repetidas
DEFAULT_LABELS = LSM_STATIC_LETTERS + (REST,)


@dataclass
class Config:
    # --- Rutas ---
    hand_model: str = "models/hand_landmarker.task"
    pose_model: str = "models/pose_landmarker_lite.task"
    classifier_model: str = "models/signa_mlp.tflite"
    meta_file: str = "models/meta.json"
    golden_file: str = "models/golden.json"
    piper_voice: str = "models/es_MX-claude-high.onnx"
    vosk_model: str = "models/vosk-model-small-es-0.42"
    data_dir: str = "data"
    reports_dir: str = "reports"
    font_path: str = ""  # vacío = buscar DejaVuSans/Arial automáticamente

    # --- Cámara ---
    camera_source: str = "0"  # índice de webcam o "picam"
    frame_width: int = 640
    frame_height: int = 480

    # --- Señante ---
    dominant_hand: str = "Right"  # "Right" | "Left"
    use_pose: bool = False        # ubicación respecto al cuerpo (vocabulario); el deletreo no la necesita
    pose_every_n: int = 3         # los hombros se mueven lento: Pose 1 de cada N cuadros

    # --- Detección y clasificación ---
    min_hand_conf: float = 0.5
    min_conf: float = 0.70        # probabilidad mínima de la clase ganadora
    min_margin: float = 0.20      # ventaja mínima sobre la segunda clase (letras parecidas)

    # --- Segmentación movimiento-detención (RF-04/05/06) ---
    move_speed_hi: float = 2.0    # palmas/s: por encima, la mano está en transición
    move_speed_lo: float = 1.0    # palmas/s: por debajo, la mano está detenida (histéresis)
    speed_tau_s: float = 0.08     # suavizado de la velocidad
    min_hold_s: float = 0.25      # detención mínima para aceptar letra
    min_move_s: float = 0.10      # movimiento mínimo que separa dos letras iguales (LL, RR, CC)
    hold_agreement: float = 0.70  # fracción de cuadros de la detención que deben coincidir
    word_end_absent_s: float = 1.0
    word_end_rest_s: float = 0.6  # contado desde que se reconoce el reposo (~0.25 s después)

    # --- Captura de datos (RF-11) ---
    collect_hz: float = 6.0
    collect_per_label: int = 150
    collect_min_hand_score: float = 0.8

    @classmethod
    def load(cls, path: str = "config.json") -> "Config":
        cfg = cls()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                overrides = json.load(f)
            valid = {f.name for f in fields(cls)}
            unknown = set(overrides) - valid
            if unknown:
                raise ValueError(f"Campos desconocidos en {path}: {sorted(unknown)}")
            for k, v in overrides.items():
                setattr(cfg, k, v)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.dominant_hand not in ("Right", "Left"):
            raise ValueError("dominant_hand debe ser 'Right' o 'Left'")
        if not self.move_speed_lo < self.move_speed_hi:
            raise ValueError("move_speed_lo debe ser menor que move_speed_hi")
        for name in ("min_conf", "min_margin", "hold_agreement"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} debe estar entre 0 y 1")

    def to_dict(self) -> dict:
        return asdict(self)

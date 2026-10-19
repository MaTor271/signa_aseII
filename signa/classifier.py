"""
Inferencia con LiteRT/TFLite (O4). Valida que el modelo corresponda a esta versión de features.py
(RNF-07): un desajuste silencioso daría predicciones malas sin ningún error visible.
"""
import json
import os
from dataclasses import dataclass
from typing import List

import numpy as np

from .features import FEATURE_VERSION, n_features


class ModelMismatchError(RuntimeError):
    pass


def load_interpreter(model_path: str, num_threads: int = 2):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
    interp = Interpreter(model_path=model_path, num_threads=num_threads)
    interp.allocate_tensors()
    return interp


@dataclass
class Prediction:
    label: str
    conf: float     # probabilidad de la clase ganadora
    margin: float   # ventaja sobre la segunda clase
    probs: np.ndarray


class SignClassifier:
    def __init__(self, model_path: str, meta_path: str):
        for p in (model_path, meta_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Falta {p}. Entrenen con: python -m signa train")
        with open(meta_path, encoding="utf-8") as f:
            self.meta = json.load(f)
        if self.meta.get("feature_version") != FEATURE_VERSION:
            raise ModelMismatchError(
                f"El modelo usa features v{self.meta.get('feature_version')} "
                f"y el código v{FEATURE_VERSION}. "
                "Reentrenen con: python -m signa train")
        self.labels: List[str] = self.meta["labels"]
        self.use_location: bool = bool(self.meta["use_location"])
        self.interpreter = load_interpreter(model_path)
        inp = self.interpreter.get_input_details()[0]
        out = self.interpreter.get_output_details()[0]
        expected = n_features(self.use_location)
        if int(inp["shape"][-1]) != expected or int(out["shape"][-1]) != len(self.labels):
            raise ModelMismatchError("El .tflite no coincide con meta.json (dimensiones). Reentrenen.")
        self._in, self._out = inp["index"], out["index"]

    def probs(self, features: np.ndarray) -> np.ndarray:
        self.interpreter.set_tensor(self._in, np.asarray(features, np.float32).reshape(1, -1))
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self._out)[0].copy()

    def predict(self, features: np.ndarray) -> Prediction:
        p = self.probs(features)
        top2 = np.argsort(p)[-2:]
        i, j = int(top2[1]), int(top2[0])
        return Prediction(self.labels[i], float(p[i]), float(p[i] - p[j]), p)

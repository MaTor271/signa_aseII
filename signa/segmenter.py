"""
Del flujo de predicciones por cuadro a letras y palabras, imitando cómo se deletrea en LSM.

El deletreo manual alterna DETENCIÓN (la mano sostiene la letra) y MOVIMIENTO (transición).
  - Solo se acepta una letra durante una detención, nunca en la transición (evita letras basura).
  - La velocidad se mide en palmas/segundo y las ventanas en segundos: mismo comportamiento a
    8 FPS en la Pi que a 30 FPS en la laptop (RNF-03).
  - Letra doble (LL, RR, CC): un rebote o deslizamiento breve entre dos detenciones iguales,
    o pasar por reposo. Sostener la misma letra NO la repite.
  - Cambio lento de forma sin mover la muñeca (p. ej. A -> S) también cuenta como letra nueva.
  - Fin de palabra: bajar la mano o sostener la posición de reposo.
"""
from collections import Counter, deque
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import REST, Config
from .features import palm_size

ABSENT, MOVING, HOLDING = "ausente", "movimiento", "detencion"
_EPS_T = 1e-6       # tolerancia de redondeo en comparaciones de tiempo
_MAX_SPEED = 8.0    # palmas/s; el deletreo real rara vez supera ~5


@dataclass(frozen=True)
class Event:
    kind: str   # "letter" | "word"
    value: str


class MotionEstimator:
    """Velocidad media de los 21 puntos, en palmas por segundo, suavizada con constante de tiempo."""

    def __init__(self, tau_s: float):
        self.tau_s = tau_s
        self.reset()

    def reset(self) -> None:
        self._prev: Optional[np.ndarray] = None
        self._t: Optional[float] = None
        self.speed = 0.0

    def update(self, t: float, hand_px: np.ndarray) -> float:
        xy = np.asarray(hand_px, np.float32)[:, :2]
        if self._prev is not None and t > self._t:
            dt = t - self._t
            palm = max(float(palm_size(hand_px)), 1e-6)
            raw = float(np.linalg.norm(xy - self._prev, axis=1).mean()) / palm / dt
            raw = min(raw, _MAX_SPEED)  # un salto del detector no debe "congelar" la lectura
            alpha = 1.0 - np.exp(-dt / self.tau_s)
            self.speed += alpha * (raw - self.speed)
        self._prev, self._t = xy, t
        return self.speed


class Speller:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.motion = MotionEstimator(cfg.speed_tau_s)
        self.word = ""
        self.sentence: List[str] = []
        self.phase = ABSENT
        self.candidate: Optional[str] = None
        self.progress = 0.0
        self._votes: deque = deque()
        self._moving = False
        self._phase_start = 0.0
        self._hold_label: Optional[str] = None
        self._last_letter: Optional[str] = None
        self._armed = True
        self._rest_start = 0.0

    # ------------------------------------------------------------------ API
    def update(self, t: float, hand_px: Optional[np.ndarray], label: Optional[str] = None,
               conf: float = 0.0, margin: float = 0.0) -> List[Event]:
        if hand_px is None:
            return self._on_absent(t)
        speed = self.motion.update(t, hand_px)
        if self._moving:
            self._moving = speed > self.cfg.move_speed_lo
        else:
            self._moving = speed > self.cfg.move_speed_hi
        if self._moving:
            return self._on_moving(t)
        ok = label is not None and conf >= self.cfg.min_conf and margin >= self.cfg.min_margin
        return self._on_holding(t, label if ok else None)

    def backspace(self) -> None:
        if self.word:
            self.word = self.word[:-1]
        elif self.sentence:
            self.word = self.sentence.pop()
        self._last_letter, self._armed = None, True

    def clear(self) -> None:
        self.word, self.sentence = "", []
        self._last_letter, self._armed = None, True

    @property
    def text(self) -> str:
        return " ".join(self.sentence + ([self.word] if self.word else []))

    # ------------------------------------------------------------ estados
    def _enter(self, phase: str, t: float) -> None:
        if self.phase == phase:
            return
        self.phase, self._phase_start = phase, t
        self._votes.clear()
        self._hold_label = None
        self.candidate, self.progress = None, 0.0

    def _on_absent(self, t: float) -> List[Event]:
        self._enter(ABSENT, t)
        self.motion.reset()
        self._moving = False
        dur = t - self._phase_start + _EPS_T
        if dur >= self.cfg.min_move_s:  # descarta parpadeos de detección de 1-2 cuadros
            self._armed = True
        if dur >= self.cfg.word_end_absent_s:
            return self._end_word()
        return []

    def _on_moving(self, t: float) -> List[Event]:
        self._enter(MOVING, t)
        if t - self._phase_start + _EPS_T >= self.cfg.min_move_s:
            self._armed = True
        return []

    def _on_holding(self, t: float, vote: Optional[str]) -> List[Event]:
        self._enter(HOLDING, t)
        self._votes.append((t, vote))
        while self._votes and self._votes[0][0] < t - self.cfg.min_hold_s - _EPS_T:
            self._votes.popleft()

        counts = Counter(v for _, v in self._votes if v is not None)
        if not counts:
            self.candidate, self.progress = None, 0.0
            return []
        winner, n = counts.most_common(1)[0]
        agreement = n / len(self._votes)
        held = min(1.0, (t - self._phase_start + _EPS_T) / self.cfg.min_hold_s)
        self.candidate = winner
        if winner == self._hold_label:
            self.progress = 1.0
        else:
            self.progress = held * min(1.0, agreement / self.cfg.hold_agreement)

        events: List[Event] = []
        if held >= 1.0 and agreement >= self.cfg.hold_agreement and winner != self._hold_label:
            self._hold_label = winner
            if winner == REST:
                self._armed = True
                self._rest_start = t
            elif winner != self._last_letter or self._armed:
                self.word += winner
                self._last_letter, self._armed = winner, False
                events.append(Event("letter", winner))
        rest_for = t - self._rest_start + _EPS_T
        if self._hold_label == REST and self.word and rest_for >= self.cfg.word_end_rest_s:
            events += self._end_word()
        return events

    def _end_word(self) -> List[Event]:
        if not self.word:
            return []
        word, self.word = self.word, ""
        self.sentence.append(word)
        self._last_letter, self._armed = None, True
        return [Event("word", word)]

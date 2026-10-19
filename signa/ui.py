"""
Interfaz pensada primero para la persona sorda que señala (RF-08, RF-09, RNF-09).
Ella no oye la voz que el dispositivo emite en su nombre, así que todo se le comunica visualmente:
  - Letra candidata grande con anillo de progreso: ve que el sistema "la está leyendo" y puede
    corregir la forma antes de que se acepte.
  - Esqueleto coloreado por fase: gris = transición, ámbar = leyendo, verde = aceptada.
  - Texto confirmado grande y con alto contraste.
  - Indicador "VOZ" mientras el dispositivo habla por ella.
  - Lo que dice la persona oyente, con acentos y ñ (cv2.putText no los dibuja: se usa Pillow).
"""
import importlib.util
import os
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import REST
from .segmenter import HOLDING, MOVING

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
                    (17, 18), (18, 19), (19, 20)]
GRAY, AMBER, GREEN, WHITE, CYAN = (150, 150, 150), (0, 176, 255), (80, 220, 80), (255, 255, 255), (255, 220, 0)
_FONT_CANDIDATES = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                    "C:/Windows/Fonts/arialbd.ttf"]


@dataclass
class ViewState:
    hand_px: Optional[np.ndarray] = None
    phase: str = ""
    candidate: Optional[str] = None
    progress: float = 0.0
    text: str = ""
    heard: str = ""
    speaking: bool = False
    debug: List[str] = field(default_factory=list)


class TextPainter:
    """Dibuja texto Unicode con Pillow en una sola pasada por cuadro (barato: ~2-3 ms a 640x480)."""

    def __init__(self, font_path: str = ""):
        self._path = font_path or next((p for p in _FONT_CANDIDATES if os.path.exists(p)), "")
        self._fonts = {}
        self._queue: List[Tuple[str, Tuple[int, int], int, Tuple[int, int, int]]] = []
        self._pil = bool(self._path) and importlib.util.find_spec("PIL") is not None

    def add(self, text: str, org: Tuple[int, int], size: int, color_bgr=WHITE) -> None:
        self._queue.append((text, org, size, color_bgr))

    def width(self, text: str, size: int) -> int:
        if self._pil:
            return int(self._font(size).getlength(text))
        return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size / 30, 2)[0][0]

    def _font(self, size: int):
        from PIL import ImageFont
        if size not in self._fonts:
            self._fonts[size] = ImageFont.truetype(self._path, size)
        return self._fonts[size]

    def flush(self, img: np.ndarray) -> np.ndarray:
        if not self._queue:
            return img
        if self._pil:
            from PIL import Image, ImageDraw
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            for text, (x, y), size, (b, g, r) in self._queue:
                draw.text((x, y), text, font=self._font(size), fill=(r, g, b),
                          stroke_width=max(2, size // 14), stroke_fill=(0, 0, 0))
            img = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        else:  # respaldo sin Pillow: quita acentos para que al menos sea legible
            for text, (x, y), size, color in self._queue:
                ascii_ = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
                cv2.putText(img, ascii_, (x, y + size), cv2.FONT_HERSHEY_SIMPLEX, size / 30, color, 2)
        self._queue.clear()
        return img


def _fit_tail(painter: TextPainter, text: str, size: int, max_w: int) -> str:
    """Si no cabe, muestra el final (lo más reciente) cortando en palabra completa: '…mañana nos vemos'."""
    if painter.width(text, size) <= max_w:
        return text
    words = text.split(" ")
    while len(words) > 1 and painter.width("…" + " ".join(words), size) > max_w:
        words.pop(0)
    tail = "…" + " ".join(words)
    while len(tail) > 1 and painter.width(tail, size) > max_w:  # una sola palabra muy larga
        tail = "…" + tail[2:]
    return tail


def _speaker_icon(img, x, y, s, color):
    pts = np.array([[x, y + s // 3], [x + s // 3, y + s // 3], [x + 2 * s // 3, y],
                    [x + 2 * s // 3, y + s], [x + s // 3, y + 2 * s // 3], [x, y + 2 * s // 3]], np.int32)
    cv2.fillPoly(img, [pts], color)
    for r in (s // 3, 2 * s // 3):
        cv2.ellipse(img, (x + 2 * s // 3, y + s // 2), (r, r), 0, -45, 45, color, 2)


class Renderer:
    """round_display=True: recorta a cuadrado, centra todo y ajusta cada línea a la cuerda del
    círculo, para la pantalla circular del dispositivo."""

    def __init__(self, font_path: str = "", round_display: bool = False):
        self.painter = TextPainter(font_path)
        self.round = round_display

    def _line(self, text: str, y: int, size: int, color, w: int, h: int) -> None:
        if self.round:
            r, cy = min(w, h) / 2, h / 2
            far = max(abs(y - cy), abs(y + size - cy))
            half = max(0.0, (r * r - far * far)) ** 0.5 - 8
            text = _fit_tail(self.painter, text, size, int(2 * half))
            x = w // 2 - self.painter.width(text, size) // 2
        else:
            text = _fit_tail(self.painter, text, size, w - 24)
            x = 12
        self.painter.add(text, (x, y), size, color)

    def render(self, frame_bgr: np.ndarray, st: ViewState) -> np.ndarray:
        img = frame_bgr
        h, w = img.shape[:2]
        if self.round:  # recorte cuadrado centrado
            s = min(h, w)
            x0, y0 = (w - s) // 2, (h - s) // 2
            img = img[y0:y0 + s, x0:x0 + s].copy()
            if st.hand_px is not None:
                st.hand_px = st.hand_px - np.array([x0, y0, 0], np.float32)
            h = w = s
        color = GRAY if st.phase == MOVING else (GREEN if st.progress >= 1.0 else AMBER)

        if st.hand_px is not None:
            pts = st.hand_px[:, :2].astype(int)
            for a, b in HAND_CONNECTIONS:
                cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, 2, cv2.LINE_AA)

        # Letra candidata + anillo de progreso
        if st.phase == HOLDING and st.candidate:
            cx, cy = (w // 2, int(0.27 * h)) if self.round else (57, 62)
            cv2.circle(img, (cx, cy), 40, (40, 40, 40), -1, cv2.LINE_AA)
            cv2.ellipse(img, (cx, cy), (40, 40), -90, 0, 360 * st.progress, color, 6, cv2.LINE_AA)
            glyph = "·" if st.candidate == REST else st.candidate
            self.painter.add(glyph, (cx - self.painter.width(glyph, 52) // 2, cy - 32), 52, WHITE)

        if st.speaking:
            x, y = (w // 2 - 50, int(0.08 * h)) if self.round else (w - 110, 12)
            _speaker_icon(img, x, y, 32, GREEN)
            self.painter.add("VOZ", (x + 42, y), 26, GREEN)

        # Banda de texto: oyente arriba, señante abajo
        top = int(0.60 * h) if self.round else h - 100
        overlay = img.copy()
        cv2.rectangle(overlay, (0, top), (w, top + 100), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
        if st.heard:
            self._line("Oyente: " + st.heard, top + 8, 26, CYAN, w, h)
        self._line(st.text + "▏", top + 48, 34, WHITE, w, h)

        for i, line in enumerate(st.debug):
            y = int(0.40 * h) + 20 * i if self.round else 120 + 20 * i
            self._line(line, y, 16, WHITE, w, h)
        img = self.painter.flush(img)

        if self.round:
            mask = np.zeros((h, w), np.uint8)
            cv2.circle(mask, (w // 2, h // 2), w // 2, 255, -1)
            img[mask == 0] = 0
        return img

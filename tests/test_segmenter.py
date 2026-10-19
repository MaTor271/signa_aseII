"""Simula deletreo real: detenciones (letra), transiciones (movimiento), ausencias y reposo."""
import numpy as np
import pytest

from conftest import make_hand
from signa.config import REST, Config
from signa.segmenter import Speller

PALM = 80.0


def spell(script, fps=20.0, cfg=None, conf=0.95, margin=0.6, seed=0):
    """script: lista de (tipo, duración_s, etiqueta). tipo: 'hold' | 'move' | 'absent'.
    'move' desplaza la mano a 4 palmas/s (transición típica); 'hold' agrega temblor de detección."""
    cfg = cfg or Config()
    rng = np.random.default_rng(seed)
    sp, t, dt = Speller(cfg), 0.0, 1.0 / fps
    base = make_hand(rng, palm=PALM)
    pos = np.zeros(2, np.float32)
    events = []
    for kind, dur, label in script:
        for _ in range(max(1, int(round(dur * fps)))):
            t += dt
            if kind == "absent":
                events += sp.update(t, None)
                continue
            if kind == "move":
                pos += np.array([4.0 * PALM * dt, 0], np.float32)
            hand = base.copy()
            hand[:, :2] += pos + rng.normal(0, 0.6, (21, 2))  # ~0.6 px de ruido de MediaPipe
            lab = label if kind == "hold" else rng.choice(list("ABCDE"))  # basura en transición
            events += sp.update(t, hand, lab, conf, margin)
    return sp, events


def letters(events):
    return "".join(e.value for e in events if e.kind == "letter")


def words(events):
    return [e.value for e in events if e.kind == "word"]


def test_spelling_word_with_transitions():
    script = [("hold", .5, "H"), ("move", .2, None), ("hold", .5, "O"), ("move", .2, None),
              ("hold", .5, "L"), ("move", .2, None), ("hold", .5, "A"), ("absent", 1.2, None)]
    sp, ev = spell(script)
    assert letters(ev) == "HOLA"
    assert words(ev) == ["HOLA"]


def test_transition_frames_never_become_letters():
    _, ev = spell([("move", 2.0, None)])
    assert letters(ev) == ""


def test_double_letter_with_bounce():
    """RF-05: LL con un rebote breve, sin bajar la mano."""
    _, ev = spell([("hold", .5, "L"), ("move", .15, None), ("hold", .5, "L")])
    assert letters(ev) == "LL"


def test_double_letter_through_rest():
    _, ev = spell([("hold", .5, "R"), ("hold", .4, REST), ("hold", .5, "R")])
    assert letters(ev) == "RR"


def test_sustained_letter_is_not_repeated():
    _, ev = spell([("hold", 3.0, "L")])
    assert letters(ev) == "L"


def test_micro_movement_does_not_double():
    _, ev = spell([("hold", .5, "L"), ("move", .05, None), ("hold", .5, "L")])
    assert letters(ev) == "L"


def test_detection_dropout_does_not_double():
    """MediaPipe pierde la mano 1-2 cuadros a veces: no debe crear una letra doble."""
    _, ev = spell([("hold", .5, "L"), ("absent", .05, None), ("hold", .5, "L")], fps=30)
    assert letters(ev) == "L"


def test_shape_change_without_wrist_movement():
    """A -> S cambia solo los dedos; también es letra nueva."""
    _, ev = spell([("hold", .6, "A"), ("hold", .6, "S")])
    assert letters(ev) == "AS"


def test_low_confidence_is_ignored():
    _, ev = spell([("hold", 1.0, "M")], conf=0.5)
    assert letters(ev) == ""


def test_ambiguous_prediction_is_ignored():
    """M y N se parecen: si el modelo duda (margen bajo), no se escribe nada."""
    _, ev = spell([("hold", 1.0, "M")], margin=0.05)
    assert letters(ev) == ""


def test_word_ends_with_rest():
    _, ev = spell([("hold", .5, "S"), ("move", .2, None), ("hold", .5, "I"), ("hold", 1.0, REST)])
    assert words(ev) == ["SI"]


def test_word_ends_when_hand_drops():
    _, ev = spell([("hold", .5, "N"), ("move", .2, None), ("hold", .5, "O"), ("absent", 1.1, None)])
    assert words(ev) == ["NO"]


def test_two_words_in_a_row():
    script = [("hold", .5, "S"), ("move", .2, None), ("hold", .5, "I"), ("hold", 1.0, REST),
              ("hold", .5, "N"), ("move", .2, None), ("hold", .5, "O"), ("hold", 1.0, REST)]
    sp, ev = spell(script)
    assert words(ev) == ["SI", "NO"] and sp.text == "SI NO"


def test_recovers_quickly_after_detector_jump():
    """Si el detector salta (p. ej. cambia de mano) la lectura no queda bloqueada en 'movimiento'."""
    cfg = Config()
    sp, hand = Speller(cfg), make_hand(np.random.default_rng(0))
    sp.update(0.05, hand, "A", 0.95, 0.6)
    jumped = hand.copy(); jumped[:, 0] += 400
    t, ev = 0.10, []
    while t < 1.0:
        ev += sp.update(t, jumped, "A", 0.95, 0.6)
        if ev:
            break
        t += 0.05
    assert ev and t - 0.10 <= 0.25 + cfg.min_hold_s


@pytest.mark.parametrize("fps", [8, 12, 15, 20, 30])
def test_fps_independence(fps):
    """RNF-03: la letra se acepta a min_hold_s ± 1 cuadro, a cualquier FPS."""
    cfg = Config()
    sp, t = Speller(cfg), 0.0
    hand = make_hand(np.random.default_rng(0))
    first = None
    while t < 2.0:
        t += 1.0 / fps
        if sp.update(t, hand, "A", 0.95, 0.6):
            break
        if first is None:
            first = t
    assert abs((t - first) - cfg.min_hold_s) <= 1.0 / fps + 1e-6


def test_backspace_and_clear():
    sp, ev = spell([("hold", .5, "S"), ("move", .2, None), ("hold", .5, "I"), ("absent", 1.2, None)])
    sp.backspace()            # quita el espacio: vuelve a editar "SI"
    assert sp.word == "SI"
    sp.backspace()
    assert sp.text == "S"
    sp.clear()
    assert sp.text == ""

import time

import numpy as np

from conftest import make_hand
from signa.features import (FEATURE_VERSION, N_HAND_FEATURES, augment_batch, extract_features,
                            extract_features_batch, n_features, to_pixel_space)

ANCHORS = np.array([[320, 120], [400, 260], [240, 260]], np.float32)  # nariz, hombros


def test_shapes_and_version():
    assert FEATURE_VERSION == 2
    assert n_features(False) == N_HAND_FEATURES == 73
    assert n_features(True) == 77


def test_invariant_to_position(rng):
    p = make_hand(rng)
    q = p.copy(); q[:, :2] += [57, -31]
    assert np.allclose(extract_features(p, "Right"), extract_features(q, "Right"), atol=1e-5)


def test_invariant_to_distance(rng):
    p = make_hand(rng)
    q = (p - p[0]) * 1.8 + p[0]
    assert np.allclose(extract_features(p, "Right"), extract_features(q, "Right"), atol=1e-5)


def test_mirror_left_hand_equals_right(rng):
    """RNF-08: una persona zurda produce las mismas características que una diestra."""
    p = make_hand(rng)
    q = p.copy(); q[:, 0] = 640 - q[:, 0]
    assert np.allclose(extract_features(p, "Right"), extract_features(q, "Left"), atol=1e-5)


def test_orientation_is_preserved(rng):
    """En LSM la orientación distingue letras: rotar la mano debe cambiar las características."""
    p = make_hand(rng)
    th = np.deg2rad(30)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], np.float32)
    q = p.copy()
    q[:, :2] = (p[:, :2] - p[0, :2]) @ rot.T + p[0, :2]
    assert np.abs(extract_features(p, "Right") - extract_features(q, "Right")).max() > 0.1


def test_augmentation_keeps_rotation_small(rng):
    p = make_hand(rng)
    q = augment_batch(np.repeat(p[None], 200, 0), np.random.default_rng(1), max_rot_deg=8, jitter=0)
    v0 = p[9, :2] - p[0, :2]
    v = q[:, 9, :2] - q[:, 0, :2]
    ang = np.degrees(np.abs(np.arctan2(v[:, 1], v[:, 0]) - np.arctan2(v0[1], v0[0])))
    assert ang.max() <= 8.0 + 1e-3


def test_single_equals_batch(rng):
    H = np.stack([make_hand(rng) for _ in range(5)])
    hands = ["Right", "Left", "Right", "Left", "Right"]
    A = np.repeat(ANCHORS[None], 5, axis=0)
    batch = extract_features_batch(H, hands, A, use_location=True)
    for i in range(5):
        assert np.allclose(batch[i], extract_features(H[i], hands[i], A[i], use_location=True))


def test_location_invariant_to_scene_shift_and_zoom(rng):
    p = make_hand(rng, center=(400, 300))
    f = extract_features(p, "Right", ANCHORS, use_location=True)[-4:]
    shift = np.array([40, -20], np.float32)
    q, a = p.copy(), ANCHORS + shift
    q[:, :2] += shift
    assert np.allclose(f, extract_features(q, "Right", a, use_location=True)[-4:], atol=1e-5)
    c = ANCHORS.mean(0)
    q, a = p.copy(), (ANCHORS - c) * 0.7 + c
    q[:, :2] = (p[:, :2] - c) * 0.7 + c
    assert np.allclose(f, extract_features(q, "Right", a, use_location=True)[-4:], atol=1e-4)
    assert f[-1] == 1.0


def test_location_missing_pose_is_flagged(rng):
    f = extract_features(make_hand(rng), "Right", None, use_location=True)
    assert np.all(f[-4:] == 0) and np.isfinite(f).all()


def test_to_pixel_space_batch_matches_single(rng):
    norm = rng.uniform(0, 1, (4, 21, 3)).astype(np.float32)
    w, h = np.array([640, 320, 640, 1280]), np.array([480, 240, 640, 720])
    batch = to_pixel_space(norm, w, h)
    for i in range(4):
        assert np.allclose(batch[i], to_pixel_space(norm[i], w[i], h[i]))


def test_feature_speed(rng):
    """RNF-11: < 50 µs por mano (mediana)."""
    p = make_hand(rng)
    times = []
    for _ in range(2000):
        t0 = time.perf_counter(); extract_features(p, "Right"); times.append(time.perf_counter() - t0)
    assert np.median(times) * 1e6 < 50

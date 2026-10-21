"""Pruebas de humo del baseline con datos sintéticos (no sustituyen los resultados reales)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lsc.features.landmarks import D, N_FEATURES, frame_vector, video_features  # noqa: E402


def _pose():
    p = np.zeros((33, 3), dtype=np.float32)
    p[11], p[12] = (0.4, 0.5, 0), (0.6, 0.5, 0)
    return p


def test_frame_vector_is_translation_and_scale_invariant():
    hand = np.random.default_rng(0).random((21, 3)).astype(np.float32)
    a = frame_vector(_pose(), hand, None)
    shifted = _pose().copy(); shifted[:, :2] = shifted[:, :2] * 2 + 0.1
    b = frame_vector(shifted, hand * 2 + 0.1, None)
    assert a.shape == (D,)
    assert np.allclose(a, b, atol=1e-5, equal_nan=True)


def test_missing_hand_is_nan_and_missing_pose_is_all_nan():
    v = frame_vector(_pose(), None, None)
    assert np.isnan(v[14:]).all() and not np.isnan(v[:14]).any()
    assert np.isnan(frame_vector(None, None, None)).all()


def test_video_features_shape_and_detection_rate():
    seq = np.full((10, D), np.nan); seq[:5, :] = 0.1
    f = video_features(seq)
    assert f.shape == (N_FEATURES,)
    assert f[-3] == 0.5 and f[-1] == 10


def test_pixel_conversion_makes_resolution_irrelevant():
    """La misma escena física grabada a 640x360 y a 640x480 debe dar el mismo vector."""
    rng = np.random.default_rng(1)
    hand_px = rng.random((21, 2)) * [300, 200] + [200, 100]          # puntos en píxeles
    pose_px = np.zeros((33, 2)); pose_px[11], pose_px[12] = (250, 180), (390, 180)
    def vec(w, h, convert):
        pose = np.concatenate([pose_px / [w, h], np.zeros((33, 1))], axis=1).astype(np.float32)
        hand = np.concatenate([hand_px / [w, h], np.zeros((21, 1))], axis=1).astype(np.float32)
        return frame_vector(pose, hand, None, *( (w, h) if convert else (1, 1) ))

    assert np.allclose(vec(640, 360, True), vec(640, 480, True), atol=1e-4, equal_nan=True)
    # sin la conversión a píxeles las dos resoluciones NO coinciden: ese es el sesgo que se evita
    assert not np.allclose(vec(640, 360, False), vec(640, 480, False), atol=1e-3, equal_nan=True)


def test_implausible_frames_are_dropped_without_touching_valid_ones():
    from lsc.features.landmarks import drop_implausible_frames
    seq = np.zeros((4, D)); seq[1, 3] = 50.0  # cuadro con hombros casi superpuestos (escala ~0)
    out = drop_implausible_frames(seq, max_abs=6)
    assert np.isnan(out[1]).all() and not np.isnan(out[[0, 2, 3]]).any()

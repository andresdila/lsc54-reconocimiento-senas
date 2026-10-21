"""Pruebas de la representación secuencial y del modelo (Entrega 2)."""
import numpy as np
import pytest

from lsc.features.landmarks import D
from lsc.features.sequence import D_IN, augment, fill_short_gaps, prepare, resample_indices


def _seq(T=40, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (T, D)).astype(np.float32)


def test_prepare_shape_and_no_nan():
    s = _seq()
    s[5:10, 14:56] = np.nan     # mano izquierda ausente 5 cuadros
    s[20:, 56:] = np.nan        # mano derecha ausente al final
    x = prepare(s, n_frames=32)
    assert x.shape == (32, D_IN)
    assert not np.isnan(x).any()


def test_short_gaps_filled_long_gaps_masked():
    s = _seq()
    s[5:10, 14:56] = np.nan          # 5 cuadros: se interpola
    s[20:30, 56:] = np.nan           # 10 cuadros: queda ausente
    f = fill_short_gaps(s, max_gap=5)
    assert not np.isnan(f[5:10, 14:56]).any()
    assert np.allclose(f[7, 14:56], s[4, 14:56] + (s[10, 14:56] - s[4, 14:56]) * 3 / 6, atol=1e-5)
    assert np.isnan(f[20:30, 56:]).all()


def test_gaps_at_edges_are_not_extrapolated():
    s = _seq()
    s[:3, 14:56] = np.nan
    assert np.isnan(fill_short_gaps(s, 5)[:3, 14:56]).all()


def test_mask_matches_absence():
    s = _seq(T=32)
    s[:, 56:] = np.nan
    x = prepare(s, n_frames=32, max_gap=0)
    assert (x[:, D + 2] == 0).all() and (x[:, D + 1] == 1).all()
    assert (x[:, 56:D] == 0).all()


def test_implausible_frames_marked_absent():
    s = _seq(T=32)
    s[10, 0] = 50.0
    x = prepare(s, n_frames=32, max_gap=0, max_abs=6.0)
    assert x[10, D] == 0  # sin pose en ese cuadro


def test_resample_endpoints():
    idx = resample_indices(53, 32)
    assert idx[0] == 0 and idx[-1] == 52 and len(idx) == 32


def test_augment_keeps_absent_at_zero_and_is_seeded():
    s = _seq()
    s[:, 56:] = np.nan
    a1 = augment(s, np.random.default_rng(1))
    a2 = augment(s, np.random.default_rng(1))
    assert np.array_equal(a1, a2)
    assert (a1[:, 56:D] == 0).all() and a1.shape == (32, D_IN)


def test_model_forward_shapes():
    torch = pytest.importorskip("torch")
    from lsc.models.sequence import SeqClassifier
    for enc in ("gru", "lstm", "conv"):
        m = SeqClassifier(D_IN, 43, encoder=enc)
        logits, w = m(torch.zeros(4, 32, D_IN), torch.zeros(4), return_attention=True)
        assert logits.shape == (4, 43) and w.shape == (4, 32)
        assert torch.allclose(w.sum(1), torch.ones(4))

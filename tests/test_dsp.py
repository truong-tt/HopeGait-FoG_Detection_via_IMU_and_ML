"""DSP guarantees: filters don't leak future, FI is freeze-band sensitive,
RobustScaler round-trips."""

import numpy as np
import tempfile

from data_pipeline.dsp import IMUFilter, freeze_index_window, RobustScaler


def test_lowpass_is_causal():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((500, 3)).astype(np.float32)
    f = IMUFilter()
    y_full = f.apply_lowpass(x)
    # Changing samples after t must not change y[t] for any prior t.
    x2 = x.copy()
    x2[300:] = rng.standard_normal((200, 3)).astype(np.float32)
    y_partial = f.apply_lowpass(x2)
    np.testing.assert_allclose(y_full[:300], y_partial[:300], atol=1e-6)


def test_filter_no_startup_transient():
    # A constant input must remain ~constant under warm-started lfilter
    # (the original bug: zero-state filter ramps from 0 -> input value).
    x = np.full((200, 3), 1.5, dtype=np.float32)
    y = IMUFilter().apply_lowpass(x)
    assert np.max(np.abs(y - 1.5)) < 1e-3


def test_freeze_index_band_sensitivity():
    fs = 100.0
    t = np.arange(0, 5, 1 / fs)
    walking = np.stack([np.sin(2 * np.pi * 1.5 * t)] * 3, axis=1)   # in loco band
    tremor = np.stack([np.sin(2 * np.pi * 5.0 * t)] * 3, axis=1)    # in freeze band
    fi_w = freeze_index_window(walking, fs=fs)
    fi_t = freeze_index_window(tremor, fs=fs)
    assert fi_t > fi_w


def test_robust_scaler_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    data = rng.standard_normal((100, 5)).astype(np.float32) * 3 + 7
    s = RobustScaler().fit(data)
    p = tmp_path / "scaler.npz"
    s.save(str(p))
    s2 = RobustScaler.load(str(p))
    np.testing.assert_allclose(s.median, s2.median)
    np.testing.assert_allclose(s.iqr, s2.iqr)

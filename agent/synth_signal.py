"""Deterministic synthesis of 6-channel lumbar IMU signals.

Renders walking + FoG phenomenology from a clinical profile and event timeline.
Output: (n, 6) float32 at `fs` Hz — gait oscillator + freeze-band injection
(3–8 Hz, Moore-Bachlin FI band) + tremor + sensor noise — plus (n,) int64 labels.
The LLM only sets parameters; all waveforms come from numpy/scipy.
"""

from __future__ import annotations

import numpy as np

from typing import Any


GAIT_AX_AMP = 1.4   # forward accel peak amplitude during normal gait (m/s^2)
GAIT_AY_AMP = 0.9   # lateral accel
GAIT_AZ_AMP = 3.5   # vertical accel (peak-to-peak heel-strike component)
GAIT_GYRO_AMP = 1.0  # rad/s peaks for trunk rotation about each axis
GRAVITY_AZ = 9.81

NOISE_ACCEL_STD = 0.08  # m/s^2 sensor noise per axis (realistic for consumer IMU)
NOISE_GYRO_STD = 0.04   # rad/s

FOG_GAIT_DAMP = 0.2     # gait amplitude during freeze (fraction of normal)
FOG_BAND_LO = 3.0       # Hz; FI numerator band start
FOG_BAND_HI = 8.0       # Hz; FI numerator band end
TRANSITION_S = 0.5      # cross-fade duration between walk/FoG segments


def _crossfade_mask(n_samples: int, fs: float, events: list[dict[str, Any]]) -> np.ndarray:
    """fog_frac[t] in [0, 1] from an event list, with 0.5-s Bartlett cross-fades.

    Events with type != "fog" are treated as walk; gaps default to walk.
    """
    fog_mask = np.zeros(n_samples, dtype=np.float32)
    for ev in events:
        if str(ev.get("type", "")).lower() != "fog":
            continue
        start = max(0, int(round(float(ev["start_s"]) * fs)))
        end = min(n_samples, start + int(round(float(ev["duration_s"]) * fs)))
        if end <= start:
            continue
        fog_mask[start:end] = 1.0

    # Linear cross-fade — convolve binary mask with a triangle kernel and clip.
    fade = max(1, int(round(TRANSITION_S * fs)))
    if fade > 1:
        kernel = np.bartlett(2 * fade + 1).astype(np.float32)
        kernel /= kernel.sum()
        smoothed = np.convolve(fog_mask, kernel, mode="same")
        fog_mask = np.clip(smoothed, 0.0, 1.0)
    return fog_mask


def _band_limited_noise(n_samples: int, fs: float, lo: float, hi: float,
                        rng: np.random.Generator) -> np.ndarray:
    """Unit-variance white noise band-passed to [lo, hi] Hz via FFT.

    Stochastic rather than a pure sine — prevents the network from ignoring
    a single predictable frequency during freeze/tremor episodes.
    """
    x = rng.standard_normal(n_samples).astype(np.float32)
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    mask = (freqs >= lo) & (freqs <= hi)
    spectrum[~mask] = 0
    out = np.fft.irfft(spectrum, n=n_samples).astype(np.float32)
    s = out.std()
    return out / (s + 1e-8)  # unit-variance for predictable amplitude scaling


def synthesize_subject(
    profile: dict[str, Any],
    events: list[dict[str, Any]],
    duration_s: float,
    fs: float = 64.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Render one subject's (n, 6) float32 signal and (n,) int64 FoG labels.

    Profile keys: gait_freq_hz, tremor_band_hz ([lo, hi] Hz),
    fog_severity (NFOG-Q 0–28), tremor_amp_g (≈0–0.3 × g). Missing
    fields fall back to safe defaults.
    """
    rng = np.random.default_rng(seed if seed is not None else int(profile.get("subject_id", 0) * 7919))
    n_samples = int(round(duration_s * fs))
    t = np.arange(n_samples, dtype=np.float32) / fs

    gait_freq = float(profile.get("gait_freq_hz", 1.8))
    tremor_band = profile.get("tremor_band_hz", [4.5, 6.0])
    tremor_lo = float(tremor_band[0])
    tremor_hi = float(tremor_band[1])
    fog_severity = float(profile.get("fog_severity", 12.0))   # NFOG-Q range
    tremor_amp_g = float(profile.get("tremor_amp_g", 0.05))   # fraction of g

    # Gait oscillator. Phase randomization avoids subjects being co-phased.
    phi = float(rng.uniform(0, 2 * np.pi))
    gait_1x = np.sin(2 * np.pi * gait_freq * t + phi)
    gait_2x = np.sin(2 * np.pi * 2 * gait_freq * t + 2 * phi)

    fog_frac = _crossfade_mask(n_samples, fs, events)
    walk_frac = 1.0 - fog_frac

    # Build each channel as a sum: walk gait + (gain-damped during fog) + freeze-band + tremor + noise.
    signal = np.empty((n_samples, 6), dtype=np.float32)

    gait_gain = walk_frac + FOG_GAIT_DAMP * fog_frac

    # Freeze band amplitude scales with severity (NFOG-Q normalized to 0..1)
    # and is gated by `fog_frac` so it only fires inside FoG segments.
    freeze_amp_accel = 0.6 + 0.05 * fog_severity   # m/s^2
    freeze_amp_gyro = 0.15 + 0.015 * fog_severity   # rad/s

    fb_ax = _band_limited_noise(n_samples, fs, FOG_BAND_LO, FOG_BAND_HI, rng) * freeze_amp_accel * fog_frac
    fb_ay = _band_limited_noise(n_samples, fs, FOG_BAND_LO, FOG_BAND_HI, rng) * freeze_amp_accel * fog_frac
    fb_az = _band_limited_noise(n_samples, fs, FOG_BAND_LO, FOG_BAND_HI, rng) * freeze_amp_accel * fog_frac
    fb_gx = _band_limited_noise(n_samples, fs, FOG_BAND_LO, FOG_BAND_HI, rng) * freeze_amp_gyro * fog_frac
    fb_gy = _band_limited_noise(n_samples, fs, FOG_BAND_LO, FOG_BAND_HI, rng) * freeze_amp_gyro * fog_frac
    fb_gz = _band_limited_noise(n_samples, fs, FOG_BAND_LO, FOG_BAND_HI, rng) * freeze_amp_gyro * fog_frac

    # Resting tremor — present everywhere (lumbar IMU picks up trunk-coupled tremor),
    # at low amplitude, in the subject-specific tremor band.
    tremor_accel_amp = tremor_amp_g * 9.81
    tr_acc = _band_limited_noise(n_samples, fs, tremor_lo, tremor_hi, rng) * tremor_accel_amp
    tr_gyr = _band_limited_noise(n_samples, fs, tremor_lo, tremor_hi, rng) * (tremor_amp_g * 0.4)

    # ax (forward): 1x gait, damped during fog, plus freeze-band + tremor + noise.
    signal[:, 0] = (
        gait_gain * GAIT_AX_AMP * gait_1x
        + fb_ax + tr_acc
        + rng.standard_normal(n_samples).astype(np.float32) * NOISE_ACCEL_STD
    )
    # ay (lateral): 1x gait at 90° phase.
    signal[:, 1] = (
        gait_gain * GAIT_AY_AMP * np.cos(2 * np.pi * gait_freq * t + phi)
        + fb_ay + tr_acc * 0.7
        + rng.standard_normal(n_samples).astype(np.float32) * NOISE_ACCEL_STD
    )
    # az (vertical): 2x heel-strike harmonic + gravity baseline.
    signal[:, 2] = (
        GRAVITY_AZ
        + gait_gain * GAIT_AZ_AMP * 0.5 * gait_2x
        + fb_az + tr_acc * 0.5
        + rng.standard_normal(n_samples).astype(np.float32) * NOISE_ACCEL_STD
    )
    # Gyro channels — 1x gait phase, smaller amplitudes.
    signal[:, 3] = (
        gait_gain * GAIT_GYRO_AMP * 0.5 * gait_1x
        + fb_gx + tr_gyr
        + rng.standard_normal(n_samples).astype(np.float32) * NOISE_GYRO_STD
    )
    signal[:, 4] = (
        gait_gain * GAIT_GYRO_AMP * 0.7 * np.cos(2 * np.pi * gait_freq * t + phi)
        + fb_gy + tr_gyr * 0.7
        + rng.standard_normal(n_samples).astype(np.float32) * NOISE_GYRO_STD
    )
    signal[:, 5] = (
        gait_gain * GAIT_GYRO_AMP * 0.4 * gait_1x
        + fb_gz + tr_gyr * 0.5
        + rng.standard_normal(n_samples).astype(np.float32) * NOISE_GYRO_STD
    )

    # Sample-level binary labels — anything past the cross-fade midpoint counts as FoG.
    labels = (fog_frac >= 0.5).astype(np.int64)
    return signal, labels


def freeze_index(linear_acc_window: np.ndarray, fs: float = 64.0) -> float:
    """Moore-Bachlin Freeze Index: power(3–8 Hz) / power(0.5–3 Hz), summed across axes.

    Per-axis-then-sum avoids the frequency-doubling that magnitude-rectification causes.
    """
    from scipy.signal import welch

    f, p = welch(linear_acc_window, fs=fs, nperseg=min(64, len(linear_acc_window)), axis=0)
    loco = float(p[(f >= 0.5) & (f <= 3.0)].sum())
    freeze = float(p[(f >= 3.0) & (f <= 8.0)].sum())
    return freeze / (loco + 1e-6)


def per_subject_summary(signal: np.ndarray, labels: np.ndarray, fs: float = 64.0) -> dict[str, Any]:
    """Compact stats dict used as Gemini-prompt context for the QC critique."""
    fog_mask = labels.astype(bool)
    walk_mask = ~fog_mask
    out: dict[str, Any] = {
        "n_samples": int(signal.shape[0]),
        "duration_s": float(signal.shape[0] / fs),
        "fog_fraction": float(fog_mask.mean()),
        "n_fog_segments": int(_count_runs(labels, 1)),
        "channel_means": signal.mean(axis=0).round(3).tolist(),
        "channel_stds": signal.std(axis=0).round(3).tolist(),
    }
    if walk_mask.any():
        out["walk_freeze_index"] = round(freeze_index(signal[walk_mask, :3], fs), 4)
    if fog_mask.any():
        out["fog_freeze_index"] = round(freeze_index(signal[fog_mask, :3], fs), 4)
    return out


def _count_runs(arr: np.ndarray, value: int) -> int:
    """Number of contiguous runs of `value` in a 1-D integer array."""
    arr = np.asarray(arr).astype(np.int64)
    if arr.size == 0:
        return 0
    edges = np.diff((arr == value).astype(np.int8))
    starts = int((edges == 1).sum()) + (1 if arr[0] == value else 0)
    return starts

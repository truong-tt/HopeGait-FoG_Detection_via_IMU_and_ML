"""Streaming post-processing for FoG probabilities.

Two stages, both causal so they're MCU-portable:

1. `smooth_probs` — boxcar moving average over `window` consecutive predictions.
   Knocks down single-window flicker that the raw classifier produces near the
   decision boundary. Larger `window` -> smoother but laggier.

2. `apply_hysteresis` — dual-threshold Schmitt trigger. Enter the FREEZE state
   only when smoothed prob crosses `high`; leave only when it drops below
   `low`. Without hysteresis a probability that hovers around a single
   threshold produces alternating 0/1 predictions; in the wild this would
   pulse the cueing belt on and off uselessly.

Both functions accept a 1D array of probabilities for a single recording (in
chronological order). For multi-recording evaluation, call them once per
recording so the state machine never crosses a recording boundary.
"""

from __future__ import annotations

import numpy as np


def smooth_probs(probs: np.ndarray, window: int = 5) -> np.ndarray:
    """Causal boxcar moving average. Output length matches input."""
    p = np.asarray(probs, dtype=np.float64)
    if window <= 1 or p.size == 0:
        return p.astype(np.float64)
    # Cumulative sum trick: smoothed[i] = mean(p[max(0, i-window+1) : i+1]).
    cs = np.concatenate(([0.0], np.cumsum(p)))
    idx = np.arange(p.size)
    lo = np.maximum(0, idx - window + 1)
    counts = (idx - lo + 1).astype(np.float64)
    return (cs[idx + 1] - cs[lo]) / counts


def apply_hysteresis(probs: np.ndarray, low: float = 0.4, high: float = 0.6,
                     initial_state: int = 0) -> np.ndarray:
    """Dual-threshold gate -> binary decisions.

    Args:
        probs: 1D array of (already smoothed) probabilities.
        low: drop below this to leave FREEZE.
        high: rise to/above this to enter FREEZE.
        initial_state: state before the first sample (0 = walking, 1 = freeze).
    Returns:
        int64 array of the same length, values in {0, 1}.
    """
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError(f"Invalid thresholds low={low} high={high}.")
    p = np.asarray(probs, dtype=np.float64)
    out = np.empty_like(p, dtype=np.int64)
    state = int(initial_state)
    for i, v in enumerate(p):
        if state == 0 and v >= high:
            state = 1
        elif state == 1 and v < low:
            state = 0
        out[i] = state
    return out


def postprocess_predictions(probs: np.ndarray, threshold: float,
                            smooth_window: int = 5, hysteresis_band: float = 0.1):
    """Convenience wrapper: smooth, then hysteresis around a base threshold.

    `threshold` is the per-fold operating point (e.g. Youden's J on inner val).
    The hysteresis band straddles it: high = threshold + band/2, low - band/2.
    """
    p_smooth = smooth_probs(probs, window=smooth_window)
    half = hysteresis_band / 2.0
    high = float(np.clip(threshold + half, 0.0, 1.0))
    low = float(np.clip(threshold - half, 0.0, 1.0))
    if high < low:
        high, low = low, high
    decisions = apply_hysteresis(p_smooth, low=low, high=high)
    return decisions, p_smooth

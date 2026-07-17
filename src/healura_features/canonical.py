"""
Canonical window contract: rate normalisation and the shared window constants.

Imports: numpy only. No sklearn/neurokit/pandas — this module must stay cheap
enough for the firmware/backend side to mirror.

Scope discipline
----------------
This module normalises RATE only. It never touches units.
Callers must supply EDA already in microsiemens (µS) and ACC already in g.
Unit conversion belongs at the point of ingestion (see datasets/loaders.py for
the WESAD counts→g conversion), not here — a function that silently rescaled
its input would reintroduce exactly the train/serve skew this module exists to
prevent.

The canonical window
--------------------
Every feature vector in feature_pipeline_version v1 is computed over a window of
WINDOW_SAMPLES (240) samples spanning WINDOW_SECONDS (60) seconds — i.e. both
EDA and ACC are carried at CANONICAL_EDA_HZ / CANONICAL_ACC_HZ (4 Hz) by the
time extract_all_features() sees them. This is the rate the v1 population model
was trained at: datasets/loaders.py decimates WESAD's 32 Hz ACC to EDA length
before windowing, so the model has only ever seen 240-row ACC windows.
"""

from typing import Tuple

import numpy as np

# ── The canonical window — single source of truth ─────────────────────────────
# These constants replace the previously unlinked literals in
# features/windows.py and models/predict/score.py.
WINDOW_SECONDS = 60
STEP_SECONDS = 10

CANONICAL_EDA_HZ = 4
CANONICAL_ACC_HZ = 4

WINDOW_SAMPLES = WINDOW_SECONDS * CANONICAL_EDA_HZ   # 240
STEP_SAMPLES = STEP_SECONDS * CANONICAL_EDA_HZ       # 40

# PROVISIONAL — how far a raw window's duration may stray from WINDOW_SECONDS
# and still be treated as a 60 s window. Device delivery jitter is real (a 60 s
# buffer at 4 Hz may arrive as 239 or 241 samples), but a 2x-off duration means
# the caller is windowing to a different contract. Tighten once pilot data
# shows the true delivery spread.
DURATION_TOLERANCE = 0.10


def implied_duration_seconds(n_samples: int, hz: float) -> float:
    """Duration in seconds that n_samples at hz represents."""
    if hz <= 0:
        raise ValueError(f"Sample rate must be positive, got {hz}")
    return n_samples / float(hz)


def duration_is_plausible(n_samples: int, hz: float) -> bool:
    """
    True if n_samples at hz spans WINDOW_SECONDS within DURATION_TOLERANCE.

    This is a duration check, not a sample-count check: 900 samples at 15 Hz and
    240 samples at 4 Hz both describe a valid 60 s window, and the old
    `len(eda) < 240` floor could not tell them apart.
    """
    duration = implied_duration_seconds(n_samples, hz)
    lo = WINDOW_SECONDS * (1.0 - DURATION_TOLERANCE)
    hi = WINDOW_SECONDS * (1.0 + DURATION_TOLERANCE)
    return lo <= duration <= hi


def resample_to_length(sig: np.ndarray, n: int) -> np.ndarray:
    """
    Resample a signal to exactly n samples by nearest-index selection.

    Deliberately identical to the method used by the dataset loaders when they
    decimated WESAD's 32 Hz ACC to EDA length:
        np.round(np.linspace(0, len(sig) - 1, n)).astype(int)
    The training features were produced by that exact arithmetic, so serving
    must reproduce it bit-for-bit. Any "better" resampler (polyphase, FFT,
    interpolation) would be a silent contract change. datasets/loaders.py now
    calls this function rather than repeating the expression.

    Parameters
    ----------
    sig : np.ndarray, shape (m,) or (m, k)
        1-D (EDA) or 2-D (ACC, m x 3). 2-D input is resampled along axis 0.
    n : int
        Target number of samples. len(sig) == n is a no-op (linspace yields the
        identity index) and is not special-cased.

    Returns
    -------
    np.ndarray with the same dimensionality as sig, first axis of length n.
    """
    sig = np.asarray(sig)

    if n <= 0:
        raise ValueError(f"Target length must be positive, got {n}")
    if sig.ndim not in (1, 2):
        raise ValueError(f"Expected a 1-D or 2-D signal, got ndim={sig.ndim}")
    if len(sig) == 0:
        raise ValueError("Cannot resample an empty signal")

    idx = np.round(np.linspace(0, len(sig) - 1, n)).astype(int)
    return sig[idx]


def to_canonical(
    eda: np.ndarray,
    acc: np.ndarray,
    *,
    eda_hz: float,
    acc_hz: float,
    n: int = WINDOW_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample one raw device window to the canonical (n,) EDA / (n, 3) ACC shape.

    RATE ONLY — units are the caller's responsibility. eda must already be in µS
    and acc must already be in g; passing ACC in raw E4 counts here will produce
    a correctly-shaped, physically wrong window (signal_quality_gate() is what
    catches that at serving time).

    Parameters
    ----------
    eda : array-like, shape (m,)      — EDA in µS at eda_hz
    acc : array-like, shape (m2, 3)   — ACC in g at acc_hz
    eda_hz, acc_hz : float
        The rate the caller's raw arrays are sampled at. Used to verify each
        array spans ~WINDOW_SECONDS; a canonical caller passes 4 / 4 and the
        resample is a no-op.
    n : int
        Target sample count. Defaults to WINDOW_SAMPLES (240).

    Returns
    -------
    (eda_canon, acc_canon) — shapes exactly (n,) and (n, 3), dtype float64.

    Raises
    ------
    ValueError
        If acc is not 2-D with 3 columns, or if either array's implied duration
        is outside WINDOW_SECONDS ± DURATION_TOLERANCE. Raising rather than
        silently stretching is the point: resampling a 25 s buffer up to 240
        samples would manufacture a 60 s window that never happened.
    """
    eda_arr = np.asarray(eda, dtype=float)
    acc_arr = np.asarray(acc, dtype=float)

    if eda_arr.ndim != 1:
        raise ValueError(f"eda must be 1-D, got shape {eda_arr.shape}")
    if acc_arr.ndim != 2 or acc_arr.shape[1] != 3:
        raise ValueError(f"acc must be 2-D with 3 columns, got shape {acc_arr.shape}")

    if not duration_is_plausible(len(eda_arr), eda_hz):
        raise ValueError(
            f"EDA window spans {implied_duration_seconds(len(eda_arr), eda_hz):.2f}s "
            f"({len(eda_arr)} samples @ {eda_hz} Hz); expected "
            f"{WINDOW_SECONDS}s ±{DURATION_TOLERANCE:.0%}"
        )
    if not duration_is_plausible(len(acc_arr), acc_hz):
        raise ValueError(
            f"ACC window spans {implied_duration_seconds(len(acc_arr), acc_hz):.2f}s "
            f"({len(acc_arr)} samples @ {acc_hz} Hz); expected "
            f"{WINDOW_SECONDS}s ±{DURATION_TOLERANCE:.0%}"
        )

    eda_canon = resample_to_length(eda_arr, n)
    acc_canon = resample_to_length(acc_arr, n)

    return eda_canon, acc_canon

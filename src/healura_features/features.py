"""
Feature extraction from cleaned E4 signals.
Produces one feature vector per window.

EDA features: mean, std, slope, SCR peak rate, tonic level, phasic amplitude
HRV features (from BVP): RMSSD, SDNN, LF/HF ratio, mean HR
ACC features: magnitude mean, magnitude std, movement variance
"""

import logging

import numpy as np
import neurokit2 as nk
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
import warnings

logger = logging.getLogger(__name__)

# ── SCR peak criterion (v3) — ABSOLUTE, not relative ─────────────────────────
# These two values are LOAD BEARING. They are named in the find_peaks call
# below, never inherited from a library default, and they are declared verbatim
# in feature_spec_v1.yaml's extraction: string for scr_peak_rate.
#
# WHY THEY EXIST:
# v2 counted peaks from nk.eda_process's own SCR_Peaks column. That detector's
# amplitude_min default of 0.1 is RELATIVE — 10% of the window maximum — and is
# not reachable through eda_process at all. On a flat window the window maximum
# is small, so the floor collapses toward zero and the detector fires on
# baseline noise: 27 peaks/min on flat traces, and the feature confidently
# INVERTED in 13 of 15 subjects (calm windows carried more counted "peaks" than
# stressed ones). At 4 Hz an SCR rise of 1-3 s is only 4-12 samples, so a
# relative criterion has no chance of separating a real sympathetic burst from
# wander.
#
# THE RESULT THIS CRITERION IS JUSTIFIED BY IS DIRECTION, NOT PERFORMANCE:
# an absolute microsiemens floor leaves the feature inverted in 0 of 15
# subjects, against 13 of 15 under the relative floor. The floor is in µS
# because a sympathetic response has a physical amplitude, so it does not scale
# with whatever else happened to be in the window.
#
# The held-out LOW-movement AUROC figures that accompany that result, 0.0915
# under v2 and 0.9829 under v3, are recorded here ONLY as the direction
# diagnostic they were computed as: the informative thing about them is that one
# sits below chance and the other above it, on the same windows. NEITHER IS A
# HEADLINE METRIC FOR THIS PACKAGE OR FOR ANY MODEL BUILT ON IT. healura_ml's
# docs/ML_CONVENTIONS.md is binding on both repositories and states it plainly:
# primary metrics are PR-AUC and recall-at-fixed-precision, never accuracy or
# AUROC as the headline number. Quote the PR-AUC and recall-at-fixed-precision
# figures from healura_ml's v3 candidate evaluation instead, and do not lift
# either AUROC out of this comment into a model card, a report or a docstring.
#
# The criterion's own derivation, including the within-amplitude-stratum
# evidence that v3 is not the same amplitude proxy with its sign corrected, is
# healura_ml's docs/SCR_V3_CRITERION.md. This package states the constants; it
# does not restate the argument for them.
#
# DO NOT reintroduce a relative threshold anywhere in this path. That is the
# defect this criterion exists to remove.
SCR_MIN_PROMINENCE_MICROSIEMENS = 0.02
SCR_MIN_INTERPEAK_INTERVAL_SECONDS = 3.0

EDA_FEATURE_KEYS = [
    "eda_tonic_mean", "eda_tonic_std", "eda_phasic_mean",
    "eda_phasic_max", "scr_peak_rate", "eda_mean",
    "eda_std", "eda_slope",
]

HRV_FEATURE_KEYS = ["hrv_rmssd", "hrv_sdnn", "hrv_mean_hr", "hrv_lf_hf"]

ACC_FEATURE_KEYS = [
    "acc_magnitude_mean", "acc_magnitude_std", "acc_magnitude_max",
    "acc_x_std", "acc_y_std", "acc_z_std",
]


def extract_eda_features(
    eda_window: np.ndarray,
    sampling_rate: int = 4
) -> dict:
    """Extract EDA features from one window."""
    features = {}

    try:
        # Process EDA — separates tonic (SCL) and phasic (SCR) components
        eda_signals, info = nk.eda_process(eda_window, sampling_rate=sampling_rate)

        # Tonic (baseline skin conductance level)
        features["eda_tonic_mean"] = eda_signals["EDA_Tonic"].mean()
        # ddof=0 — EDA_Tonic is a pandas Series, whose .std() defaults to ddof=1
        # (sample std) while every other std in this module is numpy's ddof=0
        # (population std). Pinning ddof=0 here makes all six stds in the frozen
        # 14 use the same estimator.
        features["eda_tonic_std"] = eda_signals["EDA_Tonic"].std(ddof=0)

        # Phasic (stress responses)
        features["eda_phasic_mean"] = eda_signals["EDA_Phasic"].mean()
        features["eda_phasic_max"] = eda_signals["EDA_Phasic"].max()

        # SCR peak rate (number of stress responses per minute).
        #
        # Peaks are detected on the SAME EDA_Phasic series eda_process already
        # produced above (clean -> phasic decomposition is NeuroKit's and is
        # unchanged); only the peak CRITERION applied to it differs. The
        # eda_process SCR_Peaks column is deliberately not read: its amplitude
        # floor is relative to the window maximum and cannot be overridden
        # through eda_process. See the constants at module level.
        #
        # Both parameters are passed explicitly. distance is in SAMPLES, so the
        # 3-second interval is converted at the window's own sampling rate
        # rather than assuming 4 Hz.
        phasic = eda_signals["EDA_Phasic"].values
        min_interpeak_samples = max(
            1, int(round(SCR_MIN_INTERPEAK_INTERVAL_SECONDS * sampling_rate))
        )
        scr_peaks, _ = find_peaks(
            phasic,
            prominence=SCR_MIN_PROMINENCE_MICROSIEMENS,
            distance=min_interpeak_samples,
        )
        n_peaks = len(scr_peaks)
        window_minutes = len(eda_window) / sampling_rate / 60
        features["scr_peak_rate"] = n_peaks / max(window_minutes, 0.01)

        # Raw EDA stats
        features["eda_mean"] = eda_window.mean()
        features["eda_std"] = eda_window.std()
        features["eda_slope"] = stats.linregress(
            np.arange(len(eda_window)), eda_window
        ).slope

    except Exception as e:
        # NaN signals a FAILED extraction to the caller — it is not a value.
        # The gate check in models/predict/score.py turns this into
        # gate_passed=False / reason="extraction_failed", and the training
        # assembler drops the row. Never silently swallow: a window that stops
        # producing features is a defect someone must see.
        logger.warning(
            "extract_eda_features failed (n=%d, sampling_rate=%s): %s: %s "
            "— returning NaN for all %d EDA features",
            len(eda_window) if eda_window is not None else -1,
            sampling_rate, type(e).__name__, e, len(EDA_FEATURE_KEYS),
        )
        for k in EDA_FEATURE_KEYS:
            features[k] = np.nan

    return features


def extract_hrv_features(
    bvp_window: np.ndarray,
    sampling_rate: int = 64
) -> dict:
    """Extract HRV features from one BVP window."""
    features = {}

    try:
        ppg_signals, info = nk.ppg_process(bvp_window, sampling_rate=sampling_rate)
        hrv = nk.hrv(ppg_signals, sampling_rate=sampling_rate, show=False)

        features["hrv_rmssd"] = hrv["HRV_RMSSD"].values[0]
        features["hrv_sdnn"] = hrv["HRV_SDNN"].values[0]
        features["hrv_mean_hr"] = hrv["HRV_MeanNN"].values[0]
        features["hrv_lf_hf"] = hrv.get("HRV_LFHF", pd.Series([np.nan])).values[0]

    except Exception as e:
        logger.warning(
            "extract_hrv_features failed (n=%d, sampling_rate=%s): %s: %s "
            "— returning NaN for all %d HRV features",
            len(bvp_window) if bvp_window is not None else -1,
            sampling_rate, type(e).__name__, e, len(HRV_FEATURE_KEYS),
        )
        for k in HRV_FEATURE_KEYS:
            features[k] = np.nan

    return features


def extract_acc_features(acc_window: np.ndarray) -> dict:
    """
    Extract movement features from one ACC window (shape: n x 3), in g.

    Units: acc_window must already be in g — see datasets/loaders.py for the
    WESAD counts→g conversion. These features are plain statistics with no time
    term, so they carry no sample-rate assumption, but they ARE calibrated to
    the g scale the model was trained on.
    """

    # Guard against empty or malformed arrays.
    # WHY HERE NOT JUST IN windows.py:
    # Defensive programming — features.py should never crash regardless
    # of what calls it. Belt-and-suspenders approach for a function
    # that will eventually run in real-time production code.
    # The NaN is a failure signal, not a value: callers must check for it
    # (score.py gates on it; extract.py drops the row). It is logged so a
    # malformed payload is never invisible.
    if acc_window is None or len(acc_window) == 0:
        logger.warning(
            "extract_acc_features: acc_window is None or empty — "
            "returning NaN for all %d ACC features", len(ACC_FEATURE_KEYS),
        )
        return {k: np.nan for k in ACC_FEATURE_KEYS}

    # Also guard against 1D array being passed instead of (n, 3)
    if acc_window.ndim != 2 or acc_window.shape[1] != 3:
        logger.warning(
            "extract_acc_features: expected 2-D (n, 3) ACC, got shape %s — "
            "returning NaN for all %d ACC features",
            (acc_window.shape,), len(ACC_FEATURE_KEYS),
        )
        return {k: np.nan for k in ACC_FEATURE_KEYS}

    magnitude = np.sqrt(np.sum(acc_window ** 2, axis=1))
    return {
        "acc_magnitude_mean": magnitude.mean(),
        "acc_magnitude_std":  magnitude.std(),
        "acc_magnitude_max":  magnitude.max(),
        "acc_x_std":          acc_window[:, 0].std(),
        "acc_y_std":          acc_window[:, 1].std(),
        "acc_z_std":          acc_window[:, 2].std(),
    }


def extract_all_features(
    eda_window: np.ndarray,
    bvp_window: np.ndarray = None,
    acc_window: np.ndarray = None,
    eda_hz: int = 4,
    bvp_hz: int = 64
) -> dict:
    """
    Master function: extract all features from one window.
    Returns a flat dict of feature_name -> value.
    """
    features = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # suppress NeuroKit low-frequency warning
        features.update(extract_eda_features(eda_window, sampling_rate=eda_hz))
    if bvp_window is not None:
        features.update(extract_hrv_features(bvp_window, sampling_rate=bvp_hz))
    if acc_window is not None:
        features.update(extract_acc_features(acc_window))
    return features
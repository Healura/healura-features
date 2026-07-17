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
import warnings

logger = logging.getLogger(__name__)

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

        # SCR peak rate (number of stress responses per minute)
        n_peaks = eda_signals["SCR_Peaks"].sum()
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
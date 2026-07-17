"""
Unit tests for feature extraction and the feature contract (M1).

Synthetic window: rng seed=42, EDA = sine + noise, ACC = normal noise.
All assertions use concrete expected values — a test that passes without
asserting the computed value is a defect (per CLAUDE.md hard rules).
"""

import math
import warnings

import numpy as np
import pytest

from healura_features.contract import load_spec, vectorize
from healura_features.features import extract_all_features
from fixture_gen import make_golden_window

# ── Shared spec fixture ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def spec():
    return load_spec()


# ── Deterministic synthetic windows (seed=42) ─────────────────────────────────
# The generator lives in tests/fixture_gen.py and is shared with
# scripts/make_fixtures.py, so these arrays are byte-identical to the committed
# tests/fixtures/golden_window.npz that test_parity.py checks against.
#
# EDA: 240 samples (60 s at 4 Hz) in MICROSIEMENS — nk.eda_simulate, a
#      physiological trace with 4 SCRs (≈4 peaks/min). The previous fixture was
#      a 2 Hz sine that produced 51 SCRs/min, which no wrist ever reads.
# ACC: 240 × 3 at 4 Hz in G — ~1 g of gravity on the z axis plus small movement,
#      mean magnitude ≈ 1.0. Units matter: the previous fixture was zero-mean
#      unit-normal noise with no gravity, which is not accelerometer data in any
#      unit and would now be refused by signal_quality_gate.

@pytest.fixture(scope="module")
def synthetic_windows():
    return make_golden_window()


@pytest.fixture(scope="module")
def synthetic_features(synthetic_windows, spec):
    eda, acc = synthetic_windows
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return extract_all_features(eda, bvp_window=None, acc_window=acc)


# ── TASK 3a: 14 keys present, all float / NaN (never a crash) ────────────────

EXPECTED_KEYS = {
    "eda_mean", "eda_phasic_max", "eda_phasic_mean", "eda_slope",
    "eda_std", "eda_tonic_mean", "eda_tonic_std", "scr_peak_rate",
    "acc_magnitude_max", "acc_magnitude_mean", "acc_magnitude_std",
    "acc_x_std", "acc_y_std", "acc_z_std",
}


def test_expected_keys_present(synthetic_features):
    assert set(synthetic_features.keys()) == EXPECTED_KEYS


def test_all_values_are_float_or_nan(synthetic_features):
    for k, v in synthetic_features.items():
        assert isinstance(v, float) or (isinstance(v, np.floating)), \
            f"{k} is {type(v)}, expected float"
        # Either a real finite float or NaN — never a string, None, or exception
        assert math.isfinite(v) or math.isnan(v), f"{k} = {v} is not finite or NaN"


# ── TASK 3b: vectorize returns length-14 array in EDA-first order ─────────────

def test_vectorize_length_and_dtype(synthetic_features, spec):
    vec = vectorize(synthetic_features, spec=spec)
    assert vec.shape == (14,), f"expected shape (14,), got {vec.shape}"
    assert vec.dtype == np.float64, f"expected float64, got {vec.dtype}"


def test_vectorize_position_0_is_eda_mean(synthetic_features, spec):
    vec = vectorize(synthetic_features, spec=spec)
    # Position 0 must be eda_mean — EDA-first ordering
    assert spec["frozen_order"][0] == "eda_mean", "frozen_order[0] is not eda_mean"
    assert abs(vec[0] - 0.777564188538008) < 1e-6, \
        f"vec[0] (eda_mean) = {vec[0]}, expected ~0.77756"


def test_vectorize_position_13_is_acc_z_std(synthetic_features, spec):
    vec = vectorize(synthetic_features, spec=spec)
    # Position 13 must be acc_z_std — last ACC feature
    assert spec["frozen_order"][13] == "acc_z_std", "frozen_order[13] is not acc_z_std"
    assert abs(vec[13] - 0.030306332704056617) < 1e-6, \
        f"vec[13] (acc_z_std) = {vec[13]}, expected ~0.03031"


def test_golden_acc_is_in_g_with_gravity(synthetic_windows):
    """
    The golden ACC must be physically plausible g-unit data: mean per-sample
    magnitude ≈ 1 g from gravity. This is the property the old fixture lacked,
    and the reason a units bug could hide behind a green parity test.
    """
    _, acc = synthetic_windows
    mean_mag = float(np.mean(np.sqrt(np.sum(acc ** 2, axis=1))))
    assert 0.8 <= mean_mag <= 1.2, (
        f"golden ACC mean magnitude = {mean_mag:.4f} g; expected ~1.0 g "
        "(gravity). ACC fixtures without gravity are not accelerometer data."
    )


def test_golden_scr_peak_rate_is_physiological(synthetic_features):
    """
    scr_peak_rate must sit in a plausible 1–10 peaks/min band. The previous
    fixture read 51/min — the peak detector was being exercised in a regime
    real data never reaches, so the fixture could not catch a real regression.
    """
    scr = synthetic_features["scr_peak_rate"]
    assert 1.0 <= scr <= 10.0, (
        f"golden scr_peak_rate = {scr} /min, outside the physiological band "
        "[1, 10]. Adjust GOLDEN_SCR_NUMBER in tests/fixture_gen.py."
    )


def test_vectorize_eda_block_positions(spec):
    # Positions 0-7 must all be EDA features
    eda_names = spec["frozen_order"][:8]
    assert all(n.startswith("eda_") or n == "scr_peak_rate" for n in eda_names), \
        f"Positions 0-7 are not all EDA features: {eda_names}"


def test_vectorize_acc_block_positions(spec):
    # Positions 8-13 must all be ACC features
    acc_names = spec["frozen_order"][8:]
    assert all(n.startswith("acc_") for n in acc_names), \
        f"Positions 8-13 are not all ACC features: {acc_names}"


# ── TASK 3c: NaN-path — flat zero EDA, assert NaN→0 substitution ─────────────

def test_zero_eda_does_not_raise(spec):
    zeros_eda = np.zeros(240)
    zeros_acc = np.zeros((240, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feat = extract_all_features(zeros_eda, bvp_window=None, acc_window=zeros_acc)
    assert feat is not None
    assert set(feat.keys()) == EXPECTED_KEYS


def test_zero_eda_eda_features_are_nan(spec):
    zeros_eda = np.zeros(240)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feat = extract_all_features(zeros_eda, bvp_window=None, acc_window=None)
    eda_keys = ["eda_mean", "eda_phasic_max", "eda_phasic_mean", "eda_slope",
                "eda_std", "eda_tonic_mean", "eda_tonic_std", "scr_peak_rate"]
    for k in eda_keys:
        assert math.isnan(feat[k]), f"{k} should be NaN for flat-zero EDA, got {feat[k]}"


def test_vectorize_raises_on_nan_by_default(spec):
    """
    A failed extraction must NOT be silently imputed to zero.

    Zero is not a neutral value: StandardScaler is fit on raw features, so an
    imputed 0.0 for eda_mean (training mean ≈ 0.4–2 µS) is rescaled to
    (0 − µ)/σ — an extreme point. The old default (nan_to_zero=True) therefore
    turned an unmeasurable window into a confidently-wrong score. vectorize now
    refuses, and names every offending feature so the cause is diagnosable.
    """
    zeros_eda = np.zeros(240)
    zeros_acc = np.zeros((240, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feat = extract_all_features(zeros_eda, bvp_window=None, acc_window=zeros_acc)

    with pytest.raises(ValueError) as exc_info:
        vectorize(feat, spec=spec)          # default is now nan_to_zero=False

    msg = str(exc_info.value)
    assert "NaN" in msg, f"error must say the features are NaN, got: {msg}"
    # Every failed EDA feature must be named, not just counted.
    for name in ["eda_mean", "eda_std", "eda_slope", "eda_tonic_mean",
                 "eda_tonic_std", "eda_phasic_mean", "eda_phasic_max",
                 "scr_peak_rate"]:
        assert name in msg, f"error must name the NaN feature '{name}', got: {msg}"


def test_vectorize_nan_to_zero_true_is_still_available_for_debugging(spec):
    """
    nan_to_zero=True remains as an explicit debugging escape hatch. It must be
    opt-in only — no scoring or training path may pass it.
    """
    zeros_eda = np.zeros(240)
    zeros_acc = np.zeros((240, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feat = extract_all_features(zeros_eda, bvp_window=None, acc_window=zeros_acc)

    vec = vectorize(feat, spec=spec, nan_to_zero=True)
    for i in range(8):
        assert vec[i] == 0.0, \
            f"Position {i} ({spec['frozen_order'][i]}) should be 0.0 with nan_to_zero=True, got {vec[i]}"
    for i in range(8, 14):
        assert vec[i] == 0.0, \
            f"Position {i} ({spec['frozen_order'][i]}) should be 0.0 for zero ACC, got {vec[i]}"


def test_vectorize_nan_error_names_only_the_failed_features(spec):
    """A partial failure must name only the features that actually failed."""
    zeros_eda = np.zeros(240)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feat = extract_all_features(zeros_eda, bvp_window=None, acc_window=None)

    # Supply valid ACC features so only the EDA block is NaN.
    for k in ["acc_magnitude_max", "acc_magnitude_mean", "acc_magnitude_std",
              "acc_x_std", "acc_y_std", "acc_z_std"]:
        feat[k] = 0.5

    with pytest.raises(ValueError) as exc_info:
        vectorize(feat, spec=spec)

    msg = str(exc_info.value)
    assert "eda_mean" in msg
    assert "acc_x_std" not in msg, (
        f"acc_x_std is not NaN and must not be named as failed. Got: {msg}"
    )


# ── TASK 3d: missing-feature raises ValueError ────────────────────────────────

def test_vectorize_empty_dict_raises(spec):
    with pytest.raises(ValueError) as exc_info:
        vectorize({}, spec=spec)
    assert "missing from input" in str(exc_info.value)
    # Error must list the offending keys — not just say "something went wrong"
    assert "eda_mean" in str(exc_info.value)


# ── TASK 3e: extra-feature raises ValueError ──────────────────────────────────

def test_vectorize_extra_key_raises(synthetic_features, spec):
    polluted = dict(synthetic_features)
    polluted["bogus"] = 1.0
    with pytest.raises(ValueError) as exc_info:
        vectorize(polluted, spec=spec)
    assert "extra in input" in str(exc_info.value)
    assert "bogus" in str(exc_info.value)


# ── TASK 3f: feature-count regression tripwire ────────────────────────────────

def test_frozen_order_count_is_14(spec):
    assert len(spec["frozen_order"]) == 14, \
        f"frozen_order has {len(spec['frozen_order'])} features, expected exactly 14. " \
        "If a feature was added/removed, bump feature_pipeline_version and update this test."


def test_frozen_order_exact_name_set(spec):
    expected_names = {
        "eda_mean", "eda_phasic_max", "eda_phasic_mean", "eda_slope",
        "eda_std", "eda_tonic_mean", "eda_tonic_std", "scr_peak_rate",
        "acc_magnitude_max", "acc_magnitude_mean", "acc_magnitude_std",
        "acc_x_std", "acc_y_std", "acc_z_std",
    }
    actual_names = set(spec["frozen_order"])
    added = actual_names - expected_names
    removed = expected_names - actual_names
    assert not added and not removed, (
        f"Feature set changed without bumping feature_pipeline_version. "
        f"Added: {added}. Removed: {removed}. "
        "Update this hardcoded set AND bump feature_pipeline_version."
    )

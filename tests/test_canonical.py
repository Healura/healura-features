"""
Canonical rate-normalisation tests (features/canonical.py).

The contract this pins down: a raw device window at ANY plausible rate becomes
exactly (240,) EDA and (240, 3) ACC, using the same nearest-index arithmetic the
training loaders used to decimate WESAD's 32 Hz ACC. If serving resampled
differently from training, every ACC feature would shift — silently, because the
vector would still be the right shape with the right names.

golden_resample.npz encodes a raw 15 Hz / 25 Hz window and its expected
canonical output, so the backend can verify its own resampler against ours.
"""

import io
import json
from importlib import resources

import numpy as np
import pytest

from healura_features.canonical import (
    CANONICAL_ACC_HZ,
    CANONICAL_EDA_HZ,
    DURATION_TOLERANCE,
    STEP_SAMPLES,
    WINDOW_SAMPLES,
    WINDOW_SECONDS,
    duration_is_plausible,
    resample_to_length,
    to_canonical,
)

# Resample golden fixtures ship as installed package data (see pyproject).
_FIXTURES = resources.files("healura_features").joinpath("fixtures")


@pytest.fixture(scope="module")
def golden_resample():
    return np.load(io.BytesIO(_FIXTURES.joinpath("golden_resample.npz").read_bytes()))


@pytest.fixture(scope="module")
def golden_resample_meta():
    return json.loads(_FIXTURES.joinpath("golden_resample.json").read_text())


# ── Constants ─────────────────────────────────────────────────────────────────

def test_window_samples_is_240():
    assert WINDOW_SAMPLES == 240
    assert WINDOW_SAMPLES == WINDOW_SECONDS * CANONICAL_EDA_HZ


def test_step_samples_is_40():
    assert STEP_SAMPLES == 40


def test_canonical_acc_hz_is_4_not_32():
    """
    The v1 model only ever saw ACC decimated to EDA length. acc_hz=32 described
    the sensor, not the contract.
    """
    assert CANONICAL_ACC_HZ == 4


# NOTE: healura_ml's test_canonical had a test asserting that features/windows.py
# and models/predict/score.py share WINDOW_SAMPLES/STEP_SAMPLES. That test imported
# `features.windows.create_windows` — a TRAINING-side module (batch windowing over
# full recordings) that is intentionally NOT part of this feature-layer package.
# It was the only import in the extracted tests that reached back into healura_ml,
# so it is dropped here; the shared window constants it guarded still live in and
# are exercised via healura_features.canonical (see test_window_samples_is_240 /
# test_step_samples_is_40 above).


# ── resample_to_length ────────────────────────────────────────────────────────

def test_resample_matches_loader_arithmetic():
    """
    Must be bit-identical to the expression the loaders used:
        np.round(np.linspace(0, len(sig) - 1, n)).astype(int)
    Any "better" resampler would silently change every trained feature.
    """
    sig = np.arange(1500, dtype=float)
    expected_idx = np.round(np.linspace(0, len(sig) - 1, WINDOW_SAMPLES)).astype(int)
    expected = sig[expected_idx]
    np.testing.assert_array_equal(resample_to_length(sig, WINDOW_SAMPLES), expected)


def test_resample_1d_shape():
    out = resample_to_length(np.arange(900, dtype=float), WINDOW_SAMPLES)
    assert out.shape == (WINDOW_SAMPLES,)


def test_resample_2d_shape():
    out = resample_to_length(np.zeros((1500, 3)), WINDOW_SAMPLES)
    assert out.shape == (WINDOW_SAMPLES, 3)


def test_resample_identity_when_already_canonical():
    """len(sig) == n must be an exact no-op, not an approximation."""
    sig = np.random.default_rng(0).normal(size=WINDOW_SAMPLES)
    np.testing.assert_array_equal(resample_to_length(sig, WINDOW_SAMPLES), sig)


def test_resample_is_deterministic():
    sig = np.random.default_rng(1).normal(size=907)
    a = resample_to_length(sig, WINDOW_SAMPLES)
    b = resample_to_length(sig, WINDOW_SAMPLES)
    np.testing.assert_array_equal(a, b)


def test_resample_empty_raises():
    with pytest.raises(ValueError):
        resample_to_length(np.array([]), WINDOW_SAMPLES)


def test_resample_bad_target_raises():
    with pytest.raises(ValueError):
        resample_to_length(np.arange(10, dtype=float), 0)


# ── duration_is_plausible ─────────────────────────────────────────────────────

def test_duration_plausible_canonical():
    assert duration_is_plausible(240, 4)


def test_duration_plausible_at_device_rate():
    """900 @ 15 Hz and 240 @ 4 Hz are both a valid 60 s window."""
    assert duration_is_plausible(900, 15)
    assert duration_is_plausible(1500, 25)


def test_duration_implausible_short():
    assert not duration_is_plausible(100, 4)      # 25 s


def test_duration_implausible_long():
    assert not duration_is_plausible(480, 4)      # 120 s


def test_duration_tolerance_boundary():
    """239 @ 4 Hz (59.75 s) is jitter, not a short window."""
    assert duration_is_plausible(239, 4)
    lo = int(WINDOW_SECONDS * (1 - DURATION_TOLERANCE) * CANONICAL_EDA_HZ)
    assert duration_is_plausible(lo, 4)


# ── to_canonical ──────────────────────────────────────────────────────────────

def test_to_canonical_reproduces_golden(golden_resample, golden_resample_meta):
    """The committed raw window must canonicalise to the committed output."""
    eda_canon, acc_canon = to_canonical(
        golden_resample["eda_raw"],
        golden_resample["acc_raw"],
        eda_hz=golden_resample_meta["eda_hz"],
        acc_hz=golden_resample_meta["acc_hz"],
    )
    np.testing.assert_allclose(eda_canon, golden_resample["eda_canon"], atol=1e-8)
    np.testing.assert_allclose(acc_canon, golden_resample["acc_canon"], atol=1e-8)


def test_golden_resample_raw_shapes(golden_resample):
    assert golden_resample["eda_raw"].shape == (900,), "EDA raw = 15 Hz x 60 s"
    assert golden_resample["acc_raw"].shape == (1500, 3), "ACC raw = 25 Hz x 60 s"


def test_to_canonical_output_shapes(golden_resample, golden_resample_meta):
    eda_canon, acc_canon = to_canonical(
        golden_resample["eda_raw"], golden_resample["acc_raw"],
        eda_hz=golden_resample_meta["eda_hz"],
        acc_hz=golden_resample_meta["acc_hz"],
    )
    assert eda_canon.shape == (WINDOW_SAMPLES,)
    assert acc_canon.shape == (WINDOW_SAMPLES, 3)


def test_to_canonical_is_noop_at_canonical_rates():
    """A caller already at 4 Hz must get its arrays back untouched."""
    rng = np.random.default_rng(3)
    eda = rng.normal(2.0, 0.3, WINDOW_SAMPLES)
    acc = np.column_stack([
        rng.normal(0, 0.03, WINDOW_SAMPLES),
        rng.normal(0, 0.03, WINDOW_SAMPLES),
        1.0 + rng.normal(0, 0.03, WINDOW_SAMPLES),
    ])
    eda_c, acc_c = to_canonical(eda, acc, eda_hz=4, acc_hz=4)
    np.testing.assert_array_equal(eda_c, eda)
    np.testing.assert_array_equal(acc_c, acc)


def test_to_canonical_is_deterministic(golden_resample, golden_resample_meta):
    args = dict(eda_hz=golden_resample_meta["eda_hz"],
                acc_hz=golden_resample_meta["acc_hz"])
    a_eda, a_acc = to_canonical(golden_resample["eda_raw"], golden_resample["acc_raw"], **args)
    b_eda, b_acc = to_canonical(golden_resample["eda_raw"], golden_resample["acc_raw"], **args)
    np.testing.assert_array_equal(a_eda, b_eda)
    np.testing.assert_array_equal(a_acc, b_acc)


def test_to_canonical_rejects_bad_acc_shape():
    with pytest.raises(ValueError, match="3 columns"):
        to_canonical(np.zeros(240), np.zeros((240, 2)), eda_hz=4, acc_hz=4)


def test_to_canonical_rejects_short_eda():
    """Upsampling a 25 s buffer to 240 would manufacture a window that never happened."""
    with pytest.raises(ValueError, match="EDA window spans"):
        to_canonical(np.zeros(100), np.zeros((240, 3)), eda_hz=4, acc_hz=4)


def test_to_canonical_rejects_wrong_duration_acc():
    with pytest.raises(ValueError, match="ACC window spans"):
        to_canonical(np.zeros(240), np.zeros((100, 3)), eda_hz=4, acc_hz=4)


def test_to_canonical_does_not_rescale_units():
    """
    to_canonical normalises RATE ONLY. If it silently rescaled units it would
    reintroduce exactly the skew it exists to prevent — so ACC in counts must
    come back out in counts (the gate is what refuses it, not this function).
    """
    rng = np.random.default_rng(5)
    acc_counts = np.column_stack([
        rng.normal(0, 2, WINDOW_SAMPLES),
        rng.normal(0, 2, WINDOW_SAMPLES),
        64.0 + rng.normal(0, 2, WINDOW_SAMPLES),
    ])
    _, acc_out = to_canonical(np.ones(240) * 2.0, acc_counts, eda_hz=4, acc_hz=4)
    mean_mag = float(np.mean(np.sqrt(np.sum(acc_out ** 2, axis=1))))
    assert mean_mag > 20, (
        f"to_canonical must not rescale units; mean|acc| came back {mean_mag:.2f}, "
        "expected ~64 (still counts)"
    )

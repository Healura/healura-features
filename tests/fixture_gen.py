"""
Golden-fixture generators — the single source of truth for both the test
fixtures and the committed .npz files.

WHY THIS MODULE EXISTS:
The golden window used to be generated inline in test_features.py while
golden_window.npz held a separately-produced copy of "the same" arrays. Two
generators for one fixture is the same class of drift bug this slice exists to
remove. scripts/make_fixtures.py and tests/test_features.py both import from
here, so the committed .npz and the in-test arrays cannot diverge.

Units and rates are explicit everywhere: EDA in µS, ACC in g.
"""

import numpy as np
import neurokit2 as nk

from healura_features.canonical import CANONICAL_EDA_HZ, WINDOW_SAMPLES, WINDOW_SECONDS

# Seed for every golden fixture. Changing it changes the fixtures and therefore
# the parity contract — don't, without regenerating and countersigning.
GOLDEN_SEED = 42

# scr_number=4 yields scr_peak_rate == 4.0 /min on this trace, inside the
# physiological 1–10 /min band asserted by test_features. The previous fixture
# (a 2 Hz sine + noise) produced 51 SCRs/min — an impossible rate that exercised
# NeuroKit's peak detector far outside the regime real data occupies.
GOLDEN_SCR_NUMBER = 4

# Raw device-rate fixture: deliberately NOT the canonical rates, so the
# resample path is actually exercised rather than short-circuited by a no-op.
RESAMPLE_EDA_HZ = 15
RESAMPLE_ACC_HZ = 25


def _make_acc_g(n: int, hz: float, seed: int) -> np.ndarray:
    """
    Synthetic wrist ACC in g, shape (n, 3), axis order [x, y, z].

    Physically plausible by construction: ~1 g of gravity resting on the z axis
    plus small movement on all three. Mean per-sample magnitude lands near 1.0,
    which is what models/predict/score.py::signal_quality_gate expects of real
    g-unit data — an ACC array of zero-mean noise with no gravity (what the old
    fixtures used) is not accelerometer data and is now correctly refused.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / float(hz)
    acc_x = rng.normal(0.0, 0.03, n) + 0.05 * np.sin(2 * np.pi * 0.25 * t)
    acc_y = rng.normal(0.0, 0.03, n)
    acc_z = 1.0 + rng.normal(0.0, 0.03, n)   # gravity
    return np.column_stack([acc_x, acc_y, acc_z])


def make_golden_window():
    """
    The canonical 60 s window: EDA (240,) in µS at 4 Hz, ACC (240, 3) in g.

    Returns
    -------
    (eda, acc) — shapes (WINDOW_SAMPLES,) and (WINDOW_SAMPLES, 3), float64.
    """
    eda = nk.eda_simulate(
        duration=WINDOW_SECONDS,
        sampling_rate=CANONICAL_EDA_HZ,
        scr_number=GOLDEN_SCR_NUMBER,
        random_state=GOLDEN_SEED,
    )
    eda = np.asarray(eda, dtype=float)
    acc = _make_acc_g(WINDOW_SAMPLES, CANONICAL_EDA_HZ, GOLDEN_SEED)
    return eda, acc


def make_resample_window():
    """
    A RAW device window at non-canonical rates, for the to_canonical contract.

    EDA: 900 samples  = 15 Hz x 60 s, µS
    ACC: 1500 x 3     = 25 Hz x 60 s, g

    Returns
    -------
    (eda_raw, acc_raw) — shapes (900,) and (1500, 3), float64.
    """
    n_eda = RESAMPLE_EDA_HZ * WINDOW_SECONDS    # 900
    n_acc = RESAMPLE_ACC_HZ * WINDOW_SECONDS    # 1500

    eda = nk.eda_simulate(
        duration=WINDOW_SECONDS,
        sampling_rate=RESAMPLE_EDA_HZ,
        scr_number=GOLDEN_SCR_NUMBER,
        random_state=GOLDEN_SEED,
    )
    eda = np.asarray(eda, dtype=float)
    assert len(eda) == n_eda, f"expected {n_eda} EDA samples, got {len(eda)}"

    acc = _make_acc_g(n_acc, RESAMPLE_ACC_HZ, GOLDEN_SEED)
    return eda, acc

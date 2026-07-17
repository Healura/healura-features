"""
Golden-vector parity test (M1 — cross-repo seam).

The backend imports features/ and runs this SAME fixture; if firmware/backend
compute features differently, this test goes red on their side.

Fixture files (tests/fixtures/):
    golden_window.npz — EDA + ACC arrays, rng seed=42 (see test_features.py)
    golden_vector.json — 14-value vector, keyed by feature name

If this test fails after any change to features/features.py, that change
altered the feature-extraction contract and MUST NOT land without:
  1. A new spec version (feature_pipeline_version bump)
  2. Backend + firmware countersign on the new spec
"""

import io
import json
import warnings
from importlib import resources

import numpy as np
import pytest

from healura_features.contract import load_spec, vectorize
from healura_features.features import extract_all_features

# The parity fixtures are loaded from the INSTALLED package data, not a relative
# path — this test is the acceptance gate on what actually ships in the wheel.
_FIXTURES = resources.files("healura_features").joinpath("fixtures")


@pytest.fixture(scope="module")
def golden_window():
    data = np.load(io.BytesIO(_FIXTURES.joinpath("golden_window.npz").read_bytes()))
    return data["eda"], data["acc"]


@pytest.fixture(scope="module")
def golden_vector():
    return json.loads(_FIXTURES.joinpath("golden_vector.json").read_text())


@pytest.fixture(scope="module")
def spec():
    return load_spec()


def test_parity_recomputed_matches_golden(golden_window, golden_vector, spec):
    eda, acc = golden_window
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feat = extract_all_features(eda, bvp_window=None, acc_window=acc)

    vec = vectorize(feat, spec=spec)   # nan_to_zero=False — a NaN here is a failure

    for i, name in enumerate(spec["frozen_order"]):
        expected = golden_vector[name]
        actual = vec[i]
        assert abs(actual - expected) <= 1e-8, (
            f"Feature '{name}' (position {i}) diverged from golden.\n"
            f"  expected : {expected}\n"
            f"  actual   : {actual}\n"
            f"  diff     : {abs(actual - expected)}\n"
            "This indicates a change to the feature-extraction contract. "
            "If intentional, re-run scripts/hash_spec.py and regenerate the fixture."
        )


def test_golden_vector_has_14_entries(golden_vector, spec):
    assert len(golden_vector) == 14, \
        f"golden_vector.json has {len(golden_vector)} entries, expected 14"
    assert set(golden_vector.keys()) == set(spec["frozen_order"]), \
        "golden_vector.json keys do not match frozen_order"

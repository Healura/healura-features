"""
spec_hash integrity + tripwire tests.

Two distinct jobs:

1. INTEGRITY — the hash stored in feature_spec_v1.yaml must actually be the
   hash of that file's content. Nothing asserted this before, so the stored
   value could silently drift from the spec it claimed to describe.

2. TRIPWIRE — the hash must MOVE when the contract's semantics move. Before
   this slice the hash covered names/order/dtypes only, so acc_hz could go from
   32 to 4, or acc_unit from counts to g, and every consumer verifying the hash
   would see no change while the feature values shifted underneath them. These
   tests fail if that hole is ever reopened.
"""

import copy

import pytest
import yaml

from healura_features.hash_spec import (
    HASHED_WINDOW_FIELDS,
    SPEC_PATH,
    canonical_string,
    compute_hash,
)


@pytest.fixture(scope="module")
def raw_spec():
    return yaml.safe_load(SPEC_PATH.read_text())


# ── 1. Integrity ──────────────────────────────────────────────────────────────

def test_stored_spec_hash_matches_content(raw_spec):
    recomputed = compute_hash(raw_spec)
    stored = raw_spec["spec_hash"]
    assert recomputed == stored, (
        f"feature_spec_v1.yaml spec_hash is stale.\n"
        f"  stored     : {stored}\n"
        f"  recomputed : {recomputed}\n"
        "The spec content changed without regenerating the hash. "
        "Run `python scripts/hash_spec.py`."
    )


def test_compute_hash_is_deterministic(raw_spec):
    assert compute_hash(raw_spec) == compute_hash(copy.deepcopy(raw_spec))


# ── 2. Tripwire: semantics must be hashed ─────────────────────────────────────

def test_changing_acc_hz_changes_hash(raw_spec):
    """
    acc_hz is the field that was WRONG (32, when the model was trained on 4).
    If it is not hashed, firmware can honour a stale rate while the hash says
    the contract is unchanged.
    """
    baseline = compute_hash(raw_spec)
    mutated = copy.deepcopy(raw_spec)
    mutated["window"]["acc_hz"] = 32

    assert compute_hash(mutated) != baseline, (
        "Changing window.acc_hz did NOT change spec_hash. The hash must cover "
        "sample-rate semantics — otherwise a consumer sending 1920-sample ACC "
        "windows verifies clean against a 240-sample contract."
    )


def test_changing_acc_unit_changes_hash(raw_spec):
    """
    acc_unit distinguishes g from raw E4 counts — a 64x difference in every ACC
    feature, invisible to a schema-only hash.
    """
    baseline = compute_hash(raw_spec)
    mutated = copy.deepcopy(raw_spec)
    mutated["window"]["acc_unit"] = "counts"

    assert compute_hash(mutated) != baseline, (
        "Changing window.acc_unit did NOT change spec_hash. The hash must cover "
        "the units contract — otherwise ACC in counts verifies clean against a "
        "contract that says g."
    )


def test_changing_eda_unit_changes_hash(raw_spec):
    baseline = compute_hash(raw_spec)
    mutated = copy.deepcopy(raw_spec)
    mutated["window"]["eda_unit"] = "nanosiemens"
    assert compute_hash(mutated) != baseline


@pytest.mark.parametrize("field,new_value", [
    ("eda_hz", 8),
    ("acc_hz", 32),
    ("size_seconds", 30),
    ("step_seconds", 5),
    ("eda_unit", "nanosiemens"),
    ("acc_unit", "counts"),
])
def test_every_hashed_window_field_moves_the_hash(raw_spec, field, new_value):
    """Each field named in HASHED_WINDOW_FIELDS must actually affect the digest."""
    baseline = compute_hash(raw_spec)
    mutated = copy.deepcopy(raw_spec)
    mutated["window"][field] = new_value
    assert compute_hash(mutated) != baseline, (
        f"window.{field} is declared hashed but changing it did not move the hash."
    )


# ── Schema coverage must not regress either ───────────────────────────────────

def test_changing_feature_order_changes_hash(raw_spec):
    baseline = compute_hash(raw_spec)
    mutated = copy.deepcopy(raw_spec)
    mutated["frozen_order"][0], mutated["frozen_order"][1] = (
        mutated["frozen_order"][1], mutated["frozen_order"][0]
    )
    assert compute_hash(mutated) != baseline


def test_changing_dtype_changes_hash(raw_spec):
    baseline = compute_hash(raw_spec)
    mutated = copy.deepcopy(raw_spec)
    mutated["features"][0]["dtype"] = "float32"
    assert compute_hash(mutated) != baseline


# ── bvp_hz is deliberately excluded ───────────────────────────────────────────

def test_changing_bvp_hz_does_not_change_hash(raw_spec):
    """
    bvp_hz is intentionally NOT hashed: HRV is deferred to v2 and no v1 feature
    reads BVP, so it cannot alter a v1 feature vector. This documents that the
    omission is a decision, not an oversight.
    """
    baseline = compute_hash(raw_spec)
    mutated = copy.deepcopy(raw_spec)
    mutated["window"]["bvp_hz"] = 128
    assert compute_hash(mutated) == baseline


# ── Missing semantics must fail loudly, not hash to something ─────────────────

def test_missing_window_field_raises(raw_spec):
    mutated = copy.deepcopy(raw_spec)
    del mutated["window"]["acc_unit"]
    with pytest.raises(KeyError) as exc_info:
        compute_hash(mutated)
    assert "acc_unit" in str(exc_info.value)


def test_canonical_string_contains_all_semantics(raw_spec):
    """The hashed string must literally carry every declared semantic field."""
    s = canonical_string(raw_spec)
    for field in HASHED_WINDOW_FIELDS:
        assert f"{field}=" in s, f"canonical string is missing '{field}='"
    assert "acc_unit=g" in s
    assert "acc_hz=4" in s

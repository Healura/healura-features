"""
Feature contract: spec loading and name-keyed vectorization.

Imports: numpy + pyyaml only. No sklearn/xgboost in this file.

The key invariant: the model-input vector is built by NAME from frozen_order,
never by position. Adding a feature to features.py without updating the spec
raises ValueError at vectorize() time — the ordering is non-load-bearing.
"""

import pathlib
from importlib import resources
from typing import Optional

import numpy as np
import yaml

# The frozen spec ships as package data inside this package. It is loaded via
# importlib.resources (not a relative __file__ path) so it resolves correctly
# from an installed wheel, not just an editable source checkout.
_SPEC_RESOURCE = "feature_spec_v1.yaml"


def load_spec(path: Optional[pathlib.Path] = None) -> dict:
    """
    Load feature_spec_v1.yaml and return a dict with keys:
        frozen_order : list[str]  — 14 feature names in canonical order
        spec_hash    : str        — SHA-256 of (names + dtypes) for provenance
        feature_pipeline_version : str

    With path=None the packaged spec is read via importlib.resources so it works
    from an installed wheel. Pass an explicit path to override (e.g. in tests).

    Raises FileNotFoundError if the yaml is missing.
    Raises KeyError if frozen_order or spec_hash are absent (broken spec).
    """
    if path is not None:
        with open(pathlib.Path(path)) as f:
            raw = yaml.safe_load(f)
    else:
        raw = yaml.safe_load(
            resources.files(__package__).joinpath(_SPEC_RESOURCE).read_text()
        )

    return {
        "frozen_order": raw["frozen_order"],
        "spec_hash": raw["spec_hash"],
        "feature_pipeline_version": raw["feature_pipeline_version"],
    }


def vectorize(
    features: dict,
    spec: Optional[dict] = None,
    nan_to_zero: bool = False,
) -> np.ndarray:
    """
    Build the model-input vector from a feature dict, ordered by frozen_order.

    Parameters
    ----------
    features : dict
        Output of extract_all_features() — keys are feature names, values are floats.
    spec : dict, optional
        Result of load_spec(). Loaded from the default path if None.
    nan_to_zero : bool, default False
        Substitute 0.0 for NaN instead of raising. OFF by default and reserved
        for debugging — do not enable it on any scoring or training path.

        A NaN means extract_all_features() FAILED on this window; it is a defect
        signal, not a value, and it must not be silently substituted. Zero is
        not a neutral filler: the model's StandardScaler is fit on RAW features,
        so an imputed 0.0 for eda_mean (µS, training mean ≈ 0.4–2) is rescaled
        to (0 − µ)/σ — an extreme low-end point, not the training mean. A failed
        window imputed to zero therefore scores as a specific, confidently wrong
        window rather than an unknown one. (An earlier docstring here claimed
        "StandardScaler in the pipeline treats 0 as mean" — it does not, and
        that claim is why this hole existed.)

        Callers should instead detect the NaN and refuse to score: see
        models/predict/score.py (gate_reason="extraction_failed") and
        pipelines/extract.py (drops the row).

    Returns
    -------
    np.ndarray, shape (len(frozen_order),), dtype float64

    Raises
    ------
    ValueError
        If features is missing any key in frozen_order, OR
        if features contains any key not in frozen_order.
        The message lists all offending keys so the caller can diagnose
        firmware/backend/ML divergence in one error rather than one key at a time.
    ValueError
        If nan_to_zero is False and any feature is NaN. The message names every
        offending feature.
    """
    if spec is None:
        spec = load_spec()

    frozen_order = spec["frozen_order"]
    spec_keys = set(frozen_order)
    input_keys = set(features.keys())

    missing = spec_keys - input_keys
    extra = input_keys - spec_keys

    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing from input: {sorted(missing)}")
        if extra:
            parts.append(f"extra in input (not in spec): {sorted(extra)}")
        raise ValueError(
            f"Feature dict does not match spec v1 frozen_order. "
            + "; ".join(parts)
        )

    vec = np.array([float(features[name]) for name in frozen_order], dtype=np.float64)

    if nan_to_zero:
        return np.where(np.isnan(vec), 0.0, vec)

    nan_names = [name for name, v in zip(frozen_order, vec) if np.isnan(v)]
    if nan_names:
        raise ValueError(
            f"Feature extraction failed — {len(nan_names)} of {len(frozen_order)} "
            f"v1 features are NaN: {nan_names}. A NaN means extract_all_features() "
            "could not compute the feature for this window; it must not be scored. "
            "Callers should refuse the window (see score.py 'extraction_failed') "
            "or drop it (see extract.py). Pass nan_to_zero=True only for debugging."
        )
    return vec

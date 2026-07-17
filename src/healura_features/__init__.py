"""
healura-features — the frozen shared feature layer for Healura.

This package contains ONLY the code that training and serving must run
IDENTICALLY: signal windowing/rate-normalisation, feature extraction, the
name-keyed vectoriser, and the frozen feature spec + spec_hash. It contains
nothing training-side (no sklearn/xgboost, no dataset loaders) and nothing
serving-side (no scoring, no gates). The golden parity fixture that ships in
this wheel is the acceptance gate: if extraction drifts, parity goes red.

Public API
----------
extract_all_features : one window (EDA/BVP/ACC) -> flat feature dict
to_canonical         : raw device window -> canonical (240,) EDA / (240, 3) ACC
resample_to_length   : nearest-index resampler (bit-identical to training)
vectorize            : feature dict -> model-input vector, ordered by spec
load_spec            : load the frozen feature_spec (frozen_order + spec_hash)
compute_hash         : recompute the spec_hash from a loaded spec dict
"""

from .canonical import resample_to_length, to_canonical
from .contract import load_spec, vectorize
from .features import extract_all_features
from .hash_spec import compute_hash

__version__ = "0.1.0"

__all__ = [
    "extract_all_features",
    "to_canonical",
    "resample_to_length",
    "vectorize",
    "load_spec",
    "compute_hash",
]

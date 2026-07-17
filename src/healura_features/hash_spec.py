"""
Compute, write, or verify the spec_hash for features/feature_spec_v1.yaml.

Hash input is a deterministic canonical string built from:
  - the frozen_order list (feature names in order)
  - each feature's (name, dtype) pair in frozen_order sequence
  - the canonical window semantics: eda_hz, acc_hz, size_seconds, step_seconds,
    eda_unit, acc_unit

WHY THE SEMANTICS SECTION EXISTS:
The hash used to cover names, order, and dtypes only — the SCHEMA. That meant
every semantic property of the contract was invisible to it: a consumer could
verify spec_hash, see no change, and still be sending ACC in raw counts instead
of g, or 1920-sample windows instead of 240. Both produce a correctly-shaped,
correctly-named, physically wrong feature vector. The hash now covers the rate
and unit contract too, so those changes move it and a verifying consumer breaks
loudly instead of drifting quietly.

bvp_hz is deliberately NOT hashed: HRV is deferred to v2 and no v1 feature reads
BVP, so a change to it cannot alter a v1 feature vector.

Usage
-----
python scripts/hash_spec.py            # recompute and WRITE spec_hash into the yaml
python scripts/hash_spec.py --check    # recompute and COMPARE; exit 1 on mismatch, never writes

Running this script twice on the same yaml content produces the identical hash.
"""

import argparse
import hashlib
import re
import sys
from importlib import resources

import yaml

# The spec ships as package data alongside this module. Resolve it via
# importlib.resources so --check / compute_hash work from an installed wheel,
# not only from a source checkout.
SPEC_PATH = resources.files(__package__).joinpath("feature_spec_v1.yaml")

# Fields of the `window:` block that feed the hash, in fixed order.
# Order is hardcoded, never derived from dict iteration, so the canonical
# string is stable across Python versions and yaml round-trips.
HASHED_WINDOW_FIELDS = (
    "eda_hz",
    "acc_hz",
    "size_seconds",
    "step_seconds",
    "eda_unit",
    "acc_unit",
)


def canonical_string(spec: dict) -> str:
    """
    Build the exact string that is hashed. Exposed separately so tests can
    assert on the input, not just the digest.

    Layout:
        <frozen_order names, one per line>
        ---
        <name:dtype per feature, in frozen_order sequence>
        ---
        <window_field=value, one per line, in HASHED_WINDOW_FIELDS order>
    """
    frozen_order = spec["frozen_order"]
    features_by_name = {f["name"]: f for f in spec["features"]}

    parts = []
    for name in frozen_order:
        feat = features_by_name[name]
        parts.append(f"{feat['name']}:{feat['dtype']}")

    window = spec["window"]
    missing = [f for f in HASHED_WINDOW_FIELDS if f not in window]
    if missing:
        raise KeyError(
            f"feature_spec window block is missing required field(s): {missing}. "
            f"All of {list(HASHED_WINDOW_FIELDS)} must be present — they are part "
            "of the hashed contract."
        )
    semantics = [f"{field}={window[field]}" for field in HASHED_WINDOW_FIELDS]

    return (
        "\n".join(frozen_order)
        + "\n---\n"
        + "\n".join(parts)
        + "\n---\n"
        + "\n".join(semantics)
    )


def compute_hash(spec: dict) -> str:
    return hashlib.sha256(canonical_string(spec).encode("utf-8")).hexdigest()


def _load_spec() -> dict:
    return yaml.safe_load(SPEC_PATH.read_text())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the stored spec_hash matches the yaml content. "
             "Exits 1 on mismatch and never writes.",
    )
    args = parser.parse_args(argv)

    spec = _load_spec()
    h = compute_hash(spec)
    stored = spec.get("spec_hash")

    if args.check:
        if stored == h:
            print(f"spec_hash OK: {h}")
            return 0
        print(
            f"spec_hash MISMATCH\n"
            f"  stored     : {stored}\n"
            f"  recomputed : {h}\n"
            f"The spec content has changed without regenerating the hash. "
            f"Run `python scripts/hash_spec.py` to update it — and confirm the "
            f"change is countersigned before it lands.",
            file=sys.stderr,
        )
        return 1

    print(f"spec_hash: {h}")

    # Write back — preserve existing structure, update spec_hash only.
    # Read raw text and replace the spec_hash line so surrounding formatting
    # (comments, blank lines) is left untouched.
    raw = SPEC_PATH.read_text()
    raw = re.sub(
        r'^spec_hash:.*$',
        f'spec_hash: "{h}"',
        raw,
        flags=re.MULTILINE,
    )
    SPEC_PATH.write_text(raw)
    return 0


if __name__ == "__main__":
    sys.exit(main())

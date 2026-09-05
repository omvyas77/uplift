"""Feature preparation.

There is almost nothing to do here, which is itself worth stating:

* No scaling - LightGBM is scale-invariant.
* No encoding - every column is already numeric.
* No imputation - the ingest assertions confirm there are no nulls.
* No `exposure`. It is a POST-TREATMENT variable; using it as a feature would
  break identification. `uplift.data.io.load_split` omits it structurally, and
  `load_iv_split` is the single function in the codebase that returns it.

The one real decision is which outcome is primary, and that is `visit` (4.7%)
rather than `conversion` (0.29%) - the Criteo paper's own recommendation, and
the only one of the two with enough base rate to rank on.
"""

from __future__ import annotations

import numpy as np

FEATURES: list[str] = [f"f{i}" for i in range(12)]


def check_no_leakage(columns: list[str]) -> None:
    """Raise if a post-treatment column reached a feature matrix.

    Cheap, and it turns a silent identification failure into a loud one.
    """
    banned = {"exposure", "is_exposed", "visit", "visited", "conversion", "converted", "y"}
    leaked = banned.intersection({c.lower() for c in columns})
    if leaked:
        raise ValueError(
            f"post-treatment or outcome columns in the feature matrix: {sorted(leaked)}. "
            "Conditioning on these breaks randomization - see docs/findings.md."
        )


def summarize(X: np.ndarray) -> dict[str, float]:
    return {
        "n": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_nan": int(np.isnan(X).sum()),
        "n_inf": int(np.isinf(X).sum()),
    }

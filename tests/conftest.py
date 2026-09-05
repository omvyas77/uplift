"""Shared fixtures.

Every fixture here is synthetic. Nothing in the fast test suite touches the
14M-row dataset, which is what lets CI run in minutes for free.
"""

from __future__ import annotations

import numpy as np
import pytest

from uplift.data.synthetic import make_rct


@pytest.fixture(scope="session")
def rct_small():
    """(X, w, y, true_ate) - an 85/15 RCT with a known heterogeneous effect."""
    return make_rct(n=200_000, treatment_share=0.85, seed=0)


@pytest.fixture(scope="session")
def rct_medium():
    return make_rct(n=300_000, treatment_share=0.85, seed=0)


@pytest.fixture
def rng():
    return np.random.default_rng(20260101)

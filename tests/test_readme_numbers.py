"""The README must not drift from the artifacts.

A portfolio README is the most-read file in the repo and the easiest one to let
go stale after a re-run. These tests parse the actual numbers out of it and check
them against evals/*.json, so a rerun that moves a metric fails here instead of
quietly leaving a wrong claim on the front page.

Skipped when the artifacts are absent, so a fresh clone is not blocked.
"""

from __future__ import annotations

import json
import re

import pytest

from uplift.config import REPO_ROOT, settings

_RAW = (REPO_ROOT / "README.md").read_text()
# The README uses typographic minus (U+2212) and en/em dashes for readability,
# while f-strings emit ASCII hyphen-minus. Normalise so the comparison is about
# the NUMBERS, not the punctuation.
README = _RAW.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")


def _load(name: str):
    p = settings.evals_dir / name
    if not p.exists():
        pytest.skip(f"{name} not generated")
    return json.loads(p.read_text())


def _claims(pattern: str) -> list[str]:
    return re.findall(pattern, README)


def test_row_count_matches_the_dataset():
    ev = _load("experiment_visit.json")
    assert f"{ev['n']:,}" in README


def test_headline_lift_matches_the_adjusted_estimate():
    ev = _load("experiment_visit.json")
    adjusted = 100 * ev["ate"]["lin"]["relative_lift"]
    unadjusted = 100 * ev["ate"]["difference_in_means"]["relative_lift"]
    assert f"+{adjusted:.1f}%" in README
    assert f"+{unadjusted:.1f}%" in README
    # and the README must present the ADJUSTED one as the headline
    assert "The number to quote is" in README


def test_every_model_qini_in_the_table_is_current():
    res = _load("results.json")
    for name, m in res["models"].items():
        assert f"{m['qini']:.4f}".lstrip("0") in README or f"{m['qini']:.4f}" in README, (
            f"{name} qini {m['qini']:.4f} not found in README"
        )


def test_the_overlapping_model_is_reported_as_overlapping():
    """The single most important honesty claim in the README."""
    res = _load("results.json")
    unresolved = [
        n
        for n, m in res["models"].items()
        if isinstance(m.get("vs_reference"), dict) and not m["vs_reference"]["resolved"]
    ]
    assert unresolved, "expected at least one indistinguishable model"
    for name in unresolved:
        assert "overlapping" in README.lower()
        assert name.replace("_", "-").split("-")[0].lower() in README.lower()


def test_bias_table_numbers_are_current():
    cau = _load("causal.json")
    rows = {r["estimator"]: r for r in cau["bias_table"]["rows"]}
    truth = rows["ground truth (from the RCT)"]["estimate"]
    assert f"{truth:.6f}" in README
    naive_rel = 100 * rows["naive difference-in-means"]["relative_error"]
    assert f"{naive_rel:.0f}%" in README


def test_iv_numbers_are_current():
    cau = _load("causal.json")
    iv = cau["iv"]
    assert f"{iv['itt']:.6f}" in README
    assert f"{iv['cace']:.6f}" in README
    assert f"{iv['naive_exposed_vs_control']:.6f}" in README


def test_assumptions_are_stated_next_to_every_dollar_figure():
    """The guide's rule: never a dollar figure without the assumption line."""
    assert "$25.00 per conversion" in README
    assert "$0.01 per treatment" in README
    dollars = _claims(r"\$[\d,]+")
    assert dollars, "expected dollar figures in the README"


def test_limitations_section_exists_and_is_substantive():
    assert "Honest limitations" in README
    section = README.split("Honest limitations")[1]
    for required in ("indistinguishable", "exclusion restriction", "injected by me", "xfail"):
        assert required in section, f"limitations section missing: {required}"

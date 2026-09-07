"""The dashboard must actually execute, not merely serve HTML.

Booting Streamlit and curling `/` returns 200 before the script has run at all -
it only serves the shell. That check passed while the app was raising KeyError on
a policy field whose real name was `diff_dr`, not `difference`. AppTest runs the
script for real and surfaces the exception, which is the only version of this
test worth having.
"""

from __future__ import annotations

import pytest

from uplift.config import REPO_ROOT, settings

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
# Absolute: AppTest resolves relative paths against the CALLING file, so
# "src/uplift/..." would be looked up under tests/.
APP = str(REPO_ROOT / "src" / "uplift" / "dashboard" / "app.py")


@pytest.fixture(scope="module")
def app():
    at = streamlit_testing.AppTest.from_file(APP, default_timeout=180)
    at.run()
    return at


def test_the_app_runs_without_exceptions(app):
    assert not app.exception, [str(e.value) for e in app.exception]


def test_all_five_tabs_render(app):
    assert len(app.tabs) == 5


def test_the_business_assumptions_are_sliders(app):
    """They must be adjustable, not hardcoded - that is what makes the
    sensitivity of every dollar figure visible."""
    labels = [s.label for s in app.slider]
    assert any("conversion" in lab.lower() for lab in labels)
    assert any("cost" in lab.lower() for lab in labels)


@pytest.mark.skipif(not (settings.evals_dir / "results.json").exists(), reason="no results.json")
def test_it_renders_content_when_artifacts_exist(app):
    assert len(app.dataframe) >= 4
    assert len(app.metric) >= 4


def test_changing_an_assumption_does_not_break_it(app):
    """Moving a slider re-runs the script; the profit maths must survive it."""
    if not app.slider:
        pytest.skip("no sliders rendered")
    app.slider[0].set_value(60.0).run()
    assert not app.exception, [str(e.value) for e in app.exception]

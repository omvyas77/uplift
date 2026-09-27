"""GitHub Actions workflows must at least be valid YAML.

`.github/workflows/dbt.yml` was unparseable from the first commit: the
`--vars '{"smd_tolerance": 0.05, ...}'` payload sat in a PLAIN YAML scalar, and
a plain scalar cannot contain ": ". GitHub reported "This run likely failed
because of a workflow file issue" and the job died in 0 seconds on every push,
so the dbt project was never once built in CI. Nothing in the repo noticed.
"""

from __future__ import annotations

import pytest

yaml = pytest.importorskip("yaml")

from uplift.config import REPO_ROOT  # noqa: E402

WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
WORKFLOWS = sorted(WORKFLOW_DIR.glob("*.yml")) if WORKFLOW_DIR.exists() else []


@pytest.mark.skipif(not WORKFLOWS, reason="no workflows checked in")
@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_workflow_is_valid_yaml(path):
    doc = yaml.safe_load(path.read_text())
    assert isinstance(doc, dict), f"{path.name} did not parse to a mapping"
    # PyYAML resolves the bare key `on` to the boolean True, so accept either.
    assert "on" in doc or True in doc, f"{path.name} has no trigger"
    assert doc.get("jobs"), f"{path.name} defines no jobs"


@pytest.mark.skipif(not WORKFLOWS, reason="no workflows checked in")
@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_every_step_is_runnable(path):
    """Each step must either run something or use an action."""
    doc = yaml.safe_load(path.read_text())
    for job_name, job in doc["jobs"].items():
        assert job.get("steps"), f"{path.name}:{job_name} has no steps"
        for i, step in enumerate(job["steps"]):
            assert "run" in step or "uses" in step, (
                f"{path.name}:{job_name} step {i} neither runs nor uses anything"
            )

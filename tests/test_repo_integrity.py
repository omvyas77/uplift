"""Guard against the repo being unrunnable from a clean clone.

This exists because of a real, silent failure: `.gitignore` carried an
unanchored `data/` rule, which matches a directory of that name at ANY depth.
It therefore excluded the whole `src/uplift/data/` package - download, ingest,
io, splits, synthetic - from every commit. The public repo could not import
`uplift.data.synthetic`, so every test and the entire CLI were broken on a
fresh clone, while everything passed locally because the files were on disk.

Local green means nothing if the artifact people clone is missing source.
"""

from __future__ import annotations

import subprocess

import pytest

from uplift.config import REPO_ROOT

PACKAGE_DIRS = ["src", "orchestration", "scripts", "tests"]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def _is_git_repo() -> bool:
    try:
        _git("rev-parse", "--git-dir")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


pytestmark = pytest.mark.skipif(not _is_git_repo(), reason="not a git checkout")


def test_no_source_file_is_invisible_to_git():
    """The bug that motivated this file: source silently excluded by gitignore.

    A file git has never seen is EITHER tracked OR reported as untracked. An
    ignored file is in neither list - it is invisible, and that is precisely the
    failure that shipped a repo missing `src/uplift/data/`. A brand-new file
    that simply has not been `git add`ed yet is untracked, not invisible, so
    this does not fire during normal work.
    """
    visible = set(_git("ls-files").splitlines())
    visible |= set(_git("ls-files", "--others", "--exclude-standard").splitlines())
    invisible = []
    for d in PACKAGE_DIRS:
        root = REPO_ROOT / d
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(REPO_ROOT).as_posix()
            if rel not in visible:
                invisible.append(rel)
    assert not invisible, (
        "these source files are invisible to git, so a clone cannot run them:\n  "
        + "\n  ".join(sorted(invisible))
    )


def test_no_source_path_is_gitignored():
    """Catches the failure one step earlier, with the offending rule named."""
    offenders = []
    for d in PACKAGE_DIRS:
        root = REPO_ROOT / d
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(REPO_ROOT).as_posix()
            res = subprocess.run(
                ["git", "check-ignore", "-v", rel],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            if res.returncode == 0:
                offenders.append(res.stdout.strip())
    assert not offenders, "source files matched a .gitignore rule:\n  " + "\n  ".join(offenders)


def test_the_package_exposes_its_subpackages():
    """A cheap import-level check that the tree is complete."""
    import importlib

    for mod in (
        "uplift.data.synthetic",
        "uplift.data.io",
        "uplift.data.ingest",
        "uplift.data.download",
        "uplift.data.splits",
        "uplift.causal.iv",
        "uplift.evaluation.qini",
        "uplift.policy.ope",
    ):
        assert importlib.import_module(mod) is not None


def test_data_artifacts_are_still_ignored():
    """The fix anchored the ignore rules; make sure it did not un-ignore the
    311 MB source file or the DuckDB database."""
    for path in ("data/uplift.duckdb", "data/criteo_units.parquet", "artifacts/x.parquet"):
        res = subprocess.run(
            ["git", "check-ignore", path], cwd=REPO_ROOT, capture_output=True, text=True
        )
        assert res.returncode == 0, f"{path} is NOT ignored; large data could be committed"

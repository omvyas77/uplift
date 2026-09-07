"""Publish the dashboard to the Hugging Face Space.

Assembles a self-contained Space from the repo - the dashboard module itself,
`uplift.config`, and the committed `evals/*.json` - and uploads it. The Space
runs the SAME dashboard module as `make dashboard`, via a thin entrypoint, so
the two cannot drift.

Two things worth knowing before running it:

* HF no longer offers a `streamlit` SDK for new Spaces; `create_repo` rejects it
  with 'expected one of gradio|docker|static'. This uses the **docker** SDK and
  runs Streamlit on port 7860, which is the port HF's proxy expects and must
  match `app_port` in the Space README frontmatter.
* Only pre-computed results are uploaded. No model artifacts, no data, and
  nothing that would let the Space retrain anything.

    python scripts/deploy_space.py --repo-id omvyas77/uplift
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from uplift.config import REPO_ROOT

EVALS = [
    "experiment_visit.json",
    "experiment_conversion.json",
    "results.json",
    "causal.json",
    "policy.json",
]


def stage(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    tmpl = REPO_ROOT / "deploy" / "hf_space"
    for name in ("Dockerfile", "app.py", "requirements.txt", "README.md"):
        shutil.copy(tmpl / name, dest / name)

    pkg = dest / "src" / "uplift"
    (pkg / "dashboard").mkdir(parents=True, exist_ok=True)
    src = REPO_ROOT / "src" / "uplift"
    for rel in ("__init__.py", "config.py", "dashboard/__init__.py", "dashboard/app.py"):
        shutil.copy(src / rel, pkg / rel)

    evals = dest / "evals"
    evals.mkdir(exist_ok=True)
    missing = []
    for name in EVALS:
        p = REPO_ROOT / "evals" / name
        if p.exists():
            shutil.copy(p, evals / name)
        else:
            missing.append(name)
    if missing:
        raise SystemExit(f"missing evals: {missing}. Run the reports before deploying.")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", default="omvyas77/uplift")
    ap.add_argument("--dry-run", action="store_true", help="stage only, do not upload")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        payload = stage(Path(tmp) / "space")
        files = sorted(p.relative_to(payload).as_posix() for p in payload.rglob("*") if p.is_file())
        print(f"staged {len(files)} files:")
        for f in files:
            print("  " + f)
        if args.dry_run:
            return 0

        from huggingface_hub import HfApi

        api = HfApi()
        url = api.create_repo(
            repo_id=args.repo_id,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True,
            private=False,
        )
        api.upload_folder(
            folder_path=str(payload),
            repo_id=args.repo_id,
            repo_type="space",
            commit_message="Update dashboard and results",
        )
        print(f"\npublished: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

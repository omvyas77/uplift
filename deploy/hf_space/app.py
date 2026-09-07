"""Hugging Face Space entrypoint.

The dashboard lives in the package at src/uplift/dashboard/app.py so that the
Space runs the SAME code as `make dashboard` locally, rather than a fork that
can drift. This file only puts src/ on the path and executes it.
"""

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
runpy.run_path(str(ROOT / "src" / "uplift" / "dashboard" / "app.py"), run_name="__main__")

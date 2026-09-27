"""Fetch the Criteo uplift file from the first working mirror, then verify it.

Three mirrors and a checksum, because a portfolio repo that fails on
`make download` eighteen months from now is worse than no repo. The checksum
also proves which exact file version produced every number in this repo.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path

from uplift.config import settings
from uplift.logging import get_logger

log = get_logger(__name__)
CHUNK = 1024 * 1024


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def download_criteo(force: bool = False) -> Path:
    settings.ensure_dirs()
    dest = settings.criteo_csv_path

    if dest.exists() and not force:
        if sha256_of(dest) == settings.criteo_sha256:
            log.info("already_present", path=str(dest))
            return dest
        log.warning("checksum_mismatch_redownloading", path=str(dest))

    last_error: Exception | None = None
    for url in settings.criteo_urls:
        try:
            log.info("downloading", url=url)
            tmp = dest.with_suffix(dest.suffix + ".part")
            req = urllib.request.Request(url, headers={"User-Agent": "uplift-portfolio/0.1"})
            with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as out:
                shutil.copyfileobj(resp, out, CHUNK)
            tmp.replace(dest)
            break
        except Exception as exc:
            log.warning("mirror_failed", url=url, error=str(exc))
            last_error = exc
    else:
        raise RuntimeError(f"all mirrors failed; last error: {last_error}")

    digest = sha256_of(dest)
    if digest != settings.criteo_sha256:
        raise ValueError(
            f"checksum mismatch\n  expected {settings.criteo_sha256}\n  got      {digest}"
        )
    log.info("verified", path=str(dest), sha256=digest)
    return dest

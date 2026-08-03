"""Provenance stamps for every artifact.

SPEC.md 6.1: no number may reach the manuscript without a ``git_sha``, a
``config_hash``, a ``data_hash`` and a seed list attached. This module produces those
stamps so that ``finsent.eval.report`` can embed them in each table footnote.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = ["git_sha", "git_is_dirty", "hash_array", "hash_frame", "hash_obj", "Provenance"]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), *args],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_sha(short: bool = True) -> str:
    """Current commit sha, or ``"unknown"`` outside a repository."""
    sha = _git("rev-parse", "--short=12" if short else "HEAD", "HEAD")
    if sha is None:
        return "unknown"
    return sha.splitlines()[0] if sha else "unknown"


def git_is_dirty() -> bool:
    """True when the working tree has uncommitted changes.

    A dirty tree means the artifact cannot be reproduced from the recorded sha, so
    ``Provenance`` records it and the report footnote surfaces it.
    """
    status = _git("status", "--porcelain")
    return bool(status)


def hash_array(arr: np.ndarray) -> str:
    """Content digest of a numeric array (shape and dtype included)."""
    a = np.ascontiguousarray(arr)
    hasher = hashlib.sha256()
    hasher.update(str(a.shape).encode())
    hasher.update(str(a.dtype).encode())
    hasher.update(a.tobytes())
    return hasher.hexdigest()[:12]


def hash_frame(df: pd.DataFrame) -> str:
    """Content digest of a dataframe, stable across column ordering."""
    hasher = hashlib.sha256()
    for col in sorted(map(str, df.columns)):
        hasher.update(col.encode())
        values = df[col].to_numpy()
        if values.dtype == object or str(values.dtype).startswith(("datetime", "str")):
            hasher.update(pd.Series(values).astype(str).str.cat(sep="|").encode())
        else:
            hasher.update(np.ascontiguousarray(values).tobytes())
    hasher.update(str(len(df)).encode())
    return hasher.hexdigest()[:12]


def hash_obj(obj: Any) -> str:
    """Digest of any JSON-serialisable object."""
    payload = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class Provenance:
    """Everything needed to decide whether a reported number can be trusted."""

    git_sha: str
    dirty: bool
    config_hash: str
    data_hash: str
    seeds: tuple[int, ...]
    created_utc: str
    python: str
    platform: str

    @classmethod
    def capture(
        cls,
        config_hash: str = "unknown",
        data_hash: str = "unknown",
        seeds: Iterable[int] = (),
    ) -> "Provenance":
        return cls(
            git_sha=git_sha(),
            dirty=git_is_dirty(),
            config_hash=config_hash,
            data_hash=data_hash,
            seeds=tuple(int(s) for s in seeds),
            created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            python=sys.version.split()[0],
            platform=platform.platform(),
        )

    def footnote(self) -> str:
        """One-line provenance string for a table caption."""
        dirty = " (DIRTY TREE - not reproducible)" if self.dirty else ""
        seeds = ",".join(str(s) for s in self.seeds) if self.seeds else "none"
        return (
            f"git={self.git_sha}{dirty}; config={self.config_hash}; "
            f"data={self.data_hash}; seeds=[{seeds}]; generated={self.created_utc}"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def stamp_path(self, stem: str, suffix: str) -> str:
        """Artifact filename with the config hash and sha embedded."""
        return f"{stem}__cfg-{self.config_hash}__git-{self.git_sha}{suffix}"

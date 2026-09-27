"""Repository path resolution: anchor everything at repo root."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    # src/utils/paths.py -> parents[2] == repo root
    return Path(__file__).resolve().parents[2]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return repo_root() / p

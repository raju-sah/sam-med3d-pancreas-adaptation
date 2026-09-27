"""Deterministic case-level train/val/test splitting (Phase 1).

Rules: split whole 3D cases, never slices. IDs sorted before a seeded
shuffle. Internal test IDs are FROZEN once written — see tests/test_split.py.
"""

from __future__ import annotations

import hashlib
import random


def canonical_order(case_ids: list[str]) -> list[str]:
    """Deduplicated, stably sorted canonical ID list."""
    return sorted(set(case_ids))


def canonical_hash(ordered_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ordered_ids).encode("utf-8")).hexdigest()


def make_split(
    case_ids: list[str],
    seed: int = 42,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> dict[str, list[str]]:
    """Return {'train': [...], 'val': [...], 'test': [...]} (each list sorted).

    Algorithm: sort IDs, shuffle with random.Random(seed), take
    round(n*train_frac) train, round(n*val_frac) val, remainder test.
    """
    ordered = canonical_order(case_ids)
    n = len(ordered)
    if n == 0:
        raise ValueError("empty case-ID list")
    rng = random.Random(seed)
    shuffled = ordered[:]
    rng.shuffle(shuffled)
    n_train = round(n * train_frac)
    n_val = round(n * val_frac)
    n_test = n - n_train - n_val
    if n_test < 0:
        raise ValueError(f"fractions over-allocate n={n}")
    return {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train : n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val :]),
    }

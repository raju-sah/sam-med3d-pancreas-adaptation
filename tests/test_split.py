"""Phase 1 split tests. No medical data required (synthetic IDs), except the
frozen-split regression test, which reads the tracked data/splits/*.json files
and skips if Phase 1 splits are not generated yet."""

import json
import unittest
from pathlib import Path

from src.data.split import canonical_hash, canonical_order, make_split
from src.utils.paths import repo_root

SPLITS_DIR = repo_root() / "data" / "splits"


def synthetic_ids(n: int = 281) -> list[str]:
    # Deliberately unsorted input to prove sort-stability.
    return [f"case_{i:03d}" for i in range(n, 0, -1)]


class SplitLogic(unittest.TestCase):
    def test_deterministic_same_seed(self):
        self.assertEqual(make_split(synthetic_ids()), make_split(synthetic_ids()))

    def test_different_seed_differs(self):
        self.assertNotEqual(
            make_split(synthetic_ids(), seed=42), make_split(synthetic_ids(), seed=43)
        )

    def test_mutually_exclusive_and_complete(self):
        s = make_split(synthetic_ids())
        all_ids = s["train"] + s["val"] + s["test"]
        self.assertEqual(len(all_ids), len(set(all_ids)))  # no duplicates
        self.assertEqual(set(all_ids), set(canonical_order(synthetic_ids())))

    def test_expected_counts_n281(self):
        s = make_split(synthetic_ids(281))
        self.assertEqual((len(s["train"]), len(s["val"]), len(s["test"])), (197, 42, 42))

    def test_sorted_stable_serialization(self):
        s = make_split(synthetic_ids())
        for k in ("train", "val", "test"):
            self.assertEqual(s[k], sorted(s[k]))
        blob = json.dumps(s, sort_keys=True)
        self.assertEqual(json.loads(blob), s)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            make_split([])


class FrozenSplitRegression(unittest.TestCase):
    @unittest.skipUnless(
        (SPLITS_DIR / "split_manifest.json").is_file(), "Phase 1 splits not generated yet"
    )
    def test_regeneration_reproduces_frozen_split(self):
        """Any code change that silently alters the frozen split fails here."""
        manifest = json.loads((SPLITS_DIR / "split_manifest.json").read_text())
        canonical = manifest["canonical_case_ids"]
        self.assertEqual(canonical_hash(canonical), manifest["canonical_case_ids_sha256"])
        regen = make_split(canonical, seed=manifest["seed"])
        for k in ("train", "val", "test"):
            frozen = json.loads((SPLITS_DIR / f"{k}.json").read_text())
            self.assertEqual(regen[k], frozen, f"frozen {k} split changed!")


if __name__ == "__main__":
    unittest.main()

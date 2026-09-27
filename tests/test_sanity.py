"""Phase 0 sanity tests (stdlib unittest — no pytest install needed)."""

import unittest
from pathlib import Path


class Phase0Sanity(unittest.TestCase):
    def test_package_imports(self):
        import src
        import src.data
        import src.models
        import src.training
        import src.evaluation
        import src.utils
        import src.utils.config as c
        import src.utils.paths as p
        import src.utils.seed as s

        self.assertTrue(hasattr(c, "load_config"))
        self.assertTrue(hasattr(p, "repo_root"))
        self.assertTrue(hasattr(s, "set_seed"))

    def test_paths_resolve(self):
        from src.utils.paths import repo_root, resolve

        root = repo_root()
        self.assertTrue((root / "pyproject.toml").is_file())
        self.assertEqual(resolve("configs/base.yaml"), root / "configs/base.yaml")

    def test_config_loads(self):
        from src.utils.config import load_config
        from src.utils.paths import resolve

        cfg = load_config(resolve("configs/base.yaml"))
        self.assertEqual(cfg["seed"], 42)
        self.assertIn("paths", cfg)

    def test_seed_deterministic(self):
        import random

        from src.utils.seed import set_seed

        set_seed(42)
        a = random.random()
        set_seed(42)
        b = random.random()
        self.assertEqual(a, b)

    def test_config_missing_raises(self):
        from src.utils.config import load_config

        with self.assertRaises(FileNotFoundError):
            load_config("configs/does_not_exist.yaml")


if __name__ == "__main__":
    unittest.main()

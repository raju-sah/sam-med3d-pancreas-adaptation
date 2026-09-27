"""Phase 2 preprocessing tests — synthetic arrays only, no dataset, no MONAI."""

import unittest

import numpy as np

from src.data import preprocess as P


def toy_mask() -> np.ndarray:
    m = np.zeros((8, 9, 10), dtype=np.uint8)
    m[1:4, 1:4, 1:4] = 1
    m[5:7, 5:7, 5:7] = 2
    return m


def _has_monai() -> bool:
    try:
        import monai  # noqa: F401

        return True
    except ImportError:
        return False


class PreprocessLogic(unittest.TestCase):
    def test_config_parses(self):
        from src.utils.config import load_config
        from src.utils.paths import resolve

        cfg = load_config(resolve("configs/preprocess_phase2.yaml"))
        self.assertEqual(cfg["orientation"], "RAS")
        self.assertEqual(cfg["label_mode"], "nearest")
        self.assertEqual(cfg["valid_labels"], [0, 1, 2])

    def test_valid_labels_accepted(self):
        self.assertEqual(P.validate_labels(toy_mask()), {0, 1, 2})

    def test_invalid_labels_rejected(self):
        bad = toy_mask()
        bad[0, 0, 0] = 5
        with self.assertRaises(ValueError):
            P.validate_labels(bad)

    def test_derived_tumor_mask(self):
        d = P.derive_masks(toy_mask())
        self.assertEqual(int(d["tumor"].sum()), 8)
        self.assertTrue(set(np.unique(d["tumor"])).issubset({0, 1}))

    def test_derived_whole_mask(self):
        d = P.derive_masks(toy_mask())
        self.assertEqual(int(d["whole"].sum()), 27 + 8)
        self.assertTrue(set(np.unique(d["whole"])).issubset({0, 1}))

    def test_nearest_resample_preserves_labels(self):
        out = P.resample_label_nearest(toy_mask(), (0.5, 0.5, 0.5))
        self.assertTrue(set(np.unique(out)).issubset({0, 1, 2}))
        out2 = P.resample_label_nearest(toy_mask(), (2.0, 2.0, 2.0))
        self.assertTrue(set(np.unique(out2)).issubset({0, 1, 2}))

    def test_alignment_ok_and_mismatch(self):
        img = np.zeros((1, 8, 9, 10))
        self.assertEqual(P.check_alignment(img, toy_mask()), (8, 9, 10))
        with self.assertRaises(ValueError):
            P.check_alignment(img, np.zeros((8, 8, 10), dtype=np.uint8))

    def test_determinism(self):
        a = P.resample_label_nearest(toy_mask(), (0.5, 1.0, 2.0))
        b = P.resample_label_nearest(toy_mask(), (0.5, 1.0, 2.0))
        np.testing.assert_array_equal(a, b)

    def test_channel_first(self):
        v = np.zeros((8, 9, 10))
        self.assertEqual(P.ensure_channel_first(v).shape, (1, 8, 9, 10))

    def test_test_ids_cannot_enter_profiling(self):
        with self.assertRaises(ValueError):
            P.assert_not_test(["pancreas_001", "pancreas_004"] + P.repo_split_ids("test")[:1])
        P.assert_not_test(P.repo_split_ids("train")[:5])  # train IDs pass the guard

    def test_splits_intact(self):
        tr, va, te = (set(P.repo_split_ids(s)) for s in ("train", "val", "test"))
        self.assertEqual((len(tr), len(va), len(te)), (197, 42, 42))
        self.assertFalse(tr & va or tr & te or va & te)

    @unittest.skipUnless(_has_monai(), "monai not installed locally")
    def test_monai_builder(self):
        from src.utils.config import load_config
        from src.utils.paths import resolve

        cfg = load_config(resolve("configs/preprocess_phase2.yaml"))
        t = P.build_monai_inference_transforms({**cfg, "target_spacing": [1.5, 1.5, 1.5]})
        self.assertIsNotNone(t)


if __name__ == "__main__":
    unittest.main()

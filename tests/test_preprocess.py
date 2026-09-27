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
        self.assertEqual(cfg["norm"], "positive_intensity_zscore")

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

    @unittest.skipUnless(_has_monai(), "monai not installed")
    def test_monai_builder(self):
        import tempfile

        import nibabel as nib
        from src.utils.config import load_config
        from src.utils.paths import resolve

        cfg = load_config(resolve("configs/preprocess_phase2.yaml"))
        t = P.build_monai_inference_transforms({**cfg, "target_spacing": [1.5, 1.5, 1.5]})
        self.assertIsNotNone(t)
        # builder output matches the numpy reference on synthetic data
        vals = np.array([-1000.0, -500, -100, 0, 50, 100, 500, 1000],
                        dtype=np.float32).reshape(2, 2, 2)
        with tempfile.TemporaryDirectory() as d:
            nib.save(nib.Nifti1Image(vals, np.eye(4)), f"{d}/img.nii.gz")
            nib.save(nib.Nifti1Image((vals > 0).astype(np.uint8), np.eye(4)),
                     f"{d}/lbl.nii.gz")
            out = t({"image": f"{d}/img.nii.gz", "label": f"{d}/lbl.nii.gz"})
        ref = P.positive_intensity_zscore(np.clip(vals, -1000, 1000))
        np.testing.assert_allclose(np.asarray(out["image"])[0], ref, rtol=1e-5, atol=1e-6)


def _has_torchio() -> bool:
    try:
        import torchio  # noqa: F401

        return True
    except ImportError:
        return False


class PositiveMaskZScore(unittest.TestCase):
    PROBE = np.array([-1000.0, -500, -100, 0, 50, 100, 500, 1000], dtype=np.float32)

    def test_mask_selects_strictly_positive(self):
        out = P.positive_intensity_zscore(self.PROBE)
        pos = np.array([50.0, 100, 500, 1000])
        np.testing.assert_allclose(out, (self.PROBE - pos.mean()) / pos.std(), rtol=1e-6)

    def test_monai_nonzero_would_differ(self):
        # MONAI nonzero=True masks img != 0 (includes negatives) and leaves
        # masked-out voxels untouched — both differ from upstream semantics.
        from monai.transforms import NormalizeIntensity

        monai_out = np.asarray(NormalizeIntensity(nonzero=True)(self.PROBE.copy()))
        # negatives standardized differently (or zeros untouched) -> not equal
        self.assertFalse(np.allclose(monai_out, P.positive_intensity_zscore(self.PROBE)))
        # corrected output standardizes EVERY voxel, including negatives/zeros
        corrected = P.positive_intensity_zscore(self.PROBE)
        self.assertFalse(np.allclose(corrected[self.PROBE == 0], 0.0))

    def test_deterministic_and_finite(self):
        a = P.positive_intensity_zscore(self.PROBE)
        b = P.positive_intensity_zscore(self.PROBE)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.all(np.isfinite(a)))
        self.assertEqual(a.dtype, np.float32)

    def test_degenerate_cases(self):
        np.testing.assert_array_equal(
            P.positive_intensity_zscore(np.array([-5.0, 0.0, -1.0])), np.zeros(3))
        np.testing.assert_array_equal(
            P.positive_intensity_zscore(np.array([7.0, 7.0])), np.zeros(2))

    @unittest.skipUnless(_has_torchio(), "torchio not installed")
    def test_direct_torchio_equivalence(self):
        import torch
        import torchio as tio

        data = np.random.RandomState(0).uniform(-1200, 1500, size=(1, 16, 16, 16)).astype(np.float32)
        subject = tio.Subject(img=tio.ScalarImage(tensor=torch.from_numpy(data)))
        got = tio.ZNormalization(masking_method=lambda x: x > 0)(subject).img.data.numpy()
        ref = np.stack([P.positive_intensity_zscore(ch) for ch in data])
        diff = np.abs(got - ref).max()
        print(f"\nmax |torchio - ours| = {diff:.3e}")
        # Same mask + same whole-image application; residual is float32
        # accumulation-order rounding (torch fp32 mean/std vs numpy fp64).
        self.assertLess(diff, 5e-3)


if __name__ == "__main__":
    unittest.main()

"""Phase 3 tests — synthetic tensors/volumes only. No dataset, no GPU needed."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


def toy_cfg(**over):
    cfg = {
        "seed": 42, "spatial_dims": 3, "in_channels": 1, "out_channels": 3,
        "channels": [8, 16, 32], "strides": [2, 2], "num_res_units": 1, "norm": "INSTANCE",
        "loss_include_background": False, "loss_to_onehot_y": True, "loss_softmax": True,
        "lr": 1e-4, "weight_decay": 1e-5, "max_epochs": 2,
        "orientation": "RAS", "target_spacing": None, "image_mode": "bilinear",
        "label_mode": "nearest", "clamp_min": -1000, "clamp_max": 1000,
        "patch_size": [16, 16, 16], "sampler_ratios": [1, 2, 1],
        "sampler_num_classes": 3, "sampler_num_samples": 1,
        "aug_flip_prob": 0.5, "aug_rotate90_prob": 0.0,
        "aug_scale_prob": 0.0, "aug_scale_factors": [0.9, 1.1],
        "aug_shift_prob": 0.0, "aug_shift_offsets": [-0.1, 0.1],
        "batch_size": 1, "num_workers": 0,
        "sw_roi_size": [16, 16, 16], "sw_overlap": 0.25, "sw_batch_size": 1,
        "sw_mode": "constant", "amp": False,
        "val_interval_epochs": 1, "early_stopping_min_delta": 0.001,
        "early_stopping_patience_val_events": 8,
        "best_filename": "best.pt", "last_filename": "last.pt",
    }
    cfg.update(over)
    return cfg


def toy_pair():
    img = torch.randn(1, 32, 32, 32)
    lbl = torch.zeros(1, 32, 32, 32, dtype=torch.long)
    lbl[:, 8:20, 8:20, 8:20] = 1
    lbl[:, 12:16, 12:16, 12:16] = 2
    return {"image": img, "label": lbl}


class UnetPhase3(unittest.TestCase):
    def test_model_output_shape(self):
        from src.models.unet import build_unet, count_parameters

        m = build_unet(toy_cfg())
        out = m(torch.randn(2, 1, 32, 32, 32))
        self.assertEqual(tuple(out.shape), (2, 3, 32, 32, 32))
        p = count_parameters(m)
        self.assertGreater(p["total"], 0)
        self.assertEqual(p["total"], p["trainable"])

    def test_loss_finite(self):
        from src.training.trainer import build_loss

        loss = build_loss(toy_cfg())
        m_out = torch.randn(2, 3, 16, 16, 16)
        tgt = torch.randint(0, 3, (2, 1, 16, 16, 16))
        v = loss(m_out, tgt)
        self.assertTrue(torch.isfinite(v))

    def test_metric_calculations(self):
        from src.training.metrics import binary_dice, binary_iou, multiclass_report

        a = np.zeros((6, 6, 6), bool)
        a[1:3, 1:3, 1:3] = True
        self.assertEqual(binary_dice(a, a), 1.0)
        self.assertEqual(binary_dice(a, ~a & np.ones_like(a)), 0.0)
        self.assertEqual(binary_dice(np.zeros((2, 2, 2), bool), np.zeros((2, 2, 2), bool)), 1.0)
        self.assertEqual(binary_iou(a, a), 1.0)
        pred = np.zeros((6, 6, 6), np.uint8)
        pred[1:3, 1:3, 1:3] = 1
        pred[4, 4, 4] = 2
        gt = pred.copy()
        r = multiclass_report(pred, gt)
        self.assertEqual(r["dice_macro_foreground"], 1.0)
        self.assertEqual(r["dice_whole"], 1.0)

    def test_empty_prediction_robust(self):
        from src.training.metrics import hd95_per_target, multiclass_report

        gt = np.zeros((6, 6, 6), np.uint8)
        gt[1:3, 1:3, 1:3] = 2
        r = multiclass_report(np.zeros_like(gt), gt)
        self.assertEqual(r["dice_tumor"], 0.0)
        h = hd95_per_target(np.zeros_like(gt), gt)
        self.assertTrue(np.isnan(h["hd95_tumor"]))
        # empty pred -> both whole and tumor undefined
        self.assertEqual(h["hd95_n_undefined"], 2.0)

    def test_whole_union_derivation(self):
        from src.training.metrics import multiclass_report

        pred = np.zeros((4, 4, 4), np.uint8)
        pred[0, 0, 0] = 1
        gt = np.zeros((4, 4, 4), np.uint8)
        gt[0, 0, 0] = 2
        r = multiclass_report(pred, gt)
        # whole masks both have the single voxel -> dice 1 despite class mismatch
        self.assertEqual(r["dice_whole"], 1.0)
        self.assertEqual(r["dice_class1"], 0.0)

    def test_softmax_argmax(self):
        from src.training.metrics import softmax_argmax

        logits = torch.zeros(3, 4, 4, 4)
        logits[2] = 5.0
        np.testing.assert_array_equal(softmax_argmax(logits), np.full((4, 4, 4), 2, np.uint8))

    def test_config_parses(self):
        from src.utils.config import load_config
        from src.utils.paths import resolve

        cfg = load_config(resolve("configs/unet_baseline.yaml"))
        for k in ("patch_size", "sampler_ratios", "channels", "strides",
                  "checkpoint_metric", "sw_roi_size"):
            self.assertIn(k, cfg)
        self.assertEqual(cfg["checkpoint_metric"], "macro_foreground_dice")
        self.assertEqual(cfg["out_channels"], 3)

    def test_checkpoint_save_load_resume(self):
        from src.models.unet import build_unet
        from src.training.trainer import Trainer, config_hash

        cfg = toy_cfg()
        with tempfile.TemporaryDirectory() as d:
            tr = Trainer(cfg, build_unet(cfg), [], [], Path(d))
            tr.best_macro, tr.best_epoch = 0.5, 3
            tr.save("best.pt", 3)
            tr2 = Trainer(cfg, build_unet(cfg), [], [], Path(d))
            ep = tr2.load(Path(d) / "best.pt")
            self.assertEqual(ep, 4)
            self.assertEqual(tr2.best_epoch, 3)
            self.assertEqual(tr2.best_macro, 0.5)
            # stale config rejected
            bad = toy_cfg(lr=9e-4)
            tr3 = Trainer(bad, build_unet(bad), [], [], Path(d))
            with self.assertRaises(AssertionError):
                tr3.load(Path(d) / "best.pt")
            self.assertNotEqual(config_hash(cfg), config_hash(bad))

    def test_split_exclusion(self):
        from src.data.preprocess import repo_split_ids
        from src.training import data as D

        tr, va, te = (set(repo_split_ids(s)) for s in ("train", "val", "test"))
        self.assertEqual((len(tr), len(va), len(te)), (197, 42, 42))
        self.assertFalse(tr & va or tr & te or va & te)
        with self.assertRaises(ValueError):
            D.load_split_items("test", Path("."))

    def test_sampler_valid_labels(self):
        from src.training.data import RandomizedView, training_random_stage

        items = [toy_pair() for _ in range(2)]
        view = RandomizedView(items, training_random_stage(toy_cfg()))
        got = view[0]
        self.assertEqual(tuple(got["image"].shape), (1, 16, 16, 16))
        self.assertTrue(set(torch.unique(got["label"]).tolist()) <= {0, 1, 2})

    def test_val_transforms_deterministic(self):
        import tempfile

        import nibabel as nib
        from src.data import preprocess as P

        import monai.transforms as T

        det = T.Compose([P.build_monai_inference_transforms(toy_cfg()),
                         P._T(T, "ToTensord", "ToTensorD")(keys=["image", "label"])])
        pair = toy_pair()
        with tempfile.TemporaryDirectory() as d:
            nib.save(nib.Nifti1Image(np.asarray(pair["image"][0]), np.eye(4)),
                     f"{d}/img.nii.gz")
            nib.save(nib.Nifti1Image(np.asarray(pair["label"][0].numpy()).astype(np.uint8),
                                     np.eye(4)), f"{d}/lbl.nii.gz")
            inp = {"image": f"{d}/img.nii.gz", "label": f"{d}/lbl.nii.gz"}
            a = det(dict(inp))
            b = det(dict(inp))
        torch.testing.assert_close(a["image"], b["image"])
        torch.testing.assert_close(a["label"].float(), b["label"].float())

    def test_selection_logic(self):
        from src.training.metrics import is_improvement

        self.assertTrue(is_improvement(0.5, 0.4))
        self.assertFalse(is_improvement(0.4005, 0.4))  # within min_delta
        self.assertFalse(is_improvement(0.4, 0.5))


if __name__ == "__main__":
    unittest.main()

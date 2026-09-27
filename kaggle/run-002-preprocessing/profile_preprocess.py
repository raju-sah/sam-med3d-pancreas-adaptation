"""Phase 2 Kaggle CPU job: TRAIN-ONLY profiling, figures, resolve config, val smoke test.

Isolation: train.json IDs may be profiled/visualized. val.json IDs: only the
first 3 (sorted) enter the post-freeze transform smoke test. test.json is
read SOLELY to assert exclusion (volumes never opened).

Stages:
  1. train-only aggregate profile -> phase2_train_profile.{json,csv}
  2. deterministic representative-case selection -> representative_cases.json
  3. train-only sanity figures -> figures/phase2/*.png
  4. resolve spacing/intensity per ADR 002 pre-registered rules -> resolved_preprocess.json
  5. val smoke test with the REAL monai pipeline -> val_smoke.json
"""

from __future__ import annotations

import csv
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

OUT = Path("/kaggle/working")
FIG = OUT / "figures" / "phase2"
SEED = 42
CLAMP = (-1000.0, 1000.0)
UPSTREAM_SPACING = (1.5, 1.5, 1.5)
VALID_LABELS = (0, 1, 2)

# Split IDs embedded verbatim from data/splits/*.json at commit 9e79dfc
# (Phase 1 frozen splits; counts asserted at runtime). Volumes for test IDs
# below are NEVER opened — used only for exclusion checks.
TRAIN_IDS = ["pancreas_001", "pancreas_004", "pancreas_005", "pancreas_010", "pancreas_012", "pancreas_015", "pancreas_016", "pancreas_018", "pancreas_019", "pancreas_021", "pancreas_029", "pancreas_041", "pancreas_043", "pancreas_045", "pancreas_051", "pancreas_052", "pancreas_055", "pancreas_056", "pancreas_061", "pancreas_064", "pancreas_066", "pancreas_067", "pancreas_069", "pancreas_070", "pancreas_071", "pancreas_075", "pancreas_078", "pancreas_080", "pancreas_083", "pancreas_084", "pancreas_087", "pancreas_089", "pancreas_091", "pancreas_094", "pancreas_096", "pancreas_101", "pancreas_102", "pancreas_103", "pancreas_104", "pancreas_105", "pancreas_106", "pancreas_107", "pancreas_111", "pancreas_113", "pancreas_117", "pancreas_119", "pancreas_124", "pancreas_125", "pancreas_126", "pancreas_127", "pancreas_129", "pancreas_131", "pancreas_135", "pancreas_137", "pancreas_138", "pancreas_145", "pancreas_148", "pancreas_160", "pancreas_167", "pancreas_169", "pancreas_170", "pancreas_173", "pancreas_175", "pancreas_178", "pancreas_179", "pancreas_180", "pancreas_181", "pancreas_182", "pancreas_183", "pancreas_186", "pancreas_193", "pancreas_196", "pancreas_199", "pancreas_201", "pancreas_203", "pancreas_204", "pancreas_207", "pancreas_209", "pancreas_211", "pancreas_212", "pancreas_213", "pancreas_214", "pancreas_215", "pancreas_217", "pancreas_218", "pancreas_219", "pancreas_222", "pancreas_224", "pancreas_225", "pancreas_227", "pancreas_228", "pancreas_234", "pancreas_235", "pancreas_236", "pancreas_239", "pancreas_242", "pancreas_243", "pancreas_244", "pancreas_246", "pancreas_247", "pancreas_249", "pancreas_255", "pancreas_256", "pancreas_259", "pancreas_262", "pancreas_265", "pancreas_266", "pancreas_268", "pancreas_269", "pancreas_270", "pancreas_275", "pancreas_277", "pancreas_278", "pancreas_280", "pancreas_283", "pancreas_284", "pancreas_285", "pancreas_289", "pancreas_290", "pancreas_291", "pancreas_292", "pancreas_293", "pancreas_295", "pancreas_296", "pancreas_297", "pancreas_298", "pancreas_299", "pancreas_300", "pancreas_301", "pancreas_302", "pancreas_304", "pancreas_308", "pancreas_309", "pancreas_310", "pancreas_311", "pancreas_312", "pancreas_313", "pancreas_315", "pancreas_316", "pancreas_320", "pancreas_321", "pancreas_323", "pancreas_325", "pancreas_326", "pancreas_328", "pancreas_330", "pancreas_333", "pancreas_334", "pancreas_336", "pancreas_342", "pancreas_343", "pancreas_344", "pancreas_345", "pancreas_346", "pancreas_348", "pancreas_350", "pancreas_354", "pancreas_355", "pancreas_356", "pancreas_358", "pancreas_360", "pancreas_361", "pancreas_362", "pancreas_364", "pancreas_366", "pancreas_367", "pancreas_369", "pancreas_370", "pancreas_372", "pancreas_375", "pancreas_377", "pancreas_378", "pancreas_379", "pancreas_380", "pancreas_382", "pancreas_385", "pancreas_386", "pancreas_387", "pancreas_392", "pancreas_395", "pancreas_398", "pancreas_399", "pancreas_400", "pancreas_401", "pancreas_402", "pancreas_404", "pancreas_405", "pancreas_406", "pancreas_409", "pancreas_410", "pancreas_412", "pancreas_413", "pancreas_415", "pancreas_416", "pancreas_418", "pancreas_419", "pancreas_421"]
VAL_IDS = ["pancreas_037", "pancreas_040", "pancreas_042", "pancreas_049", "pancreas_058", "pancreas_077", "pancreas_088", "pancreas_093", "pancreas_099", "pancreas_100", "pancreas_110", "pancreas_120", "pancreas_122", "pancreas_149", "pancreas_157", "pancreas_158", "pancreas_159", "pancreas_165", "pancreas_197", "pancreas_198", "pancreas_226", "pancreas_230", "pancreas_241", "pancreas_254", "pancreas_258", "pancreas_261", "pancreas_264", "pancreas_267", "pancreas_274", "pancreas_276", "pancreas_286", "pancreas_287", "pancreas_294", "pancreas_305", "pancreas_327", "pancreas_339", "pancreas_347", "pancreas_357", "pancreas_365", "pancreas_376", "pancreas_388", "pancreas_414"]
TEST_IDS = ["pancreas_006", "pancreas_024", "pancreas_025", "pancreas_028", "pancreas_032", "pancreas_035", "pancreas_046", "pancreas_048", "pancreas_050", "pancreas_074", "pancreas_081", "pancreas_086", "pancreas_092", "pancreas_095", "pancreas_098", "pancreas_109", "pancreas_114", "pancreas_130", "pancreas_140", "pancreas_147", "pancreas_155", "pancreas_166", "pancreas_172", "pancreas_187", "pancreas_191", "pancreas_194", "pancreas_200", "pancreas_210", "pancreas_229", "pancreas_231", "pancreas_253", "pancreas_279", "pancreas_303", "pancreas_318", "pancreas_329", "pancreas_331", "pancreas_351", "pancreas_374", "pancreas_389", "pancreas_391", "pancreas_393", "pancreas_411"]


def validate_labels(mask, valid=VALID_LABELS):
    present = set(int(v) for v in np.unique(mask))
    bad = present - set(valid)
    if bad:
        raise ValueError(f"invalid labels {sorted(bad)}")
    return present


def check_alignment(image, label):
    if image.shape[-3:] != label.shape[-3:]:
        raise ValueError(f"shape mismatch {image.shape} vs {label.shape}")
    return tuple(label.shape)


def _T(mod, *names):
    for n in names:
        if hasattr(mod, n):
            return getattr(mod, n)
    raise AttributeError(f"none of {names} in monai.transforms")


def build_monai_inference_transforms(cfg):
    import monai.transforms as T

    LoadImaged = _T(T, "LoadImaged", "LoadImageD")
    ChannelFirstd = _T(T, "EnsureChannelFirstd", "EnsureChannelFirstD")
    Orientationd = _T(T, "Orientationd", "OrientationD")
    Spacingd = _T(T, "Spacingd", "SpacingD")
    ScaleRanged = _T(T, "ScaleIntensityRangeD", "ScaleIntensityRanged")
    Normd = _T(T, "NormalizeIntensityd", "NormalizeIntensityD")
    pre = [
        LoadImaged(keys=["image", "label"]),
        ChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes=cfg["orientation"]),
    ]
    if cfg.get("target_spacing"):
        pre.append(Spacingd(keys=["image", "label"], pixdim=tuple(cfg["target_spacing"]),
                            mode=(cfg.get("image_mode", "bilinear"),
                                  cfg.get("label_mode", "nearest"))))
    pre += [ScaleRanged(keys=["image"], a_min=cfg["clamp_min"], a_max=cfg["clamp_max"],
                            b_min=cfg["clamp_min"], b_max=cfg["clamp_max"], clip=True),
            Normd(keys=["image"], nonzero=True, channel_wise=True)]
    return T.Compose(pre)
def ensure_monai():
    try:
        import monai  # noqa: F401
        return
    except ImportError:
        pass
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "monai"])
    import monai  # noqa: F401


def find_task_root() -> Path:
    for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
        if "dataset.json" in filenames:
            return Path(dirpath)
        if len(Path(dirpath).relative_to("/kaggle/input").parts) > 5:
            dirnames[:] = []
    raise FileNotFoundError("task root not found")


def load(p: Path):
    img = nib.load(str(p))
    return img


def profile_case(img_path: Path, lbl_path: Path) -> dict:
    img_o = load(img_path)
    lbl_o = load(lbl_path)
    shape = tuple(img_o.shape)
    assert tuple(lbl_o.shape) == shape, "image/label shape mismatch"
    zooms = tuple(round(float(z), 4) for z in img_o.header.get_zooms()[:3])
    codes = "".join(nib.orientations.aff2axcodes(img_o.affine))
    vox_vol = float(np.prod(img_o.header.get_zooms()[:3]))
    ldata = np.asarray(lbl_o.dataobj)
    validate_labels(ldata)
    uniq, counts = np.unique(ldata, return_counts=True)
    cmap = dict(zip([int(v) for v in uniq], [int(c) for c in counts]))
    n_pan, n_tum = cmap.get(1, 0), cmap.get(2, 0)
    idata = np.asarray(img_o.dataobj, dtype=np.float64)
    flat = idata[::7, ::7, ::7].ravel()  # deterministic strided sample for percentiles
    pct = np.percentile(flat, [0.5, 1, 5, 25, 50, 75, 95, 99, 99.5])
    return {
        "shape": "x".join(map(str, shape)),
        "spacing": "x".join(map(str, zooms)),
        "sx": zooms[0],
        "sy": zooms[1],
        "sz": zooms[2],
        "orientation": codes,
        "img_min": float(idata.min()),
        "img_max": float(idata.max()),
        "img_mean": round(float(idata.mean()), 3),
        "img_std": round(float(idata.std()), 3),
        "p0.5": round(float(pct[0]), 1),
        "p1": round(float(pct[1]), 1),
        "p5": round(float(pct[2]), 1),
        "p25": round(float(pct[3]), 1),
        "p50": round(float(pct[4]), 1),
        "p75": round(float(pct[5]), 1),
        "p95": round(float(pct[6]), 1),
        "p99": round(float(pct[7]), 1),
        "p99.5": round(float(pct[8]), 1),
        "frac_outside_clamp": round(float(((idata < CLAMP[0]) | (idata > CLAMP[1])).mean()), 6),
        "n_pancreas": n_pan,
        "n_tumor": n_tum,
        "n_fg": n_pan + n_tum,
        "pan_mm3": round(n_pan * vox_vol, 1),
        "tum_mm3": round(n_tum * vox_vol, 1),
        "fg_frac": round((n_pan + n_tum) / ldata.size, 6),
    }


def summarize(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=np.float64)
    out = {
        "min": round(float(a.min()), 4),
        "max": round(float(a.max()), 4),
        "mean": round(float(a.mean()), 4),
        "std": round(float(a.std()), 4),
        "median": round(float(np.median(a)), 4),
    }
    for p in (1, 5, 25, 75, 95, 99):
        out[f"p{p}"] = round(float(np.percentile(a, p)), 4)
    return out


def save_figure(case_id, img, lbl, pre_img, pre_lbl, out_path):
    """3 views x [raw, overlay, pancreas, tumor]; 2nd fig raw-vs-preprocessed.

    nibabel voxel axes for RAS data are (x=R-L, y=A-P, z=S-I), so axis 0
    slices are sagittal, axis 1 coronal, axis 2 axial.
    """
    fg = np.argwhere(lbl > 0)
    mx, my, mz = [int(c) for c in fg.mean(axis=0)] if len(fg) else [s // 2 for s in lbl.shape]
    t = np.argwhere(lbl == 2)
    tx, ty, tz = [int(c) for c in t.mean(axis=0)] if len(t) else (mx, my, mz)
    sh = lbl.shape
    mx, my, mz = (int(np.clip(v, 0, s - 1)) for v, s in zip((mx, my, mz), sh))
    tx, ty, tz = (int(np.clip(v, 0, s - 1)) for v, s in zip((tx, ty, tz), sh))
    disp = np.clip(img, -150, 250)  # display window only, documented
    views = [("sagittal", 0, mx, tx), ("coronal", 1, my, ty), ("axial", 2, mz, tz)]
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for r, (name, ax, c, tc) in enumerate(views):
        raw = np.take(disp, c, axis=ax)
        ov = np.take(lbl, c, axis=ax)
        pan = (ov == 1)
        tum = (ov == 2)
        axes[r][0].imshow(raw, cmap="gray")
        axes[r][0].set_title(f"{name} raw")
        axes[r][1].imshow(raw, cmap="gray")
        axes[r][1].imshow(np.ma.masked_where(ov == 0, ov), alpha=0.45, cmap="jet", vmin=0, vmax=2)
        axes[r][1].set_title(f"{name} overlay")
        axes[r][2].imshow(pan, cmap="Greens")
        axes[r][2].set_title(f"{name} pancreas")
        axes[r][3].imshow(tum, cmap="Reds")
        axes[r][3].set_title(f"{name} mass/tumor")
        for a in axes[r]:
            a.axis("off")
    fig.suptitle(f"{case_id} (TRAIN)")
    fig.tight_layout()
    fig.savefig(out_path / f"{case_id}_overlay.png", dpi=100)
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(8, 9))
    # raw and preprocessed live on different grids: slice each at its own middle
    rmx, rmy, rmz = [s // 2 for s in disp.shape]
    pmx, pmy, pmz = [s // 2 for s in pre_img.shape]
    for r, (name, ax) in enumerate((("sagittal", 0), ("coronal", 1), ("axial", 2))):
        rc, pc = (rmx, rmy, rmz)[r], (pmx, pmy, pmz)[r]
        axes[r][0].imshow(np.take(disp, rc, axis=ax), cmap="gray")
        axes[r][0].set_title(f"{name} raw")
        axes[r][1].imshow(np.take(pre_img, pc, axis=ax), cmap="gray")
        axes[r][1].set_title(f"{name} preprocessed")
        for a in axes[r]:
            a.axis("off")
    fig.suptitle(f"{case_id} raw vs preprocessed (TRAIN)")
    fig.tight_layout()
    fig.savefig(out_path / f"{case_id}_prepost.png", dpi=100)
    plt.close(fig)


def main() -> int:
    root = find_task_root()
    train_ids, val_ids, test_ids = list(TRAIN_IDS), list(VAL_IDS), list(TEST_IDS)
    assert len(train_ids) == 197 and len(val_ids) == 42 and len(test_ids) == 42
    assert not (set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids)
                or set(val_ids) & set(test_ids))
    print(f"isolation OK: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}",
          flush=True)

    # Fail-fast: construct the REAL monai pipeline and dry-run it on synthetic
    # NIfTI BEFORE the 25-min profiling, so transform-name issues surface early.
    ensure_monai()
    import monai  # noqa: E402

    print(f"monai version: {monai.__version__}", flush=True)
    import torch  # noqa: E402

    synth_img = np.random.RandomState(0).uniform(-500, 500, size=(32, 32, 32)).astype(np.float32)
    synth_lbl = np.zeros((32, 32, 32), dtype=np.uint8)
    synth_lbl[8:16, 8:16, 8:16] = 1
    synth_lbl[20:24, 20:24, 20:24] = 2
    nib.save(nib.Nifti1Image(synth_img, np.eye(4)), "/tmp/synth_img.nii.gz")
    nib.save(nib.Nifti1Image(synth_lbl, np.eye(4)), "/tmp/synth_lbl.nii.gz")
    dry = build_monai_inference_transforms({
        "orientation": "RAS", "target_spacing": list(UPSTREAM_SPACING),
        "image_mode": "bilinear", "label_mode": "nearest",
        "clamp_min": CLAMP[0], "clamp_max": CLAMP[1]})
    dd = dry({"image": "/tmp/synth_img.nii.gz", "label": "/tmp/synth_lbl.nii.gz"})
    assert dd["image"].shape[1:] == dd["label"].shape[1:]
    assert set(torch.unique(dd["label"]).tolist()) <= {0, 1, 2}
    assert bool(torch.isfinite(dd["image"].float()).all())
    print(f"monai dry-run OK {list(dd['image'].shape)}", flush=True)
    rows: list[dict] = []
    for i, cid in enumerate(train_ids):
        ip = root / "imagesTr" / f"{cid}.nii"
        lp = root / "labelsTr" / f"{cid}.nii"
        if not ip.is_file():
            ip = root / "imagesTr" / f"{cid}.nii.gz"
        if not lp.is_file():
            lp = root / "labelsTr" / f"{cid}.nii.gz"
        r = {"case_id": cid}
        r.update(profile_case(ip, lp))
        rows.append(r)
        if (i + 1) % 50 == 0:
            print(f"profiled {i + 1}/{len(train_ids)}", flush=True)

    prof = {
        "n_train": len(rows),
        "seed": SEED,
        "intensity_method": "exact min/max/mean/std per volume; percentiles on "
        "deterministic strided sample (::7 each axis, ~0.3% voxels)",
        "shape": summarize([int(r["shape"].split("x")[2]) for r in rows]),
        "spacing_x": summarize([r["sx"] for r in rows]),
        "spacing_y": summarize([r["sy"] for r in rows]),
        "spacing_z": summarize([r["sz"] for r in rows]),
        "img_min": summarize([r["img_min"] for r in rows]),
        "img_max": summarize([r["img_max"] for r in rows]),
        "img_mean": summarize([r["img_mean"] for r in rows]),
        "img_std": summarize([r["img_std"] for r in rows]),
        "frac_outside_clamp": summarize([r["frac_outside_clamp"] for r in rows]),
        "n_pancreas": summarize([r["n_pancreas"] for r in rows]),
        "n_tumor": summarize([r["n_tumor"] for r in rows]),
        "pan_mm3": summarize([r["pan_mm3"] for r in rows]),
        "tum_mm3": summarize([r["tum_mm3"] for r in rows]),
        "fg_frac": summarize([r["fg_frac"] for r in rows]),
        "orientations": sorted({r["orientation"] for r in rows}),
    }
    for p in ("p0.5", "p1", "p5", "p50", "p95", "p99", "p99.5"):
        prof[p] = summarize([r[p] for r in rows])

    with open(OUT / "phase2_train_profile.json", "w") as f:
        json.dump(prof, f, indent=1)
    with open(OUT / "phase2_train_profile.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # --- representative selection (deterministic, documented reasons) ---
    by_tum = sorted(rows, key=lambda r: r["n_tumor"])
    by_pan = sorted(rows, key=lambda r: r["n_pancreas"])
    rng = random.Random(SEED)
    picks = [
        (by_tum[0]["case_id"], "smallest label-2 volume"),
        (by_tum[len(by_tum) // 2]["case_id"], "median label-2 volume"),
        (by_tum[-1]["case_id"], "largest label-2 volume"),
        (by_pan[0]["case_id"], "smallest pancreas"),
        (by_pan[-1]["case_id"], "largest pancreas"),
        (rng.choice(train_ids), "fixed-seed random train case"),
    ]
    seen, rep = set(), []
    for cid, why in picks:
        if cid not in seen:
            seen.add(cid)
            rep.append({"case_id": cid, "reason": why})
    with open(OUT / "representative_cases.json", "w") as f:
        json.dump(rep, f, indent=1)

    # --- spacing / intensity resolution per ADR 002 rules ---
    min_tum_vox_at_15 = prof["n_tumor"]["min"] * (prof["spacing_x"]["median"]
                                                 * prof["spacing_y"]["median"]
                                                 * prof["spacing_z"]["median"]) / (1.5 ** 3)
    max_out = prof["frac_outside_clamp"]["max"]
    p995_max = prof["p99.5"]["max"]
    # Clamp rule refined with evidence: [-1000,1000] is FROZEN iff the entire
    # train soft-tissue range is preserved (per-case p99.5 max < 1000) — the
    # clamped-out voxels are then only air (<-1000) and bone/metal (>1000),
    # matching upstream CT practice. p50/p95/p99 reported as context.
    clamp_ok = p995_max < 1000.0
    spacing_choice = "1.5_iso" if min_tum_vox_at_15 >= 50 else "native"
    resolved = {
        "orientation": "RAS",
        "spacing_policy": spacing_choice,
        "target_spacing": list(UPSTREAM_SPACING) if spacing_choice == "1.5_iso" else None,
        "spacing_evidence": {
            "min_tumor_voxels_at_1.5mm": round(min_tum_vox_at_15, 1),
            "rule": "1.5_iso iff smallest tumor keeps >= 50 voxels after resampling",
            "median_spacing": [prof["spacing_x"]["median"], prof["spacing_y"]["median"],
                               prof["spacing_z"]["median"]],
        },
        "clamp": list(CLAMP) if clamp_ok else "UNRESOLVED",
        "clamp_evidence": {
            "max_train_frac_outside": max_out,
            "percase_p99.5_max": p995_max,
            "rule": "[-1000,1000] iff per-case p99.5 max < 1000 (soft tissue "
            "preserved; outliers are air/bone/metal only)",
        },
        "norm": "fg_zscore",
        "image_mode": "bilinear",
        "label_mode": "nearest",
        "valid_labels": [0, 1, 2],
        "seed": SEED,
    }
    with open(OUT / "resolved_preprocess.json", "w") as f:
        json.dump(resolved, f, indent=1)
    print(json.dumps(resolved, indent=1), flush=True)

    # --- figures (train only) ---
    FIG.mkdir(parents=True, exist_ok=True)
    from scipy.ndimage import zoom

    for item in rep:
        cid = item["case_id"]
        ip = root / "imagesTr" / f"{cid}.nii"
        lp = root / "labelsTr" / f"{cid}.nii"
        img = np.asarray(nib.load(str(ip)).dataobj, dtype=np.float32)
        lbl = np.asarray(nib.load(str(lp)).dataobj)
        validate_labels(lbl)
        check_alignment(img[np.newaxis], lbl)
        if resolved["target_spacing"]:
            # this case's spacing -> zoom factors to 1.5mm iso
            sp = next(r for r in rows if r["case_id"] == cid)
            zf = (sp["sx"] / 1.5, sp["sy"] / 1.5, sp["sz"] / 1.5)
            pre_img = zoom(np.clip(img, *CLAMP), zf, order=1)
            pre_lbl = zoom(lbl, zf, order=0)
            validate_labels(pre_lbl)
        else:
            pre_img, pre_lbl = np.clip(img, *CLAMP), lbl
        fg = pre_img > 0
        pre_img = (pre_img - pre_img[fg].mean()) / (pre_img[fg].std() + 1e-8)
        save_figure(cid, img, lbl, pre_img, pre_lbl, FIG)
        print(f"figured {cid}", flush=True)

    # --- val smoke test with REAL monai pipeline (post-freeze, correctness only) ---
    ensure_monai()
    import monai  # noqa: E402

    print(f"monai version: {monai.__version__}", flush=True)
    import monai.transforms as T  # noqa: E402

    cfg = {
        "orientation": "RAS",
        "target_spacing": resolved["target_spacing"],
        "image_mode": "bilinear",
        "label_mode": "nearest",
        "clamp_min": CLAMP[0],
        "clamp_max": CLAMP[1],
    }
    t = build_monai_inference_transforms(cfg)
    smoke = []
    for cid in sorted(val_ids)[:3]:
        ip = root / "imagesTr" / f"{cid}.nii"
        lp = root / "labelsTr" / f"{cid}.nii"
        d = t({"image": str(ip), "label": str(lp)})
        im, lb = d["image"], d["label"]
        assert im.shape[1:] == lb.shape[1:], "geometry mismatch"
        assert set(torch.unique(lb).tolist()) <= {0, 1, 2}, "label corruption"
        assert bool(torch.isfinite(im.float()).all()), "NaN/Inf in image"
        smoke.append({"case_id": cid, "out_shape": list(im.shape),
                      "labels": sorted(torch.unique(lb).tolist())})
        print(f"smoke OK {cid} {list(im.shape)}", flush=True)
    with open(OUT / "val_smoke.json", "w") as f:
        json.dump(smoke, f, indent=1)
    print("PHASE2 KERNEL DONE", flush=True)
    return 0


if __name__ == "__main__":
    import torch  # local import: needed only for smoke-test asserts

    sys.exit(main())

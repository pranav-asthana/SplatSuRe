"""
eval_unsparse_scenes.py
------------------------
Evaluate predictions for the 4 scenes that did NOT ship with sparse/ data
(aeroplane, cycle, face, still3) — i.e. the scenes where you ran your own
SfM (LoMa/LofTR) and need to verify the whole pipeline works.

Reports:
  * Per-scene PSNR / SSIM / LPIPS (optional)
  * Per-scene COMPETITION SCORE: 0.5 * SSIM + 0.5 * (PSNR / 30)
  * Macro-mean across scenes (the leaderboard metric is image-mean of the
    competition score, but image-vs-scene macro can differ when scenes have
    very different image counts — both are reported)
  * Worst N images globally (for debugging which views to fix)
  * Optional CSV of per-image scores

Expected layout
---------------
  <dataset_root>/<scene>/images/<file>            <- LR ground truth (every 8th held out)
  <pred_root>/<scene>/<file>                       <- your prediction with same filename

Usage
-----
python eval_unsparse_scenes.py \
    --dataset-root /home/pranav-htic/3DGS-SR/RealSr4x_Swin \
    --pred-root    /home/pranav-htic/3DGS-SR/Splatsure-v7-final/submissions/ \
    --scenes       aeroplane \
    --gt-folder    images \
    --save-csv     aeroplane_lr_diagnostic.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

try:
    import torch
    import lpips as lpips_lib
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False


IMAGE_EXTS = {".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"}
DEFAULT_SCENES = ["aeroplane", "cycle", "face", "still3"]


# ── helpers ───────────────────────────────────────────────────────────────────

def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    imgs = [p for p in folder.iterdir() if p.is_file() and p.suffix in IMAGE_EXTS]
    imgs.sort(key=lambda p: (p.stem.lower(), p.suffix.lower()))
    return imgs


def get_test_images(scene_dir: Path, gt_folder: str, llffhold: int) -> list[Path]:
    """Every Nth image (LLFF convention)."""
    all_imgs = list_images(scene_dir / gt_folder)
    return [img for idx, img in enumerate(all_imgs) if idx % llffhold == 0]


def load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def resize_to(img: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    H, W = target_hw
    if img.shape[:2] == (H, W):
        return img
    pil = Image.fromarray((img * 255.0).clip(0, 255).astype(np.uint8))
    return np.asarray(pil.resize((W, H), Image.BICUBIC), dtype=np.float32) / 255.0


def competition_score(psnr: float, ssim: float) -> float:
    """0.5 * SSIM + 0.5 * (PSNR / 30) — clamped to [0, 1] roughly."""
    return 0.5 * ssim + 0.5 * (psnr / 30.0)


def compute_metrics(render: np.ndarray, gt: np.ndarray, lpips_model=None) -> dict:
    psnr = float(psnr_metric(gt, render, data_range=1.0))
    ssim = float(ssim_metric(gt, render, channel_axis=2, data_range=1.0))
    out = {"psnr": psnr, "ssim": ssim, "score": competition_score(psnr, ssim)}
    if lpips_model is not None:
        with torch.no_grad():
            r = torch.from_numpy(render).permute(2, 0, 1).unsqueeze(0).cuda() * 2 - 1
            g = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).cuda() * 2 - 1
            out["lpips"] = float(lpips_model(r, g).item())
    return out


# ── per-scene evaluation ─────────────────────────────────────────────────────

def evaluate_scene(scene_name: str, scene_dir: Path, pred_dir: Path,
                   llffhold: int, gt_folder: str,
                   match_render_res: bool, lpips_model=None) -> dict:
    """Returns per-image rows + scene-level summary."""
    test_gts = get_test_images(scene_dir, gt_folder, llffhold)
    if not test_gts:
        return {"scene": scene_name, "rows": [], "missing": [], "error": "no test images"}

    pred_files = {p.stem.lower(): p for p in list_images(pred_dir)} if pred_dir.exists() else {}

    rows, missing = [], []
    for gt_path in test_gts:
        pred_path = pred_files.get(gt_path.stem.lower())
        if pred_path is None:
            missing.append(gt_path.name)
            continue

        gt     = load_image(gt_path)
        render = load_image(pred_path)
        if gt.shape[:2] != render.shape[:2]:
            if match_render_res:
                gt = resize_to(gt, render.shape[:2])
            else:
                render = resize_to(render, gt.shape[:2])

        m = compute_metrics(render, gt, lpips_model)
        m["scene"]    = scene_name
        m["filename"] = gt_path.name
        rows.append(m)

    return {"scene": scene_name, "rows": rows, "missing": missing}


def summarize_scene(scene_result: dict) -> dict:
    rows = scene_result["rows"]
    if not rows:
        return {"scene": scene_result["scene"], "n": 0}
    psnrs   = np.array([r["psnr"]   for r in rows])
    ssims   = np.array([r["ssim"]   for r in rows])
    scores  = np.array([r["score"]  for r in rows])
    out = {
        "scene":      scene_result["scene"],
        "n":          len(rows),
        "psnr_mean":  float(psnrs.mean()),
        "ssim_mean":  float(ssims.mean()),
        "score_mean": float(scores.mean()),
        "score_min":  float(scores.min()),
        "score_max":  float(scores.max()),
    }
    if "lpips" in rows[0]:
        out["lpips_mean"] = float(np.mean([r["lpips"] for r in rows]))
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Evaluate predictions on the 4 SfM-only scenes")
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="Root containing <scene>/images/")
    p.add_argument("--pred-root",    type=Path, required=True,
                   help="Root containing <scene>/<pred>.png predictions")
    p.add_argument("--scenes",       nargs="+", default=DEFAULT_SCENES,
                   help=f"Scenes to evaluate. Default: {DEFAULT_SCENES}")
    p.add_argument("--gt-folder",    type=str, default="images")
    p.add_argument("--llffhold",     type=int, default=8)
    p.add_argument("--match-render-res", action="store_true",
                   help="Upsample GT to render res. Default: downscale render to GT res "
                        "(matches what most graders do).")
    p.add_argument("--lpips",        action="store_true")
    p.add_argument("--worst-n",      type=int, default=5,
                   help="Print N worst-scoring images globally (default 5; 0 to disable)")
    p.add_argument("--save-csv",     type=Path, default=None)
    args = p.parse_args()

    # LPIPS model (optional)
    lpips_model = None
    if args.lpips:
        if not LPIPS_AVAILABLE:
            print("[WARN] --lpips requested but `lpips` not installed. Skipping.")
        else:
            print("Loading LPIPS (alex)…")
            lpips_model = lpips_lib.LPIPS(net='alex').cuda().eval()

    # Per-scene evaluation
    print(f"\n{'='*72}")
    print(f"Evaluating {len(args.scenes)} scenes (llffhold={args.llffhold})")
    print(f"  dataset : {args.dataset_root}")
    print(f"  preds   : {args.pred_root}")
    print(f"  scenes  : {args.scenes}")
    print(f"{'='*72}")

    all_rows: list[dict] = []
    summaries: list[dict] = []

    for scene_name in args.scenes:
        scene_dir = args.dataset_root / scene_name
        pred_dir  = args.pred_root    / scene_name

        if not scene_dir.exists():
            print(f"\n[SKIP] {scene_name}: dataset dir missing -> {scene_dir}")
            continue
        if not pred_dir.exists():
            print(f"\n[SKIP] {scene_name}: pred dir missing -> {pred_dir}")
            continue

        result = evaluate_scene(scene_name, scene_dir, pred_dir,
                                args.llffhold, args.gt_folder,
                                args.match_render_res, lpips_model)
        all_rows.extend(result["rows"])
        s = summarize_scene(result)
        summaries.append(s)

        if s["n"] == 0:
            print(f"\n[{scene_name}] no paired images")
            continue

        # Resolution sanity check (one-line)
        sample_gt   = load_image(get_test_images(scene_dir, args.gt_folder, args.llffhold)[0])
        sample_pred = load_image(list_images(pred_dir)[0])
        gt_hw, pr_hw = sample_gt.shape[:2], sample_pred.shape[:2]
        res_note = "" if gt_hw == pr_hw else \
            f" | res mismatch: GT {gt_hw[1]}x{gt_hw[0]} vs pred {pr_hw[1]}x{pr_hw[0]}"

        lpips_str = f" | LPIPS {s['lpips_mean']:.4f}" if "lpips_mean" in s else ""
        print(f"\n[{s['scene']:>10}]  N={s['n']:>3}  "
              f"PSNR {s['psnr_mean']:.3f}  SSIM {s['ssim_mean']:.4f}  "
              f"SCORE {s['score_mean']:.4f}  "
              f"(min {s['score_min']:.4f}, max {s['score_max']:.4f}){lpips_str}{res_note}")
        if result["missing"]:
            preview = result["missing"][:3]
            ell = "…" if len(result["missing"]) > 3 else ""
            print(f"             [WARN] missing renders for {len(result['missing'])} images: {preview}{ell}")

    # Aggregate
    if not all_rows:
        print("\nNo images evaluated. Check paths.")
        return 1

    print(f"\n{'='*72}")
    print(f"AGGREGATE")
    print(f"{'='*72}")

    psnrs   = np.array([r["psnr"]  for r in all_rows])
    ssims   = np.array([r["ssim"]  for r in all_rows])
    scores  = np.array([r["score"] for r in all_rows])
    print(f"  Total images          : {len(all_rows)}")
    print(f"  Image-mean PSNR       : {psnrs.mean():.3f}")
    print(f"  Image-mean SSIM       : {ssims.mean():.4f}")
    print(f"  Image-mean SCORE      : {scores.mean():.4f}    "
          f"<-- proxy for leaderboard contribution from these 4 scenes")
    macro_score = float(np.mean([s["score_mean"] for s in summaries if s["n"] > 0]))
    print(f"  Scene-macro SCORE     : {macro_score:.4f}    "
          f"(unweighted mean over scenes — useful when scene sizes differ)")

    # Worst images
    if args.worst_n > 0:
        print(f"\n  Worst {args.worst_n} images by competition score:")
        worst = sorted(all_rows, key=lambda r: r["score"])[:args.worst_n]
        for r in worst:
            print(f"    {r['scene']:>10}/{r['filename']:<25} "
                  f"PSNR {r['psnr']:.2f}  SSIM {r['ssim']:.4f}  SCORE {r['score']:.4f}")

    # CSV
    if args.save_csv:
        fieldnames = ["scene", "filename", "psnr", "ssim", "score"]
        if "lpips" in all_rows[0]:
            fieldnames.append("lpips")
        with open(args.save_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in all_rows:
                w.writerow({k: r.get(k) for k in fieldnames})
        print(f"\n  Saved CSV : {args.save_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

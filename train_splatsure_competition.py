#!/usr/bin/env python3
"""
train_splatsure_competition.py

Competition-oriented launcher for SplatSuRe training on custom COLMAP scenes.

How SplatSuRe differs from SRGS
--------------------------------
SplatSuRe uses a THREE-STEP pipeline per scene (per official repo):

  Step 1 – LR training  [train_lr.py]
    Trains a vanilla 3DGS on your low-resolution `images/` folder.

  Step 2 – Weight map generation  [weight_maps.py]
    Uses the trained LR model to produce per-view weight maps that encode
    which regions are undersampled. These maps guide selective SR in Step 3.

  Step 3 – SR training  [train.py]
    Re-trains 3DGS using the SR images (`images_SR/`) as the high-resolution
    supervision target, weighted by the maps from Step 2.
    This is what the paper calls "selective SR".

Dataset layout expected by SplatSuRe (per scene)
-------------------------------------------------
    {scene}/
    ├── images/          <- LR training images  (COLMAP undistorted)
    ├── images_SR/       <- SR images, same filenames as images/
    ├── sparse/
    │   └── 0/           <- cameras.bin  images.bin  points3D.bin
    └── input/           <- (optional) original HR GT for eval metrics

Your dataset already has:
    {scene}/images/      <- OK
    {scene}/images_sr/   <- auto-symlinked to images_SR/ by this script
    {scene}/sparse/0/    <- OK  (.bin format)

Extension handling
------------------
Each scene uses ONE extension consistently (e.g. aeroplane uses .jpg
throughout, bike uses .JPG throughout). LR and SR pairs share the same
filename+extension. This script auto-detects the extension per scene from
the `images/` folder and uses it for both phases — no global flag needed.
You can still override with --ext-override if required.

python train_splatsure_competition.py --repo-root     "/home/pranav-htic/3DGS-SR/SplatSuRe" --dataset-root "/home/pranav-htic/3DGS-SR/ClassicalSr4x_Swin_v2" --output-root  "/home/pranav-htic/3DGS-SR/Splatsure-Classical4x_v2" --pred-root    "/home/pranav-htic/3DGS-SR/Splatsure-Classical4x_v2/submissions" --lr-iterations 20000  --sr-iterations 50000 --test-iterations 7000 15000 30000 50000 --ratio-threshold 1.5  --upscale 4 --extra-train-args "--opacity_reset_interval 1500 --densify_grad_threshold 0.0003 --densify_until_iter 20000"

Usage
-----
python train_splatsure_competition.py \\
    --repo-root    /path/to/SplatSuRe \\
    --dataset-root /path/to/dataset \\
    --output-root  /path/to/outputs \\
    --pred-root    /path/to/predictions \\
    --upscale 4
"""



from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

# All recognised image extensions (case-sensitive variants included)
IMAGE_EXTS: set[str] = {
    ".png", ".PNG",
    ".jpg", ".JPG",
    ".jpeg", ".JPEG",
    ".webp", ".bmp", ".tif", ".tiff",
}

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def natural_key(path: Path) -> tuple:
    """Sort key: case-insensitive stem, then suffix."""
    return (path.stem.lower(), path.suffix.lower(), path.name.lower())


def list_scene_images(scene_dir: Path, folder: str = "images") -> list[Path]:
    images_dir = scene_dir / folder
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing folder: {images_dir}")
    imgs = [p for p in images_dir.iterdir()
            if p.is_file() and p.suffix in IMAGE_EXTS]
    imgs.sort(key=natural_key)
    if not imgs:
        raise RuntimeError(f"No images found in {images_dir}")
    return imgs


def detect_scene_ext(scene_dir: Path, folder: str = "images") -> str:
    """
    Auto-detect the image extension used in a scene's image folder.

    Each scene is expected to use exactly ONE extension throughout.
    Returns the extension WITHOUT a leading dot, preserving original case,
    e.g. "jpg" or "JPG" or "PNG".
    Raises if zero or multiple distinct extensions are found.
    """
    images_dir = scene_dir / folder
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing folder: {images_dir}")

    found_exts: set[str] = set()
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix in IMAGE_EXTS:
            found_exts.add(p.suffix)   # keep original case, e.g. ".JPG"

    if not found_exts:
        raise RuntimeError(f"No recognised image files in {images_dir}")
    if len(found_exts) > 1:
        raise RuntimeError(
            f"Scene '{scene_dir.name}/{folder}' contains multiple extensions: "
            f"{sorted(found_exts)}. Each scene must use exactly one extension."
        )

    ext_with_dot = next(iter(found_exts))   # e.g. ".JPG"
    return ext_with_dot.lstrip(".")          # -> "JPG"


def llff_split(images: Sequence[Path], llffhold: int) -> tuple[list[Path], list[Path]]:
    if llffhold <= 0:
        raise ValueError("--llffhold must be a positive integer")
    train = [img for idx, img in enumerate(images) if idx % llffhold != 0]
    test  = [img for idx, img in enumerate(images) if idx % llffhold == 0]
    return train, test


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: list[str], cwd: Path, log_file: Path | None = None) -> None:
    pretty = " ".join(str(c) for c in cmd)
    print(f"\n$ {pretty}")
    if log_file:
        ensure_dir(log_file.parent)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] CMD: {pretty}\n")

    proc = subprocess.Popen(
        cmd, cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        if log_file:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(line)
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def find_latest_ply(search_root: Path) -> Path | None:
    candidates = [p for p in search_root.rglob("*.ply") if p.is_file()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: (p.stat().st_mtime, len(p.parts)))
    return candidates[-1]


def copy_point_cloud(
    scene_out: Path,
    scene_name: str,
    export_root: Path,
    source_scene_dir: Path,
) -> Path | None:
    ensure_dir(export_root)
    target = export_root / f"{scene_name}.ply"
    ply = find_latest_ply(scene_out)
    if ply is None:
        src_ply = source_scene_dir / "sparse" / "0" / "points3D.ply"
        if src_ply.exists():
            ply = src_ply
    if ply is None:
        return None
    shutil.copy2(ply, target)
    return target


def discover_scenes(dataset_root: Path) -> list[str]:
    scenes = []
    for p in sorted(dataset_root.iterdir()):
        if p.is_dir() and (p / "images").exists():
            scenes.append(p.name)
    return scenes


# ──────────────────────────────────────────────────────────────────────────────
# Dataset preparation
# ──────────────────────────────────────────────────────────────────────────────

def ensure_images_SR_symlink(scene_dir: Path, sr_folder_name: str) -> Path:
    """
    SplatSuRe expects the SR folder to be named `images_SR/`.
    If your folder uses a different name (e.g. `images_sr`), this creates a
    relative symlink  images_SR -> <sr_folder_name>  inside the scene directory.
    Safe to call multiple times — skips silently if already in place.
    Returns the Path to images_SR/.
    """
    src = scene_dir / sr_folder_name   # what you have, e.g. images_sr/
    dst = scene_dir / "images_4x"      # what SplatSuRe expects

    if not src.exists():
        raise FileNotFoundError(
            f"SR folder not found: {src}\n"
            f"Tip: check --sr-folder matches the actual folder name inside each scene."
        )

    if dst.exists() or dst.is_symlink():
        return dst   # already in place, nothing to do

    dst.symlink_to(src.name)   # relative symlink — portable
    print(f"[setup] Symlink created: {dst} -> {src.name}")
    return dst


def validate_scene(scene_dir: Path, llffhold: int, skip: bool) -> None:
    """Quick sanity checks before spending GPU time."""
    if skip:
        return
    sparse = scene_dir / "sparse" / "0"
    if not sparse.exists():
        raise FileNotFoundError(f"Missing sparse/0 for scene '{scene_dir.name}'")
    for b in ["cameras.bin", "images.bin", "points3D.bin"]:
        if not (sparse / b).exists():
            raise FileNotFoundError(f"Missing {b} in {sparse}")
    images = list_scene_images(scene_dir, "images")
    if len(images) < llffhold + 1:
        raise RuntimeError(
            f"Scene '{scene_dir.name}' has only {len(images)} images — "
            f"need at least {llffhold + 1} for llffhold={llffhold}."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 – LR training  [train_lr.py]
# ──────────────────────────────────────────────────────────────────────────────

def train_lr_phase(
    repo_root: Path,
    scene_dir: Path,
    lr_out: Path,
    iterations: int,
    test_iterations: list[int],
    save_iterations: list[int],
    checkpoint_iterations: list[int],
    white_bg: bool,
    extra_train_args: list[str],
    log_file: Path,
) -> None:
    """
    Train a low-resolution 3DGS model using the official train_lr.py script.

    Uses -r 1 because the images/ folder already contains LR images at their
    native resolution — no additional downsampling needed.

    Official command (from README):
        python train_lr.py -s <data_dir> -m <out_dir>/lr/<scene> -r <r> --eval
    """
    cmd = [
        sys.executable, str(repo_root / "train_lr.py"),
        "-s", str(scene_dir),
        "-m", str(lr_out),
        "-r", "1",          # images/ already at LR resolution, no further downscale
        "--eval",
        "--iterations",            str(iterations),
        "--test_iterations",       *[str(i) for i in test_iterations],
        "--save_iterations",       *[str(i) for i in save_iterations],
        "--checkpoint_iterations", *[str(i) for i in sorted(set(list(checkpoint_iterations) + [iterations]))],
        "--quiet",
    ]
    if white_bg:
        cmd.append("--white_background")
    cmd.extend(extra_train_args)
    run_cmd(cmd, cwd=repo_root, log_file=log_file)


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 – Weight map generation  [weight_maps.py]
# ──────────────────────────────────────────────────────────────────────────────

WEIGHT_MAPS_DIRNAME = "weight_maps"

def generate_weight_maps(
    repo_root: Path,
    scene_dir: Path,
    lr_out: Path,
    ratio_threshold: float,
    log_file: Path,
    extra_train_args: list[str], # <--- 1. Add this parameter
) -> Path:
    
    cmd = [
        sys.executable, str(repo_root / "weight_maps.py"),
        "-s", str(scene_dir),
        "-m", str(lr_out),
        "-r", "1",                                     
        "--eval",
        "--weight_maps_dirname", WEIGHT_MAPS_DIRNAME,  
        "--ratio_threshold",     str(ratio_threshold),
        "--iterations", "-1",    
    ]
    cmd.extend(extra_train_args) # <--- 2. Add this line
    run_cmd(cmd, cwd=repo_root, log_file=log_file)
    weight_maps_path = lr_out / WEIGHT_MAPS_DIRNAME
    if not weight_maps_path.is_dir():
        raise FileNotFoundError(
            f"weight_maps.py finished but '{weight_maps_path}' was not created.\n"
            f"Contents of {lr_out}: {sorted(p.name for p in lr_out.iterdir())}"
        )
    return weight_maps_path


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 – SR training  [train.py]
# ──────────────────────────────────────────────────────────────────────────────

def train_sr_phase(
    repo_root: Path,
    scene_dir: Path,
    sr_out: Path,
    weight_maps_path: Path,
    iterations: int,
    test_iterations: list[int],
    save_iterations: list[int],
    checkpoint_iterations: list[int],
    white_bg: bool,
    upscale: int,
    img_ext: str,
    extra_train_args: list[str],
    log_file: Path,
) -> None:
    """
    Train SplatSuRe with selective SR supervision.

    Official command (from README):
        python train.py -s <data_dir> -m <out_dir>/<scene> -r 1 --eval
            --images images_SR --img_ext png --upscale 4
            --weight_maps_path <out_dir>/lr/<scene>/weight_maps

    -r 1 is critical: images_SR/ is already at SR resolution, no further
    downscaling should happen. Without it, auto-scaling may kick in if SR
    images are wide (>1600px) and corrupt the supervision signal.
    """
    cmd = [
        sys.executable, str(repo_root / "train.py"),
        "-s", str(scene_dir),
        "-m", str(sr_out),
        "-r", "1",            # SR images are already at target resolution
        "--eval",
        "--images",           "images_4x",
        "--img_ext",          img_ext,
        "--upscale",          str(upscale),
        "--weight_maps_path", str(weight_maps_path),
        "--iterations",            str(iterations),
        "--test_iterations",       *[str(i) for i in test_iterations],
        "--save_iterations",       *[str(i) for i in save_iterations],
        "--checkpoint_iterations", *[str(i) for i in checkpoint_iterations],
        "--quiet",
    ]
    if white_bg:
        cmd.append("--white_background")
    cmd.extend(extra_train_args)
    run_cmd(cmd, cwd=repo_root, log_file=log_file)


# ──────────────────────────────────────────────────────────────────────────────
# Render + rename
# ──────────────────────────────────────────────────────────────────────────────

def render_predictions(
    repo_root: Path,
    scene_dir: Path,
    scene_out: Path,
    iteration: int,
    white_bg: bool,
    upscale: int,
    img_ext: str,
    extra_render_args: list[str],
) -> Path:
    """Render held-out test views from the trained SR model.

    --use_trained_exp is intentionally NOT passed: the SR phase trains with
    train_test_exp=False, so pretrained_exposures is never set on the loaded
    GaussianModel and passing --use_trained_exp here would AttributeError.
    """
    cmd = [
        sys.executable, str(repo_root / "render.py"),
        "--model_path", str(scene_out),
        "--iteration",  str(iteration),
        "-s",           str(scene_dir),
        "--images",     "images_4x",
        "--img_ext",    img_ext,
        "--upscale",    "1",
        "--skip_train",
        "--eval",
        "--quiet",
    ]
    if white_bg:
        cmd.append("--white_background")
    cmd.extend(extra_render_args)
    run_cmd(cmd, cwd=repo_root)
    return scene_out / "test" / f"ours_{iteration}" / "renders"


def rename_renders_to_test_filenames(
    renders_dir: Path,
    test_images: Sequence[Path],
    pred_root: Path,
    scene_name: str,
) -> Path:
    if not renders_dir.exists():
        raise FileNotFoundError(f"Render directory not found: {renders_dir}")

    rendered = sorted(
        p for p in renders_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".png"
    )
    if len(rendered) != len(test_images):
        raise RuntimeError(
            f"Rendered count ({len(rendered)}) != held-out count "
            f"({len(test_images)}) for scene '{scene_name}'."
        )

    scene_pred_dir = pred_root / scene_name
    ensure_dir(scene_pred_dir)
    for src, gt in zip(rendered, test_images):
        shutil.copy2(src, scene_pred_dir / gt.name)
    return scene_pred_dir


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Competition launcher for SplatSuRe (two-phase LR -> SR training)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    parser.add_argument("--repo-root",    type=Path, default=Path.cwd())
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root",  type=Path, required=True)
    parser.add_argument("--pred-root",    type=Path, required=True)

    # Scene selection
    parser.add_argument("--scenes", nargs="+", default=None,
                        help="Explicit scene names (default: auto-discover)")

    # Training schedule
    parser.add_argument("--lr-iterations",  type=int, default=7000,
                        help="Iterations for Phase 1 (LR / weight-map phase)")
    parser.add_argument("--sr-iterations",  type=int, default=30000,
                        help="Iterations for Phase 2 (SR training phase)")
    parser.add_argument("--test-iterations",       nargs="+", type=int,
                        default=[7000, 15000, 30000])
    parser.add_argument("--save-iterations",       nargs="+", type=int,
                        default=[7000, 30000])
    parser.add_argument("--checkpoint-iterations", nargs="+", type=int,
                        default=[7000, 30000])

    # Dataset / SR settings
    parser.add_argument("--llffhold",  type=int, default=8,
                        help="Every Nth image held out for test (LLFF convention)")
    parser.add_argument("--upscale",   type=int, default=4,
                        help="SR scale factor (4 for 4x SwinIR-L)")
    parser.add_argument("--ratio-threshold", type=float, default=1.1,
                        help="Ratio threshold for weight_maps.py (Section 4.1 of paper). "
                             "Controls how aggressively SR maps are applied. Default: 1.1")
    parser.add_argument("--sr-folder", type=str, default="images_4x",
                        help="Name of your SR folder inside each scene dir")
    parser.add_argument("--ext-override", type=str, default=None,
                        help="Force one extension for ALL scenes (no dot, e.g. 'jpg'). "
                             "Default: auto-detect per scene from images/.")
    parser.add_argument("--white-bg", action="store_true")

    # Misc
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-lr-phase",   action="store_true",
                        help="Skip Phase 1 (weight maps must already exist in output-root/lr/)")
    parser.add_argument("--skip-sr-phase",   action="store_true",
                        help="Skip Phase 3 SR training, going straight to rendering. "
                             "Requires the trained SR model (point_cloud at --sr-iterations) "
                             "to already exist in output-root/sr/<scene>/. Useful for "
                             "re-rendering after fixing a render-only bug without retraining.")
    parser.add_argument("--dry-run",         action="store_true")
    parser.add_argument("--extra-train-args",  type=str, default="",
                        help="Extra args forwarded to train_lr.py, weight_maps.py AND train.py "
                             "(use for args accepted by all three, e.g. --opacity_reset_interval)")
    parser.add_argument("--extra-lr-args",     type=str, default="",
                        help="Extra args forwarded to train_lr.py and weight_maps.py ONLY "
                             "(not to train.py). Use this only if you have args specific "
                             "to the LR phase that must not reach the SR phase. "
                             "Note: canonical SplatSuRe (train_and_eval.sh in the official "
                             "repo) does not pass --train_test_exp at any phase, and for "
                             "competition runs with strict held-out splits this should be "
                             "left empty.")
    parser.add_argument("--extra-sr-args",     type=str, default="",
                        help="Extra args forwarded to train.py (SR phase) ONLY "
                             "(use for SR-specific args such as --gamma that train_lr.py does not accept)")
    parser.add_argument("--extra-render-args", type=str, default="")

    args = parser.parse_args()

    repo_root    = args.repo_root.resolve()
    dataset_root = args.dataset_root.resolve()
    output_root  = args.output_root.resolve()
    pred_root    = args.pred_root.resolve()

    for required, label in [
        (repo_root / "train.py",      "train.py"),
        (repo_root / "train_lr.py",   "train_lr.py"),
        (repo_root / "weight_maps.py","weight_maps.py"),
        (repo_root / "render.py",     "render.py"),
    ]:
        if not required.exists():
            raise FileNotFoundError(f"{label} not found under {repo_root}")

    scenes = args.scenes or discover_scenes(dataset_root)
    if not scenes:
        raise RuntimeError(f"No scenes found under {dataset_root}")

    extra_train_args  = args.extra_train_args.split()  if args.extra_train_args.strip()  else []
    extra_lr_args     = args.extra_lr_args.split()     if args.extra_lr_args.strip()     else []
    extra_sr_args     = args.extra_sr_args.split()     if args.extra_sr_args.strip()     else []
    extra_render_args = args.extra_render_args.split() if args.extra_render_args.strip() else []

    ensure_dir(output_root)
    ensure_dir(pred_root)

    summary: list[tuple[str, bool, str]] = []

    for scene_name in scenes:
        scene_dir = dataset_root / scene_name
        lr_out    = output_root / "lr" / scene_name
        sr_out    = output_root / "sr" / scene_name

        print(f"\n{'='*60}")
        print(f"  Scene : {scene_name}")
        print(f"{'='*60}")

        if not scene_dir.exists():
            summary.append((scene_name, False, f"Scene dir not found: {scene_dir}"))
            print(f"[ERROR] Scene dir not found: {scene_dir}")
            continue

        # ── Dataset setup ──────────────────────────────────────────────────────
        try:
            ensure_images_SR_symlink(scene_dir, args.sr_folder)
            validate_scene(scene_dir, args.llffhold, args.skip_validation)

            # Per-scene extension detection
            if args.ext_override:
                img_ext = args.ext_override.lstrip(".")
                print(f"[ext] Using override: .{img_ext}")
            else:
                img_ext = detect_scene_ext(scene_dir, "images")
                print(f"[ext] Auto-detected from images/: .{img_ext}")
                sr_ext = detect_scene_ext(scene_dir, args.sr_folder)
                if img_ext.lower() != sr_ext.lower():
                    raise RuntimeError(
                        f"LR extension '.{img_ext}' != SR extension '.{sr_ext}'. "
                        f"LR and SR pairs must share the same extension per scene."
                    )

            all_images   = list_scene_images(scene_dir, "images")
            _, test_imgs = llff_split(all_images, args.llffhold)

        except Exception as e:
            summary.append((scene_name, False, f"Dataset setup failed: {e}"))
            print(f"[ERROR] {e}")
            continue

        if args.dry_run:
            n_train = sum(1 for i in range(len(all_images)) if i % args.llffhold != 0)
            print(f"  ext={img_ext}  train={n_train}  test={len(test_imgs)}")
            summary.append((scene_name, True, "dry-run"))
            continue

        ensure_dir(lr_out)
        ensure_dir(sr_out)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = output_root / "_logs" / f"{scene_name}_{ts}.log"

        try:
            # ── Step 1 : LR training ──────────────────────────────────────────
            if not args.skip_lr_phase:
                print(f"\n[Step 1] LR training ({args.lr_iterations} iters) -> {lr_out}")
                train_lr_phase(
                    repo_root=repo_root,
                    scene_dir=scene_dir,
                    lr_out=lr_out,
                    iterations=args.lr_iterations,
                    test_iterations=[i for i in args.test_iterations
                                     if i <= args.lr_iterations],
                    save_iterations=[i for i in args.save_iterations
                                     if i <= args.lr_iterations],
                    checkpoint_iterations=[i for i in args.checkpoint_iterations
                                           if i <= args.lr_iterations],
                    white_bg=args.white_bg,
                    extra_train_args=extra_train_args + extra_lr_args,
                    log_file=log_file,
                )
            else:
                print(f"[Step 1] Skipped (--skip-lr-phase).")

            # ── Step 2 : Weight map generation ───────────────────────────────
            if not args.skip_lr_phase:
                print(f"\n[Step 2] Generating weight maps -> {lr_out / WEIGHT_MAPS_DIRNAME}")
                weight_maps_path = generate_weight_maps(
                    repo_root=repo_root,
                    scene_dir=scene_dir,
                    lr_out=lr_out,
                    ratio_threshold=args.ratio_threshold,
                    log_file=log_file,
                    extra_train_args=extra_train_args + extra_lr_args, # <--- 3. Pass the argument here
                )
            else:
                # If LR phase was skipped, weight maps must already exist
                weight_maps_path = lr_out / WEIGHT_MAPS_DIRNAME
                if not weight_maps_path.is_dir():
                    raise FileNotFoundError(
                        f"--skip-lr-phase was set but weight maps not found at: {weight_maps_path}\n"
                        f"Run Step 1+2 first, or check --output-root path."
                    )
            print(f"[Step 2] Weight maps: {weight_maps_path}")

            # ── Step 3 : SR training ──────────────────────────────────────────
            # Phase 2 inherits Phase 1 geometry via --start_checkpoint.
            # This transfers xyz, scales, rotations, opacities from the LR
            # model so Phase 2 starts from geometry already shaped by the
            # train views rather than re-initialising from the sparse COLMAP
            # point cloud. SH coefficients also transfer but are quickly
            # corrected by the SR supervision signal. Exposure optimizer state
            # is NOT carried over (separate optimizer; both phases run with
            # train_test_exp=False under canonical SplatSuRe).
            final_sr_args = extra_train_args + extra_sr_args

            if not args.skip_sr_phase:
                print(f"\n[Step 3] SR training ({args.sr_iterations} iters) -> {sr_out}")
                train_sr_phase(
                    repo_root=repo_root,
                    scene_dir=scene_dir,
                    sr_out=sr_out,
                    weight_maps_path=weight_maps_path,
                    iterations=args.sr_iterations,
                    test_iterations=[i for i in args.test_iterations
                                     if i <= args.sr_iterations],
                    save_iterations=[i for i in args.save_iterations
                                     if i <= args.sr_iterations],
                    checkpoint_iterations=[i for i in args.checkpoint_iterations
                                           if i <= args.sr_iterations],
                    white_bg=args.white_bg,
                    upscale=args.upscale,
                    img_ext=img_ext,
                    extra_train_args=final_sr_args,
                    log_file=log_file,
                )
            else:
                # Verify the SR model actually exists before trying to render
                expected_ply = sr_out / "point_cloud" / f"iteration_{args.sr_iterations}" / "point_cloud.ply"
                if not expected_ply.exists():
                    raise FileNotFoundError(
                        f"--skip-sr-phase was set but SR model not found at:\n  {expected_ply}\n"
                        f"Run the full pipeline first, or check --output-root and --sr-iterations."
                    )
                print(f"[Step 3] Skipped (--skip-sr-phase). Using existing model at {expected_ply}")

            # ── Render held-out views ──────────────────────────────────────────
            print(f"\n[Render] Rendering {len(test_imgs)} held-out views ...")
            renders_dir = render_predictions(
                repo_root=repo_root,
                scene_dir=scene_dir,
                scene_out=sr_out,
                iteration=args.sr_iterations,
                white_bg=args.white_bg,
                upscale=args.upscale,
                img_ext=img_ext,
                extra_render_args=extra_render_args,
            )

            # ── Rename renders -> original held-out filenames ──────────────────
            scene_pred_dir = rename_renders_to_test_filenames(
                renders_dir=renders_dir,
                test_images=test_imgs,
                pred_root=pred_root,
                scene_name=scene_name,
            )

            # ── Export point cloud ─────────────────────────────────────────────
            ply_path = copy_point_cloud(
                scene_out=sr_out,
                scene_name=scene_name,
                export_root=output_root / "point_clouds",
                source_scene_dir=scene_dir,
            )
            if ply_path:
                print(f"[PLY] Exported: {ply_path}")
            else:
                print(f"[WARN] No .ply found for {scene_name}")

            summary.append((scene_name, True, str(scene_pred_dir)))

        except Exception as e:
            summary.append((scene_name, False, str(e)))
            print(f"[ERROR] {scene_name}: {e}")
            import traceback; traceback.print_exc()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for sname, ok, info in summary:
        print(f"{'OK  ' if ok else 'FAIL'}  {sname}: {info}")
    print(f"\nPredictions : {pred_root}")
    print(f"Outputs     : {output_root}")

    return 0 if all(ok for _, ok, _ in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())

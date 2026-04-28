#!/usr/bin/env python3
import argparse
import json
import sys
import struct
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"}
ALL_SCENES = ["aeroplane", "bike", "buddha", "cycle", "face", "firehydrant", "still3", "toy"]

# ── Depth-Anything-V2 wrapper ────────────────────────────────────────────────
class DepthAnythingV2Wrapper:
    HF_MODELS = {
        "small": "depth-anything/Depth-Anything-V2-Small-hf",
        "base":  "depth-anything/Depth-Anything-V2-Base-hf",
        "large": "depth-anything/Depth-Anything-V2-Large-hf",
    }

    def __init__(self, size: str = "large", device: str = "cuda", invert: bool = False):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        model_id = self.HF_MODELS[size]
        print(f"[depth] loading {model_id} on {device}…")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.invert = invert
        if invert:
            print(f"[depth] --invert-depth set: output will be 1 / (d + 1e-3)")

    @torch.no_grad()
    def infer(self, pil_image: Image.Image) -> np.ndarray:
        W, H = pil_image.size
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        pred = outputs.predicted_depth
        pred_resized = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False
        ).squeeze().cpu().numpy().astype(np.float32)
        pred_resized = np.clip(pred_resized, 0.0, None)
        if self.invert:
            pred_resized = 1.0 / (pred_resized + 1e-3)
        return pred_resized

# ── COLMAP loading ──────────────────────────────────────────────────────────
def _read_points3D_binary_as_dict(path: Path) -> dict:
    points3D: dict[int, np.ndarray] = {}
    with open(path, "rb") as f:
        (num_points,) = struct.unpack("<Q", f.read(8))
        for _ in range(num_points):
            (point_id,) = struct.unpack("<Q", f.read(8))
            xyz = np.array(struct.unpack("<ddd", f.read(24)), dtype=np.float64)
            _ = f.read(3)
            _ = f.read(8)
            (track_length,) = struct.unpack("<Q", f.read(8))
            _ = f.read(8 * track_length)
            points3D[int(point_id)] = xyz
    return points3D

def _read_points3D_text_as_dict(path: Path) -> dict:
    points3D: dict[int, np.ndarray] = {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            pid = int(parts[0])
            xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
            points3D[pid] = xyz
    return points3D

def load_colmap(repo_root: Path, scene_dir: Path):
    sys.path.insert(0, str(repo_root))
    from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, read_extrinsics_text, read_intrinsics_text
    sparse = scene_dir / "sparse" / "0"
    if (sparse / "cameras.bin").exists():
        cams_intrinsic = read_intrinsics_binary(str(sparse / "cameras.bin"))
        cams_extrinsic = read_extrinsics_binary(str(sparse / "images.bin"))
        points3D = _read_points3D_binary_as_dict(sparse / "points3D.bin")
    else:
        cams_intrinsic = read_intrinsics_text(str(sparse / "cameras.txt"))
        cams_extrinsic = read_extrinsics_text(str(sparse / "images.txt"))
        points3D = _read_points3D_text_as_dict(sparse / "points3D.txt")
    return cams_intrinsic, cams_extrinsic, points3D

def qvec_to_rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w    ],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w    ],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
    ])

PINHOLE_MODELS = {"PINHOLE", "SIMPLE_PINHOLE"}

def get_intrinsic_matrix(cam_intrinsic):
    model = cam_intrinsic.model
    if isinstance(model, bytes):
        model = model.decode("utf-8")
    p = cam_intrinsic.params

    if model == "SIMPLE_PINHOLE":
        f, cx, cy = float(p[0]), float(p[1]), float(p[2])
        fx = fy = f
    elif model == "PINHOLE":
        fx, fy, cx, cy = float(p[0]), float(p[1]), float(p[2]), float(p[3])
    elif model in ("SIMPLE_RADIAL", "RADIAL"):
        f, cx, cy = float(p[0]), float(p[1]), float(p[2])
        fx = fy = f
    elif model in ("OPENCV", "FULL_OPENCV"):
        fx, fy, cx, cy = float(p[0]), float(p[1]), float(p[2]), float(p[3])
    else:
        raise ValueError(f"Camera model '{model}' is not supported here.")
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

def check_scene_camera_models(cams_intrinsic: dict, scene_name: str) -> bool:
    seen = {}
    for c in cams_intrinsic.values():
        m = c.model.decode("utf-8") if isinstance(c.model, bytes) else c.model
        seen[m] = seen.get(m, 0) + 1
    all_pinhole = set(seen.keys()).issubset(PINHOLE_MODELS)
    tag = "OK" if all_pinhole else "WARN"
    print(f"  [{tag}] camera models in {scene_name}: {seen}")
    return all_pinhole

# ── alignment ───────────────────────────────────────────────────────────────
def align_one_image(da_invdepth: np.ndarray, image_record, intrinsic_record, points3D: dict, min_points: int = 8, max_points: int = 5000) -> tuple[float, float, int, float]:
    K = get_intrinsic_matrix(intrinsic_record)
    R = qvec_to_rotmat(image_record.qvec)
    t = np.asarray(image_record.tvec).reshape(3, 1)
    H, W = da_invdepth.shape

    p3d_ids = image_record.point3D_ids
    valid = p3d_ids != -1
    p3d_ids = p3d_ids[valid]
    if p3d_ids.size == 0:
        return 1.0, 0.0, 0, 0.0

    coords = [points3D[int(pid)] for pid in p3d_ids if int(pid) in points3D]
    if not coords:
        return 1.0, 0.0, 0, 0.0
    coords = np.asarray(coords, dtype=np.float64)

    cam = (R @ coords.T + t).T
    z = cam[:, 2]
    in_front = z > 1e-3
    cam = cam[in_front]
    if cam.shape[0] < min_points:
        return 1.0, 0.0, 0, 0.0
    z = cam[:, 2]

    uv = (K @ cam.T).T
    u = uv[:, 0] / uv[:, 2]
    v = uv[:, 1] / uv[:, 2]
    on_screen = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if on_screen.sum() < min_points:
        return 1.0, 0.0, 0, 0.0

    u = u[on_screen].astype(np.int64)
    v = v[on_screen].astype(np.int64)
    z = z[on_screen]
    colmap_inv = (1.0 / z).astype(np.float64)
    da_inv = da_invdepth[v, u].astype(np.float64)

    pearson = 0.0
    if da_inv.size > 2 and da_inv.std() > 1e-12 and colmap_inv.std() > 1e-12:
        pearson = float(np.corrcoef(da_inv, colmap_inv)[0, 1])

    if da_inv.size > max_points:
        idx = np.random.choice(da_inv.size, size=max_points, replace=False)
        da_inv, colmap_inv = da_inv[idx], colmap_inv[idx]

    rng = np.random.default_rng(0)
    n = da_inv.size
    best_inl, best_scale, best_off = 0, 1.0, 0.0
    if n < 3:
        return 1.0, 0.0, 0, pearson
    n_iters = max(50, min(300, n))
    
    A0 = np.stack([da_inv, np.ones_like(da_inv)], axis=1)
    s0, o0 = np.linalg.lstsq(A0, colmap_inv, rcond=None)[0]
    resid0 = colmap_inv - (s0 * da_inv + o0)
    mad = np.median(np.abs(resid0 - np.median(resid0))) + 1e-9
    thresh = 3.0 * mad

    for _ in range(n_iters):
        idx = rng.choice(n, size=min(3, n), replace=False)
        A = np.stack([da_inv[idx], np.ones(idx.size)], axis=1)
        try:
            s, o = np.linalg.lstsq(A, colmap_inv[idx], rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        if not np.isfinite(s) or not np.isfinite(o):
            continue
        resid = colmap_inv - (s * da_inv + o)
        inl = np.abs(resid) < thresh
        if inl.sum() > best_inl:
            best_inl = inl.sum()
            best_scale, best_off = float(s), float(o)

    final_resid = colmap_inv - (best_scale * da_inv + best_off)
    inliers = np.abs(final_resid) < thresh
    if inliers.sum() >= 3:
        A = np.stack([da_inv[inliers], np.ones(inliers.sum())], axis=1)
        s, o = np.linalg.lstsq(A, colmap_inv[inliers], rcond=None)[0]
        if np.isfinite(s) and np.isfinite(o):
            best_scale, best_off = float(s), float(o)
            best_inl = int(inliers.sum())

    return best_scale, best_off, best_inl, pearson

# ── per-scene pipeline ──────────────────────────────────────────────────────
def list_images(folder: Path) -> list[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix in IMAGE_EXTS])

def stem_no_ext(image_name: str) -> str:
    return Path(image_name).stem

def run_inference_for_scene(scene_dir: Path, depths_dir: Path, model: DepthAnythingV2Wrapper, overwrite: bool):
    images_dir = scene_dir / "images"
    images = list_images(images_dir)
    depths_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(images, desc=f"  DA-V2 [{scene_dir.name}]", leave=False):
        out_npy = depths_dir / f"{img_path.stem}.npy"
        if out_npy.exists() and not overwrite:
            continue
        pil = Image.open(img_path).convert("RGB")
        invdepth = model.infer(pil)
        np.save(out_npy, invdepth.astype(np.float32))

def align_scene(scene_dir: Path, depths_dir: Path, repo_root: Path) -> dict:
    cams_intrinsic, cams_extrinsic, points3D = load_colmap(repo_root, scene_dir)
    check_scene_camera_models(cams_intrinsic, scene_dir.name)
    intrinsic_by_id = {c.id: c for c in cams_intrinsic.values()}

    depth_params: dict[str, dict] = {}
    correlations: list[float] = []
    n_aligned, n_total = 0, 0
    for img_id, img_record in tqdm(cams_extrinsic.items(), desc=f"  align [{scene_dir.name}]", leave=False):
        n_total += 1
        stem = stem_no_ext(img_record.name)
        depth_npy = depths_dir / f"{stem}.npy"
        if not depth_npy.exists():
            continue
        da_inv = np.load(depth_npy).astype(np.float32)

        intr = intrinsic_by_id.get(img_record.camera_id)
        if intr is None:
            continue

        scale, offset, n_inl, pearson = align_one_image(da_inv, img_record, intr, points3D)
        depth_params[stem] = {
            "scale":  float(scale),
            "offset": float(offset),
            "n_inliers": int(n_inl),
            "pearson": float(pearson),
        }
        if n_inl >= 8:
            n_aligned += 1
            correlations.append(pearson)

    valid_scales = [v["scale"] for v in depth_params.values() if v["n_inliers"] >= 8]
    med_scale = float(np.median(valid_scales)) if valid_scales else 1.0
    
    for k in depth_params:
        depth_params[k]["med_scale"] = med_scale

    if correlations:
        med_corr = float(np.median(correlations))
        n_pos = sum(1 for c in correlations if c > 0.3)
        n_neg = sum(1 for c in correlations if c < -0.3)
        corr_status = "healthy"
        if med_corr < -0.3:
            corr_status = "INVERTED (DA-V2 looks like depth, not inverse depth — try --invert-depth)"
        elif med_corr < 0.3:
            corr_status = "weak (depth prior may not help training)"
        print(f"  Pearson corr (DA vs 1/z): median={med_corr:+.3f}  ({n_pos} strong+, {n_neg} strong-)  → {corr_status}")

    return {"params": depth_params, "n_aligned": n_aligned, "n_total": n_total,
            "med_scale": med_scale, "median_corr": float(np.median(correlations)) if correlations else 0.0}

def write_depth_params(scene_dir: Path, params: dict):
    out_path = scene_dir / "sparse" / "0" / "depth_params.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(params, f, indent=2)
    return out_path

# ── main ────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Generate DA-V2 depths as Float32 NPY for SplatSuRe")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--repo-root",    type=Path, required=True)
    p.add_argument("--scenes",       nargs="+", default=ALL_SCENES)
    p.add_argument("--depths-dirname", type=str, default="depths_npy")
    p.add_argument("--model-size",   choices=["small", "base", "large"], default="large")
    p.add_argument("--device",       type=str, default="cuda")
    p.add_argument("--overwrite",    action="store_true")
    p.add_argument("--skip-inference", action="store_true")
    p.add_argument("--invert-depth", action="store_true")
    args = p.parse_args()

    if not args.dataset_root.exists():
        sys.exit(f"dataset-root not found: {args.dataset_root}")
    if not (args.repo_root / "scene" / "colmap_loader.py").exists():
        sys.exit(f"repo-root missing scene/colmap_loader.py: {args.repo_root}")

    model = None
    if not args.skip_inference:
        model = DepthAnythingV2Wrapper(size=args.model_size, device=args.device, invert=args.invert_depth)

    summary = []
    for scene_name in args.scenes:
        scene_dir = args.dataset_root / scene_name
        if not (scene_dir / "images").exists() or not (scene_dir / "sparse" / "0").exists():
            continue

        depths_dir = scene_dir / args.depths_dirname
        print(f"\n── {scene_name} ──")

        if not args.skip_inference:
            run_inference_for_scene(scene_dir, depths_dir, model, args.overwrite)

        align_result = align_scene(scene_dir, depths_dir, args.repo_root)
        out_path = write_depth_params(scene_dir, align_result["params"])

        n_total = align_result["n_total"]
        n_aligned = align_result["n_aligned"]
        
        # New scale-invariant reliability check based on pearson > 0.3
        n_reliable = sum(1 for v in align_result["params"].values() if v["n_inliers"] >= 8 and v["pearson"] > 0.3)
        
        print(f"  aligned : {n_aligned}/{n_total}  (median scale {align_result['med_scale']:.6f})")
        print(f"  reliable: {n_reliable}/{n_total}  (will receive float32 depth supervision)")
        print(f"  wrote   : {out_path}")
        summary.append({
            "scene": scene_name, "total": n_total,
            "aligned": n_aligned, "reliable": n_reliable,
            "med_scale": align_result["med_scale"], "median_corr": align_result.get("median_corr", 0.0),
        })

    print(f"\n{'='*84}")
    print(f"{'scene':<14}{'total':>8}{'aligned':>10}{'reliable':>10}{'med_scale':>14}{'med_corr':>12}")
    print(f"{'-'*84}")
    for s in summary:
        warn = "  <- NEGATIVE (try --invert-depth)" if s["median_corr"] < -0.3 else ("  <- weak" if s["median_corr"] < 0.3 else "")
        print(f"{s['scene']:<14}{s['total']:>8}{s['aligned']:>10}{s['reliable']:>10}{s['med_scale']:>14.6f}{s['median_corr']:>+12.3f}{warn}")
    print(f"{'='*84}")

if __name__ == "__main__":
    main()
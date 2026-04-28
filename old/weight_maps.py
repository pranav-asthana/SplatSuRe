import os
import torch
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import save_image
import torch

if __name__ != "__main__":
    exit(0)

# Set up command line argument parser
parser = ArgumentParser(description="Training script parameters")
lp = ModelParams(parser)
op = OptimizationParams(parser)
pp = PipelineParams(parser)
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--weight_maps_dirname", type=str, default = "weight_maps")
parser.add_argument("--ratio_threshold", type=float, default = 1.7)
parser.add_argument("--render_debug", action="store_true", default=False)

args = parser.parse_args(sys.argv[1:])

print("Optimizing " + args.model_path)

SPARSE_ADAM_AVAILABLE = False

# Initialize system state (RNG)
safe_state(args.quiet)

args.weight_maps_path = os.path.join(args.model_path, args.weight_maps_dirname)
os.makedirs(args.weight_maps_path, exist_ok=True)

dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)
background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
dataset.img_ext = None

gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
scene = Scene(dataset, gaussians, load_iteration=args.iterations, shuffle=False)

Ng = gaussians._opacity.shape[0]
cam = scene.getTestCameras()[0]
H, W = cam.image_height, cam.image_width
num_cams = len(scene.getTrainCameras())

minmax_radius_img_for_G = torch.zeros((Ng, 2), dtype=torch.float32) # (MIN, MAX)
minmax_index_img_for_G = torch.zeros((Ng, 2), dtype=torch.int32) - 1 # (Idx of image with MIN, Idx of image with MAX)
minmax_radius_img_for_G[:, 0] = 1e6 # MIN
minmax_radius_img_for_G[:, 1] = -1 # MAX
num_times_seen = torch.zeros((Ng,), dtype=torch.int32)

for img_idx, cam in enumerate(tqdm(scene.getTrainCameras())):
    render_pkg = render(cam, gaussians, pipe, background, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
    radii = render_pkg['true_radii']
    contributions_mask = render_pkg['is_contributing'].cpu()>0
    visible_mask = (radii > 0) & contributions_mask
    if not visible_mask.any():
        continue
    if visible_mask.any():
        update_mask = visible_mask.nonzero().squeeze(-1)
        num_times_seen[update_mask] += 1
        curr_min_lower_indices = torch.where(radii[update_mask]<minmax_radius_img_for_G[update_mask, 0])[0]
        minmax_radius_img_for_G[update_mask[curr_min_lower_indices], 0] = radii[update_mask][curr_min_lower_indices]
        minmax_index_img_for_G[update_mask[curr_min_lower_indices], 0] = img_idx
        curr_max_lower_indices = torch.where(radii[update_mask]>minmax_radius_img_for_G[update_mask, 1])[0]
        minmax_radius_img_for_G[update_mask[curr_max_lower_indices], 1] = radii[update_mask[curr_max_lower_indices]]
        minmax_index_img_for_G[update_mask[curr_max_lower_indices], 1] = img_idx

ratio = minmax_radius_img_for_G[:, 1]/minmax_radius_img_for_G[:, 0]
ratio = ratio.clamp_min(1)
ratio_threshold = args.ratio_threshold

# Ratio to Gaussian fidelity score
g_scores = torch.sigmoid((ratio-ratio_threshold)/0.05) # Higher for good Gaussians, lower for bad Gaussians
g_scores[num_times_seen<3] = 0.0  # Not enough views

with torch.no_grad():
    for img_idx, cam in enumerate(tqdm(scene.getTrainCameras())):
        # Get weight map where LR is sufficient (high g_scores), and converse is where SR is needed
        override_color=g_scores.unsqueeze(-1).repeat(1, 3).cuda()
        render_pkg = render(cam, gaussians, pipe, background, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, override_color=override_color)
        image_good = render_pkg['render'].cpu()
        one_m_good = 1-image_good
        # Also get largest Gaussians for current Image and visualize it
        override_color = torch.zeros((Ng, 3), dtype=torch.float32).cuda()
        good_gs_for_curr_img = torch.where(minmax_index_img_for_G[:, 1]==img_idx)[0]
        override_color[good_gs_for_curr_img] = 1.0
        render_pkg = render(cam, gaussians, pipe, background, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, override_color=override_color)
        image_good_curr = render_pkg['render'].cpu()
        sr_map = (one_m_good + image_good_curr)[0]
        if sr_map.max()<0.9:
            sr_map = (sr_map - sr_map.min())/(sr_map.max()-sr_map.min()+1e-6)
        sr_map = sr_map.clip(0, 1)
        if args.render_debug:
            render_pkg = render(cam, gaussians, pipe, background, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
            rendering = render_pkg['render'].cpu()
            save_image([image_good, image_good_curr, one_m_good, sr_map.unsqueeze(0).repeat(3, 1, 1), rendering], f'{args.weight_maps_path}/importance_{img_idx}.jpg', pad_value=1)
        torch.save(sr_map, f'{args.weight_maps_path}/SR_{cam.image_name}.pty')

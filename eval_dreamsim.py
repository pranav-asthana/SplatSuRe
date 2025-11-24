
from pathlib import Path
import os
from PIL import Image
import torch
from tqdm import tqdm
from argparse import ArgumentParser
from dreamsim import dreamsim

dreamsim_model, dreamsim_preprocess = dreamsim(pretrained=True, device='cuda')


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(render)
        gts.append(gt)
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths):
    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                dreamsim_distances = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    # Preprocess for dreamsim
                    render_ds = dreamsim_preprocess(renders[idx]).cuda()
                    gt_ds = dreamsim_preprocess(gts[idx]).cuda()
                    dreamsim_distance = dreamsim_model(render_ds, gt_ds)
                    dreamsim_distances.append(dreamsim_distance.item())
                    

                print(torch.tensor(dreamsim_distances).mean())

        except Exception as e:
            print("Unable to compute metrics for model", scene_dir)
            print(e)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', nargs="+", type=str, default=[])
    parser.add_argument('-p1', type=str, default=None)
    parser.add_argument('-p2', type=str, default=None)
    args = parser.parse_args()
    evaluate(args.model_paths)
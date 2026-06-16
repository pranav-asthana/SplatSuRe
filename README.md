# SplatSuRe: Selective Super-Resolution for Multi-view Consistent 3D Gaussian Splatting
[Pranav Asthana](https://pranav-asthana.github.io/), [Alex Hanson](https://www.cs.umd.edu/~hanson/), [Allen Tu](https://tuallen.github.io/), [Tom Goldstein](https://www.cs.umd.edu/~tomg/), [Matthias Zwicker](https://www.cs.umd.edu/~zwicker/), [Amitabh Varshney](https://www.cs.umd.edu/~varshney/)<br>
University of Maryland, College Park<br>
[Webpage](https://splatsure.github.io) | [arXiv](https://arxiv.org/abs/2512.02172) | [StableSR outputs(T&T, DB, MipNerf360)](https://drive.google.com/drive/folders/1mhEKcvJtxhPCrsTveRFbEkTSqerJyjvb) | [COLMAP(T&T)](https://drive.google.com/drive/folders/1iNMynWtvRg1N--YyqpB1HaPkt-kURj04)<br>
<br>
![Teaser image](assets/teaser.png)

This repository contains the official implementation of the paper "SplatSuRe: Selective Super-Resolution for Multi-view consistent 3D Gaussian Splatting". We build from the original [3DGS repository](https://github.com/graphdeco-inria/gaussian-splatting).

**Abstract**: *3D Gaussian Splatting (3DGS) enables high-quality novel view synthesis, motivating interest in generating higher-resolution renders than those available during training. A natural strategy is to apply super-resolution (SR) to low-resolution (LR) input views, but independently enhancing each image introduces multi-view inconsistencies, leading to blurry renders. Prior methods attempt to mitigate these inconsistencies through learned neural components, temporally consistent video priors, or joint optimization on LR and SR views, but all uniformly apply SR across every image. In contrast, our key insight is that close-up LR views may contain high-frequency information for regions also captured in more distant views, and that we can use the camera pose relative to scene geometry to inform where to add SR content. Building from this insight, we propose SplatSuRe, a method that selectively applies SR content only in undersampled regions lacking high-frequency supervision, yielding sharper and more consistent results. Across Tanks & Temples, Deep Blending and Mip-NeRF 360, our approach surpasses baselines in both fidelity and perceptual quality. Notably, our gains are most significant in localized foreground regions where higher detail is desired.*

## Setup
We follow the setup of [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and install additional libraries for evaluation. Alternatively, you can follow the setup steps in `setup_main.sh` using conda and pip:

```
git clone git@github.com:pranav-asthana/SplatSuRe.git --recursive
cd SplatSuRe
conda create -y --name splatsure python=3.11
conda activate splatsure
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install --no-build-isolation submodules/*
pip install -r requirements.txt
```

To evaluate additional metrics CMMD, DreamSim, NIQE and MUSIQ, install the additional dependencies following the steps in `setup_metrics.sh`: 
```
# Install other dependencies for additional metrics (Optional)
pip install dreamsim pyiqa
pip install peft==0.10.0
git clone git@github.com:sayakpaul/cmmd-pytorch.git
cd cmmd-pytorch
pip install -r requirements.txt
cd ..
```

After setup, store data in COLMAP format, same as that used in 3DGS. Run Single Image Super-Resolution (SISR) using any SISR model and store the super-resolved images alongside `images` in the COLMAP directory. We use [StableSR](https://github.com/IceClear/StableSR) as the SISR model in our paper. StableSR outputs for all scenes used in our work can be found [here](https://drive.google.com/drive/folders/1mhEKcvJtxhPCrsTveRFbEkTSqerJyjvb?usp=share_link). Additionally, COLMAP outputs for scenes in Tanks \& Temples can also be found [here](https://drive.google.com/drive/folders/1iNMynWtvRg1N--YyqpB1HaPkt-kURj04?usp=share_link). The data directory structure should look like this:
```
{data_dir}
├── distorted    # COLMAP generated
├── images       # Undistorted images produced by COLMAP
├── images_SR    # Super-resolved images
├── input        # Input ground truth images
├── sparse       # COLMAP generated
└── stereo       # COLMAP generated
```

## Training and Evaluation
A training and evaluation script has been provided in `train_and_eval.sh`.
First, set the following variables
```
scene='Train'
upscale=4                        # Super-resolution factor
ratio_threshold=1.1              # Ratio threshold used for generating weight maps, refer to Section 4.1 of the paper
weight_maps_dirname=weight_maps  # Directory to store generated weight maps
output_dir=outputs_${upscale}x   # Path to output directory
data_dir=data/tandt/${scene}     # Path to data directory in COLMAP format
r=8                              # Downsampling factor for original images
sr_images_dir=images_SR          # Directory name where SR images are stored (within data directory)
```

After setting the variables, train a low-resolution model and generate the weight maps:
```
# Train LR model
python train_lr.py \
  -s ${data_dir} \
  -m ${output_dir}/lr/${scene} \
  -r ${r} \
  --eval

# Get weight maps
python weight_maps.py \
  -s ${data_dir} \
  -m ${output_dir}/lr/${scene} \
  -r ${r} \
  --eval \
  --weight_maps_dirname ${weight_maps_dirname} \
  --ratio_threshold ${ratio_threshold}
```
Here, additional arguments (apart from those used in 3DGS) include `weight_maps_dirname` and `ratio_threshold` explained above. The original images are downsampled by a factor `r` to serve as the low-resolution. This value can be set to 1 if using the images directly as LR.

Now train the high-resolution model using selective weighted super-resolution images and render the test views:
```
# Train SR model
python train.py \
  -s ${data_dir} \
  -m ${output_dir}/${scene} \
  -r 1 \
  --eval \
  --images ${sr_images_dir} \
  --img_ext png \
  --upscale ${upscale} \
  --weight_maps_path ${output_dir}/lr/${scene}/${weight_maps_dirname}

python render.py \
  --model_path ${output_dir}/${scene} \
  --skip_train \
  --images images \
  -r ${r} \
  --img_ext jpg \
  --upscale ${upscale}
```
Here, `img_ext` refers to the filename extension of SR images (since this may be different from that in the LR images) and `upscale` refers to the upscaling or SR factor.

Finally, evaluate metrics SSIM, PSNR, LPIPS and FID using the modified `metrics.py` script. We use additional scripts to evaluate [DreamSim](https://github.com/ssundaram21/dreamsim) and [CMMD](https://github.com/sayakpaul/cmmd-pytorch), and use [pyiqa](https://github.com/chaofengc/IQA-PyTorch) to evaluate NIQE and MUSIQ.
```
# Metrics
python metrics.py -m ${output_dir}/${scene}
```

Another script has been provided in `additional_metrics.sh` to evaluate additional metrics using [DreamSim](https://github.com/ssundaram21/dreamsim), and [CMMD](https://github.com/sayakpaul/cmmd-pytorch), and we use [pyiqa](https://github.com/chaofengc/IQA-PyTorch) to evaluate NIQE and MUSIQ.

```
# Additional Metrics
python cmmd-pytorch/main.py ${output_dir}/${scene}/test/ours_30000/renders ${output_dir}/${scene}/test/ours_30000/gt > ${output_dir}/${scene}/cmmd.txt
python eval_dreamsim.py -m ${output_dir}/${scene} > ${output_dir}/${scene}/dreamsim.txt
pyiqa niqe musiq -t ${output_dir}/${scene}/test/ours_30000/renders > ${output_dir}/${scene}/niqe_musiq.txt
```
The metrics are stored in the output directory, with SSIM, PSNR, LPIPS and FID in `results.json` and the additional metrics in their corresponding text files for DreamSim, CMMD and NIQE and MUSIQ.

## Results
![Qualiitative Figure](assets/main_qualitative.png)

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@InProceedings{Asthana2025SplatSuRe,
    author    = {Asthana, Pranav and Hanson, Alex and Tu, Allen and Goldstein, Tom and Zwicker, Matthias and Varshney, Amitabh},
    title     = {SplatSuRe: Selective Super-Resolution for Multi-view Consistent 3D Gaussian Splatting},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {11840-11849},
    url       = {https://splatsure.github.io/}
}
</code></pre>
  </div>
</section>



## Funding and Acknowledgments
This work was made possible by NSF Grants 21-37229 and 22-35050, DARPA TIAMAT, and the NSF TRAILS Institute (2229885). This research is based upon work supported by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA), via IARPA R&D Contract No. 140D0423C0076. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of the ODNI, IARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright annotation thereon. Additional support was provided by Coefficient Giving.

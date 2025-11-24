#!/bin/bash

set -e

scene='Train'
upscale=4
ratio_threshold=1.1
weight_maps_dirname=weight_maps
output_dir=./outputs_${upscale}x

if [[ $scene = @(Auditorium|Ignatius|Palace|Ballroom|Courthouse|Panther|Barn|Lighthouse|Playground|Courtroom|M60|Temple|Caterpillar|Family|Meetingroom|Train|Francis|Truck|Church|Horse|Museum) ]]; then
  data_dir=./data/tandt/${scene}
  r=8
fi
if [[ $scene = @(bicycle|bonsai|counter|flowers|garden|kitchen|room|stump|treehill) ]]; then
  data_dir=data/mipnerf_data/${scene}
  r=8
fi
if [[ $scene = @(drjohnson|playroom) ]]; then
  data_dir=data/deep_blending/${scene}/colmap
  r=4
fi

source ~/.bashrc;

# Train LR model
conda activate gaussian_splatting;
python train_lr.py -s ${data_dir} -m ${output_dir}/lr/${scene} -r ${r} --eval

conda activate mine_3dgs;

# Get weight masks
python weight_masks.py -s ${data_dir} -m ${output_dir}/lr/${scene} -r ${r} --eval --img_ext png --weight_maps_dirname ${weight_maps_dirname} --ratio_threshold ${ratio_threshold}


# # Train SR model
python train.py -s ${data_dir} -m ${output_dir}/${scene} -r 1 --eval --images images_${r}_${upscale}x --img_ext png --upscale ${upscale} --weight_maps_path ${output_dir}/lr/${scene}/${weight_maps_dirname}

python render.py --model_path ${output_dir}/${scene} --skip_train --images images -r ${r} --img_ext jpg --upscale ${upscale}

# # Metrics
python metrics.py -m ${output_dir}/${scene}
python cmmd-pytorch/main.py ${output_dir}/${scene}/test/ours_30000/renders ${output_dir}/${scene}/test/ours_30000/gt > ${output_dir}/${scene}/cmmd.txt
python eval_dreamsim.py -m ${output_dir}/${scene} > ${output_dir}/${scene}/dreamsim.txt
pyiqa niqe musiq -t ${output_dir}/${scene}/test/ours_30000/renders > ${output_dir}/${scene}/niqe_musiq.txt

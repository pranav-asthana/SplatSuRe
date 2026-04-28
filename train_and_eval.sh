#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

scene='toy'
upscale=4
ratio_threshold=1.1
r=4

dataset_root="${DATASET_ROOT:-${SCRIPT_DIR}/dataset}"
if [ ! -d "${dataset_root}" ] && [ -d "${SCRIPT_DIR}/../dataset" ]; then
	dataset_root="${SCRIPT_DIR}/../dataset"
fi
data_dir="${dataset_root}/${scene}"
sr_images_dirname="images_4x"
weight_maps_dirname="weight_maps"
output_dir="outputs_${upscale}x"

if [ ! -d "${data_dir}" ]; then
	echo "Dataset path not found: ${data_dir}" >&2
	exit 1
fi

if [ ! -d "${data_dir}/images" ]; then
	echo "Missing LR images directory: ${data_dir}/images" >&2
	exit 1
fi

if [ ! -d "${data_dir}/${sr_images_dirname}" ]; then
	if [ -d "${data_dir}/images_SR" ]; then
		sr_images_dirname="images_SR"
	else
		echo "Missing SR images directory: ${data_dir}/images_4x (or images_SR)" >&2
		exit 1
	fi
fi

if ! python -c "import torch" >/dev/null 2>&1; then
	echo "PyTorch is not installed in the current Python environment." >&2
	echo "Activate your env first (e.g., conda activate splatsure)." >&2
	exit 1
fi

get_ext() {
	local folder="$1"
	local first_file
	first_file="$(find "$folder" -maxdepth 1 -type f | head -n 1)"
	if [ -z "$first_file" ]; then
		echo ""
		return
	fi
	basename "$first_file" | awk -F. 'NF>1 {print $NF}'
}

sr_img_ext="$(get_ext "${data_dir}/${sr_images_dirname}")"
lr_img_ext="$(get_ext "${data_dir}/images")"

if [ -z "${sr_img_ext}" ] || [ -z "${lr_img_ext}" ]; then
	echo "Could not infer image extensions from ${data_dir}/images and ${data_dir}/${sr_images_dirname}" >&2
	exit 1
fi

lr_model_path="${output_dir}/lr/${scene}"
sr_model_path="${output_dir}/${scene}"
weight_maps_path="${lr_model_path}/${weight_maps_dirname}"

# Train LR model
python train_lr.py -s "${data_dir}" -m "${lr_model_path}" -r "${r}" --eval

# Get weight maps
python weight_maps.py -s "${data_dir}" -m "${lr_model_path}" -r "${r}" --eval --weight_maps_dirname "${weight_maps_dirname}" --ratio_threshold "${ratio_threshold}"

# Train SR model
python train.py -s "${data_dir}" -m "${sr_model_path}" -r 1 --eval --images "${sr_images_dirname}" --img_ext "${sr_img_ext}" --upscale "${upscale}" --weight_maps_path "${weight_maps_path}"

# Render
python render.py --model_path "outputs_4x/toy" --skip_train --images "images" -r 4 --img_ext "${lr_img_ext}" --upscale 4
# removing skip train does not work properly
# Metrics
python metrics.py -m "${sr_model_path}"
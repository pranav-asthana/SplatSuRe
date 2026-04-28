#!/bin/bash
# =============================================================================
# v7_final.sh — Universal Dense Init + Unified Dataset Root
# =============================================================================
#
# Final Configuration:
#  1. Unified Dataset: All scenes pull from /RealSr4x_Swin.
#  2. Universal MVS: fused.ply used for initialization across ALL scenes.
#  3. Adaptive Depth: Depth L1 and Pseudo-view Pearson are ZEROED where 
#     ρ < 0.1 (cycle, bike, face, still3).
#  4. Uniform Pruning: Threshold locked at 0.005 in train.py to protect 
#     smooth surfaces.
# =============================================================================

set -e

REPO="/home/suresh/Documents/SplatSuRe"
DATASET="/home/suresh/Documents/dataset"
OUTPUT="/home/suresh/Documents/OUTPUTS/Splatsure_v1"
PRED="$OUTPUT/submissions"

cd "$REPO"

#── PASS 1A: Thin + reliable depth (aeroplane, toy) ──────────────────────────
#Active depth signal (ρ=0.855/0.911) + Universal MVS Init
python train_splatsure_competition.py \
  --repo-root    "$REPO" \
  --dataset-root "$DATASET" \
  --output-root  "$OUTPUT" \
  --pred-root    "$PRED" \
  --scenes aeroplane \
  --lr-iterations  15000 \
  --sr-iterations  50000 \
  --ratio-threshold 1.1 \
  --upscale 4 \
  --extra-train-args "--opacity_reset_interval 3000 --densify_grad_threshold 0.00015" \
  --extra-lr-args "--pseudo_view_weight 0.03 \
                 --pseudo_view_interval 10 \
                 --pseudo_view_start 1000 \
                 --pseudo_view_end 8000 \
                 --pseudo_view_model_size small \
                 --densify_until_iter 10000 \
                 --position_lr_max_steps 15000" \
  --extra-sr-args "--gamma 0.4 \
                 --position_lr_max_steps 50000"

# ── PASS 2B: Smooth + no depth (face, still3) ────────────────────────────────
# Depth Loss ZERO + Universal MVS Init + Conservative ratio
python train_splatsure_competition.py \
  --repo-root    "$REPO" \
  --dataset-root "$DATASET" \
  --output-root  "$OUTPUT" \
  --pred-root    "$PRED" \
  --scenes still3 \
  --lr-iterations  15000 \
  --sr-iterations  50000 \
  --ratio-threshold 1.5 \
  --upscale 4 \
  --extra-train-args "--opacity_reset_interval 3000 --densify_grad_threshold 0.0003 --depth_l1_weight_init 0 --depth_l1_weight_final 0" \
  --extra-lr-args "--pseudo_view_weight 0.0" \
  --extra-sr-args "--gamma 0.4"

# ── PASS 1B: Thin + no depth (cycle, bike) ───────────────────────────────────
# Depth Loss ZERO + Universal MVS Init
python train_splatsure_competition.py \
  --repo-root    "$REPO" \
  --dataset-root "$DATASET" \
  --output-root  "$OUTPUT" \
  --pred-root    "$PRED" \
  --scenes cycle bike \
  --lr-iterations  15000 \
  --sr-iterations  50000 \
  --ratio-threshold 1.1 \
  --upscale 4 \
  --extra-train-args "--opacity_reset_interval 3000 --densify_grad_threshold 0.00015 --depth_l1_weight_init 0 --depth_l1_weight_final 0" \
  --extra-lr-args "--pseudo_view_weight 0.0" \
  --extra-sr-args "--gamma 0.4"

# ── PASS 2A: Smooth + reliable depth (buddha, firehydrant) ───────────────────
# Active depth signal (ρ=0.890/0.985) + Universal MVS Init
python train_splatsure_competition.py \
  --repo-root    "$REPO" \
  --dataset-root "$DATASET" \
  --output-root  "$OUTPUT" \
  --pred-root    "$PRED" \
  --scenes buddha firehydrant \
  --lr-iterations  15000 \
  --sr-iterations  50000 \
  --ratio-threshold 1.5 \
  --upscale 4 \
  --extra-train-args "--opacity_reset_interval 3000 --densify_grad_threshold 0.0003" \
  --extra-lr-args "--pseudo_view_weight 0.05 --pseudo_view_interval 10 --pseudo_view_start 2000 --pseudo_view_end 10000 --pseudo_view_model_size small" \
  --extra-sr-args "--gamma 0.4"



echo "============================================================"
echo "  v7_final complete. Initialization: Universal fused.ply"
echo "============================================================"
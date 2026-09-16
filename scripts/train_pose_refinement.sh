#!/usr/bin/env bash
set -euo pipefail

dataset=$1
workspace=$2
gpu_id=$3
geometry_mod_multiscale=${4:-1}
audio_extractor=${5:-deepspeech}

export CUDA_VISIBLE_DEVICES=$gpu_id

python train_face.py -s "$dataset" -m "$workspace" --init_num 2000 \
    --densify_grad_threshold 0.0005 --audio_extractor "$audio_extractor" \
    --geometry_mod_multiscale "$geometry_mod_multiscale" --pose_refinement \
    --pose_max_translation_ratio 0.02

python train_mouth.py -s "$dataset" -m "$workspace" \
    --audio_extractor "$audio_extractor" --pose_refinement

python train_fuse.py -s "$dataset" -m "$workspace" --opacity_lr 0.001 \
    --audio_extractor "$audio_extractor" \
    --geometry_mod_multiscale "$geometry_mod_multiscale" --pose_refinement

python synthesize_fuse.py -s "$dataset" -m "$workspace" --eval \
    --audio_extractor "$audio_extractor" \
    --geometry_mod_multiscale "$geometry_mod_multiscale"

python metrics.py "$workspace/test/ours_None/renders/out.mp4" \
    "$workspace/test/ours_None/gt/out.mp4"

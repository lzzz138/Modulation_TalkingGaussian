#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${3:?GPU ID required}"
dataset=${1:?Dataset required}
workspace=${2:?Output directory required}
resume_args=()
if [[ -n "${5:-}" ]]; then
    resume_args=(--resume "$5")
fi
python train_uvw_pose.py -s "$dataset" -m "$workspace" --audio_extractor deepspeech --geometry_mod_multiscale "${4:-1}" "${resume_args[@]}"
python train_fuse.py --uvw_pose -s "$dataset" -m "$workspace" --opacity_lr 0.001 --audio_extractor deepspeech --geometry_mod_multiscale "${4:-1}"
python synthesize_fuse.py --uvw_pose -s "$dataset" -m "$workspace" --eval --audio_extractor deepspeech --geometry_mod_multiscale "${4:-1}"
python metrics.py "$workspace/test/ours_None/renders/out.mp4" "$workspace/test/ours_None/gt/out.mp4"

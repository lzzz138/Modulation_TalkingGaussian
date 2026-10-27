#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${3:?GPU ID required}"
dataset=${1:?Dataset required}
workspace=${2:?Output required}
python_bin=${PYTHON:-python}
condition=${PCV_CONDITION:-pose_visibility}
if [[ -e "$workspace/chkpnt_face_latest.pth" ]]; then
    echo 'Use a fresh output directory; resume individual stages with --start_checkpoint.' >&2
    exit 1
fi
if [[ -n "${4:-}" ]]; then
    "$python_bin" -m utils.pcv_checkpoint --mouth "$4" --source "$dataset" --output "$workspace"
else
    "$python_bin" train_mouth.py -s "$dataset" -m "$workspace" --audio_extractor deepspeech
fi
common=(-s "$dataset" -m "$workspace" --audio_extractor deepspeech --geometry_mod_multiscale 1 --pose_visibility_deformation --pcv_condition "$condition")
"$python_bin" train_face.py "${common[@]}" --init_num 2000 --densify_grad_threshold 0.0005
"$python_bin" train_fuse.py "${common[@]}" --opacity_lr 0.001
"$python_bin" synthesize_fuse.py "${common[@]}" --eval
"$python_bin" metric.py "$workspace/test/ours_None/renders/out.mp4" "$workspace/test/ours_None/gt/out.mp4" --batch_size 4

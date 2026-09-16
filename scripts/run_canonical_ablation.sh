#!/usr/bin/env bash
set -euo pipefail

dataset=$1
canonical_python=$2
densemarks_repo=$3
densemarks_weights=$4
gpu_id=$5

export CUDA_VISIBLE_DEVICES=$gpu_id

for mode in universal identity expression_invariant; do
    python data_utils/align_canonical.py \
        --data "$dataset" \
        --canonical_python "$canonical_python" \
        --densemarks_repo "$densemarks_repo" \
        --densemarks_weights "$densemarks_weights" \
        --template_mode "$mode" \
        --output "track_params_canonical_${mode}.pt" \
        --keep_cache
done

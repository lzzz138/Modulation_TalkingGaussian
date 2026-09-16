# Direct canonical head alignment

This preprocessing path uses frozen DenseMarks UVW predictions to refine every
frame directly against a fixed canonical template. It does not use optical flow
or the pose from a neighboring frame.

DenseMarks requires Python 3.10 or newer, while TalkingGaussian commonly runs
under Python 3.7. Install DenseMarks in a separate environment following its
official instructions, including the `third_party_dinov3` checkout and the
released `model.safetensors` checkpoint.

The checkpoint is CC BY-NC 4.0. Keep it outside this repository and pass its
path explicitly.

```bash
python data_utils/process.py data/ID/ID.mp4 \
  --head_stabilizer canonical \
  --canonical_python /path/to/densemarks-env/bin/python \
  --densemarks_repo /path/to/densemarks \
  --densemarks_weights /path/to/model.safetensors \
  --canonical_template_mode expression_invariant \
  --canonical_keep_cache
```

For already processed data, run only canonical alignment and then regenerate
the transforms:

```bash
python data_utils/align_canonical.py \
  --data data/ID \
  --canonical_python /path/to/densemarks-env/bin/python \
  --densemarks_repo /path/to/densemarks \
  --densemarks_weights /path/to/model.safetensors \
  --keep_cache

python data_utils/process.py data/ID/ID.mp4 --task 9 \
  --track_params track_params_canonical.pt
```

Use `universal`, `identity`, and `expression_invariant` as the three template
ablation modes. Keep the feature cache when running more than one mode so the
expensive DenseMarks inference is performed only once.

`--overwrite` replaces alignment outputs while reusing a validated feature
cache. Use `--rebuild_cache` only when the images, DenseMarks checkout, weights,
or feature resolution intentionally changed.

The helper below runs all three modes and keeps separate parameter, template,
and diagnostic files:

```bash
bash scripts/run_canonical_ablation.sh data/ID \
  /path/to/densemarks-env/bin/python \
  /path/to/densemarks \
  /path/to/model.safetensors \
  0
```

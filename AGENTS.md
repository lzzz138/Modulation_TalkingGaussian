# Repository Guidelines

## Project Structure & Module Organization

This repository implements TalkingGaussian training, preprocessing, synthesis, and metrics. Top-level training entry points are `train_mouth.py`, `train_face.py`, and `train_fuse.py`; `synthesize_fuse.py` renders trained models, and `metrics.py` evaluates generated videos. Core model and scene logic lives in `scene/`, rendering code in `gaussian_renderer/`, shared helpers in `utils/`, CLI argument definitions in `arguments/`, and CUDA/grid encoding code in `gridencoder/` plus `submodules/`. Data preparation tools are under `data_utils/`, including face parsing, tracking, and DeepSpeech feature extraction. Scripts are in `scripts/`; datasets and generated outputs are expected under `data/<ID>/` and `output/<run_name>/`.

## Build, Test, and Development Commands

Create the expected environment:

```bash
conda env create --file environment.yml
conda activate talking_gaussian
```

Prepare repository assets:

```bash
bash scripts/prepare.sh
python data_utils/process.py data/<ID>/<ID>.mp4
```

Train and render:

```bash
bash scripts/train_xx.sh data/<ID> output/<run_name> <GPU_ID>
python synthesize_fuse.py -S data/<ID> -M output/<run_name> --eval
python metrics.py output/<run_name>/test/ours_None/renders/out.mp4 output/<run_name>/test/ours_None/gt/out.mp4
```

Use `python -m py_compile <files>` for a quick syntax check after Python edits. There is no formal test suite in this repository.

## Coding Style & Naming Conventions

Use Python with 4-space indentation and keep changes localized to the relevant training, scene, utility, or preprocessing module. Follow existing naming: snake_case for functions and variables, PascalCase for classes such as `GaussianModel` or `MotionNetwork`, and short CLI flags consistent with `arguments/`. Prefer structured subprocess calls over shell-string execution for new scripts. Avoid committing generated datasets, checkpoints, cache folders, or render outputs.

## Testing Guidelines

Validate preprocessing by checking that `data/<ID>/` contains `aud.npy`, `au.csv`, `bc.jpg`, `transforms_train.json`, `transforms_val.json`, `ori_imgs/*.lms`, `gt_imgs/`, `torso_imgs/`, and `parsing/`. For training changes, run a short smoke test on a small dataset or a fresh output directory. For CUDA-related edits, verify both import/syntax and at least one forward/training step when hardware is available.

## Commit & Pull Request Guidelines

Recent history uses short imperative summaries, for example `accelerate loading a bit` or `add data loading on the fly`. Keep commit subjects concise and focused. Pull requests should include the motivation, affected modules, exact commands run, environment notes such as CUDA/PyTorch versions, and representative output paths or screenshots for rendering changes.

## Security & Configuration Tips

This project processes real videos and voice features. Keep private datasets and model outputs out of version control. Respect the README usage restrictions and source-video licenses. Pin CUDA, PyTorch, TensorFlow, and protobuf versions carefully; mismatches can cause runtime crashes in preprocessing or training.

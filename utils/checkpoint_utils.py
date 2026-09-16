"""Training-state helpers for reproducible pose-refinement resumes."""

import random

import numpy as np
import torch


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def camera_stack_from_ids(cameras, frame_ids):
    by_id = {int(camera.talking_dict["img_id"]): camera for camera in cameras}
    missing = [frame_id for frame_id in frame_ids if frame_id not in by_id]
    if missing:
        raise RuntimeError("Checkpoint camera stack contains unknown frame IDs: {}".format(missing[:5]))
    return [by_id[frame_id] for frame_id in frame_ids]


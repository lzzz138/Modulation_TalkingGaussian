"""Extract DenseMarks UVW maps in a separate Python 3.10+ environment."""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch
from PIL import Image


def _sha256(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            block = file.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _numeric_images(image_dir):
    result = []
    for name in os.listdir(image_dir):
        stem, extension = os.path.splitext(name)
        if stem.isdigit() and extension.lower() in (".jpg", ".jpeg", ".png"):
            result.append((int(stem), os.path.join(image_dir, name)))
    return sorted(result)


def _head_mask(parsing_path, size):
    if not os.path.exists(parsing_path):
        return np.ones((size, size), dtype=np.uint8)
    parsing = np.asarray(
        Image.open(parsing_path).convert("RGB").resize((size, size), Image.NEAREST)
    )
    white = np.all(parsing == np.asarray([255, 255, 255]), axis=-1)
    green = np.all(parsing == np.asarray([0, 255, 0]), axis=-1)
    blue = np.all(parsing == np.asarray([0, 0, 255]), axis=-1)
    return (~(white | green | blue)).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Cache DenseMarks canonical UVW maps")
    parser.add_argument("--images", required=True)
    parser.add_argument("--parsing", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--densemarks_repo", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if sys.version_info < (3, 10):
        raise RuntimeError("DenseMarks extraction requires Python 3.10 or newer")
    model_file = os.path.join(args.densemarks_repo, "dense_marks_model.py")
    dinov3_dir = os.path.join(args.densemarks_repo, "third_party_dinov3")
    if not os.path.exists(model_file):
        raise RuntimeError("Missing DenseMarks model file: %s" % model_file)
    if not os.path.isdir(dinov3_dir):
        raise RuntimeError("Missing DenseMarks DINOv3 checkout: %s" % dinov3_dir)
    if not os.path.isfile(args.weights):
        raise RuntimeError("Missing DenseMarks weights: %s" % args.weights)
    frames = _numeric_images(args.images)
    if not frames:
        raise RuntimeError("No numeric image frames found in %s" % args.images)
    os.makedirs(args.output, exist_ok=True)

    sys.path.insert(0, args.densemarks_repo)
    from dense_marks_model import DenseMarksModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseMarksModel(args.weights).to(device).eval()
    records = []
    for start in range(0, len(frames), args.batch_size):
        batch = frames[start:start + args.batch_size]
        arrays = []
        for _, path in batch:
            arrays.append(np.asarray(
                Image.open(path).convert("RGB").resize(
                    (args.size, args.size), Image.BILINEAR
                )
            ))
        with torch.no_grad():
            uvw_batch = model(np.stack(arrays)).detach().cpu().numpy()
        for (frame_id, path), uvw in zip(batch, uvw_batch):
            output_path = os.path.join(args.output, "%d.npz" % frame_id)
            if os.path.exists(output_path) and not args.overwrite:
                raise RuntimeError(
                    "DenseMarks cache already exists; pass --overwrite: %s" % output_path
                )
            parsing_path = os.path.join(args.parsing, "%d.png" % frame_id)
            np.savez_compressed(
                output_path,
                uvw=uvw.astype(np.float16),
                head_mask=_head_mask(parsing_path, args.size),
            )
            records.append({
                "frame_id": frame_id,
                "image": os.path.basename(path),
                "sha256": _sha256(path),
            })
        print("[DenseMarks] %d/%d" % (min(start + len(batch), len(frames)), len(frames)))

    manifest = {
        "format": "talking_gaussian_densemarks_v1",
        "feature_size": args.size,
        "frame_count": len(frames),
        "weights_path": os.path.abspath(args.weights),
        "weights_sha256": _sha256(args.weights),
        "densemarks_repo": os.path.abspath(args.densemarks_repo),
        "frames": records,
    }
    with open(os.path.join(args.output, "manifest.json"), "w") as file:
        json.dump(manifest, file, indent=2)


if __name__ == "__main__":
    main()

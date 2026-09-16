import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils.canonical_alignment import (
    CanonicalAlignmentConfig,
    run_canonical_alignment,
)


def _require_file(path, description):
    if not path or not os.path.isfile(path):
        raise RuntimeError("Missing %s: %s" % (description, path))


def _require_dir(path, description):
    if not path or not os.path.isdir(path):
        raise RuntimeError("Missing %s: %s" % (description, path))


def _sha256(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            block = file.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Direct DenseMarks canonical pose alignment")
    parser.add_argument("--data", required=True)
    parser.add_argument("--canonical_python", required=True)
    parser.add_argument("--densemarks_repo", required=True)
    parser.add_argument("--densemarks_weights", required=True)
    parser.add_argument(
        "--template_mode",
        choices=["universal", "identity", "expression_invariant"],
        default="expression_invariant",
    )
    parser.add_argument("--feature_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--feature_cache", default=None)
    parser.add_argument(
        "--output", default=None,
        help="output parameter file; defaults to track_params_canonical.pt",
    )
    parser.add_argument("--keep_cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--rebuild_cache", action="store_true",
        help="re-run DenseMarks even when a validated feature cache exists",
    )
    parser.add_argument("--optimization_steps_coarse", type=int, default=80)
    parser.add_argument("--optimization_steps_fine", type=int, default=40)
    args = parser.parse_args()

    _require_file(args.canonical_python, "canonical Python executable")
    _require_dir(args.densemarks_repo, "DenseMarks repository")
    _require_file(args.densemarks_weights, "DenseMarks weights")
    image_dir = os.path.join(args.data, "ori_imgs")
    parsing_dir = os.path.join(args.data, "parsing")
    _require_dir(image_dir, "input image directory")
    _require_dir(parsing_dir, "face parsing directory")
    cache_dir = args.feature_cache or os.path.join(args.data, "canonical_features")
    manifest = os.path.join(cache_dir, "manifest.json")
    if not os.path.exists(manifest) or args.rebuild_cache:
        command = [
            args.canonical_python,
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "canonical_alignment", "extract_densemarks.py",
            ),
            "--images", image_dir,
            "--parsing", parsing_dir,
            "--output", cache_dir,
            "--densemarks_repo", args.densemarks_repo,
            "--weights", args.densemarks_weights,
            "--size", str(args.feature_size),
            "--batch_size", str(args.batch_size),
        ]
        if args.rebuild_cache:
            command.append("--overwrite")
        print("[INFO] running DenseMarks extractor")
        subprocess.run(command, check=True)
    else:
        with open(manifest) as file:
            cache_manifest = json.load(file)
        if cache_manifest.get("weights_sha256") != _sha256(args.densemarks_weights):
            raise RuntimeError(
                "DenseMarks cache was produced by different weights; "
                "pass --rebuild_cache to rebuild it"
            )
        cached_repo = cache_manifest.get("densemarks_repo")
        if cached_repo and os.path.realpath(cached_repo) != os.path.realpath(args.densemarks_repo):
            raise RuntimeError(
                "DenseMarks cache was produced by a different repository checkout; "
                "pass --rebuild_cache to rebuild it"
            )

    config = CanonicalAlignmentConfig(
        template_mode=args.template_mode,
        feature_size=args.feature_size,
        optimization_steps_coarse=args.optimization_steps_coarse,
        optimization_steps_fine=args.optimization_steps_fine,
    )
    output, diagnostics = run_canonical_alignment(
        args.data, cache_dir, config=config, overwrite=args.overwrite,
        output_path=args.output,
    )
    print("Canonical parameters saved to %s" % output)
    print(
        "Accepted frames: %d/%d"
        % (diagnostics["accepted_frames"], diagnostics["frames"])
    )
    if not args.keep_cache:
        shutil.rmtree(cache_dir)
        print("Removed DenseMarks feature cache: %s" % cache_dir)


if __name__ == "__main__":
    main()

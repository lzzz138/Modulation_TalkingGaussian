import hashlib
import json
import os
import tempfile
import unittest

import cv2
import numpy as np
import torch

from data_utils.canonical_alignment.aligner import (
    CanonicalAlignmentConfig,
    DenseMarksCache,
    _build_template,
    _optimize_frames,
    _project,
    _sample_map,
)


class _SyntheticCache(object):
    def __init__(self, feature, frame_count):
        self.feature = feature
        self.frame_count = frame_count

    def load(self, index, device):
        return (
            self.feature.to(device),
            torch.ones(1, 1, self.feature.shape[-2], self.feature.shape[-1], device=device),
        )


class CanonicalTemplateTest(unittest.TestCase):
    def test_expression_regression_recovers_neutral_template(self):
        frame_count = 24
        vertex_count = 12
        expressions = np.zeros((frame_count, 4), dtype=np.float32)
        expressions[:, 0] = np.linspace(-2.0, 2.0, frame_count)
        expressions[:, 1] = np.sin(np.linspace(-np.pi, np.pi, frame_count))
        neutral = np.random.RandomState(3).uniform(0.2, 0.8, (vertex_count, 3)).astype(np.float32)
        coefficients = np.random.RandomState(4).normal(0.0, 0.03, (2, vertex_count, 3))
        values = neutral[None] + np.einsum(
            "tk,kvc->tvc", expressions[:, :2], coefficients
        )
        weights = np.ones((frame_count, vertex_count), dtype=np.float32)
        geometry = torch.zeros(frame_count, 68 + vertex_count, 3)
        config = CanonicalAlignmentConfig(expression_pca_components=2)
        target, reliability = _build_template(
            "expression_invariant", values, weights, geometry, expressions,
            np.repeat(np.arange(4), 6), config,
        )
        self.assertLess(float(np.abs(target - neutral).max()), 2e-3)
        self.assertTrue(np.all(reliability > 0.9))

    def test_identity_template_robustly_rejects_one_outlier(self):
        values = np.zeros((9, 5, 3), dtype=np.float32) + 0.4
        values[-1] = 1.0
        weights = np.ones((9, 5), dtype=np.float32)
        target, _ = _build_template(
            "identity", values, weights, torch.zeros(9, 73, 3),
            np.zeros((9, 2)), np.zeros(9, dtype=np.int32),
            CanonicalAlignmentConfig(),
        )
        self.assertLess(float(np.abs(target - 0.4).max()), 0.02)


class CanonicalOptimizationTest(unittest.TestCase):
    def test_per_frame_alignment_reduces_direct_canonical_error(self):
        size = 128
        yy, xx = torch.meshgrid(
            torch.linspace(0.0, 1.0, size),
            torch.linspace(0.0, 1.0, size),
            indexing="ij",
        )
        feature = torch.stack((xx, yy, torch.zeros_like(xx)))[None]
        generator = torch.Generator().manual_seed(7)
        rigid = torch.rand(100, 3, generator=generator) - 0.5
        rigid[:, :2] *= 1.5
        rigid[:, 2] *= 0.2
        geometry = torch.zeros(1, 168, 3)
        geometry[0, 68:] = rigid
        base_rotation = torch.eye(3)[None]
        base_translation = torch.tensor([[0.0, 0.0, -5.0]])
        true_translation = torch.tensor([[0.08, -0.04, -5.0]])
        focal = 120.0
        center = torch.tensor((size / 2.0, size / 2.0))
        true_points = _project(
            geometry[:, 68:], base_rotation, true_translation, focal, center
        )[0]
        target = _sample_map(feature, true_points, size, size).numpy()
        visibility = torch.ones(1, 168)
        config = CanonicalAlignmentConfig(
            min_alignment_anchors=64,
            optimization_steps_coarse=60,
            optimization_steps_fine=30,
            lambda_pose_prior=1e-4,
            minimum_improvement=0.01,
        )
        _, translation, diagnostics = _optimize_frames(
            _SyntheticCache(feature, 1), geometry, base_rotation,
            base_translation, visibility, target, np.ones(100, dtype=np.float32),
            focal, center, size, size, config, torch.device("cpu"),
        )
        self.assertTrue(diagnostics[0]["accepted"])
        self.assertGreater(diagnostics[0]["improvement"], 0.5)
        self.assertLess(float(torch.linalg.norm(translation - true_translation)), 0.03)


class DenseMarksCacheTest(unittest.TestCase):
    def test_manifest_rejects_changed_image(self):
        with tempfile.TemporaryDirectory() as directory:
            image_dir = os.path.join(directory, "images")
            cache_dir = os.path.join(directory, "cache")
            os.makedirs(image_dir)
            os.makedirs(cache_dir)
            image_path = os.path.join(image_dir, "0.jpg")
            cv2.imwrite(image_path, np.zeros((8, 8, 3), dtype=np.uint8))
            with open(image_path, "rb") as image_file:
                digest = hashlib.sha256(image_file.read()).hexdigest()
            np.savez_compressed(
                os.path.join(cache_dir, "0.npz"),
                uvw=np.zeros((3, 16, 16), dtype=np.float16),
                head_mask=np.ones((16, 16), dtype=np.uint8),
            )
            manifest = {
                "format": "talking_gaussian_densemarks_v1",
                "feature_size": 16,
                "frame_count": 1,
                "frames": [{"frame_id": 0, "sha256": digest}],
            }
            with open(os.path.join(cache_dir, "manifest.json"), "w") as file:
                json.dump(manifest, file)
            DenseMarksCache(cache_dir, [image_path], 16)
            cv2.imwrite(image_path, np.ones((8, 8, 3), dtype=np.uint8) * 255)
            with self.assertRaises(RuntimeError):
                DenseMarksCache(cache_dir, [image_path], 16)


if __name__ == "__main__":
    unittest.main()

"""Render-guided temporal SE(3) pose refinement."""

from dataclasses import dataclass
import hashlib
import math

import torch
from torch import nn

from utils.se3 import compose_increment, matrix_multiply, matrix_vector, se3_log


@dataclass
class RefinedPose:
    world_to_camera: torch.Tensor
    rotation: torch.Tensor
    translation: torch.Tensor
    camera_center: torch.Tensor
    delta_xi: torch.Tensor
    normalized_delta: torch.Tensor


class TemporalPoseRefiner(nn.Module):
    def __init__(self, window=9, hidden_dim=64, max_rotation_deg=5.0,
                 max_translation_ratio=0.02, median_depth=1.0):
        super().__init__()
        if window < 3 or window % 2 != 1:
            raise ValueError("pose_window must be an odd integer >= 3")
        self.window = int(window)
        self.max_rotation_rad = math.radians(max_rotation_deg)
        self.max_translation = float(max_translation_ratio * median_depth)
        self.input = nn.Sequential(nn.Conv1d(18, hidden_dim, 3, padding=1), nn.SiLU())
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.GroupNorm(8, hidden_dim), nn.SiLU(),
                          nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
                          nn.GroupNorm(8, hidden_dim), nn.SiLU(),
                          nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1))
            for _ in range(2)
        ])
        self.head = nn.Conv1d(hidden_dim, 6, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, features):
        # features: [B, window, 18]
        x = self.input(features.transpose(1, 2))
        for block in self.blocks:
            x = x + block(x)
        normalized = torch.tanh(self.head(x)[:, :, self.window // 2])
        scale = normalized.new_tensor([
            self.max_rotation_rad, self.max_rotation_rad, self.max_rotation_rad,
            self.max_translation, self.max_translation, self.max_translation,
        ])
        return normalized * scale, normalized


def _camera_id(camera):
    return int(camera.talking_dict["img_id"])


def _camera_matrix(camera, device):
    matrix = getattr(camera, "coarse_world_view_transform", camera.world_view_transform)
    return matrix.transpose(0, 1).detach().to(device=device, dtype=torch.float32)


def _relative_twist(current, previous):
    current_rotation = current[..., :3, :3]
    previous_rotation = previous[..., :3, :3]
    relative_rotation = matrix_multiply(current_rotation, previous_rotation.transpose(-1, -2))
    relative_translation = current[..., :3, 3] - matrix_vector(
        relative_rotation, previous[..., :3, 3]
    )
    return se3_log(relative_rotation, relative_translation)


class PoseRefinementRuntime:
    """Owns trajectory features while the network remains a normal nn.Module."""

    FORMAT_VERSION = 1

    def __init__(self, train_cameras, test_cameras, window=9, max_rotation_deg=5.0,
                 max_translation_ratio=0.02, device="cuda"):
        self.device = torch.device(device)
        train_ids, train_matrices = self._collect(train_cameras)
        test_ids, test_matrices = self._collect(test_cameras)
        if not train_ids:
            raise ValueError("Pose refinement requires at least one training camera")

        self.reference = train_matrices[len(train_matrices) // 2]
        centers = -matrix_vector(train_matrices[:, :3, :3].transpose(1, 2),
                                 train_matrices[:, :3, 3])
        median_depth = torch.median(torch.linalg.norm(centers, dim=-1)).clamp_min(1e-3).item()
        self.median_depth = median_depth
        raw_train = self._features(train_matrices)
        self.feature_mean = raw_train.mean(dim=0)
        self.feature_std = raw_train.std(dim=0, unbiased=False).clamp_min(1e-4)
        self.ids = {"train": train_ids, "test": test_ids}
        self.matrices = {"train": train_matrices, "test": test_matrices}
        self.features = {
            "train": (raw_train - self.feature_mean) / self.feature_std,
            "test": (self._features(test_matrices) - self.feature_mean) / self.feature_std
                    if len(test_ids) else raw_train[:0],
        }
        self.indices = {split: {frame_id: i for i, frame_id in enumerate(ids)}
                        for split, ids in self.ids.items()}
        self.refiner = TemporalPoseRefiner(window, 64, max_rotation_deg,
                                           max_translation_ratio, median_depth).to(self.device)
        self.train_transform_hash = self._hash(("train",))
        self.transform_hash = self._hash()

    def _collect(self, cameras):
        ordered = sorted(cameras, key=_camera_id)
        ids = [_camera_id(camera) for camera in ordered]
        matrices = torch.stack([_camera_matrix(camera, self.device) for camera in ordered]) \
            if ordered else torch.empty((0, 4, 4), device=self.device)
        return ids, matrices

    def _features(self, matrices):
        if matrices.shape[0] == 0:
            return torch.empty((0, 18), device=self.device)
        reference = self.reference.expand(matrices.shape[0], -1, -1)
        absolute = _relative_twist(matrices, reference)
        velocity = torch.zeros_like(absolute)
        if matrices.shape[0] > 1:
            velocity[1:] = _relative_twist(matrices[1:], matrices[:-1])
            velocity[0] = velocity[1]
        acceleration = torch.zeros_like(velocity)
        if matrices.shape[0] > 2:
            acceleration[1:] = velocity[1:] - velocity[:-1]
            acceleration[0] = acceleration[1]
        absolute[:, 3:] /= self.median_depth if hasattr(self, "median_depth") else 1.0
        velocity[:, 3:] /= self.median_depth if hasattr(self, "median_depth") else 1.0
        acceleration[:, 3:] /= self.median_depth if hasattr(self, "median_depth") else 1.0
        return torch.cat((absolute, velocity, acceleration), dim=-1)

    def _window(self, split, index):
        radius = self.refiner.window // 2
        indices = torch.arange(index - radius, index + radius + 1, device=self.device)
        indices = indices.clamp(0, self.features[split].shape[0] - 1)
        return self.features[split][indices].unsqueeze(0)

    def pose(self, camera, split="train", detach=False):
        frame_id = _camera_id(camera)
        if frame_id not in self.indices[split]:
            raise KeyError("Frame {} is absent from the {} pose trajectory".format(frame_id, split))
        index = self.indices[split][frame_id]
        delta, normalized = self.refiner(self._window(split, index))
        rotation, translation = compose_increment(
            delta[0], self.matrices[split][index, :3, :3], self.matrices[split][index, :3, 3]
        )
        world_to_camera = torch.cat((
            torch.cat((rotation, translation[:, None]), dim=1),
            rotation.new_tensor([[0.0, 0.0, 0.0, 1.0]]),
        ), dim=0)
        camera_center = -matrix_vector(rotation.transpose(0, 1), translation)
        result = RefinedPose(world_to_camera, rotation, translation, camera_center,
                             delta[0], normalized[0])
        if detach:
            return RefinedPose(*(value.detach() for value in result.__dict__.values()))
        return result

    def temporal_loss(self, camera, split="train"):
        index = self.indices[split][_camera_id(camera)]
        count = len(self.ids[split])
        indices = [max(0, index - 1), index, min(count - 1, index + 1)]
        normalized = [self.refiner(self._window(split, i))[1][0] for i in indices]
        return (normalized[0] - 2.0 * normalized[1] + normalized[2]).square().mean()

    def _hash(self, splits=("train", "test")):
        digest = hashlib.sha256()
        for split in splits:
            digest.update(torch.tensor(self.ids[split], dtype=torch.int64).numpy().tobytes())
            digest.update(self.matrices[split].detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    def checkpoint(self, optimizer=None, scheduler=None):
        state_dict = {
            name: value.detach().cpu().clone()
            for name, value in self.refiner.state_dict().items()
        }
        state_digest = hashlib.sha256()
        for name, value in sorted(state_dict.items()):
            state_digest.update(name.encode("utf8"))
            state_digest.update(value.numpy().tobytes())
        return {
            "format_version": self.FORMAT_VERSION,
            "state_dict": state_dict,
            "state_hash": state_digest.hexdigest(),
            "config": {
                "window": self.refiner.window,
                "max_rotation_deg": math.degrees(self.refiner.max_rotation_rad),
                "max_translation_ratio": self.refiner.max_translation / self.median_depth,
            },
            "feature_mean": self.feature_mean.detach().cpu(),
            "feature_std": self.feature_std.detach().cpu(),
            "reference": self.reference.detach().cpu(),
            "median_depth": self.median_depth,
            "transform_hash": self.transform_hash,
            "train_transform_hash": self.train_transform_hash,
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scheduler": scheduler.state_dict() if scheduler else None,
        }

    def restore(self, payload, strict_hash=True):
        if payload.get("format_version") != self.FORMAT_VERSION:
            raise RuntimeError("Unsupported pose checkpoint format")
        expected_train_hash = payload.get("train_transform_hash", payload["transform_hash"])
        actual_hash = self.train_transform_hash if "train_transform_hash" in payload else self.transform_hash
        if strict_hash and expected_train_hash != actual_hash:
            raise RuntimeError("Pose checkpoint training transforms do not match this dataset")
        expected = payload["config"]
        actual = {
            "window": self.refiner.window,
            "max_rotation_deg": math.degrees(self.refiner.max_rotation_rad),
            "max_translation_ratio": self.refiner.max_translation / self.median_depth,
        }
        for key in actual:
            if abs(float(expected[key]) - float(actual[key])) > 1e-7:
                raise RuntimeError("Pose checkpoint {} does not match runtime configuration".format(key))
        self.refiner.load_state_dict(payload["state_dict"], strict=True)
        # In evaluation mode the repository aliases validation cameras as the
        # training list. Rebuild trajectory features with the training-time
        # reference and normalization saved in the checkpoint.
        self.reference = payload["reference"].to(self.device)
        self.median_depth = float(payload["median_depth"])
        self.feature_mean = payload["feature_mean"].to(self.device)
        self.feature_std = payload["feature_std"].to(self.device)
        self.refiner.max_translation = (
            float(expected["max_translation_ratio"]) * self.median_depth
        )
        self.features = {
            split: (self._features(self.matrices[split]) - self.feature_mean) / self.feature_std
            if self.matrices[split].shape[0] else self.feature_mean.new_empty((0, 18))
            for split in ("train", "test")
        }

    @torch.no_grad()
    def export_tables(self):
        result = {}
        for split in ("train", "test"):
            result[split] = {
                frame_id: self.pose_by_index(split, index).world_to_camera.cpu()
                for index, frame_id in enumerate(self.ids[split])
            }
        return result

    def pose_by_index(self, split, index):
        camera = type("CameraId", (), {})()
        camera.talking_dict = {"img_id": self.ids[split][index]}
        return self.pose(camera, split)

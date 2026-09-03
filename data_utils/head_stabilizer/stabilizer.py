import json
import math
import os
import warnings
from dataclasses import asdict, dataclass

import cv2
import numpy as np
import torch
from tqdm import tqdm

from data_utils.face_tracking.facemodel import Face_3DMM
from data_utils.face_tracking.render_3dmm import Render_3DMM
from data_utils.face_tracking.util import euler2rot, rot_trans_pts

from .flow import FlowEstimator, forward_backward_track
from .se3 import (
    adaptive_temporal_weight,
    compose_increment,
    relative_twist,
    rotation_to_euler,
)


@dataclass
class LPHSConfig:
    flow_backend: str = "raft"
    preset: str = "balanced"
    flow_max_side: int = 512
    visibility_max_side: int = 256
    visibility_batch_size: int = 4
    reanchor_interval: int = 25
    yaw_bin_degrees: float = 15.0
    max_keyframes: int = 7
    flow_confidence_threshold: float = 0.5
    min_effective_anchors: int = 12
    optimization_steps: int = 300
    learning_rate: float = 1e-3
    lambda_track: float = 1.0
    lambda_acceleration: float = 0.25
    lambda_prior: float = 0.01

    def apply_preset(self):
        if self.preset == "fast":
            self.flow_max_side = 384
            self.reanchor_interval = 50
        elif self.preset == "quality":
            self.flow_max_side = 100000
            self.reanchor_interval = 1
        elif self.preset != "balanced":
            raise ValueError("Unknown LPHS preset: %s" % self.preset)


def _numeric_frame_paths(image_dir):
    paths = []
    for name in os.listdir(image_dir):
        stem, extension = os.path.splitext(name)
        if extension.lower() == ".jpg" and stem.isdigit():
            paths.append((int(stem), os.path.join(image_dir, name)))
    return [path for _, path in sorted(paths)]


def _load_landmarks(image_paths):
    landmarks = []
    for image_path in image_paths:
        landmark_path = os.path.splitext(image_path)[0] + ".lms"
        if not os.path.exists(landmark_path):
            raise RuntimeError("Missing landmark file: %s" % landmark_path)
        landmarks.append(np.loadtxt(landmark_path, dtype=np.float32).reshape(68, 2))
    return np.stack(landmarks)


def _normalize_detector_confidence(scores):
    normalized = np.zeros_like(scores, dtype=np.float32)
    valid = scores > 0
    if not valid.any():
        return normalized
    low, high = np.percentile(scores[valid], [5.0, 95.0])
    scale = max(float(high - low), 1e-6)
    normalized[valid] = np.clip((scores[valid] - low) / scale, 0.0, 1.0)
    return normalized


def _semantic_weights():
    weights = np.full(68 + 464, 0.1, dtype=np.float32)
    weights[:17] = 0.5
    weights[27:36] = 1.0
    weights[68:] = 1.0
    return weights


def _project(geometry, rotation, translation, focal, center):
    camera = torch.matmul(rotation, geometry.transpose(1, 2)).transpose(1, 2)
    camera = camera + translation[:, None]
    depth = torch.clamp(camera[..., 2], max=-1e-4)
    x = -focal * camera[..., 0] / depth + center[0]
    y = focal * camera[..., 1] / depth + center[1]
    return torch.stack((x, y), dim=-1)


def _build_geometry_and_visibility(params, height, width, config, device):
    frame_count = params["exp"].shape[0]
    model_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "face_tracking", "3DMM"
    )
    model = Face_3DMM(model_dir, 100, 79, 100, 34650).to(device)
    rotation = euler2rot(params["euler"].to(device).float())
    translation = params["trans"].to(device).float()
    focal = params["focal"].to(device).float()
    center = torch.tensor((width / 2.0, height / 2.0), device=device)
    scale = min(1.0, float(config.visibility_max_side) / max(height, width))
    render_h = max(16, int(round(height * scale)))
    render_w = max(16, int(round(width * scale)))

    all_geometry = []
    all_vertex_ids = []
    all_visibility = []
    batch_size = config.visibility_batch_size
    renderer = Render_3DMM(
        float(focal.item()) * scale,
        render_h,
        render_w,
        batch_size,
        device,
    )
    for start in tqdm(range(0, frame_count, batch_size), desc="LPHS visibility"):
        end = min(start + batch_size, frame_count)
        count = end - start
        identity = params["id"].to(device).float().expand(count, -1)
        expression = params["exp"][start:end].to(device).float()
        landmarks, landmark_ids = model.get_3dlandmarks(
            identity,
            expression,
            params["euler"][start:end].to(device).float(),
            translation[start:end],
            focal,
            center,
            return_indices=True,
        )
        full_geometry = model.forward_geo(identity, expression)
        rigid_geometry = full_geometry[:, model.rigid_ids]
        anchor_geometry = torch.cat((landmarks, rigid_geometry), dim=1)
        anchor_ids = torch.cat(
            (landmark_ids, model.rigid_ids[None].expand(count, -1)), dim=1
        )

        camera_geometry = rot_trans_pts(
            full_geometry, rotation[start:end], translation[start:end]
        )
        if count < batch_size:
            padding = camera_geometry[-1:].expand(batch_size - count, -1, -1)
            visibility_geometry = torch.cat((camera_geometry, padding), dim=0)
        else:
            visibility_geometry = camera_geometry
        z_visible, front_score = renderer.vertex_visibility(visibility_geometry)
        z_visible = z_visible[:count]
        front_score = front_score[:count]
        visibility = torch.gather(
            z_visible.float() * front_score, 1, anchor_ids
        )
        all_geometry.append(anchor_geometry.cpu())
        all_vertex_ids.append(anchor_ids.cpu())
        all_visibility.append(visibility.cpu())
        del full_geometry, camera_geometry, visibility_geometry

    del renderer

    return (
        torch.cat(all_geometry),
        torch.cat(all_vertex_ids),
        torch.cat(all_visibility),
        rotation.cpu(),
        translation.cpu(),
    )


def _yaw_degrees(rotation):
    return torch.atan2(rotation[:, 2, 0], rotation[:, 2, 2]).numpy() * 180.0 / math.pi


def _select_keyframes(rotation, detector_confidence, config):
    yaw = _yaw_degrees(rotation)
    bins = np.round(yaw / config.yaw_bin_degrees).astype(np.int32)
    confidence = detector_confidence.mean(axis=1)
    selected = []
    for bin_id in np.unique(bins):
        candidates = np.flatnonzero(bins == bin_id)
        selected.append(int(candidates[np.argmax(confidence[candidates])]))
    frontal = int(np.argmin(np.abs(yaw)))
    if frontal not in selected:
        selected.append(frontal)
    if len(selected) > config.max_keyframes:
        ranked = sorted(selected, key=lambda index: abs(yaw[index]), reverse=True)
        selected = ranked[: config.max_keyframes - 1] + [frontal]
    return sorted(set(selected)), yaw, bins


class _ImageCache(object):
    def __init__(self, paths, capacity=6):
        self.paths = paths
        self.capacity = capacity
        self.cache = {}
        self.order = []

    def get(self, index):
        if index in self.cache:
            return self.cache[index]
        image = cv2.imread(self.paths[index], cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Could not read image: %s" % self.paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.cache[index] = image
        self.order.append(index)
        if len(self.order) > self.capacity:
            removed = self.order.pop(0)
            self.cache.pop(removed, None)
        return image


def _build_flow_tracks(
    image_paths, landmarks, detector_confidence, seed_points, rotation, config
):
    frame_count, anchor_count = seed_points.shape[:2]
    estimator = FlowEstimator(config.flow_backend, config.flow_max_side)
    cache = _ImageCache(image_paths)
    keyframes, yaw, yaw_bins = _select_keyframes(
        rotation, detector_confidence, config
    )
    tracks = np.zeros_like(seed_points, dtype=np.float32)
    flow_confidence = np.zeros((frame_count, anchor_count), dtype=np.float32)
    source_confidence = np.zeros_like(flow_confidence)
    tracks[0] = seed_points[0]
    flow_confidence[0] = 1.0
    source_confidence[0, :68] = detector_confidence[0]
    source_confidence[0, 68:] = 1.0

    for frame_id in tqdm(range(1, frame_count), desc="LPHS bidirectional flow"):
        forward, backward = estimator.bidirectional(
            cache.get(frame_id - 1), cache.get(frame_id)
        )
        sequential, sequential_conf = forward_backward_track(
            forward, backward, tracks[frame_id - 1]
        )
        source = source_confidence[frame_id - 1] * 0.98
        trigger = (
            frame_id % config.reanchor_interval == 0
            or yaw_bins[frame_id] != yaw_bins[frame_id - 1]
            or np.median(sequential_conf) < config.flow_confidence_threshold
        )
        if trigger:
            keyframe = min(keyframes, key=lambda index: abs(yaw[index] - yaw[frame_id]))
            if keyframe == frame_id:
                anchored = seed_points[frame_id]
                anchor_conf = np.ones(anchor_count, dtype=np.float32)
            else:
                key_forward, key_backward = estimator.bidirectional(
                    cache.get(keyframe), cache.get(frame_id)
                )
                anchored, anchor_conf = forward_backward_track(
                    key_forward, key_backward, seed_points[keyframe]
                )
            total = sequential_conf + anchor_conf + 1e-6
            tracks[frame_id] = (
                sequential * sequential_conf[:, None]
                + anchored * anchor_conf[:, None]
            ) / total[:, None]
            flow_confidence[frame_id] = np.maximum(sequential_conf, anchor_conf)
            source = np.maximum(source, anchor_conf)
        else:
            tracks[frame_id] = sequential
            flow_confidence[frame_id] = sequential_conf
        source[:68] = np.maximum(source[:68], detector_confidence[frame_id])
        source_confidence[frame_id] = source

    return tracks, flow_confidence, source_confidence, keyframes, estimator.backend


def _huber_reprojection(predicted, observed, weights, focal, delta_pixels=3.0):
    residual = torch.linalg.norm(predicted - observed, dim=-1) / focal
    delta = delta_pixels / focal
    loss = torch.where(
        residual <= delta,
        0.5 * residual * residual,
        delta * (residual - 0.5 * delta),
    )
    weight_sum = weights.sum(dim=1)
    frame_loss = (loss * weights).sum(dim=1) / torch.clamp(weight_sum, min=1e-8)
    valid = weight_sum > 1e-8
    if valid.any():
        return frame_loss[valid].mean()
    return loss.sum() * 0.0


def _pose_metrics(rotation, translation, geometry, observations, focal, center):
    with torch.no_grad():
        projected = _project(geometry, rotation, translation, focal, center)
        reprojection = torch.linalg.norm(projected - observations, dim=-1)
        twists = relative_twist(rotation, translation)
        acceleration = twists[1:] - twists[:-1]
        if acceleration.shape[0] > 0:
            median_acceleration = float(
                torch.median(torch.linalg.norm(acceleration[:, :3], dim=-1)).item()
                * 180.0 / math.pi
            )
        else:
            median_acceleration = 0.0
        return {
            "median_track_reprojection_px": float(torch.median(reprojection).item()),
            "p95_track_reprojection_px": float(
                torch.quantile(reprojection.reshape(-1), 0.95).item()
            ),
            "median_rotation_velocity_deg": float(
                torch.median(torch.linalg.norm(twists[:, :3], dim=-1)).item()
                * 180.0 / math.pi
            ),
            "median_rotation_acceleration_deg": median_acceleration,
        }


def _optimize_pose(
    geometry, base_rotation, base_translation, landmarks, tracks,
    detector_weights, track_weights, focal, center, config, device,
):
    geometry = geometry.to(device).float()
    base_rotation = base_rotation.to(device).float()
    base_translation = base_translation.to(device).float()
    landmarks = torch.from_numpy(landmarks).to(device)
    tracks = torch.from_numpy(tracks).to(device)
    detector_weights = torch.from_numpy(detector_weights).to(device)
    track_weights = torch.from_numpy(track_weights).to(device)
    focal = torch.as_tensor(float(focal), device=device)
    center = torch.as_tensor(center, dtype=torch.float32, device=device)
    depth = torch.median(torch.abs(base_translation[:, 2])).detach()
    base_twist = relative_twist(base_rotation, base_translation).detach()
    temporal_weight = adaptive_temporal_weight(base_twist, depth).detach()
    delta = torch.nn.Parameter(
        torch.zeros(base_rotation.shape[0], 6, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam([delta], lr=config.learning_rate)
    history = []
    stable_steps = 0
    previous_loss = None

    for step in tqdm(range(config.optimization_steps), desc="LPHS SE(3) optimization"):
        rotation, translation = compose_increment(delta, base_rotation, base_translation)
        projected = _project(geometry, rotation, translation, focal, center)
        detector_loss = _huber_reprojection(
            projected[:, :68], landmarks, detector_weights, focal
        )
        track_loss = _huber_reprojection(projected, tracks, track_weights, focal)
        twists = relative_twist(rotation, translation)
        scaled_twists = torch.cat(
            (twists[:, :3], twists[:, 3:] / depth), dim=-1
        )
        acceleration = scaled_twists[1:] - scaled_twists[:-1]
        acceleration_weight = 0.5 * (
            temporal_weight[1:] + temporal_weight[:-1]
        )
        if acceleration.shape[0] > 0:
            acceleration_loss = (
                acceleration.square().sum(dim=-1) * acceleration_weight
            ).mean()
        else:
            acceleration_loss = delta.sum() * 0.0
        scaled_delta = torch.cat((delta[:, :3], delta[:, 3:] / depth), dim=-1)
        prior_loss = scaled_delta.square().mean()
        loss = (
            detector_loss
            + config.lambda_track * track_loss
            + config.lambda_acceleration * acceleration_loss
            + config.lambda_prior * prior_loss
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            rotation_norm = torch.linalg.norm(delta[:, :3], dim=-1, keepdim=True)
            delta[:, :3] *= torch.clamp(
                math.radians(10.0) / torch.clamp(rotation_norm, min=1e-8), max=1.0
            )
            translation_norm = torch.linalg.norm(delta[:, 3:], dim=-1, keepdim=True)
            delta[:, 3:] *= torch.clamp(
                (0.05 * depth) / torch.clamp(translation_norm, min=1e-8), max=1.0
            )
        loss_value = float(loss.detach().item())
        history.append(loss_value)
        if previous_loss is not None:
            relative_change = abs(previous_loss - loss_value) / max(abs(previous_loss), 1e-12)
            stable_steps = stable_steps + 1 if relative_change < 1e-5 else 0
            if stable_steps >= 20:
                break
        previous_loss = loss_value

    with torch.no_grad():
        rotation, translation = compose_increment(delta, base_rotation, base_translation)
    return rotation, translation, history


def run_lphs(data_dir, config=None, overwrite=False):
    config = config or LPHSConfig()
    config.apply_preset()
    output_path = os.path.join(data_dir, "track_params_lphs.pt")
    if os.path.exists(output_path) and not overwrite:
        raise RuntimeError("LPHS output already exists; pass --overwrite: %s" % output_path)
    source_path = os.path.join(data_dir, "track_params.pt")
    confidence_path = os.path.join(data_dir, "landmark_scores.npy")
    image_dir = os.path.join(data_dir, "ori_imgs")
    if not os.path.exists(source_path):
        raise RuntimeError("Missing coarse tracking parameters: %s" % source_path)
    if not os.path.exists(confidence_path):
        raise RuntimeError(
            "Missing landmark_scores.npy. Re-run data_utils/process.py with --task 7."
        )

    image_paths = _numeric_frame_paths(image_dir)
    if not image_paths:
        raise RuntimeError("No numeric JPG frames found in %s" % image_dir)
    first_image = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)
    height, width = first_image.shape[:2]
    params = torch.load(source_path, map_location="cpu")
    frame_count = params["euler"].shape[0]
    if frame_count < 3:
        raise RuntimeError("LPHS requires at least 3 tracked frames")
    if len(image_paths) != frame_count:
        raise RuntimeError(
            "Frame/pose mismatch: %d images versus %d poses" % (len(image_paths), frame_count)
        )
    landmarks = _load_landmarks(image_paths)
    detector_scores = np.load(confidence_path).astype(np.float32)
    if detector_scores.shape != (frame_count, 68):
        raise RuntimeError(
            "landmark_scores.npy must have shape (%d, 68), got %s"
            % (frame_count, detector_scores.shape)
        )
    detector_confidence = _normalize_detector_confidence(detector_scores)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("LPHS geometry visibility requires CUDA, like the base face tracker")

    geometry, vertex_ids, visibility, base_rotation, base_translation = (
        _build_geometry_and_visibility(params, height, width, config, device)
    )
    focal = float(params["focal"].reshape(-1)[0].item())
    center = (width / 2.0, height / 2.0)
    with torch.no_grad():
        projected = _project(
            geometry.float(), base_rotation.float(), base_translation.float(),
            focal, center,
        ).numpy()
    seed_points = projected.copy()
    seed_points[:, :68] = landmarks
    tracks, flow_confidence, source_confidence, keyframes, actual_backend = (
        _build_flow_tracks(
            image_paths, landmarks, detector_confidence, seed_points,
            base_rotation, config,
        )
    )

    semantic = _semantic_weights()[None]
    visibility_np = visibility.numpy().astype(np.float32)
    detector_weights = (
        semantic[:, :68]
        * visibility_np[:, :68]
        * flow_confidence[:, :68]
        * detector_confidence
    )
    track_weights = semantic * visibility_np * flow_confidence * source_confidence
    effective = (track_weights > 0.05).sum(axis=1)
    weak_frames = effective < config.min_effective_anchors
    detector_weights[weak_frames] = 0.0
    track_weights[weak_frames] = 0.0

    before_metrics = _pose_metrics(
        base_rotation.float(), base_translation.float(), geometry.float(),
        torch.from_numpy(tracks), focal, center,
    )
    rotation, translation, loss_history = _optimize_pose(
        geometry, base_rotation, base_translation, landmarks, tracks,
        detector_weights, track_weights, focal, center, config, device,
    )
    after_metrics = _pose_metrics(
        rotation, translation, geometry.to(device).float(),
        torch.from_numpy(tracks).to(device), focal,
        torch.as_tensor(center, device=device),
    )

    result = dict(params)
    result["rot"] = rotation.detach().cpu()
    result["trans"] = translation.detach().cpu()
    result["euler"] = rotation_to_euler(rotation).detach().cpu()
    result["lphs"] = {
        "config": asdict(config),
        "flow_backend": actual_backend,
        "keyframes": keyframes,
        "before": before_metrics,
        "after": after_metrics,
    }
    torch.save(result, output_path)
    np.savez_compressed(
        os.path.join(data_dir, "lphs_observations.npz"),
        tracks=tracks,
        detector_landmarks=landmarks,
        semantic_weights=semantic[0],
        visibility_weights=visibility_np,
        flow_weights=flow_confidence,
        confidence_weights=source_confidence,
        detector_confidence=detector_confidence,
        vertex_ids=vertex_ids.numpy(),
        keyframes=np.asarray(keyframes, dtype=np.int64),
    )
    diagnostics = {
        "frames": frame_count,
        "anchors": int(geometry.shape[1]),
        "weak_frames": np.flatnonzero(weak_frames).tolist(),
        "flow_backend": actual_backend,
        "keyframes": keyframes,
        "optimization_steps": len(loss_history),
        "initial_loss": loss_history[0] if loss_history else None,
        "final_loss": loss_history[-1] if loss_history else None,
        "before": before_metrics,
        "after": after_metrics,
        "config": asdict(config),
    }
    with open(os.path.join(data_dir, "lphs_diagnostics.json"), "w") as file:
        json.dump(diagnostics, file, indent=2)
    return output_path, diagnostics

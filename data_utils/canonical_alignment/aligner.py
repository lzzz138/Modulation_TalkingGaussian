import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data_utils.face_tracking.util import euler2rot
from data_utils.head_stabilizer.se3 import (
    compose_increment,
    relative_twist,
    rotation_to_euler,
)
from data_utils.head_stabilizer.stabilizer import (
    LPHSConfig,
    _build_geometry_and_visibility,
    _load_landmarks,
    _normalize_detector_confidence,
    _numeric_frame_paths,
    _project,
    _yaw_degrees,
)


@dataclass
class CanonicalAlignmentConfig:
    template_mode: str = "expression_invariant"
    feature_size: int = 512
    max_template_frames: int = 128
    yaw_bin_degrees: float = 15.0
    max_frames_per_yaw_bin: int = 16
    template_train_fraction: float = 10.0 / 11.0
    min_template_frames: int = 8
    min_visible_ratio: float = 0.60
    min_detector_confidence: float = 0.50
    min_alignment_anchors: int = 64
    optimization_steps_coarse: int = 80
    optimization_steps_fine: int = 40
    learning_rate_coarse: float = 1e-3
    learning_rate_fine: float = 2e-4
    lambda_pose_prior: float = 0.01
    max_rotation_degrees: float = 10.0
    max_translation_ratio: float = 0.05
    minimum_improvement: float = 0.05
    confidence_anchor_high_ratio: float = 1.50
    confidence_spatial_extent_low: float = 0.025
    confidence_spatial_extent_high: float = 0.075
    confidence_coverage_retention_low: float = 0.70
    confidence_bound_margin_high: float = 0.25
    lambda_temporal_velocity: float = 0.15
    lambda_temporal_acceleration: float = 0.05
    lambda_temporal_coarse: float = 0.10
    temporal_optimization_steps: int = 100
    temporal_learning_rate: float = 0.05
    expression_pca_components: int = 8
    expression_ridge: float = 1e-3
    visibility_max_side: int = 256
    visibility_batch_size: int = 4
    alignment_batch_size: int = 32

    def validate(self):
        if self.template_mode not in ("universal", "identity", "expression_invariant"):
            raise ValueError("Unknown canonical template mode: %s" % self.template_mode)
        if self.feature_size <= 0 or self.feature_size % 16 != 0:
            raise ValueError("feature_size must be a positive multiple of 16")
        if self.min_template_frames < 1:
            raise ValueError("min_template_frames must be positive")
        if not 0.0 < self.template_train_fraction <= 1.0:
            raise ValueError("template_train_fraction must be in (0, 1]")
        if self.confidence_anchor_high_ratio <= 1.0:
            raise ValueError("confidence_anchor_high_ratio must be greater than 1")
        if not 0.0 <= self.confidence_spatial_extent_low < self.confidence_spatial_extent_high:
            raise ValueError("canonical spatial extent thresholds are invalid")
        if not 0.0 <= self.confidence_coverage_retention_low < 1.0:
            raise ValueError("confidence_coverage_retention_low must be in [0, 1)")
        if not 0.0 < self.confidence_bound_margin_high <= 1.0:
            raise ValueError("confidence_bound_margin_high must be in (0, 1]")
        if min(
            self.lambda_temporal_velocity,
            self.lambda_temporal_acceleration,
            self.lambda_temporal_coarse,
        ) < 0.0:
            raise ValueError("canonical temporal loss weights must be non-negative")
        if self.temporal_optimization_steps < 0:
            raise ValueError("temporal_optimization_steps must be non-negative")
        if self.temporal_learning_rate <= 0.0:
            raise ValueError("temporal_learning_rate must be positive")


def _sha256(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            block = file.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class DenseMarksCache(object):
    def __init__(self, cache_dir, image_paths, feature_size):
        manifest_path = os.path.join(cache_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise RuntimeError("Missing DenseMarks manifest: %s" % manifest_path)
        with open(manifest_path) as file:
            self.manifest = json.load(file)
        if self.manifest.get("format") != "talking_gaussian_densemarks_v1":
            raise RuntimeError("Unsupported DenseMarks cache format")
        if self.manifest.get("feature_size") != feature_size:
            raise RuntimeError(
                "DenseMarks feature size mismatch: cache=%s requested=%s"
                % (self.manifest.get("feature_size"), feature_size)
            )
        if self.manifest.get("frame_count") != len(image_paths):
            raise RuntimeError("DenseMarks cache frame count does not match images")
        records = self.manifest.get("frames", [])
        if len(records) != len(image_paths):
            raise RuntimeError("DenseMarks manifest has an invalid frame list")
        self.cache_dir = cache_dir
        self.frame_ids = []
        for record, image_path in zip(records, image_paths):
            frame_id = int(os.path.splitext(os.path.basename(image_path))[0])
            if record.get("frame_id") != frame_id:
                raise RuntimeError("DenseMarks cache frame order does not match images")
            if record.get("sha256") != _sha256(image_path):
                raise RuntimeError("Image changed after DenseMarks extraction: %s" % image_path)
            cache_path = os.path.join(cache_dir, "%d.npz" % frame_id)
            if not os.path.exists(cache_path):
                raise RuntimeError("Missing DenseMarks frame cache: %s" % cache_path)
            self.frame_ids.append(frame_id)

    def load(self, index, device):
        path = os.path.join(self.cache_dir, "%d.npz" % self.frame_ids[index])
        with np.load(path) as data:
            uvw = np.asarray(data["uvw"], dtype=np.float32)
            mask = np.asarray(data["head_mask"], dtype=np.float32)
        if uvw.shape[0] != 3 or uvw.shape[1:] != mask.shape:
            raise RuntimeError("Invalid DenseMarks cache shape in %s" % path)
        return (
            torch.from_numpy(uvw).to(device)[None],
            torch.from_numpy(mask).to(device)[None, None],
        )


def _sample_map(feature, points, width, height, mode="bilinear"):
    x = points[..., 0] * 2.0 / max(width - 1, 1) - 1.0
    y = points[..., 1] * 2.0 / max(height - 1, 1) - 1.0
    squeeze_batch = points.ndim == 2
    if squeeze_batch:
        points = points[None]
        x = x[None]
        y = y[None]
    grid = torch.stack((x, y), dim=-1).unsqueeze(1)
    sampled = F.grid_sample(
        feature, grid, mode=mode, padding_mode="zeros", align_corners=True
    )
    sampled = sampled[:, :, 0].transpose(1, 2)
    return sampled[0] if squeeze_batch else sampled


def _in_bounds(points, width, height):
    return (
        (points[..., 0] >= 1.0)
        & (points[..., 0] <= width - 2.0)
        & (points[..., 1] >= 1.0)
        & (points[..., 1] <= height - 2.0)
    )


def _reliable_frames(
    geometry, visibility, rotation, translation, landmarks, detector_confidence,
    focal, center, width, height, config,
):
    projected = _project(geometry, rotation, translation, focal, center)
    landmark_error = torch.linalg.norm(
        projected[:, :68] - torch.from_numpy(landmarks), dim=-1
    ).median(dim=1).values.numpy()
    detector_score = np.median(detector_confidence, axis=1)
    rigid_visibility = visibility[:, 68:]
    visible_ratio = (rigid_visibility > 0.5).float().mean(dim=1).numpy()
    median = float(np.median(landmark_error))
    mad = float(np.median(np.abs(landmark_error - median)))
    reprojection_limit = min(
        median + 2.5 * max(mad, 1e-6),
        0.03 * math.sqrt(width * width + height * height),
    )
    valid = (
        (visible_ratio >= config.min_visible_ratio)
        & (detector_score >= config.min_detector_confidence)
        & (landmark_error <= reprojection_limit)
    )
    template_frame_count = max(
        1, int(len(valid) * config.template_train_fraction)
    )
    valid[template_frame_count:] = False
    yaw = _yaw_degrees(rotation)
    yaw_bins = np.round(yaw / config.yaw_bin_degrees).astype(np.int32)
    quality = detector_score * visible_ratio / np.maximum(landmark_error, 1.0)
    selected = []
    for bin_id in np.unique(yaw_bins[valid]):
        candidates = np.flatnonzero(valid & (yaw_bins == bin_id))
        ranked = candidates[np.argsort(quality[candidates])[::-1]]
        selected.extend(ranked[:config.max_frames_per_yaw_bin].tolist())
    selected = sorted(selected, key=lambda index: quality[index], reverse=True)
    selected = sorted(selected[:config.max_template_frames])
    if len(selected) < config.min_template_frames:
        raise RuntimeError(
            "Canonical template has only %d reliable frames; at least %d are required"
            % (len(selected), config.min_template_frames)
        )
    return selected, {
        "landmark_error": landmark_error,
        "detector_score": detector_score,
        "visible_ratio": visible_ratio,
        "yaw": yaw,
        "yaw_bins": yaw_bins,
        "reprojection_limit": reprojection_limit,
        "quality": quality,
        "template_frame_count": template_frame_count,
    }


def _collect_observations(
    cache, selected, geometry, rotation, translation, visibility, focal,
    center, width, height, detector_scores, device,
):
    observations = []
    weights = []
    rigid_geometry = geometry[:, 68:].to(device)
    for frame_id in selected:
        uvw, head_mask = cache.load(frame_id, device)
        points = _project(
            rigid_geometry[frame_id:frame_id + 1],
            rotation[frame_id:frame_id + 1].to(device),
            translation[frame_id:frame_id + 1].to(device),
            focal, center,
        )[0]
        sampled = _sample_map(uvw, points, width, height)
        sampled_mask = _sample_map(head_mask, points, width, height)[:, 0]
        weight = visibility[frame_id, 68:].to(device)
        weight = weight * sampled_mask * _in_bounds(points, width, height).float()
        weight = weight * float(detector_scores[frame_id])
        observations.append(sampled.cpu().numpy())
        weights.append(weight.cpu().numpy())
    return np.stack(observations), np.stack(weights)


def _huber_location(values, weights, iterations=6):
    weight_sum = np.maximum(weights.sum(axis=0), 1e-8)
    center = (values * weights[..., None]).sum(axis=0) / weight_sum[:, None]
    for _ in range(iterations):
        residual = np.linalg.norm(values - center[None], axis=-1)
        valid_residual = residual[weights > 0]
        scale = np.median(valid_residual) if valid_residual.size else 1.0
        scale = max(float(scale), 1e-4)
        robust = np.minimum(1.0, (1.5 * scale) / np.maximum(residual, 1e-8))
        combined = weights * robust
        denom = np.maximum(combined.sum(axis=0), 1e-8)
        center = (values * combined[..., None]).sum(axis=0) / denom[:, None]
    return center.astype(np.float32)


def _template_reliability(values, weights, target, yaw_bins):
    residual = np.linalg.norm(values - target[None], axis=-1)
    median_residual = np.ones(weights.shape[1], dtype=np.float32)
    for vertex in range(weights.shape[1]):
        valid = weights[:, vertex] > 0
        if valid.any():
            median_residual[vertex] = np.median(residual[valid, vertex])
    scale = max(float(np.nanmedian(median_residual)), 1e-4)
    repeatability = np.exp(-median_residual / (2.0 * scale))
    coverage = (weights > 0).sum(axis=0).astype(np.float32) / max(weights.shape[0], 1)
    bin_coverage = np.zeros(weights.shape[1], dtype=np.float32)
    unique_bins = np.unique(yaw_bins)
    for vertex in range(weights.shape[1]):
        present = set(yaw_bins[weights[:, vertex] > 0].tolist())
        bin_coverage[vertex] = len(present) / max(len(unique_bins), 1)
    return np.clip(repeatability * np.sqrt(coverage * bin_coverage), 0.0, 1.0)


def _build_template(mode, values, weights, geometry, expressions, yaw_bins, config):
    reliability_values = values
    if mode == "universal":
        frame_weight = weights.sum(axis=1)
        reference = int(np.argmax(frame_weight))
        source = geometry[reference, 68:].cpu().numpy()
        source = np.concatenate((source, np.ones((source.shape[0], 1))), axis=1)
        valid = weights[reference] > 0
        weighted_source = source[valid] * np.sqrt(weights[reference, valid, None])
        weighted_target = values[reference, valid] * np.sqrt(weights[reference, valid, None])
        affine = np.linalg.lstsq(weighted_source, weighted_target, rcond=None)[0]
        target = np.matmul(source, affine).astype(np.float32)
    elif mode == "identity":
        target = _huber_location(values, weights)
    else:
        expression = expressions.astype(np.float64)
        expression -= np.median(expression, axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(expression, full_matrices=False)
        components = min(config.expression_pca_components, vt.shape[0])
        z = np.matmul(expression, vt[:components].T)
        design = np.concatenate((np.ones((len(z), 1)), z), axis=1)
        target = np.zeros((values.shape[1], 3), dtype=np.float32)
        reliability_values = values.copy()
        identity = np.eye(design.shape[1]) * config.expression_ridge
        identity[0, 0] = 0.0
        for vertex in range(values.shape[1]):
            w = weights[:, vertex].astype(np.float64)
            if np.count_nonzero(w) < max(3, design.shape[1]):
                target[vertex] = _huber_location(
                    values[:, vertex:vertex + 1], weights[:, vertex:vertex + 1]
                )[0]
                continue
            normal = np.matmul(design.T, design * w[:, None]) + identity
            rhs = np.matmul(design.T, values[:, vertex].astype(np.float64) * w[:, None])
            try:
                coefficients = np.linalg.solve(normal, rhs)
            except np.linalg.LinAlgError:
                coefficients = np.linalg.lstsq(normal, rhs, rcond=None)[0]
            target[vertex] = coefficients[0]
            reliability_values[:, vertex] -= np.matmul(
                design[:, 1:], coefficients[1:]
            ).astype(np.float32)
    reliability = _template_reliability(
        reliability_values, weights, target, yaw_bins
    )
    return target, reliability.astype(np.float32)


def _alignment_loss(feature, mask, points, target, weights, width, height):
    sampled = _sample_map(feature, points, width, height)
    sampled_mask = _sample_map(mask, points, width, height)[..., 0].detach()
    active = weights * sampled_mask * _in_bounds(points, width, height).float()
    residual = torch.sqrt((sampled - target).square().sum(dim=-1) + 1e-6)
    if residual.ndim == 1:
        loss = (residual * active).sum() / active.sum().clamp_min(1e-8)
    else:
        loss = (residual * active).sum(dim=1) / active.sum(dim=1).clamp_min(1e-8)
    return loss, active


def _smoothstep(value, low, high):
    """Map a reliability measurement continuously to [0, 1]."""
    if high <= low:
        raise ValueError("smoothstep requires high > low")
    normalized = ((value - low) / (high - low)).clamp(0.0, 1.0)
    return normalized.square() * (3.0 - 2.0 * normalized)


def _spatial_extent(points, active, width, height):
    """Return the geometric mean of weighted x/y spread in image coordinates."""
    weights = active.clamp_min(0.0)
    weight_sum = weights.sum(dim=1).clamp_min(1e-8)
    mean = (points * weights[..., None]).sum(dim=1) / weight_sum[:, None]
    variance = (
        (points - mean[:, None]).square() * weights[..., None]
    ).sum(dim=1) / weight_sum[:, None]
    std = variance.clamp_min(0.0).sqrt()
    normalized_x = std[:, 0] / max(float(width), 1.0)
    normalized_y = std[:, 1] / max(float(height), 1.0)
    return (normalized_x * normalized_y).clamp_min(0.0).sqrt()


def _temporal_refine_deltas(deltas, confidence, max_rotation, max_translation, config):
    """Refine the complete correction sequence without batch-boundary gaps."""
    if len(deltas) < 2 or config.temporal_optimization_steps == 0:
        return deltas.detach()
    scale = deltas.new_tensor([
        max_rotation, max_rotation, max_rotation,
        float(max_translation), float(max_translation), float(max_translation),
    ]).clamp_min(1e-8)
    target = (deltas / scale).detach()
    refined = torch.nn.Parameter(target.clone())
    confidence = confidence.detach().clamp(0.0, 1.0)
    data_weight = 0.25 + 0.75 * confidence
    optimizer = torch.optim.Adam([refined], lr=config.temporal_learning_rate)
    for _ in range(config.temporal_optimization_steps):
        data_loss = (
            (refined - target).square() * data_weight[:, None]
        ).mean()
        coarse_loss = (
            refined.square() * (1.0 - confidence[:, None])
        ).mean()
        velocity_loss = (refined[1:] - refined[:-1]).square().mean()
        if len(refined) > 2:
            acceleration = refined[2:] - 2.0 * refined[1:-1] + refined[:-2]
            acceleration_loss = acceleration.square().mean()
        else:
            acceleration_loss = refined.new_zeros(())
        loss = (
            data_loss
            + config.lambda_temporal_coarse * coarse_loss
            + config.lambda_temporal_velocity * velocity_loss
            + config.lambda_temporal_acceleration * acceleration_loss
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return (refined.detach() * scale).to(deltas.dtype)


def _optimize_frames(
    cache, geometry, base_rotation, base_translation, visibility, target,
    reliability, focal, center, width, height, config, device,
):
    applied_deltas = []
    confidences = []
    frame_diagnostics = []
    target = torch.from_numpy(target).to(device)
    reliability = torch.from_numpy(reliability).to(device)
    reliable_anchor_count = int((reliability > 0.05).sum().item())
    required_anchors = min(
        config.min_alignment_anchors,
        max(12, int(round(0.15 * reliable_anchor_count))),
    )
    depth = torch.median(torch.abs(base_translation[:, 2])).to(device).clamp_min(1e-6)
    max_rotation = math.radians(config.max_rotation_degrees)
    max_translation = config.max_translation_ratio * depth
    rigid_geometry = geometry[:, 68:].to(device)

    frame_count = len(geometry)
    batch_size = max(1, int(config.alignment_batch_size))
    progress = tqdm(total=frame_count, desc="Canonical batched alignment")
    for start in range(0, frame_count, batch_size):
        end = min(start + batch_size, frame_count)
        loaded = [cache.load(frame_id, device) for frame_id in range(start, end)]
        feature = torch.cat([item[0] for item in loaded], dim=0)
        head_mask = torch.cat([item[1] for item in loaded], dim=0)
        base_r = base_rotation[start:end].to(device)
        base_t = base_translation[start:end].to(device)
        weight = visibility[start:end, 68:].to(device) * reliability[None]
        delta = torch.nn.Parameter(torch.zeros(end - start, 6, device=device))

        def evaluate():
            rotation, translation = compose_increment(delta, base_r, base_t)
            points = _project(
                rigid_geometry[start:end], rotation, translation, focal, center,
            )
            align_loss, active = _alignment_loss(
                feature, head_mask, points, target[None], weight, width, height
            )
            scaled = torch.cat(
                (delta[:, :3] / max_rotation, delta[:, 3:] / max_translation), dim=1
            )
            objective = align_loss + config.lambda_pose_prior * scaled.square().mean(dim=1)
            return align_loss, objective, active, points

        with torch.no_grad():
            initial_loss, _, initial_active, initial_points = evaluate()
        effective = (initial_active > 0.05).sum(dim=1)
        finite_initial = torch.isfinite(initial_loss)
        minimum_optimization_anchors = max(6, required_anchors // 2)
        optimizable = (effective >= minimum_optimization_anchors) & finite_initial
        if bool(optimizable.any().item()):
            for steps, learning_rate in (
                (config.optimization_steps_coarse, config.learning_rate_coarse),
                (config.optimization_steps_fine, config.learning_rate_fine),
            ):
                optimizer = torch.optim.Adam([delta], lr=learning_rate)
                for _ in range(steps):
                    _, objective, _, _ = evaluate()
                    loss = objective[optimizable].mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        rotation_norm = torch.linalg.norm(
                            delta[:, :3], dim=1, keepdim=True
                        ).clamp_min(1e-8)
                        delta[:, :3] *= torch.clamp(
                            max_rotation / rotation_norm, max=1.0
                        )
                        translation_norm = torch.linalg.norm(
                            delta[:, 3:], dim=1, keepdim=True
                        ).clamp_min(1e-8)
                        delta[:, 3:] *= torch.clamp(
                            max_translation / translation_norm, max=1.0
                        )
        with torch.no_grad():
            final_loss, _, final_active, final_points = evaluate()
            finite_final = torch.isfinite(final_loss)
            improvement = torch.where(
                finite_initial & finite_final,
                (initial_loss - final_loss) / initial_loss.clamp_min(1e-8),
                torch.full_like(initial_loss, -1.0),
            )
            rotation_norm = torch.linalg.norm(delta[:, :3], dim=1)
            translation_norm = torch.linalg.norm(delta[:, 3:], dim=1)
            final_effective = (final_active > 0.05).sum(dim=1)
            anchor_confidence = _smoothstep(
                effective.float(), float(minimum_optimization_anchors),
                float(required_anchors) * config.confidence_anchor_high_ratio,
            )
            improvement_confidence = _smoothstep(
                improvement, 0.0, max(config.minimum_improvement * 2.0, 1e-6)
            )
            coverage_retention = (
                final_effective.float() / effective.float().clamp_min(1.0)
            ).clamp(max=1.0)
            coverage_confidence = _smoothstep(
                coverage_retention,
                config.confidence_coverage_retention_low,
                1.0,
            )
            initial_extent = _spatial_extent(
                initial_points, initial_active, width, height
            )
            final_extent = _spatial_extent(final_points, final_active, width, height)
            spatial_extent = torch.minimum(initial_extent, final_extent)
            spatial_confidence = _smoothstep(
                spatial_extent,
                config.confidence_spatial_extent_low,
                config.confidence_spatial_extent_high,
            )
            bound_ratio = torch.maximum(
                rotation_norm / max_rotation,
                translation_norm / max_translation,
            )
            bound_confidence = _smoothstep(
                1.0 - bound_ratio,
                0.0,
                config.confidence_bound_margin_high,
            )
            confidence_product = (
                anchor_confidence
                * improvement_confidence
                * coverage_confidence
                * spatial_confidence
                * bound_confidence
            )
            confidence = confidence_product.clamp_min(0.0).pow(1.0 / 5.0)
            confidence = torch.where(
                optimizable & finite_final, confidence, torch.zeros_like(confidence)
            )
            applied_delta = delta * confidence[:, None]
            # Backward-compatible high-confidence diagnostic; it no longer
            # controls an all-or-nothing pose switch.
            accepted = confidence >= 0.5
        applied_deltas.append(applied_delta.detach())
        confidences.append(confidence.detach())
        for local_index, frame_id in enumerate(range(start, end)):
            frame_diagnostics.append({
                "frame": frame_id,
                "effective_anchors": int(effective[local_index]),
                "final_effective_anchors": int(final_effective[local_index]),
                "required_anchors": required_anchors,
                "minimum_optimization_anchors": minimum_optimization_anchors,
                "initial_loss": float(initial_loss[local_index])
                if finite_initial[local_index] else None,
                "final_loss": float(final_loss[local_index])
                if finite_final[local_index] else None,
                "improvement": float(improvement[local_index]),
                "confidence": float(confidence[local_index]),
                "anchor_confidence": float(anchor_confidence[local_index]),
                "improvement_confidence": float(improvement_confidence[local_index]),
                "coverage_retention": float(coverage_retention[local_index]),
                "coverage_confidence": float(coverage_confidence[local_index]),
                "spatial_extent": float(spatial_extent[local_index]),
                "spatial_confidence": float(spatial_confidence[local_index]),
                "bound_confidence": float(bound_confidence[local_index]),
                "accepted": bool(accepted[local_index]),
                "rotation_correction_deg": float(
                    torch.linalg.norm(applied_delta[local_index, :3])
                    * 180.0 / math.pi
                ),
                "translation_correction_ratio": float(
                    torch.linalg.norm(applied_delta[local_index, 3:]) / depth
                ),
                "raw_rotation_correction_deg": float(
                    rotation_norm[local_index] * 180.0 / math.pi
                ),
                "raw_translation_correction_ratio": float(
                    translation_norm[local_index] / depth
                ),
            })
        progress.update(end - start)
    progress.close()
    pre_temporal_delta = torch.cat(applied_deltas)
    confidence = torch.cat(confidences)
    temporal_delta = _temporal_refine_deltas(
        pre_temporal_delta, confidence, max_rotation, max_translation, config
    )
    rotation, translation = compose_increment(
        temporal_delta,
        base_rotation.to(device),
        base_translation.to(device),
    )
    for index, diagnostic in enumerate(frame_diagnostics):
        before = pre_temporal_delta[index]
        after = temporal_delta[index]
        diagnostic["pre_temporal_rotation_correction_deg"] = float(
            torch.linalg.norm(before[:3]) * 180.0 / math.pi
        )
        diagnostic["pre_temporal_translation_correction_ratio"] = float(
            torch.linalg.norm(before[3:]) / depth
        )
        diagnostic["rotation_correction_deg"] = float(
            torch.linalg.norm(after[:3]) * 180.0 / math.pi
        )
        diagnostic["translation_correction_ratio"] = float(
            torch.linalg.norm(after[3:]) / depth
        )
        diagnostic["temporal_adjustment_norm"] = float(
            torch.linalg.norm((after - before) / torch.cat((
                before.new_full((3,), max_rotation),
                before.new_full((3,), float(max_translation)),
            )))
        )
    return rotation.detach().cpu(), translation.detach().cpu(), frame_diagnostics


def _return_events(yaw, diagnostics):
    events = []
    state = "frontal"
    turn_start = None
    for index, value in enumerate(yaw):
        if state == "frontal" and abs(value) > 45.0:
            turn_start = index
            state = "turned"
        elif state == "turned" and abs(value) < 15.0:
            end = min(index + 10, len(yaw))
            subset = diagnostics[index:end]
            events.append({
                "turn_frame": turn_start,
                "return_frame": index,
                "return_mean_final_loss": float(np.mean([
                    item["final_loss"] for item in subset if item["final_loss"] is not None
                ])) if subset else None,
                "return_acceptance_rate": float(np.mean([
                    item["accepted"] for item in subset
                ])) if subset else None,
                "return_mean_confidence": float(np.mean([
                    item.get("confidence", float(item["accepted"])) for item in subset
                ])) if subset else None,
                "return_landmark_reprojection_px": float(np.mean([
                    item["landmark_reprojection_after_px"] for item in subset
                ])) if subset else None,
                "return_rotation_correction_deg": float(np.mean([
                    item["rotation_correction_deg"] for item in subset
                ])) if subset else None,
            })
            state = "frontal"
    return events


def _pose_diagnostics(rotation, translation, geometry, landmarks, focal, center):
    projected = _project(
        geometry[:, :68].float(), rotation.float(), translation.float(),
        focal, center,
    )
    error = torch.linalg.norm(
        projected - torch.from_numpy(landmarks).float(), dim=-1
    )
    frame_error = error.median(dim=1).values
    twists = relative_twist(rotation.float(), translation.float())
    rotation_velocity = torch.linalg.norm(twists[:, :3], dim=-1) * 180.0 / math.pi
    if len(twists) > 1:
        acceleration = torch.linalg.norm(twists[1:, :3] - twists[:-1, :3], dim=-1)
        acceleration = acceleration * 180.0 / math.pi
    else:
        acceleration = torch.zeros(1)
    return {
        "frame_landmark_reprojection_px": frame_error.numpy(),
        "median_landmark_reprojection_px": float(torch.median(error)),
        "p95_landmark_reprojection_px": float(torch.quantile(error.reshape(-1), 0.95)),
        "median_rotation_velocity_deg": float(torch.median(rotation_velocity)) if len(rotation_velocity) else 0.0,
        "median_rotation_acceleration_deg": float(torch.median(acceleration)),
    }


def run_canonical_alignment(
    data_dir, feature_cache, config=None, overwrite=False, output_path=None
):
    config = config or CanonicalAlignmentConfig()
    config.validate()
    output_path = output_path or os.path.join(data_dir, "track_params_canonical.pt")
    if not os.path.isabs(output_path):
        output_path = os.path.join(data_dir, output_path)
    if os.path.exists(output_path) and not overwrite:
        raise RuntimeError("Canonical output already exists; pass --overwrite: %s" % output_path)
    source_path = os.path.join(data_dir, "track_params.pt")
    confidence_path = os.path.join(data_dir, "landmark_scores.npy")
    image_paths = _numeric_frame_paths(os.path.join(data_dir, "ori_imgs"))
    if not os.path.exists(source_path):
        raise RuntimeError("Missing coarse tracking parameters: %s" % source_path)
    if not os.path.exists(confidence_path):
        raise RuntimeError("Missing landmark confidence file: %s" % confidence_path)
    if not image_paths:
        raise RuntimeError("No numeric image frames found")
    first_image = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)
    height, width = first_image.shape[:2]
    params = torch.load(source_path, map_location="cpu")
    frame_count = int(params["euler"].shape[0])
    if frame_count != len(image_paths):
        raise RuntimeError("Frame/pose count mismatch")
    landmarks = _load_landmarks(image_paths)
    detector_scores = np.load(confidence_path).astype(np.float32)
    detector_confidence = _normalize_detector_confidence(detector_scores)
    positive_scores = detector_scores[detector_scores > 0]
    if positive_scores.size and np.ptp(positive_scores) < 1e-6:
        detector_confidence[detector_scores > 0] = 1.0
    if detector_confidence.shape != (frame_count, 68):
        raise RuntimeError("landmark_scores.npy has an invalid shape")
    cache = DenseMarksCache(feature_cache, image_paths, config.feature_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Canonical visibility and alignment require CUDA")

    visibility_config = LPHSConfig(
        visibility_max_side=config.visibility_max_side,
        visibility_batch_size=config.visibility_batch_size,
    )
    geometry, vertex_ids, visibility, base_rotation, base_translation = (
        _build_geometry_and_visibility(params, height, width, visibility_config, device)
    )
    focal = float(params["focal"].reshape(-1)[0])
    center = torch.tensor((width / 2.0, height / 2.0), device=device)
    selected, selection = _reliable_frames(
        geometry, visibility, base_rotation, base_translation, landmarks,
        detector_confidence, focal, (width / 2.0, height / 2.0), width, height,
        config,
    )
    observations, observation_weights = _collect_observations(
        cache, selected, geometry, base_rotation, base_translation, visibility,
        focal, center, width, height, selection["detector_score"], device,
    )
    target, reliability = _build_template(
        config.template_mode, observations, observation_weights, geometry[selected],
        params["exp"][selected].numpy(), selection["yaw_bins"][selected], config,
    )
    first_rotation, first_translation, first_diagnostics = _optimize_frames(
        cache, geometry, base_rotation, base_translation, visibility, target,
        reliability, focal, center, width, height, config, device,
    )

    confidence = np.asarray([item["confidence"] for item in first_diagnostics])
    second_selected = [index for index in selected if confidence[index] >= 0.5]
    if len(second_selected) < config.min_template_frames:
        second_selected = selected
    refined_params = dict(params)
    refined_params["euler"] = rotation_to_euler(first_rotation)
    refined_params["trans"] = first_translation
    geometry_second, vertex_ids_second, visibility_second, _, _ = (
        _build_geometry_and_visibility(
            refined_params, height, width, visibility_config, device
        )
    )
    observations, observation_weights = _collect_observations(
        cache, second_selected, geometry_second, first_rotation, first_translation,
        visibility_second, focal, center, width, height,
        selection["detector_score"], device,
    )
    target, reliability = _build_template(
        config.template_mode, observations, observation_weights,
        geometry_second[second_selected], params["exp"][second_selected].numpy(),
        selection["yaw_bins"][second_selected], config,
    )
    rotation, translation, frame_diagnostics = _optimize_frames(
        cache, geometry, base_rotation, base_translation, visibility_second,
        target, reliability, focal, center, width, height, config, device,
    )
    before_pose = _pose_diagnostics(
        base_rotation, base_translation, geometry, landmarks, focal,
        (width / 2.0, height / 2.0),
    )
    after_pose = _pose_diagnostics(
        rotation, translation, geometry, landmarks, focal,
        (width / 2.0, height / 2.0),
    )
    for index, item in enumerate(frame_diagnostics):
        item["landmark_reprojection_before_px"] = float(
            before_pose["frame_landmark_reprojection_px"][index]
        )
        item["landmark_reprojection_after_px"] = float(
            after_pose["frame_landmark_reprojection_px"][index]
        )

    result = dict(params)
    result["rot"] = rotation
    result["trans"] = translation
    result["euler"] = rotation_to_euler(rotation)
    result["canonical_alignment"] = {
        "format": "talking_gaussian_canonical_alignment_v1",
        "config": asdict(config),
        "weights_sha256": cache.manifest["weights_sha256"],
        "template_frames": second_selected,
    }
    torch.save(result, output_path)
    output_stem = os.path.splitext(os.path.basename(output_path))[0]
    if output_stem == "track_params_canonical":
        template_name = "canonical_template.npz"
        diagnostics_name = "canonical_diagnostics.json"
    else:
        template_name = output_stem + "_template.npz"
        diagnostics_name = output_stem + "_diagnostics.json"
    np.savez_compressed(
        os.path.join(data_dir, template_name),
        vertex_ids=vertex_ids_second[0, 68:].numpy(),
        target_uvw=target,
        reliability=reliability,
        template_frames=np.asarray(second_selected, dtype=np.int64),
        yaw_bins=selection["yaw_bins"][second_selected],
    )
    diagnostics = {
        "format": "talking_gaussian_canonical_diagnostics_v1",
        "frames": frame_count,
        "rigid_vertices": int(target.shape[0]),
        "template_mode": config.template_mode,
        "template_frames_initial": selected,
        "template_frames_final": second_selected,
        "selection_reprojection_limit": selection["reprojection_limit"],
        "template_source_frame_count": selection["template_frame_count"],
        "accepted_frames": int(sum(item["accepted"] for item in frame_diagnostics)),
        "applied_frames": int(sum(
            item["confidence"] > 0.0 for item in frame_diagnostics
        )),
        "mean_confidence": float(np.mean([
            item["confidence"] for item in frame_diagnostics
        ])),
        "median_confidence": float(np.median([
            item["confidence"] for item in frame_diagnostics
        ])),
        "before": {
            key: value for key, value in before_pose.items()
            if key != "frame_landmark_reprojection_px"
        },
        "after": {
            key: value for key, value in after_pose.items()
            if key != "frame_landmark_reprojection_px"
        },
        "frame_diagnostics": frame_diagnostics,
        "return_events": _return_events(selection["yaw"], frame_diagnostics),
        "config": asdict(config),
    }
    with open(os.path.join(data_dir, diagnostics_name), "w") as file:
        json.dump(diagnostics, file, indent=2)
    return output_path, diagnostics

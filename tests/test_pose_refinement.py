import math

import torch

from scene.pose_refiner import PoseRefinementRuntime, TemporalPoseRefiner
from utils.se3 import compose_increment, se3_exp, se3_log


class _Camera:
    def __init__(self, frame_id, translation=0.0):
        matrix = torch.eye(4)
        matrix[0, 3] = translation
        self.world_view_transform = matrix.transpose(0, 1)
        self.talking_dict = {"img_id": frame_id}


def test_se3_round_trip_and_gradient():
    twist = torch.tensor([0.02, -0.03, 0.01, 0.1, -0.05, 0.02], requires_grad=True)
    rotation, translation = se3_exp(twist)
    recovered = se3_log(rotation, translation)
    assert torch.allclose(recovered, twist, atol=2e-5)
    recovered.square().sum().backward()
    assert torch.isfinite(twist.grad).all()


def test_pose_refiner_is_identity_at_initialization():
    cameras = [_Camera(index, index * 0.01) for index in range(7)]
    runtime = PoseRefinementRuntime(cameras, [], window=5, device="cpu")
    pose = runtime.pose(cameras[3])
    coarse = cameras[3].world_view_transform.transpose(0, 1)
    assert torch.allclose(pose.world_to_camera, coarse, atol=1e-6)
    assert torch.count_nonzero(pose.delta_xi) == 0


def test_photometric_proxy_reaches_shared_temporal_network():
    network = TemporalPoseRefiner(window=5, max_rotation_deg=5.0,
                                  max_translation_ratio=0.02, median_depth=2.0)
    features = torch.randn(1, 5, 18)
    delta, _ = network(features)
    rotation, translation = compose_increment(delta[0], torch.eye(3), torch.zeros(3))
    point = (rotation * torch.tensor([0.2, 0.1, 1.0])).sum(-1) + translation
    point.square().sum().backward()
    assert network.head.weight.grad is not None
    assert torch.isfinite(network.head.weight.grad).all()


def test_rotation_and_translation_are_bounded():
    network = TemporalPoseRefiner(window=3, max_rotation_deg=5.0,
                                  max_translation_ratio=0.02, median_depth=4.0)
    with torch.no_grad():
        network.head.bias.fill_(100.0)
    delta, _ = network(torch.zeros(1, 3, 18))
    assert torch.all(delta[0, :3] <= math.radians(5.0) + 1e-7)
    assert torch.all(delta[0, 3:] <= 0.08 + 1e-7)


def test_checkpoint_round_trip_is_strict():
    cameras = [_Camera(index, index * 0.01) for index in range(5)]
    runtime = PoseRefinementRuntime(cameras, [], window=3, device="cpu")
    payload = runtime.checkpoint()
    original_hash = payload["state_hash"]
    with torch.no_grad():
        runtime.refiner.head.bias.add_(0.5)
    runtime.restore(payload)
    assert runtime.checkpoint()["state_hash"] == original_hash


def test_checkpoint_allows_a_new_test_trajectory():
    train = [_Camera(index, index * 0.01) for index in range(5)]
    original = PoseRefinementRuntime(train, [_Camera(10, 0.1)], window=3, device="cpu")
    replacement = PoseRefinementRuntime(train, [_Camera(20, 0.2), _Camera(21, 0.3)],
                                        window=3, device="cpu")
    replacement.restore(original.checkpoint())
    replacement_camera = _Camera(20, 0.2)
    assert torch.isfinite(replacement.pose(replacement_camera, "test").delta_xi).all()


def test_eval_alias_uses_checkpoint_training_statistics():
    train = [_Camera(index, index * 0.01) for index in range(5)]
    runtime = PoseRefinementRuntime(train, [_Camera(10, 0.1)], window=3, device="cpu")
    payload = runtime.checkpoint()
    validation = [_Camera(20, 0.3), _Camera(21, 0.4)]
    eval_runtime = PoseRefinementRuntime(validation, validation, window=3, device="cpu")
    eval_runtime.restore(payload, strict_hash=False)
    assert torch.allclose(eval_runtime.reference, payload["reference"])
    assert eval_runtime.median_depth == payload["median_depth"]


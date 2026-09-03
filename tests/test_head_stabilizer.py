import math
import unittest

import numpy as np
import torch

from data_utils.face_tracking.util import euler2rot
from data_utils.head_stabilizer.flow import forward_backward_track
from data_utils.head_stabilizer.se3 import (
    adaptive_temporal_weight,
    rotation_to_euler,
    se3_exp,
    se3_log,
)


class SE3Test(unittest.TestCase):
    def test_exp_log_round_trip_and_gradient(self):
        twist = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1e-5, -2e-5, 3e-5, 0.01, -0.02, 0.03],
                [0.1, -0.05, 0.02, 0.2, 0.1, -0.1],
            ],
            requires_grad=True,
        )
        rotation, translation = se3_exp(twist)
        recovered = se3_log(rotation, translation)
        self.assertLess(float((twist - recovered).abs().max()), 1e-5)
        loss = rotation.square().mean() + translation.square().mean()
        loss.backward()
        self.assertTrue(bool(torch.isfinite(twist.grad).all()))

    def test_repository_euler_convention_round_trip(self):
        euler = torch.tensor([[0.1, -0.2, 0.05], [-0.3, 0.4, -0.1]])
        rotation = euler2rot(euler)
        recovered = rotation_to_euler(rotation)
        self.assertTrue(torch.allclose(euler, recovered, atol=1e-6))

    def test_adaptive_weight_reduces_for_fast_motion(self):
        twists = torch.zeros(2, 6)
        twists[0, 1] = math.radians(0.5)
        twists[1, 1] = math.radians(6.0)
        weights = adaptive_temporal_weight(twists, torch.tensor(7.0))
        self.assertGreater(float(weights[0]), float(weights[1]))
        self.assertAlmostEqual(float(weights[1]), 0.25, places=4)


class FlowTest(unittest.TestCase):
    def test_forward_backward_consistent_translation(self):
        forward = np.zeros((16, 16, 2), dtype=np.float32)
        backward = np.zeros_like(forward)
        forward[..., 0] = 2.0
        forward[..., 1] = 1.0
        backward[..., 0] = -2.0
        backward[..., 1] = -1.0
        points = np.asarray([[3.0, 4.0], [8.5, 7.5]], dtype=np.float32)
        tracked, confidence = forward_backward_track(forward, backward, points)
        self.assertTrue(np.allclose(tracked, points + [2.0, 1.0]))
        self.assertTrue(np.allclose(confidence, 1.0))

    def test_out_of_bounds_track_is_rejected(self):
        forward = np.zeros((8, 8, 2), dtype=np.float32)
        backward = np.zeros_like(forward)
        forward[..., 0] = 10.0
        points = np.asarray([[4.0, 4.0]], dtype=np.float32)
        _, confidence = forward_backward_track(forward, backward, points)
        self.assertEqual(float(confidence[0]), 0.0)


if __name__ == "__main__":
    unittest.main()

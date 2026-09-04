import unittest
from types import SimpleNamespace

import torch

from scene.motion_net import (
    AudioUpperFaceGeometryModulator,
    GaussianScaleRouter,
    MotionNetwork,
)


class GaussianScaleRouterTest(unittest.TestCase):
    def test_initial_routing_prefers_16(self):
        router = GaussianScaleRouter(feature_dim=36)
        feature = torch.randn(5, 36)
        scaling = torch.rand(5, 3) + 0.01

        weights = router(feature, scaling)
        expected = torch.softmax(torch.tensor([-1.0, 2.0, -1.0]), dim=0)

        self.assertEqual(tuple(weights.shape), (5, 3))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(5)))
        self.assertTrue(torch.allclose(weights, expected.expand_as(weights)))

    def test_scale_is_detached_but_spatial_feature_is_trainable(self):
        torch.manual_seed(3)
        router = GaussianScaleRouter(feature_dim=36)
        torch.nn.init.normal_(router.fc2.weight, std=0.1)
        feature = torch.randn(4, 36, requires_grad=True)
        scaling = (torch.rand(4, 3) + 0.01).requires_grad_()

        router(feature, scaling)[:, 0].sum().backward()

        self.assertIsNotNone(feature.grad)
        self.assertIsNone(scaling.grad)


class MultiScaleModulatorTest(unittest.TestCase):
    def test_pyramid_shapes_and_zero_initialization(self):
        module = AudioUpperFaceGeometryModulator(
            audio_dim=32,
            upper_dim=6,
            plane_dim=12,
            multiscale=True,
        )
        maps, gate = module(torch.randn(1, 32), torch.randn(1, 6))

        self.assertEqual(
            [tuple(value.shape) for value in maps],
            [(1, 3, 2, 12, 8, 8),
             (1, 3, 2, 12, 16, 16),
             (1, 3, 2, 12, 32, 32)],
        )
        self.assertEqual(tuple(gate.shape), (1, 3, 2, 16, 16))
        for value in maps:
            self.assertEqual(int(torch.count_nonzero(value)), 0)

    def test_fixed_modulator_keeps_original_shape(self):
        module = AudioUpperFaceGeometryModulator(
            audio_dim=32,
            upper_dim=6,
            plane_dim=12,
            multiscale=False,
        )
        modulation, gate = module(torch.randn(1, 32), torch.randn(1, 6))
        self.assertEqual(tuple(modulation.shape), (1, 3, 2, 12, 16, 16))
        self.assertEqual(tuple(gate.shape), (1, 3, 2, 16, 16))

    def test_gaussian_level_weighted_sampling(self):
        args = SimpleNamespace(
            audio_extractor='deepspeech',
            geometry_mod_map_res=16,
            geometry_mod_condition_scale=0.1,
            geometry_mod_multiscale=1,
        )
        network = MotionNetwork(args=args)
        coords = tuple(torch.zeros(2, 2) for _ in range(3))
        features = tuple(torch.zeros(2, 12) for _ in range(3))
        maps = []
        for resolution, beta in ((8, 1.0), (16, 2.0), (32, 4.0)):
            value = torch.zeros(1, 3, 2, 12, resolution, resolution)
            value[:, :, 1] = beta
            maps.append(value)
        routing = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.25, 0.75]])

        output = network.apply_geometry_modulation(
            coords, features, tuple(maps), routing
        )

        for plane in output:
            self.assertTrue(torch.allclose(plane[0], torch.ones(12)))
            self.assertTrue(torch.allclose(plane[1], torch.full((12,), 3.5)))

    def test_checkpoint_mode_mismatch_is_rejected(self):
        adaptive_args = SimpleNamespace(
            audio_extractor='deepspeech',
            geometry_mod_map_res=16,
            geometry_mod_condition_scale=0.1,
            geometry_mod_multiscale=1,
        )
        fixed_args = SimpleNamespace(**vars(adaptive_args))
        fixed_args.geometry_mod_multiscale = 0
        adaptive = MotionNetwork(args=adaptive_args)
        fixed = MotionNetwork(args=fixed_args)

        adaptive.validate_multiscale_checkpoint(adaptive.state_dict())
        fixed.validate_multiscale_checkpoint(fixed.state_dict())
        with self.assertRaises(RuntimeError):
            adaptive.validate_multiscale_checkpoint(fixed.state_dict())


if __name__ == '__main__':
    unittest.main()

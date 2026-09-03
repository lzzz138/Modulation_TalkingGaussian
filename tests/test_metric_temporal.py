import unittest

import numpy as np

from metric import VideoMetricsCalculator


def make_landmarks(frame_count):
    landmarks = np.zeros((frame_count, 68, 2), dtype=np.float32)
    landmarks[..., 0] = np.arange(68, dtype=np.float32)[None]
    landmarks[:, 36, 0] = 10.0
    landmarks[:, 45, 0] = 30.0
    return landmarks


class LandmarkJitterTest(unittest.TestCase):
    @staticmethod
    def make_insightface_features(frame_count):
        keypoints = np.zeros((frame_count, 5, 2), dtype=np.float32)
        keypoints[:, 0, 0] = 10.0
        keypoints[:, 1, 0] = 30.0
        return {
            'kps5': keypoints,
            'valid': np.ones(frame_count, dtype=bool),
        }

    def test_kps5_matching_motion_has_zero_jitter(self):
        real = self.make_insightface_features(6)
        generated = self.make_insightface_features(6)
        displacement = np.arange(6, dtype=np.float32)[:, None, None]
        real['kps5'] += displacement
        generated['kps5'] += displacement
        result = VideoMetricsCalculator.calculate_landmark_velocity_distance_from_features(
            generated, real
        )
        self.assertAlmostEqual(result['pixels'], 0.0, places=7)
        self.assertAlmostEqual(result['normalized'], 0.0, places=7)
        self.assertEqual(result['valid_pairs'], 5)

    def test_kps5_motion_difference_is_detected(self):
        real = self.make_insightface_features(5)
        generated = self.make_insightface_features(5)
        generated['kps5'][3:, :, 1] += 4.0
        result = VideoMetricsCalculator.calculate_landmark_velocity_distance_from_features(
            generated, real
        )
        self.assertGreater(result['pixels'], 0.0)
        self.assertGreater(result['normalized'], 0.0)

    def test_constant_velocity_has_zero_jitter(self):
        real = make_landmarks(6)
        generated = real.copy()
        generated[..., 0] += np.arange(6, dtype=np.float32)[:, None] * 2.0
        result = VideoMetricsCalculator.calculate_landmark_jitter(
            generated, real, region='head'
        )
        self.assertAlmostEqual(result['gen'], 0.0, places=7)
        self.assertAlmostEqual(result['error'], 0.0, places=7)
        self.assertEqual(result['valid_triplets'], 4)

    def test_fan_landmark_velocity_distance(self):
        real = make_landmarks(6)
        generated = real.copy()
        generated[3:, :, 1] += 4.0
        result = VideoMetricsCalculator.calculate_landmark_velocity_distance(
            generated, real, region='head'
        )
        self.assertGreater(result['pixels'], 0.0)
        self.assertGreater(result['normalized'], 0.0)
        self.assertEqual(result['valid_pairs'], 5)

    def test_single_frame_jump_is_detected(self):
        real = make_landmarks(7)
        generated = real.copy()
        generated[3, :, 1] += 5.0
        result = VideoMetricsCalculator.calculate_landmark_jitter(
            generated, real, region='all'
        )
        self.assertGreater(result['gen'], 0.0)
        self.assertGreater(result['error'], 0.0)
        self.assertAlmostEqual(result['real'], 0.0, places=7)

    def test_matching_real_acceleration_has_zero_error(self):
        real = make_landmarks(6)
        displacement = np.asarray([0.0, 1.0, 4.0, 9.0, 16.0, 25.0])
        real[..., 1] += displacement[:, None]
        generated = real.copy()
        result = VideoMetricsCalculator.calculate_landmark_jitter(
            generated, real, region='head'
        )
        self.assertGreater(result['gen'], 0.0)
        self.assertAlmostEqual(result['error'], 0.0, places=7)


class WarpHelperTest(unittest.TestCase):
    def test_temporal_flow_endpoint_error(self):
        real_flow = np.zeros((4, 4, 2), dtype=np.float32)
        generated_flow = np.zeros_like(real_flow)
        generated_flow[..., 0] = 3.0
        generated_flow[..., 1] = 4.0
        valid = np.ones((4, 4), dtype=bool)
        error = VideoMetricsCalculator._temporal_flow_error(
            generated_flow, real_flow, valid
        )
        self.assertAlmostEqual(error, 5.0, places=6)

    def test_zero_flow_preserves_image(self):
        image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        flow = np.zeros((8, 8, 2), dtype=np.float32)
        warped, valid = VideoMetricsCalculator._warp_previous_frame(image, flow)
        self.assertTrue(np.array_equal(warped, image))
        self.assertTrue(valid.all())

    def test_consistency_mask_rejects_inconsistent_flow(self):
        forward = np.zeros((8, 8, 2), dtype=np.float32)
        backward = np.zeros_like(forward)
        forward[..., 0] = 3.0
        mask = VideoMetricsCalculator._forward_backward_mask(
            forward, backward, threshold=1.5
        )
        self.assertFalse(mask.any())


if __name__ == '__main__':
    unittest.main()

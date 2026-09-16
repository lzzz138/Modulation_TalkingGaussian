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


class LandmarkDistanceAndStabilityTest(unittest.TestCase):
    def test_lmd_matches_legacy_metrics_definition(self):
        real = make_landmarks(4)
        generated = real.copy()
        generated[:, 48, 1] += np.asarray([0.0, 2.0, 4.0, 6.0])
        gen_mouth = generated[:, 48:68]
        real_mouth = real[:, 48:68]
        gen_mouth = gen_mouth - gen_mouth.mean(axis=1, keepdims=True)
        real_mouth = real_mouth - real_mouth.mean(axis=1, keepdims=True)
        expected = np.sqrt(
            ((gen_mouth - real_mouth) ** 2).sum(axis=-1)
        ).mean(axis=-1).mean()

        result = VideoMetricsCalculator.calculate_lmd_from_landmarks(
            generated, real, lmd_region='mouth'
        )
        extended = VideoMetricsCalculator.calculate_lmd_auc_from_landmarks(
            generated, real, lmd_region='mouth'
        )
        self.assertAlmostEqual(result, expected, places=7)
        self.assertAlmostEqual(extended['lmd'], expected, places=7)

    def test_identical_landmarks_have_zero_lmd(self):
        real = make_landmarks(8)
        result = VideoMetricsCalculator.calculate_lmd_auc_from_landmarks(
            real.copy(), real, lmd_region='mouth'
        )
        self.assertAlmostEqual(result['lmd'], 0.0, places=7)
        self.assertAlmostEqual(result['lmd_normalized'], 0.0, places=7)
        self.assertAlmostEqual(result['auc'], 1.0, places=6)
        self.assertEqual(result['valid_frames'], 8)

    def test_lmd_ignores_region_translation_but_detects_deformation(self):
        real = make_landmarks(5)
        translated = real.copy()
        translated[:, 48:68] += np.asarray([8.0, 3.0], dtype=np.float32)
        translated_result = VideoMetricsCalculator.calculate_lmd_auc_from_landmarks(
            translated, real, lmd_region='mouth'
        )
        self.assertAlmostEqual(translated_result['lmd'], 0.0, places=6)

        deformed = real.copy()
        deformed[:, 48, 1] += 4.0
        deformed_result = VideoMetricsCalculator.calculate_lmd_auc_from_landmarks(
            deformed, real, lmd_region='mouth'
        )
        self.assertGreater(deformed_result['lmd'], 0.0)
        self.assertGreater(deformed_result['lmd_normalized'], 0.0)

    def test_matching_trajectory_has_zero_stability_error(self):
        real = make_landmarks(16)
        real[:, 27:36, 1] += np.sin(np.arange(16, dtype=np.float32))[:, None]
        result = VideoMetricsCalculator.calculate_gaussianheadtalk_stability(
            real.copy(), real
        )
        self.assertAlmostEqual(result['score'], 0.0, places=7)
        self.assertEqual(result['valid_frames'], 16)

    def test_high_frequency_nose_wobble_increases_stability_error(self):
        real = make_landmarks(32)
        generated = real.copy()
        wobble = ((np.arange(32) % 2) * 2 - 1).astype(np.float32) * 2.0
        generated[:, 27:36, 1] += wobble[:, None]
        result = VideoMetricsCalculator.calculate_gaussianheadtalk_stability(
            generated, real
        )
        self.assertGreater(result['score'], 0.0)
        self.assertGreater(result['mean_motion_difference'], 0.0)
        self.assertGreater(result['high_frequency_power'], 0.0)

    def test_stability_uses_longest_contiguous_valid_run(self):
        real = make_landmarks(10)
        generated = real.copy()
        generated[3] = np.nan
        result = VideoMetricsCalculator.calculate_gaussianheadtalk_stability(
            generated, real
        )
        self.assertEqual(result['valid_frames'], 6)

    def test_invalid_stability_frequency_ratio_is_rejected(self):
        real = make_landmarks(4)
        with self.assertRaises(ValueError):
            VideoMetricsCalculator.calculate_gaussianheadtalk_stability(
                real, real, high_frequency_ratio=0.0
            )


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

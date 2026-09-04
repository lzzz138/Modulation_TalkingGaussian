import argparse
import math
import os
import subprocess
import tempfile

import cv2
import lpips
import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from scipy.io import wavfile
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision import models

class VideoMetricsCalculator:
    def __init__(self, device='cuda', batch_size=32, det_size=640):
        self.device = torch.device(device if device == 'cuda' and torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.det_size = (det_size, det_size)

        self.loss_fn_alex = lpips.LPIPS(net='alex').eval().to(self.device)
        self.inception_model = None

        self.face_analyzer = None
        self.fan_predictor = None
        self.fan_unavailable = False

    def _get_inception_model(self):
        model = models.inception_v3(pretrained=True, transform_input=False)
        model.fc = torch.nn.Identity()
        return model

    def load_video_frames(self, video_path, max_frames=None):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if max_frames and len(frames) >= max_frames:
                break
        cap.release()
        return frames, float(fps)

    def _frames_to_tensor(self, frames, size=None, value_range='lpips'):
        arr = []
        for frame in frames:
            if size is not None:
                frame = cv2.resize(frame, size)
            arr.append(frame)
        tensor = torch.from_numpy(np.stack(arr)).permute(0, 3, 1, 2).float().to(self.device)
        if value_range == 'lpips':
            return tensor / 127.5 - 1.0
        if value_range == '01':
            return tensor / 255.0
        return tensor

    def _iter_batches(self, frames):
        for start in range(0, len(frames), self.batch_size):
            yield frames[start:start + self.batch_size]

    def calculate_psnr_ssim(self, frames_gen, frames_real):
        psnr_vals = []
        ssim_vals = []
        for gen, real in zip(frames_gen, frames_real):
            psnr_vals.append(psnr(real, gen))
            ssim_vals.append(ssim(real, gen, channel_axis=2, data_range=255))
        return float(np.mean(psnr_vals)), float(np.mean(ssim_vals))

    def calculate_lpips(self, frames_gen, frames_real):
        vals = []
        with torch.no_grad():
            for gen_batch, real_batch in zip(self._iter_batches(frames_gen), self._iter_batches(frames_real)):
                gen = self._frames_to_tensor(gen_batch, value_range='lpips')
                real = self._frames_to_tensor(real_batch, value_range='lpips')
                vals.extend(self.loss_fn_alex(real, gen).view(-1).detach().cpu().numpy().tolist())
        return float(np.mean(vals)) if vals else 0.0

    def calculate_tlpips(self, frames_gen, frames_real):
        min_len = min(len(frames_gen), len(frames_real))
        if min_len < 2:
            return 0.0
        real_delta = [
            ((frames_real[i].astype(np.float32) - frames_real[i - 1].astype(np.float32)) / 255.0)
            for i in range(1, min_len)
        ]
        gen_delta = [
            ((frames_gen[i].astype(np.float32) - frames_gen[i - 1].astype(np.float32)) / 255.0)
            for i in range(1, min_len)
        ]
        vals = []
        with torch.no_grad():
            for gen_batch, real_batch in zip(self._iter_batches(gen_delta), self._iter_batches(real_delta)):
                gen = torch.from_numpy(np.stack(gen_batch)).permute(0, 3, 1, 2).float().to(self.device)
                real = torch.from_numpy(np.stack(real_batch)).permute(0, 3, 1, 2).float().to(self.device)
                vals.extend(self.loss_fn_alex(real, gen).view(-1).detach().cpu().numpy().tolist())
        return float(np.mean(vals)) if vals else 0.0

    @staticmethod
    def _warp_previous_frame(previous, backward_flow):
        height, width = previous.shape[:2]
        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        map_x = grid_x + backward_flow[..., 0]
        map_y = grid_y + backward_flow[..., 1]
        warped = cv2.remap(
            previous,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        valid = (
            (map_x >= 0.0)
            & (map_x <= width - 1)
            & (map_y >= 0.0)
            & (map_y <= height - 1)
        )
        return warped, valid

    @staticmethod
    def _forward_backward_mask(forward_flow, backward_flow, threshold):
        height, width = backward_flow.shape[:2]
        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        map_x = grid_x + backward_flow[..., 0]
        map_y = grid_y + backward_flow[..., 1]
        sampled_forward = cv2.remap(
            forward_flow,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        valid = (
            (map_x >= 0.0)
            & (map_x <= width - 1)
            & (map_y >= 0.0)
            & (map_y <= height - 1)
        )
        consistency = np.linalg.norm(
            backward_flow + sampled_forward, axis=-1
        )
        return valid & (consistency <= threshold)

    @staticmethod
    def _compute_farneback_flow(frame_previous, frame_current, max_side=512):
        previous_gray = cv2.cvtColor(frame_previous, cv2.COLOR_RGB2GRAY)
        current_gray = cv2.cvtColor(frame_current, cv2.COLOR_RGB2GRAY)
        original_height, original_width = previous_gray.shape
        scale = min(1.0, float(max_side) / max(original_height, original_width))
        if scale < 1.0:
            resized_size = (
                max(8, int(round(original_width * scale))),
                max(8, int(round(original_height * scale))),
            )
            previous_gray = cv2.resize(previous_gray, resized_size)
            current_gray = cv2.resize(current_gray, resized_size)
        flow = cv2.calcOpticalFlowFarneback(
            previous_gray,
            current_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        if scale < 1.0:
            flow = cv2.resize(flow, (original_width, original_height))
            flow[..., 0] *= float(original_width) / previous_gray.shape[1]
            flow[..., 1] *= float(original_height) / previous_gray.shape[0]
        return flow.astype(np.float32)

    @staticmethod
    def _face_region_mask(landmarks, height, width, expansion=0.25):
        if landmarks is None or not np.isfinite(landmarks).all():
            return np.ones((height, width), dtype=bool)
        x_min, y_min = landmarks.min(axis=0)
        x_max, y_max = landmarks.max(axis=0)
        face_width = max(float(x_max - x_min), 1.0)
        face_height = max(float(y_max - y_min), 1.0)
        x_min = max(0, int(np.floor(x_min - expansion * face_width)))
        x_max = min(width, int(np.ceil(x_max + expansion * face_width)) + 1)
        y_min = max(0, int(np.floor(y_min - expansion * face_height)))
        y_max = min(height, int(np.ceil(y_max + expansion * face_height)) + 1)
        mask = np.zeros((height, width), dtype=bool)
        mask[y_min:y_max, x_min:x_max] = True
        return mask

    @staticmethod
    def _temporal_flow_error(generated_flow, real_flow, valid_mask):
        endpoint_error = np.linalg.norm(generated_flow - real_flow, axis=-1)
        if not valid_mask.any():
            return 0.0
        return float(endpoint_error[valid_mask].mean())

    def calculate_flow_warp_error(
        self,
        frames_gen,
        frames_real,
        real_landmarks=None,
        backend='farneback',
        max_side=512,
        fb_threshold=1.5,
        region='face',
    ):
        min_len = min(len(frames_gen), len(frames_real))
        if min_len < 2:
            return {
                'gen': 0.0,
                'real': 0.0,
                'excess': 0.0,
                'tof_px': 0.0,
                'tof_normalized': 0.0,
                'valid_pairs': 0,
                'backend': backend,
            }
        estimator = None
        actual_backend = backend
        if backend != 'farneback':
            from data_utils.head_stabilizer.flow import FlowEstimator

            estimator = FlowEstimator(
                backend=backend, max_side=max_side, device=str(self.device)
            )
        gen_errors = []
        real_errors = []
        tof_errors = []
        tof_normalized_errors = []
        for frame_id in range(1, min_len):
            previous_real = frames_real[frame_id - 1]
            current_real = frames_real[frame_id]
            previous_gen = frames_gen[frame_id - 1]
            current_gen = frames_gen[frame_id]
            if previous_real.shape != current_real.shape or previous_gen.shape != current_gen.shape:
                raise RuntimeError('Flow Warp Error requires constant frame resolution.')
            if current_gen.shape != current_real.shape:
                raise RuntimeError('Generated and real videos must have the same resolution.')

            if estimator is None:
                forward = self._compute_farneback_flow(
                    previous_real, current_real, max_side=max_side
                )
                # cv2.remap needs a current-to-previous flow field.
                backward = self._compute_farneback_flow(
                    current_real, previous_real, max_side=max_side
                )
                generated_forward = self._compute_farneback_flow(
                    previous_gen, current_gen, max_side=max_side
                )
            else:
                forward, backward = estimator.bidirectional(
                    previous_real, current_real
                )
                generated_forward = estimator.flow(previous_gen, current_gen)
                actual_backend = estimator.backend
            warped_gen, warp_valid = self._warp_previous_frame(previous_gen, backward)
            warped_real, _ = self._warp_previous_frame(previous_real, backward)
            valid = warp_valid & self._forward_backward_mask(
                forward, backward, fb_threshold
            )
            if region == 'face':
                landmarks = None
                if real_landmarks is not None and frame_id < len(real_landmarks):
                    landmarks = real_landmarks[frame_id]
                valid &= self._face_region_mask(
                    landmarks, current_real.shape[0], current_real.shape[1]
                )
            if not valid.any():
                continue
            gen_error = np.abs(
                current_gen.astype(np.float32) - warped_gen.astype(np.float32)
            ).mean(axis=-1) / 255.0
            real_error = np.abs(
                current_real.astype(np.float32) - warped_real.astype(np.float32)
            ).mean(axis=-1) / 255.0
            frame_diagonal = math.sqrt(
                current_real.shape[0] ** 2 + current_real.shape[1] ** 2
            )
            tof_px = self._temporal_flow_error(
                generated_forward, forward, valid
            )
            gen_errors.append(float(gen_error[valid].mean()))
            real_errors.append(float(real_error[valid].mean()))
            tof_errors.append(tof_px)
            tof_normalized_errors.append(tof_px / frame_diagonal)
            if frame_id % 100 == 0:
                print(f'[INFO] Flow warp: {frame_id}/{min_len - 1}')

        if not gen_errors:
            return {
                'gen': 0.0,
                'real': 0.0,
                'excess': 0.0,
                'tof_px': 0.0,
                'tof_normalized': 0.0,
                'valid_pairs': 0,
                'backend': actual_backend,
            }
        gen_mean = float(np.mean(gen_errors))
        real_mean = float(np.mean(real_errors))
        return {
            'gen': gen_mean,
            'real': real_mean,
            'excess': max(0.0, gen_mean - real_mean),
            'tof_px': float(np.mean(tof_errors)),
            'tof_normalized': float(np.mean(tof_normalized_errors)),
            'valid_pairs': len(gen_errors),
            'backend': actual_backend,
        }

    def _get_inception_features(self, frames):
        if self.inception_model is None:
            self.inception_model = self._get_inception_model().eval().to(self.device)
        features = []
        with torch.no_grad():
            for batch in self._iter_batches(frames):
                inp = self._frames_to_tensor(batch, size=(299, 299), value_range='01')
                feat = self.inception_model(inp)
                features.append(feat.detach().cpu().numpy())
        return np.concatenate(features, axis=0)

    def calculate_fid(self, frames_gen, frames_real):
        act1 = self._get_inception_features(frames_real)
        act2 = self._get_inception_features(frames_gen)

        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

        ssdiff = np.sum((mu1 - mu2) ** 2)
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean))

    def _init_face_analyzer(self):
        if self.face_analyzer is not None:
            return True
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            print('[WARN] insightface is not installed; skip CSIM.')
            return False

        providers = ['CPUExecutionProvider']
        if self.device.type == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.face_analyzer = FaceAnalysis(providers=providers)
        self.face_analyzer.prepare(ctx_id=0 if self.device.type == 'cuda' else -1, det_size=self.det_size)
        return True

    @staticmethod
    def _largest_face(faces):
        if len(faces) == 0:
            return None
        def area(face):
            box = face.bbox if hasattr(face, 'bbox') else face['bbox']
            return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
        return max(faces, key=area)

    def extract_insightface_features(self, frames, name='video'):
        if not self._init_face_analyzer():
            return None
        embeddings = []
        kps = []
        valid = []
        for idx, frame in enumerate(frames):
            faces = self.face_analyzer.get(frame)
            face = self._largest_face(faces)
            if face is None:
                embeddings.append(np.full((512,), np.nan, dtype=np.float32))
                kps.append(np.full((5, 2), np.nan, dtype=np.float32))
                valid.append(False)
                continue
            emb = face.embedding if hasattr(face, 'embedding') else face['embedding']
            lm = face.kps if hasattr(face, 'kps') else face['kps']
            embeddings.append(np.asarray(emb, dtype=np.float32))
            kps.append(np.asarray(lm, dtype=np.float32))
            valid.append(True)
            if (idx + 1) % 200 == 0:
                print(f'[INFO] InsightFace {name}: {idx + 1}/{len(frames)}')
        return {
            'embedding': np.stack(embeddings, axis=0),
            'kps5': np.stack(kps, axis=0),
            'valid': np.asarray(valid, dtype=bool),
        }

    def calculate_csim_from_features(self, gen_feat, real_feat):
        if gen_feat is None or real_feat is None:
            return None
        valid = gen_feat['valid'] & real_feat['valid']
        if not valid.any():
            return 0.0
        gen = torch.from_numpy(gen_feat['embedding'][valid]).float().to(self.device)
        real = torch.from_numpy(real_feat['embedding'][valid]).float().to(self.device)
        gen = F.normalize(gen, dim=-1)
        real = F.normalize(real, dim=-1)
        return float((gen * real).sum(dim=-1).mean().detach().cpu().item())

    def calculate_csim_reference_from_features(self, gen_feat, real_feat):
        if gen_feat is None or real_feat is None:
            return None
        real_valid_idx = np.where(real_feat['valid'])[0]
        gen_valid = gen_feat['valid']
        if len(real_valid_idx) == 0 or not gen_valid.any():
            return 0.0
        ref = torch.from_numpy(real_feat['embedding'][real_valid_idx[0]:real_valid_idx[0] + 1]).float().to(self.device)
        gen = torch.from_numpy(gen_feat['embedding'][gen_valid]).float().to(self.device)
        ref = F.normalize(ref, dim=-1)
        gen = F.normalize(gen, dim=-1)
        return float((gen * ref).sum(dim=-1).mean().detach().cpu().item())

    @staticmethod
    def calculate_landmark_velocity_distance_from_features(gen_feat, real_feat):
        """Compare generated and real inter-frame velocity of InsightFace kps5."""
        if gen_feat is None or real_feat is None:
            return None
        min_len = min(len(gen_feat['kps5']), len(real_feat['kps5']))
        if min_len < 2:
            return {
                'pixels': 0.0,
                'normalized': 0.0,
                'gen_motion': 0.0,
                'real_motion': 0.0,
                'valid_pairs': 0,
            }
        gen_kps = np.asarray(gen_feat['kps5'][:min_len], dtype=np.float32)
        real_kps = np.asarray(real_feat['kps5'][:min_len], dtype=np.float32)
        frame_valid = (
            gen_feat['valid'][:min_len]
            & real_feat['valid'][:min_len]
            & np.isfinite(gen_kps).all(axis=(1, 2))
            & np.isfinite(real_kps).all(axis=(1, 2))
        )
        pair_valid = frame_valid[:-1] & frame_valid[1:]
        if not pair_valid.any():
            return {
                'pixels': 0.0,
                'normalized': 0.0,
                'gen_motion': 0.0,
                'real_motion': 0.0,
                'valid_pairs': 0,
            }

        gen_motion = gen_kps[1:] - gen_kps[:-1]
        real_motion = real_kps[1:] - real_kps[:-1]
        motion_error = np.linalg.norm(gen_motion - real_motion, axis=-1).mean(axis=-1)
        gen_magnitude = np.linalg.norm(gen_motion, axis=-1).mean(axis=-1)
        real_magnitude = np.linalg.norm(real_motion, axis=-1).mean(axis=-1)
        # InsightFace kps5 stores left/right eye at indices 0 and 1.
        interocular = np.linalg.norm(real_kps[1:, 0] - real_kps[1:, 1], axis=-1)
        interocular = np.maximum(interocular, 1e-6)
        return {
            'pixels': float(motion_error[pair_valid].mean()),
            'normalized': float((motion_error / interocular)[pair_valid].mean()),
            'gen_motion': float(gen_magnitude[pair_valid].mean()),
            'real_motion': float(real_magnitude[pair_valid].mean()),
            'valid_pairs': int(pair_valid.sum()),
        }

    @staticmethod
    def calculate_landmark_velocity_distance(gen_lms, real_lms, region='head'):
        """Landmark velocity distance using already extracted FAN landmarks."""
        if gen_lms is None or real_lms is None:
            return None
        min_len = min(len(gen_lms), len(real_lms))
        if min_len < 2:
            return {
                'pixels': 0.0,
                'normalized': 0.0,
                'gen_motion': 0.0,
                'real_motion': 0.0,
                'valid_pairs': 0,
            }
        gen = np.asarray(gen_lms[:min_len], dtype=np.float32)
        real = np.asarray(real_lms[:min_len], dtype=np.float32)
        if region == 'head':
            indices = np.concatenate((np.arange(17), np.arange(27, 36)))
        elif region == 'mouth':
            indices = np.arange(48, 68)
        elif region == 'all':
            indices = np.arange(68)
        else:
            raise ValueError('Unknown LVD region: %s' % region)

        frame_valid = (
            np.isfinite(gen).all(axis=(1, 2))
            & np.isfinite(real).all(axis=(1, 2))
        )
        pair_valid = frame_valid[:-1] & frame_valid[1:]
        if not pair_valid.any():
            return {
                'pixels': 0.0,
                'normalized': 0.0,
                'gen_motion': 0.0,
                'real_motion': 0.0,
                'valid_pairs': 0,
            }

        gen_motion = gen[1:, indices] - gen[:-1, indices]
        real_motion = real[1:, indices] - real[:-1, indices]
        motion_error = np.linalg.norm(gen_motion - real_motion, axis=-1).mean(axis=-1)
        gen_magnitude = np.linalg.norm(gen_motion, axis=-1).mean(axis=-1)
        real_magnitude = np.linalg.norm(real_motion, axis=-1).mean(axis=-1)
        interocular = np.linalg.norm(real[1:, 36] - real[1:, 45], axis=-1)
        interocular = np.maximum(interocular, 1e-6)
        return {
            'pixels': float(motion_error[pair_valid].mean()),
            'normalized': float((motion_error / interocular)[pair_valid].mean()),
            'gen_motion': float(gen_magnitude[pair_valid].mean()),
            'real_motion': float(real_magnitude[pair_valid].mean()),
            'valid_pairs': int(pair_valid.sum()),
        }

    @staticmethod
    def calculate_landmark_jitter_from_features(gen_feat, real_feat):
        # Backward-compatible alias for earlier metric.py callers.
        return VideoMetricsCalculator.calculate_landmark_velocity_distance_from_features(
            gen_feat, real_feat
        )

    def _init_fan(self):
        if self.fan_predictor is not None:
            return True
        if self.fan_unavailable:
            return False
        try:
            import face_alignment
        except ImportError:
            self.fan_unavailable = True
            print('[WARN] face_alignment is not installed; LVD will use InsightFace kps5.')
            return False

        torch_load = torch.load

        def compatible_torch_load(*args, **kwargs):
            kwargs.pop('weights_only', None)
            return torch_load(*args, **kwargs)

        torch.load = compatible_torch_load
        try:
            try:
                landmark_type = face_alignment.LandmarksType._2D
            except AttributeError:
                landmark_type = face_alignment.LandmarksType.TWO_D
            self.fan_predictor = face_alignment.FaceAlignment(
                landmark_type,
                flip_input=False,
                device=str(self.device),
            )
        finally:
            torch.load = torch_load
        return True

    def extract_fan_landmarks(self, frames, name='video'):
        if not self._init_fan():
            return None
        results = []
        for start in range(0, len(frames), self.batch_size):
            batch = frames[start:start + self.batch_size]
            tensor = self._frames_to_tensor(batch, value_range='255')
            with torch.no_grad():
                pred_batch = self.fan_predictor.get_landmarks_from_batch(tensor)
            for pred in pred_batch:
                if pred is None or len(pred) == 0:
                    results.append(np.full((68, 2), np.nan, dtype=np.float32))
                else:
                    results.append(np.asarray(pred[:68], dtype=np.float32))
            print(f'[INFO] FAN landmarks {name}: {min(start + len(batch), len(frames))}/{len(frames)}')
        return np.stack(results, axis=0)

    @staticmethod
    def calculate_lmd_auc_from_landmarks(gen_lms, real_lms, lmd_region='mouth', auc_threshold=0.08):
        if gen_lms is None or real_lms is None:
            return None
        min_len = min(len(gen_lms), len(real_lms))
        gen_lms = gen_lms[:min_len]
        real_lms = real_lms[:min_len]
        valid = np.isfinite(gen_lms).all(axis=(1, 2)) & np.isfinite(real_lms).all(axis=(1, 2))
        if not valid.any():
            return {'lmd': 0.0, 'auc': 0.0, 'failure_rate': 1.0, 'valid_frames': 0}

        gen = gen_lms[valid]
        real = real_lms[valid]
        if lmd_region == 'mouth':
            idx = np.arange(48, 68)
        else:
            idx = np.arange(68)

        gen_region = gen[:, idx] - gen[:, idx].mean(axis=1, keepdims=True)
        real_region = real[:, idx] - real[:, idx].mean(axis=1, keepdims=True)
        lmd = np.linalg.norm(gen_region - real_region, axis=-1).mean()

        iod = np.linalg.norm(real[:, 36] - real[:, 45], axis=-1)
        iod = np.maximum(iod, 1e-6)
        nme = np.linalg.norm(gen - real, axis=-1).mean(axis=-1) / iod
        xs = np.linspace(0.0, auc_threshold, 100)
        ced = np.asarray([(nme <= x).mean() for x in xs], dtype=np.float32)
        auc = np.trapz(ced, xs) / auc_threshold
        failure_rate = float((nme > auc_threshold).mean())
        return {
            'lmd': float(lmd),
            'auc': float(auc),
            'failure_rate': failure_rate,
            'valid_frames': int(valid.sum()),
        }

    @staticmethod
    def calculate_landmark_jitter(gen_lms, real_lms, region='head'):
        if gen_lms is None or real_lms is None:
            return None
        min_len = min(len(gen_lms), len(real_lms))
        if min_len < 3:
            return {
                'gen': 0.0,
                'real': 0.0,
                'error': 0.0,
                'valid_triplets': 0,
            }
        gen = np.asarray(gen_lms[:min_len], dtype=np.float32)
        real = np.asarray(real_lms[:min_len], dtype=np.float32)
        if region == 'head':
            indices = np.concatenate((np.arange(17), np.arange(27, 36)))
        elif region == 'mouth':
            indices = np.arange(48, 68)
        elif region == 'all':
            indices = np.arange(68)
        else:
            raise ValueError('Unknown landmark jitter region: %s' % region)

        frame_valid = (
            np.isfinite(gen).all(axis=(1, 2))
            & np.isfinite(real).all(axis=(1, 2))
        )
        triplet_valid = frame_valid[:-2] & frame_valid[1:-1] & frame_valid[2:]
        if not triplet_valid.any():
            return {
                'gen': 0.0,
                'real': 0.0,
                'error': 0.0,
                'valid_triplets': 0,
            }

        gen_acceleration = gen[2:] - 2.0 * gen[1:-1] + gen[:-2]
        real_acceleration = real[2:] - 2.0 * real[1:-1] + real[:-2]
        interocular = np.linalg.norm(real[1:-1, 36] - real[1:-1, 45], axis=-1)
        interocular = np.maximum(interocular, 1e-6)

        gen_value = np.linalg.norm(
            gen_acceleration[:, indices], axis=-1
        ).mean(axis=-1) / interocular
        real_value = np.linalg.norm(
            real_acceleration[:, indices], axis=-1
        ).mean(axis=-1) / interocular
        error_value = np.linalg.norm(
            gen_acceleration[:, indices] - real_acceleration[:, indices], axis=-1
        ).mean(axis=-1) / interocular
        return {
            'gen': float(gen_value[triplet_valid].mean()),
            'real': float(real_value[triplet_valid].mean()),
            'error': float(error_value[triplet_valid].mean()),
            'valid_triplets': int(triplet_valid.sum()),
        }

    @staticmethod
    def _mouth_opening_from_landmarks(lms):
        valid = np.isfinite(lms).all(axis=(1, 2))
        opening = np.full((len(lms),), np.nan, dtype=np.float32)
        pairs = [(62, 66), (63, 65), (61, 67)]
        for i in np.where(valid)[0]:
            vals = [np.linalg.norm(lms[i, a] - lms[i, b]) for a, b in pairs]
            opening[i] = np.mean(vals)
        return opening

    @staticmethod
    def _load_audio_mono(audio_path, target_sr=16000):
        sr, audio = wavfile.read(audio_path)
        if audio.ndim > 1:
            audio = audio[:, 0]
        if np.issubdtype(audio.dtype, np.integer):
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
        else:
            audio = audio.astype(np.float32)
        if sr != target_sr:
            raise RuntimeError(f'Audio sample rate is {sr}, expected {target_sr}.')
        return sr, audio

    @staticmethod
    def _extract_audio_from_video(video_path, out_path):
        subprocess.run(
            ['ffmpeg', '-y', '-i', video_path, '-vn', '-ac', '1', '-ar', '16000', out_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def calculate_sync(self, gen_lms, fps, audio_path=None, video_path=None, max_offset_frames=15):
        if gen_lms is None:
            return None
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = audio_path
            if wav_path is None:
                if video_path is None:
                    return None
                wav_path = os.path.join(tmpdir, 'audio.wav')
                try:
                    self._extract_audio_from_video(video_path, wav_path)
                except Exception:
                    return None

            sr, audio = self._load_audio_mono(wav_path)
            mouth = self._mouth_opening_from_landmarks(gen_lms)
            valid = np.isfinite(mouth)
            if valid.sum() < 5:
                return {'sync_corr': 0.0, 'sync_offset_frames': 0, 'valid_frames': int(valid.sum())}

            hop = int(round(sr / fps))
            energy = []
            for i in range(len(mouth)):
                start = i * hop
                end = min(len(audio), start + hop)
                if end <= start:
                    energy.append(np.nan)
                else:
                    energy.append(float(np.mean(np.abs(audio[start:end]))))
            energy = np.asarray(energy, dtype=np.float32)

            valid = valid & np.isfinite(energy)
            mouth = mouth[valid]
            energy = energy[valid]
            if len(mouth) < 5:
                return {'sync_corr': 0.0, 'sync_offset_frames': 0, 'valid_frames': int(len(mouth))}
            mouth = (mouth - mouth.mean()) / (mouth.std() + 1e-6)
            energy = (energy - energy.mean()) / (energy.std() + 1e-6)

            best_corr = -1.0
            best_lag = 0
            max_lag = min(max_offset_frames, len(mouth) - 2)
            for lag in range(-max_lag, max_lag + 1):
                if lag < 0:
                    a, b = mouth[-lag:], energy[:lag]
                elif lag > 0:
                    a, b = mouth[:-lag], energy[lag:]
                else:
                    a, b = mouth, energy
                if len(a) < 3:
                    continue
                corr = float(np.mean(a * b))
                if corr > best_corr:
                    best_corr = corr
                    best_lag = lag
            return {
                'sync_corr': float(best_corr),
                'sync_offset_frames': int(best_lag),
                'valid_frames': int(len(mouth)),
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('gen_video', nargs='?', default='/home/lzq/paperCode/talkinghead/TalkingGaussian/output/exp_adaptive/test/ours_None/renders/out.mp4')
    parser.add_argument('real_video', nargs='?', default='/home/lzq/paperCode/talkinghead/TalkingGaussian/output/pose/test/ours_None/gt/out.mp4')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--det_size', type=int, default=640)
    parser.add_argument(
        '--flow_backend', choices=['farneback', 'dis', 'raft'],
        default='farneback',
    )
    parser.add_argument('--flow_max_side', type=int, default=512)
    parser.add_argument('--flow_fb_threshold', type=float, default=1.5)
    parser.add_argument('--flow_region', choices=['face', 'full'], default='full')
    args = parser.parse_args()

    if not os.path.exists(args.gen_video) or not os.path.exists(args.real_video):
        raise FileNotFoundError('Video path does not exist.')

    calculator = VideoMetricsCalculator(device=args.device, batch_size=args.batch_size, det_size=args.det_size)

    print('[INFO] loading video frames...')
    frames_gen, fps_gen = calculator.load_video_frames(args.gen_video, max_frames=args.max_frames)
    frames_real, fps_real = calculator.load_video_frames(args.real_video, max_frames=args.max_frames)
    min_len = min(len(frames_gen), len(frames_real))
    frames_gen = frames_gen[:min_len]
    frames_real = frames_real[:min_len]
    print(f'[INFO] frames: gen={len(frames_gen)}, real={len(frames_real)}, fps={fps_gen:.3f}/{fps_real:.3f}')

    print('\n[METRIC] PSNR / SSIM')
    avg_psnr, avg_ssim = calculator.calculate_psnr_ssim(frames_gen, frames_real)
    print(f'PSNR: {avg_psnr:.4f}')
    print(f'SSIM: {avg_ssim:.4f}')

    print('\n[METRIC] LPIPS / tLPIPS')
    avg_lpips = calculator.calculate_lpips(frames_gen, frames_real)
    avg_tlpips = calculator.calculate_tlpips(frames_gen, frames_real)
    print(f'LPIPS: {avg_lpips:.4f}')
    print(f'tLPIPS: {avg_tlpips:.4f}')

    print('\n[METRIC] FID')
    print(f'FID: {calculator.calculate_fid(frames_gen, frames_real):.4f}')

    print('\n[METRIC] CSIM')
    real_if = calculator.extract_insightface_features(frames_real, name='real')
    gen_if = calculator.extract_insightface_features(frames_gen, name='gen')
    csim = calculator.calculate_csim_from_features(gen_if, real_if)
    if csim is None:
        print('CSIM: skipped (InsightFace is unavailable)')
    else:
        print(f'CSIM: {csim:.6f}')

    # FAN landmarks are shared by LVD and optional face-region flow masking.
    print('\n[INFO] extracting FAN landmarks...')
    real_lms = calculator.extract_fan_landmarks(frames_real, name='real')
    gen_lms = (
        calculator.extract_fan_landmarks(frames_gen, name='gen')
        if real_lms is not None else None
    )

    print('\n[METRIC] Landmark Velocity Distance')
    if gen_lms is not None and real_lms is not None:
        lvd = calculator.calculate_landmark_velocity_distance(
            gen_lms, real_lms, region='head'
        )
        lvd_source = 'FAN-68 head landmarks'
    else:
        lvd = calculator.calculate_landmark_velocity_distance_from_features(
            gen_if, real_if
        )
        lvd_source = 'InsightFace kps5 fallback'
    if lvd is None:
        print('LVD: skipped (no landmark detector is available)')
    else:
        print(f'LVD_source: {lvd_source}')
        print(f'LVD_px: {lvd["pixels"]:.6f}')
        print(f'LVD_normalized: {lvd["normalized"]:.6f}')
        print(f'Landmark_Motion_gen_px: {lvd["gen_motion"]:.6f}')
        print(f'Landmark_Motion_real_px: {lvd["real_motion"]:.6f}')
        print(f'LVD_valid_pairs: {lvd["valid_pairs"]}')

    print('\n[METRIC] Flow Warp Error')
    flow_warp = calculator.calculate_flow_warp_error(
        frames_gen,
        frames_real,
        real_landmarks=real_lms,
        backend=args.flow_backend,
        max_side=args.flow_max_side,
        fb_threshold=args.flow_fb_threshold,
        region=args.flow_region,
    )
    print(f'Flow_Warp_Error_gen: {flow_warp["gen"]:.6f}')
    print(f'Flow_Warp_Error_real: {flow_warp["real"]:.6f}')
    print(f'Flow_Warp_Error_excess: {flow_warp["excess"]:.6f}')
    print(f'tOF_px: {flow_warp["tof_px"]:.6f}')
    print(f'tOF_normalized: {flow_warp["tof_normalized"]:.6f}')
    print(f'Flow_Warp_valid_pairs: {flow_warp["valid_pairs"]}')
    print(f'Flow_Warp_backend: {flow_warp["backend"]}')


if __name__ == '__main__':
    main()

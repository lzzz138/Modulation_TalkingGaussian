import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def sample_flow(flow, points):
    height, width = flow.shape[:2]
    x = points[:, 0]
    y = points[:, 1]
    valid = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    x0 = np.floor(np.clip(x, 0, width - 1)).astype(np.int64)
    y0 = np.floor(np.clip(y, 0, height - 1)).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (np.clip(x, 0, width - 1) - x0)[:, None]
    wy = (np.clip(y, 0, height - 1) - y0)[:, None]
    top = flow[y0, x0] * (1.0 - wx) + flow[y0, x1] * wx
    bottom = flow[y1, x0] * (1.0 - wx) + flow[y1, x1] * wx
    values = top * (1.0 - wy) + bottom * wy
    values[~valid] = 0.0
    return values, valid


def forward_backward_track(forward, backward, points):
    displacement, valid_a = sample_flow(forward, points)
    tracked = points + displacement
    reverse, valid_b = sample_flow(backward, tracked)
    error = np.linalg.norm(displacement + reverse, axis=-1)
    valid = valid_a & valid_b
    error[~valid] = np.inf
    confidence = np.exp(-error / 1.5).astype(np.float32)
    confidence[~valid] = 0.0
    return tracked.astype(np.float32), confidence


class FlowEstimator(object):
    def __init__(self, backend="raft", max_side=512, device="cuda"):
        self.backend = backend
        self.max_side = max_side
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.dis = None
        if backend == "raft":
            try:
                from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

                weights = Raft_Large_Weights.DEFAULT
                self.model = raft_large(weights=weights, progress=True).to(self.device).eval()
            except Exception as error:
                warnings.warn("RAFT unavailable; falling back to OpenCV DIS: %s" % error)
                self.backend = "dis"
        if self.backend == "dis":
            self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    def _raft_flow(self, image_a, image_b):
        original_h, original_w = image_a.shape[:2]
        scale = min(1.0, float(self.max_side) / max(original_h, original_w))
        height = max(8, int(round(original_h * scale / 8.0)) * 8)
        width = max(8, int(round(original_w * scale / 8.0)) * 8)

        def prepare(image):
            tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()[None]
            tensor = F.interpolate(tensor, (height, width), mode="bilinear", align_corners=False)
            return (tensor.to(self.device) / 255.0) * 2.0 - 1.0

        with torch.no_grad():
            flow = self.model(prepare(image_a), prepare(image_b))[-1]
            flow = F.interpolate(
                flow, (original_h, original_w), mode="bilinear", align_corners=False
            )[0]
            flow[0] *= float(original_w) / width
            flow[1] *= float(original_h) / height
        return flow.permute(1, 2, 0).cpu().numpy().astype(np.float32)

    def _dis_flow(self, image_a, image_b):
        gray_a = cv2.cvtColor(image_a, cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(image_b, cv2.COLOR_RGB2GRAY)
        return self.dis.calc(gray_a, gray_b, None).astype(np.float32)

    def flow(self, image_a, image_b):
        if self.backend == "raft":
            try:
                return self._raft_flow(image_a, image_b)
            except RuntimeError as error:
                if "out of memory" not in str(error).lower():
                    raise
                warnings.warn("RAFT ran out of memory; switching to OpenCV DIS")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.backend = "dis"
                self.model = None
                self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        return self._dis_flow(image_a, image_b)

    def bidirectional(self, image_a, image_b):
        initial_backend = self.backend
        forward = self.flow(image_a, image_b)
        backward = self.flow(image_b, image_a)
        if initial_backend == "raft" and self.backend == "dis":
            forward = self._dis_flow(image_a, image_b)
            backward = self._dis_flow(image_b, image_a)
        return forward, backward

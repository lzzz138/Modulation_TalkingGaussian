"""Training-only DenseMarks observations and conservative mesh surface masks."""
import hashlib
import json
import os
from collections import OrderedDict
import cv2
import numpy as np
import torch
from data_utils.head_stabilizer.se3 import matrix_multiply
import torch.nn.functional as F
from data_utils.canonical_alignment.aligner import DenseMarksCache
from data_utils.face_tracking.util import euler2rot


def dataset_digest(root):
    digest = hashlib.sha256()
    for name in ('transforms_train.json', 'transforms_val.json', 'track_params_canonical.pt',
                 'canonical_features/manifest.json', 'canonical_diagnostics.json'):
        with open(os.path.join(root, name), 'rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


def preflight(root):
    names = ('transforms_train.json', 'transforms_val.json', 'track_params_canonical.pt',
             'canonical_features/manifest.json', 'canonical_diagnostics.json')
    for name in names:
        if not os.path.isfile(os.path.join(root, name)):
            raise RuntimeError('Missing %s. Run data_utils/process.py VIDEO --task 11 '
                               '--canonical_keep_cache (with DenseMarks environment/weights), '
                               'then --task 9 --track_params track_params_canonical.pt.' % name)
    train = json.load(open(os.path.join(root, names[0])))
    val = json.load(open(os.path.join(root, names[1])))
    ids = [int(f['img_id']) for f in train['frames']]
    val_ids = [int(f['img_id']) for f in val['frames']]
    if len(ids) != len(set(ids)) or set(ids) & set(val_ids):
        raise ValueError('Duplicate or overlapping training/validation frame IDs')
    params = torch.load(os.path.join(root, names[2]), map_location='cpu')
    rot = params.get('rot', euler2rot(params['euler']))
    trans = params['trans'] / 10
    for f in train['frames'] + val['frames']:
        i = int(f['img_id'])
        expected = torch.eye(4)
        expected[:3, :3] = rot[i].T
        expected[:3, 3] = -(rot[i].T @ trans[i])
        if not torch.allclose(torch.tensor(f['transform_matrix']), expected, atol=2e-5):
            raise ValueError('transforms disagree with offline CFHS at frame %d' % i)
    manifest = json.load(open(os.path.join(root, names[3])))
    paths = [os.path.join(root, 'ori_imgs', '%d.jpg' % f['frame_id']) for f in manifest['frames']]
    cache = DenseMarksCache(os.path.join(root, 'canonical_features'), paths, manifest['feature_size'])
    if not set(ids).issubset(set(cache.frame_ids)):
        raise ValueError('Training frame missing from DenseMarks cache')
    return params, cache, dataset_digest(root)


def sample(image, xy, width, height):
    grid = torch.stack((xy[:, 0] * 2 / (width - 1) - 1,
                        xy[:, 1] * 2 / (height - 1) - 1), -1)[None, None]
    return F.grid_sample(image[None], grid, align_corners=True)[0, :, 0].T


class Observations:
    def __init__(self, root, cameras, params, cache, output, capacity=16):
        from data_utils.face_tracking.facemodel import Face_3DMM
        self.root, self.output, self.capacity = root, output, capacity
        self.cameras = {int(c.talking_dict['img_id']): c for c in cameras}
        self.params, self.cache = params, cache
        self.indices = {i: k for k, i in enumerate(cache.frame_ids)}
        self.memory = OrderedDict()
        self.mesh_model = Face_3DMM('data_utils/face_tracking/3DMM', 100, 79, 100, 34650)
        # Geometry is preprocessing only; CPU GEMM avoids unsupported legacy CUDA BLAS.
        for name in ('base_id', 'base_exp', 'mu', 'sig_id', 'sig_exp'):
            setattr(self.mesh_model, name, getattr(self.mesh_model, name).cpu())
        neutral = self.mesh_model.forward_geo(
            params['id'].cpu(), torch.zeros_like(params['exp'][:1]).cpu()
        )[0]
        # keyinds contain fixed landmarks 17..67 (contours are dynamic).
        keys = neutral[self.mesh_model.keyinds.long()[17:]]
        brow_y = keys[:10, 1].median()
        eye_y = keys[19:31, 1].median()
        nose_top = keys[10]
        nose_bottom = keys[13]
        width = (keys[19, 0] - keys[28, 0]).abs().clamp_min(1e-6)
        x = (neutral[:, 0] - nose_top[0]).abs()
        y = neutral[:, 1]
        # Mesh-local anatomical regions, excluding cheeks/jaw and eye/brow bands.
        forehead = (y > brow_y + 0.12 * width) & (x < 0.70 * width)
        bridge = (x < 0.09 * width) & (y < nose_top[1]) & (y > nose_bottom[1] + 0.08 * width)
        temples = (x > 0.48 * width) & (x < 0.72 * width) & (y > eye_y) & (y < brow_y + 0.35 * width)
        self.stable_vertices = forehead | bridge | temples
        d = json.load(open(os.path.join(root, 'canonical_diagnostics.json')))
        self.conf = {int(f['frame']): float(f.get('confidence', float(f['accepted']))) for f in d['frame_diagnostics']}
        self.yaw = {i: float(params['euler'][i, 1]) * 180 / np.pi for i in self.cameras}

    def selected(self, maximum=64):
        bins = {}
        for i in self.cameras:
            bins.setdefault(int(round(self.yaw[i] / 15)), []).append(i)
        for b in bins:
            bins[b].sort(key=lambda i: (-self.conf.get(i, 0), i))
        selected = []
        while len(selected) < maximum and any(bins.values()):
            for b in sorted(bins):
                if bins[b] and len(selected) < maximum:
                    selected.append(bins[b].pop(0))
        return selected

    @torch.no_grad()
    def get(self, i):
        if i not in self.cameras:
            raise ValueError('UVW observations may only be read for training frames')
        if i in self.memory:
            self.memory.move_to_end(i)
            return {k: v.cuda() for k, v in self.memory[i].items()}
        from data_utils.face_tracking.render_3dmm import Render_3DMM
        from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
        from pytorch3d.structures import Meshes
        c = self.cameras[i]
        h, w = c.image_height, c.image_width
        # Match full image projection; mask rasterization is deliberately no-grad.
        raw = cv2.imread(os.path.join(self.root, 'ori_imgs', '%d.jpg' % i))
        if raw is None:
            raise ValueError('Missing source image for frame %d' % i)
        focal = float(self.params['focal'].item()) * w / raw.shape[1]
        # Some supported legacy CUDA/cuBLAS combinations fail inside
        # PyTorch3D's Transform3d batched matrix multiplication. Projection is
        # preprocessing-only, so perform it on CPU and send the already
        # projected mesh to the CUDA rasterizer.
        renderer = Render_3DMM(focal, h, w, 1, torch.device('cpu'))
        geometry = self.mesh_model.forward_geo(
            self.params['id'].cpu(), self.params['exp'][i:i+1].cpu()
        )
        rot = self.params.get('rot')
        r = rot[i].cpu() if rot is not None else euler2rot(self.params['euler'][i:i+1].cpu())[0]
        camera_geometry = matrix_multiply(geometry, r.T) + self.params['trans'][i].cpu()
        mesh = Meshes(camera_geometry, renderer.tris[None])
        mesh_ndc = renderer.renderer.rasterizer.transform(mesh).cuda()
        settings = renderer.renderer.rasterizer.raster_settings
        pix_to_face = rasterize_meshes(
            mesh_ndc,
            image_size=settings.image_size,
            blur_radius=settings.blur_radius,
            faces_per_pixel=settings.faces_per_pixel,
            bin_size=settings.bin_size,
            max_faces_per_bin=settings.max_faces_per_bin,
            clip_barycentric_coords=(settings.clip_barycentric_coords
                                     if settings.clip_barycentric_coords is not None
                                     else settings.blur_radius > 0.0),
            perspective_correct=settings.perspective_correct,
            cull_backfaces=settings.cull_backfaces,
            z_clip_value=settings.z_clip_value,
            cull_to_frustum=settings.cull_to_frustum,
        )[0]
        face_ids = pix_to_face[0, ..., 0]
        stable_faces = self.stable_vertices[renderer.tris].all(-1).cuda()
        mask = ((face_ids >= 0) & stable_faces[face_ids.clamp_min(0)]).cpu().numpy().astype(np.uint8)
        parsing = cv2.imread(os.path.join(self.root, 'parsing', '%d.png' % i))
        if parsing is None:
            raise ValueError('Missing parsing for frame %d' % i)
        parsing = cv2.resize(parsing, (w, h), interpolation=cv2.INTER_NEAREST)
        skin = (parsing[..., 0] == 255) & (parsing[..., 1] == 0) & (parsing[..., 2] == 0)
        mask &= skin.astype(np.uint8)
        lms = np.loadtxt(os.path.join(self.root, 'ori_imgs', '%d.lms' % i)).astype(np.float32)
        raw = cv2.imread(os.path.join(self.root, 'ori_imgs', '%d.jpg' % i))
        lms *= np.array([w / raw.shape[1], h / raw.shape[0]])
        for idx in (list(range(17, 27)), list(range(36, 42)), list(range(42, 48)), list(range(31, 36)), list(range(48, 68))):
            excluded = np.zeros_like(mask)
            cv2.fillConvexPoly(excluded, cv2.convexHull(lms[idx].astype(np.int32)), 1)
            excluded = cv2.dilate(excluded, np.ones((7, 7), np.uint8))
            mask[excluded > 0] = 0
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
        uvw, _ = self.cache.load(self.indices[i], torch.device('cpu'))
        uvw = F.interpolate(uvw, size=(h, w), mode='bilinear', align_corners=True)[0]
        result = dict(uvw=uvw, mask=torch.from_numpy(mask).bool())
        self.memory[i] = result
        if len(self.memory) > self.capacity:
            self.memory.popitem(last=False)
        overlay_dir = os.path.join(self.output, 'mask_overlays')
        os.makedirs(overlay_dir, exist_ok=True)
        if i in self.selected():
            overlay = cv2.resize(raw, (w, h))
            overlay[mask > 0] = (0.5 * overlay[mask > 0] + np.array([0, 127, 0])).astype(np.uint8)
            cv2.imwrite(os.path.join(overlay_dir, '%d.jpg' % i), overlay)
        return {k: v.cuda() for k, v in result.items()}

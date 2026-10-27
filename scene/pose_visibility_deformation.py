"""Pose-conditioned residuals. All geometric observation inputs are detached."""
import math
import torch
from torch import nn
import torch.nn.functional as F


def linear(x, layer):
    if x.shape[0] > 2048:
        return torch.cat([linear(chunk, layer) for chunk in x.split(2048)], dim=0)
    return (x.unsqueeze(-2) * layer.weight).sum(-1) + layer.bias


def row_transform(points, matrix):
    # Row-vector convention used by GaussianRasterizer; avoids small CUDA GEMMs.
    return (points[..., :, None] * matrix).sum(-2)


@torch.no_grad()
def pose_condition(camera, centers):
    # Stored world_view_transform is W2C.T. Its upper 3x3 is C2W rotation.
    rotation = camera.world_view_transform.detach()[:3, :3]
    code = rotation[:, :2].T.reshape(1, 6).expand(len(centers), -1)
    direction = F.normalize(camera.camera_center.detach()[None] - centers.detach(), dim=-1, eps=1e-8)
    return torch.cat((code, direction), -1)


@torch.no_grad()
def visibility_descriptor(camera, centers, scales, depth, alpha):
    hom = torch.cat((centers.detach(), torch.ones_like(centers[:, :1])), -1)
    view = row_transform(hom, camera.world_view_transform.detach())
    clip = row_transform(hom, camera.full_proj_transform.detach())
    denom = clip[:, 3:4]
    ndc = clip[:, :2] / torch.where(denom.abs() > 1e-8, denom, torch.ones_like(denom))
    grid = torch.nan_to_num(ndc).reshape(1, -1, 1, 2)
    # Rasterizer ndc2Pix maps NDC endpoints to -0.5 and W-0.5.
    sampled_alpha = F.grid_sample(alpha[None], grid, align_corners=False).reshape(-1)
    normalized_depth = depth / alpha.clamp_min(1e-8)
    sampled_depth = F.grid_sample(normalized_depth[None], grid, align_corners=False).reshape(-1)
    valid = (torch.isfinite(clip).all(-1) & torch.isfinite(view).all(-1)
             & (view[:, 2] > .2) & (denom[:, 0] > 1e-8)
             & (ndc.abs() <= 1).all(-1) & (sampled_alpha > 1e-6)
             & torch.isfinite(sampled_depth))
    tolerance = torch.maximum(view[:, 2].abs() * .01, scales.detach().amax(-1) * 2).clamp_min(1e-8)
    error = (view[:, 2] - sampled_depth) / tolerance
    confidence = sampled_alpha * torch.exp(-.5 * error.square())
    features = torch.stack((valid.to(centers.dtype), sampled_alpha, error.clamp(-3, 3), confidence), -1)
    return torch.where(valid[:, None], features, torch.zeros_like(features))


@torch.no_grad()
def prerender_visibility(camera, centers, scales, rotations, opacity, resolution, modifier=1., debug=False):
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    factor = resolution / min(camera.image_height, camera.image_width)
    settings = GaussianRasterizationSettings(
        image_height=max(1, round(camera.image_height * factor)),
        image_width=max(1, round(camera.image_width * factor)),
        tanfovx=math.tan(camera.FoVx * .5), tanfovy=math.tan(camera.FoVy * .5),
        bg=centers.new_zeros(3), scale_modifier=modifier,
        viewmatrix=camera.world_view_transform.detach(), projmatrix=camera.full_proj_transform.detach(),
        sh_degree=0, campos=camera.camera_center.detach(), prefiltered=False, debug=debug)
    _, _, depth, alpha = GaussianRasterizer(settings)(
        means3D=centers.detach(), means2D=torch.zeros_like(centers), shs=None,
        colors_precomp=torch.zeros_like(centers), opacities=opacity.detach(),
        scales=scales.detach(), rotations=rotations.detach(), cov3D_precomp=None)
    return visibility_descriptor(camera, centers, scales * modifier, depth, alpha)


class PoseVisibilityResidual(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.condition = args.pcv_condition
        self.maximum = args.pcv_max_displacement
        self.resolution = args.pcv_visibility_resolution
        self.fc1, self.fc2 = nn.Linear(87, 64), nn.Linear(64, 64)
        self.output, self.gate = nn.Linear(64, 3), nn.Linear(64, 1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, -2.)
        self.register_buffer('strength', torch.tensor(0.))

    def set_iteration(self, iteration, start, ramp):
        self.strength.fill_(max(0., min(1., (iteration - start) / ramp)))

    def forward(self, features, pose, visibility):
        if self.condition == 'none':
            pose, visibility = torch.zeros_like(pose), torch.zeros_like(visibility)
        elif self.condition == 'pose':
            visibility = torch.zeros_like(visibility)
        x = torch.cat((features, pose.detach(), visibility.detach()), -1)
        h = torch.relu(linear(torch.relu(linear(x, self.fc1)), self.fc2))
        gate = linear(h, self.gate).sigmoid()
        raw_residual = linear(h, self.output).tanh()
        normalized = gate * raw_residual
        delta = self.strength * self.maximum * normalized
        regularization = normalized.square().sum(-1).mean()
        diagnostics = dict(gate_mean=gate.detach().mean(), residual_rms=delta.detach().square().mean().sqrt(),
                           saturation=(raw_residual.detach().abs() > .95).float().mean(),
                           gate_saturation=((gate.detach() < .01) | (gate.detach() > .99)).float().mean(),
                           valid_fraction=visibility[:, 0].mean(), visibility_mean=visibility[:, 3].mean())
        return delta, regularization, diagnostics


def apply_pose_residual(camera, pc, network, predictions, scales, rotations, modifier=1., debug=False):
    module = network.pose_residual
    predictions['pcv_regularization'] = pc.get_xyz.new_zeros(())
    if module.strength.item() == 0 or module.maximum == 0:
        predictions.pop('pcv_features', None)
        return predictions
    profile = getattr(network, 'pcv_profile', False)
    if profile:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
    centers = pc.get_xyz + predictions['d_xyz']
    visibility = (prerender_visibility(camera, centers, scales, rotations, pc.get_opacity,
                   module.resolution, modifier, debug) if module.condition == 'pose_visibility'
                   else centers.new_zeros(len(centers), 4))
    delta, reg, diag = module(predictions.pop('pcv_features'), pose_condition(camera, centers), visibility)
    if profile:
        end.record(); end.synchronize()
        diag['extra_forward_ms'] = start.elapsed_time(end)
    predictions['d_xyz'] = predictions['d_xyz'] + delta
    predictions['pcv_regularization'] = reg
    predictions['pcv_diagnostics'] = diag
    return predictions


def validate_options(args):
    if not getattr(args, 'pose_visibility_deformation', False):
        return
    if getattr(args, 'uvw_pose', False) or getattr(args, 'uvw_pose_post_topology', False):
        raise ValueError('PCV-DF requires fixed CFHS cameras, not online pose optimization')
    if args.pcv_condition not in ('none', 'pose', 'pose_visibility'):
        raise ValueError('Unknown pcv_condition')
    if args.pcv_start < 3000 or args.pcv_ramp_steps < 1 or args.pcv_visibility_resolution < 16 or args.pcv_max_displacement < 0 or args.pcv_lr <= 0 or args.pcv_reg_weight < 0:
        raise ValueError('Invalid PCV-DF configuration')


def yaw_groups(cameras):
    """Training-only circular centering avoids the +/-180 degree Euler seam."""
    import numpy as np
    values = []
    for c in cameras:
        r = c.world_view_transform[:3, :3]
        values.append(math.atan2(float(r[0, 2]), float(r[2, 2])))
    values = np.asarray(values)
    center = math.atan2(float(np.sin(values).mean()), float(np.cos(values).mean()))
    angles = np.degrees(np.arctan2(np.sin(values-center), np.cos(values-center)))
    edges = np.quantile(angles, [1/3, 2/3]).tolist()
    by_id = {int(c.talking_dict['img_id']): dict(yaw_degrees=float(a), yaw_group=int(np.searchsorted(edges, a, side='right')))
             for c, a in zip(cameras, angles)}
    return dict(center_radians=center, tertile_edges_degrees=edges), by_id

"""Camera-space rasterization: pose gradients travel through centers/covariances."""
import math
from contextlib import nullcontext
import torch
from data_utils.head_stabilizer.se3 import matrix_multiply
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.general_utils import build_rotation, strip_symmetric
from utils.sh_utils import eval_sh


def render_uvw(camera, pc, motion, pipe, background, world_to_camera=None,
               freeze_field=False, render_attributes=True, mouth=False):
    context = torch.no_grad() if freeze_field else nullcontext()
    with context:
        xyz, scales, rotation = pc.get_xyz, pc.get_scaling, pc.get_rotation
        predictions = None
        if motion is not None:
            args = (xyz, camera.talking_dict['auds'].cuda(), camera.talking_dict['au_exp'].cuda())
            predictions = motion(*args[:2]) if mouth else motion(*args, gaussian_scaling=pc.get_scaling.detach())
            xyz = xyz + predictions['d_xyz']
            if not mouth:
                scales = pc.scaling_activation(pc._scaling + predictions['d_scale'])
                rotation = pc.rotation_activation(pc._rotation + predictions['d_rot'])
        # R @ diag(scales) is exactly R with each column scaled. Writing it
        # directly avoids the batched cuBLAS GEMM which is unsupported by the
        # legacy CUDA stack used by this project.
        factor = build_rotation(rotation) * scales[:, None, :]
        cov = matrix_multiply(factor, factor.transpose(1, 2))
        opacity, sh = pc.get_opacity, pc.get_features
    if freeze_field:
        xyz, cov, opacity, sh = [v.detach() for v in (xyz, cov, opacity, sh)]
    w = camera.world_view_transform.T if world_to_camera is None else world_to_camera
    r, t = w[:3, :3], w[:3, 3]
    means = matrix_multiply(xyz, r.T) + t
    covariance = strip_symmetric(matrix_multiply(matrix_multiply(r[None], cov), r.T[None]))
    center = -((r.T * t[None]).sum(-1))
    directions = xyz - center
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    colors = (eval_sh(pc.active_sh_degree, sh.transpose(1, 2), directions) + 0.5).clamp_min(0)
    screen = torch.zeros_like(means, requires_grad=True) + 0
    if screen.requires_grad:
        screen.retain_grad()

    def raster(color, bg):
        settings = GaussianRasterizationSettings(
            image_height=camera.image_height, image_width=camera.image_width,
            tanfovx=math.tan(camera.FoVx / 2), tanfovy=math.tan(camera.FoVy / 2),
            bg=bg, scale_modifier=1.0, viewmatrix=torch.eye(4, device=xyz.device),
            projmatrix=camera.projection_matrix, sh_degree=0,
            campos=torch.zeros(3, device=xyz.device), prefiltered=False, debug=pipe.debug)
        return GaussianRasterizer(raster_settings=settings)(
            means3D=means, means2D=screen, shs=None, colors_precomp=color,
            opacities=opacity, scales=None, rotations=None, cov3D_precomp=covariance)

    rgb, radii, depth, alpha = raster(colors, background)
    result = dict(render=rgb, alpha=alpha, depth=depth, radii=radii,
                  visibility_filter=radii > 0, viewspace_points=screen, motion=predictions)
    if render_attributes and getattr(pc, 'fixed_uvw', None) is not None:
        fixed = pc.fixed_uvw
        q = fixed.weights(pc.get_xyz).detach()
        numerator = raster(q * fixed.uvw, torch.zeros_like(background))[0]
        denominator = raster(q.expand(-1, 3).contiguous(), torch.zeros_like(background))[0][:1]
        result.update(uvw=numerator / denominator.clamp_min(1e-6), uvw_coverage=denominator)
    return result

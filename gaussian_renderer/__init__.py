#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.motion_net import MotionNetwork, MouthMotionNetwork
from utils.sh_utils import eval_sh
from utils.general_utils import build_rotation
from utils.se3 import matrix_multiply, matrix_vector


def _unpack_covariance(covariance):
    matrix = covariance.new_zeros((covariance.shape[0], 3, 3))
    matrix[:, 0, 0] = covariance[:, 0]
    matrix[:, 0, 1] = matrix[:, 1, 0] = covariance[:, 1]
    matrix[:, 0, 2] = matrix[:, 2, 0] = covariance[:, 2]
    matrix[:, 1, 1] = covariance[:, 3]
    matrix[:, 1, 2] = matrix[:, 2, 1] = covariance[:, 4]
    matrix[:, 2, 2] = covariance[:, 5]
    return matrix


def _pack_covariance(matrix):
    return torch.stack((matrix[:, 0, 0], matrix[:, 0, 1], matrix[:, 0, 2],
                        matrix[:, 1, 1], matrix[:, 1, 2], matrix[:, 2, 2]), dim=-1)


def _pose_raster_settings(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, refined_pose):
    if refined_pose is None:
        viewmatrix = viewpoint_camera.world_view_transform
        projmatrix = viewpoint_camera.full_proj_transform
        campos = viewpoint_camera.camera_center
    else:
        viewmatrix = torch.eye(4, device=refined_pose.rotation.device,
                               dtype=refined_pose.rotation.dtype)
        projmatrix = viewpoint_camera.projection_matrix
        campos = torch.zeros(3, device=refined_pose.rotation.device,
                             dtype=refined_pose.rotation.dtype)
    return GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=math.tan(viewpoint_camera.FoVx * 0.5),
        tanfovy=math.tan(viewpoint_camera.FoVy * 0.5),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=pc.active_sh_degree,
        campos=campos,
        prefiltered=False,
        debug=pipe.debug,
    )


def _apply_refined_pose(means3D, pc, refined_pose, scaling_modifier=1.0,
                        scales=None, rotations=None, colors_precomp=None, shs=None):
    if refined_pose is None:
        return means3D, scales, rotations, None, colors_precomp, shs

    rotation = refined_pose.rotation
    means_camera = matrix_vector(rotation.unsqueeze(0), means3D) + refined_pose.translation
    if scales is None or rotations is None:
        scales, rotations = pc.get_scaling, pc.get_rotation
    gaussian_rotation = build_rotation(rotations)
    squared_scale = (scaling_modifier * scales).square()
    covariance_world = (
        gaussian_rotation[:, :, None, :]
        * gaussian_rotation[:, None, :, :]
        * squared_scale[:, None, None, :]
    ).sum(dim=-1)
    rotated_once = matrix_multiply(rotation.unsqueeze(0), covariance_world)
    covariance_camera = matrix_multiply(rotated_once, rotation.transpose(0, 1).unsqueeze(0))

    if colors_precomp is None and shs is not None:
        shs_view = shs.transpose(1, 2).reshape(-1, 3, (pc.max_sh_degree + 1) ** 2)
        direction = means3D - refined_pose.camera_center.unsqueeze(0)
        direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-8)
        colors_precomp = torch.clamp_min(eval_sh(pc.active_sh_degree, shs_view, direction) + 0.5, 0.0)
    return means_camera, None, None, _pack_covariance(covariance_camera), colors_precomp, None

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, refined_pose=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = _pose_raster_settings(viewpoint_camera, pc, pipe, bg_color,
                                              scaling_modifier, refined_pose)

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    if refined_pose is not None:
        means3D, scales, rotations, cov3D_precomp, colors_precomp, shs = _apply_refined_pose(
            means3D, pc, refined_pose, scaling_modifier, scales, rotations,
            colors_precomp, shs)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "depth": rendered_depth, 
            "alpha": rendered_alpha,
            "radii": radii}


def render_motion(viewpoint_camera, pc : GaussianModel, motion_net : MotionNetwork, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, frame_idx = None, return_attn = False, refined_pose=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = _pose_raster_settings(viewpoint_camera, pc, pipe, bg_color,
                                              scaling_modifier, refined_pose)

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    audio_feat = viewpoint_camera.talking_dict["auds"].cuda()
    exp_feat = viewpoint_camera.talking_dict["au_exp"].cuda()

    # ind_code = motion_net.individual_codes[frame_idx if frame_idx is not None else viewpoint_camera.talking_dict["img_id"]]
    ind_code = None
    motion_preds = motion_net(
        pc.get_xyz,
        audio_feat,
        exp_feat,
        ind_code,
        gaussian_scaling=pc.get_scaling.detach(),
    )
    means3D = pc.get_xyz + motion_preds['d_xyz']
    means2D = screenspace_points
    # opacity = pc.opacity_activation(pc._opacity + motion_preds['d_opa'])
    opacity = pc.get_opacity

    cov3D_precomp = None
    # scales = pc.get_scaling
    scales = pc.scaling_activation(pc._scaling + motion_preds['d_scale'])
    rotations = pc.rotation_activation(pc._rotation + motion_preds['d_rot'])

    colors_precomp = None
    shs = pc.get_features

    if refined_pose is not None:
        means3D, scales, rotations, cov3D_precomp, colors_precomp, shs = _apply_refined_pose(
            means3D, pc, refined_pose, scaling_modifier, scales, rotations,
            colors_precomp, shs)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    # Attn
    rendered_attn = None
    if return_attn:
        attn_precomp = torch.cat([motion_preds['ambient_aud'], motion_preds['ambient_eye'], torch.zeros_like(motion_preds['ambient_eye'])], dim=-1)
        rendered_attn, _, _, _ = rasterizer(
            means3D = means3D.detach(),
            means2D = means2D,
            shs = None,
            colors_precomp = attn_precomp,
            opacities = opacity.detach(),
            scales = scales.detach() if scales is not None else None,
            rotations = rotations.detach() if rotations is not None else None,
            cov3D_precomp = cov3D_precomp.detach() if cov3D_precomp is not None else None)


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "depth": rendered_depth, 
            "alpha": rendered_alpha,
            "radii": radii,
            "motion": motion_preds,
            'attn': rendered_attn}





def render_motion_mouth(viewpoint_camera, pc : GaussianModel, motion_net : MouthMotionNetwork, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, frame_idx = None, return_attn = False, refined_pose=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = _pose_raster_settings(viewpoint_camera, pc, pipe, bg_color,
                                              scaling_modifier, refined_pose)

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    audio_feat = viewpoint_camera.talking_dict["auds"].cuda()

    motion_preds = motion_net(pc.get_xyz, audio_feat)
    means3D = pc.get_xyz + motion_preds['d_xyz']
    means2D = screenspace_points
    opacity = pc.get_opacity

    cov3D_precomp = None
    scales = pc.get_scaling
    rotations = pc.get_rotation

    colors_precomp = None
    shs = pc.get_features

    if refined_pose is not None:
        means3D, scales, rotations, cov3D_precomp, colors_precomp, shs = _apply_refined_pose(
            means3D, pc, refined_pose, scaling_modifier, scales, rotations,
            colors_precomp, shs)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "depth": rendered_depth, 
            "alpha": rendered_alpha,
            "radii": radii,
            "motion": motion_preds}

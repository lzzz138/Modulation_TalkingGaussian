"""Strict UVW checkpoint loading for frozen-pose Fuse and synthesis."""
import copy
import torch
from data_utils.head_stabilizer.se3 import matrix_multiply
from scene.uvw_pose import PoseTable, FixedUVW
from scene.uvw_observations import dataset_digest


def attach_pose(scene, dataset, checkpoint):
    if checkpoint['dataset_digest'] != dataset_digest(dataset.source_path):
        raise ValueError('UVW checkpoint belongs to a different dataset/transforms/cache')
    state = checkpoint['pose']
    # Synthesis can load validation cameras only. Construct the complete saved table
    # without requiring training images or accidentally reindexing validation frames.
    table = PoseTable([], device='cuda')
    table.base = {i: w.cuda() for i, w in state['base'].items()}
    table.parameters = {i: torch.nn.Parameter(p.cuda(), requires_grad=False) for i,p in state['parameters'].items()}
    table.angle, table.translation_ratio = state['angle'], state['translation_ratio']
    table.acceleration_delay = state.get('acceleration_delay', table.acceleration_delay)
    table.acceleration_ramp = state.get('acceleration_ramp', table.acceleration_ramp)
    table.accepted_updates = state.get('accepted_updates', 0)
    table.center = state['center'].cuda()
    for cameras in list(scene.train_cameras.values()) + list(scene.test_cameras.values()):
        for c in cameras:
            i = int(c.talking_dict['img_id'])
            if i not in table.base:
                continue
            if not torch.allclose(c.world_view_transform.T, table.base[i], atol=1e-6):
                raise ValueError('UVW checkpoint camera differs at frame %d' % i)
            w = table.matrix(c).detach()
            c.world_view_transform = w.T.contiguous()
            c.full_proj_transform = matrix_multiply(c.world_view_transform, c.projection_matrix)
            c.camera_center = -((w[:3,:3].T * w[:3,3][None]).sum(-1))
    return table


def load_joint(checkpoint, gaussians, motion, mouth_gaussians, mouth_motion, opt=None):
    if checkpoint.get('format') not in ('uvw_pose_v1', 'uvw_pose_fuse_v1'):
        raise ValueError('Unsupported UVW checkpoint format')
    for name, g, net in (('face', gaussians, motion), ('mouth', mouth_gaussians, mouth_motion)):
        b = checkpoint['branches'][name]
        g.restore(b['gaussians'], opt)
        net.load_state_dict(b['motion'], strict=True)
        if name == 'face':
            if b['uvw'] is None:
                raise ValueError('UVW training has not reached the binding step')
            g.fixed_uvw = FixedUVW.restore(b['uvw'], 'cuda')

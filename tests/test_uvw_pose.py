import copy
import unittest
from types import SimpleNamespace
import torch
from scene.uvw_pose import FixedUVW, PoseTable, robust_binding


def camera(i=0, device='cpu'):
    w = torch.eye(4, device=device)
    w[2,3] = 2.0
    return SimpleNamespace(talking_dict={'img_id': i}, world_view_transform=w.T)


class UVWPoseTest(unittest.TestCase):
    def test_rollback_and_no_unsampled_drift(self):
        table = PoseTable([camera(2), camera(3)], device='cpu')
        p = table.parameters[2]
        other = table.parameters[3].clone()
        accepted = table.update(2, lambda: ((p - .3).square().mean(), 1.0))
        self.assertTrue(accepted['accepted'])
        self.assertEqual(accepted['reason'], 'accepted')
        self.assertIsNotNone(accepted['gradient'])
        self.assertIsNotNone(accepted['candidate_correction'])
        saved = p.detach().clone()
        step = table.optimizers[2].state[p]['step'].clone()
        rejected = table.update(2, lambda: ((p - .3).square().mean(), 0.0))
        self.assertFalse(rejected['accepted'])
        self.assertEqual(rejected['reason'], 'coverage_below_minimum')
        self.assertNotEqual(rejected['candidate_correction'], saved.tolist())
        torch.testing.assert_close(saved, p)
        torch.testing.assert_close(step, table.optimizers[2].state[p]['step'])
        torch.testing.assert_close(other, table.parameters[3])
        table.parameters[2].data[:] = torch.tensor([0, .5, 0, 0, 0, 0])
        torch.testing.assert_close(table.matrix(camera(2))[:3,3], torch.tensor([0.,0.,2.]))

    def test_uvw_inheritance_and_drift(self):
        uvw = FixedUVW(torch.rand(3,3), torch.ones(3), torch.ones(3), torch.zeros(3,3), torch.ones(3))
        original = uvw.uvw.clone()
        uvw.append_parents(torch.tensor([2,0,2]))
        torch.testing.assert_close(uvw.uvw[3:], original[[2,0,2]])
        uvw.select(torch.tensor([False, True, True, True, False, True]))
        self.assertEqual(len(uvw.uvw),4)
        xyz = torch.tensor([[0.,0,0],[2.,0,0],[3.,0,0],[4.,0,0]])
        torch.testing.assert_close(uvw.weights(xyz).flatten(), torch.tensor([1.,1.,.5,0.]))

    def test_robust_binding_rejects_single_view_and_outliers(self):
        values = torch.zeros(8,2,3) + .4
        values[-1,0] = .8
        weights = torch.ones(8,2)
        weights[2:,1] = 0
        result = robust_binding(values, weights, torch.tensor([0,0,0,0,1,1,1,1]),
                                torch.zeros(2,3),torch.ones(2),torch.ones(3))
        self.assertTrue(result.valid[0])
        self.assertFalse(result.valid[1])
        self.assertLess((result.uvw[0] - .4).abs().max(), .02)

    def test_resume_next_update(self):
        first = PoseTable([camera(0)], device='cpu')
        p = first.parameters[0]
        first.update(0,lambda: ((p-.2).square().mean(),1.0))
        second = PoseTable([camera(0)], device='cpu')
        second.load_state_dict(copy.deepcopy(first.state_dict()))
        first.update(0,lambda: ((p-.2).square().mean(),1.0))
        q = second.parameters[0]
        second.update(0,lambda: ((q-.2).square().mean(),1.0))
        torch.testing.assert_close(p,q)
        self.assertEqual(first.accepted_updates, second.accepted_updates)

    def test_temporal_terms_are_unscaled_and_acceleration_warms_up(self):
        table = PoseTable([camera(0), camera(1), camera(2)], device='cpu',
                          acceleration_delay=2, acceleration_ramp=2)
        table.parameters[1].data[0] = 0.1
        # prior=.01*.01/6, velocity=.15*2*.01/6, acceleration disabled
        expected = .01 * .01 / 6 + .15 * 2 * .01 / 6
        self.assertAlmostEqual(float(table.regularization(1)), expected, places=7)
        table.accepted_updates = 3
        self.assertAlmostEqual(table.acceleration_weight(), .025)
        expected += .025 * .04 / 6
        self.assertAlmostEqual(float(table.regularization(1)), expected, places=7)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class UVWRenderTest(unittest.TestCase):
    def test_identity_equivalence_and_pose_gradient(self):
        from scene.gaussian_model import GaussianModel
        from gaussian_renderer import render, render_motion
        from gaussian_renderer.uvw import render_uvw
        from utils.graphics_utils import getProjectionMatrix
        g = GaussianModel(2)
        device='cuda'
        g._xyz = torch.nn.Parameter(torch.tensor([[-.1,.03,0],[.12,-.1,.05],[0,.1,.1]],device=device))
        g._scaling = torch.nn.Parameter(torch.tensor([[.08,.05,.04],[.04,.09,.05],[.07,.05,.1]],device=device).log())
        g._rotation = torch.nn.Parameter(torch.tensor([[1.,.2,.1,.3]]*3,device=device))
        g._opacity = torch.nn.Parameter(torch.ones(3,1,device=device))
        g._features_dc = torch.nn.Parameter(torch.randn(3,1,3,device=device)*.1)
        g._features_rest = torch.nn.Parameter(torch.randn(3,8,3,device=device)*.03)
        g.active_sh_degree=2
        c = camera(0,device)
        from data_utils.head_stabilizer.se3 import so3_exp
        w = c.world_view_transform.T.clone()
        w[:3,:3] = so3_exp(torch.tensor([.1,.2,-.1],device=device))
        c.world_view_transform=w.T.contiguous()
        c.FoVx=c.FoVy=.6
        c.image_width=c.image_height=64
        c.projection_matrix=getProjectionMatrix(.01,100,.6,.6).T.cuda()
        c.full_proj_transform=c.world_view_transform @ c.projection_matrix
        c.camera_center=-((w[:3,:3].T * w[:3,3][None]).sum(-1))
        pipe=SimpleNamespace(debug=False,compute_cov3D_python=False,convert_SHs_python=False)
        bg=torch.zeros(3,device=device)
        a=render(c,g,pipe,bg)
        b=render_uvw(c,g,None,pipe,bg)
        for key in ('render','alpha','depth'):
            torch.testing.assert_close(a[key],b[key],atol=2e-4,rtol=2e-4)
        class Motion(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.offset = torch.nn.Parameter(torch.tensor(.01, device=device))
            def forward(self, xyz, audio, expression, code=None, gaussian_scaling=None):
                return dict(d_xyz=xyz * self.offset, d_scale=torch.ones_like(xyz) * self.offset,
                            d_rot=torch.ones(len(xyz), 4, device=device) * self.offset)
        motion = Motion()
        c.talking_dict.update(auds=torch.zeros(8,29,16,device=device),au_exp=torch.zeros(6,device=device))
        dynamic_original = render_motion(c,g,motion,pipe,bg)
        dynamic_new = render_uvw(c,g,motion,pipe,bg)
        for key in ('render','alpha','depth'):
            torch.testing.assert_close(dynamic_original[key],dynamic_new[key],atol=2e-4,rtol=2e-4)
        table=PoseTable([c])
        target=torch.linspace(0,1,64,device=device)[None,None,:]
        def loss():
            out=render_uvw(c,g,motion,pipe,bg,table.matrix(c,True),True)
            return (out['render']*target).sum()
        loss().backward()
        grad=table.parameters[0].grad.clone()
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.norm()),0)
        self.assertIsNone(g._xyz.grad)
        self.assertIsNone(motion.offset.grad)
        for axis in range(6):
            with torch.no_grad():
                table.parameters[0][axis]=.002
                plus=float(loss())
                table.parameters[0][axis]=-.002
                minus=float(loss())
                table.parameters[0][axis]=0
            numerical=(plus-minus)/.004
            self.assertLess(abs(float(grad[axis])-numerical), max(.05,abs(numerical)*.15))
        g.fixed_uvw=FixedUVW(torch.ones(3,3,device=device)*.4,torch.ones(3,device=device),
                             torch.ones(3,device=device),g.get_xyz,g.get_scaling.max(-1).values)
        out=render_uvw(c,g,None,pipe,bg,freeze_field=True)
        valid=out['uvw_coverage'][0]>.1
        torch.testing.assert_close(out['uvw'][:,valid],torch.full_like(out['uvw'][:,valid],.4),atol=1e-5,rtol=1e-5)


        # An invalid front point still occludes valid attributes behind it.
        with torch.no_grad():
            g._xyz[:] = 0
            g._xyz[0, 2] = -.1
            g.fixed_uvw.valid[0] = False
            g.fixed_uvw.position.copy_(g._xyz)
            g._opacity[:] = 2
        occluded = render_uvw(c,g,None,pipe,bg,freeze_field=True)['uvw_coverage']
        with torch.no_grad():
            g._opacity[0] = -20
        unoccluded = render_uvw(c,g,None,pipe,bg,freeze_field=True)['uvw_coverage']
        self.assertGreater(float((unoccluded-occluded).max()), .01)

        from argparse import ArgumentParser
        from arguments import OptimizationParams
        parser = ArgumentParser()
        op = OptimizationParams(parser)
        options = op.extract(parser.parse_args([]))
        g._identity = torch.nn.Parameter(torch.zeros(3,1,device=device))
        g.max_radii2D = torch.zeros(3,device=device)
        g.spatial_lr_scale = 1.0
        g.training_setup(options)
        g.fixed_uvw.uvw = torch.arange(3,device=device)[:,None].expand(-1,3).float().clone()
        with torch.no_grad():
            g._scaling.fill_(-8)
        g.densify_and_clone(torch.ones(3,1,device=device),.1,1.0)
        torch.testing.assert_close(g.fixed_uvw.uvw[:,0],torch.tensor([0,1,2,0,1,2],device=device).float())
        with torch.no_grad():
            g._scaling.fill_(-1)
        g.densify_and_split(torch.ones(6,1,device=device),.1,1.0)
        torch.testing.assert_close(g.fixed_uvw.uvw[:,0],torch.tensor([0,1,2,0,1,2]*2,device=device).float())
        prune = torch.arange(12,device=device) % 2 == 0
        expected = g.fixed_uvw.uvw[~prune].clone()
        g.prune_points(prune)
        torch.testing.assert_close(g.fixed_uvw.uvw,expected)
        self.assertEqual(len(g.get_xyz),len(expected))


if __name__ == '__main__':
    unittest.main()

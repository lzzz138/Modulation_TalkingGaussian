import copy
import math
import tempfile
import unittest
from types import SimpleNamespace
import torch
from scene.pose_visibility_deformation import (PoseVisibilityResidual, pose_condition,
    visibility_descriptor, prerender_visibility, row_transform)
from utils.graphics_utils import getProjectionMatrix


def options(condition='pose_visibility'):
    return SimpleNamespace(pose_visibility_deformation=True, pcv_condition=condition,
        pcv_start=6000, pcv_ramp_steps=3000, pcv_max_displacement=.001,
        pcv_visibility_resolution=32, pcv_lr=.0005, pcv_reg_weight=.001,
        geometry_mod_multiscale=1, geometry_mod_map_res=16, geometry_mod_condition_scale=.1,
        audio_extractor='deepspeech', sh_degree=2)


def camera(device='cpu'):
    w=torch.eye(4,device=device);w[2,3]=2
    proj=getProjectionMatrix(.01,100,.6,.6).to(device).T
    full=(w.T[:,:,None]*proj[None,:,:]).sum(1)
    return SimpleNamespace(world_view_transform=w.T.contiguous(), full_proj_transform=full,
        camera_center=torch.tensor([0.,0.,-2.],device=device),FoVx=.6,FoVy=.6,
        image_width=64,image_height=64,talking_dict={'img_id':0,
            'auds':torch.randn(8,29,16,device=device),'au_exp':torch.zeros(6,device=device)})


class PCVTest(unittest.TestCase):
    def test_depth_normalization_occlusion_invalid(self):
        c=camera();x=torch.tensor([[0.,0.,0.],[0.,0.,.5],[100.,0.,0.],[0.,0.,-3.]],requires_grad=True)
        alpha=torch.full((1,32,32),.5);depth=alpha*2
        v=visibility_descriptor(c,x,torch.ones_like(x)*.01,depth,alpha)
        self.assertFalse(v.requires_grad)
        self.assertAlmostEqual(float(v[0,3]),.5)
        self.assertLess(float(v[1,3]),1e-5)
        self.assertEqual(float(v[2:].sum()),0)
        self.assertEqual(float(visibility_descriptor(c,x,torch.ones_like(x),depth,alpha*0).sum()),0)

    def test_rotation_convention(self):
        c=camera();angle=.35
        rotation=torch.tensor([[math.cos(angle),0,math.sin(angle)],[0,1,0],[-math.sin(angle),0,math.cos(angle)]])
        c.world_view_transform[:3,:3]=rotation
        result=pose_condition(c,torch.zeros(2,3))
        torch.testing.assert_close(result[0,:6],rotation[:,:2].T.reshape(-1))
        torch.testing.assert_close(result[0,6:],torch.tensor([0.,0.,-1.]))

    def test_residual_schedule_bounds_and_detach(self):
        m=PoseVisibilityResidual(options());f=torch.randn(8,74,requires_grad=True)
        p=torch.randn(8,9,requires_grad=True);v=torch.randn(8,4,requires_grad=True)
        m.set_iteration(9000,6000,3000)
        d,reg,_=m(f,p,v);self.assertEqual(float(d.abs().max()),0)
        d.sum().backward();self.assertGreater(float(m.output.weight.grad.abs().sum()),0)
        self.assertIsNone(p.grad);self.assertIsNone(v.grad)
        torch.nn.init.normal_(m.output.weight)
        d,_,_=m(f,p,v);self.assertLessEqual(float(d.abs().max()),.001)
        m.set_iteration(6000,6000,3000);self.assertEqual(float(m(f,p,v)[0].abs().sum()),0)

    def test_ablation_and_optimizer_restore(self):
        m=PoseVisibilityResidual(options('none'));m.strength.fill_(1)
        torch.nn.init.normal_(m.output.weight)
        f=torch.randn(5,74);p=torch.randn(5,9);v=torch.randn(5,4)
        torch.testing.assert_close(m(f,p,v)[0],m(f,p*10,v*7)[0],rtol=0,atol=0)
        opt=torch.optim.Adam(m.parameters(),lr=.001)
        def step(module,optimizer):
            optimizer.zero_grad();d,r,_=module(f,p,v);(d.sum()+r*.001).backward();optimizer.step()
        step(m,opt);state=copy.deepcopy(m.state_dict());ostate=copy.deepcopy(opt.state_dict())
        other=PoseVisibilityResidual(options('none'));other.load_state_dict(state)
        otheropt=torch.optim.Adam(other.parameters(),lr=.001);otheropt.load_state_dict(ostate)
        step(m,opt);step(other,otheropt)
        for a,b in zip(m.parameters(),other.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_new_defaults_do_not_change_d_checkpoint_config(self):
        from train_uvw_pose import Runtime
        r = Runtime.__new__(Runtime)
        r.dataset = SimpleNamespace(source_path='x', model_path='y', eval=False, audio_extractor='deepspeech')
        r.options = SimpleNamespace(iterations=50000)
        r.args = SimpleNamespace(pose_interval=5, pose_start=40005, pose_stop=50001,
            fixed_pose_control=False, pose_acceleration_delay=500, pose_acceleration_ramp=1500,
            topology_freeze_iter=40000)
        r.post_topology = True
        before = r.configuration()
        r.dataset.pose_visibility_deformation = False
        for k,v in vars(options()).items():
            if k.startswith('pcv_'):setattr(r.dataset,k,v)
        self.assertEqual(before,r.configuration())

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_render_equivalence_gradient_and_visibility(self):
        from scene.gaussian_model import GaussianModel
        from scene.motion_net import MotionNetwork
        from gaussian_renderer import render_motion
        from unittest.mock import patch
        g=GaussianModel(2)
        g._xyz=torch.nn.Parameter(torch.tensor([[-.08,.03,0.],[.08,-.03,.01],[0.,0.,.3]],device='cuda'))
        g._scaling=torch.nn.Parameter(torch.tensor([[.04,.05,.02]]*3,device='cuda').log())
        g._rotation=torch.nn.Parameter(torch.tensor([[1.,0,0,0]]*3,device='cuda'))
        g._opacity=torch.nn.Parameter(torch.ones(3,1,device='cuda')*2)
        g._features_dc=torch.nn.Parameter(torch.randn(3,1,3,device='cuda')*.1)
        g._features_rest=torch.nn.Parameter(torch.zeros(3,8,3,device='cuda'))
        c=camera('cuda');pipe=SimpleNamespace(debug=False);bg=torch.zeros(3,device='cuda')
        args=options();torch.manual_seed(12);model=MotionNetwork(args=args).cuda()
        baseargs=copy.copy(args);baseargs.pose_visibility_deformation=False
        torch.manual_seed(12);base=MotionNetwork(args=baseargs).cuda()
        for key,value in base.state_dict().items():torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
        a=render_motion(c,g,base,pipe,bg)
        model.pose_residual.strength.fill_(1)
        with patch('scene.pose_visibility_deformation.prerender_visibility', wraps=prerender_visibility) as pre:
            b=render_motion(c,g,model,pipe,bg);self.assertEqual(pre.call_count,1)
        for key in ('render','depth','alpha'):torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        b['render'].square().mean().backward()
        self.assertIsNotNone(b['viewspace_points'].grad)
        self.assertGreater(float(model.pose_residual.output.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.sigma_net.net[-1].weight.grad.abs().sum()),0)
        groups=model.get_params(.005,.0005);ids=[id(p) for group in groups for p in group['params']]
        self.assertEqual(len(ids),len(set(ids)))
        vis=prerender_visibility(c,g.get_xyz,g.get_scaling,g.get_rotation,g.get_opacity,32)
        self.assertFalse(vis.requires_grad)
        # Repeated positions mimic clone/split topology; no persistent per-point state.
        vis2=prerender_visibility(c,g.get_xyz.repeat(2,1),g.get_scaling.repeat(2,1),g.get_rotation.repeat(2,1),g.get_opacity.repeat(2,1),32)
        self.assertEqual(vis2.shape,(6,4))

if __name__=='__main__':unittest.main()
